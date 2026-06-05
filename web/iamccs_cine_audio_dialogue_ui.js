import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

console.info("[IAMCCS AudioDialogueUI] module loaded", { ts: new Date().toISOString() });

const EMOTIONS = [
    ["calm", "Calm", "Restrained delivery, steady breath"],
    ["fear", "Fear", "Tension, fragile voice"],
    ["anger", "Anger", "Compressed energy, hard consonants"],
    ["sad", "Sad", "Soft pace, low intensity"],
    ["joy", "Joy", "Warm and open"],
    ["shock", "Shock", "Interrupted breath, sudden reaction"],
    ["whisper", "Whisper", "Close microphone, low volume"],
    ["urgent", "Urgent", "Faster delivery, forward pressure"],
    ["wonder", "Wonder", "Curious, suspended tone"],
    ["resolve", "Resolve", "Firm, controlled confidence"],
];

const PRESETS = [
    ["tense_dialogue", "Tense Dialogue", ["urgent", "whisper", "fear"]],
    ["quiet_drama", "Quiet Drama", ["calm", "sad", "resolve"]],
    ["reveal", "Reveal", ["shock", "wonder", "whisper"]],
    ["conflict", "Conflict", ["anger", "urgent", "resolve"]],
];

function ensureStyles() {
    if (document.getElementById("iamccs-cine-audio-dialogue-style")) return;
    const style = document.createElement("style");
    style.id = "iamccs-cine-audio-dialogue-style";
    style.textContent = `
        .iamccs-emotion-panel {
            box-sizing: border-box;
            width: 100%;
            padding: 10px;
            color: #d9e2ea;
            font: 12px/1.35 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: linear-gradient(180deg, rgba(24,31,39,.98), rgba(15,19,24,.98));
            border: 1px solid rgba(133, 154, 177, .22);
            border-radius: 8px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.035);
        }
        .iamccs-emotion-head {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 10px;
            margin-bottom: 8px;
        }
        .iamccs-emotion-title {
            color: #f2f6f9;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0;
        }
        .iamccs-emotion-subtitle {
            margin-top: 2px;
            color: #8ea0ae;
            font-size: 10px;
        }
        .iamccs-emotion-clear {
            flex: 0 0 auto;
            padding: 5px 8px;
            color: #b9c7d2;
            background: rgba(255,255,255,.055);
            border: 1px solid rgba(160,176,194,.18);
            border-radius: 6px;
            cursor: pointer;
            font-size: 10px;
        }
        .iamccs-emotion-clear:hover {
            color: #fff;
            border-color: rgba(213,226,239,.32);
            background: rgba(255,255,255,.09);
        }
        .iamccs-emotion-selected {
            min-height: 26px;
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 5px;
            margin-bottom: 9px;
            padding: 7px;
            background: rgba(0,0,0,.24);
            border: 1px solid rgba(255,255,255,.07);
            border-radius: 7px;
        }
        .iamccs-emotion-empty {
            color: #728292;
            font-size: 10px;
        }
        .iamccs-emotion-chip {
            padding: 3px 7px;
            color: #dfe8f0;
            background: rgba(93, 134, 181, .18);
            border: 1px solid rgba(119, 164, 215, .26);
            border-radius: 999px;
            font-size: 10px;
        }
        .iamccs-emotion-section {
            margin-top: 9px;
            margin-bottom: 5px;
            color: #9cadba;
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .04em;
        }
        .iamccs-emotion-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 6px;
        }
        .iamccs-emotion-btn,
        .iamccs-emotion-preset {
            min-height: 34px;
            text-align: left;
            color: #c6d3df;
            background: rgba(255,255,255,.045);
            border: 1px solid rgba(160,176,194,.16);
            border-radius: 7px;
            padding: 6px 8px;
            cursor: pointer;
        }
        .iamccs-emotion-btn:hover,
        .iamccs-emotion-preset:hover {
            color: #f5f8fb;
            background: rgba(255,255,255,.075);
            border-color: rgba(194,211,229,.30);
        }
        .iamccs-emotion-btn.is-active {
            color: #ffffff;
            background: linear-gradient(180deg, rgba(44, 113, 184, .52), rgba(32, 73, 129, .50));
            border-color: rgba(137, 190, 245, .60);
        }
        .iamccs-emotion-btn strong,
        .iamccs-emotion-preset strong {
            display: block;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0;
        }
        .iamccs-emotion-btn span,
        .iamccs-emotion-preset span {
            display: block;
            margin-top: 1px;
            color: #8fa0af;
            font-size: 9px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .iamccs-emotion-btn.is-active span {
            color: #cde5ff;
        }
        .iamccs-emotion-presets {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 6px;
        }
    `;
    document.head.appendChild(style);
}

function findWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function parseSelected(value) {
    return String(value || "")
        .split(/[,;\n]+/)
        .map((part) => part.trim())
        .filter(Boolean);
}

function writeSelected(node, values) {
    const widget = findWidget(node, "selected_emotions");
    if (!widget) return;
    const unique = [...new Set(values.filter(Boolean))];
    widget.value = unique.join(", ");
    widget.callback?.(widget.value);
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function hideWidget(widget) {
    if (!widget) return;
    widget.hidden = true;
    widget.serialize = true;
    widget.type = "hidden";
    widget.computeSize = () => [0, -4];
    widget.draw = () => {};
    widget.options = { ...(widget.options || {}), hidden: true };
    if (widget.inputEl) {
        widget.inputEl.style.display = "none";
        widget.inputEl.style.height = "0";
        widget.inputEl.style.opacity = "0";
    }
}

function widgetValue(node, name, fallback = "") {
    const widget = findWidget(node, name);
    return widget ? widget.value : fallback;
}

function setWidgetValue(node, name, value) {
    const widget = findWidget(node, name);
    if (!widget) return false;
    widget.value = value;
    widget.callback?.(value);
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
    return true;
}

function firstUiValue(message, key) {
    if (!message || typeof message !== "object") return "";
    const value = message[key];
    if (Array.isArray(value)) return String(value[0] || "");
    return String(value || "");
}

function addText(parent, className, text, tag = "div") {
    const el = document.createElement(tag);
    el.className = className;
    el.textContent = text;
    parent.appendChild(el);
    return el;
}

function ensureDialogueFoleyStyles() {
    if (document.getElementById("iamccs-dialogue-foley-boardmaker-style")) return;
    const style = document.createElement("style");
    style.id = "iamccs-dialogue-foley-boardmaker-style";
    style.textContent = `
        .iamccs-dfb {
            box-sizing: border-box;
            width: 100%;
            min-height: 720px;
            padding: 8px;
            color: #ecf2f4;
            --dfb-text-scale: 1;
            font: 12px/1.35 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: linear-gradient(180deg, #0f171a 0%, #101315 100%);
            border: 1px solid rgba(154, 184, 190, .22);
            border-radius: 8px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.045);
        }
        .iamccs-dfb * { box-sizing: border-box; letter-spacing: 0; }
        .iamccs-dfb-head {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 10px;
            background: #182428;
            border: 1px solid rgba(162, 198, 206, .18);
            border-radius: 7px;
        }
        .iamccs-dfb-title { font-size: 15px; font-weight: 800; color: #ffffff; }
        .iamccs-dfb-subtitle { margin-top: 2px; font-size: 11px; color: #9fb1b7; font-weight: 650; }
        .iamccs-dfb-actions { margin-left: auto; display: flex; gap: 7px; flex-wrap: wrap; }
        .iamccs-dfb-btn {
            height: 28px;
            padding: 0 9px;
            color: #eef6f7;
            background: #24343a;
            border: 1px solid rgba(172, 206, 214, .22);
            border-radius: 6px;
            font-weight: 750;
            cursor: pointer;
            white-space: nowrap;
        }
        .iamccs-dfb-btn:hover { background: #2e444b; border-color: rgba(205, 230, 235, .38); }
        .iamccs-dfb-btn.primary { background: #2a6370; border-color: #70aeba; color: #fff; }
        .iamccs-dfb-btn.danger { background: #523033; border-color: #9c5b61; }
        .iamccs-dfb-btn.is-active { background: #f1eee8; border-color: #d7c7b7; color: #111; }
        .iamccs-dfb-btn.inject {
            height: 38px;
            padding: 0 16px;
            color: #ffffff;
            background: linear-gradient(180deg, #2f7f8d 0%, #21545f 100%);
            border-color: #8fd1dc;
            font-size: 12px;
            font-weight: 900;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.16), 0 0 0 1px rgba(143,209,220,.12);
        }
        .iamccs-dfb-btn.inject:hover { background: linear-gradient(180deg, #3a94a4 0%, #286470 100%); }
        .iamccs-dfb-top {
            display: grid;
            grid-template-columns: minmax(240px, 1fr) 76px 98px 98px 92px 98px 98px 150px;
            gap: 7px;
            margin-top: 7px;
            padding: 7px;
            background: #121c1f;
            border: 1px solid rgba(156, 184, 190, .14);
            border-radius: 7px;
        }
        .iamccs-dfb-field { display: flex; flex-direction: column; gap: 3px; min-width: 0; color: #aebec3; font-size: 9px; font-weight: 800; text-transform: uppercase; }
        .iamccs-dfb-input, .iamccs-dfb-text, .iamccs-dfb-select {
            width: 100%;
            color: #f3f7f8;
            background: #0b1113;
            border: 1px solid rgba(156, 184, 190, .22);
            border-radius: 6px;
            outline: none;
            font: calc(11px * var(--dfb-text-scale)) / 1.28 Inter, ui-sans-serif, system-ui, sans-serif;
        }
        .iamccs-dfb-input, .iamccs-dfb-select { height: 28px; padding: 0 7px; }
        .iamccs-dfb-text { min-height: 48px; padding: 6px 7px; resize: vertical; }
        .iamccs-dfb.paper .iamccs-dfb-input,
        .iamccs-dfb.paper .iamccs-dfb-text {
            color: #090909;
            background: #f8f4ed;
            border-color: #b7a795;
            font-family: "Courier New", ui-monospace, monospace;
            font-weight: 700;
        }
        .iamccs-dfb.paper .iamccs-dfb-input::placeholder,
        .iamccs-dfb.paper .iamccs-dfb-text::placeholder { color: #685f55; }
        .iamccs-dfb-input:focus, .iamccs-dfb-text:focus, .iamccs-dfb-select:focus { border-color: #75bac6; box-shadow: 0 0 0 1px rgba(117,186,198,.18); }
        .iamccs-dfb-panels {
            display: grid;
            grid-template-columns: minmax(420px, .9fr) minmax(520px, 1.1fr);
            gap: 7px;
            margin-top: 7px;
        }
        .iamccs-dfb-panel {
            padding: 7px;
            background: #131d20;
            border: 1px solid rgba(156, 184, 190, .14);
            border-radius: 7px;
        }
        .iamccs-dfb-panel-title {
            margin-bottom: 5px;
            color: #eff7f8;
            font-size: 10px;
            font-weight: 850;
            text-transform: uppercase;
        }
        .iamccs-dfb-status {
            min-height: 26px;
            margin-top: 7px;
            padding: 6px 8px;
            color: #b8c8cc;
            background: #0b1113;
            border: 1px solid rgba(156, 184, 190, .15);
            border-radius: 7px;
            font-weight: 700;
        }
        .iamccs-dfb-list {
            display: flex;
            flex-direction: column;
            gap: 6px;
            margin-top: 7px;
            max-height: 500px;
            overflow: auto;
            padding-right: 3px;
        }
        .iamccs-dfb-card {
            display: grid;
            grid-template-columns: 44px 96px 150px minmax(240px, 1fr) minmax(260px, 1fr) minmax(180px, .75fr) 58px;
            gap: 6px;
            padding: 6px;
            background: #172327;
            border: 1px solid rgba(156, 184, 190, .18);
            border-radius: 7px;
        }
        .iamccs-dfb-badge {
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            white-space: pre-line;
            color: #fff;
            background: linear-gradient(180deg, #2a6370, #172327);
            border: 1px solid #70aeba;
            border-radius: 6px;
            font-weight: 850;
            line-height: 1.16;
            font-size: 10px;
        }
        .iamccs-dfb-card-actions { display: grid; gap: 5px; align-content: start; }
        .iamccs-speech1 {
            box-sizing: border-box;
            width: 100%;
            padding: 10px;
            color: #ecf2f4;
            font: 12px/1.35 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: linear-gradient(180deg, #121c1f 0%, #0d1315 100%);
            border: 1px solid rgba(154, 184, 190, .24);
            border-radius: 8px;
        }
        .iamccs-speech1-head { display: grid; gap: 3px; margin-bottom: 9px; }
        .iamccs-speech1-title { font-size: 14px; font-weight: 900; color: #fff; }
        .iamccs-speech1-subtitle { font-size: 10px; color: #9fb1b7; font-weight: 700; }
        .iamccs-speech1-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; margin-bottom: 8px; }
        .iamccs-speech1-field { display: grid; gap: 3px; color: #aebec3; font-size: 9px; font-weight: 850; text-transform: uppercase; }
        .iamccs-speech1-input {
            height: 30px;
            padding: 0 8px;
            color: #f3f7f8;
            background: #0b1113;
            border: 1px solid rgba(156, 184, 190, .22);
            border-radius: 6px;
            outline: none;
        }
        .iamccs-speech1-check { display: flex; align-items: center; gap: 6px; min-height: 30px; color: #d8e3e6; text-transform: none; font-size: 11px; }
        .iamccs-speech1-inject {
            width: 100%;
            height: 42px;
            color: #fff;
            background: linear-gradient(180deg, #2f7f8d 0%, #21545f 100%);
            border: 1px solid #8fd1dc;
            border-radius: 7px;
            font-size: 12px;
            font-weight: 950;
            cursor: pointer;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.16), 0 0 0 1px rgba(143,209,220,.12);
        }
        .iamccs-speech1-inject:hover { background: linear-gradient(180deg, #3a94a4 0%, #286470 100%); }
        .iamccs-speech1-status {
            min-height: 28px;
            margin-top: 8px;
            padding: 6px 8px;
            color: #b8c8cc;
            background: #0b1113;
            border: 1px solid rgba(156, 184, 190, .15);
            border-radius: 7px;
            font-weight: 750;
        }
    `;
    document.head.appendChild(style);
}

function protectDrag(el) {
    for (const eventName of ["pointerdown", "mousedown", "dblclick", "wheel"]) {
        el.addEventListener(eventName, (event) => event.stopPropagation(), { capture: true });
    }
    return el;
}

function makeInput(value, onInput, opts = {}) {
    const el = document.createElement(opts.multiline ? "textarea" : "input");
    if (!opts.multiline) el.type = opts.type || "text";
    el.className = opts.multiline ? "iamccs-dfb-text" : "iamccs-dfb-input";
    el.value = value ?? "";
    if (opts.placeholder) el.placeholder = opts.placeholder;
    if (opts.type === "number") {
        el.step = opts.step ?? "0.1";
        if (opts.min != null) el.min = String(opts.min);
        if (opts.max != null) el.max = String(opts.max);
    }
    el.oninput = () => onInput(opts.type === "number" ? Number(el.value) : el.value);
    return protectDrag(el);
}

function makeSelect(value, options, onChange) {
    const el = document.createElement("select");
    el.className = "iamccs-dfb-select";
    for (const option of options) {
        const opt = document.createElement("option");
        opt.value = option;
        opt.textContent = option;
        el.appendChild(opt);
    }
    el.value = options.includes(String(value)) ? String(value) : options[0];
    el.onchange = () => onChange(el.value);
    return protectDrag(el);
}

function field(label, control) {
    const wrap = document.createElement("label");
    wrap.className = "iamccs-dfb-field";
    const span = document.createElement("span");
    span.textContent = label;
    wrap.append(span, control);
    return wrap;
}

function parseDialogueRows(text) {
    try {
        const data = JSON.parse(String(text || "").trim() || "{}");
        const raw = Array.isArray(data) ? data : Array.isArray(data.lines) ? data.lines : [];
        return raw.length ? raw.map((row, index) => normalizeDialogueRow(row, index)) : defaultDialogueRows();
    } catch {
        return defaultDialogueRows();
    }
}

function normalizeDialogueRow(row = {}, index = 0) {
    const speaker = String(row.speaker || (index % 2 ? "B" : "A")).toUpperCase().startsWith("B") ? "B" : "A";
    return {
        speaker,
        seconds: Math.max(0.25, Number(row.seconds ?? row.duration ?? 3.5) || 3.5),
        ref: Math.max(1, Math.round(Number(row.ref ?? (speaker === "A" ? 1 : 2)) || 1)),
        label: String(row.label || (speaker === "A" ? `campo_A_${index + 1}` : `controcampo_B_${index + 1}`)),
        framing: String(row.framing || (speaker === "A" ? "medium close-up" : "reverse angle")),
        dialogue: String(row.dialogue || row.text || ""),
        voice: String(row.voice || row.voice_direction || ""),
        local_prompt: String(row.local_prompt || row.prompt || ""),
        foley: String(row.foley || ""),
    };
}

function defaultDialogueRows() {
    return [
        normalizeDialogueRow({ speaker: "A", seconds: 4, ref: 1, label: "campo_A", dialogue: "Non guardare la finestra. Se capisce che l'abbiamo visto, siamo finiti.", voice: "low tense voice, almost whispered", local_prompt: "man A in field shot, tense medium close-up, restrained fear, coherent eyeline", foley: "soft breathing, cloth rustle, distant room tone" }, 0),
        normalizeDialogueRow({ speaker: "B", seconds: 3.5, ref: 2, label: "controcampo_B", dialogue: "L'ha gia capito. Sta aspettando che uno di noi faccia il primo passo.", voice: "controlled low voice, guarded", local_prompt: "man B in reverse field shot, listening then answering, controlled fear", foley: "small chair creak, quiet breath, low room pressure" }, 1),
        normalizeDialogueRow({ speaker: "A", seconds: 3.5, ref: 1, label: "ritorno_A", dialogue: "Allora il primo passo lo fai tu, lentamente, verso l'uscita.", voice: "firm whisper, urgent but contained", local_prompt: "return to man A close-up, slight push-in, decision in the eyes", foley: "subtle foot shift, jacket movement, controlled breath" }, 2),
        normalizeDialogueRow({ speaker: "B", seconds: 4, ref: 2, label: "chiusura_B", dialogue: "No. Se esco da solo, tu resti qui a spiegargli dove siamo.", voice: "dry, bitter, quietly direct", local_prompt: "man B reverse close-up, hard pause before the last words, tense eye line", foley: "distant door tension, faint wood creak, breath held" }, 3),
    ];
}

function renderDialogueFoleyBoardMaker(node) {
    if (node.__iamccsDialogueFoleyReady) return;
    node.__iamccsDialogueFoleyReady = true;
    ensureDialogueFoleyStyles();

    const widgetNames = [
        "board_name", "global_prompt", "scene_context", "foley_prompt", "dialogue_data",
        "frame_rate", "image_width", "image_height", "default_force", "guide_policy",
        "woosh_chunk_seconds", "woosh_overlap_seconds",
    ];
    const hideNativeWidgets = () => widgetNames.forEach((name) => hideWidget(findWidget(node, name)));
    hideNativeWidgets();
    requestAnimationFrame?.(hideNativeWidgets);
    setTimeout(hideNativeWidgets, 50);
    node.color = "#1c3035";
    node.bgcolor = "#101719";
    node.size = [Math.max(node.size?.[0] || 1480, 1540), Math.max(node.size?.[1] || 760, 900)];

    let rows = parseDialogueRows(widgetValue(node, "dialogue_data", ""));
    let paperMode = Boolean(node.properties?.iamccs_dfb_paper_mode);
    let textScale = Number(node.properties?.iamccs_dfb_text_scale || 1);

    const root = document.createElement("div");
    root.className = "iamccs-dfb";
    root.classList.toggle("paper", paperMode);
    root.style.setProperty("--dfb-text-scale", String(textScale));

    const head = document.createElement("div");
    head.className = "iamccs-dfb-head";
    const headText = document.createElement("div");
    addText(headText, "iamccs-dfb-title", "IAMCCS BoardMaker Dialogue Foley");
    addText(headText, "iamccs-dfb-subtitle", "Scrivi board A/B. Il cine_linx e l'inject verso Shotboard, TTS e Woosh.");
    const actions = document.createElement("div");
    actions.className = "iamccs-dfb-actions";
    const addBtn = document.createElement("button");
    addBtn.className = "iamccs-dfb-btn primary";
    addBtn.textContent = "Add Line";
    const presetBtn = document.createElement("button");
    presetBtn.className = "iamccs-dfb-btn";
    presetBtn.textContent = "Preset A/B";
    const injectBtn = document.createElement("button");
    injectBtn.className = "iamccs-dfb-btn inject";
    injectBtn.textContent = "INJECT TO SHOTBOARD";
    injectBtn.title = "Sync this BoardMaker into cine_linx. The connected Speech1 Compiler injects compiled global/local prompt, dialogue and foley into Shotboard/TTS/Woosh on queue.";
    const paperBtn = document.createElement("button");
    paperBtn.className = "iamccs-dfb-btn";
    paperBtn.textContent = "Paper";
    const fontDownBtn = document.createElement("button");
    fontDownBtn.className = "iamccs-dfb-btn";
    fontDownBtn.textContent = "A-";
    const fontUpBtn = document.createElement("button");
    fontUpBtn.className = "iamccs-dfb-btn";
    fontUpBtn.textContent = "A+";
    const fitBtn = document.createElement("button");
    fitBtn.className = "iamccs-dfb-btn";
    fitBtn.textContent = "Fit";
    const resetBtn = document.createElement("button");
    resetBtn.className = "iamccs-dfb-btn danger";
    resetBtn.textContent = "Reset";
    actions.append(addBtn, presetBtn, injectBtn, paperBtn, fontDownBtn, fontUpBtn, fitBtn, resetBtn);
    head.append(headText, actions);
    root.appendChild(head);

    const top = document.createElement("div");
    top.className = "iamccs-dfb-top";
    const boardName = makeInput(widgetValue(node, "board_name", "dialogue_foley_board"), (v) => setWidgetValue(node, "board_name", v));
    const fps = makeInput(widgetValue(node, "frame_rate", 24), (v) => setWidgetValue(node, "frame_rate", v), { type: "number", step: "1", min: 1, max: 120 });
    const width = makeInput(widgetValue(node, "image_width", 1280), (v) => setWidgetValue(node, "image_width", v), { type: "number", step: "32", min: 64 });
    const height = makeInput(widgetValue(node, "image_height", 736), (v) => setWidgetValue(node, "image_height", v), { type: "number", step: "32", min: 64 });
    const force = makeInput(widgetValue(node, "default_force", 0.65), (v) => setWidgetValue(node, "default_force", v), { type: "number", step: "0.01", min: 0, max: 1 });
    const chunk = makeInput(widgetValue(node, "woosh_chunk_seconds", 8), (v) => setWidgetValue(node, "woosh_chunk_seconds", v), { type: "number", step: "0.1", min: 1, max: 8 });
    const overlap = makeInput(widgetValue(node, "woosh_overlap_seconds", 1), (v) => setWidgetValue(node, "woosh_overlap_seconds", v), { type: "number", step: "0.1", min: 0, max: 4 });
    const policy = makeSelect(widgetValue(node, "guide_policy", "every_checked_row"), ["every_checked_row", "safe_core_guides", "prompt_only"], (v) => setWidgetValue(node, "guide_policy", v));
    top.append(field("Board", boardName), field("FPS", fps), field("Width", width), field("Height", height), field("Force", force), field("Woosh max", chunk), field("Overlap", overlap), field("Guide policy", policy));
    root.appendChild(top);

    const panels = document.createElement("div");
    panels.className = "iamccs-dfb-panels";
    const globalPanel = document.createElement("div");
    globalPanel.className = "iamccs-dfb-panel";
    addText(globalPanel, "iamccs-dfb-panel-title", "Scene architecture");
    const global = makeInput(widgetValue(node, "global_prompt", ""), (v) => setWidgetValue(node, "global_prompt", v), { multiline: true, placeholder: "Global prompt..." });
    const scene = makeInput(widgetValue(node, "scene_context", ""), (v) => setWidgetValue(node, "scene_context", v), { multiline: true, placeholder: "Scene context..." });
    globalPanel.append(field("Global prompt", global), field("Scene context", scene));
    const foleyPanel = document.createElement("div");
    foleyPanel.className = "iamccs-dfb-panel";
    addText(foleyPanel, "iamccs-dfb-panel-title", "Foley plan");
    const foley = makeInput(widgetValue(node, "foley_prompt", ""), (v) => setWidgetValue(node, "foley_prompt", v), { multiline: true, placeholder: "Global foley prompt..." });
    foleyPanel.append(field("Global foley prompt", foley));
    panels.append(globalPanel, foleyPanel);
    root.appendChild(panels);

    const status = document.createElement("div");
    status.className = "iamccs-dfb-status";
    root.appendChild(status);
    const list = document.createElement("div");
    list.className = "iamccs-dfb-list";
    root.appendChild(list);

    function sync() {
        rows = rows.map(normalizeDialogueRow);
        setWidgetValue(node, "dialogue_data", JSON.stringify({ lines: rows }, null, 2));
        const total = rows.reduce((sum, row) => sum + Number(row.seconds || 0), 0);
        const aCount = rows.filter((row) => row.speaker === "A").length;
        const bCount = rows.filter((row) => row.speaker === "B").length;
        status.textContent = `Lines: ${rows.length} | A: ${aCount} | B: ${bCount} | Duration: ${total.toFixed(1)}s | Inject: cine_linx | Token local prompt: <speech1>`;
    }

    function draw() {
        list.innerHTML = "";
        rows.forEach((row, index) => {
            const card = document.createElement("div");
            card.className = "iamccs-dfb-card";
            const badge = document.createElement("div");
            badge.className = "iamccs-dfb-badge";
            badge.textContent = `${row.speaker}\nLINE\n${String(index + 1).padStart(2, "0")}`;

            const meta = document.createElement("div");
            const speaker = makeSelect(row.speaker, ["A", "B"], (v) => { row.speaker = v; row.ref = v === "A" ? 1 : 2; sync(); draw(); });
            const seconds = makeInput(row.seconds, (v) => { row.seconds = Math.max(0.25, Number(v) || 0.25); sync(); }, { type: "number", step: "0.1", min: 0.25 });
            const ref = makeInput(row.ref, (v) => { row.ref = Math.max(1, Math.round(Number(v) || 1)); sync(); }, { type: "number", step: "1", min: 1 });
            meta.append(field("Speaker", speaker), field("Seconds", seconds), field("Ref", ref));

            const shot = document.createElement("div");
            const rowLabelInput = makeInput(row.label, (v) => { row.label = v; sync(); });
            const framing = makeInput(row.framing, (v) => { row.framing = v; sync(); });
            shot.append(field("Label", rowLabelInput), field("Framing", framing));

            const dialogue = makeInput(row.dialogue, (v) => { row.dialogue = v; sync(); }, { multiline: true, placeholder: "Dialogue line..." });
            const prompt = makeInput(row.local_prompt, (v) => { row.local_prompt = v; sync(); }, { multiline: true, placeholder: "Local visual prompt..." });
            const voice = makeInput(row.voice, (v) => { row.voice = v; sync(); }, { multiline: true, placeholder: "Voice direction..." });

            const rowActions = document.createElement("div");
            rowActions.className = "iamccs-dfb-card-actions";
            const copy = document.createElement("button");
            copy.className = "iamccs-dfb-btn";
            copy.textContent = "Copy";
            copy.onclick = () => { rows.splice(index + 1, 0, normalizeDialogueRow({ ...row, label: `${row.label}_copy` }, index + 1)); sync(); draw(); };
            const del = document.createElement("button");
            del.className = "iamccs-dfb-btn danger";
            del.textContent = "Del";
            del.onclick = () => { rows.splice(index, 1); if (!rows.length) rows = defaultDialogueRows(); sync(); draw(); };
            rowActions.append(copy, del);

            card.append(badge, meta, shot, field("Dialogue / speech1", dialogue), field("Local prompt / Shotboard", prompt), field("Voice / TTS style", voice), rowActions);
            list.appendChild(card);
        });
        sync();
    }

    function updateVisualMode() {
        paperBtn.classList.toggle("is-active", paperMode);
        root.classList.toggle("paper", paperMode);
        root.style.setProperty("--dfb-text-scale", String(textScale));
        node.properties = node.properties || {};
        node.properties.iamccs_dfb_paper_mode = paperMode;
        node.properties.iamccs_dfb_text_scale = textScale;
        node.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas?.(true, true);
    }

    addBtn.onclick = () => {
        rows.push(normalizeDialogueRow({ speaker: rows.length % 2 ? "B" : "A", dialogue: "New line.", local_prompt: "new dialogue beat, coherent eyeline", voice: "natural low voice", foley: "room tone and cloth movement" }, rows.length));
        sync();
        draw();
    };
    presetBtn.onclick = () => { rows = defaultDialogueRows(); sync(); draw(); };
    injectBtn.onclick = () => {
        sync();
        status.textContent = "Inject armato: BoardMaker -> Speech1 Compiler -> Shotboard/TTS/Woosh tramite cine_linx. Premi Queue per eseguire l'iniezione.";
    };
    paperBtn.onclick = () => { paperMode = !paperMode; updateVisualMode(); };
    fontDownBtn.onclick = () => { textScale = Math.max(0.85, Math.round((textScale - 0.1) * 100) / 100); updateVisualMode(); };
    fontUpBtn.onclick = () => { textScale = Math.min(1.45, Math.round((textScale + 0.1) * 100) / 100); updateVisualMode(); };
    fitBtn.onclick = () => { const total = rows.reduce((sum, row) => sum + Number(row.seconds || 0), 0); status.textContent = `Board duration will be ${total.toFixed(1)}s when queued.`; };
    resetBtn.onclick = () => { rows = defaultDialogueRows(); sync(); draw(); };

    updateVisualMode();
    draw();
    const domWidget = node.addDOMWidget("Dialogue Foley BoardMaker", "iamccs_dialogue_foley_boardmaker", root, { serialize: false });
    domWidget.computeSize = (width) => [Math.max(1480, Number(width || 1480)), 840];
}

function renderSpeech1Compiler(node) {
    if (node.__iamccsSpeech1CompilerReady) return;
    node.__iamccsSpeech1CompilerReady = true;
    ensureDialogueFoleyStyles();

    const widgetNames = ["speech_token", "append_scene_context", "append_voice_style"];
    const hideNativeWidgets = () => widgetNames.forEach((name) => hideWidget(findWidget(node, name)));
    hideNativeWidgets();
    requestAnimationFrame?.(hideNativeWidgets);
    setTimeout(hideNativeWidgets, 50);

    node.color = "#253236";
    node.bgcolor = "#11191b";
    node.size = [Math.max(node.size?.[0] || 430, 460), Math.max(node.size?.[1] || 230, 250)];

    const root = document.createElement("div");
    root.className = "iamccs-speech1";

    const head = document.createElement("div");
    head.className = "iamccs-speech1-head";
    addText(head, "iamccs-speech1-title", "Speech1 Prompt Compiler");
    addText(head, "iamccs-speech1-subtitle", "Compila <speech1> e passa il cine_linx allo Shotboard.");
    root.appendChild(head);

    const grid = document.createElement("div");
    grid.className = "iamccs-speech1-grid";

    const tokenInput = document.createElement("input");
    tokenInput.className = "iamccs-speech1-input";
    tokenInput.value = widgetValue(node, "speech_token", "<speech1>") || "<speech1>";
    tokenInput.oninput = () => setWidgetValue(node, "speech_token", tokenInput.value || "<speech1>");
    protectDrag(tokenInput);
    const tokenWrap = document.createElement("label");
    tokenWrap.className = "iamccs-speech1-field";
    tokenWrap.append(addText(document.createElement("span"), "", "Token"), tokenInput);

    const sceneCheck = document.createElement("input");
    sceneCheck.type = "checkbox";
    sceneCheck.checked = Boolean(widgetValue(node, "append_scene_context", true));
    sceneCheck.onchange = () => setWidgetValue(node, "append_scene_context", Boolean(sceneCheck.checked));
    protectDrag(sceneCheck);
    const sceneWrap = document.createElement("label");
    sceneWrap.className = "iamccs-speech1-check";
    sceneWrap.append(sceneCheck, document.createTextNode("append scene context"));

    const voiceCheck = document.createElement("input");
    voiceCheck.type = "checkbox";
    voiceCheck.checked = Boolean(widgetValue(node, "append_voice_style", true));
    voiceCheck.onchange = () => setWidgetValue(node, "append_voice_style", Boolean(voiceCheck.checked));
    protectDrag(voiceCheck);
    const voiceWrap = document.createElement("label");
    voiceWrap.className = "iamccs-speech1-check";
    voiceWrap.append(voiceCheck, document.createTextNode("append voice style"));

    grid.append(tokenWrap, sceneWrap, voiceWrap);
    root.appendChild(grid);

    const inject = document.createElement("button");
    inject.type = "button";
    inject.className = "iamccs-speech1-inject";
    inject.textContent = "COMPILE + INJECT TO SHOTBOARD";
    inject.title = "On Queue this node replaces <speech1>, writes local_prompts/segment_lengths/dialogue_script, and sends compiled cine_linx to Shotboard, TTS and Woosh.";
    protectDrag(inject);
    root.appendChild(inject);

    const status = document.createElement("div");
    status.className = "iamccs-speech1-status";
    status.textContent = "Ready: BoardMaker cine_linx in, compiled cine_linx out.";
    root.appendChild(status);

    inject.onclick = () => {
        setWidgetValue(node, "speech_token", tokenInput.value || "<speech1>");
        setWidgetValue(node, "append_scene_context", Boolean(sceneCheck.checked));
        setWidgetValue(node, "append_voice_style", Boolean(voiceCheck.checked));
        status.textContent = "Inject armato: in Queue questo nodo compila <speech1> e invia global/local/dialogue/foley allo Shotboard.";
    };

    const domWidget = node.addDOMWidget("Speech1 Compiler", "iamccs_speech1_compiler", root, { serialize: false });
    domWidget.computeSize = (width) => [Math.max(430, Number(width || 430)), 230];
}

function renderEmotionPanel(node) {
    if (node.__iamccsEmotionButtonsReady) return;
    node.__iamccsEmotionButtonsReady = true;
    ensureStyles();

    const selectedWidget = findWidget(node, "selected_emotions");
    hideWidget(selectedWidget);

    node.color = "#26323d";
    node.bgcolor = "#141a20";
    node.size = [
        Math.max(node.size?.[0] || 380, 420),
        Math.max(node.size?.[1] || 360, 500),
    ];

    const root = document.createElement("div");
    root.className = "iamccs-emotion-panel";

    const head = document.createElement("div");
    head.className = "iamccs-emotion-head";
    const titleWrap = document.createElement("div");
    addText(titleWrap, "iamccs-emotion-title", "Emotion Palette");
    addText(titleWrap, "iamccs-emotion-subtitle", "Seleziona il tono per voce, performance e prompt.");
    const clear = document.createElement("button");
    clear.type = "button";
    clear.className = "iamccs-emotion-clear";
    clear.textContent = "Clear";
    head.append(titleWrap, clear);
    root.appendChild(head);

    const selected = document.createElement("div");
    selected.className = "iamccs-emotion-selected";
    root.appendChild(selected);

    addText(root, "iamccs-emotion-section", "Preset rapidi");
    const presets = document.createElement("div");
    presets.className = "iamccs-emotion-presets";
    root.appendChild(presets);

    addText(root, "iamccs-emotion-section", "Emozioni");
    const grid = document.createElement("div");
    grid.className = "iamccs-emotion-grid";
    root.appendChild(grid);

    const buttons = new Map();

    function currentValues() {
        return parseSelected(findWidget(node, "selected_emotions")?.value);
    }

    function refresh() {
        const values = currentValues();
        const set = new Set(values);
        selected.innerHTML = "";
        if (!values.length) {
            addText(selected, "iamccs-emotion-empty", "Nessuna emozione selezionata");
        } else {
            for (const value of values) addText(selected, "iamccs-emotion-chip", value);
        }
        for (const [key, button] of buttons) {
            button.classList.toggle("is-active", set.has(key));
        }
    }

    function toggle(key) {
        const values = currentValues();
        const found = values.includes(key);
        writeSelected(node, found ? values.filter((value) => value !== key) : [...values, key]);
        refresh();
    }

    function applyPreset(values) {
        writeSelected(node, values);
        refresh();
    }

    for (const [, label, values] of PRESETS) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "iamccs-emotion-preset";
        button.innerHTML = `<strong>${label}</strong><span>${values.join(", ")}</span>`;
        button.onclick = () => applyPreset(values);
        presets.appendChild(button);
    }

    for (const [key, label, description] of EMOTIONS) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "iamccs-emotion-btn";
        button.title = description;
        button.innerHTML = `<strong>${label}</strong><span>${description}</span>`;
        button.onclick = () => toggle(key);
        buttons.set(key, button);
        grid.appendChild(button);
    }

    clear.onclick = () => {
        writeSelected(node, []);
        refresh();
    };

    refresh();
    const domWidget = node.addDOMWidget("Emotion Palette", "iamccs_emotion_palette", root, {
        serialize: false,
    });
    domWidget.computeSize = (width) => [width, 390];
}


function ensureAudioBoardArrangerStyles() {
    let style = document.getElementById("iamccs-audio-board-arranger-style");
    if (!style) {
        style = document.createElement("style");
        style.id = "iamccs-audio-board-arranger-style";
        document.head.appendChild(style);
    }
    style.textContent = `
        .iamccs-audio-board {
            box-sizing: border-box;
            width: 100%;
            height: auto;
            min-height: 0;
            max-height: none;
            padding: 10px 10px 18px;
            overflow: hidden;
            contain: layout paint style;
            content-visibility: visible;
            contain-intrinsic-size: 1880px 1220px;
            transform: translateZ(0);
            color: #dbe7ea;
            background: linear-gradient(180deg, #151b1f, #0f1317);
            border: 1px solid rgba(150, 174, 184, .22);
            border-radius: 8px;
            font: 11px/1.25 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        .iamccs-audio-board.is-tool-cut .iamccs-audio-board-track,
        .iamccs-audio-board.is-tool-cut .iamccs-audio-clip {
            cursor: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24'%3E%3Cpath fill='%23f4d49e' stroke='%230d1114' stroke-width='1.25' d='M9.2 7.4a2.6 2.6 0 1 1-5.2 0a2.6 2.6 0 0 1 5.2 0Zm0 9.2a2.6 2.6 0 1 1-5.2 0a2.6 2.6 0 0 1 5.2 0ZM9 8.6l10.4-4.2l.6 1.5l-7.7 4.4l7.7 7.8l-1.1 1.1l-8.8-6.7l-1.2.8l-.8-1.3l1.1-.8l-.7-.5l.8-1.3l.9.5l7.4-4.3l-8 3.2Z'/%3E%3C/svg%3E") 4 4, crosshair;
        }
        .iamccs-audio-board.is-tool-trim .iamccs-audio-board-track,
        .iamccs-audio-board.is-tool-trim .iamccs-audio-clip {
            cursor: ew-resize;
        }
        .iamccs-audio-board.is-fullscreen {
            display: flex;
            flex-direction: column;
            min-height: calc(100vh - 96px);
            height: calc(100vh - 96px);
            padding: 4px 8px 4px !important;
            border: 0;
            box-shadow: none;
        }
        .iamccs-audio-board.is-fullscreen .iamccs-audio-board-dynamic {
            flex: 1 1 0;
            height: auto;
            min-height: 0;
            display: flex;
            flex-direction: column;
            min-width: 0;
            overflow: hidden;
        }
        .iamccs-audio-board-dynamic {
            height: auto;
            min-height: 0;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            padding-bottom: 4px;
        }
        .iamccs-audio-board * { box-sizing: border-box; letter-spacing: 0; }
        .iamccs-audio-board-head,
        .iamccs-audio-board-tools,
        .iamccs-audio-board-status,
        .iamccs-audio-board-transport,
        .iamccs-audio-board-master {
            display: flex;
            align-items: center;
            gap: 6px;
            flex-wrap: wrap;
        }
        .iamccs-audio-board-head {
            display: grid;
            grid-template-columns: minmax(0, 1fr);
            align-items: start;
            justify-content: stretch;
            gap: 6px;
            margin-bottom: 8px;
        }
        .iamccs-audio-board-head > div { justify-self: stretch !important; width: 100%; }
        .iamccs-audio-board-tools {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(88px, 132px));
            justify-content: start;
            gap: 8px;
            width: 100%;
            padding: 9px;
            border: 1px solid rgba(143,208,204,.22);
            border-radius: 7px;
            background: linear-gradient(180deg, rgba(25,35,38,.94), rgba(8,13,15,.94));
            box-shadow: inset 0 1px 0 rgba(255,255,255,.04);
        }
        .iamccs-audio-board-tools button {
            width: 100%;
            min-height: 39px;
            padding: 0 9px;
            font-size: 11px;
        }
        .iamccs-audio-board-title { color: #f4f8f9; font-size: 13px; font-weight: 900; }
        .iamccs-audio-board-sub { color: #8fa5ac; font-size: 10px; font-weight: 750; }
        .iamccs-audio-board button {
            min-height: 25px;
            padding: 0 8px;
            border: 1px solid rgba(143, 208, 204, .35);
            border-radius: 5px;
            background: linear-gradient(180deg, #284a4e, #1a3034);
            color: #ecffff;
            cursor: pointer;
            font-size: 10px;
            font-weight: 850;
            white-space: nowrap;
        }
        .iamccs-audio-board button:hover { border-color: rgba(244, 212, 158, .65); }
        .iamccs-audio-board button.is-active {
            color: #101315;
            background: linear-gradient(180deg, #f2d79a, #c79e59);
            border-color: #ffe6ae;
        }
        .iamccs-audio-board button.is-values {
            color: #fff4c8;
            background: linear-gradient(180deg, rgba(133,93,37,.96), rgba(79,53,24,.96));
            border-color: rgba(255,219,139,.52);
        }
        .iamccs-audio-board button.danger {
            background: linear-gradient(180deg, #65342f, #42221f);
            border-color: rgba(216, 105, 90, .6);
        }
        .iamccs-audio-board-multi {
            display: grid;
            grid-template-columns: auto repeat(5, minmax(88px, 1fr)) auto auto;
            gap: 7px;
            align-items: end;
            margin-top: 7px;
            padding: 8px;
            border: 1px solid rgba(244,212,158,.22);
            border-radius: 7px;
            background: linear-gradient(180deg, rgba(33,38,31,.94), rgba(9,14,13,.94));
        }
        .iamccs-audio-board-multi-title {
            display: grid;
            gap: 1px;
            min-width: 120px;
            color: #fff1ba;
            font-size: 11px;
            font-weight: 950;
        }
        .iamccs-audio-board-multi-title small {
            color: #8fb7b5;
            font-size: 9px;
            font-weight: 800;
        }
        .iamccs-audio-board-multi label {
            display: grid;
            gap: 3px;
            color: #91a5ac;
            font-size: 8px;
            font-weight: 950;
            text-transform: uppercase;
        }
        .iamccs-audio-board-multi select,
        .iamccs-audio-board-multi input {
            min-width: 0;
            height: 26px;
            border: 1px solid rgba(143,208,204,.38);
            border-radius: 5px;
            background: #081115;
            color: #ecffff;
            font-size: 10px;
            font-weight: 850;
            padding: 0 7px;
        }
        .iamccs-audio-board-multi button.is-primary {
            color: #151005;
            background: linear-gradient(180deg, #f2d79a, #c79e59);
            border-color: #ffe6ae;
        }
        .iamccs-audio-board-multi-map {
            grid-column: 1 / -1;
            min-height: 18px;
            padding: 4px 6px;
            border: 1px solid rgba(255,255,255,.08);
            border-radius: 5px;
            background: #05090a;
            color: #b8fff1;
            font: 10px Consolas, monospace;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .iamccs-audio-board-transport,
        .iamccs-audio-board-master {
            margin-top: 7px;
            padding: 8px;
            border: 1px solid rgba(255,255,255,.10);
            border-radius: 7px;
            background: rgba(0,0,0,.20);
        }
        .iamccs-audio-board-transport {
            display: grid;
            grid-template-columns: minmax(190px, 1fr) auto minmax(260px, 1fr);
            align-items: center;
        }
        .iamccs-transport-left,
        .iamccs-transport-center,
        .iamccs-transport-right,
        .iamccs-clip-action-bar {
            display: flex;
            align-items: center;
            gap: 6px;
            flex-wrap: wrap;
        }
        .iamccs-transport-center {
            justify-content: center;
            padding: 4px 10px;
            border: 1px solid rgba(244,212,158,.22);
            border-radius: 7px;
            background: linear-gradient(180deg, rgba(31,42,45,.94), rgba(9,13,16,.94));
        }
        .iamccs-transport-center button { min-width: 44px; min-height: 31px; }
        .iamccs-transport-center button.is-play {
            min-width: 64px;
            color: #101315;
            background: linear-gradient(180deg, #f2d79a, #c79e59);
            border-color: #ffe6ae;
            font-size: 11px;
        }
        .iamccs-transport-center button.is-loop {
            min-width: 54px;
        }
        .iamccs-transport-center button.is-marker {
            min-width: 34px;
            color: #132017;
            background: linear-gradient(180deg, #f8fff2, #cfeec6);
            border-color: rgba(80,180,92,.55);
        }
        .iamccs-transport-right { justify-content: flex-end; min-width: 0; }
        .iamccs-transport-time {
            min-width: 96px;
            color: #ffe2a8;
            font: 900 13px/1 ui-monospace, SFMono-Regular, Consolas, monospace;
        }
        .iamccs-helper-monitor {
            flex: 1 1 220px;
            min-height: 25px;
            padding: 6px 8px;
            color: #b9cbd0;
            background: #0a0e11;
            border: 1px solid rgba(255,255,255,.10);
            border-radius: 5px;
            font: 10px/1.25 ui-monospace, SFMono-Regular, Consolas, monospace;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .iamccs-audio-board-timeline {
            position: relative;
            margin-top: 8px;
            height: auto;
            overflow-x: auto;
            overflow-y: hidden;
            min-width: 0;
            border: 1px solid rgba(255,255,255,.10);
            border-radius: 7px;
            background: #0a0e11;
        }
        .iamccs-audio-board-ruler {
            position: sticky;
            top: 0;
            z-index: 8;
            height: 28px;
            background: linear-gradient(180deg, rgba(26,35,39,.98), rgba(13,17,20,.96));
            border-bottom: 1px solid rgba(255,255,255,.08);
            cursor: crosshair;
            touch-action: none;
        }
        .iamccs-loop-range,
        .iamccs-loop-range-ruler {
            position: absolute;
            pointer-events: none;
            background: linear-gradient(90deg, rgba(106,229,132,.14), rgba(244,212,158,.18));
            border-left: 1px solid rgba(98,232,121,.72);
            border-right: 1px solid rgba(244,212,158,.72);
            box-shadow: inset 0 0 16px rgba(94,225,111,.08);
        }
        .iamccs-loop-range-ruler {
            top: 0;
            bottom: 0;
            z-index: 1;
        }
        .iamccs-loop-range {
            top: 0;
            bottom: 0;
            z-index: 1;
        }
        .iamccs-loop-marker {
            position: absolute;
            top: 2px;
            z-index: 10;
            height: 23px;
            width: 31px;
            transform: translateX(-50%);
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px solid rgba(54,153,71,.68);
            border-radius: 4px;
            color: #0d7a24;
            background: #fafff3;
            box-shadow: 0 0 0 1px rgba(255,255,255,.18), inset 0 0 0 1px rgba(15,102,24,.18);
            font: 950 9px/1 "Courier New", ui-monospace, Consolas, monospace;
            pointer-events: none;
        }
        .iamccs-loop-marker.out {
            color: #9b5b00;
            border-color: rgba(210,147,52,.75);
        }
        .iamccs-audio-board.is-fullscreen .iamccs-audio-board-timeline {
            flex: 1 1 0;
            height: auto !important;
            min-height: 220px;
            overflow-y: auto !important;
        }
        .iamccs-audio-board-track {
            position: relative;
            height: var(--iamccs-track-height, 188px);
            border-bottom: 2px solid rgba(255,255,255,.10);
            background-image:
                linear-gradient(180deg, var(--iamccs-track-fill, rgba(255,255,255,.02)), rgba(0,0,0,.02)),
                linear-gradient(180deg, rgba(255,255,255,.018), rgba(0,0,0,.10)),
                linear-gradient(90deg, rgba(244,212,158,.10) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px);
            background-size: 100% 100%, 100% 100%, calc(var(--iamccs-px-per-frame, 2px) * 24) 100%, calc(var(--iamccs-px-per-frame, 2px) * 6) 100%;
            background-position: 0 0, 0 0, 246px 0, 246px 0;
        }
        .iamccs-audio-board-track:nth-child(even) { background-color: rgba(255,255,255,.012); }
        .iamccs-audio-board-track-label {
            position: sticky;
            left: 0;
            z-index: 15;
            width: 246px;
            height: var(--iamccs-track-height, 188px);
            box-sizing: border-box;
            display: grid;
            grid-template-rows: 21px 26px minmax(62px, 1fr) 30px;
            align-content: start;
            gap: 5px;
            padding: 9px 11px 10px;
            color: #f3e2c0;
            background:
                linear-gradient(90deg, var(--iamccs-track-glow, rgba(244,212,158,.18)), transparent 38%),
                linear-gradient(180deg, rgba(31,39,37,.99), rgba(9,13,15,.99));
            background-color: #0e1518;
            border-right: 2px solid rgba(244,212,158,.28);
            border-bottom: 1px solid rgba(244,212,158,.12);
            box-shadow: inset -10px 0 18px rgba(0,0,0,.22);
            font-size: 9px;
            font-weight: 900;
            overflow: hidden;
        }
        .iamccs-audio-board-track-label:hover {
            background:
                linear-gradient(90deg, rgba(72,122,118,.20), transparent 38%),
                linear-gradient(180deg, rgba(35,48,45,.99), rgba(10,15,16,.99));
        }
        .iamccs-audio-board-track-label.is-selected {
            background:
                linear-gradient(90deg, rgba(205,151,76,.32), transparent 45%),
                linear-gradient(180deg, rgba(68,48,26,.98), rgba(11,13,14,.99));
            border-right-color: rgba(255,218,151,.82);
            box-shadow: inset -10px 0 18px rgba(0,0,0,.22), inset 0 0 0 1px rgba(255,218,151,.30);
        }
        .iamccs-audio-board-track:has(.iamccs-audio-board-track-label.is-selected),
        .iamccs-audio-board-track.is-selected {
            background-image:
                linear-gradient(180deg, rgba(244,212,158,.09), rgba(180,140,60,.05)),
                linear-gradient(180deg, rgba(255,255,255,.018), rgba(0,0,0,.10)),
                linear-gradient(90deg, rgba(244,212,158,.22) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,.07) 1px, transparent 1px);
            border-bottom: 2px solid rgba(244,212,158,.55);
            outline: 1px solid rgba(244,212,158,.28);
            outline-offset: -1px;
        }
        .iamccs-audio-board-track-label.is-muted {
            filter: saturate(.82) brightness(.86);
        }
        .iamccs-audio-board-track-label.is-locked {
            box-shadow: inset -10px 0 18px rgba(0,0,0,.22), inset 0 0 0 1px rgba(130,182,214,.18);
        }
        .iamccs-track-strip-top,
        .iamccs-track-strip-controls,
        .iamccs-track-strip-bottom {
            display: flex;
            align-items: center;
            gap: 5px;
            min-width: 0;
        }
        .iamccs-track-strip-bottom {
            min-height: 30px;
            margin-top: 0;
            padding-bottom: 2px;
        }
        .iamccs-track-strip-top {
            justify-content: space-between;
            padding-bottom: 2px;
            border-bottom: 1px solid rgba(255,255,255,.08);
        }
        .iamccs-track-name {
            display: grid;
            grid-template-columns: auto 1fr;
            align-items: baseline;
            gap: 5px;
            min-width: 96px;
            line-height: 1;
        }
        .iamccs-track-name span { font-size: 12px; color: #fff0c8; }
        .iamccs-track-name small {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 42px;
            height: 15px;
            padding: 0 6px;
            color: #aebec2;
            background: rgba(255,255,255,.055);
            border: 1px solid rgba(255,255,255,.07);
            border-radius: 999px;
            font-size: 7px;
            white-space: nowrap;
        }
        .iamccs-track-strip-controls {
            display: grid;
            grid-template-columns: repeat(7, 1fr);
            gap: 4px;
            margin-top: 1px;
        }
        .iamccs-track-mini {
            min-width: 0;
            width: 100%;
            height: 24px;
            min-height: 24px;
            padding: 0;
            border: 1px solid rgba(143,208,204,.34);
            border-radius: 6px;
            background: linear-gradient(180deg, #193236, #112226);
            color: #dff7f8;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.08), inset 0 -8px 12px rgba(0,0,0,.18);
            font: 950 9px/1 ui-monospace, SFMono-Regular, Consolas, monospace;
            cursor: pointer;
            transition: background .12s ease, border-color .12s ease, color .12s ease, box-shadow .12s ease, transform .08s ease;
        }
        .iamccs-track-mini:hover {
            border-color: rgba(191,236,233,.52);
            background: linear-gradient(180deg, #23454a, #13282d);
        }
        .iamccs-track-mini.is-active {
            color: #161109;
            border-color: #ffe0a4;
            background: linear-gradient(180deg, #f7e3b2, #c78d43);
            box-shadow: inset 0 1px 0 rgba(255,255,255,.26), 0 0 0 1px rgba(255,224,164,.18), 0 0 12px rgba(255,205,126,.16);
        }
        .iamccs-track-mini:active {
            transform: translateY(1px);
        }
        .iamccs-track-color-chip {
            min-width: 0;
            width: 100%;
            height: 24px;
            min-height: 24px;
            padding: 0;
            border: 1px solid rgba(255,226,168,.42);
            border-radius: 6px;
            background:
                linear-gradient(180deg, rgba(255,255,255,.18), rgba(0,0,0,.18)),
                var(--iamccs-track-color, #315f8f);
            color: #fff8d8;
            font: 950 9px/1 ui-monospace, SFMono-Regular, Consolas, monospace;
            cursor: pointer;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.16), 0 0 0 1px rgba(0,0,0,.18);
        }
        .iamccs-track-fx-select {
            position: relative;
            z-index: 20;
            width: 100%;
            min-width: 0;
            height: 30px;
            min-height: 30px;
            padding: 0 8px;
            border: 1px solid rgba(143,208,204,.38);
            border-radius: 6px;
            background: #071012;
            color: #e9ffff;
            font: 900 9px/1 ui-monospace, SFMono-Regular, Consolas, monospace;
        }
        .iamccs-audio-board-meter,
        .iamccs-master-meter {
            position: relative;
            display: grid;
            grid-template-rows: 1fr;
            gap: 1px;
            padding: 1px;
            box-sizing: border-box;
            height: 9px;
            border: 1px solid rgba(255,255,255,.18);
            border-radius: 999px;
            overflow: hidden;
            background: linear-gradient(90deg, rgba(255,255,255,.06), rgba(0,0,0,.45));
        }
        .iamccs-audio-board-meter { width: 50px; }
        .iamccs-master-meter { width: 180px; height: 12px; }
        .iamccs-audio-board-meter.is-stereo { height: 14px; grid-template-rows: repeat(2, 1fr); }
        .iamccs-master-meter.is-stereo { height: 16px; grid-template-rows: repeat(2, 1fr); }
        .iamccs-audio-board-master > .iamccs-master-meter { flex: 0 0 180px; }
        .iamccs-audio-board-master {
            min-height: 58px;
            border-color: rgba(244,212,158,.24);
            background: linear-gradient(180deg, rgba(25,32,35,.94), rgba(7,10,12,.94));
            flex-wrap: nowrap;
        }
        .iamccs-master-title {
            min-width: 92px;
            color: #ffe2a8;
            font-size: 11px;
            font-weight: 950;
        }
        .iamccs-master-controls {
            display: flex;
            align-items: center;
            gap: 7px;
            flex-wrap: wrap;
            flex: 1 1 auto;
            min-width: 0;
        }
        .iamccs-master-fx-select {
            position: relative;
            z-index: 20;
            width: 118px;
            height: 25px;
            border: 1px solid rgba(143,208,204,.30);
            border-radius: 5px;
            background: #091113;
            color: #e9ffff;
            font: 850 9px/1 ui-monospace, SFMono-Regular, Consolas, monospace;
        }
        .iamccs-master-toggle {
            min-height: 25px;
            padding: 0 8px;
            border: 1px solid rgba(143,208,204,.32);
            border-radius: 5px;
            background: rgba(15,26,28,.95);
            color: #a9c2c5;
            font-size: 9px;
            font-weight: 900;
            cursor: pointer;
        }
        .iamccs-master-toggle.is-active {
            color: #101315;
            border-color: #ffe6ae;
            background: linear-gradient(180deg, #f2d79a, #c79e59);
        }
        .iamccs-audio-board-meter > i,
        .iamccs-master-meter > i {
            display: block;
            width: 0%;
            height: auto;
            background: linear-gradient(90deg, #55c7b9 0%, #cfe37c 58%, #f0b857 78%, #dc5c42 100%);
            transition: width .045s linear;
        }
        .iamccs-meter-readout {
            min-width: 52px;
            color: #a9bec4;
            font: 900 10px/1 ui-monospace, SFMono-Regular, Consolas, monospace;
        }
        .iamccs-master-crt {
            margin-left: auto;
            flex: 0 0 132px;
            width: 132px;
            min-width: 132px;
            height: 34px;
            display: grid;
            place-items: center;
            padding: 3px 7px;
            color: #008a1f;
            background: #f8fff0;
            border: 2px solid #b7cfad;
            border-radius: 5px;
            box-shadow: inset 0 0 0 1px rgba(0,130,28,.22), inset 0 -6px 12px rgba(37,100,32,.08), 0 1px 0 rgba(255,255,255,.14);
            font: 950 11px/1 "Courier New", ui-monospace, Consolas, monospace;
            white-space: nowrap;
            text-align: center;
            overflow: hidden;
            text-shadow: 0 0 3px rgba(0,142,31,.22);
        }
        .iamccs-audio-clip {
            position: absolute;
            top: 14px;
            z-index: 1;
            height: calc(var(--iamccs-track-height, 72px) - 28px);
            min-width: 18px;
            border: 1px solid rgba(143, 208, 204, .56);
            border-radius: 6px;
            overflow: hidden;
            background: linear-gradient(180deg, rgba(40,82,116,.98), rgba(24,54,81,.98));
            box-shadow: inset 0 1px 0 rgba(255,255,255,.10), 0 4px 10px rgba(0,0,0,.22);
            cursor: grab;
            user-select: none;
            touch-action: none;
        }
        .iamccs-audio-clip:active { cursor: grabbing; }
        .iamccs-audio-clip.is-trimming { cursor: ew-resize; }
        .iamccs-audio-clip.is-moving { cursor: grabbing; }
        .iamccs-audio-clip.is-selected { border-color: #f4d49e; box-shadow: 0 0 0 1px rgba(244,212,158,.45), 0 4px 12px rgba(0,0,0,.30); }
        .iamccs-audio-clip.is-muted { opacity: .46; }
        .iamccs-audio-clip-name {
            position: absolute;
            left: 12px;
            top: 5px;
            right: 12px;
            z-index: 2;
            color: #fff2cf;
            font: 9px/1 monospace;
            font-weight: 900;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            text-shadow: 0 1px 2px #000;
        }
        .iamccs-audio-clip-time {
            position: absolute;
            left: 12px;
            bottom: 5px;
            z-index: 2;
            color: #b6ced1;
            font: 8px/1 ui-monospace, SFMono-Regular, Consolas, monospace;
            text-shadow: 0 1px 2px #000;
        }
        .iamccs-audio-clip-trim {
            position: absolute;
            right: 10px;
            bottom: 5px;
            z-index: 2;
            color: #ffe2a8;
            font: 8px/1 ui-monospace, SFMono-Regular, Consolas, monospace;
            text-shadow: 0 1px 2px #000;
        }
        .iamccs-clip-source-marker {
            position: absolute;
            top: 0;
            bottom: 0;
            z-index: 28;
            display: none;
            min-width: 72px;
            padding: 4px 6px;
            color: #111712;
            background: linear-gradient(180deg, rgba(255,226,168,.98), rgba(211,151,70,.92));
            border-left: 1px solid rgba(255,255,255,.75);
            border-right: 1px solid rgba(0,0,0,.35);
            font: 900 8px/1.1 ui-monospace, SFMono-Regular, Consolas, monospace;
            pointer-events: none;
            text-shadow: none;
            box-shadow: 0 0 12px rgba(244,212,158,.38);
        }
        .iamccs-audio-clip svg,
        .iamccs-audio-clip canvas {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            opacity: .92;
            pointer-events: none;
        }
        .iamccs-audio-clip canvas { opacity: 1; }
        .iamccs-clip-handle {
            position: absolute;
            top: 0;
            bottom: 0;
            z-index: 30;
            width: 24px;
            background: linear-gradient(90deg, rgba(244,212,158,.34), rgba(244,212,158,.08));
            cursor: ew-resize;
            touch-action: none;
            pointer-events: auto;
        }
        .iamccs-clip-handle.left { left: 0; border-right: 1px solid rgba(244,212,158,.50); }
        .iamccs-clip-handle.right { right: 0; border-left: 1px solid rgba(244,212,158,.50); }
        .iamccs-clip-handle::after {
            content: "";
            position: absolute;
            top: 8px;
            bottom: 8px;
            width: 2px;
            background: rgba(255,226,168,.82);
            box-shadow: 5px 0 0 rgba(255,226,168,.35);
        }
        .iamccs-clip-handle.left::after { left: 6px; }
        .iamccs-clip-handle.right::after { right: 11px; }
        .iamccs-playhead {
            position: absolute;
            top: 0;
            bottom: 0;
            z-index: 6;
            width: 2px;
            background: #ffdc87;
            box-shadow: 0 0 0 1px rgba(0,0,0,.45), 0 0 10px rgba(255,220,135,.35);
            pointer-events: none;
        }
        .iamccs-playhead::before {
            content: "";
            position: absolute;
            top: 0;
            left: -5px;
            border-left: 6px solid transparent;
            border-right: 6px solid transparent;
            border-top: 8px solid #ffdc87;
        }
        .iamccs-clip-action-bar {
            margin-top: 8px;
            padding: 7px;
            background: linear-gradient(180deg, rgba(23,31,35,.94), rgba(8,12,15,.94));
            border: 1px solid rgba(244,212,158,.16);
            border-radius: 7px;
        }
        .iamccs-clip-action-bar strong {
            color: #ffe2a8;
            font-size: 10px;
            font-weight: 950;
            margin-right: 4px;
        }
        .iamccs-audio-board-editor {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(78px, 1fr));
            gap: 6px;
            margin-top: 8px;
            padding: 8px;
            border: 1px solid rgba(255,226,168,.48);
            border-radius: 7px;
            background:
                linear-gradient(180deg, rgba(126,91,39,.96), rgba(57,40,18,.97)),
                repeating-linear-gradient(90deg, rgba(255,255,255,.035) 0 1px, transparent 1px 16px);
            box-shadow: inset 0 1px 0 rgba(255,247,205,.10), 0 6px 18px rgba(0,0,0,.18);
        }
        .iamccs-audio-board-editor::before {
            content: "CLIP VALUES";
            grid-column: 1 / -1;
            color: #fff1ba;
            font: 950 9px/1 ui-monospace, SFMono-Regular, Consolas, monospace;
            letter-spacing: .02em;
            text-transform: uppercase;
            text-shadow: 0 1px 0 rgba(0,0,0,.45);
        }
        .iamccs-audio-board-editor label {
            display: flex;
            flex-direction: column;
            gap: 3px;
            min-width: 0;
            color: #f9ddb0;
            font-size: 8px;
            font-weight: 900;
            text-transform: uppercase;
        }
        .iamccs-audio-board-editor input,
        .iamccs-audio-board-editor select {
            width: 100%;
            min-width: 0;
            height: 23px;
            border: 1px solid rgba(255,232,169,.28);
            border-radius: 4px;
            background: #10120d;
            color: #fff7d8;
            font-size: 10px;
            padding: 2px 5px;
        }
        .iamccs-audio-board-editor input[type="number"],
        .iamccs-master-controls input[type="number"] {
            cursor: ew-resize;
        }
        .iamccs-audio-board-editor input[type="checkbox"] { width: 16px; height: 16px; }
        .iamccs-editor-wide { grid-column: span 2; }
        .iamccs-audio-board-lower {
            display: grid;
            grid-template-columns: minmax(280px, .36fr) minmax(900px, 1.64fr);
            gap: 8px;
            margin: 8px 0 12px;
            flex: 0 0 var(--iamccs-lower-height, 430px);
            width: 100%;
            max-width: 100%;
            height: var(--iamccs-lower-height, 430px);
            min-height: 0;
            min-width: 0;
            box-sizing: border-box;
            overflow: hidden;
        }
        .iamccs-audio-board-lower-handle {
            height: 10px;
            margin: 8px 0 4px;
            border-radius: 999px;
            border: 1px solid rgba(255,226,168,.18);
            background:
                linear-gradient(180deg, rgba(77,97,92,.82), rgba(18,24,26,.94)),
                repeating-linear-gradient(90deg, rgba(255,255,255,.12) 0 10px, transparent 10px 20px);
            box-shadow: inset 0 1px 0 rgba(255,255,255,.08);
            cursor: ns-resize;
        }
        .iamccs-audio-board-lower.no-monitor { grid-template-columns: 1fr; }
        .iamccs-audio-board.is-fullscreen .iamccs-audio-board-lower {
            flex: 0 0 var(--iamccs-lower-height, 360px);
            height: var(--iamccs-lower-height, 360px);
            min-height: 200px;
            overflow: hidden;
        }
        .iamccs-event-console,
        .iamccs-inline-efx {
            width: 100%;
            min-width: 0;
            min-height: 0;
            border: 1px solid rgba(244,212,158,.16);
            border-radius: 7px;
            background:
                linear-gradient(180deg, rgba(18,26,27,.88), rgba(4,8,10,.94)),
                repeating-linear-gradient(0deg, rgba(255,255,255,.025) 0 1px, transparent 1px 4px);
            box-shadow: inset 0 0 0 1px rgba(255,255,255,.035);
            overflow: hidden;
        }
        .iamccs-audio-board.is-fullscreen .iamccs-inline-efx,
        .iamccs-audio-board.is-fullscreen .iamccs-event-console {
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        .iamccs-event-console-head,
        .iamccs-inline-efx-head {
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            padding: 0 8px;
            color: #ffe2a8;
            background: linear-gradient(180deg, rgba(42,55,51,.94), rgba(13,17,19,.94));
            border-bottom: 1px solid rgba(244,212,158,.12);
            font: 950 9px/1 ui-monospace, SFMono-Regular, Consolas, monospace;
            text-transform: uppercase;
        }
        .iamccs-event-console-body {
            height: calc(100% - 24px);
            min-height: 110px;
            padding: 8px;
            color: #a7f1ce;
            font: 10px/1.45 ui-monospace, SFMono-Regular, Consolas, monospace;
            overflow: auto;
            text-shadow: 0 0 6px rgba(85,199,185,.15);
            white-space: pre-wrap;
        }
        .iamccs-event-console-body b { color: #f8d890; }
        .iamccs-inline-efx-grid {
            display: block;
            padding: 8px;
            min-width: 0;
            width: 100%;
            height: calc(100% - 24px);
            box-sizing: border-box;
            overflow-x: auto;
            overflow-y: auto;
        }
        .iamccs-audio-board.is-fullscreen .iamccs-inline-efx-grid {
            flex: 1 1 auto;
            height: auto;
            min-height: 360px;
        }
        .iamccs-device-chain {
            min-width: 0;
            width: max-content;
            min-width: 100%;
            height: auto;
            min-height: 100%;
            display: flex;
            gap: 7px;
            overflow-x: auto;
            overflow-y: visible;
            padding-bottom: 8px;
            align-items: flex-start;
            contain: layout paint;
            box-sizing: border-box;
        }
        .iamccs-audio-board.is-fullscreen .iamccs-device-chain {
            min-height: 100%;
        }
        .iamccs-device-module {
            flex: 0 0 816px;
            min-width: 816px;
            min-height: 0;
            height: auto;
            display: flex;
            gap: 9px;
            align-items: flex-start;
        }
        .iamccs-audio-board.is-fullscreen .iamccs-device-module {
            min-height: 0;
        }
        .iamccs-audio-device {
            flex: 0 0 330px;
            min-height: 320px;
            height: auto;
            padding: 8px 9px 8px;
            border: 1px solid rgba(244,212,158,.24);
            border-radius: 7px;
            display: grid;
            grid-template-rows: auto 1fr;
            align-content: start;
            background:
                radial-gradient(circle at 26px 20px, rgba(255,224,164,.20), transparent 22px),
                linear-gradient(180deg, var(--device-hi, #5a3c29), var(--device-mid, #2c261f) 56%, var(--device-lo, #0b0d0c));
            box-shadow: inset 0 1px 0 rgba(255,255,255,.08), 0 5px 14px rgba(0,0,0,.24);
            color: #f7e6be;
        }
        .iamccs-audio-board.is-fullscreen .iamccs-audio-device {
            min-height: 356px;
        }
        .iamccs-audio-device.iamccs-device-eq {
            flex-basis: 384px;
        }
        .iamccs-audio-device.iamccs-device-eq .iamccs-device-knobs {
            grid-template-columns: repeat(4, minmax(66px, 1fr));
            gap: 10px 8px;
        }
        .iamccs-audio-device.iamccs-device-eq .iamccs-device-knob i {
            width: 40px;
            height: 40px;
        }
        .iamccs-audio-device.iamccs-device-eq .iamccs-device-knob input[type="range"] {
            width: 72px;
            max-width: 100%;
        }
        .iamccs-device-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 4px;
            margin-bottom: 8px;
            font: 950 9px/1 ui-monospace, SFMono-Regular, Consolas, monospace;
            text-transform: uppercase;
        }
        .iamccs-device-head-actions {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .iamccs-device-power {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            border: 1px solid rgba(255,226,168,.48);
            background: radial-gradient(circle at 40% 35%, #ffe6a7, #b97536 42%, #2d1d13 72%);
            box-shadow: 0 0 9px rgba(244,178,88,.28);
            cursor: pointer;
        }
        .iamccs-device-power.is-off {
            background: radial-gradient(circle at 40% 35%, #6d7774, #263033 48%, #101315 74%);
            box-shadow: none;
        }
        .iamccs-device-lamp {
            width: 9px;
            height: 9px;
            border-radius: 50%;
            border: 1px solid rgba(255,226,168,.32);
            background: radial-gradient(circle at 35% 35%, #eaffc8, #72ff8d 45%, #214122 78%);
            box-shadow: 0 0 9px rgba(114,255,141,.72), 0 0 2px rgba(255,255,255,.48) inset;
        }
        .iamccs-device-lamp.is-off {
            background: radial-gradient(circle at 35% 35%, #68716b, #1c2422 58%, #090d0c 82%);
            box-shadow: inset 0 0 2px rgba(255,255,255,.18);
            opacity: .72;
        }
        .iamccs-device-remove {
            width: 20px;
            height: 20px;
            min-height: 20px;
            padding: 0;
            border-radius: 5px;
            color: #ffdfd7;
            background: linear-gradient(180deg, rgba(112,49,43,.95), rgba(56,20,18,.95));
            border-color: rgba(255,128,112,.48);
            font: 950 12px/1 ui-monospace, SFMono-Regular, Consolas, monospace;
            cursor: pointer;
        }
        .iamccs-track-knob-row {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 8px;
            align-items: center;
            min-height: 42px;
        }
        .iamccs-track-effect-row {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            min-height: 56px;
            align-content: flex-start;
            margin-top: 7px;
            padding-top: 6px;
            border-top: 1px solid rgba(255,255,255,.08);
            overflow: hidden;
        }
        .iamccs-track-effect-chip {
            display: inline-flex;
            align-items: center;
            min-height: 20px;
            padding: 0 7px;
            border-radius: 999px;
            border: 1px solid rgba(255,226,168,.22);
            background: linear-gradient(180deg, rgba(29,38,41,.92), rgba(8,11,12,.96));
            color: #cfe0e4;
            font: 900 8px/1 ui-monospace, SFMono-Regular, Consolas, monospace;
            letter-spacing: .02em;
            text-transform: uppercase;
            white-space: nowrap;
        }
        .iamccs-track-effect-chip.is-empty {
            color: #7f9498;
            border-style: dashed;
        }
        .iamccs-track-effect-chip.is-active {
            color: #1d1610;
            border-color: rgba(255,226,168,.52);
            background: linear-gradient(180deg, #f2d79a, #b9823f);
        }
        .iamccs-track-knob-wrap {
            display: grid;
            grid-template-columns: 30px 1fr;
            grid-template-rows: 12px 14px;
            align-items: center;
            column-gap: 6px;
            color: #d9c79f;
            font: 850 7px/1 ui-monospace, SFMono-Regular, Consolas, monospace;
            text-transform: uppercase;
            min-width: 0;
        }
        .iamccs-track-knob-label {
            grid-column: 2;
            grid-row: 1;
            color: #9fb7bc;
            font-size: 7px;
            letter-spacing: 0;
        }
        .iamccs-track-knob {
            grid-column: 1;
            grid-row: 1 / span 2;
            width: 30px;
            height: 30px;
            min-height: 30px;
            padding: 0;
            border-radius: 50%;
            border: 1px solid rgba(255,226,168,.42);
            background:
                conic-gradient(from 220deg, rgba(255,226,168,.12) 0 18%, rgba(255,226,168,.72) var(--iamccs-knob-fill, 50%), rgba(0,0,0,.38) var(--iamccs-knob-fill, 50%) 78%, transparent 78% 100%),
                radial-gradient(circle at 35% 28%, #8a8270, #32342a 55%, #090a09 100%);
            box-shadow: inset 0 1px 2px rgba(255,255,255,.16), inset 0 -3px 6px rgba(0,0,0,.52), 0 1px 4px rgba(0,0,0,.38);
            cursor: ew-resize;
            position: relative;
        }
        .iamccs-track-knob::after {
            content: "";
            position: absolute;
            left: 50%;
            top: 4px;
            width: 2px;
            height: 9px;
            border-radius: 999px;
            background: #ffe2a8;
            transform-origin: 50% 11px;
            transform: translateX(-50%) rotate(var(--iamccs-knob-angle, 0deg));
            box-shadow: 0 0 4px rgba(255,226,168,.45);
        }
        .iamccs-track-knob-readout {
            grid-column: 2;
            grid-row: 2;
            color:#ffe2a8;
            font-size:9px;
            white-space:nowrap;
            overflow:hidden;
            text-overflow:ellipsis;
        }
        .iamccs-eq-switches {
            display: flex;
            gap: 5px;
            margin: 0 0 8px;
        }
        .iamccs-eq-switches button {
            flex: 1 1 0;
            min-height: 22px;
            padding: 0 5px;
            color: #a7bec1;
            background: linear-gradient(180deg, rgba(18,28,31,.94), rgba(7,12,13,.96));
            border-color: rgba(143,208,204,.25);
            font-size: 8px;
        }
        .iamccs-eq-switches button.is-active {
            color: #161109;
            background: linear-gradient(180deg, #f2d79a, #b9823f);
            border-color: #ffe6ae;
        }
        .iamccs-device-knobs {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 14px 11px;
            align-content: start;
            padding-top: 4px;
            padding-bottom: 10px;
        }
        .iamccs-device-knob {
            display: grid;
            grid-template-rows: auto auto auto auto;
            justify-items: center;
            gap: 5px;
            color: #c9b890;
            font: 800 8px/1 ui-monospace, SFMono-Regular, Consolas, monospace;
            text-transform: uppercase;
            min-width: 0;
        }
        .iamccs-device-knob i {
            display: block;
            width: 38px;
            height: 38px;
            border-radius: 50%;
            border: 1px solid rgba(255,226,168,.36);
            background:
                linear-gradient(var(--knob-angle, 140deg), transparent 0 49%, rgba(255,226,168,.75) 50% 53%, transparent 54%),
                radial-gradient(circle at 35% 28%, #5d5b50, #24251f 58%, #090a09 100%);
            box-shadow: inset 0 2px 2px rgba(255,255,255,.10), inset 0 -3px 5px rgba(0,0,0,.42);
        }
        .iamccs-device-knob input[type="range"] {
            width: 68px;
            height: 12px;
            min-height: 12px;
            padding: 0;
            accent-color: #d6a75e;
            background: transparent;
            border: 0;
            cursor: ew-resize;
        }
        .iamccs-device-knob em {
            color: #ffe2a8;
            font-style: normal;
            font-size: 8px;
        }
        .iamccs-inline-efx-panel {
            flex: 1 1 435px;
            min-width: 435px;
            border: 1px solid rgba(255,255,255,.09);
            border-radius: 6px;
            background: #071012;
            overflow: hidden;
        }
        .iamccs-inline-efx-panel label {
            display: block;
            padding: 5px 7px 0;
            color: #ffe2a8;
            font: 950 8px/1 ui-monospace, SFMono-Regular, Consolas, monospace;
            text-transform: uppercase;
        }
        .iamccs-inline-efx-canvas {
            display: block;
            width: 100%;
            height: 318px;
            cursor: crosshair;
        }
        .iamccs-efx-placeholder {
            min-height: 220px;
            display: grid;
            place-items: center;
            color: #8fa5ac;
            border: 1px dashed rgba(244,212,158,.20);
            border-radius: 6px;
            background: repeating-linear-gradient(135deg, rgba(255,255,255,.025) 0 1px, transparent 1px 9px);
            font: 900 10px/1.35 ui-monospace, SFMono-Regular, Consolas, monospace;
            text-align: center;
        }
        .iamccs-audio-board.is-fullscreen .iamccs-inline-efx-canvas,
        .iamccs-audio-board.is-fullscreen .iamccs-efx-placeholder { height: 340px; }
        .iamccs-audio-board-track-label.is-selected {
            border-right-color: #ffe2a8;
            box-shadow: inset 0 0 0 1px rgba(244,212,158,.55), 0 0 22px rgba(244,212,158,.22);
            background: linear-gradient(90deg, rgba(120,90,40,.70) 0px, rgba(65,52,35,.99) 40px, rgba(13,15,15,.99));
        }
        @media (max-width: 980px) {
            .iamccs-audio-board-lower { grid-template-columns: 1fr; }
            .iamccs-inline-efx-grid { grid-template-columns: 1fr; }
        }
    `;
    document.head.appendChild(style);
}

const IAMCCS_AUDIO_BOARD_FIXED_WIDTH = 1880;
const IAMCCS_AUDIO_BOARD_FIXED_HEIGHT = 1650;
const IAMCCS_AUDIO_BOARD_MIN_HEIGHT = 560;
const IAMCCS_AUDIO_BOARD_VISIBLE_TRACKS = 3;
// By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
const TRACK_COLORS = ["#315f8f", "#2f7f71", "#7a5a34", "#6b4c80", "#8a4b45", "#556b36", "#3c6478", "#7a6738"];
const normalizeTrackColor = (color, index = 0) => /^#[0-9a-f]{6}$/i.test(String(color || "")) ? String(color) : TRACK_COLORS[index % TRACK_COLORS.length];
// By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
const trackColorWithAlpha = (color, alpha = .14, index = 0) => {
    const hex = normalizeTrackColor(color, index).replace("#", "");
    const red = parseInt(hex.slice(0, 2), 16);
    const green = parseInt(hex.slice(2, 4), 16);
    const blue = parseInt(hex.slice(4, 6), 16);
    return `rgba(${red}, ${green}, ${blue}, ${Math.max(0, Math.min(1, Number(alpha || 0)))})`;
};
const nextTrackColor = (color, index = 0) => {
    const current = normalizeTrackColor(color, index);
    const pos = TRACK_COLORS.findIndex((item) => item.toLowerCase() === current.toLowerCase());
    return TRACK_COLORS[((pos >= 0 ? pos : index) + 1) % TRACK_COLORS.length];
};
// By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
const EFFECT_CHOICES = [
    ["eq", "EQ"],
    ["compressor", "Compressor"],
    ["limiter", "Limiter"],
    ["gate", "Gate"],
    ["reverb", "Reverb"],
    ["delay", "Delay"],
    ["saturator", "Saturator"],
    ["utility", "Utility"],
    ["stereo", "Stereo"],
    ["deesser", "De-esser"],
    ["transient", "Transient"],
    ["tape", "Tape"],
    ["chorus", "Chorus"],
];
// By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com

function iamccsAudioBoardFixedSizeFromValue(rawValue) {
    let data = {};
    try { data = JSON.parse(String(rawValue || "{}")); } catch {}
    const trackCount = Math.max(1, Number(data?.audioTrackCount || 1));
    const visibleTrackCount = Math.min(trackCount, IAMCCS_AUDIO_BOARD_VISIBLE_TRACKS);
    const trackHeight = Math.max(184, Math.min(240, Number(data?.view?.trackHeight || 188) || 188));
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    // Normal node contains header, transport, master and the visible timeline only.
    // Device editing lives in the dedicated ControlAudEfx Panel; do not reserve a phantom rack.
    const toolbarsH = 190;
    const timelineH = 28 + visibleTrackCount * trackHeight;
    const desiredHeight = Math.max(
        IAMCCS_AUDIO_BOARD_MIN_HEIGHT,
        Math.min(1120, Math.round(toolbarsH + timelineH + 34))
    );
    return [IAMCCS_AUDIO_BOARD_FIXED_WIDTH, desiredHeight];
}

function iamccsAudioBoardFixedSizeFromDom(root, rawValue) {
    const base = iamccsAudioBoardFixedSizeFromValue(rawValue);
    const measuredHeight = Math.max(0, Math.ceil(Number(root?.scrollHeight || root?.offsetHeight || 0)));
    if (!measuredHeight) return base;
    // LiteGraph places the DOM widget starting at y≈34px (title 30 + margin 4) from the node top.
    // node.size[1] must be >= measuredHeight + 34 + bottom_buffer so the element stays inside the border.
    // Using +50 gives ~16px clearance between element bottom and node border.
    // However the Arranger node has 3 output slots (cine_linx, audio_timeline_json, report) each adding
    // ~20px (LiteGraph.NODE_SLOT_HEIGHT), pushing the widget start down by ~60px.
    // Effective y-start ≈ 30 (title) + 60 (3 output slots) + 4 (margin) ≈ 94px.
    // Therefore offset must be >= 94px to contain the element. Using +110 for 16px bottom clearance.
    return [
        base[0],
        Math.max(IAMCCS_AUDIO_BOARD_MIN_HEIGHT, Math.min(IAMCCS_AUDIO_BOARD_FIXED_HEIGHT, measuredHeight + 110)),
    ];
}

function renderAudioBoardArranger(node) {
    ensureAudioBoardArrangerStyles();
    if (node._iamccsAudioBoardReady) {
        const runtimeWidget = findWidget(node, "arranger_data");
        let runtimeValue = String(runtimeWidget?.value || "").trim();
        // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        // If widget value is still empty/default after load, try localStorage backup
        if (!runtimeValue || runtimeValue === "{}") {
            try {
                const lsRaw = localStorage.getItem(`iamccs_audioboard_${node.id}`);
                if (lsRaw) { const lsParsed = JSON.parse(lsRaw); if (lsParsed?.audioSegments?.length) runtimeValue = lsRaw; }
            } catch {}
        }
        if (runtimeValue && typeof node._iamccsAudioBoardApplyRuntimeTimeline === "function") {
            try { node._iamccsAudioBoardApplyRuntimeTimeline(runtimeValue, "rerender_ready_sync"); } catch {}
        }
        const domWidget = (node.widgets || []).find((w) => w?.type === "iamccs_audio_board_arranger" || w?.name === "AudioBoard Arranger");
        const existingRoot = domWidget?.element || domWidget?.inputEl || null;
        const fixed = iamccsAudioBoardFixedSizeFromDom(existingRoot, runtimeValue);
        node._iamccsAudioBoardFixedSize = fixed;
        node.resizable = false;
        node.resizeable = false;
        node.flags = { ...(node.flags || {}), resizable: false };
        node.min_size = fixed.slice();
        if (domWidget) {
            domWidget.computeSize = () => existingRoot?._iamccsAudioFullscreenState
                ? [IAMCCS_AUDIO_BOARD_FIXED_WIDTH, 24]
                : [fixed[0], Math.max(24, fixed[1] - 70)];
        }
        if (Number(node.size?.[0] || 0) !== fixed[0] || Number(node.size?.[1] || 0) !== fixed[1]) {
            if (typeof node.setSize === "function") node.setSize(fixed.slice());
            else node.size = fixed.slice();
        }
        try { node.setDirtyCanvas?.(true, true); app.graph?.setDirtyCanvas?.(true, true); } catch {}
        return;
    }
    console.info("[IAMCCS AudioBoardArranger] render start", {
        nodeId: node?.id,
        type: node?.type,
        comfyClass: node?.comfyClass,
        addDOMWidget: typeof node?.addDOMWidget,
        widgets: (node?.widgets || []).map((w) => ({ name: w?.name, type: w?.type, hidden: w?.hidden })),
    });
    const dataWidget = findWidget(node, "arranger_data");
    console.info("[IAMCCS AudioBoardArranger] arranger_data widget", {
        found: Boolean(dataWidget),
        type: dataWidget?.type,
        valueLength: String(dataWidget?.value || "").length,
    });
    hideWidget(dataWidget);
    const fpsWidget = findWidget(node, "frame_rate");
    const syncPolicyWidget = findWidget(node, "sync_policy");
    hideWidget(fpsWidget);
    hideWidget(syncPolicyWidget);
    const root = document.createElement("div");
    root.className = "iamccs-audio-board";
    root.tabIndex = 0;
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = "audio/*,.wav,.mp3,.m4a,.flac,.ogg,.aac,.aif,.aiff,.wma";
    fileInput.multiple = true;
    fileInput.style.display = "none";
    root.appendChild(fileInput);
    const audioBoardInput = document.createElement("input");
    audioBoardInput.type = "file";
    audioBoardInput.accept = ".json,application/json";
    audioBoardInput.multiple = false;
    audioBoardInput.style.display = "none";
    root.appendChild(audioBoardInput);

    const LABEL_W = 246;
    const VIEWPORT_SECONDS = 26;
    const DEFAULT_SECONDS = VIEWPORT_SECONDS;
    const AUDIO_BOARD_FIXED_WIDTH = IAMCCS_AUDIO_BOARD_FIXED_WIDTH;
    const AUDIO_BOARD_FIXED_HEIGHT = IAMCCS_AUDIO_BOARD_FIXED_HEIGHT;
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    const floatingTrackColorInput = document.createElement("input");
    floatingTrackColorInput.type = "color";
    floatingTrackColorInput.style.cssText = "position:fixed;left:-9999px;top:-9999px;opacity:0;pointer-events:none;z-index:999999;";
    root.appendChild(floatingTrackColorInput);
    let floatingTrackColorTarget = null;
    const openTrackColorPicker = (track, event = null) => {
        const trackIndex = Math.max(0, Number(track || 0));
        const trackState = trackSettings(trackIndex);
        floatingTrackColorTarget = trackIndex;
        floatingTrackColorInput.value = normalizeTrackColor(trackState.color, trackIndex);
        if (event && Number.isFinite(event.clientX) && Number.isFinite(event.clientY)) {
            floatingTrackColorInput.style.left = `${Math.max(8, Math.round(event.clientX) - 18)}px`;
            floatingTrackColorInput.style.top = `${Math.max(8, Math.round(event.clientY) - 18)}px`;
        }
        floatingTrackColorInput.click();
    };
    floatingTrackColorInput.addEventListener("input", () => {
        if (floatingTrackColorTarget == null) return;
        const trackIndex = Math.max(0, Number(floatingTrackColorTarget || 0));
        trackSettings(trackIndex).color = normalizeTrackColor(floatingTrackColorInput.value, trackIndex);
        noteTrackMixLocalState(trackIndex, "color", 1, "color_live");
        scheduleSilentStateWrite("track_color_live");
        scheduleShotboardSync("track_color_live");
        draw();
    });
    floatingTrackColorInput.addEventListener("change", () => {
        if (floatingTrackColorTarget == null) return;
        const trackIndex = Math.max(0, Number(floatingTrackColorTarget || 0));
        trackSettings(trackIndex).color = normalizeTrackColor(floatingTrackColorInput.value, trackIndex);
        noteTrackMixLocalState(trackIndex, "color", 1, "color");
        addEdit(`A${trackIndex + 1} color changed.`);
        writeState("track_color");
        floatingTrackColorTarget = null;
    });
    floatingTrackColorInput.addEventListener("blur", () => {
        floatingTrackColorTarget = null;
        floatingTrackColorInput.style.left = "-9999px";
        floatingTrackColorInput.style.top = "-9999px";
    });
    const DEFAULT_MASTER_CHAIN = [
        { id: "master_eq", type: "eq", enabled: true, amount: .5, params: { low: 0, mid: 0, high: 0, q: 1.2 } },
        { id: "master_comp", type: "compressor", enabled: true, amount: .45, params: { threshold: -18, ratio: 4, attack: 6, release: 180, knee: 12, makeup: 0 } },
        { id: "master_limit", type: "limiter", enabled: true, amount: 1, params: { input: 0, ceiling: -1, lookahead: 3, release: 120, output: 0, softclip: .25 } },
    ];
    const audioBuffers = new Map();
    const audioUrls = new Map();
    const waveformLoading = new Set();
    const contextMenuEl = document.createElement("div");
    contextMenuEl.className = "iamccs-audio-context-menu";
    contextMenuEl.style.display = "none";
    document.body.appendChild(contextMenuEl);
    let selectedId = "";
    let audioContext = null;
    let shotboardSyncTimer = 0;
    let liveStateWriteTimer = 0;
    let graphChangeTimer = 0;
    let dragPxPerFrame = 0;
    let trackMixDebugAt = 0;
    let lastTrackMixLocalAt = 0;
    const TRACK_MIX_LOCAL_GUARD_MS = 1800;
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    const ARRANGER_LS_KEY = `iamccs_audioboard_${node.id}`;
    const saveToLocalStorage = (json) => {
        try { localStorage.setItem(ARRANGER_LS_KEY, String(json || "")); } catch {}
    };
    const loadFromLocalStorage = () => {
        try {
            const raw = localStorage.getItem(ARRANGER_LS_KEY);
            if (!raw) return null;
            const parsed = JSON.parse(raw);
            return parsed && typeof parsed === "object" ? parsed : null;
        } catch { return null; }
    };
    const shotboardSyncSignatures = new Map();
    let transport = {
        playing: false,
        playhead: 0,
        startedAt: 0,
        sources: [],
        analysers: new Map(),
        trackMixNodes: new Map(),
        masterAnalyser: null,
        masterGainNode: null,
        audioContext: null,
        lastMeterSnapshot: { tracks: {}, master: { peak: 0, rms: 0 }, playing: false, playhead: 0 },
        raf: 0,
        efxFrame: 0,
        lastMeterAt: 0,
        pendingLoopRestart: false,
        dom: null,
        helper: "Ready. Import audio, drag clips, trim with handles, then play.",
    };
    const markCanvasDirty = (full = false) => {
        try {
            node.setDirtyCanvas?.(true, Boolean(full));
            app.graph?.setDirtyCanvas?.(true, Boolean(full));
        } catch {}
    };
    const isLiveSyncReason = (reason) => /^(live_|fx_|eq_point|compressor_point)/.test(String(reason || ""));
    const finiteNumber = (value, fallback = 0) => {
        const next = Number(value);
        return Number.isFinite(next) ? next : fallback;
    };
    const smoothAudioParam = (param, value) => {
        if (!param) return;
        const ctx = audioContext;
        const next = Number(value);
        if (!Number.isFinite(next)) return;
        try {
            if (ctx?.currentTime != null && typeof param.cancelScheduledValues === "function") {
                param.cancelScheduledValues(ctx.currentTime);
                param.setTargetAtTime(next, ctx.currentTime, 0.015);
            } else {
                param.value = next;
            }
        } catch {
            try { param.value = next; } catch {}
        }
    };
    const restartPlaybackIfNeeded = (reason = "live_fx_restart") => {
        if (!transport.playing) return;
        transport.helper = `Live FX rebuild: ${reason}`;
        void playPlayback();
    };
    const scheduleSilentStateWrite = (reason = "live") => {
        if (liveStateWriteTimer) window.clearTimeout(liveStateWriteTimer);
        liveStateWriteTimer = window.setTimeout(() => {
            liveStateWriteTimer = 0;
            writeState(reason, false, { quiet: true });
        }, 140);
    };
    const toggleAudioFullscreen = () => {
        if (root._iamccsAudioFullscreenState) {
            const stateFs = root._iamccsAudioFullscreenState;
            root.classList.remove("is-fullscreen");
            root.style.cssText = stateFs.rootCss;
            if (stateFs.placeholder?.parentNode) {
                stateFs.placeholder.parentNode.insertBefore(root, stateFs.placeholder);
                stateFs.placeholder.remove();
            }
            stateFs.overlay?.remove();
            document.removeEventListener("keydown", stateFs.keyHandler);
            root._iamccsAudioFullscreenState = null;
            root.dispatchEvent(new CustomEvent("iamccs:audio-fullscreen", { detail: { open: false } }));
            markCanvasDirty(true);
            return;
        }
        const parent = root.parentNode;
        if (!parent) return;
        const placeholder = document.createComment("iamccs-audio-board-home");
        parent.insertBefore(placeholder, root);
        const overlay = document.createElement("div");
        overlay.style.cssText = "position:fixed;inset:0;z-index:999991;background:rgba(4,8,11,.84);display:flex;flex-direction:column;padding:18px;box-sizing:border-box;pointer-events:auto;";
        const bar = document.createElement("div");
        bar.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;padding:0 0 10px;color:#e8eef2;font:12px Inter,Arial,sans-serif;";
        const title = document.createElement("div");
        title.textContent = "IAMCCS AudioBoard Arranger Full Editor";
        title.style.cssText = "font-weight:900;color:#fff;";
        const close = document.createElement("button");
        close.type = "button";
        close.textContent = "Close Editor";
        close.style.cssText = "min-height:30px;padding:0 12px;border:1px solid rgba(244,212,158,.55);border-radius:6px;background:#284a4e;color:#fff;font-weight:900;cursor:pointer;";
        bar.append(title, close);
        const panel = document.createElement("div");
        panel.style.cssText = "flex:1;min-height:0;overflow:hidden;border:1px solid #557062;border-radius:8px;background:#12191f;box-shadow:0 22px 80px rgba(0,0,0,.65);padding:10px;box-sizing:border-box;";
        const keyHandler = (event) => { if (event.key === "Escape") toggleAudioFullscreen(); };
        document.addEventListener("keydown", keyHandler);
        close.onclick = () => toggleAudioFullscreen();
        root._iamccsAudioFullscreenState = { overlay, placeholder, rootCss: root.style.cssText, keyHandler };
        root.classList.add("is-fullscreen");
        root.style.cssText += "max-height:none!important;height:calc(100vh - 92px)!important;min-height:calc(100vh - 92px)!important;overflow:hidden!important;";
        panel.appendChild(root);
        overlay.append(bar, panel);
        document.body.appendChild(overlay);
        root.dispatchEvent(new CustomEvent("iamccs:audio-fullscreen", { detail: { open: true } }));
        markCanvasDirty(true);
    };

    const newId = (prefix) => `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;
    const fps = () => Math.max(1, Number(fpsWidget?.value || state.frame_rate || 24) || 24);
    const framesToSeconds = (frames) => Number(frames || 0) / fps();
    const secondsToFrames = (seconds) => Math.max(0, Math.round(Number(seconds || 0) * fps()));
    const fmtTime = (frames) => {
        const total = Math.max(0, framesToSeconds(frames));
        const minutes = Math.floor(total / 60);
        const seconds = Math.floor(total % 60);
        const hundredths = Math.floor((total - Math.floor(total)) * 100);
        return `${minutes}:${String(seconds).padStart(2, "0")}.${String(hundredths).padStart(2, "0")}`;
    };
    const closeContextMenu = () => {
        contextMenuEl.style.display = "none";
        contextMenuEl.innerHTML = "";
    };
    const setTool = (tool, helper = "") => {
        view().tool = tool;
        if (helper) transport.helper = helper;
        addEdit(`Tool: ${tool}.`);
        writeState("tool", false);
        draw();
    };
    const parseState = () => {
        const fallback = {
            schema: "iamccs.audio_board_arranger",
            schema_version: 1,
            audioSegments: [],
            audioTrackCount: 5,
            masterAudioGain: 1,
            masterAudioNormalize: false,
            masterMono: false,
            masterBus: { limiter: true, ceilingDb: -1, compressor: .45, width: 1, reverbSend: 0, delaySend: 0, effectChain: JSON.parse(JSON.stringify(DEFAULT_MASTER_CHAIN)) },
            duration_seconds: DEFAULT_SECONDS,
            frame_rate: Number(fpsWidget?.value || 24),
            status: { edits: [] },
            view: { timeZoom: 1, trackHeight: 188, tool: "cursor", visibleSeconds: VIEWPORT_SECONDS },
            showEventMonitor: false,
            showClipValues: false,
            showMultiGeneration: false,
            fullscreenShowDevices: false,
            shotboardAutoSyncEnabled: false,
            lowerPanelHeight: 390,
            trackSettings: [],
            audioBusMode: "all_tracks",
            onlyFirstTrack: false,
            loopEnabled: false,
            loopInFrame: 0,
            loopOutFrame: 0,
            selectedMixer: { type: "track", track: 0 },
        };
        try {
            let rawWidgetValue = String(dataWidget?.value || "").trim();
            // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
            // If the widget has no segments (empty or default), check localStorage backup
            if (!rawWidgetValue || rawWidgetValue === "{}") {
                const lsData = loadFromLocalStorage();
                if (lsData && Array.isArray(lsData.audioSegments) && lsData.audioSegments.length > 0) {
                    rawWidgetValue = JSON.stringify(lsData);
                }
            } else {
                // Peek at segments count without full parse to decide on LS merge
                try {
                    const peek = JSON.parse(rawWidgetValue);
                    if (Array.isArray(peek.audioSegments) && peek.audioSegments.length === 0) {
                        const lsData = loadFromLocalStorage();
                        if (lsData && Array.isArray(lsData.audioSegments) && lsData.audioSegments.length > 0) {
                            rawWidgetValue = JSON.stringify(lsData);
                        }
                    }
                } catch {}
            }
            const data = JSON.parse(rawWidgetValue);
            const out = { ...fallback, ...(data && typeof data === "object" ? data : {}) };
            // A compact workflow draft may retain the clip but lose embedded media
            // fields. Recover only matching segment media from this node's own local
            // backup; never replace current timing/edit truth.
            const localBackup = loadFromLocalStorage();
            const localSegments = Array.isArray(localBackup?.audioSegments) ? localBackup.audioSegments : [];
            if (Array.isArray(out.audioSegments) && localSegments.length) {
                const localById = new Map(localSegments.filter((seg) => seg?.id).map((seg) => [String(seg.id), seg]));
                for (const seg of out.audioSegments) {
                    if (!seg || typeof seg !== "object" || String(seg.audioFile || "").trim() || String(seg.audioB64 || "").trim()) continue;
                    const local = localById.get(String(seg.id || ""));
                    if (!local) continue;
                    if (String(local.audioFile || "").trim()) seg.audioFile = local.audioFile;
                    if (String(local.audioB64 || "").trim()) seg.audioB64 = local.audioB64;
                    if (local.audioUploadType) seg.audioUploadType = local.audioUploadType;
                }
            }
            const storedDuration = Math.max(0, Number(out.duration_seconds || 0));
            const storedFps = Math.max(1, Number(out.frame_rate || fallback.frame_rate || 24) || 24);
            const clipEndFrames = Math.max(0, ...(Array.isArray(out.audioSegments) ? out.audioSegments : []).map((seg) => {
                return Math.max(0, Number(seg?.start || 0)) + Math.max(1, Number(seg?.length || 1));
            }));
            const clipDuration = clipEndFrames > 0 ? clipEndFrames / storedFps : 0;
            const computedDuration = Math.max(VIEWPORT_SECONDS, clipDuration);
            out.duration_seconds = storedDuration > 0 && storedDuration <= computedDuration * 1.35
                ? Math.max(VIEWPORT_SECONDS, storedDuration)
                : computedDuration;
            out.audioTrackCount = Math.max(1, Number(out.audioTrackCount || 1));
            out.masterBus = { ...fallback.masterBus, ...(out.masterBus && typeof out.masterBus === "object" ? out.masterBus : {}) };
            out.masterBus.effectChain = Array.isArray(out.masterBus.effectChain)
                ? out.masterBus.effectChain
                : JSON.parse(JSON.stringify(DEFAULT_MASTER_CHAIN));
            out.trackSettings = Array.isArray(out.trackSettings) ? out.trackSettings : [];
            out.audioBusMode = (out.audioBusMode === "only_first" || out.onlyFirstTrack) ? "only_first" : "all_tracks";
            out.onlyFirstTrack = out.audioBusMode === "only_first";
            out.masterMono = Boolean(out.masterMono);
            out.loopEnabled = Boolean(out.loopEnabled);
            out.loopInFrame = Math.max(0, Math.round(Number(out.loopInFrame || 0)));
            out.loopOutFrame = Math.max(0, Math.round(Number(out.loopOutFrame || 0)));
            out.selectedMixer = out.selectedMixer && typeof out.selectedMixer === "object" ? out.selectedMixer : fallback.selectedMixer;
            out.selectedMixer.track = Math.max(0, Math.min(out.audioTrackCount - 1, Number(out.selectedMixer.track || 0)));
            out.showEventMonitor = Boolean(out.showEventMonitor);
            out.showClipValues = Boolean(out.showClipValues);
            out.showMultiGeneration = Boolean(out.showMultiGeneration);
            out.fullscreenShowDevices = Boolean(out.fullscreenShowDevices);
            out.shotboardAutoSyncEnabled = Boolean(out.shotboardAutoSyncEnabled);
            out.lowerPanelHeight = Math.max(250, Math.min(760, Number(out.lowerPanelHeight || 390) || 390));
            return out;
        } catch {
            return fallback;
        }
    };
    let state = parseState();
    const patchIncomingTrackMixState = (rawValue, reason = "runtime_sync") => {
        const protectLocalTrackMix = /(runtime_sync|rerender_ready_sync|onExecuted|history_sync)/.test(String(reason || ""))
            || Date.now() - lastTrackMixLocalAt < TRACK_MIX_LOCAL_GUARD_MS;
        if (!protectLocalTrackMix) return rawValue;
        try {
            let currentWidgetTrackSettings = [];
            try {
                const currentWidgetData = JSON.parse(String(dataWidget?.value || "{}"));
                currentWidgetTrackSettings = Array.isArray(currentWidgetData?.trackSettings) ? currentWidgetData.trackSettings : [];
            } catch {}
            const localTrackSettings = Array.isArray(currentWidgetTrackSettings) && currentWidgetTrackSettings.length
                ? currentWidgetTrackSettings
                : state.trackSettings;
            if (!Array.isArray(localTrackSettings) || !localTrackSettings.length) return rawValue;
            const data = JSON.parse(String(rawValue || "{}"));
            let currentWidgetData = {};
            try { currentWidgetData = JSON.parse(String(dataWidget?.value || "{}")); } catch {}
            const currentSegments = Array.isArray(currentWidgetData?.audioSegments) ? currentWidgetData.audioSegments : [];
            const incomingSegments = Array.isArray(data?.audioSegments) ? data.audioSegments : [];
            const hasCurrentMedia = currentSegments.some((seg) => String(seg?.audioFile || "").trim() || String(seg?.audioB64 || "").trim());
            const hasIncomingMedia = incomingSegments.some((seg) => String(seg?.audioFile || "").trim() || String(seg?.audioB64 || "").trim());
            if (hasCurrentMedia && !hasIncomingMedia) {
                data.audioSegments = JSON.parse(JSON.stringify(currentSegments));
                data.audioTrackCount = Math.max(1, Number(currentWidgetData.audioTrackCount || data.audioTrackCount || 1));
                data.duration_seconds = Math.max(Number(data.duration_seconds || 0), Number(currentWidgetData.duration_seconds || 0));
            }
            data.trackSettings = Array.isArray(data.trackSettings) ? data.trackSettings : [];
            if (!data.trackSettings.length) {
                data.trackSettings = JSON.parse(JSON.stringify(localTrackSettings || []));
                return JSON.stringify(data, null, 2);
            }
            let merged = 0;
            const protectedKeys = ["volume", "gainDb", "pan", "mute", "solo", "normalize", "bypassEffects", "reverb", "reverbSend", "lock", "color", "effectChain", "noAutoEq"];
            const sameValue = (left, right) => {
                try { return JSON.stringify(left) === JSON.stringify(right); }
                catch { return left === right; }
            };
            const cloneValue = (value) => {
                if (value == null || typeof value !== "object") return value;
                try { return JSON.parse(JSON.stringify(value)); }
                catch { return value; }
            };
            for (let trackIndex = 0; trackIndex < localTrackSettings.length; trackIndex += 1) {
                const localTrack = localTrackSettings[trackIndex];
                if (!localTrack || typeof localTrack !== "object") continue;
                const incoming = data.trackSettings[trackIndex] && typeof data.trackSettings[trackIndex] === "object" ? data.trackSettings[trackIndex] : {};
                const nextTrack = { ...incoming };
                let changed = false;
                for (const key of protectedKeys) {
                    if (!Object.prototype.hasOwnProperty.call(localTrack, key)) continue;
                    let localValue = cloneValue(localTrack[key]);
                    if (key === "volume") localValue = Math.max(0, Math.min(2, finiteNumber(localValue, 1)));
                    if (key === "gainDb") localValue = Math.max(-24, Math.min(24, finiteNumber(localValue, 0)));
                    if (key === "pan") localValue = Math.max(-1, Math.min(1, finiteNumber(localValue, 0)));
                    if (["mute", "solo", "normalize", "bypassEffects", "reverb", "lock", "noAutoEq"].includes(key)) localValue = Boolean(localValue);
                    if (key === "reverbSend") localValue = Math.max(0, Math.min(1, finiteNumber(localValue, 0)));
                    if (!sameValue(nextTrack[key], localValue)) changed = true;
                    nextTrack[key] = localValue;
                }
                if (!changed) continue;
                data.trackSettings[trackIndex] = nextTrack;
                merged += 1;
            }
            if (merged && Date.now() - trackMixDebugAt > 220) {
                trackMixDebugAt = Date.now();
                addEdit(`[debug] preserved local track mix over ${reason} on ${merged} track${merged === 1 ? "" : "s"}`);
            }
            return merged ? JSON.stringify(data, null, 2) : rawValue;
        } catch {
            return rawValue;
        }
    };
    const applyRuntimeTimeline = (rawValue, reason = "runtime_sync") => {
        const nextValue = String(patchIncomingTrackMixState(rawValue, reason) || "").trim();
        if (!nextValue) return false;
        if (!setWidgetValue(node, "arranger_data", nextValue)) return false;
        state = parseState();
        if (!segments().some((seg) => String(seg?.id || "") === String(selectedId || ""))) {
            selectedId = segments()[0]?.id || "";
        }
        if (typeof node._iamccsAudioBoardLastRuntimeReason !== "string") {
            node._iamccsAudioBoardLastRuntimeReason = "";
        }
        node._iamccsAudioBoardLastRuntimeReason = String(reason || "runtime_sync");
        draw();
        markCanvasDirty(true);
        return true;
    };
    node._iamccsAudioBoardApplyRuntimeTimeline = applyRuntimeTimeline;
    const historySyncTimestamp = (entry) => {
        const messages = Array.isArray(entry?.status?.messages) ? entry.status.messages : [];
        for (let index = messages.length - 1; index >= 0; index -= 1) {
            const detail = messages[index]?.[1];
            const timestamp = Number(detail?.timestamp || 0);
            if (timestamp > 0) return timestamp;
        }
        return 0;
    };
    const loadLatestRuntimeTimelineFromHistory = async () => {
        const response = await api.fetchApi("/history");
        if (!response?.ok) return null;
        const history = await response.json();
        const entries = Object.entries(history || {}).sort((left, right) => historySyncTimestamp(right[1]) - historySyncTimestamp(left[1]));
        for (const [promptId, entry] of entries) {
            const output = entry?.outputs?.[String(node.id)];
            const rawValue = firstUiValue(output, "iamccs_audio_board");
            if (rawValue) return { promptId, rawValue };
        }
        return null;
    };
    const scheduleHistoryRuntimeSync = (reason = "history_sync") => {
        if (node._iamccsAudioBoardHistorySyncTimer) {
            window.clearTimeout(node._iamccsAudioBoardHistorySyncTimer);
        }
        node._iamccsAudioBoardHistorySyncTimer = window.setTimeout(async () => {
            node._iamccsAudioBoardHistorySyncTimer = 0;
            // History is only a recovery source for an empty node. Once the current
            // AudioBoard owns real media, its widget state is the authoritative truth.
            const currentWidgetState = (() => {
                try { return JSON.parse(String(dataWidget?.value || "{}")); } catch { return state || {}; }
            })();
            const currentOwnsMedia = (Array.isArray(currentWidgetState?.audioSegments) ? currentWidgetState.audioSegments : [])
                .some((seg) => String(seg?.audioFile || "").trim() || String(seg?.audioB64 || "").trim());
            if (currentOwnsMedia) return;
            if (node._iamccsAudioBoardHistorySyncInFlight) return;
            node._iamccsAudioBoardHistorySyncInFlight = true;
            try {
                const latest = await loadLatestRuntimeTimelineFromHistory();
                if (!latest || latest.promptId === node._iamccsAudioBoardLastHistoryPromptId) return;
                const applied = applyRuntimeTimeline(latest.rawValue, reason);
                if (!applied) return;
                node._iamccsAudioBoardLastHistoryPromptId = latest.promptId;
                node.properties = node.properties || {};
                node.properties.iamccsAudioBoardRuntimeSyncAt = new Date().toISOString();
                renderIamccsAudioDialogueNode(node, reason);
            } catch (err) {
                console.warn("[IAMCCS AudioBoardArranger] history runtime sync failed", err);
            } finally {
                node._iamccsAudioBoardHistorySyncInFlight = false;
            }
        }, 90);
    };
    if (!node._iamccsAudioBoardStatusListenerBound && typeof api?.addEventListener === "function") {
        node._iamccsAudioBoardStatusListenerBound = true;
        api.addEventListener("status", (event) => {
            const detail = event?.detail || {};
            const queueRemaining = Number(detail?.exec_info?.queue_remaining ?? detail?.status?.exec_info?.queue_remaining ?? -1);
            if (queueRemaining === 0) scheduleHistoryRuntimeSync("status_idle_history_sync");
        });
    }
    // History may populate a genuinely empty node, but must never replace the current
    // AudioBoard truth after refresh.
    if (!(Array.isArray(state?.audioSegments) ? state.audioSegments : []).some((seg) => String(seg?.audioFile || "").trim() || String(seg?.audioB64 || "").trim())) {
        scheduleHistoryRuntimeSync("render_bootstrap_history_sync");
    }
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    const segments = () => Array.isArray(state.audioSegments) ? state.audioSegments : (state.audioSegments = []);
    const trackSettings = (track) => {
        state.trackSettings = Array.isArray(state.trackSettings) ? state.trackSettings : [];
        const index = Math.max(0, Number(track || 0));
        state.trackSettings[index] = {
            mute: false,
            solo: false,
            normalize: false,
            reverb: false,
            reverbSend: 0,
            volume: 1,
            gainDb: 0,
            pan: 0,
            bypassEffects: false,
            color: normalizeTrackColor("", index),
            lock: false,
            noAutoEq: true,
            effectChain: [],
            ...(state.trackSettings[index] && typeof state.trackSettings[index] === "object" ? state.trackSettings[index] : {}),
        };
        state.trackSettings[index].effectChain = Array.isArray(state.trackSettings[index].effectChain)
            ? state.trackSettings[index].effectChain
            : [];
        return state.trackSettings[index];
    };
    const selectedMixer = () => {
        state.selectedMixer = state.selectedMixer && typeof state.selectedMixer === "object" ? state.selectedMixer : { type: "track", track: 0 };
        state.selectedMixer.type = state.selectedMixer.type === "track" ? "track" : "master";
        state.selectedMixer.track = Math.max(0, Math.min(Math.max(0, Number(state.audioTrackCount || 1) - 1), Number(state.selectedMixer.track || 0)));
        return state.selectedMixer;
    };
    const effectParamSpecs = (type) => ({
        eq: [
            ["low", "LOW", -24, 24, .5, 0, "dB"],
            ["mid", "MID", -24, 24, .5, 0, "dB"],
            ["high", "HIGH", -24, 24, .5, 0, "dB"],
            ["q", "Q", .2, 8, .1, 1.2, ""],
            ["lowCutFreq", "LC Hz", 20, 400, 5, 80, "Hz"],
            ["lowCutLevel", "LC LVL", -48, 0, 1, -30, "dB"],
            ["highCutFreq", "HC Hz", 2000, 22000, 250, 12000, "Hz"],
            ["highCutLevel", "HC LVL", -48, 0, 1, -30, "dB"],
        ],
        compressor: [
            ["threshold", "THR", -60, 0, 1, -18, "dB"],
            ["ratio", "RATIO", 1, 20, .5, 4, ":1"],
            ["attack", "ATK", 1, 100, 1, 6, "ms"],
            ["release", "REL", 20, 800, 10, 180, "ms"],
            ["knee", "KNEE", 0, 36, 1, 12, "dB"],
            ["makeup", "MAKE", -12, 12, .5, 0, "dB"],
        ],
        limiter: [
            ["input", "IN", -12, 12, .5, 0, "dB"],
            ["ceiling", "CEIL", -12, 0, .5, -1, "dB"],
            ["lookahead", "LOOK", 0, 10, .5, 3, "ms"],
            ["release", "REL", 20, 800, 10, 120, "ms"],
            ["output", "OUT", -12, 6, .5, 0, "dB"],
            ["softclip", "SOFT", 0, 1, .05, .25, ""],
        ],
        gate: [
            ["threshold", "THR", -80, 0, 1, -45, "dB"],
            ["attack", "ATK", 1, 80, 1, 8, "ms"],
            ["hold", "HOLD", 0, 400, 10, 80, "ms"],
            ["release", "REL", 20, 1200, 10, 260, "ms"],
        ],
        reverb: [
            ["size", "SIZE", 0, 1, .01, .45, ""],
            ["decay", "DECAY", .1, 8, .1, 1.8, "s"],
            ["damp", "DAMP", 0, 1, .01, .35, ""],
            ["mix", "MIX", 0, 1, .01, .18, ""],
        ],
        delay: [
            ["time", "TIME", .03, 1.5, .01, .18, "s"],
            ["feedback", "FDBK", 0, .9, .01, .28, ""],
            ["filter", "FILT", 200, 12000, 100, 4200, "Hz"],
            ["mix", "MIX", 0, 1, .01, .2, ""],
        ],
        saturator: [
            ["drive", "DRIVE", 0, 24, .5, 4, "dB"],
            ["color", "COLOR", 0, 1, .01, .45, ""],
            ["tone", "TONE", 200, 12000, 100, 2600, "Hz"],
            ["mix", "MIX", 0, 1, .01, .5, ""],
        ],
        utility: [
            ["gain", "GAIN", -24, 24, .5, 0, "dB"],
            ["width", "WIDTH", 0, 2, .05, 1, ""],
            ["pan", "PAN", -1, 1, .05, 0, ""],
            ["mono", "MONO", 0, 1, 1, 0, ""],
        ],
        stereo: [
            ["width", "WIDTH", 0, 2, .05, 1.15, ""],
            ["angle", "ANGLE", -45, 45, 1, 0, "deg"],
            ["bassMono", "BASS", 0, 300, 10, 120, "Hz"],
            ["mix", "MIX", 0, 1, .01, .5, ""],
        ],
        deesser: [
            ["freq", "FREQ", 2500, 10000, 100, 6200, "Hz"],
            ["threshold", "THR", -60, 0, 1, -26, "dB"],
            ["range", "RANGE", 0, 24, .5, 8, "dB"],
            ["mix", "MIX", 0, 1, .01, .75, ""],
        ],
        transient: [
            ["attack", "ATK", -1, 1, .05, .15, ""],
            ["sustain", "SUS", -1, 1, .05, 0, ""],
            ["drive", "DRIVE", 0, 12, .5, 0, "dB"],
            ["mix", "MIX", 0, 1, .01, .6, ""],
        ],
        tape: [
            ["bias", "BIAS", 0, 1, .01, .45, ""],
            ["wow", "WOW", 0, 1, .01, .08, ""],
            ["sat", "SAT", 0, 1, .01, .35, ""],
            ["hiss", "HISS", 0, 1, .01, .04, ""],
        ],
        chorus: [
            ["rate", "RATE", .05, 8, .05, .8, "Hz"],
            ["depth", "DEPTH", 0, 1, .01, .35, ""],
            ["phase", "PHASE", 0, 180, 1, 90, "deg"],
            ["mix", "MIX", 0, 1, .01, .22, ""],
        ],
    }[type] || [
        ["amount", "AMT", 0, 1, .01, .5, ""],
        ["mix", "MIX", 0, 1, .01, .5, ""],
    ]);
    const normalizeEffect = (fx) => {
        const out = fx && typeof fx === "object" ? fx : {};
        out.id = out.id || newId(`fx_${out.type || "device"}`);
        out.type = String(out.type || "utility");
        out.enabled = out.enabled !== false;
        out.params = out.params && typeof out.params === "object" ? out.params : {};
        for (const [key, , , , , fallback] of effectParamSpecs(out.type)) {
            if (out.params[key] == null || Number.isNaN(Number(out.params[key]))) out.params[key] = fallback;
        }
        if (out.type === "eq") {
            out.params.lowCut = Boolean(out.params.lowCut);
            out.params.highCut = Boolean(out.params.highCut);
            out.params.lowFreq = Math.max(40, Math.min(600, Number(out.params.lowFreq || 140)));
            out.params.midFreq = Math.max(300, Math.min(5000, Number(out.params.midFreq || 1200)));
            out.params.highFreq = Math.max(2000, Math.min(16000, Number(out.params.highFreq || 6200)));
            if (out.params.lowFreq >= out.params.midFreq) out.params.midFreq = Math.min(5000, out.params.lowFreq + 160);
            if (out.params.midFreq >= out.params.highFreq) out.params.highFreq = Math.min(16000, out.params.midFreq + 500);
            out.params.lowCutFreq = Math.max(20, Math.min(400, Number(out.params.lowCutFreq || 80)));
            out.params.highCutFreq = Math.max(2000, Math.min(22000, Number(out.params.highCutFreq || 12000)));
            out.params.lowCutLevel = Math.max(-48, Math.min(0, Number(out.params.lowCutLevel ?? -30)));
            out.params.highCutLevel = Math.max(-48, Math.min(0, Number(out.params.highCutLevel ?? -30)));
        }
        out.amount = Math.max(0, Math.min(1, Number(out.amount ?? .5)));
        return out;
    };
    const eqBandFrequencies = (fx) => {
        normalizeEffect(fx);
        return {
            low: Number(fx?.params?.lowFreq || 140),
            mid: Number(fx?.params?.midFreq || 1200),
            high: Number(fx?.params?.highFreq || 6200),
        };
    };
    const eqVisualRatioForFreq = (hz) => {
        const value = Math.max(20, Math.min(22000, Number(hz || 20)));
        const bands = [
            [20, 250],
            [250, 4000],
            [4000, 22000],
        ];
        for (let index = 0; index < bands.length; index += 1) {
            const [minHz, maxHz] = bands[index];
            if (value <= maxHz || index === bands.length - 1) {
                const local = (Math.log10(value) - Math.log10(minHz)) / Math.max(.0001, Math.log10(maxHz) - Math.log10(minHz));
                return Math.max(0, Math.min(1, (index + local) / bands.length));
            }
        }
        return .5;
    };
    const eqFreqForVisualRatio = (ratio) => {
        const bands = [
            [20, 250],
            [250, 4000],
            [4000, 22000],
        ];
        const clamped = Math.max(0, Math.min(0.999999, Number(ratio || 0)));
        const scaled = clamped * bands.length;
        const index = Math.max(0, Math.min(bands.length - 1, Math.floor(scaled)));
        const [minHz, maxHz] = bands[index];
        const local = scaled - index;
        return Math.round(Math.pow(10, Math.log10(minHz) + local * (Math.log10(maxHz) - Math.log10(minHz))));
    };
    const eqVisualX = (hz, width) => Math.max(0, Math.min(Number(width || 0), eqVisualRatioForFreq(hz) * Number(width || 0)));
    const chainForTarget = () => {
        const target = selectedMixer();
        if (target.type === "track") {
            const chain = trackSettings(target.track).effectChain;
            chain.forEach(normalizeEffect);
            return chain;
        }
        state.masterBus = state.masterBus && typeof state.masterBus === "object" ? state.masterBus : {};
        state.masterBus.effectChain = Array.isArray(state.masterBus.effectChain)
            ? state.masterBus.effectChain
            : JSON.parse(JSON.stringify(DEFAULT_MASTER_CHAIN));
        state.masterBus.effectChain.forEach(normalizeEffect);
        return state.masterBus.effectChain;
    };
    const masterEffectByType = (type) => chainForTarget().find((fx) => String(fx?.type || "") === String(type || "")) || null;
    const syncMasterFlagsFromChain = () => {
        const compFx = masterEffectByType("compressor");
        const limiterFx = masterEffectByType("limiter");
        state.masterBus.compressor = compFx && compFx.enabled !== false ? Math.max(0.01, Number(state.masterBus?.compressor || compFx.amount || .45)) : 0;
        state.masterBus.limiter = Boolean(limiterFx && limiterFx.enabled !== false);
    };
    const addEffectToChain = (type, target = selectedMixer()) => {
        const cleanType = String(type || "").trim();
        if (!cleanType) return;
        const chain = target.type === "track" ? trackSettings(target.track).effectChain : chainForTarget();
        chain.push(normalizeEffect({ id: newId(`fx_${cleanType}`), type: cleanType, enabled: true, amount: cleanType === "limiter" ? 1 : .5 }));
        if (target.type === "master") {
            if (cleanType === "compressor") state.masterBus.compressor = Math.max(.45, Number(state.masterBus.compressor || 0));
            if (cleanType === "limiter") state.masterBus.limiter = true;
            syncMasterFlagsFromChain();
        }
        addEdit(`Inserted ${cleanType} on ${target.type === "track" ? `A${target.track + 1}` : "Master"}.`);
        writeState("insert_effect");
        restartPlaybackIfNeeded(`insert_${cleanType}`);
        draw();
    };
    const removeEffectFromChain = (fx, target = selectedMixer()) => {
        const chain = target.type === "track" ? trackSettings(target.track).effectChain : chainForTarget();
        const index = chain.findIndex((item) => item === fx || item.id === fx.id);
        if (index < 0) return;
        const [removed] = chain.splice(index, 1);
        if (removed?.type === "eq") {
            if (target.type === "track") trackSettings(target.track).noAutoEq = true;
            else state.masterBus.noAutoEq = true;
        }
        if (target.type === "master") syncMasterFlagsFromChain();
        addEdit(`Removed ${String(removed?.type || "device")} from ${target.type === "track" ? `A${target.track + 1}` : "Master"}.`);
        writeState("remove_effect");
        restartPlaybackIfNeeded(`remove_${String(removed?.type || "fx")}`);
        draw();
    };
    const hasMedia = (seg) => Boolean(seg && (String(seg.audioFile || "").trim() || String(seg.audioB64 || "").trim()));
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    const isDialogueInjectPlaceholder = (seg) => Boolean(
        seg
        && (
            seg.pendingTTS === true
            || String(seg.purpose || "").trim() === "dialogue_pending_tts"
            || String(seg.source || "").trim() === "IAMCCS_DialogueTagEditor_UI_Inject"
        )
    );
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    const stripDialogueInjectPlaceholders = (items, options = {}) => {
        const list = Array.isArray(items) ? items : [];
        const hasConcreteMedia = list.some((seg) => !isDialogueInjectPlaceholder(seg) && hasMedia(seg) && !seg?.mute);
        if (!options.force && !hasConcreteMedia) return { segments: list, removed: 0, hasConcreteMedia };
        const segments = list.filter((seg) => !isDialogueInjectPlaceholder(seg));
        return { segments, removed: Math.max(0, list.length - segments.length), hasConcreteMedia };
    };
    const effectiveGain = (seg) => Math.max(0, Math.min(4, Number(seg?.gain ?? 1) || 1));
    const effectiveDisplayGain = (seg) => Math.max(.1, Math.min(4, effectiveGain(seg)));
    const isStereoSegment = (seg) => String(seg?.channelMode || "").toLowerCase() === "stereo" || Number(seg?.channelCount || 1) > 1 || Math.abs(Number(seg?.stereoWidth ?? 1) - 1) > .05;
    const trackHasStereoContent = (track) => segments().some((seg) => Number(seg?.track || 0) === Number(track || 0) && isStereoSegment(seg));
    const peakFor = (seg) => {
        const peaks = Array.isArray(seg?.waveformPeaks) ? seg.waveformPeaks.map((item) => {
            if (item && typeof item === "object") return Math.max(Math.abs(Number(item.min) || 0), Math.abs(Number(item.max) || 0));
            return Math.abs(Number(item) || 0);
        }) : [];
        return peaks.length ? Math.min(1, Math.max(...peaks) * effectiveGain(seg)) : 0;
    };
    const linkedShotboardNodes = () => {
        const found = [];
        const foundIds = new Set();
        const visited = new Set();
        const queue = [node];
        while (queue.length) {
            const current = queue.shift();
            const currentId = Number(current?.id || 0);
            if (!current || !currentId || visited.has(currentId)) continue;
            visited.add(currentId);
            for (const output of current.outputs || []) {
                for (const linkId of output.links || []) {
                    const link = app.graph?.links?.[linkId];
                    const target = link ? app.graph?.getNodeById?.(link.target_id) : null;
                    const targetId = Number(target?.id || 0);
                    const type = String(target?.comfyClass || target?.type || "");
                    if (!target || !targetId) continue;
                    if (type === "IAMCCS_CineShotboardPlannerV3") {
                        if (!foundIds.has(targetId)) {
                            foundIds.add(targetId);
                            found.push(target);
                        }
                        continue;
                    }
                    queue.push(target);
                }
            }
        }
        return found;
    };
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    const upstreamDialogueTagEditor = () => {
        const visited = new Set();
        const queue = [node];
        while (queue.length) {
            const current = queue.shift();
            const currentId = Number(current?.id || 0);
            if (!current || !currentId || visited.has(currentId)) continue;
            visited.add(currentId);
            for (const input of current.inputs || []) {
                const link = input?.link != null ? app.graph?.links?.[input.link] : null;
                const origin = link ? app.graph?.getNodeById?.(link.origin_id) : null;
                const originId = Number(origin?.id || 0);
                const type = String(origin?.comfyClass || origin?.type || "");
                if (!origin || !originId || visited.has(originId)) continue;
                if (type === "IAMCCS_DialogueTagEditor") return origin;
                queue.push(origin);
            }
        }
        return null;
    };
    const dialoguePromptSyncSource = () => {
        const source = upstreamDialogueTagEditor();
        const widget = source ? findWidget(source, "dialogue_data") : null;
        try {
            const data = JSON.parse(String(widget?.value || "{}"));
            const lines = Array.isArray(data?.lines) ? data.lines : [];
            const localPrompts = lines
                .map((line) => String(line?.local_prompt || line?.shot_prompt || "").trim())
                .filter(Boolean);
            return {
                source,
                globalPrompt: String(data?.global_prompt || data?.prompt || "").trim(),
                localPrompts,
                promptRelayEnabled: localPrompts.length > 0,
            };
        } catch {
            return null;
        }
    };
    const dialoguePromptSyncLabel = () => {
        const source = dialoguePromptSyncSource();
        if (!source) return "Prompt sync: OFF";
        const sourceName = String(source?.title || source?.type || "DialogueTagEditor").trim() || "DialogueTagEditor";
        return `Prompt sync: ON (${sourceName})`;
    };
    const shotboardDurationFrames = () => {
        let maxFrames = 0;
        for (const board of linkedShotboardNodes()) {
            const widget = findWidget(board, "timeline_data");
            try {
                const data = JSON.parse(String(widget?.value || "{}"));
                const boardFps = Math.max(1, Number(data.frame_rate || fps()) || fps());
                const byDuration = Math.round(Math.max(0, Number(data.duration_seconds || 0)) * boardFps);
                const byVisual = Math.max(0, ...(Array.isArray(data.segments) ? data.segments : []).map((seg) => Number(seg.start || 0) + Number(seg.length || 1)));
                maxFrames = Math.max(maxFrames, byDuration, byVisual);
            } catch {}
        }
        return maxFrames;
    };
    const totalFrames = () => {
        const end = Math.max(1, ...segments().map((seg) => Math.max(0, Number(seg.start || 0)) + Math.max(1, Number(seg.length || 1))));
        const storedDurationFrames = secondsToFrames(Math.max(0, Number(state.duration_seconds || 0)));
        return Math.max(secondsToFrames(VIEWPORT_SECONDS), storedDurationFrames, shotboardDurationFrames(), Math.ceil(end));
    };
    const view = () => {
        state.view = state.view && typeof state.view === "object" ? state.view : {};
        // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        // Allow zoom down to 0.05 so long clips (5+ min) can be shown fully at low zoom
        state.view.timeZoom = Math.max(0.05, Math.min(8, Number(state.view.timeZoom || 1)));
        state.view.trackHeight = Math.max(184, Math.min(240, Number(state.view.trackHeight || 188)));
        state.view.tool = String(state.view.tool || "cursor");
        if (!["cursor", "move", "trim", "cut"].includes(state.view.tool)) state.view.tool = "cursor";
        state.view.visibleSeconds = VIEWPORT_SECONDS;
        return state.view;
    };
    const visibleTimelineWidth = () => {
        // The first DOM draw happens before LiteGraph applies the fixed node width.
        // Keep the normal timeline filled, while still allowing Open Editor to use a wider viewport.
        const viewportWidth = Math.max(Number(root.clientWidth || 0), AUDIO_BOARD_FIXED_WIDTH - 24);
        return Math.max(820, Math.round(viewportWidth - LABEL_W - 48));
    };
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    // The viewport scale is stable. Long timelines grow horizontally instead of shrinking
    // under the pointer while a clip is moved.
    const effectiveViewSeconds = () => VIEWPORT_SECONDS;
    const pxPerFrame = () => dragPxPerFrame || Math.max(0.01, (visibleTimelineWidth() / Math.max(1, secondsToFrames(effectiveViewSeconds()))) * view().timeZoom);
    const meterEndFrames = () => secondsToFrames(Math.ceil(framesToSeconds(totalFrames())));
    const contentWidth = () => LABEL_W + Math.max(visibleTimelineWidth(), Math.ceil(meterEndFrames() * pxPerFrame())) + 8;
    const frameToX = (frame) => LABEL_W + Math.round(Math.max(0, Number(frame || 0)) * pxPerFrame());
    const xToFrame = (x) => Math.max(0, Math.round((Number(x || 0) - LABEL_W) / pxPerFrame()));
    const loopRange = () => {
        const start = Math.max(0, Math.min(totalFrames(), Math.round(Number(state.loopInFrame || 0))));
        const end = Math.max(0, Math.min(totalFrames(), Math.round(Number(state.loopOutFrame || 0))));
        return end > start ? { start, end } : null;
    };
    const setLoopIn = () => {
        state.loopInFrame = Math.max(0, Math.min(totalFrames() - 1, Math.round(Number(transport.playhead || 0))));
        if (!state.loopOutFrame || state.loopOutFrame <= state.loopInFrame) state.loopOutFrame = Math.min(totalFrames(), state.loopInFrame + secondsToFrames(2));
        addEdit(`Loop IN set at ${fmtTime(state.loopInFrame)}.`);
        writeState("loop_in", false);
        draw();
    };
    const setLoopOut = () => {
        state.loopOutFrame = Math.max(Math.round(Number(state.loopInFrame || 0)) + 1, Math.min(totalFrames(), Math.round(Number(transport.playhead || 0))));
        if (state.loopOutFrame <= state.loopInFrame) state.loopInFrame = Math.max(0, state.loopOutFrame - secondsToFrames(2));
        addEdit(`Loop OUT set at ${fmtTime(state.loopOutFrame)}.`);
        writeState("loop_out", false);
        draw();
    };
    const toggleLoop = () => {
        const range = loopRange();
        if (!range) {
            state.loopInFrame = Math.max(0, Math.min(totalFrames() - 1, Math.round(Number(transport.playhead || 0))));
            state.loopOutFrame = Math.min(totalFrames(), state.loopInFrame + secondsToFrames(4));
        }
        state.loopEnabled = !state.loopEnabled;
        addEdit(`Loop ${state.loopEnabled ? "enabled" : "disabled"} ${loopRange() ? `${fmtTime(state.loopInFrame)}-${fmtTime(state.loopOutFrame)}` : ""}.`);
        writeState("loop_toggle", false);
        draw();
    };
    const collectTransportDom = () => {
        const trackBars = new Map();
        root.querySelectorAll(".iamccs-track-meter").forEach((meter) => {
            const track = Number(meter.dataset.track || 0);
            const fills = Array.from(meter.querySelectorAll("i"));
            if (!fills.length) return;
            if (!trackBars.has(track)) trackBars.set(track, []);
            fills.forEach((fill) => trackBars.get(track).push(fill));
        });
        transport.dom = {
            playheads: Array.from(root.querySelectorAll(".iamccs-playhead")),
            time: root.querySelector(".iamccs-transport-time"),
            masterBars: Array.from(root.querySelectorAll(".iamccs-master-meter i")),
            masterReadouts: Array.from(root.querySelectorAll(".iamccs-master-readout")),
            trackBars,
        };
    };
    const addEdit = (text) => {
        state.status = state.status && typeof state.status === "object" ? state.status : {};
        state.status.edits = Array.isArray(state.status.edits) ? state.status.edits : [];
        state.status.edits.unshift(`${new Date().toLocaleTimeString()} ${text}`);
        state.status.edits = state.status.edits.slice(0, 14);
        transport.helper = text;
    };
    const noteTrackMixLocalState = (track, key, value, phase = "local") => {
        const trackIndex = Math.max(0, Number(track || 0));
        lastTrackMixLocalAt = Date.now();
        if (phase === "end" || Date.now() - trackMixDebugAt > 220) {
            trackMixDebugAt = Date.now();
            addEdit(`[debug] A${trackIndex + 1} ${String(key || "mix")} ${phase}: ${typeof value === "number" ? value.toFixed(3) : value}`);
        }
    };
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    const pushArrangerTrackToggle = (track, key, enabled) => node._iamccsAudioBoardTransport?.setTrackToggle?.(track, key, enabled) === true;
    const visualOptions = () => {
        const board = linkedShotboardNodes()[0];
        const widget = board ? findWidget(board, "timeline_data") : null;
        try {
            const data = JSON.parse(String(widget?.value || "{}"));
            return (Array.isArray(data.segments) ? data.segments : [])
                .filter((seg) => String(seg.type || "image") !== "audio" && !seg.placeholder)
                .map((seg, index) => ({ id: String(seg.id || ""), label: String(seg.label || seg.name || `shot_${index + 1}`), start: Number(seg.start || 0) }));
        } catch {
            return [];
        }
    };
    const multiTimelineIdForTake = (takeIndex) => `T${String(Math.max(1, Math.round(Number(takeIndex) || 1))).padStart(2, "0")}`;
    const normalizeMultiTimelineId = (value, fallbackTake = 1) => {
        const raw = String(value || "").trim();
        const take = Math.max(1, Math.round(Number(raw.replace(/\D/g, "")) || Number(fallbackTake) || 1));
        return multiTimelineIdForTake(take);
    };
    const shotboardAudioForActiveTake = (allAudioSegments, activeTake, activeTimelineId) => {
        const take = Math.max(1, Math.round(Number(activeTake) || 1));
        const timelineId = normalizeMultiTimelineId(activeTimelineId, take);
        const all = Array.isArray(allAudioSegments) ? allAudioSegments : [];
        const multiMatches = all.filter((seg) => {
            const segTake = Math.max(0, Math.round(Number(seg?.multiTakeIndex || 0)));
            const rawTimelineId = String(seg?.timelineId || "").trim();
            const segTimelineId = rawTimelineId ? normalizeMultiTimelineId(rawTimelineId, segTake || take) : "";
            const isMulti = Boolean(seg?.multiGenerationClip) || /^T\d+/i.test(rawTimelineId) || segTake > 0;
            if (!isMulti) return false;
            return segTimelineId === timelineId || segTake === take;
        });
        const source = multiMatches.length
            ? multiMatches
            : (take === 1 ? all.filter((seg) => Number(seg?.track || 0) === 0) : []);
        return source.map((seg) => {
            const localStart = Number(seg?.localStart);
            const next = JSON.parse(JSON.stringify(seg || {}));
            next.track = 0;
            next.start = Number.isFinite(localStart)
                ? Math.max(0, Math.round(localStart))
                : (Boolean(seg?.multiGenerationClip) || /^T\d+/i.test(String(seg?.timelineId || ""))
                    ? 0
                    : Math.max(0, Math.round(Number(seg?.start || 0))));
            next.timelineId = timelineId;
            next.multiTakeIndex = take;
            next.shotboardActiveTakeAudio = true;
            next.sourceTrackOriginal = Math.max(0, Math.round(Number(seg?.track || 0)));
            return next;
        });
    };
    const syncToShotboard = (reason = "sync", options = {}) => {
        const liveSync = isLiveSyncReason(reason);
        const renderTarget = options.render !== false && !liveSync;
        const callbackTarget = options.callback !== false && !liveSync;
        const boards = linkedShotboardNodes();
        const promptSource = dialoguePromptSyncSource();
        for (const board of boards) {
            const widget = findWidget(board, "timeline_data");
            if (!widget) continue;
            let data = {};
            try { data = JSON.parse(String(widget.value || "{}")); } catch { data = {}; }
            if (!data || typeof data !== "object") data = {};
            data.schema = data.schema || "iamccs.cine.filmmaker_timeline";
            data.schema_version = Math.max(2, Number(data.schema_version || 2));
            if (promptSource) {
                const globalPromptWidget = findWidget(board, "global_prompt");
                if (promptSource.globalPrompt) {
                    data.global_prompt = promptSource.globalPrompt;
                    data.prompt = promptSource.globalPrompt;
                    if (globalPromptWidget) {
                        globalPromptWidget.value = promptSource.globalPrompt;
                        if (callbackTarget) {
                            try { globalPromptWidget.callback?.(globalPromptWidget.value); } catch {}
                        }
                    }
                }
                if (promptSource.localPrompts.length) {
                    const visualSegments = Array.isArray(data.segments) ? data.segments : [];
                    const promptTargets = visualSegments.filter((seg) => String(seg?.type || "image").toLowerCase() !== "audio" && !seg?.placeholder);
                    promptTargets.forEach((seg, index) => {
                        const nextPrompt = String(promptSource.localPrompts[index] || "").trim();
                        if (!nextPrompt) return;
                        seg.prompt = nextPrompt;
                        seg.local_prompt = nextPrompt;
                        seg.use_prompt = true;
                    });
                    const syncedPrompts = promptTargets
                        .map((seg) => String(seg?.prompt || seg?.local_prompt || "").trim())
                        .filter(Boolean);
                    if (syncedPrompts.length) {
                        data.director_local_prompts = syncedPrompts.join(" | ");
                        data.local_prompts = syncedPrompts.join(" | ");
                        data.promptrelay_enabled = Boolean(promptSource.promptRelayEnabled);
                    }
                }
            }
            const allAudioSegments = JSON.parse(JSON.stringify(segments()));
            const shotboardOnlyFirst = state.audioBusMode === "only_first" || state.onlyFirstTrack;
            let nextShotboardAudioSegments = shotboardOnlyFirst
                ? allAudioSegments.filter((seg) => Number(seg.track || 0) === 0)
                : allAudioSegments;
            const publishCleanup = stripDialogueInjectPlaceholders(nextShotboardAudioSegments, { force: reason === "manual_publish" });
            if (publishCleanup.removed > 0) nextShotboardAudioSegments = publishCleanup.segments;
            data.audioTrackCount = shotboardOnlyFirst ? 1 : Math.max(1, Number(state.audioTrackCount || 4));
            data.masterAudioGain = Math.max(0, Math.min(2, Number(state.masterAudioGain ?? 1) || 1));
            data.masterAudioNormalize = Boolean(state.masterAudioNormalize);
            data.masterBus = JSON.parse(JSON.stringify(state.masterBus || {}));
            const allTrackSettings = JSON.parse(JSON.stringify(state.trackSettings || []));
            data.trackSettings = shotboardOnlyFirst ? allTrackSettings.slice(0, 1) : allTrackSettings;
            data.selectedMixer = shotboardOnlyFirst ? { type: "track", track: 0 } : JSON.parse(JSON.stringify(selectedMixer()));
            data.audioBusMode = shotboardOnlyFirst ? "shotboard_only_first" : "all_tracks";
            data.onlyFirstTrack = shotboardOnlyFirst;
            data.audioSyncMode = String(state.audioSyncMode || "timeline_audio");
            const arrangerMulti = state.multiGeneration && typeof state.multiGeneration === "object" ? state.multiGeneration : {};
            const boardMulti = data.multiGeneration && typeof data.multiGeneration === "object" ? data.multiGeneration : {};
            if (arrangerMulti.enabled) {
                const clonedMulti = JSON.parse(JSON.stringify(arrangerMulti));
                if (clonedMulti.sourceSegment && typeof clonedMulti.sourceSegment === "object") {
                    const source = clonedMulti.sourceSegment;
                    clonedMulti.sourceSegment = {
                        id: String(source.id || ""),
                        name: String(source.name || source.fileName || source.audioFile || "audio"),
                        track: Number(source.track || 0),
                        start: Number(source.start || 0),
                        length: Number(source.length || source.audioDurationFrames || 0),
                        trimStart: Number(source.trimStart || 0),
                        audioDurationFrames: Number(source.audioDurationFrames || source.length || 0),
                    };
                }
                const visualTimelines = boardMulti.visualTimelines && typeof boardMulti.visualTimelines === "object"
                    ? boardMulti.visualTimelines
                    : (clonedMulti.visualTimelines && typeof clonedMulti.visualTimelines === "object" ? clonedMulti.visualTimelines : {});
                const firstTimeline = Array.isArray(clonedMulti.timelineIds) && clonedMulti.timelineIds.length ? String(clonedMulti.timelineIds[0]) : "T01";
                data.multiGeneration = {
                    ...boardMulti,
                    ...clonedMulti,
                    audioSegmentsAll: allAudioSegments,
                    visualTimelines,
                    activeTake: Math.max(1, Number(boardMulti.activeTake || clonedMulti.activeTake || 1)),
                    activeTimelineId: String(boardMulti.activeTimelineId || clonedMulti.activeTimelineId || firstTimeline),
                    shotboardDurationPolicy: "chunk_duration_from_arranger",
                };
                if (shotboardOnlyFirst) {
                    nextShotboardAudioSegments = shotboardAudioForActiveTake(
                        allAudioSegments,
                        data.multiGeneration.activeTake,
                        data.multiGeneration.activeTimelineId
                    );
                    const activeSourceTrack = Math.max(0, Math.round(Number(nextShotboardAudioSegments[0]?.sourceTrackOriginal || 0)));
                    data.trackSettings = [allTrackSettings[activeSourceTrack] || allTrackSettings[0] || {}];
                }
            } else if (Object.keys(boardMulti).length) {
                data.multiGeneration = boardMulti;
            }
            data.audioSegments = nextShotboardAudioSegments;

            // Only publish tracks that have at least one clip — strip empty tracks.
            // Build the compacted index: active track indices in ascending order.
            const activeTrackIndices = [...new Set(data.audioSegments.map((seg) => Number(seg.track || 0)))].sort((a, b) => a - b);
            const activeTrackIndexSet = new Set(activeTrackIndices);
            // Remap segment .track values to a dense 0-based index so downstream
            // consumers see contiguous track numbers (0, 1, 2 …) with no gaps.
            const remapTrack = (oldTrack) => {
                const pos = activeTrackIndices.indexOf(Number(oldTrack || 0));
                return pos >= 0 ? pos : 0;
            };
            data.audioSegments = data.audioSegments.map((seg) => ({ ...seg, track: remapTrack(seg.track) }));
            // Keep only the trackSettings entries that correspond to active tracks.
            const compactedTrackSettings = activeTrackIndices.map((idx) => allTrackSettings[idx] || {});
            if (!shotboardOnlyFirst && activeTrackIndices.length > 0 && activeTrackIndices.length < Number(state.audioTrackCount || 4)) {
                data.audioTrackCount = activeTrackIndices.length;
                data.trackSettings = compactedTrackSettings;
            }

            data.use_custom_audio = data.audioSegments.some((seg) => hasMedia(seg) && !seg.mute);
            data.audio_data = JSON.stringify({
                audioSegments: data.audioSegments,
                audioTrackCount: data.audioTrackCount,
                use_custom_audio: data.use_custom_audio,
                masterAudioGain: data.masterAudioGain,
                masterAudioNormalize: data.masterAudioNormalize,
                masterBus: data.masterBus,
                trackSettings: data.trackSettings,
                audioBusMode: data.audioBusMode,
                onlyFirstTrack: data.onlyFirstTrack,
                audioSyncMode: data.audioSyncMode,
                duration_seconds: data.duration_seconds,
            });
            const syncSignature = JSON.stringify({
                audioSegments: data.audioSegments,
                audioTrackCount: data.audioTrackCount,
                masterAudioGain: data.masterAudioGain,
                masterAudioNormalize: data.masterAudioNormalize,
                masterBus: data.masterBus,
                trackSettings: data.trackSettings,
                selectedMixer: data.selectedMixer,
                audioBusMode: data.audioBusMode,
                onlyFirstTrack: data.onlyFirstTrack,
                audioSyncMode: data.audioSyncMode,
                multiGeneration: data.multiGeneration || {},
            });
            const cacheKey = `${board.id || "board"}:${renderTarget ? "render" : "silent"}`;
            if (shotboardSyncSignatures.get(cacheKey) === syncSignature && reason !== "manual_sync" && reason !== "manual_publish") continue;
            shotboardSyncSignatures.set(cacheKey, syncSignature);
            const boardDurationWidget = findWidget(board, "duration_seconds");
            const boardWidgetDuration = Math.max(0, Number(boardDurationWidget?.value || 0));
            const audioEndSeconds = Math.max(0, ...data.audioSegments.map((seg) => {
                return (Math.max(0, Number(seg.start || 0)) + Math.max(1, Number(seg.length || 1))) / fps();
            }));
            const visualEndSeconds = Math.max(0, ...(Array.isArray(data.segments) ? data.segments : []).map((seg) => {
                return (Math.max(0, Number(seg.start || seg.frame || 0)) + Math.max(1, Number(seg.length || seg.len || 1))) / fps();
            }));
            const multiDurationSeconds = data.multiGeneration?.enabled
                ? Math.max(0, Number(data.multiGeneration.chunkSeconds || 0))
                : 0;
            const visualTimelineDuration = data.multiGeneration?.visualTimelines && typeof data.multiGeneration.visualTimelines === "object"
                ? Math.max(0, ...Object.values(data.multiGeneration.visualTimelines).map((item) => Math.max(0, Number(item?.duration_seconds || 0))))
                : 0;
            const visualOverflow = multiDurationSeconds > 0
                && Math.max(visualEndSeconds, visualTimelineDuration) > multiDurationSeconds + 0.05;
            const mediaDuration = Math.max(VIEWPORT_SECONDS, audioEndSeconds, visualEndSeconds, multiDurationSeconds);
            const storedBoardDuration = Math.max(0, Number(data.duration_seconds || 0));
            if (multiDurationSeconds > 0 && !visualOverflow) {
                data.duration_seconds = multiDurationSeconds;
                if (boardDurationWidget) {
                    boardDurationWidget.value = Number(multiDurationSeconds.toFixed(3));
                    if (callbackTarget) {
                        try { boardDurationWidget.callback?.(boardDurationWidget.value); } catch {}
                    }
                }
                if (data.multiGeneration) data.multiGeneration.durationWarning = "";
                if (state.multiGeneration && typeof state.multiGeneration === "object") state.multiGeneration.lastDurationWarning = "";
            } else {
                data.duration_seconds = boardWidgetDuration > 0
                    ? boardWidgetDuration
                    : (storedBoardDuration > 0 && storedBoardDuration <= mediaDuration * 1.35 ? storedBoardDuration : mediaDuration);
                if (visualOverflow && data.multiGeneration) {
                    const warning = `MULTI duration warning: arranger split is ${multiDurationSeconds.toFixed(2)}s, but one Shotboard visual timeline is longer. Resize/split that timeline before queueing indexed takes.`;
                    data.multiGeneration.durationWarning = warning;
                    if (state.multiGeneration && typeof state.multiGeneration === "object" && state.multiGeneration.lastDurationWarning !== warning) {
                        state.multiGeneration.lastDurationWarning = warning;
                        addEdit(warning);
                    }
                }
            }
            data.frame_rate = fps();
            data.truth_revision = Math.max(Number(data.truth_revision || 0), Number(board.properties?.iamccs_v3_timeline_revision || 0), 0) + 1;
            data.truth_updated_at = new Date().toISOString();
            const nextValue = JSON.stringify(data, null, 2);
            widget.value = nextValue;
            if (callbackTarget) widget.callback?.(widget.value);
            board.properties = board.properties || {};
            board.properties.iamccs_v3_timeline_revision = data.truth_revision;
            board.properties.iamccs_v3_timeline_data_backup = widget.value;
            board.properties.iamccs_v3_timeline_updated_at = data.truth_updated_at;
            if (renderTarget) {
                board._iamccsCineShotboardV3Ready = false;
                board._iamccsCineShotboardV3Version = "";
                document.dispatchEvent(new CustomEvent("iamccs:planner_rows_updated", { detail: { node_id: board.id, source: "IAMCCS_AudioBoardArranger", reason, render: true } }));
            }
        }
    };
    const scheduleShotboardSync = (reason = "live") => {
        if (!state.shotboardAutoSyncEnabled) return;
        if (shotboardSyncTimer) window.clearTimeout(shotboardSyncTimer);
        shotboardSyncTimer = window.setTimeout(() => {
            shotboardSyncTimer = 0;
            syncToShotboard(reason, { render: false, callback: false });
        }, 650);
    };
    const writeState = (reason = "edit", sync = true, options = {}) => {
        if (!options.quiet && liveStateWriteTimer) {
            window.clearTimeout(liveStateWriteTimer);
            liveStateWriteTimer = 0;
        }
        state.schema = "iamccs.audio_board_arranger";
        state.schema_version = 1;
        state.frame_rate = fps();
        state.duration_seconds = Math.max(VIEWPORT_SECONDS, framesToSeconds(totalFrames()));
        state.masterBus = state.masterBus && typeof state.masterBus === "object" ? state.masterBus : {};
        state.masterBus.effectChain = Array.isArray(state.masterBus.effectChain)
            ? state.masterBus.effectChain
            : JSON.parse(JSON.stringify(DEFAULT_MASTER_CHAIN));
        state.trackSettings = Array.isArray(state.trackSettings) ? state.trackSettings : [];
        state.audioBusMode = (state.audioBusMode === "only_first" || state.onlyFirstTrack) ? "only_first" : "all_tracks";
        state.onlyFirstTrack = state.audioBusMode === "only_first";
        state.loopEnabled = Boolean(state.loopEnabled);
        state.loopInFrame = Math.max(0, Math.min(totalFrames(), Math.round(Number(state.loopInFrame || 0))));
        state.loopOutFrame = Math.max(0, Math.min(totalFrames(), Math.round(Number(state.loopOutFrame || 0))));
        if (state.loopOutFrame && state.loopOutFrame <= state.loopInFrame) state.loopOutFrame = Math.min(totalFrames(), state.loopInFrame + Math.max(1, secondsToFrames(1)));
        state.selectedClipId = selectedId || "";
        selectedMixer();
        view();
        state.audioSegments = segments().sort((a, b) => (Number(a.track || 0) - Number(b.track || 0)) || (Number(a.start || 0) - Number(b.start || 0)));
        const persistedState = JSON.parse(JSON.stringify(state));
        for (const segment of (Array.isArray(persistedState.audioSegments) ? persistedState.audioSegments : [])) {
            delete segment.waveformPeaks;
            delete segment.waveformCache;
            delete segment.decodedWaveform;
        }
        if (dataWidget) {
            dataWidget.value = JSON.stringify(persistedState, null, 2);
            if (!options.quiet) dataWidget.callback?.(dataWidget.value);
        }
        // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        // Persist to localStorage as backup (survives refresh and workspace tab switch)
        if (!options.quiet) saveToLocalStorage(dataWidget?.value || "");
        // Trigger ComfyUI graph change so its autosave captures updated widget values
        if (!options.quiet) {
            if (graphChangeTimer) window.clearTimeout(graphChangeTimer);
            graphChangeTimer = window.setTimeout(() => {
                graphChangeTimer = 0;
                try { app.graph?.change?.(); } catch {}
            }, 400);
        }
        if (sync && state.shotboardAutoSyncEnabled) syncToShotboard(reason);
        document.dispatchEvent(new CustomEvent("iamccs:audio_arranger_state_changed", {
            detail: {
                node_id: node.id,
                reason,
                selectedMixer: JSON.parse(JSON.stringify(state.selectedMixer || { type: "track", track: 0 })),
            },
        }));
        if (!options.quiet) {
            markCanvasDirty(false);
        }
    };
    const publishToShotboard = () => {
        const boards = linkedShotboardNodes();
        if (!boards.length) {
            addEdit("Publish skipped: no connected Shotboard V3.");
            draw();
            return;
        }
        // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        const cleanup = stripDialogueInjectPlaceholders(segments(), { force: false });
        if (cleanup.removed > 0) {
            state.audioSegments = cleanup.segments;
            writeState("publish_placeholder_cleanup", false);
            addEdit(`Removed ${cleanup.removed} Inject UI placeholder clip${cleanup.removed === 1 ? "" : "s"} before Publish.`);
        }
        syncToShotboard("manual_publish");
        addEdit(`Published ${segments().length} audio clip${segments().length === 1 ? "" : "s"} to connected Shotboard V3.`);
        draw();
    };
    const pullFromShotboard = () => {
        const board = linkedShotboardNodes()[0];
        const widget = board ? findWidget(board, "timeline_data") : null;
        if (!widget) return;
        try {
            const data = JSON.parse(String(widget.value || "{}"));
            state.audioSegments = Array.isArray(data.audioSegments) ? data.audioSegments : [];
            state.audioTrackCount = Math.max(1, Number(data.audioTrackCount || 4));
            state.masterAudioGain = Math.max(0, Math.min(2, Number(data.masterAudioGain ?? 1) || 1));
            state.masterAudioNormalize = Boolean(data.masterAudioNormalize);
            state.masterBus = data.masterBus && typeof data.masterBus === "object" ? data.masterBus : state.masterBus;
            state.trackSettings = Array.isArray(data.trackSettings) ? data.trackSettings : state.trackSettings;
            state.selectedMixer = data.selectedMixer && typeof data.selectedMixer === "object" ? data.selectedMixer : state.selectedMixer;
            state.audioSyncMode = String(data.audioSyncMode || "timeline_audio");
            addEdit("Pulled audio lanes from connected Shotboard V3.");
            writeState("pull", false);
            draw();
        } catch {}
    };
    const ensureAudioContext = async (resume = true) => {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextClass) throw new Error("WebAudio unavailable");
        if (!audioContext) audioContext = new AudioContextClass();
        if (resume && audioContext.state === "suspended") await audioContext.resume();
        return audioContext;
    };
    let saveAudioBoardInFlight = false;
    const defaultAudioBoardPackageName = () => `audioboard_${new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19)}`;
    const cleanAudioBoardPackageName = (value) => String(value || "").trim().replace(/[\\/:*?"<>|]+/g, "_").replace(/\s+/g, "_").slice(0, 120);
    const saveAudioBoardPackage = async (options = {}) => {
        if (saveAudioBoardInFlight) {
            transport.helper = "Save AudioBoard already running...";
            draw();
            return;
        }
        saveAudioBoardInFlight = true;
        const requestedLabel = cleanAudioBoardPackageName(options.packageName || state.audioBoardPackageName || "");
        const label = requestedLabel || defaultAudioBoardPackageName();
        state.audioBoardPackageName = label;
        try {
            console.info("[IAMCCS AudioBoardArranger] Save AudioBoard clicked", { label, nodeId: node.id || null });
            addEdit(`Saving AudioBoard package: ${label}`);
            transport.helper = `Saving AudioBoard package: ${label}`;
            draw();
            writeState("save_audioboard_prepare", false, { quiet: true });
            const response = await api.fetchApi("/api/iamccs/audio/save_audioboard_package", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    package_name: label,
                    board: JSON.parse(JSON.stringify(state)),
                    workflow_node_id: node.id || null,
                }),
            });
            let result = {};
            try { result = await response.json(); }
            catch { result = { error: `save failed: ${response.status}` }; }
            if (!response.ok || !result?.ok) throw new Error(result?.error || `save failed: ${response.status}`);
            state.audioBoardPackageName = result.package_name || label;
            addEdit(`Saved AudioBoard package: ${result.package_name}`);
            transport.helper = `Saved AudioBoard package: ${result.package_dir}`;
            console.info("[IAMCCS AudioBoardArranger] Save AudioBoard complete", result);
            writeState("save_audioboard_done", false, { quiet: true });
        } catch (err) {
            console.warn("[IAMCCS AudioBoardArranger] save package failed", err);
            addEdit(`Save AudioBoard failed: ${err?.message || err}`);
            transport.helper = `Save AudioBoard failed: ${err?.message || err}`;
        } finally {
            saveAudioBoardInFlight = false;
            draw();
        }
    };
    const saveAudioBoardPackageAs = () => {
        const proposed = state.audioBoardPackageName || defaultAudioBoardPackageName();
        const chosen = window.prompt("Save AudioBoard package as", proposed);
        if (chosen == null) {
            transport.helper = "Save As cancelled.";
            draw();
            return;
        }
        const label = cleanAudioBoardPackageName(chosen);
        if (!label) {
            transport.helper = "Save As cancelled: empty package name.";
            draw();
            return;
        }
        saveAudioBoardPackage({ packageName: label });
    };
    const uploadAudioFile = async (file) => {
        const body = new FormData();
        body.append("image", file);
        const resp = await api.fetchApi("/upload/image", { method: "POST", body });
        if (!resp || resp.status !== 200) throw new Error(`upload failed: ${resp?.status || "no response"}`);
        const data = await resp.json();
        const filename = data?.name || file.name;
        const subfolder = data?.subfolder || "";
        return { path: subfolder ? `${subfolder}/${filename}` : filename, type: data?.type || "input" };
    };
    const audioViewUrl = (seg) => {
        const file = String(seg.audioFile || "").trim();
        if (!file) return "";
        const parts = file.split(/[\\/]+/).filter(Boolean);
        const filename = parts.pop() || file;
        const subfolder = parts.join("/");
        return `/view?filename=${encodeURIComponent(filename)}&type=${encodeURIComponent(seg.audioUploadType || "input")}&subfolder=${encodeURIComponent(subfolder)}`;
    };
    const getBuffer = async (seg, resume = true) => {
        if (!seg) return null;
        if (audioBuffers.has(seg.id)) return audioBuffers.get(seg.id);
        const ctx = await ensureAudioContext(resume);
        const url = audioUrls.get(seg.id) || audioViewUrl(seg);
        if (!url) return null;
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`audio fetch failed ${resp.status}`);
        const arrayBuffer = await resp.arrayBuffer();
        const decoded = await ctx.decodeAudioData(arrayBuffer.slice(0));
        audioBuffers.set(seg.id, decoded);
        return decoded;
    };
    const peaksFromBuffer = (decoded, count = 1200) => {
        const channelCount = Math.max(1, decoded?.numberOfChannels || 1);
        const frameCount = Math.max(1, decoded?.length || 1);
        const buckets = Math.max(180, Math.min(2400, Number(count || 1200)));
        const peaks = [];
        for (let i = 0; i < buckets; i += 1) {
            const start = Math.floor((i / buckets) * frameCount);
            const end = Math.max(start + 1, Math.floor(((i + 1) / buckets) * frameCount));
            let min = 0;
            let max = 0;
            let sum = 0;
            let n = 0;
            for (let channelIndex = 0; channelIndex < channelCount; channelIndex += 1) {
                const channel = decoded.getChannelData(channelIndex);
                for (let j = start; j < end; j += 1) {
                    const v = Number(channel[j] || 0);
                    min = Math.min(min, v);
                    max = Math.max(max, v);
                    sum += v * v;
                    n += 1;
                }
            }
            peaks.push({
                min: Number(min.toFixed(5)),
                max: Number(max.toFixed(5)),
                rms: Number(Math.sqrt(sum / Math.max(1, n)).toFixed(5)),
            });
        }
        return peaks;
    };
    const ensureSegmentWaveform = (seg) => {
        const hasDecodedPeaks = Array.isArray(seg?.waveformPeaks) && seg.waveformPeaks.length > 8;
        if (!seg || !hasMedia(seg) || (seg.waveformReal === true && hasDecodedPeaks) || waveformLoading.has(seg.id)) return;
        // Regenerable waveform caches may be compacted from workflow drafts. A stale
        // waveformReal flag must never prevent decoding the still-valid audio file.
        seg.waveformReal = false;
        waveformLoading.add(seg.id);
        getBuffer(seg, false)
            .then((buffer) => {
                if (!buffer) return;
                seg.audioDurationFrames = Math.max(1, Math.round(buffer.duration * fps()));
                seg.waveformPeaks = peaksFromBuffer(buffer, Math.max(900, Math.min(2200, Math.round(buffer.duration * 70))));
                seg.waveformReal = true;
                waveformLoading.delete(seg.id);
                writeState("waveform_decode", false, { quiet: true });
                draw();
            })
            .catch((err) => {
                waveformLoading.delete(seg.id);
                console.warn("[IAMCCS AudioBoardArranger] waveform decode failed", seg?.audioFile || seg?.fileName || seg?.id, err);
            });
    };
    const decodeAudioInfo = async (file) => {
        const ctx = await ensureAudioContext();
        const arrayBuffer = await file.arrayBuffer();
        const decoded = await ctx.decodeAudioData(arrayBuffer.slice(0));
        const peaks = peaksFromBuffer(decoded, Math.max(900, Math.min(2200, Math.round(decoded.duration * 70))));
        return {
            durationFrames: Math.max(1, Math.round(decoded.duration * fps())),
            peaks,
            buffer: decoded,
            channelCount: Math.max(1, Number(decoded.numberOfChannels || 1)),
        };
    };
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    const isProbablyAudioFile = (file) => {
        if (!file) return false;
        const mime = String(file.type || "").toLowerCase();
        if (mime.startsWith("audio/")) return true;
        if (mime === "application/ogg") return true;
        const name = String(file.name || "").toLowerCase();
        return /\.(wav|wave|mp3|m4a|flac|ogg|aac|aif|aiff|wma|opus)$/i.test(name);
    };
    const importFiles = async (files) => {
        const offeredFiles = Array.from(files || []);
        const list = offeredFiles.filter((file) => isProbablyAudioFile(file));
        if (!list.length) {
            const skipped = offeredFiles.map((file) => String(file?.name || "file")).filter(Boolean);
            const detail = skipped.length ? ` No supported audio detected in: ${skipped.join(", ")}.` : "";
            addEdit(`Import skipped.${detail}`);
            transport.helper = `Import skipped.${detail}`;
            draw();
            return;
        }
        const firstVisual = visualOptions()[0] || null;
        let cursor = firstVisual ? Number(firstVisual.start || 0) : Math.max(0, ...segments().map((seg) => Number(seg.start || 0) + Number(seg.length || 1)));
        for (const file of list) {
            try {
                const info = await decodeAudioInfo(file);
                const uploaded = await uploadAudioFile(file);
                const track = Math.max(0, Math.min(Math.max(1, Number(state.audioTrackCount || 4)) - 1, segments().length % Math.max(1, Number(state.audioTrackCount || 4))));
                const seg = {
                    id: newId("aud"),
                    type: "audio",
                    start: cursor,
                    length: info.durationFrames,
                    track,
                    trimStart: 0,
                    audioDurationFrames: info.durationFrames,
                    audioFile: uploaded.path,
                    audioUploadType: uploaded.type,
                    fileName: file.name,
                    name: file.name,
                    mime: file.type,
                    size: file.size,
                    waveformPeaks: info.peaks,
                    waveformReal: true,
                    channelCount: Math.max(1, Number(info.channelCount || 1)),
                    channelMode: Number(info.channelCount || 1) > 1 ? "stereo" : "mono",
                    purpose: "dialogue_or_music",
                    gain: 1,
                    pan: 0,
                    fadeInFrames: 0,
                    fadeOutFrames: 0,
                    pitchSemitones: 0,
                    timeStretch: 1,
                    hpfHz: 0,
                    lpfHz: 22000,
                    eqLowDb: 0,
                    eqMidDb: 0,
                    eqHighDb: 0,
                    compressor: 0,
                    noiseGateDb: -60,
                    ducking: 0,
                    reverbSend: 0,
                    delaySend: 0,
                    stereoWidth: 1,
                    transient: 0,
                    denoise: 0,
                    reverse: false,
                    lock: false,
                    normalizeAudio: false,
                    mute: false,
                    solo: false,
                    linkedVisualId: firstVisual ? firstVisual.id : "",
                };
                audioBuffers.set(seg.id, info.buffer);
                audioUrls.set(seg.id, URL.createObjectURL(file));
                segments().push(seg);
                selectedId = seg.id;
                if (!firstVisual) cursor += info.durationFrames;
                addEdit(`Imported ${file.name} (${framesToSeconds(info.durationFrames).toFixed(2)}s).`);
            } catch (err) {
                addEdit(`Import failed: ${file.name} (${err?.message || err})`);
                console.warn("[IAMCCS AudioBoardArranger] import failed", err);
            }
        }
        writeState("import");
        draw();
    };
    const extractAudioBoardState = (parsed) => {
        const parseMaybe = (value) => {
            if (value && typeof value === "object") return value;
            if (typeof value !== "string") return null;
            try { return JSON.parse(value); } catch { return null; }
        };
        const queue = [{ value: parsed, depth: 0 }];
        const seen = new Set();
        while (queue.length) {
            const { value, depth } = queue.shift();
            const candidate = parseMaybe(value);
            if (!candidate || typeof candidate !== "object" || seen.has(candidate)) continue;
            seen.add(candidate);
            if (Array.isArray(candidate.audioSegments)) return candidate;
            if (depth >= 6) continue;
            if (Array.isArray(candidate)) {
                candidate.slice(0, 400).forEach((item) => queue.push({ value: item, depth: depth + 1 }));
            } else {
                Object.values(candidate).slice(0, 400).forEach((item) => queue.push({ value: item, depth: depth + 1 }));
            }
        }
        return null;
    };
    const importAudioBoardPackageFile = async (file) => {
        if (!file) return;
        try {
            const raw = await file.text();
            const parsed = JSON.parse(raw);
            const board = extractAudioBoardState(parsed);
            if (!board) {
                throw new Error("Invalid AudioBoard package: no arranger audioSegments found");
            }
            stopPlayback(false);
            const next = {
                ...parseState(),
                ...board,
                schema: "iamccs.audio_board_arranger",
                schema_version: Math.max(1, Number(board.schema_version || 1) || 1),
                audioBoardPackageName: board.audioBoardPackageName || parsed.package_name || file.name.replace(/\.json$/i, ""),
            };
            next.audioSegments = Array.isArray(next.audioSegments) ? next.audioSegments : [];
            next.audioTrackCount = Math.max(1, Number(next.audioTrackCount || 1) || 1);
            next.duration_seconds = Math.max(DEFAULT_SECONDS, Number(next.duration_seconds || DEFAULT_SECONDS) || DEFAULT_SECONDS);
            next.view = { ...parseState().view, ...(board.view || {}) };
            next.status = { edits: Array.isArray(board.status?.edits) ? board.status.edits.slice(-80) : [] };
            state = next;
            selectedId = segments()[0]?.id || "";
            addEdit(`Imported AudioBoard package: ${file.name}`);
            transport.helper = `Imported AudioBoard package: ${file.name}`;
            writeState("import_audioboard");
            draw();
        } catch (err) {
            console.warn("[IAMCCS AudioBoardArranger] import AudioBoard failed", err);
            addEdit(`Import AudioBoard failed: ${err?.message || err}`);
            transport.helper = `Import AudioBoard failed: ${err?.message || err}`;
            draw();
        }
    };
    fileInput.onchange = async (event) => {
        await importFiles(event.target.files || []);
        fileInput.value = "";
    };
    audioBoardInput.onchange = async (event) => {
        await importAudioBoardPackageFile((event.target.files || [])[0]);
        audioBoardInput.value = "";
    };
    const setPlayhead = (frame, redraw = false) => {
        transport.playhead = Math.max(0, Math.min(totalFrames(), Math.round(Number(frame || 0))));
        if (redraw) draw();
        else {
            const dom = transport.dom || {};
            (dom.playheads || root.querySelectorAll(".iamccs-playhead")).forEach((el) => { el.style.left = `${frameToX(transport.playhead)}px`; });
            const readout = dom.time || root.querySelector(".iamccs-transport-time");
            if (readout) readout.textContent = `${fmtTime(transport.playhead)} / ${fmtTime(totalFrames())}`;
        }
    };
    const stopPlayback = (redraw = true) => {
        for (const item of transport.sources) {
            try { item.source.stop(); } catch {}
        }
        transport.sources = [];
        transport.playing = false;
        transport.analysers = new Map();
        transport.trackMixNodes = new Map();
        transport.masterAnalyser = null;
        transport.masterGainNode = null;
        transport.lastMeterSnapshot = { tracks: {}, master: { peak: 0, rms: 0 }, playing: false, playhead: transport.playhead };
        transport.efxFrame = 0;
        transport.lastMeterAt = 0;
        transport.pendingLoopRestart = false;
        if (transport.raf) cancelAnimationFrame(transport.raf);
        transport.raf = 0;
        const dom = transport.dom || {};
        (dom.masterBars || root.querySelectorAll(".iamccs-master-meter i")).forEach((el) => { el.style.width = "0%"; });
        if (dom.trackBars) dom.trackBars.forEach((items) => items.forEach((el) => { el.style.width = "0%"; }));
        else root.querySelectorAll(".iamccs-track-meter i").forEach((el) => { el.style.width = "0%"; });
        (dom.masterReadouts || root.querySelectorAll(".iamccs-master-readout")).forEach((el) => { el.textContent = "PK 000 RMS 000"; });
        if (redraw) draw();
    };
    const meterFromAnalyser = (analyser) => {
        if (!analyser) return { peak: 0, rms: 0 };
        if (!analyser._iamccsMeterData || analyser._iamccsMeterData.length !== analyser.fftSize) analyser._iamccsMeterData = new Uint8Array(analyser.fftSize);
        const data = analyser._iamccsMeterData;
        analyser.getByteTimeDomainData(data);
        let peak = 0;
        let sum = 0;
        for (const value of data) {
            const v = (value - 128) / 128;
            const a = Math.abs(v);
            peak = Math.max(peak, a);
            sum += v * v;
        }
        return { peak: Math.min(1, peak), rms: Math.min(1, Math.sqrt(sum / Math.max(1, data.length))) };
    };
    const activeAnalyser = () => {
        const target = selectedMixer();
        if (target.type === "track") return transport.analysers.get(target.track) || transport.masterAnalyser;
        return transport.masterAnalyser;
    };
    const paintRealtimeAnalyser = (canvas, kind, analyser) => {
        if (!canvas || !analyser) return false;
        const rect = canvas.getBoundingClientRect();
        const w = Math.max(180, Math.round(rect.width || 260));
        const h = Math.max(70, Math.round(rect.height || 90));
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d");
        if (!ctx) return false;
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = "#050d0f";
        ctx.fillRect(0, 0, w, h);
        ctx.strokeStyle = "rgba(255,255,255,.08)";
        ctx.lineWidth = 1;
        for (let x = 0; x <= w; x += Math.max(24, w / 8)) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, h);
            ctx.stroke();
        }
        if (kind === "eq") {
            const fx = canvas._iamccsFx?.type === "eq" ? canvas._iamccsFx : null;
            const freq = new Uint8Array(analyser.frequencyBinCount);
            analyser.getByteFrequencyData(freq);
            ctx.fillStyle = "rgba(255,255,255,.03)";
            ctx.fillRect(0, 0, w / 3, h);
            ctx.fillStyle = "rgba(255,255,255,.02)";
            ctx.fillRect(w / 3, 0, w / 3, h);
            ctx.fillStyle = "rgba(255,255,255,.03)";
            ctx.fillRect((w / 3) * 2, 0, w / 3, h);
            const bars = 88;
            for (let i = 0; i < bars; i += 1) {
                const t = i / Math.max(1, bars - 1);
                const idx = Math.min(freq.length - 1, Math.round(Math.pow(t, 1.85) * (freq.length - 1)));
                const v = Math.max(0, Math.min(1, Number(freq[idx] || 0) / 255));
                const x = (i / bars) * w;
                const bw = Math.max(2, w / bars - 1);
                const bh = Math.max(1, v * h * .45);
                ctx.fillStyle = v > .72 ? "rgba(240,184,87,.34)" : "rgba(85,199,185,.24)";
                ctx.fillRect(x, h - bh, bw, bh);
            }
            normalizeEffect(fx);
            const low = Number(fx?.params?.low ?? 0);
            const midDb = Number(fx?.params?.mid ?? 0);
            const high = Number(fx?.params?.high ?? 0);
            const hpf = fx?.params?.lowCut ? Number(fx.params.lowCutFreq || 80) : 0;
            const lpf = fx?.params?.highCut ? Number(fx.params.highCutFreq || 12000) : 22000;
            const lowCutLevel = Math.max(-48, Math.min(0, Number(fx?.params?.lowCutLevel ?? -30)));
            const highCutLevel = Math.max(-48, Math.min(0, Number(fx?.params?.highCutLevel ?? -30)));
            const center = h * .5;
            const freqToX = (hz) => eqVisualX(hz, w);
            const yFromDb = (db) => center - (Math.max(-24, Math.min(24, Number(db || 0))) / 48) * h;
            const yCutFromDb = (db) => center - (Math.max(-48, Math.min(0, Number(db || 0))) / 72) * h;
            const drawSmooth = (points, stroke, width = 2.2) => {
                ctx.strokeStyle = stroke;
                ctx.lineWidth = width;
                ctx.beginPath();
                points.forEach((p, i) => {
                    if (i === 0) ctx.moveTo(p[0], p[1]);
                    else {
                        const p0 = points[Math.max(0, i - 2)];
                        const p1 = points[i - 1];
                        const p2 = p;
                        const p3 = points[Math.min(points.length - 1, i + 1)];
                        ctx.bezierCurveTo(
                            p1[0] + (p2[0] - p0[0]) / 6,
                            p1[1] + (p2[1] - p0[1]) / 6,
                            p2[0] - (p3[0] - p1[0]) / 6,
                            p2[1] - (p3[1] - p1[1]) / 6,
                            p2[0],
                            p2[1]
                        );
                    }
                });
                ctx.stroke();
            };
            const eqFreq = eqBandFrequencies(fx);
            [-24, -12, 0, 12, 24].forEach((db) => {
                const y = yFromDb(db);
                ctx.strokeStyle = db === 0 ? "rgba(255,255,255,.14)" : "rgba(255,255,255,.06)";
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(w, y);
                ctx.stroke();
            });
            [60, 250, 1000, 4000, 12000].forEach((hz) => {
                const x = freqToX(hz);
                ctx.strokeStyle = (hz === 250 || hz === 4000) ? "rgba(255,255,255,.14)" : "rgba(255,255,255,.08)";
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, h);
                ctx.stroke();
            });
            ctx.fillStyle = "rgba(255,226,168,.72)";
            ctx.font = "900 9px ui-monospace, Consolas, monospace";
            ctx.fillText("LOW", 8, 12);
            ctx.fillText("MID", Math.max(8, w * .5 - 10), 12);
            ctx.fillText("HIGH", Math.max(8, w - 34), 12);
            const lowX = freqToX(eqFreq.low);
            const midX = freqToX(eqFreq.mid);
            const highX = freqToX(eqFreq.high);
            const lowCutX = freqToX(hpf || 20);
            const highCutX = freqToX(lpf || 22000);
            if (hpf > 20) {
                ctx.fillStyle = "rgba(216,91,66,.12)";
                ctx.fillRect(0, 0, lowCutX, h);
            }
            if (lpf < 22000) {
                ctx.fillStyle = "rgba(216,91,66,.10)";
                ctx.fillRect(highCutX, 0, w - highCutX, h);
            }
            drawSmooth([
                [0, hpf > 20 ? yCutFromDb(lowCutLevel) : center],
                ...(hpf > 20 ? [[lowCutX, center]] : []),
                [lowX, yFromDb(low)],
                [midX, yFromDb(midDb)],
                [highX, yFromDb(high)],
                ...(lpf < 22000 ? [[highCutX, center]] : []),
                [w, lpf < 22000 ? yCutFromDb(highCutLevel) : center],
            ].sort((a, b) => a[0] - b[0]), "#f5d08e", 2.4);
            ctx.fillStyle = "#ffc67f";
            ctx.font = "900 9px ui-monospace, Consolas, monospace";
            if (hpf > 20) ctx.fillText(`LC ${Math.round(hpf)}Hz ${Math.round(lowCutLevel)}dB`, 8, h - 9);
            if (lpf < 22000) ctx.fillText(`HC ${(lpf / 1000).toFixed(1)}k ${Math.round(highCutLevel)}dB`, Math.max(8, w - 116), h - 9);
            return true;
        }
        if (kind === "wave") {
            const data = new Uint8Array(analyser.fftSize);
            analyser.getByteTimeDomainData(data);
            ctx.strokeStyle = "#f5d08e";
            ctx.lineWidth = 2;
            ctx.beginPath();
            for (let i = 0; i < data.length; i += 1) {
                const x = (i / Math.max(1, data.length - 1)) * w;
                const y = ((data[i] || 128) / 255) * h;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
            return true;
        }
        if (kind === "bus") {
            const fx = canvas._iamccsFx || {};
            const meter = meterFromAnalyser(analyser);
            const wave = new Uint8Array(analyser.fftSize);
            analyser.getByteTimeDomainData(wave);
            ctx.strokeStyle = "rgba(255,226,168,.34)";
            ctx.lineWidth = 1.4;
            ctx.beginPath();
            wave.forEach((value, i) => {
                const x = (i / Math.max(1, wave.length - 1)) * w;
                const y = h * .26 + ((value - 128) / 128) * h * .16;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
            const peakX = Math.round(meter.peak * w);
            const rmsX = Math.round(meter.rms * w);
            const grad = ctx.createLinearGradient(0, 0, w, 0);
            grad.addColorStop(0, "#55c7b9");
            grad.addColorStop(.65, "#d6df76");
            grad.addColorStop(.84, "#e8aa4e");
            grad.addColorStop(1, "#d75b42");
            ctx.fillStyle = grad;
            ctx.fillRect(0, h * .34, peakX, h * .25);
            ctx.fillStyle = "rgba(255,255,255,.42)";
            ctx.fillRect(0, h * .68, rmsX, h * .12);
            const threshold = Math.max(-80, Math.min(0, Number(fx.params?.threshold ?? fx.params?.ceiling ?? -18)));
            const ratio = Math.max(1, Math.min(30, Number(fx.params?.ratio ?? (fx.type === "limiter" ? 20 : 4))));
            const knee = fx.type === "limiter" ? 4 : 12;
            ctx.strokeStyle = "rgba(143,208,204,.86)";
            ctx.lineWidth = 1.8;
            ctx.beginPath();
            for (let x = 0; x <= w; x += 1) {
                const inDb = -60 + (x / Math.max(1, w)) * 66;
                const over = inDb - threshold;
                const kneeMix = Math.max(0, Math.min(1, (over + knee / 2) / Math.max(.001, knee)));
                const compressed = over > 0 ? threshold + over / ratio : inDb;
                const soft = inDb * (1 - kneeMix) + compressed * kneeMix;
                const outDb = over < -knee / 2 ? inDb : soft;
                const y = h - ((outDb + 60) / 66) * h;
                if (x === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
            const inputPeakDb = meter.peak > .0001 ? 20 * Math.log10(meter.peak) : -80;
            const reduction = Math.max(0, inputPeakDb - threshold) * (1 - 1 / ratio);
            ctx.fillStyle = "rgba(216,91,66,.82)";
            ctx.fillRect(w - 18, h - Math.min(h, reduction / 24 * h), 10, Math.min(h, reduction / 24 * h));
            ctx.fillStyle = "#f5d08e";
            ctx.font = "900 10px ui-monospace, Consolas, monospace";
            ctx.fillText(`PK ${Math.round(meter.peak * 100)} RMS ${Math.round(meter.rms * 100)} GR ${reduction.toFixed(1)}dB`, 8, 15);
            return true;
        }
        return false;
    };
    const updateInlineEfxRealtime = () => {
        if (!transport.playing) return;
        transport.efxFrame = (transport.efxFrame || 0) + 1;
        if (transport.efxFrame % 3 !== 0) return;
        const analyser = activeAnalyser();
        if (!analyser) return;
        root.querySelectorAll(".iamccs-inline-efx-canvas[data-kind='bus'], .iamccs-inline-efx-canvas[data-kind='eq']").forEach((canvas) => {
            const rect = canvas.getBoundingClientRect();
            if (rect.width <= 0 || rect.height <= 0 || rect.right < 0 || rect.left > window.innerWidth) return;
            const kind = canvas.dataset.kind || "";
            paintRealtimeAnalyser(canvas, kind, analyser);
        });
    };
    const updateMeters = () => {
        if (!transport.playing) return;
        const elapsed = Math.max(0, (audioContext?.currentTime || 0) - transport.startedAt);
        const nextFrame = transport.startFrame + secondsToFrames(elapsed);
        const range = loopRange();
        if (state.loopEnabled && range && nextFrame >= range.end && !transport.pendingLoopRestart) {
            transport.pendingLoopRestart = true;
            stopPlayback(false);
            setPlayhead(range.start, false);
            playPlayback();
            return;
        }
        setPlayhead(nextFrame, false);
        const now = performance.now ? performance.now() : Date.now();
        if (now - Number(transport.lastMeterAt || 0) > 42) {
            transport.lastMeterAt = now;
            const dom = transport.dom || {};
            const meterSnapshot = { tracks: {}, master: { peak: 0, rms: 0 }, playing: true, playhead: transport.playhead };
            for (const [track, analyser] of transport.analysers.entries()) {
                const meter = meterFromAnalyser(analyser);
                meterSnapshot.tracks[track] = meter;
                const bars = dom.trackBars?.get(track) || [];
                bars.forEach((el) => { el.style.width = `${Math.round(meter.peak * 100)}%`; });
            }
            const master = meterFromAnalyser(transport.masterAnalyser);
            meterSnapshot.master = master;
            transport.lastMeterSnapshot = meterSnapshot;
            (dom.masterBars || []).forEach((el) => { el.style.width = `${Math.round(master.peak * 100)}%`; });
            (dom.masterReadouts || []).forEach((el) => { el.textContent = `PK ${String(Math.round(master.peak * 100)).padStart(3, "0")} RMS ${String(Math.round(master.rms * 100)).padStart(3, "0")}`; });
            updateInlineEfxRealtime();
        }
        if (transport.playhead >= totalFrames()) {
            stopPlayback(true);
            setPlayhead(0, false);
            return;
        }
        transport.raf = requestAnimationFrame(updateMeters);
    };
    const playPlayback = async () => {
        stopPlayback(false);
        const ctx = await ensureAudioContext();
        transport.audioContext = ctx;
        const impulseCache = new Map();
        const buildImpulse = (seconds = 1.8, damp = .35) => {
            const key = `${Number(seconds || 1.8).toFixed(2)}:${Number(damp || .35).toFixed(2)}:${ctx.sampleRate}`;
            if (impulseCache.has(key)) return impulseCache.get(key);
            const length = Math.max(1, Math.round(ctx.sampleRate * Math.max(.1, Math.min(8, Number(seconds || 1.8)))));
            const impulse = ctx.createBuffer(2, length, ctx.sampleRate);
            for (let channel = 0; channel < impulse.numberOfChannels; channel += 1) {
                const data = impulse.getChannelData(channel);
                for (let i = 0; i < length; i += 1) {
                    const decay = Math.pow(1 - i / length, 1 + Math.max(0, Math.min(1, Number(damp || .35))) * 5);
                    data[i] = (Math.random() * 2 - 1) * decay;
                }
            }
            impulseCache.set(key, impulse);
            return impulse;
        };
        const applyWetDryInsert = (inputNode, mixValue, buildWetChain) => {
            const mix = Math.max(0, Math.min(1, Number(mixValue || 0)));
            if (mix <= 0.0001) return inputNode;
            const dry = ctx.createGain();
            const wetInput = ctx.createGain();
            const wetGain = ctx.createGain();
            const output = ctx.createGain();
            dry.gain.value = Math.sqrt(Math.max(0, 1 - mix));
            wetGain.gain.value = Math.sqrt(mix);
            inputNode.connect(dry);
            dry.connect(output);
            inputNode.connect(wetInput);
            const wetTail = buildWetChain(wetInput) || wetInput;
            wetTail.connect(wetGain);
            wetGain.connect(output);
            return output;
        };
        const applyStereoFieldStage = (inputNode, options = {}) => {
            const widthValue = Math.max(0, Math.min(1, Number(options.width ?? 1)));
            const monoValue = Number(options.mono || 0) >= 0.5;
            const mixValue = options.mix == null ? 1 : Math.max(0, Math.min(1, Number(options.mix || 0)));
            const panValue = Math.max(-1, Math.min(1, Number(options.pan || 0) + (Number(options.angle || 0) / 45)));
            if (monoValue || Math.abs(widthValue - 1) > 0.001) {
                if (ctx.createChannelSplitter && ctx.createChannelMerger) {
                    inputNode = applyWetDryInsert(inputNode, mixValue, (entry) => {
                        const width = monoValue ? 0 : widthValue;
                        const splitter = ctx.createChannelSplitter(2);
                        const merger = ctx.createChannelMerger(2);
                        const leftToLeft = ctx.createGain();
                        const rightToLeft = ctx.createGain();
                        const leftToRight = ctx.createGain();
                        const rightToRight = ctx.createGain();
                        leftToLeft.gain.value = 0.5 * (1 + width);
                        rightToLeft.gain.value = 0.5 * (1 - width);
                        leftToRight.gain.value = 0.5 * (1 - width);
                        rightToRight.gain.value = 0.5 * (1 + width);
                        entry.connect(splitter);
                        splitter.connect(leftToLeft, 0);
                        splitter.connect(rightToLeft, 1);
                        splitter.connect(leftToRight, 0);
                        splitter.connect(rightToRight, 1);
                        leftToLeft.connect(merger, 0, 0);
                        rightToLeft.connect(merger, 0, 0);
                        leftToRight.connect(merger, 0, 1);
                        rightToRight.connect(merger, 0, 1);
                        return merger;
                    });
                }
            }
            if (Math.abs(panValue) > 0.001 && ctx.createStereoPanner) {
                const panNode = ctx.createStereoPanner();
                panNode.pan.value = panValue;
                inputNode.connect(panNode);
                return panNode;
            }
            return inputNode;
        };
        const applyMasterInsertFx = (inputNode, fx) => {
            const kind = String(fx?.type || "");
            if (kind === "reverb" && ctx.createConvolver) {
                return applyWetDryInsert(inputNode, fx?.params?.mix ?? .18, (entry) => {
                    const convolver = ctx.createConvolver();
                    convolver.normalize = true;
                    convolver.buffer = buildImpulse(fx?.params?.decay ?? 1.8, fx?.params?.damp ?? .35);
                    entry.connect(convolver);
                    return convolver;
                });
            }
            if (kind === "delay") {
                return applyWetDryInsert(inputNode, fx?.params?.mix ?? .2, (entry) => {
                    const delay = ctx.createDelay(2.5);
                    const feedback = ctx.createGain();
                    const tone = ctx.createBiquadFilter();
                    delay.delayTime.value = Math.max(.03, Math.min(1.5, Number(fx?.params?.time ?? .18)));
                    feedback.gain.value = Math.max(0, Math.min(.9, Number(fx?.params?.feedback ?? .28)));
                    tone.type = "lowpass";
                    tone.frequency.value = Math.max(200, Math.min(12000, Number(fx?.params?.filter ?? 4200)));
                    entry.connect(delay);
                    delay.connect(tone);
                    tone.connect(feedback);
                    feedback.connect(delay);
                    return tone;
                });
            }
            if (kind === "utility") return applyStereoFieldStage(inputNode, {
                width: fx?.params?.width ?? 1,
                mono: fx?.params?.mono ?? 0,
                pan: fx?.params?.pan ?? 0,
                mix: 1,
            });
            if (kind === "stereo") return applyStereoFieldStage(inputNode, {
                width: fx?.params?.width ?? 1.15,
                angle: fx?.params?.angle ?? 0,
                mix: fx?.params?.mix ?? .5,
            });
            return inputNode;
        };
        const applyTrackInsertFx = (inputNode, fx) => {
            const kind = String(fx?.type || "");
            if (kind === "utility") {
                const gainNode = ctx.createGain();
                gainNode.gain.value = Math.pow(10, Math.max(-24, Math.min(24, Number(fx?.params?.gain ?? 0))) / 20);
                inputNode.connect(gainNode);
                return applyStereoFieldStage(gainNode, {
                    width: fx?.params?.width ?? 1,
                    mono: fx?.params?.mono ?? 0,
                    pan: fx?.params?.pan ?? 0,
                    mix: 1,
                });
            }
            if (kind === "stereo") return applyStereoFieldStage(inputNode, {
                width: fx?.params?.width ?? 1.15,
                angle: fx?.params?.angle ?? 0,
                mix: fx?.params?.mix ?? .5,
            });
            if (kind === "reverb" || kind === "delay") return applyMasterInsertFx(inputNode, fx);
            return inputNode;
        };
        const soloed = segments().some((seg) => seg.solo);
        const trackSoloed = Array.from({ length: Math.max(1, Number(state.audioTrackCount || 4)) }, (_, index) => trackSettings(index)).some((item) => item.solo);
        const range = loopRange();
        if (state.loopEnabled && range && (transport.playhead < range.start || transport.playhead >= range.end)) {
            setPlayhead(range.start, false);
        }
        const masterGain = ctx.createGain();
        masterGain.gain.value = Math.max(0, Math.min(2, Number(state.masterAudioGain ?? 1) || 1));
        transport.masterGainNode = masterGain;
        const masterAnalyser = ctx.createAnalyser();
        masterAnalyser.fftSize = 256;
        let masterLast = masterGain;
        state.masterBus.effectChain = Array.isArray(state.masterBus.effectChain) ? state.masterBus.effectChain : JSON.parse(JSON.stringify(DEFAULT_MASTER_CHAIN));
        const masterChain = state.masterBus.effectChain.map(normalizeEffect);
        const masterEqFx = masterChain.find((fx) => fx.enabled !== false && fx.type === "eq");
        const masterCompFx = masterChain.find((fx) => fx.enabled !== false && fx.type === "compressor");
        const masterLimFx = masterChain.find((fx) => fx.enabled !== false && fx.type === "limiter");
        const addMasterBiquad = (type, freq, gain = 0, q = .707) => {
            const filter = ctx.createBiquadFilter();
            filter.type = type;
            filter.frequency.value = Math.max(10, Math.min(22000, Number(freq || 0)));
            filter.Q.value = Math.max(.1, Math.min(18, Number(q || .707)));
            if ("gain" in filter) filter.gain.value = Math.max(-36, Math.min(36, Number(gain || 0)));
            masterLast.connect(filter);
            masterLast = filter;
        };
        if (masterEqFx) {
            const masterEqFreq = eqBandFrequencies(masterEqFx);
            if (masterEqFx.params?.lowCut) addMasterBiquad("highpass", masterEqFx.params.lowCutFreq || 80);
            if (Number(masterEqFx.params?.low || 0)) addMasterBiquad("lowshelf", masterEqFreq.low, masterEqFx.params.low, .7);
            if (Number(masterEqFx.params?.mid || 0)) addMasterBiquad("peaking", masterEqFreq.mid, masterEqFx.params.mid, Math.max(.2, Number(masterEqFx.params?.q || 1.2)));
            if (Number(masterEqFx.params?.high || 0)) addMasterBiquad("highshelf", masterEqFreq.high, masterEqFx.params.high, .7);
            if (masterEqFx.params?.highCut) addMasterBiquad("lowpass", masterEqFx.params.highCutFreq || 12000);
        }
        if (state.masterAudioNormalize && ctx.createDynamicsCompressor) {
            const normalizer = ctx.createDynamicsCompressor();
            normalizer.threshold.value = -24;
            normalizer.knee.value = 18;
            normalizer.ratio.value = 3.5;
            normalizer.attack.value = .004;
            normalizer.release.value = .14;
            masterLast.connect(normalizer);
            masterLast = normalizer;
        }
        const masterCompAmount = masterCompFx ? Math.max(.01, Math.min(1, Math.abs(Number(masterCompFx.params?.threshold ?? -18)) / 60)) : Math.max(0, Math.min(1, Number(state.masterBus?.compressor || 0)));
        if ((masterLimFx || state.masterBus?.limiter || masterCompAmount > 0) && ctx.createDynamicsCompressor) {
            const comp = ctx.createDynamicsCompressor();
            comp.threshold.value = masterLimFx ? Math.max(-24, Math.min(0, Number(masterLimFx.params?.ceiling ?? -1) - 5)) : masterCompFx ? Math.max(-60, Math.min(0, Number(masterCompFx.params?.threshold ?? -18))) : -20 - masterCompAmount * 18;
            comp.knee.value = masterLimFx ? 3 : 18;
            comp.ratio.value = masterLimFx ? 18 : masterCompFx ? Math.max(1, Math.min(20, Number(masterCompFx.params?.ratio ?? 4))) : 1 + masterCompAmount * 7;
            comp.attack.value = masterCompFx ? Math.max(.001, Number(masterCompFx.params?.attack ?? 6) / 1000) : .003;
            comp.release.value = masterLimFx ? Math.max(.02, Number(masterLimFx.params?.release ?? 120) / 1000) : masterCompFx ? Math.max(.02, Number(masterCompFx.params?.release ?? 180) / 1000) : .12;
            masterLast.connect(comp);
            masterLast = comp;
        }
        for (const fx of masterChain) {
            if (!fx || fx.enabled === false || !["reverb", "delay"].includes(String(fx.type || ""))) continue;
            masterLast = applyMasterInsertFx(masterLast, fx);
        }
        if (!masterChain.some((fx) => fx?.enabled !== false && fx?.type === "reverb") && Number(state.masterBus?.reverbSend || 0) > 0) {
            masterLast = applyMasterInsertFx(masterLast, { type: "reverb", params: { decay: .9 + Number(state.masterBus?.reverbSend || 0) * 3.5, damp: .28 + Number(state.masterBus?.reverbSend || 0) * .45, mix: Math.min(.8, Math.max(.04, Number(state.masterBus?.reverbSend || 0) * .7)) } });
        }
        if (!masterChain.some((fx) => fx?.enabled !== false && fx?.type === "delay") && Number(state.masterBus?.delaySend || 0) > 0) {
            masterLast = applyMasterInsertFx(masterLast, { type: "delay", params: { time: .18, feedback: Math.min(.55, Number(state.masterBus?.delaySend || 0) * .45), filter: 4200, mix: Math.min(.7, Number(state.masterBus?.delaySend || 0)) } });
        }
        if (state.masterMono && ctx.createChannelSplitter && ctx.createChannelMerger) {
            const splitter = ctx.createChannelSplitter(2);
            const mono = ctx.createGain();
            const merger = ctx.createChannelMerger(2);
            mono.gain.value = .5;
            masterLast.connect(splitter);
            splitter.connect(mono, 0);
            splitter.connect(mono, 1);
            mono.connect(merger, 0, 0);
            mono.connect(merger, 0, 1);
            masterLast = merger;
        }
        masterLast.connect(masterAnalyser);
        masterAnalyser.connect(ctx.destination);
        const trackNodes = new Map();
        const getTrack = (track) => {
            if (trackNodes.has(track)) return trackNodes.get(track);
            const trackState = trackSettings(track);
            const gain = ctx.createGain();
            const trackGainLinear = Math.pow(10, Math.max(-24, Math.min(24, Number(trackState.gainDb ?? 0) || 0)) / 20);
            gain.gain.value = Math.max(0, Math.min(4, Number(trackState.volume ?? 1) * trackGainLinear));
            let last = gain;
            const liveTrackNodes = { gainNode: gain, panNode: null };
            if (ctx.createStereoPanner) {
                const trackPan = ctx.createStereoPanner();
                trackPan.pan.value = Math.max(-1, Math.min(1, Number(trackState.pan || 0)));
                last.connect(trackPan);
                last = trackPan;
                liveTrackNodes.panNode = trackPan;
            }
            if (trackState.normalize && ctx.createDynamicsCompressor) {
                const normalizer = ctx.createDynamicsCompressor();
                normalizer.threshold.value = -24;
                normalizer.knee.value = 18;
                normalizer.ratio.value = 3.5;
                normalizer.attack.value = .004;
                normalizer.release.value = .14;
                last.connect(normalizer);
                last = normalizer;
            }
            const analyser = ctx.createAnalyser();
            analyser.fftSize = 256;
            last.connect(analyser);
            if (!trackState.bypassEffects && trackState.reverb && ctx.createConvolver) {
                const send = ctx.createGain();
                const convolver = ctx.createConvolver();
                const wet = ctx.createGain();
                send.gain.value = .2;
                wet.gain.value = .18;
                convolver.buffer = buildImpulse(2.4, .36);
                last.connect(send);
                send.connect(convolver);
                convolver.connect(wet);
                wet.connect(masterGain);
            }
            analyser.connect(masterGain);
            trackNodes.set(track, gain);
            transport.trackMixNodes.set(track, liveTrackNodes);
            transport.analysers.set(track, analyser);
            return gain;
        };
        const startFrame = transport.playhead;
        transport.startFrame = startFrame;
        transport.startedAt = ctx.currentTime;
        transport.playing = true;
        transport.pendingLoopRestart = false;
        transport.lastMeterAt = 0;
        for (const seg of segments()) {
            const trackIndex = Math.max(0, Number(seg.track || 0));
            const trackState = trackSettings(trackIndex);
            if (trackState.mute || (trackSoloed && !trackState.solo)) continue;
            if (seg.mute || (soloed && !seg.solo)) continue;
            const segStart = Math.round(Number(seg.start || 0));
            const segLen = Math.max(1, Math.round(Number(seg.length || 1)));
            const segEnd = segStart + segLen;
            if (segEnd <= startFrame) continue;
            try {
                const buffer = await getBuffer(seg);
                if (!buffer) continue;
                const relStart = Math.max(0, startFrame - segStart);
                const when = Math.max(0, framesToSeconds(segStart - startFrame));
                const offset = Math.max(0, framesToSeconds(Number(seg.trimStart || 0) + relStart));
                const dur = Math.max(0.02, Math.min(framesToSeconds(segLen - relStart), buffer.duration - offset));
                if (dur <= 0) continue;
                const source = ctx.createBufferSource();
                source.buffer = buffer;
                source.playbackRate.value = Math.max(.25, Math.min(4, Number(seg.timeStretch || 1) || 1)) * Math.pow(2, Number(seg.pitchSemitones || 0) / 12);
                const clipGain = ctx.createGain();
                clipGain.gain.value = effectiveGain(seg);
                const nodes = [];
                const addBiquad = (type, freq, gain = 0, q = .707) => {
                    const node = ctx.createBiquadFilter();
                    node.type = type;
                    node.frequency.value = Math.max(10, Math.min(22000, Number(freq || 0)));
                    node.Q.value = Math.max(.1, Math.min(18, Number(q || .707)));
                    if ("gain" in node) node.gain.value = Math.max(-36, Math.min(36, Number(gain || 0)));
                    nodes.push(node);
                    return node;
                };
                if (!trackState.bypassEffects && Number(seg.hpfHz || 0) > 10) addBiquad("highpass", seg.hpfHz, 0, .7);
                if (!trackState.bypassEffects && Number(seg.lpfHz || 22000) < 21950) addBiquad("lowpass", seg.lpfHz, 0, .7);
                if (!trackState.bypassEffects && Number(seg.eqLowDb || 0)) addBiquad("lowshelf", 140, seg.eqLowDb, .7);
                if (!trackState.bypassEffects && Number(seg.eqMidDb || 0)) addBiquad("peaking", 1200, seg.eqMidDb, 1.1);
                if (!trackState.bypassEffects && Number(seg.eqHighDb || 0)) addBiquad("highshelf", 6200, seg.eqHighDb, .7);
                const trackEq = trackState.bypassEffects ? null : trackState.effectChain.find((fx) => fx.enabled !== false && fx.type === "eq");
                if (trackEq) {
                    const trackEqFreq = eqBandFrequencies(trackEq);
                    if (trackEq.params?.lowCut) addBiquad("highpass", trackEq.params.lowCutFreq || 80, 0, .7);
                    if (Number(trackEq.params?.low || 0)) addBiquad("lowshelf", trackEqFreq.low, trackEq.params.low, .7);
                    if (Number(trackEq.params?.mid || 0)) addBiquad("peaking", trackEqFreq.mid, trackEq.params.mid, Math.max(.2, Number(trackEq.params?.q || 1.2)));
                    if (Number(trackEq.params?.high || 0)) addBiquad("highshelf", trackEqFreq.high, trackEq.params.high, .7);
                    if (trackEq.params?.highCut) addBiquad("lowpass", trackEq.params.highCutFreq || 12000, 0, .7);
                }
                if (!trackState.bypassEffects && Number(seg.compressor || 0) > 0 && ctx.createDynamicsCompressor) {
                    const comp = ctx.createDynamicsCompressor();
                    const amount = Math.max(0, Math.min(1, Number(seg.compressor || 0)));
                    comp.threshold.value = -18 - amount * 24;
                    comp.knee.value = 18;
                    comp.ratio.value = 1 + amount * 11;
                    comp.attack.value = .006;
                    comp.release.value = .18;
                    nodes.push(comp);
                }
                const trackComp = trackState.bypassEffects ? null : trackState.effectChain.find((fx) => fx.enabled !== false && fx.type === "compressor");
                if (trackComp && ctx.createDynamicsCompressor) {
                    const comp = ctx.createDynamicsCompressor();
                    comp.threshold.value = Math.max(-60, Math.min(0, Number(trackComp.params?.threshold ?? -18)));
                    comp.knee.value = 12;
                    comp.ratio.value = Math.max(1, Math.min(20, Number(trackComp.params?.ratio ?? 4)));
                    comp.attack.value = Math.max(.001, Number(trackComp.params?.attack ?? 6) / 1000);
                    comp.release.value = Math.max(.02, Number(trackComp.params?.release ?? 180) / 1000);
                    nodes.push(comp);
                }
                source.connect(clipGain);
                let last = clipGain;
                for (const node of nodes) {
                    last.connect(node);
                    last = node;
                }
                const trackInsertChain = !trackState.bypassEffects && Array.isArray(trackState.effectChain)
                    ? trackState.effectChain.map(normalizeEffect).filter((fx) => fx.enabled !== false && ["reverb", "delay", "utility"].includes(String(fx.type || "")))
                    : [];
                for (const fx of trackInsertChain) {
                    last = applyTrackInsertFx(last, fx);
                }
                if (Number(seg.stereoWidth ?? 1) !== 1 && Number(buffer?.numberOfChannels || 1) > 1) {
                    last = applyStereoFieldStage(last, { width: seg.stereoWidth ?? 1, mix: 1 });
                }
                if (ctx.createStereoPanner) {
                    const pan = ctx.createStereoPanner();
                    pan.pan.value = Math.max(-1, Math.min(1, Number(seg.pan || 0)));
                    last.connect(pan);
                    last = pan;
                }
                if (!trackState.bypassEffects && Number(seg.delaySend || 0) > 0) {
                    const delay = ctx.createDelay(1.5);
                    const feedback = ctx.createGain();
                    const send = ctx.createGain();
                    delay.delayTime.value = .18;
                    feedback.gain.value = Math.min(.55, Number(seg.delaySend || 0) * .45);
                    send.gain.value = Math.min(.7, Number(seg.delaySend || 0));
                    last.connect(send);
                    send.connect(delay);
                    delay.connect(feedback);
                    feedback.connect(delay);
                    delay.connect(getTrack(trackIndex));
                }
                if (!trackState.bypassEffects && Number(seg.reverbSend || 0) > 0 && ctx.createConvolver) {
                    const convolver = ctx.createConvolver();
                    const send = ctx.createGain();
                    const wet = ctx.createGain();
                    convolver.buffer = buildImpulse(.9 + Number(seg.reverbSend || 0) * 3.5, .28 + Number(seg.reverbSend || 0) * .45);
                    send.gain.value = Math.min(.85, Math.max(.04, Number(seg.reverbSend || 0) * .75));
                    wet.gain.value = Math.min(.8, Math.max(.04, Number(seg.reverbSend || 0) * .7));
                    last.connect(send);
                    send.connect(convolver);
                    convolver.connect(wet);
                    wet.connect(getTrack(trackIndex));
                }
                last.connect(getTrack(trackIndex));
                source.start(ctx.currentTime + when, offset, dur);
                transport.sources.push({ source, seg });
            } catch (err) {
                console.warn("[IAMCCS AudioBoardArranger] playback skipped clip", seg, err);
            }
        }
        transport.masterAnalyser = masterAnalyser;
        transport.helper = "Playing timeline. Meters are live from WebAudio.";
        draw();
        updateMeters();
    };
    const exposeTransportApi = () => {
        node._iamccsAudioBoardTransport = {
            play: () => playPlayback(),
            stop: (redraw = true) => stopPlayback(redraw),
            toggle: () => transport.playing ? stopPlayback(true) : playPlayback(),
            rewind: () => {
                stopPlayback(false);
                setPlayhead(0, true);
            },
            setPlayhead: (frame, redraw = false) => setPlayhead(frame, redraw),
            setTrackVolume: (track, value) => {
                trackSettings(track).volume = Math.max(0, Math.min(2, Number(value ?? 1) || 1));
                const live = transport.trackMixNodes.get(Math.max(0, Number(track || 0)));
                if (!live?.gainNode) return false;
                const gainLinear = Math.pow(10, Math.max(-24, Math.min(24, Number(trackSettings(track).gainDb ?? 0) || 0)) / 20);
                smoothAudioParam(live.gainNode.gain, Math.max(0, Math.min(4, trackSettings(track).volume * gainLinear)));
                return true;
            },
            setTrackGainDb: (track, value) => {
                trackSettings(track).gainDb = Math.max(-24, Math.min(24, Number(value ?? 0) || 0));
                const live = transport.trackMixNodes.get(Math.max(0, Number(track || 0)));
                if (!live?.gainNode) return false;
                const gainLinear = Math.pow(10, trackSettings(track).gainDb / 20);
                smoothAudioParam(live.gainNode.gain, Math.max(0, Math.min(4, trackSettings(track).volume * gainLinear)));
                return true;
            },
            setTrackPan: (track, value) => {
                trackSettings(track).pan = Math.max(-1, Math.min(1, Number(value ?? 0) || 0));
                const live = transport.trackMixNodes.get(Math.max(0, Number(track || 0)));
                if (!live?.panNode?.pan) return false;
                smoothAudioParam(live.panNode.pan, trackSettings(track).pan);
                return true;
            },
            setTrackToggle: (track, key, enabled) => {
                const cleanKey = ["mute", "solo", "normalize", "bypassEffects", "reverb", "lock"].includes(String(key || "")) ? String(key) : "";
                if (!cleanKey) return false;
                trackSettings(track)[cleanKey] = Boolean(enabled);
                restartPlaybackIfNeeded(`track_${cleanKey}_external`);
                return true;
            },
            setMasterGain: (value) => {
                state.masterAudioGain = Math.max(0, Math.min(2, Number(value ?? 1) || 1));
                if (!transport.masterGainNode?.gain) return false;
                smoothAudioParam(transport.masterGainNode.gain, state.masterAudioGain);
                return true;
            },
            rebuildFx: (reason = "transport_rebuild") => {
                restartPlaybackIfNeeded(reason);
                return true;
            },
            applyExternalState: (nextState, reason = "external_state") => {
                if (!nextState || typeof nextState !== "object") return false;
                const clean = JSON.parse(JSON.stringify(nextState));
                Object.keys(state).forEach((key) => delete state[key]);
                Object.assign(state, clean);
                restartPlaybackIfNeeded(reason);
                draw();
                return true;
            },
            snapshot: () => ({
                playing: Boolean(transport.playing),
                playhead: Math.max(0, Math.round(Number(transport.playhead || 0))),
                totalFrames: totalFrames(),
                fps: fps(),
                loopEnabled: Boolean(state.loopEnabled),
                helper: String(transport.helper || ""),
                meters: transport.lastMeterSnapshot || { tracks: {}, master: { peak: 0, rms: 0 }, playing: false, playhead: transport.playhead },
            }),
        };
    };
    exposeTransportApi();
    const drawWave = (clip, seg) => {
        ensureSegmentWaveform(seg);
        const allPeaks = Array.isArray(seg.waveformPeaks) ? seg.waveformPeaks : [];
        const durationFrames = Math.max(1, Number(seg.audioDurationFrames || seg.length || 1));
        const trimStart = Math.max(0, Number(seg.trimStart || 0));
        const trimEnd = Math.max(trimStart + 1, trimStart + Math.max(1, Number(seg.length || 1)));
        const startIndex = allPeaks.length ? Math.max(0, Math.min(allPeaks.length - 1, Math.floor((trimStart / durationFrames) * allPeaks.length))) : 0;
        const endIndex = allPeaks.length ? Math.max(startIndex + 1, Math.min(allPeaks.length, Math.ceil((trimEnd / durationFrames) * allPeaks.length))) : 0;
        const peaks = allPeaks.slice(startIndex, endIndex);
        const canvas = document.createElement("canvas");
        canvas.classList.add("iamccs-clip-wave-svg", "iamccs-clip-wave-canvas");
        canvas.dataset.sourceStart = String(trimStart);
        canvas.dataset.sourceEnd = String(trimEnd);
        const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
        const cssW = Math.max(64, Math.round(Number(seg.length || 1) * pxPerFrame()));
        const cssH = Math.max(46, Math.round(view().trackHeight - 28));
        const w = Math.max(96, Math.min(4096, Math.round(cssW * dpr)));
        const h = Math.max(50, Math.min(260, Math.round(cssH * dpr)));
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d");
        if (!ctx) {
            clip.appendChild(canvas);
            return;
        }
        const bg = ctx.createLinearGradient(0, 0, 0, h);
        bg.addColorStop(0, "#376a9b");
        bg.addColorStop(.52, "#315f8f");
        bg.addColorStop(1, "#23496f");
        ctx.fillStyle = bg;
        ctx.fillRect(0, 0, w, h);
        ctx.strokeStyle = "rgba(255,255,255,.12)";
        ctx.lineWidth = 1;
        for (let x = 0; x <= w; x += Math.max(36, Math.round(w / 18))) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, h);
            ctx.stroke();
        }
        ctx.strokeStyle = "rgba(255,255,255,.28)";
        ctx.beginPath();
        ctx.moveTo(0, h * .5);
        ctx.lineTo(w, h * .5);
        ctx.stroke();
        const normPeak = (raw) => {
            if (raw && typeof raw === "object") {
                return {
                    min: Math.max(-1, Math.min(1, Number(raw.min) || 0)),
                    max: Math.max(-1, Math.min(1, Number(raw.max) || 0)),
                    rms: Math.max(0, Math.min(1, Math.abs(Number(raw.rms) || 0))),
                };
            }
            const p = Math.max(0, Math.min(1, Math.abs(Number(raw) || 0)));
            return { min: -p, max: p, rms: p * .66 };
        };
        const peakValue = (raw) => {
            const p = normPeak(raw);
            return Math.max(Math.abs(p.min), Math.abs(p.max), p.rms);
        };
        if (!peaks.length) {
            ctx.fillStyle = "rgba(235,248,255,.76)";
            ctx.font = `900 ${Math.max(10, Math.round(12 * dpr))}px ui-monospace, Consolas, monospace`;
            ctx.textAlign = "center";
            ctx.fillText(waveformLoading.has(seg.id) ? "decoding real waveform..." : "no waveform peaks", w * .5, h * .53);
            clip.appendChild(canvas);
            return;
        }
        const visualMax = Math.max(.05, ...peaks.map(peakValue));
        const scale = Math.min(1.65, .94 / visualMax);
        const columnPeak = (x) => {
            const from = Math.floor((x / Math.max(1, w)) * peaks.length);
            const to = Math.max(from + 1, Math.floor(((x + 1) / Math.max(1, w)) * peaks.length));
            let min = 0;
            let max = 0;
            let rms = 0;
            let n = 0;
            for (let i = from; i < Math.min(peaks.length, to); i += 1) {
                const p = normPeak(peaks[i]);
                min = Math.min(min, p.min);
                max = Math.max(max, p.max);
                rms += p.rms;
                n += 1;
            }
            if (!n) {
                const p = normPeak(peaks[Math.min(peaks.length - 1, Math.max(0, from))]);
                return p;
            }
            return { min, max, rms: rms / n };
        };
        const center = h * .5;
        const amp = h * .46;
        const top = [];
        const bottom = [];
        const rmsTop = [];
        const rmsBottom = [];
        for (let x = 0; x < w; x += 1) {
            const p = columnPeak(x);
            top.push([x, center - Math.max(1, p.max * scale * amp)]);
            bottom.unshift([x, center + Math.max(1, Math.abs(p.min) * scale * amp)]);
            rmsTop.push([x, center - Math.max(.5, p.rms * scale * amp * .62)]);
            rmsBottom.unshift([x, center + Math.max(.5, p.rms * scale * amp * .62)]);
        }
        const body = ctx.createLinearGradient(0, 0, 0, h);
        body.addColorStop(0, "rgba(236,249,255,.96)");
        body.addColorStop(.48, "rgba(178,221,245,.82)");
        body.addColorStop(.52, "rgba(172,215,241,.80)");
        body.addColorStop(1, "rgba(236,249,255,.94)");
        ctx.fillStyle = "rgba(255,255,255,.16)";
        ctx.beginPath();
        rmsTop.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
        rmsBottom.forEach(([x, y]) => ctx.lineTo(x, y));
        ctx.closePath();
        ctx.fill();
        ctx.fillStyle = body;
        ctx.beginPath();
        top.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
        bottom.forEach(([x, y]) => ctx.lineTo(x, y));
        ctx.closePath();
        ctx.fill();
        ctx.strokeStyle = "rgba(255,255,255,.84)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        top.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
        bottom.slice().reverse().forEach(([x, y]) => ctx.lineTo(x, y));
        ctx.stroke();
        ctx.strokeStyle = "rgba(255,255,255,.32)";
        const detailStep = w > 2200 ? 2 : 1;
        for (let x = 0; x < w; x += detailStep) {
            const p = columnPeak(x);
            const y1 = center - Math.max(1, p.max * scale * amp);
            const y2 = center + Math.max(1, Math.abs(p.min) * scale * amp);
            ctx.beginPath();
            ctx.moveTo(x + .5, y1);
            ctx.lineTo(x + .5, y2);
            ctx.stroke();
        }
        clip.appendChild(canvas);
    };
    const updateClipStyle = (clip, seg) => {
        clip.style.left = `${frameToX(seg.start)}px`;
        clip.style.width = `${Math.max(18, Math.round(Number(seg.length || 1) * pxPerFrame()))}px`;
        const time = clip.querySelector(".iamccs-audio-clip-time");
        if (time) time.textContent = `${fmtTime(seg.start)} - ${fmtTime(Number(seg.start || 0) + Number(seg.length || 1))}`;
        const trim = clip.querySelector(".iamccs-audio-clip-trim");
        const srcEnd = Number(seg.trimStart || 0) + Number(seg.length || 1);
        if (trim) trim.textContent = `src +${fmtTime(Number(seg.trimStart || 0))} -> +${fmtTime(srcEnd)}`;
        const marker = clip.querySelector(".iamccs-clip-source-marker");
        if (marker) marker.textContent = `SRC ${fmtTime(srcEnd)}`;
        const waveKey = `${Math.round(Number(seg.trimStart || 0))}:${Math.round(Number(seg.length || 1))}:${Array.isArray(seg.waveformPeaks) ? seg.waveformPeaks.length : 0}:${seg.waveformReal ? 1 : 0}`;
        if (clip.dataset.waveKey !== waveKey) {
            clip.dataset.waveKey = waveKey;
            clip.querySelectorAll(".iamccs-clip-wave-svg").forEach((el) => el.remove());
            drawWave(clip, seg);
        }
    };
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    const refreshTrackClipVisuals = (trackIndex) => {
        const track = Math.max(0, Number(trackIndex || 0));
        root.querySelectorAll(".iamccs-audio-clip[data-seg-id]").forEach((clipEl) => {
            const seg = segments().find((item) => String(item.id || "") === String(clipEl.dataset.segId || ""));
            if (!seg || Math.max(0, Number(seg.track || 0)) !== track) return;
            clipEl.dataset.waveKey = "";
            updateClipStyle(clipEl, seg);
        });
    };
    const attachClipPointer = (clip, seg) => {
        const beginDrag = (event, forcedMode = "") => {
            if (seg.lock || trackSettings(seg.track).lock) return;
            event.stopPropagation();
            event.preventDefault();
            root.focus();
            selectedId = seg.id;
            state.selectedMixer = { type: "track", track: Math.max(0, Number(seg.track || 0)) };
            const rect = clip.getBoundingClientRect();
            const tool = view().tool;
            const pickedMode = forcedMode || "move";
            if (pickedMode === "cut" || (!forcedMode && tool === "cut")) {
                // scaleX converts screen-pixels (getBoundingClientRect) to CSS-pixels
                // necessary because LiteGraph applies a CSS transform scale to the widget.
                const scaleX = Math.max(0.001, clip.offsetWidth / Math.max(1, rect.width));
                const clickInClip = (event.clientX - rect.left) * scaleX; // CSS px from clip left
                const frame = Math.max(
                    Number(seg.start || 0),
                    Math.min(
                        Number(seg.start || 0) + Number(seg.length || 1),
                        Math.round(Number(seg.start || 0) + clickInClip / pxPerFrame())
                    )
                );
                setPlayhead(frame, false);
                splitSelectedAtPlayhead();
                return;
            }
            const mode = pickedMode;
            dragPxPerFrame = Math.max(0.01, pxPerFrame());
            const startX = event.clientX;
            const startY = event.clientY;
            const start = Number(seg.start || 0);
            const length = Number(seg.length || 1);
            const trimStart = Number(seg.trimStart || 0);
            const track = Number(seg.track || 0);
            try { clip.setPointerCapture?.(event.pointerId); } catch {}
            clip.classList.toggle("is-trimming", mode === "trim_left" || mode === "trim_right" || mode === "trim");
            clip.classList.toggle("is-moving", mode === "move");
            const marker = clip.querySelector(".iamccs-clip-source-marker");
            if (marker) {
                marker.style.display = mode === "trim_left" || mode === "trim_right" || mode === "trim" ? "block" : "none";
                marker.style.left = mode === "trim_left" ? "0" : "auto";
                marker.style.right = mode === "trim_left" ? "auto" : "0";
            }
            transport.helper = `${mode.replace("_", " ")} ${String(seg.name || seg.fileName || "clip")}`;
            const move = (moveEvent) => {
                const delta = Math.round((moveEvent.clientX - startX) / pxPerFrame());
                const trackDelta = Math.round((moveEvent.clientY - startY) / Math.max(1, view().trackHeight));
                if (mode === "trim_left") {
                    const applied = Math.max(-trimStart, Math.min(length - 1, delta));
                    seg.start = Math.max(0, start + applied);
                    seg.trimStart = Math.max(0, trimStart + applied);
                    seg.length = Math.max(1, length - applied);
                } else if (mode === "trim_right" || mode === "trim") {
                    seg.length = Math.max(1, Math.min(Number(seg.audioDurationFrames || 999999) - Number(seg.trimStart || 0), length + delta));
                } else {
                    seg.start = Math.max(0, start + delta);
                    seg.track = Math.max(0, Math.min(Math.max(1, Number(state.audioTrackCount || 4)) - 1, track + trackDelta));
                    clip.style.transform = `translateY(${trackDelta * Math.max(1, view().trackHeight)}px)`;
                }
                transport.helper = `${mode.replace("_", " ")} | ${fmtTime(seg.start)} - ${fmtTime(Number(seg.start || 0) + Number(seg.length || 1))} | source +${fmtTime(seg.trimStart || 0)}`;
                updateClipStyle(clip, seg);
                scheduleSilentStateWrite(`live_${mode}`);
                scheduleShotboardSync(`live_${mode}`);
                moveEvent.preventDefault();
                moveEvent.stopPropagation();
            };
            const up = (upEvent) => {
                window.removeEventListener("pointermove", move);
                window.removeEventListener("pointerup", up);
                if (liveStateWriteTimer) {
                    window.clearTimeout(liveStateWriteTimer);
                    liveStateWriteTimer = 0;
                }
                try { clip.releasePointerCapture?.(upEvent.pointerId); } catch {}
                clip.classList.remove("is-trimming", "is-moving");
                clip.style.transform = "";
                const marker = clip.querySelector(".iamccs-clip-source-marker");
                if (marker) marker.style.display = "none";
                if (shotboardSyncTimer) {
                    window.clearTimeout(shotboardSyncTimer);
                    shotboardSyncTimer = 0;
                }
                dragPxPerFrame = 0;
                addEdit(`${mode.replace("_", " ")} applied to clip.`);
                writeState(mode);
                draw();
            };
            window.addEventListener("pointermove", move);
            window.addEventListener("pointerup", up, { once: true });
        };
        const edgeMode = (event) => {
            const rect = clip.getBoundingClientRect();
            const edgeWidth = Math.min(40, Math.max(24, rect.width * .28));
            if (event.target?.closest?.(".iamccs-clip-handle.left")) return "trim_left";
            if (event.target?.closest?.(".iamccs-clip-handle.right")) return "trim_right";
            if (event.clientX - rect.left <= edgeWidth) return "trim_left";
            if (rect.right - event.clientX <= edgeWidth) return "trim_right";
            return "";
        };
        clip.onpointerdown = (event) => {
            if (event.button != null && event.button !== 0) return;
            const edge = edgeMode(event);
            if (edge) return beginDrag(event, edge);
            const tool = view().tool;
            if (tool === "cut") return beginDrag(event, "cut");
            if (tool === "trim") return beginDrag(event, "trim_right");
            beginDrag(event, "move");
        };
        clip.ondblclick = (event) => {
            event.stopPropagation();
            setPlayhead(Number(seg.start || 0), false);
            selectedId = seg.id;
            draw();
        };
        clip.oncontextmenu = (event) => {
            event.preventDefault();
            event.stopPropagation();
            selectedId = seg.id;
            state.selectedMixer = { type: "track", track: Math.max(0, Number(seg.track || 0)) };
            const frameAtClick = Number(seg.start || 0) + Math.round((event.clientX - clip.getBoundingClientRect().left) / Math.max(1, pxPerFrame()));
            setPlayhead(Math.max(Number(seg.start || 0), Math.min(Number(seg.start || 0) + Number(seg.length || 1), frameAtClick)), false);
            const actions = [
                { label: "Cut Here", run: () => splitSelectedAtPlayhead() },
                { label: "Trim In To Cursor", run: () => trimStartToPlayhead() },
                { label: "Trim Out To Cursor", run: () => trimEndToPlayhead() },
            ];
            for (let laneIndex = 0; laneIndex < Math.max(1, Number(state.audioTrackCount || 4)); laneIndex += 1) {
                if (laneIndex === Number(seg.track || 0)) continue;
                actions.push({ label: `Move To A${laneIndex + 1}`, run: () => moveSelectedClipToTrack(laneIndex) });
            }
            actions.push({ label: "Convert To Mono", run: () => applySelectedChannelMode("mono") });
            actions.push({ label: "Convert To Stereo", run: () => applySelectedChannelMode("stereo") });
            actions.push({ label: "Delete Clip", run: () => deleteSelectedClip(), klass: "danger" });
            showContextMenu(String(seg.name || seg.fileName || `Clip A${Number(seg.track || 0) + 1}`), event.clientX, event.clientY, actions);
            draw();
        };
    };
    const selectedClip = () => segments().find((seg) => seg.id === selectedId) || segments()[0] || null;
    const splitSelectedAtPlayhead = () => {
        const seg = selectedClip();
        if (!seg) return;
        const rel = transport.playhead - Number(seg.start || 0);
        if (rel <= 0 || rel >= Number(seg.length || 1)) {
            addEdit("Cut ignored: playhead is outside selected clip.");
            draw();
            return;
        }
        const right = { ...seg, id: newId("aud"), start: Number(seg.start || 0) + rel, length: Number(seg.length || 1) - rel, trimStart: Number(seg.trimStart || 0) + rel, name: `${String(seg.name || "clip")} B` };
        seg.length = rel;
        segments().push(right);
        selectedId = right.id;
        addEdit("Split selected clip at playhead.");
        writeState("cut");
        draw();
    };
    const trimStartToPlayhead = () => {
        const seg = selectedClip();
        if (!seg) return;
        const rel = transport.playhead - Number(seg.start || 0);
        if (rel <= 0 || rel >= Number(seg.length || 1)) return;
        seg.start = transport.playhead;
        seg.trimStart = Number(seg.trimStart || 0) + rel;
        seg.length = Number(seg.length || 1) - rel;
        addEdit("Trimmed clip start to playhead.");
        writeState("trim_start");
        draw();
    };
    const trimEndToPlayhead = () => {
        const seg = selectedClip();
        if (!seg) return;
        const rel = transport.playhead - Number(seg.start || 0);
        if (rel <= 0 || rel >= Number(seg.length || 1)) return;
        seg.length = rel;
        addEdit("Trimmed clip end to playhead.");
        writeState("trim_end");
        draw();
    };
    const nudgeSelected = (frames) => {
        const seg = selectedClip();
        if (!seg || seg.lock) return;
        seg.start = Math.max(0, Number(seg.start || 0) + frames);
        addEdit(`Nudged clip ${frames}f.`);
        writeState("nudge");
        draw();
    };
    const makeDraggableNumber = (input, options = {}) => {
        void options;
        return input;
    };
    root.addEventListener("keydown", (event) => {
        const tag = String(event.target?.tagName || "").toLowerCase();
        if (tag === "input" || tag === "select" || tag === "textarea") return;
        const selected = selectedClip();
        const step = event.shiftKey ? 12 : 1;
        if (event.code === "Space") {
            transport.playing ? stopPlayback(true) : playPlayback();
            event.preventDefault();
            return;
        }
        if (event.key === "+" || event.key === "=") {
            view().timeZoom = Math.min(8, view().timeZoom * 1.2);
            addEdit("Shortcut zoom in.");
            writeState("shortcut_zoom", false);
            draw();
            event.preventDefault();
            return;
        }
        if (event.key === "-" || event.key === "_") {
            view().timeZoom = Math.max(.35, view().timeZoom / 1.2);
            addEdit("Shortcut zoom out.");
            writeState("shortcut_zoom", false);
            draw();
            event.preventDefault();
            return;
        }
        if (event.key === "c" || event.key === "C") {
            setTool("cut", "Tool: cut. Click a clip to split at the clicked frame.");
            event.preventDefault();
            return;
        }
        if (event.key === "t" || event.key === "T") {
            setTool("trim", "Tool: trim. Drag a clip edge to trim.");
            event.preventDefault();
            return;
        }
        if (!selected) return;
        if (event.key === "Delete" || event.key === "Backspace") {
            state.audioSegments = segments().filter((seg) => seg.id !== selected.id);
            selectedId = "";
            addEdit("Shortcut deleted selected clip.");
            writeState("shortcut_delete");
            draw();
            event.preventDefault();
            return;
        }
        if (event.key === "m" || event.key === "M") selected.mute = !selected.mute;
        else if (event.key === "s" || event.key === "S") selected.solo = !selected.solo;
        else if (event.key === "n" || event.key === "N") selected.normalizeAudio = !selected.normalizeAudio;
        else if (event.key === "l" || event.key === "L") selected.lock = !selected.lock;
        else if (event.key === "ArrowLeft" || event.key === "ArrowRight") nudgeSelected((event.key === "ArrowLeft" ? -1 : 1) * step);
        else if (event.key === "[") trimStartToPlayhead();
        else if (event.key === "]") trimEndToPlayhead();
        else return;
        addEdit(`Shortcut edited selected clip.`);
        writeState("shortcut_edit");
        draw();
        event.preventDefault();
    });
    const addButton = (parent, label, handler, klass = "") => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = label;
        btn.className = klass;
        btn.onclick = handler;
        parent.appendChild(btn);
        return btn;
    };
    const deleteSelectedClip = () => {
        const selected = selectedClip();
        if (!selected) return;
        state.audioSegments = segments().filter((seg) => seg.id !== selected.id);
        selectedId = segments()[0]?.id || "";
        addEdit("Deleted selected clip.");
        writeState("delete");
        draw();
    };
    const moveSelectedClipToTrack = (track) => {
        const selected = selectedClip();
        if (!selected) return;
        selected.track = Math.max(0, Math.min(Math.max(1, Number(state.audioTrackCount || 4)) - 1, Number(track || 0)));
        state.selectedMixer = { type: "track", track: Number(selected.track || 0) };
        addEdit(`Moved selected clip to A${Number(selected.track || 0) + 1}.`);
        writeState("move_track");
        draw();
    };
    const applySelectedChannelMode = (mode) => {
        const selected = selectedClip();
        if (!selected) return;
        const nextMode = String(mode || "mono").toLowerCase() === "stereo" ? "stereo" : "mono";
        selected.channelMode = nextMode;
        selected.channelCount = nextMode === "stereo" ? 2 : 1;
        selected.stereoWidth = nextMode === "stereo" ? Math.max(.85, Number(selected.stereoWidth || 1)) : 0;
        selected.forceMono = nextMode === "mono";
        addEdit(`${String(selected.name || selected.fileName || "clip")} -> ${nextMode}.`);
        writeState(`channel_mode_${nextMode}`);
        draw();
    };
    const showContextMenu = (title, x, y, actions) => {
        contextMenuEl.innerHTML = "";
        const titleEl = document.createElement("div");
        titleEl.className = "iamccs-audio-context-menu-title";
        titleEl.textContent = title;
        contextMenuEl.appendChild(titleEl);
        actions.filter(Boolean).forEach((action) => {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.textContent = action.label;
            if (action.klass) btn.className = action.klass;
            btn.onclick = () => {
                closeContextMenu();
                action.run();
            };
            contextMenuEl.appendChild(btn);
        });
        contextMenuEl.style.display = "grid";
        const vw = window.innerWidth || 0;
        const vh = window.innerHeight || 0;
        const rect = contextMenuEl.getBoundingClientRect();
        contextMenuEl.style.left = `${Math.max(8, Math.min(x, vw - rect.width - 8))}px`;
        contextMenuEl.style.top = `${Math.max(8, Math.min(y, vh - rect.height - 8))}px`;
    };
    const openTrackContextMenu = (track, event) => {
        event.preventDefault();
        event.stopPropagation();
        state.selectedMixer = { type: "track", track };
        const trackCount = Math.max(1, Number(state.audioTrackCount || 4));
        const actions = [
            {
                label: `Select A${track + 1}`,
                run: () => {
                    state.selectedMixer = { type: "track", track };
                    transport.helper = `Selected A${track + 1} device chain.`;
                    writeState("select_track", false);
                    draw();
                },
            },
            { label: `Color A${track + 1}`, run: () => openTrackColorPicker(track, event) },
        ];
        const selected = selectedClip();
        if (selected) {
            actions.push({ label: "Delete selected clip", run: () => deleteSelectedClip(), klass: "danger" });
            for (let laneIndex = 0; laneIndex < trackCount; laneIndex += 1) {
                if (laneIndex === Number(selected.track || 0)) continue;
                actions.push({ label: `Move selected clip to A${laneIndex + 1}`, run: () => moveSelectedClipToTrack(laneIndex) });
            }
        }
        showContextMenu(`Track A${track + 1}`, event.clientX, event.clientY, actions);
    };
    window.addEventListener("pointerdown", (event) => {
        if (!contextMenuEl.contains(event.target)) closeContextMenu();
    });
    window.addEventListener("blur", closeContextMenu);
    const multiGenerationClips = () => segments().filter((seg) => Boolean(seg.multiGenerationClip) || String(seg.timelineId || "").startsWith("T"));
    const multiGenerationMapText = () => {
        const clips = multiGenerationClips()
            .slice()
            .sort((a, b) => (Number(a.multiTakeIndex || 0) - Number(b.multiTakeIndex || 0)) || (Number(a.track || 0) - Number(b.track || 0)));
        if (!clips.length) return "MULTI inactive. Split source audio to create T1-A1, T2-A2, T3-A3 indexed lanes.";
        return clips
            .map((seg) => {
                const localStart = Number(seg.start || 0);
                const globalStart = Number(seg.sourceGlobalStart ?? seg.globalStart ?? localStart);
                const suffix = Math.abs(globalStart - localStart) > 0.5 ? ` src ${fmtTime(globalStart)}` : "";
                return `${String(seg.timelineId || `T${String(seg.multiTakeIndex || 1).padStart(2, "0")}`).replace(/^T0/, "T")} - A${Number(seg.track || 0) + 1} ${fmtTime(localStart)}+${fmtTime(Number(seg.length || 1))}${suffix}`;
            })
            .join(" | ");
    };
    const clearMultiGenerationClips = () => {
        const beforeSegments = segments();
        const source = state.multiGeneration?.sourceSegment && typeof state.multiGeneration.sourceSegment === "object"
            ? JSON.parse(JSON.stringify(state.multiGeneration.sourceSegment))
            : null;
        const sourceId = String(state.multiGeneration?.sourceSegmentId || source?.id || "");
        const removed = beforeSegments.filter((seg) => Boolean(seg.multiGenerationClip) || String(seg.timelineId || "").startsWith("T")).length;
        state.audioSegments = beforeSegments.filter((seg) => !(Boolean(seg.multiGenerationClip) || String(seg.timelineId || "").startsWith("T")));
        if (source && !state.audioSegments.some((seg) => String(seg.id || "") === sourceId)) {
            state.audioSegments.push(source);
        }
        state.multiGeneration = { enabled: false, removed, clearedAt: new Date().toISOString() };
        addEdit(`Cleared ${removed} MULTI take clip(s).${source ? " Source restored." : ""}`);
        writeState("multi_clear");
        draw();
    };
    const applyMultiGenerationSplit = (options = {}) => {
        const chunkSeconds = Math.max(1, Number(options.chunkSeconds || 20));
        const chunkFrames = Math.max(1, secondsToFrames(chunkSeconds));
        const sourceTrack = Math.max(0, Number(options.sourceTrack || 0));
        const destinationStartTrack = Math.max(0, Number(options.destinationStartTrack || 0));
        const takeCountRaw = String(options.takeCount || "auto");
        const splitStartMode = String(options.splitStartMode || state.multiGeneration?.splitStartMode || "all_zero") === "global_source"
            ? "global_source"
            : "all_zero";
        const localizeStarts = splitStartMode !== "global_source";
        const sourceCandidates = segments()
            .filter((seg) => !seg.multiGenerationClip && !String(seg.timelineId || "").startsWith("T"))
            .filter((seg) => Number(seg.track || 0) === sourceTrack)
            .filter((seg) => hasMedia(seg))
            .sort((a, b) => Number(b.length || b.audioDurationFrames || 0) - Number(a.length || a.audioDurationFrames || 0));
        const source = sourceCandidates[0];
        if (!source) {
            addEdit(`MULTI split failed: no source audio on A${sourceTrack + 1}.`);
            draw();
            return;
        }
        const sourceStart = Math.max(0, Math.round(Number(source.start || 0)));
        const sourceLength = Math.max(1, Math.round(Number(source.length || source.audioDurationFrames || 1)));
        const sourceTrim = Math.max(0, Math.round(Number(source.trimStart || 0)));
        const autoTakes = Math.max(1, Math.ceil(sourceLength / chunkFrames));
        const takeCount = takeCountRaw === "auto"
            ? autoTakes
            : Math.max(1, Math.min(autoTakes, Math.round(Number(takeCountRaw) || autoTakes)));
        const sourceId = String(source.id || "");
        const baseSegments = segments().filter((seg) => {
            if (Boolean(seg.multiGenerationClip) || String(seg.timelineId || "").startsWith("T")) return false;
            return String(seg.id || "") !== sourceId;
        });
        const created = [];
        for (let index = 0; index < takeCount; index += 1) {
            const offset = index * chunkFrames;
            if (offset >= sourceLength) break;
            const length = Math.max(1, Math.min(chunkFrames, sourceLength - offset));
            const takeIndex = index + 1;
            const timelineId = `T${String(takeIndex).padStart(2, "0")}`;
            const clone = {
                ...JSON.parse(JSON.stringify(source)),
                id: newId(`multi_${timelineId.toLowerCase()}`),
                name: `${String(source.name || source.fileName || source.audioFile || "audio").slice(0, 42)} ${timelineId}`,
                track: destinationStartTrack + index,
                start: localizeStarts ? 0 : sourceStart + offset,
                length,
                trimStart: sourceTrim + offset,
                audioDurationFrames: Math.max(sourceTrim + offset + length, Number(source.audioDurationFrames || sourceLength)),
                mute: false,
                solo: false,
                multiGenerationClip: true,
                multiTakeIndex: takeIndex,
                timelineId,
                sourceSegmentId: sourceId,
                sourceTrack,
                sourceGlobalStart: sourceStart + offset,
                sourceGlobalEnd: sourceStart + offset + length,
                globalStart: sourceStart + offset,
                globalEnd: sourceStart + offset + length,
                localStart: 0,
                splitStartMode,
                chunkFrames,
                chunkSeconds,
            };
            created.push(clone);
        }
        state.audioSegments = baseSegments.concat(created);
        state.audioTrackCount = Math.max(Number(state.audioTrackCount || 1), destinationStartTrack + created.length);
        state.audioBusMode = "only_first";
        state.onlyFirstTrack = true;
        state.multiGeneration = {
            enabled: true,
            template: `${chunkSeconds}s`,
            chunkSeconds,
            chunkFrames,
            sourceTrack,
            sourceSegment: JSON.parse(JSON.stringify(source)),
            sourceSegmentId: sourceId,
            destinationStartTrack,
            splitStartMode,
            takeCount: created.length,
            timelineIds: created.map((seg) => seg.timelineId),
            laneIndex: created.map((seg) => ({
                timeline_id: seg.timelineId,
                take_index: seg.multiTakeIndex,
                track_index: seg.track,
                track_name: `A${Number(seg.track || 0) + 1}`,
                audio_chunk_id: seg.id,
                source_segment_id: sourceId,
                start: seg.start,
                local_start: seg.start,
                source_global_start: seg.sourceGlobalStart,
                source_global_end: seg.sourceGlobalEnd,
                length: seg.length,
            })),
            updatedAt: new Date().toISOString(),
        };
        addEdit(`MULTI split created ${created.length} take lane(s): ${multiGenerationMapText()}.`);
        writeState("multi_split");
        draw();
    };
    const makeMultiSelect = (labelText, values, value, onChange) => {
        const wrap = document.createElement("label");
        wrap.textContent = labelText;
        const select = document.createElement("select");
        for (const [optionValue, optionLabel] of values) {
            const option = document.createElement("option");
            option.value = String(optionValue);
            option.textContent = optionLabel;
            select.appendChild(option);
        }
        select.value = String(value);
        select.onchange = () => onChange(select.value);
        wrap.appendChild(select);
        return { wrap, select };
    };
    const appendMultiGenerationStrip = (parent) => {
        const multi = state.multiGeneration && typeof state.multiGeneration === "object" ? state.multiGeneration : {};
        const strip = document.createElement("div");
        strip.className = "iamccs-audio-board-multi";
        const title = document.createElement("div");
        title.className = "iamccs-audio-board-multi-title";
        title.innerHTML = `<span>MULTIGENERATION</span><small>audio chunk lanes</small>`;
        let chunkSeconds = Number(multi.chunkSeconds || 20);
        let takeCount = "auto";
        let sourceTrack = Number(multi.sourceTrack || 0);
        let destinationStartTrack = Number(multi.destinationStartTrack || 0);
        let splitStartMode = String(multi.splitStartMode || "all_zero") === "global_source" ? "global_source" : "all_zero";
        const chunk = makeMultiSelect("Template", [["10", "10 sec"], ["15", "15 sec"], ["20", "20 sec"], ["25", "25 sec"], ["custom", "custom"]], [10, 15, 20, 25].includes(chunkSeconds) ? String(chunkSeconds) : "custom", (value) => {
            if (value !== "custom") customInput.value = value;
            chunkSeconds = Math.max(1, Number(customInput.value || value || 20));
        });
        const customLabel = document.createElement("label");
        customLabel.textContent = "Seconds";
        const customInput = document.createElement("input");
        customInput.type = "number";
        customInput.min = "1";
        customInput.max = "300";
        customInput.step = "0.25";
        customInput.value = String(chunkSeconds);
        customInput.onchange = () => { chunkSeconds = Math.max(1, Number(customInput.value || 20)); };
        customLabel.appendChild(customInput);
        const takes = makeMultiSelect("Takes", [["auto", "auto"], ["2", "2 takes"], ["3", "3 takes"], ["4", "4 takes"], ["5", "5 takes"]], takeCount, (value) => { takeCount = value; });
        const source = makeMultiSelect("Source", Array.from({ length: Math.max(1, Number(state.audioTrackCount || 1)) }, (_, i) => [String(i), `A${i + 1}`]), sourceTrack, (value) => { sourceTrack = Number(value || 0); });
        const dest = makeMultiSelect("T1 lane", Array.from({ length: Math.max(1, Number(state.audioTrackCount || 1)) }, (_, i) => [String(i), `A${i + 1}`]), destinationStartTrack, (value) => { destinationStartTrack = Number(value || 0); });
        const startMode = makeMultiSelect("Start", [["all_zero", "all T @ 0"], ["global_source", "keep source time"]], splitStartMode, (value) => { splitStartMode = String(value || "all_zero"); });
        const split = document.createElement("button");
        split.type = "button";
        split.className = "is-primary";
        split.textContent = "Split To T Lanes";
        split.onclick = () => applyMultiGenerationSplit({ chunkSeconds, sourceTrack, destinationStartTrack, takeCount, splitStartMode });
        const clear = document.createElement("button");
        clear.type = "button";
        clear.className = "danger";
        clear.textContent = "Clear MULTI";
        clear.onclick = clearMultiGenerationClips;
        const map = document.createElement("div");
        map.className = "iamccs-audio-board-multi-map";
        map.textContent = multiGenerationMapText();
        strip.append(title, chunk.wrap, customLabel, takes.wrap, source.wrap, dest.wrap, startMode.wrap, split, clear, map);
        parent.appendChild(strip);
    };
    const syncInlineFxControls = (fx) => {
        const syncers = Array.isArray(fx?._iamccsInlineControlSync) ? fx._iamccsInlineControlSync : [];
        syncers.forEach((syncer) => {
            try { syncer?.(); } catch {}
        });
    };
    const registerInlineFxControl = (fx, syncer) => {
        if (!fx) return;
        fx._iamccsInlineControlSync = Array.isArray(fx._iamccsInlineControlSync) ? fx._iamccsInlineControlSync : [];
        fx._iamccsInlineControlSync.push(syncer);
    };
    const mirrorDynamicsFxToState = (fx, target) => {
        if (!fx || target?.type !== "master") return;
        if (fx.type === "compressor") state.masterBus.compressor = Math.max(.01, Math.min(1, (Math.abs(Number(fx.params?.threshold || -18)) / 60) + .15));
        if (fx.type === "limiter") {
            state.masterBus.limiter = true;
            state.masterBus.ceilingDb = Number(fx.params?.ceiling ?? state.masterBus.ceilingDb ?? -1);
        }
    };
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    const paintInlineCanvasDeferred = (canvas, kind, seg, staticPeak = 0) => {
        if (!canvas) return;
        canvas._iamccsInlinePaintState = { kind, seg, staticPeak };
        const run = () => {
            const live = canvas._iamccsInlinePaintState || { kind, seg, staticPeak };
            paintInlineCanvas(canvas, live.kind, live.seg, live.staticPeak);
        };
        requestAnimationFrame(run);
        window.setTimeout(run, 48);
        if (!canvas._iamccsInlineResizeObserver && typeof ResizeObserver !== "undefined") {
            canvas._iamccsInlineResizeObserver = new ResizeObserver(() => requestAnimationFrame(run));
            try { canvas._iamccsInlineResizeObserver.observe(canvas); } catch {}
        }
    };
    const paintInlineCanvas = (canvas, kind, seg, staticPeak = 0) => {
        const rect = canvas.getBoundingClientRect();
        const w = Math.max(180, Math.round(rect.width || 260));
        const h = Math.max(70, Math.round(rect.height || 90));
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = "#061012";
        ctx.fillRect(0, 0, w, h);
        ctx.strokeStyle = "rgba(255,255,255,.08)";
        ctx.lineWidth = 1;
        for (let x = 0; x <= w; x += Math.max(24, w / 8)) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, h);
            ctx.stroke();
        }
        for (let y = 0; y <= h; y += Math.max(18, h / 4)) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }
        if (kind === "wave") {
            const peaks = Array.isArray(seg?.waveformPeaks) ? seg.waveformPeaks : [];
            const mid = h * .5;
            const amp = h * .42;
            ctx.strokeStyle = "rgba(244,212,158,.92)";
            ctx.fillStyle = "rgba(85,199,185,.34)";
            ctx.beginPath();
            if (!peaks.length) {
                ctx.moveTo(0, mid);
                ctx.lineTo(w, mid);
            } else {
                for (let i = 0; i < peaks.length; i += 1) {
                    const item = peaks[i] || {};
                    const x = (i / Math.max(1, peaks.length - 1)) * w;
                    const max = Math.max(0, Math.abs(Number(item.max ?? item) || 0));
                    const min = Math.max(0, Math.abs(Number(item.min ?? item) || 0));
                    const top = mid - Math.max(max, min) * amp;
                    if (i === 0) ctx.moveTo(x, top);
                    else ctx.lineTo(x, top);
                }
                for (let i = peaks.length - 1; i >= 0; i -= 1) {
                    const item = peaks[i] || {};
                    const x = (i / Math.max(1, peaks.length - 1)) * w;
                    const max = Math.max(0, Math.abs(Number(item.max ?? item) || 0));
                    const min = Math.max(0, Math.abs(Number(item.min ?? item) || 0));
                    ctx.lineTo(x, mid + Math.max(max, min) * amp);
                }
            }
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
            ctx.strokeStyle = "rgba(255,255,255,.18)";
            ctx.beginPath();
            ctx.moveTo(0, mid);
            ctx.lineTo(w, mid);
            ctx.stroke();
            return;
        }
        if (kind === "eq") {
            const eqFx = canvas._iamccsFx?.type === "eq" ? canvas._iamccsFx : chainForTarget().find((fx) => fx.enabled !== false && fx.type === "eq");
            normalizeEffect(eqFx);
            const low = Number(eqFx?.params?.low ?? seg?.eqLowDb ?? 0);
            const midDb = Number(eqFx?.params?.mid ?? seg?.eqMidDb ?? 0);
            const high = Number(eqFx?.params?.high ?? seg?.eqHighDb ?? 0);
            const hpf = eqFx?.params?.lowCut ? Number(eqFx.params.lowCutFreq || 80) : Number(seg?.hpfHz || 0);
            const lpf = eqFx?.params?.highCut ? Number(eqFx.params.highCutFreq || 12000) : Number(seg?.lpfHz || 22000);
            const lowCutLevel = Math.max(-48, Math.min(0, Number(eqFx?.params?.lowCutLevel ?? -30)));
            const highCutLevel = Math.max(-48, Math.min(0, Number(eqFx?.params?.highCutLevel ?? -30)));
            const center = h * .5;
            const clampDb = (db) => Math.max(-24, Math.min(24, Number(db || 0)));
            const yFromDb = (db) => center - (clampDb(db) / 48) * h;
            const yCutFromDb = (db) => center - (Math.max(-48, Math.min(0, Number(db || 0))) / 72) * h;
            const freqToX = (hz) => eqVisualX(hz, w);
            const drawSmooth = (points, stroke, width = 2.2) => {
                ctx.strokeStyle = stroke;
                ctx.lineWidth = width;
                ctx.beginPath();
                points.forEach((p, i) => {
                    if (i === 0) ctx.moveTo(p[0], p[1]);
                    else {
                        const p0 = points[Math.max(0, i - 2)];
                        const p1 = points[i - 1];
                        const p2 = p;
                        const p3 = points[Math.min(points.length - 1, i + 1)];
                        const cp1x = p1[0] + (p2[0] - p0[0]) / 6;
                        const cp1y = p1[1] + (p2[1] - p0[1]) / 6;
                        const cp2x = p2[0] - (p3[0] - p1[0]) / 6;
                        const cp2y = p2[1] - (p3[1] - p1[1]) / 6;
                        ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, p2[0], p2[1]);
                    }
                });
                ctx.stroke();
            };
            const eqFreq = eqBandFrequencies(eqFx || seg || {});
            [-24, -12, 0, 12, 24].forEach((db) => {
                const y = yFromDb(db);
                ctx.strokeStyle = db === 0 ? "rgba(255,255,255,.14)" : "rgba(255,255,255,.06)";
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(w, y);
                ctx.stroke();
            });
            ctx.fillStyle = "rgba(255,255,255,.03)";
            ctx.fillRect(0, 0, w / 3, h);
            ctx.fillStyle = "rgba(255,255,255,.02)";
            ctx.fillRect(w / 3, 0, w / 3, h);
            ctx.fillStyle = "rgba(255,255,255,.03)";
            ctx.fillRect((w / 3) * 2, 0, w / 3, h);
            [60, 250, 1000, 4000, 12000].forEach((hz) => {
                const x = freqToX(hz);
                ctx.strokeStyle = (hz === 250 || hz === 4000) ? "rgba(255,255,255,.14)" : "rgba(255,255,255,.08)";
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, h);
                ctx.stroke();
            });
            const lowX = freqToX(eqFreq.low);
            const midX = freqToX(eqFreq.mid);
            const highX = freqToX(eqFreq.high);
            const lowCutX = freqToX(hpf || 20);
            const highCutX = freqToX(lpf || 22000);
            if (hpf > 20) {
                ctx.fillStyle = "rgba(214,91,66,.11)";
                ctx.fillRect(0, 0, Math.max(0, lowCutX), h);
            }
            if (lpf < 22000) {
                ctx.fillStyle = "rgba(214,91,66,.09)";
                ctx.fillRect(highCutX, 0, Math.max(0, w - highCutX), h);
            }
            const mainPoints = [
                [0, hpf > 20 ? yCutFromDb(lowCutLevel) : center],
                ...(hpf > 20 ? [[lowCutX, center]] : []),
                [lowX, yFromDb(low)],
                [midX, yFromDb(midDb)],
                [highX, yFromDb(high)],
                ...(lpf < 22000 ? [[highCutX, center]] : []),
                [w, lpf < 22000 ? yCutFromDb(highCutLevel) : center],
            ].sort((a, b) => a[0] - b[0]);
            ctx.fillStyle = "rgba(85,199,185,.08)";
            ctx.beginPath();
            mainPoints.forEach((p, i) => {
                if (i === 0) ctx.moveTo(p[0], p[1]);
                else ctx.lineTo(p[0], p[1]);
            });
            ctx.lineTo(w, center);
            ctx.lineTo(0, center);
            ctx.closePath();
            ctx.fill();
            drawSmooth(mainPoints, "#f5d08e", 2.4);
            if (hpf > 20) {
                ctx.fillStyle = "#ffac72";
                ctx.beginPath();
                ctx.arc(lowCutX, center, 5, 0, Math.PI * 2);
                ctx.fill();
            }
            if (lpf < 22000) {
                ctx.fillStyle = "#ffac72";
                ctx.beginPath();
                ctx.arc(highCutX, center, 5, 0, Math.PI * 2);
                ctx.fill();
            }
            const points = [
                [lowX / Math.max(1, w), low, "#d6a75e"],
                [midX / Math.max(1, w), midDb, "#55c7b9"],
                [highX / Math.max(1, w), high, "#d08963"],
            ];
            for (const [tx, db, color] of points) {
                const x = tx * w;
                const y = yFromDb(db);
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(x, y, 6, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = "rgba(0,0,0,.72)";
                ctx.stroke();
            }
            ctx.fillStyle = "rgba(255,226,168,.72)";
            ctx.font = "900 9px ui-monospace, Consolas, monospace";
            ctx.fillText("LOW", 8, 12);
            ctx.fillText("MID", Math.max(8, w * .5 - 10), 12);
            ctx.fillText("HIGH", Math.max(8, w - 34), 12);
            ctx.fillStyle = "#ffc67f";
            ctx.font = "900 9px ui-monospace, Consolas, monospace";
            if (hpf > 20) ctx.fillText(`LC ${Math.round(hpf)}Hz ${Math.round(lowCutLevel)}dB`, 8, h - 9);
            if (lpf < 22000) ctx.fillText(`HC ${(lpf / 1000).toFixed(1)}k ${Math.round(highCutLevel)}dB`, Math.max(8, w - 116), h - 9);
            return;
        }
        const currentFx = canvas._iamccsFx || null;
        const activeFx = currentFx || chainForTarget().find((fx) => fx.enabled !== false && (fx.type === "compressor" || fx.type === "limiter"));
        if (["reverb", "delay", "stereo", "utility", "saturator", "chorus", "tape"].includes(String(activeFx?.type || ""))) {
            normalizeEffect(activeFx);
            const meter = Math.max(0, Math.min(1, staticPeak));
            const drawLabel = (text, x = 8, y = 14, color = "#ffe2a8") => {
                ctx.fillStyle = color;
                ctx.font = "900 10px ui-monospace, Consolas, monospace";
                ctx.fillText(text, x, y);
            };
            if (activeFx.type === "reverb") {
                const size = Math.max(0, Math.min(1, Number(activeFx.params?.size ?? .45)));
                const decay = Math.max(.1, Math.min(8, Number(activeFx.params?.decay ?? 1.8)));
                const mix = Math.max(0, Math.min(1, Number(activeFx.params?.mix ?? .18)));
                const damp = Math.max(0, Math.min(1, Number(activeFx.params?.damp ?? .35)));
                ctx.strokeStyle = "rgba(85,199,185,.95)";
                ctx.lineWidth = 2;
                ctx.beginPath();
                for (let x = 0; x < w; x += 1) {
                    const t = x / Math.max(1, w - 1);
                    const env = Math.exp(-t * (1.2 + decay * 1.6));
                    const ripple = Math.sin((t * (12 + size * 22)) * Math.PI) * env * (1 - damp * .7);
                    const y = h * .22 + ripple * h * (.18 + mix * .24) + env * h * (.18 + size * .14);
                    if (x === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }
                ctx.stroke();
                ctx.fillStyle = "rgba(85,199,185,.16)";
                ctx.fillRect(8, h - 18, Math.max(8, mix * (w - 16)), 8);
                ctx.fillStyle = "rgba(240,184,87,.26)";
                ctx.fillRect(8, h - 30, Math.max(8, size * (w - 16)), 7);
                drawLabel(`REV size ${size.toFixed(2)} decay ${decay.toFixed(2)} mix ${mix.toFixed(2)}`);
                return;
            }
            if (activeFx.type === "delay") {
                const time = Math.max(.03, Math.min(1.5, Number(activeFx.params?.time ?? .18)));
                const feedback = Math.max(0, Math.min(.9, Number(activeFx.params?.feedback ?? .28)));
                const mix = Math.max(0, Math.min(1, Number(activeFx.params?.mix ?? .2)));
                const taps = 6;
                for (let i = 0; i < taps; i += 1) {
                    const t = Math.min(1, ((i + 1) / taps) * (time / 1.5) * 1.9);
                    const x = 12 + t * (w - 24);
                    const height = Math.max(8, (1 - i / taps) * (feedback * .9 + .2) * h * .56);
                    ctx.fillStyle = `rgba(240,184,87,${Math.max(.18, .72 - i * .1)})`;
                    ctx.fillRect(x - 4, h * .62 - height, 8, height);
                }
                ctx.fillStyle = "rgba(85,199,185,.28)";
                ctx.fillRect(8, h - 16, Math.max(8, mix * (w - 16)), 8);
                drawLabel(`DLY ${time.toFixed(2)}s fb ${feedback.toFixed(2)} mix ${mix.toFixed(2)}`);
                return;
            }
            if (activeFx.type === "stereo" || activeFx.type === "utility") {
                const width = Math.max(0, Math.min(2, Number(activeFx.params?.width ?? 1)));
                const pan = Math.max(-1, Math.min(1, Number(activeFx.params?.pan ?? 0)));
                const mono = Number(activeFx.params?.mono ?? 0) >= .5;
                const spread = mono ? 0 : Math.min(.44, width / 2.4);
                const centerX = w * (.5 + pan * .24);
                ctx.strokeStyle = "rgba(255,255,255,.14)";
                ctx.beginPath();
                ctx.moveTo(w * .5, 0);
                ctx.lineTo(w * .5, h);
                ctx.stroke();
                ctx.fillStyle = "rgba(85,199,185,.24)";
                ctx.fillRect(centerX - spread * w, h * .2, spread * w * 2, h * .56);
                ctx.strokeStyle = mono ? "#f0b857" : "#55c7b9";
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(centerX - spread * w, h * .28);
                ctx.lineTo(centerX + spread * w, h * .28);
                ctx.moveTo(centerX - spread * w, h * .72);
                ctx.lineTo(centerX + spread * w, h * .72);
                ctx.stroke();
                drawLabel(`${String(activeFx.type).toUpperCase()} width ${width.toFixed(2)} pan ${pan.toFixed(2)}${mono ? " mono" : " stereo"}`);
                ctx.fillStyle = mono ? "rgba(240,184,87,.82)" : "rgba(85,199,185,.82)";
                ctx.fillRect(8, h - 14, Math.max(8, meter * (w - 16)), 6);
                return;
            }
            const amount = Math.max(0, Math.min(1, Number(activeFx.params?.mix ?? activeFx.params?.sat ?? activeFx.params?.color ?? .35)));
            ctx.strokeStyle = "rgba(240,184,87,.86)";
            ctx.lineWidth = 2;
            ctx.beginPath();
            for (let x = 0; x < w; x += 1) {
                const n = (x / Math.max(1, w - 1)) * 2 - 1;
                const curved = Math.tanh(n * (1 + amount * 5));
                const y = h * .5 - curved * h * .28;
                if (x === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
            drawLabel(`${String(activeFx.type).toUpperCase()} amt ${amount.toFixed(2)}`);
            return;
        }
        const isLimiter = activeFx?.type === "limiter";
        const compFx = activeFx?.type === "compressor" ? activeFx : chainForTarget().find((fx) => fx.enabled !== false && fx.type === "compressor");
        const limFx = activeFx?.type === "limiter" ? activeFx : chainForTarget().find((fx) => fx.enabled !== false && fx.type === "limiter");
        const threshold = Math.max(-60, Math.min(0, Number(compFx?.params?.threshold ?? -18)));
        const ratio = Math.max(1, Math.min(20, Number(compFx?.params?.ratio ?? 4)));
        const ceiling = Math.max(-12, Math.min(0, Number(limFx?.params?.ceiling ?? -1)));
        const inputDb = Math.max(-12, Math.min(12, Number(limFx?.params?.input ?? 0)));
        const outputDb = Math.max(-12, Math.min(6, Number(limFx?.params?.output ?? 0)));
        const dbToX = (db, minDb, maxDb) => ((Math.max(minDb, Math.min(maxDb, db)) - minDb) / Math.max(.001, maxDb - minDb)) * w;
        const compY = (db) => h - ((Math.max(-60, Math.min(0, db)) + 60) / 60) * h;
        const limY = (db) => h - ((Math.max(-12, Math.min(6, db)) + 12) / 18) * h;
        ctx.strokeStyle = "rgba(255,226,168,.18)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, h - 18);
        ctx.lineTo(w, h - 18);
        ctx.moveTo(28, 0);
        ctx.lineTo(28, h);
        ctx.stroke();
        ctx.fillStyle = "rgba(255,226,168,.76)";
        ctx.font = "900 9px ui-monospace, Consolas, monospace";
        if (isLimiter) {
            ctx.fillText("IN -12dB", 34, h - 7);
            ctx.fillText("0dB", w - 35, h - 7);
            ctx.fillText("OUT", 6, 13);
            ctx.strokeStyle = "rgba(143,208,204,.86)";
            ctx.lineWidth = 2.2;
            ctx.beginPath();
            for (let x = 0; x <= w; x += 1) {
                const inDb = -12 + (x / Math.max(1, w)) * 12 + inputDb;
                const limited = Math.min(ceiling, inDb);
                const outDb = Math.max(-12, Math.min(6, limited + outputDb));
                const y = limY(outDb);
                if (x === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
            const px = dbToX(ceiling, -12, 0);
            const py = limY(outputDb);
            ctx.strokeStyle = "rgba(245,208,142,.38)";
            ctx.beginPath();
            ctx.moveTo(px, 0);
            ctx.lineTo(px, h);
            ctx.moveTo(0, py);
            ctx.lineTo(w, py);
            ctx.stroke();
            ctx.fillStyle = "#f5d08e";
            ctx.beginPath();
            ctx.arc(px, py, 9, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = "rgba(0,0,0,.75)";
            ctx.stroke();
            ctx.fillStyle = "#fff2cf";
            ctx.fillText(`CEIL ${ceiling.toFixed(1)} OUT ${outputDb.toFixed(1)}`, Math.min(w - 142, px + 12), Math.max(14, py - 10));
        } else {
            ctx.fillText("-60dB", 34, h - 7);
            ctx.fillText("0dB", w - 35, h - 7);
            ctx.fillText("OUT", 6, 13);
            ctx.strokeStyle = "rgba(255,255,255,.18)";
            ctx.beginPath();
            ctx.moveTo(28, h - 18);
            ctx.lineTo(w, 8);
            ctx.stroke();
            ctx.strokeStyle = "rgba(143,208,204,.86)";
            ctx.lineWidth = 2.4;
            ctx.beginPath();
            for (let x = 28; x <= w; x += 1) {
                const inDb = -60 + ((x - 28) / Math.max(1, w - 28)) * 60;
                const over = Math.max(0, inDb - threshold);
                const outDb = inDb <= threshold ? inDb : threshold + over / ratio;
                const y = compY(outDb);
                if (x === 28) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
            const px = 28 + ((threshold + 60) / 60) * Math.max(1, w - 28);
            const py = h - ((ratio - 1) / 19) * (h - 24) - 12;
            ctx.strokeStyle = "rgba(245,208,142,.38)";
            ctx.beginPath();
            ctx.moveTo(px, 0);
            ctx.lineTo(px, h);
            ctx.moveTo(0, py);
            ctx.lineTo(w, py);
            ctx.stroke();
            ctx.fillStyle = "#d6a75e";
            ctx.beginPath();
            ctx.arc(px, py, 9, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = "rgba(0,0,0,.75)";
            ctx.stroke();
            ctx.fillStyle = "#fff2cf";
            ctx.fillText(`THR ${threshold.toFixed(0)}dB  R ${ratio.toFixed(1)}:1`, Math.min(w - 150, px + 12), Math.max(14, py - 10));
        }
        const peak = Math.max(0, Math.min(1, staticPeak));
        ctx.fillStyle = "rgba(85,199,185,.86)";
        ctx.fillRect(8, h - Math.max(2, peak * h), 8, Math.max(2, peak * h));
    };
    const appendInlineEfxPreview = (parent, selected, staticPeak) => {
        const efx = document.createElement("div");
        efx.className = "iamccs-inline-efx";
        const target = selectedMixer();
        const targetName = target.type === "track" ? `A${target.track + 1}` : "MASTER OUT";
        const chain = chainForTarget();
        const deviceColors = {
            eq: ["#6a5532", "#29251c", "#0b0d0c"],
            compressor: ["#5a3f2c", "#2d241b", "#0b0d0c"],
            limiter: ["#673631", "#2c1e1b", "#0b0d0c"],
            gate: ["#394d3f", "#1d2922", "#0b0d0c"],
            reverb: ["#35445f", "#1b2232", "#090b12"],
            delay: ["#5b4a2d", "#2a2418", "#0d0b08"],
            saturator: ["#6b4026", "#322116", "#0d0907"],
            utility: ["#3e514d", "#1c2927", "#071010"],
            stereo: ["#423c60", "#211f32", "#0a0912"],
            deesser: ["#4b5a35", "#242d1c", "#090d08"],
            transient: ["#614c34", "#2c251a", "#0c0a08"],
            tape: ["#57412b", "#2b2118", "#0d0906"],
            chorus: ["#34565d", "#1a2b30", "#071012"],
        };
        efx.innerHTML = `<div class="iamccs-inline-efx-head"><span>DEVICE RACK / AUDIO CONTROL EFX</span><span>${targetName}</span></div>`;
        const grid = document.createElement("div");
        grid.className = "iamccs-inline-efx-grid";
        grid.onscroll = () => { root._iamccsFxRackScrollLeft = Number(grid.scrollLeft || 0); };
        const chainEl = document.createElement("div");
        chainEl.className = "iamccs-device-chain";
        const previewForFx = (type) => {
            if (type === "eq") return ["eq", "EQ 3-Point Editor"];
            if (["compressor", "limiter", "gate", "deesser", "transient", "reverb", "delay", "stereo", "utility", "saturator", "chorus", "tape"].includes(type)) return ["bus", `${String(type || "FX").toUpperCase()} Editor`];
            return ["placeholder", `${String(type || "FX").toUpperCase()} Preview`];
        };
        if (!chain.length) {
            const module = document.createElement("div");
            module.className = "iamccs-device-module";
            const empty = document.createElement("div");
            empty.className = "iamccs-audio-device";
            empty.innerHTML = `<div class="iamccs-device-head"><span>NO DEVICE</span></div><div style="color:#b3a37e;font:800 9px/1.35 ui-monospace,SFMono-Regular,Consolas,monospace;">Use the ${target.type === "track" ? "track" : "master"} FX dropdown to insert a device.</div>`;
            const ph = document.createElement("div");
            ph.className = "iamccs-inline-efx-panel";
            ph.innerHTML = `<label>Realtime Preview</label><div class="iamccs-efx-placeholder">Insert an effect to show its control preview here.</div>`;
            module.append(empty, ph);
            chainEl.appendChild(module);
        }
        chain.forEach((fx, index) => {
            normalizeEffect(fx);
            fx._iamccsInlineControlSync = [];
            const module = document.createElement("div");
            module.className = "iamccs-device-module";
            const repaintModuleCanvas = () => {
                module.querySelectorAll(".iamccs-inline-efx-canvas").forEach((currentCanvas) => {
                    paintInlineCanvasDeferred(currentCanvas, currentCanvas.dataset.kind || "", selected, staticPeak);
                });
            };
            const device = document.createElement("div");
            device.className = `iamccs-audio-device iamccs-device-${String(fx.type || "device").replace(/[^a-z0-9_-]/gi, "")}`;
            const palette = deviceColors[fx.type] || ["#3a3328", "#171714", "#0b0d0c"];
            device.style.setProperty("--device-hi", palette[0]);
            device.style.setProperty("--device-mid", palette[1]);
            device.style.setProperty("--device-lo", palette[2]);
            const name = EFFECT_CHOICES.find(([value]) => value === fx.type)?.[1] || String(fx.type || "FX");
            const power = document.createElement("button");
            power.type = "button";
            power.className = `iamccs-device-power${fx.enabled === false ? " is-off" : ""}`;
            power.title = "Enable / bypass";
            power.onpointerdown = (event) => event.stopPropagation();
            power.onclick = (event) => {
                event.stopPropagation();
                fx.enabled = fx.enabled === false;
                addEdit(`${targetName} ${name} ${fx.enabled ? "enabled" : "bypassed"}.`);
                writeState("toggle_effect");
                restartPlaybackIfNeeded(`toggle_${fx.type}`);
                draw();
            };
            const head = document.createElement("div");
            head.className = "iamccs-device-head";
            const deviceTitle = document.createElement("span");
            deviceTitle.textContent = `${index + 1}. ${name}`;
            const headActions = document.createElement("div");
            headActions.className = "iamccs-device-head-actions";
            const remove = document.createElement("button");
            remove.type = "button";
            remove.className = "iamccs-device-remove";
            remove.title = "Remove device";
            remove.textContent = "x";
            remove.onpointerdown = (event) => event.stopPropagation();
            remove.onclick = (event) => {
                event.stopPropagation();
                removeEffectFromChain(fx, target);
            };
            const lamp = document.createElement("span");
            lamp.className = `iamccs-device-lamp${fx.enabled === false ? " is-off" : ""}`;
            lamp.title = fx.enabled === false ? "Device bypassed" : "Device active";
            headActions.append(lamp, power, remove);
            head.append(deviceTitle, headActions);
            const knobs = document.createElement("div");
            knobs.className = "iamccs-device-knobs";
            effectParamSpecs(fx.type).forEach(([key, knobName, min, max, step, fallback, unit]) => {
                if (fx.params[key] == null) fx.params[key] = fallback;
                const knob = document.createElement("div");
                knob.className = "iamccs-device-knob";
                const dial = document.createElement("i");
                const value = Math.max(Number(min), Math.min(Number(max), Number(fx.params[key] ?? fallback)));
                const amount = (value - Number(min)) / Math.max(.0001, Number(max) - Number(min));
                dial.style.setProperty("--knob-angle", `${-135 + amount * 270}deg`);
                const text = document.createElement("span");
                text.textContent = knobName;
                const valueText = document.createElement("em");
                valueText.textContent = `${Number(value.toFixed(2))}${unit || ""}`;
                const range = document.createElement("input");
                range.type = "range";
                range.min = String(min);
                range.max = String(max);
                range.step = String(step);
                range.value = String(value);
                const syncControl = () => {
                    const liveValue = Math.max(Number(min), Math.min(Number(max), Number(fx.params[key] ?? fallback)));
                    range.value = String(liveValue);
                    const t = (liveValue - Number(min)) / Math.max(.0001, Number(max) - Number(min));
                    dial.style.setProperty("--knob-angle", `${-135 + t * 270}deg`);
                    valueText.textContent = `${Number(liveValue.toFixed(2))}${unit || ""}`;
                };
                registerInlineFxControl(fx, syncControl);
                const applyParam = (nextValue, commit = false) => {
                    const next = Math.max(Number(min), Math.min(Number(max), Number(nextValue ?? range.value ?? fallback)));
                    range.value = String(next);
                    fx.params[key] = next;
                    mirrorDynamicsFxToState(fx, target);
                    syncInlineFxControls(fx);
                    repaintModuleCanvas();
                    if (commit) {
                        addEdit(`${targetName} ${name}: ${knobName} ${valueText.textContent}.`);
                        writeState(`fx_${fx.type}_${key}`);
                        restartPlaybackIfNeeded(`fx_${fx.type}_${key}`);
                    } else {
                        scheduleSilentStateWrite(`fx_${fx.type}_${key}`);
                        scheduleShotboardSync(`fx_${fx.type}_${key}`);
                    }
                };
                range.oninput = (event) => {
                    event.stopPropagation();
                    applyParam(Number(range.value || fallback), false);
                };
                range.onchange = () => {
                    if (shotboardSyncTimer) {
                        window.clearTimeout(shotboardSyncTimer);
                        shotboardSyncTimer = 0;
                    }
                    applyParam(Number(range.value || fallback), true);
                };
                dial.onpointerdown = (event) => {
                    event.stopPropagation();
                    event.preventDefault();
                    const startX = event.clientX;
                    const startValue = Number(fx.params[key] ?? fallback);
                    try { dial.setPointerCapture?.(event.pointerId); } catch {}
                    const move = (moveEvent) => {
                        const mult = moveEvent.shiftKey ? 10 : moveEvent.altKey ? .1 : 1;
                        let next = startValue + Math.round((moveEvent.clientX - startX) / 5) * Number(step) * mult;
                        next = Math.max(Number(min), Math.min(Number(max), next));
                        range.value = String(Number(next.toFixed(4)));
                        applyParam(next, false);
                        moveEvent.preventDefault();
                    };
                    const up = (upEvent) => {
                        window.removeEventListener("pointermove", move);
                        window.removeEventListener("pointerup", up);
                        try { dial.releasePointerCapture?.(upEvent.pointerId); } catch {}
                        if (shotboardSyncTimer) {
                            window.clearTimeout(shotboardSyncTimer);
                            shotboardSyncTimer = 0;
                        }
                        applyParam(Number(range.value || fallback), true);
                    };
                    window.addEventListener("pointermove", move);
                    window.addEventListener("pointerup", up, { once: true });
                };
                knob.append(dial, text, valueText, range);
                knobs.appendChild(knob);
                syncControl();
            });
            const eqSwitches = document.createElement("div");
            eqSwitches.className = "iamccs-eq-switches";
            if (fx.type === "eq") {
                [["LOW CUT", "lowCut"], ["HIGH CUT", "highCut"]].forEach(([labelText, key]) => {
                    const btn = document.createElement("button");
                    btn.type = "button";
                    btn.textContent = labelText;
                    btn.className = fx.params[key] ? "is-active" : "";
                    btn.onclick = (event) => {
                        event.stopPropagation();
                        fx.params[key] = !fx.params[key];
                        addEdit(`${targetName} EQ ${labelText} ${fx.params[key] ? "on" : "off"}.`);
                        writeState(`eq_${key}`);
                        restartPlaybackIfNeeded(`eq_${key}`);
                        draw();
                    };
                    eqSwitches.appendChild(btn);
                });
            }
            device.append(head);
            if (fx.type === "eq") device.appendChild(eqSwitches);
            device.appendChild(knobs);
            const [kind, labelText] = previewForFx(fx.type);
            const panel = document.createElement("div");
            panel.className = "iamccs-inline-efx-panel";
            const previewLabel = document.createElement("label");
            previewLabel.textContent = labelText;
            let canvas = null;
            if (kind === "placeholder") {
                const ph = document.createElement("div");
                ph.className = "iamccs-efx-placeholder";
                ph.textContent = `${labelText}: realtime preview placeholder`;
                panel.append(previewLabel, ph);
            } else {
                canvas = document.createElement("canvas");
                canvas.className = "iamccs-inline-efx-canvas";
                canvas.dataset.kind = kind;
                canvas._iamccsFx = fx;
                panel.append(previewLabel, canvas);
            }
            if (kind === "eq") {
                canvas.onpointerdown = (event) => {
                    const eqFx = chainForTarget().find((fx) => fx.type === "eq");
                    if (!eqFx) return;
                    normalizeEffect(eqFx);
                    const rect = canvas.getBoundingClientRect();
                    const update = (ev, commit = false) => {
                        const x = Math.max(0, Math.min(1, (ev.clientX - rect.left) / Math.max(1, rect.width)));
                        const y = Math.max(0, Math.min(1, (ev.clientY - rect.top) / Math.max(1, rect.height)));
                        const eqFreq = eqBandFrequencies(eqFx);
                        const points = [
                            { key: "low", fxKey: "lowFreq", x: eqVisualRatioForFreq(eqFreq.low), min: 40, max: Math.max(400, eqFreq.mid - 120) },
                            { key: "mid", fxKey: "midFreq", x: eqVisualRatioForFreq(eqFreq.mid), min: eqFreq.low + 120, max: Math.max(eqFreq.low + 240, eqFreq.high - 400) },
                            { key: "high", fxKey: "highFreq", x: eqVisualRatioForFreq(eqFreq.high), min: eqFreq.mid + 400, max: 16000 },
                        ];
                        const nearest = points.reduce((best, item) => Math.abs(item.x - x) < Math.abs(best.x - x) ? item : best, points[0]);
                        const key = nearest.key;
                        eqFx.params[key] = Number(((.5 - y) * 48).toFixed(2));
                        const nextFreq = Math.max(nearest.min, Math.min(nearest.max, eqFreqForVisualRatio(x)));
                        eqFx.params[nearest.fxKey] = nextFreq;
                        syncInlineFxControls(eqFx);
                        paintInlineCanvasDeferred(canvas, "eq", selected, staticPeak);
                        if (commit) {
                            writeState(`eq_point_${key}`);
                            restartPlaybackIfNeeded(`eq_point_${key}`);
                        } else scheduleSilentStateWrite(`eq_point_${key}`);
                        if (!commit) scheduleShotboardSync(`eq_point_${key}`);
                    };
                    update(event, false);
                    const move = (ev) => { update(ev, false); ev.preventDefault(); };
                    const up = (ev) => {
                        window.removeEventListener("pointermove", move);
                        window.removeEventListener("pointerup", up);
                        update(ev, true);
                    };
                    window.addEventListener("pointermove", move);
                    window.addEventListener("pointerup", up, { once: true });
                };
            }
            if (kind === "bus") {
                canvas.onpointerdown = (event) => {
                    const activeFx = canvas._iamccsFx || chainForTarget().find((fx) => ["compressor", "limiter", "reverb", "delay", "stereo", "utility", "saturator", "chorus", "tape", "gate", "deesser", "transient"].includes(fx.type));
                    if (!activeFx) return;
                    normalizeEffect(activeFx);
                    const rect = canvas.getBoundingClientRect();
                    const update = (ev, commit = false) => {
                        const x = Math.max(0, Math.min(1, (ev.clientX - rect.left) / Math.max(1, rect.width)));
                        const y = Math.max(0, Math.min(1, (ev.clientY - rect.top) / Math.max(1, rect.height)));
                        if (activeFx.type === "limiter") {
                            activeFx.params.ceiling = Number((-12 + x * 12).toFixed(1));
                            activeFx.params.output = Number((-12 + (1 - y) * 18).toFixed(1));
                        } else if (activeFx.type === "reverb") {
                            activeFx.params.size = Number(x.toFixed(2));
                            activeFx.params.decay = Number((.1 + (1 - y) * 7.9).toFixed(2));
                            activeFx.params.mix = Number((Math.max(.02, y)).toFixed(2));
                        } else if (activeFx.type === "delay") {
                            activeFx.params.time = Number((.03 + x * 1.47).toFixed(2));
                            activeFx.params.feedback = Number((Math.max(0, Math.min(.9, 1 - y * .95))).toFixed(2));
                            activeFx.params.mix = Number((Math.max(.02, y)).toFixed(2));
                        } else if (activeFx.type === "stereo") {
                            activeFx.params.width = Number((x * 2).toFixed(2));
                            activeFx.params.angle = Number((-45 + (1 - y) * 90).toFixed(1));
                            activeFx.params.mix = Number((Math.max(.02, y)).toFixed(2));
                        } else if (activeFx.type === "utility") {
                            activeFx.params.pan = Number((-1 + x * 2).toFixed(2));
                            activeFx.params.gain = Number((-24 + (1 - y) * 48).toFixed(1));
                            activeFx.params.width = Number((x * 2).toFixed(2));
                            activeFx.params.mono = x < .08 ? 1 : 0;
                        } else if (["saturator", "chorus", "tape"].includes(activeFx.type)) {
                            const primaryKey = activeFx.type === "tape" ? "sat" : activeFx.type === "chorus" ? "depth" : "mix";
                            activeFx.params[primaryKey] = Number((Math.max(.02, x)).toFixed(2));
                            activeFx.params.mix = Number((Math.max(.02, y)).toFixed(2));
                        } else {
                            activeFx.params.threshold = Number((-60 + x * 60).toFixed(1));
                            activeFx.params.ratio = Number((1 + (1 - y) * 19).toFixed(1));
                        }
                        mirrorDynamicsFxToState(activeFx, target);
                        syncInlineFxControls(activeFx);
                        paintInlineCanvasDeferred(canvas, "bus", selected, staticPeak);
                        if (commit) {
                            writeState(`${activeFx.type}_point`);
                            restartPlaybackIfNeeded(`${activeFx.type}_point`);
                        } else scheduleSilentStateWrite(`${activeFx.type}_point`);
                        if (!commit) scheduleShotboardSync(`${activeFx.type}_point`);
                    };
                    update(event, false);
                    const move = (ev) => { update(ev, false); ev.preventDefault(); };
                    const up = (ev) => {
                        window.removeEventListener("pointermove", move);
                        window.removeEventListener("pointerup", up);
                        update(ev, true);
                    };
                    window.addEventListener("pointermove", move);
                    window.addEventListener("pointerup", up, { once: true });
                };
            }
            module.append(device, panel);
            chainEl.appendChild(module);
            if (canvas) paintInlineCanvasDeferred(canvas, kind, selected, staticPeak);
        });
        grid.appendChild(chainEl);
        efx.appendChild(grid);
        parent.appendChild(efx);
        requestAnimationFrame(() => {
            grid.scrollLeft = Math.max(0, Number(root._iamccsFxRackScrollLeft || 0));
        });
    };
    const buildTrackEffectLabels = (trackState) => {
        const labels = [];
        const activeEffects = Array.isArray(trackState?.effectChain)
            ? trackState.effectChain.filter((fx) => fx && fx.enabled !== false).map((fx) => String(fx.type || "FX").toUpperCase())
            : [];
        labels.push(...activeEffects);
        if (trackState?.mute) labels.push("MUTE");
        if (trackState?.solo) labels.push("SOLO");
        if (trackState?.normalize) labels.push("NORMALIZE");
        if (trackState?.bypassEffects) labels.push("DRY FX");
        if (trackState?.reverb) labels.push("REV SEND");
        return labels;
    };
    const bindLowerPanelResize = (handle) => {
        handle.onpointerdown = (event) => {
            event.preventDefault();
            const startY = event.clientY;
            const startHeight = Math.max(250, Math.min(760, Number(state.lowerPanelHeight || 390) || 390));
            try { handle.setPointerCapture?.(event.pointerId); } catch {}
            const move = (moveEvent) => {
                const nextHeight = Math.max(250, Math.min(760, startHeight + (moveEvent.clientY - startY)));
                state.lowerPanelHeight = nextHeight;
                writeState("lower_panel_resize", false, { quiet: true });
                draw();
                moveEvent.preventDefault();
            };
            const up = (upEvent) => {
                window.removeEventListener("pointermove", move);
                window.removeEventListener("pointerup", up);
                try { handle.releasePointerCapture?.(upEvent.pointerId); } catch {}
                addEdit(`FX rack height ${Math.round(Number(state.lowerPanelHeight || startHeight))} px.`);
                writeState("lower_panel_resize", false);
            };
            window.addEventListener("pointermove", move, { passive: false });
            window.addEventListener("pointerup", up, { passive: false, once: true });
        };
    };
    let refreshFixedAudioBoardSize = () => {};
    const draw = () => {
        closeContextMenu();
        root.classList.toggle("is-tool-cut", view().tool === "cut");
        root.classList.toggle("is-tool-trim", view().tool === "trim");
        root.querySelectorAll(".iamccs-audio-board-dynamic").forEach((el) => el.remove());
        const dynamic = document.createElement("div");
        dynamic.className = "iamccs-audio-board-dynamic";
        const head = document.createElement("div");
        head.className = "iamccs-audio-board-head";
        const left = document.createElement("div");
        left.innerHTML = `<div class="iamccs-audio-board-sub">${segments().length} clips / ${Math.max(1, Number(state.audioTrackCount || 4))} tracks / ${fmtTime(totalFrames())} / Shotboard links: ${linkedShotboardNodes().length} / auto publish: ${state.shotboardAutoSyncEnabled ? "ON" : "OFF"} / ${dialoguePromptSyncLabel()}</div>`;
        const tools = document.createElement("div");
        tools.className = "iamccs-audio-board-tools";
        const toolsStack = document.createElement("div");
        toolsStack.style.cssText = "display:grid;gap:8px;justify-items:stretch;align-items:start;min-width:0;";
        const selected = selectedClip();
        const setSelectedExternal = (key, value) => {
            if (!selected) return;
            selected[key] = value;
            addEdit(`Edited ${String(selected.name || selected.fileName || "clip")}: ${key}`);
            writeState(`edit_${key}`);
            draw();
        };
        const clipActions = document.createElement("div");
        clipActions.className = "iamccs-clip-action-bar";
        clipActions.innerHTML = `<strong>CLIP EDIT</strong>`;
        addButton(tools, root._iamccsAudioFullscreenState ? "Close Editor" : "Open Editor", () => toggleAudioFullscreen(), root._iamccsAudioFullscreenState ? "is-active" : "");
        addButton(tools, "MULTI", () => { state.showMultiGeneration = !state.showMultiGeneration; addEdit(`Multigeneration panel ${state.showMultiGeneration ? "shown" : "hidden"}.`); writeState("toggle_multi", false); draw(); }, state.showMultiGeneration ? "is-active" : "");
        const saveAudioBoardBtn = addButton(tools, "Save AudioBoard", (event) => { event?.preventDefault?.(); event?.stopPropagation?.(); saveAudioBoardPackage(); }, "is-save-audioboard");
        saveAudioBoardBtn.dataset.iamccsAudioAction = "save_audioboard";
        saveAudioBoardBtn.onpointerdown = (event) => { event.preventDefault(); event.stopPropagation(); saveAudioBoardPackage(); };
        saveAudioBoardBtn.onmousedown = (event) => { event.preventDefault(); event.stopPropagation(); };
        const saveAsBtn = addButton(tools, "Save As", (event) => { event?.preventDefault?.(); event?.stopPropagation?.(); saveAudioBoardPackageAs(); }, "is-save-audioboard");
        saveAsBtn.dataset.iamccsAudioAction = "save_audioboard_as";
        saveAsBtn.onpointerdown = (event) => { event.preventDefault(); event.stopPropagation(); saveAudioBoardPackageAs(); };
        saveAsBtn.onmousedown = (event) => { event.preventDefault(); event.stopPropagation(); };
        const importBoardBtn = addButton(tools, "Import AudioBoard", (event) => { event?.preventDefault?.(); event?.stopPropagation?.(); audioBoardInput.click(); });
        importBoardBtn.dataset.iamccsAudioAction = "import_audioboard";
        importBoardBtn.onpointerdown = (event) => { event.preventDefault(); event.stopPropagation(); audioBoardInput.click(); };
        importBoardBtn.onmousedown = (event) => { event.preventDefault(); event.stopPropagation(); };
        addButton(tools, "Import Audio", () => fileInput.click());
        addButton(tools, "Add Track", () => { state.audioTrackCount = Math.max(1, Number(state.audioTrackCount || 4)) + 1; addEdit("Added audio track."); writeState("add_track"); draw(); });
        addButton(tools, "ONLY FIRST", () => {
            state.audioBusMode = state.audioBusMode === "only_first" ? "all_tracks" : "only_first";
            state.onlyFirstTrack = state.audioBusMode === "only_first";
            addEdit(state.onlyFirstTrack ? "Shotboard V3 sync uses only track A1." : "Shotboard V3 sync uses all tracks.");
            writeState("bus_mode");
            draw();
        }, state.audioBusMode === "only_first" ? "is-active" : "");
        for (const tool of ["cursor", "move", "trim", "cut"]) {
            const helper = tool === "cut"
                ? "Tool: cut. Click a clip to split at the clicked frame."
                : tool === "trim"
                    ? "Tool: trim. Drag a clip edge to trim."
                    : "";
            addButton(tools, tool.toUpperCase(), () => setTool(tool, helper), view().tool === tool ? "is-active" : "");
        }
        addButton(tools, "Delete Selected", () => deleteSelectedClip(), "danger");
        addButton(tools, "Zoom +", () => { view().timeZoom = Math.min(8, view().timeZoom * 1.25); addEdit("Zoomed timeline in."); writeState("zoom", false); draw(); });
        addButton(tools, "Zoom -", () => { view().timeZoom = Math.max(.35, view().timeZoom / 1.25); addEdit("Zoomed timeline out."); writeState("zoom", false); draw(); });
        addButton(tools, "Tall +", () => { view().trackHeight = Math.min(240, view().trackHeight + 10); addEdit("Increased track height."); writeState("track_height", false); draw(); });
        addButton(tools, "Tall -", () => { view().trackHeight = Math.max(184, view().trackHeight - 10); addEdit("Reduced track height."); writeState("track_height", false); draw(); });
        addButton(tools, "Event Monitor", () => { state.showEventMonitor = !state.showEventMonitor; addEdit(`Event monitor ${state.showEventMonitor ? "shown" : "hidden"}.`); writeState("event_monitor", false); draw(); }, state.showEventMonitor ? "is-active" : "");
        addButton(tools, "AUTO PUBLISH", () => { state.shotboardAutoSyncEnabled = !state.shotboardAutoSyncEnabled; addEdit(`Shotboard auto publish ${state.shotboardAutoSyncEnabled ? "enabled" : "disabled"}.`); writeState("toggle_shotboard_auto_publish", false); draw(); }, state.shotboardAutoSyncEnabled ? "is-active" : "");
        addButton(tools, "Publish", () => publishToShotboard());
        addButton(tools, "Clear", () => { stopPlayback(false); state.audioSegments = []; selectedId = ""; addEdit("Cleared arranger clips."); writeState("clear"); draw(); }, "danger");
        addButton(clipActions, "Track Up", () => selected && setSelectedExternal("track", Math.max(0, Number(selected.track || 0) - 1)));
        addButton(clipActions, "Track Down", () => selected && setSelectedExternal("track", Math.min(Math.max(1, Number(state.audioTrackCount || 4)) - 1, Number(selected.track || 0) + 1)));
        addButton(clipActions, "Trim In", () => trimStartToPlayhead());
        addButton(clipActions, "Trim Out", () => trimEndToPlayhead());
        addButton(clipActions, "Split", () => splitSelectedAtPlayhead());
        addButton(clipActions, "Delete Clip", () => {
            deleteSelectedClip();
        }, "danger");
        toolsStack.append(tools, clipActions);
        head.append(left, toolsStack);
        dynamic.appendChild(head);
        if (state.showMultiGeneration || (state.multiGeneration && state.multiGeneration.enabled)) appendMultiGenerationStrip(dynamic);

        const transportBar = document.createElement("div");
        transportBar.className = "iamccs-audio-board-transport";
        const transportLeft = document.createElement("div");
        transportLeft.className = "iamccs-transport-left";
        const transportCenter = document.createElement("div");
        transportCenter.className = "iamccs-transport-center";
        const transportRight = document.createElement("div");
        transportRight.className = "iamccs-transport-right";
        addButton(transportCenter, "|<", () => { stopPlayback(false); setPlayhead(0, true); addEdit("Rewind to start."); });
        addButton(transportCenter, "<<", () => { setPlayhead(transport.playhead - secondsToFrames(1), true); addEdit("Back 1 second."); });
        addButton(transportCenter, transport.playing ? "Stop" : "Play", () => transport.playing ? stopPlayback(true) : playPlayback(), `${transport.playing ? "is-active " : ""}is-play`);
        addButton(transportCenter, ">>", () => { setPlayhead(transport.playhead + secondsToFrames(1), true); addEdit("Forward 1 second."); });
        addButton(transportCenter, ">|", () => { setPlayhead(totalFrames(), true); addEdit("Go to end."); });
        addButton(transportCenter, "I", () => setLoopIn(), "is-marker");
        addButton(transportCenter, "O", () => setLoopOut(), "is-marker");
        addButton(transportCenter, "Loop", () => toggleLoop(), `${state.loopEnabled ? "is-active " : ""}is-loop`);
        addButton(transportLeft, "Cut", () => splitSelectedAtPlayhead());
        addButton(transportLeft, "Trim In", () => trimStartToPlayhead());
        addButton(transportLeft, "Trim Out", () => trimEndToPlayhead());
        addButton(transportLeft, "Nudge -", () => nudgeSelected(-1));
        addButton(transportLeft, "Nudge +", () => nudgeSelected(1));
        const time = document.createElement("div");
        time.className = "iamccs-transport-time";
        time.textContent = `${fmtTime(transport.playhead)} / ${fmtTime(totalFrames())}`;
        const monitor = document.createElement("div");
        monitor.className = "iamccs-helper-monitor";
        monitor.textContent = transport.helper || "Ready.";
        transportRight.append(time, monitor);
        transportBar.append(transportLeft, transportCenter, transportRight);
        dynamic.appendChild(transportBar);

        const master = document.createElement("div");
        master.className = "iamccs-audio-board-master";
        master.onclick = () => {
            state.selectedMixer = { type: "master", track: 0 };
            transport.helper = "Selected MASTER OUT device chain.";
            writeState("select_master", false);
            draw();
        };
        master.innerHTML = `<div class="iamccs-master-title">MASTER OUT</div><span class="iamccs-master-meter is-stereo"><i style="width:0%"></i><i style="width:0%"></i></span>`;
        const masterControls = document.createElement("div");
        masterControls.className = "iamccs-master-controls";
        masterControls.onclick = (event) => event.stopPropagation();
        const masterLimiterFx = masterEffectByType("limiter");
        const masterCompFx = masterEffectByType("compressor");
        const limiterToggle = document.createElement("button");
        limiterToggle.type = "button";
        limiterToggle.className = `iamccs-master-toggle${masterLimiterFx?.enabled !== false ? " is-active" : ""}`;
        limiterToggle.textContent = "LIMITER";
        limiterToggle.onclick = () => {
            const limiterFx = masterEffectByType("limiter");
            if (limiterFx) {
                limiterFx.enabled = limiterFx.enabled === false;
                state.masterBus.limiter = limiterFx.enabled !== false;
            } else {
                addEffectToChain("limiter", { type: "master", track: 0 });
                return;
            }
            syncMasterFlagsFromChain();
            addEdit(`Master limiter ${state.masterBus.limiter ? "on" : "off"}.`);
            writeState("master_limiter");
            restartPlaybackIfNeeded("master_limiter_toggle");
            draw();
        };
        const compToggle = document.createElement("button");
        compToggle.type = "button";
        compToggle.className = `iamccs-master-toggle${masterCompFx?.enabled !== false ? " is-active" : ""}`;
        compToggle.textContent = "COMP";
        compToggle.onclick = () => {
            const compFx = masterEffectByType("compressor");
            if (compFx) {
                compFx.enabled = compFx.enabled === false;
                state.masterBus.compressor = compFx.enabled !== false ? Math.max(.45, Number(state.masterBus?.compressor || compFx.amount || .45)) : 0;
            } else {
                addEffectToChain("compressor", { type: "master", track: 0 });
                return;
            }
            syncMasterFlagsFromChain();
            addEdit(`Master compressor ${Number(state.masterBus.compressor || 0) > 0 ? "on" : "off"}.`);
            writeState("master_compressor");
            restartPlaybackIfNeeded("master_compressor_toggle");
            draw();
        };
        masterControls.append(limiterToggle, compToggle);
        const masterFxSelect = document.createElement("select");
        masterFxSelect.className = "iamccs-master-fx-select";
        masterFxSelect.innerHTML = `<option value="">+ Master FX</option>`;
        for (const [value, label] of EFFECT_CHOICES) {
            const option = document.createElement("option");
            option.value = value;
            option.textContent = label;
            masterFxSelect.appendChild(option);
        }
        masterFxSelect.onchange = (event) => {
            event.stopPropagation();
            if (!masterFxSelect.value) return;
            addEffectToChain(masterFxSelect.value, { type: "master", track: 0 });
            masterFxSelect.value = "";
        };
        masterFxSelect.onpointerdown = (event) => event.stopPropagation();
        masterFxSelect.onclick = (event) => event.stopPropagation();
        masterControls.appendChild(masterFxSelect);
        const masterFields = [
            ["Gain", "masterAudioGain", 0, 2, .05],
            ["Ceil dB", "masterBus.ceilingDb", -12, 0, .5],
            ["Comp", "masterBus.compressor", 0, 1, .05],
            ["Width", "masterBus.width", 0, 2, .05],
            ["Rev", "masterBus.reverbSend", 0, 1, .05],
            ["Delay", "masterBus.delaySend", 0, 1, .05],
        ];
        for (const [label, key, min, max, step] of masterFields) {
            const wrap = document.createElement("label");
            wrap.style.cssText = "display:grid;gap:2px;color:#91a5ac;font-size:8px;font-weight:900;text-transform:uppercase;";
            wrap.textContent = label;
            const input = document.createElement("input");
            input.type = "number";
            input.min = String(min);
            input.max = String(max);
            input.step = String(step);
            input.style.cssText = "width:54px;height:22px;background:#10171a;color:#fff;border:1px solid rgba(255,255,255,.16);border-radius:4px;";
            input.value = key.startsWith("masterBus.") ? String(state.masterBus?.[key.split(".")[1]] ?? 0) : String(state[key] ?? 1);
            input.onchange = () => {
                if (key.startsWith("masterBus.")) state.masterBus[key.split(".")[1]] = Number(input.value || 0);
                else state[key] = Number(input.value || 1);
                addEdit(`Changed master ${label}.`);
                writeState(`master_${label}`);
                restartPlaybackIfNeeded(`master_${String(label || "field").toLowerCase().replace(/[^a-z0-9]+/g, "_")}`);
                draw();
            };
            makeDraggableNumber(input, { step, min, max });
            wrap.appendChild(input);
            masterControls.appendChild(wrap);
        }
        const norm = document.createElement("label");
        norm.style.cssText = "display:flex;align-items:center;gap:5px;color:#91a5ac;font-size:9px;font-weight:900;text-transform:uppercase;";
        const normInput = document.createElement("input");
        normInput.type = "checkbox";
        normInput.checked = Boolean(state.masterAudioNormalize);
        normInput.onchange = () => { state.masterAudioNormalize = Boolean(normInput.checked); addEdit("Toggled master normalize."); writeState("master_normalize"); restartPlaybackIfNeeded("master_normalize_toggle"); draw(); };
        norm.append(normInput, "Normalize");
        masterControls.appendChild(norm);
        const mono = document.createElement("label");
        mono.style.cssText = "display:flex;align-items:center;gap:5px;color:#91a5ac;font-size:9px;font-weight:900;text-transform:uppercase;";
        const monoInput = document.createElement("input");
        monoInput.type = "checkbox";
        monoInput.checked = Boolean(state.masterMono);
        monoInput.onchange = () => { state.masterMono = Boolean(monoInput.checked); addEdit(`Master out ${state.masterMono ? "mono" : "stereo"}.`); writeState("master_mono"); restartPlaybackIfNeeded("master_mono_toggle"); draw(); };
        mono.append(monoInput, "Mono");
        masterControls.appendChild(mono);
        master.appendChild(masterControls);
        const masterMonitor = document.createElement("span");
        masterMonitor.className = "iamccs-master-crt iamccs-master-readout";
        masterMonitor.textContent = "PK 000 RMS 000";
        master.appendChild(masterMonitor);
        dynamic.appendChild(master);

        const trackCount = Math.max(1, Number(state.audioTrackCount || 1));
        const visibleTrackCount = Math.min(trackCount, IAMCCS_AUDIO_BOARD_VISIBLE_TRACKS);
        const timelineHeight = 28 + visibleTrackCount * Math.max(184, Math.min(240, Number(view().trackHeight || 188) || 188));
        const timeline = document.createElement("div");
        timeline.className = "iamccs-audio-board-timeline";
        timeline.style.setProperty("--iamccs-track-height", `${view().trackHeight}px`);
        timeline.style.setProperty("--iamccs-px-per-frame", `${pxPerFrame()}px`);
        timeline.style.height = `${timelineHeight}px`;
        timeline.style.overflowY = trackCount > IAMCCS_AUDIO_BOARD_VISIBLE_TRACKS ? "auto" : "hidden";
        const editorContentWidth = contentWidth();
        const ruler = document.createElement("div");
        ruler.className = "iamccs-audio-board-ruler";
        ruler.style.width = `${editorContentWidth}px`;
        ruler.style.minWidth = `${editorContentWidth}px`;
        const activeLoopRange = loopRange();
        const addLoopRange = (parent, klass) => {
            if (!activeLoopRange) return;
            const region = document.createElement("div");
            region.className = klass;
            region.style.left = `${frameToX(activeLoopRange.start)}px`;
            region.style.width = `${Math.max(2, frameToX(activeLoopRange.end) - frameToX(activeLoopRange.start))}px`;
            parent.appendChild(region);
        };
        const scrubToEvent = (event, redraw = false) => {
            const rect = ruler.getBoundingClientRect();
            const scaleX = Math.max(.001, ruler.offsetWidth / Math.max(1, rect.width));
            const contentX = Math.max(0, (event.clientX - rect.left) * scaleX);
            const frame = xToFrame(contentX);
            setPlayhead(frame, redraw);
            transport.helper = `Scrub ${fmtTime(transport.playhead)} / ${fmtTime(totalFrames())}`;
        };
        ruler.onpointerdown = (event) => {
            event.preventDefault();
            scrubToEvent(event, false);
            ruler.setPointerCapture?.(event.pointerId);
            const move = (moveEvent) => {
                scrubToEvent(moveEvent, false);
                moveEvent.preventDefault();
            };
            const up = () => {
                window.removeEventListener("pointermove", move);
                window.removeEventListener("pointerup", up);
                draw();
            };
            window.addEventListener("pointermove", move);
            window.addEventListener("pointerup", up, { once: true });
        };
        addLoopRange(ruler, "iamccs-loop-range-ruler");
        const rulerSeconds = Math.max(VIEWPORT_SECONDS, Math.ceil(framesToSeconds(totalFrames())));
        for (let s = 0; s <= rulerSeconds + .01; s += 1) {
            const x = frameToX(secondsToFrames(s));
            const tick = document.createElement("div");
            tick.style.cssText = `position:absolute;left:${x}px;top:0;bottom:0;width:1px;background:rgba(244,212,158,.25);`;
            const rulerTickLabel = document.createElement("span");
            rulerTickLabel.textContent = `${s}s`;
            rulerTickLabel.style.cssText = `position:absolute;left:${x + 4}px;top:5px;color:#9fb1b8;font-size:9px;font-weight:850;`;
            ruler.append(tick, rulerTickLabel);
        }
        const addLoopMarker = (frame, label, out = false) => {
            const marker = document.createElement("div");
            marker.className = `iamccs-loop-marker${out ? " out" : ""}`;
            marker.style.left = `${frameToX(frame)}px`;
            marker.textContent = label;
            ruler.appendChild(marker);
        };
        addLoopMarker(Math.max(0, Number(state.loopInFrame || 0)), "IN");
        if (Number(state.loopOutFrame || 0) > Number(state.loopInFrame || 0)) addLoopMarker(Number(state.loopOutFrame || 0), "OUT", true);
        const rulerHead = document.createElement("div");
        rulerHead.style.cssText = `position:sticky;left:0;z-index:9;width:${LABEL_W}px;height:28px;background:#11181c;border-right:1px solid rgba(255,255,255,.12);display:flex;align-items:center;justify-content:center;color:#ffe2a8;font-size:9px;font-weight:900;`;
        rulerHead.textContent = "TIME";
        ruler.appendChild(rulerHead);
        timeline.appendChild(ruler);
        for (let track = 0; track < trackCount; track += 1) {
            const isTrackSelected = selectedMixer().type === "track" && selectedMixer().track === track;
            const lane = document.createElement("div");
            lane.className = `iamccs-audio-board-track${isTrackSelected ? " is-selected" : ""}`;
            lane.style.width = `${editorContentWidth}px`;
            lane.style.minWidth = `${editorContentWidth}px`;
            addLoopRange(lane, "iamccs-loop-range");
            const trackSegs = segments().filter((seg) => Number(seg.track || 0) === track);
            lane.oncontextmenu = (event) => openTrackContextMenu(track, event);
            const trackState = trackSettings(track);
            trackState.volume = Math.max(0, Math.min(2, finiteNumber(trackState.volume, 1)));
            trackState.pan = Math.max(-1, Math.min(1, finiteNumber(trackState.pan, 0)));
            trackState.color = normalizeTrackColor(trackState.color, track);
            lane.style.setProperty("--iamccs-track-color", trackState.color);
            lane.style.setProperty("--iamccs-track-fill", trackColorWithAlpha(trackState.color, .13, track));
            lane.style.setProperty("--iamccs-track-glow", trackColorWithAlpha(trackState.color, .22, track));
            const trackPeak = trackState.mute ? 0 : Math.min(1, trackSegs.reduce((sum, seg) => sum + peakFor(seg), 0));
            const trackLabel = document.createElement("div");
            trackLabel.className = `iamccs-audio-board-track-label${selectedMixer().type === "track" && selectedMixer().track === track ? " is-selected" : ""}${trackState.mute ? " is-muted" : ""}${trackState.lock ? " is-locked" : ""}`;
            trackLabel.style.borderLeft = `4px solid ${trackState.color}`;
            trackLabel.style.setProperty("--iamccs-track-glow", trackColorWithAlpha(trackState.color, .22, track));
            trackLabel.oncontextmenu = (event) => openTrackContextMenu(track, event);
            trackLabel.onclick = (event) => {
                event.stopPropagation();
                const wasSelected = selectedMixer().type === "track" && selectedMixer().track === track;
                state.selectedMixer = { type: "track", track };
                transport.helper = `Selected A${track + 1} device chain.`;
                writeState("select_track", false, { quiet: true });
                if (!wasSelected) draw();
            };
            const top = document.createElement("div");
            top.className = "iamccs-track-strip-top";
            top.innerHTML = `<div class="iamccs-track-name"><span>A${track + 1}</span><small>${trackSegs.length} clips</small></div><span class="iamccs-audio-board-meter iamccs-track-meter ${trackHasStereoContent(track) ? "is-stereo" : ""}" data-track="${track}"><i style="width:${transport.playing ? Math.round(trackPeak * 100) : 0}%"></i>${trackHasStereoContent(track) ? `<i style="width:${transport.playing ? Math.round(trackPeak * 100) : 0}%"></i>` : ""}</span>`;
            const controls = document.createElement("div");
            controls.className = "iamccs-track-strip-controls";
            const addTrackToggle = (labelText, key) => {
                const btn = document.createElement("button");
                btn.type = "button";
                btn.className = `iamccs-track-mini${trackState[key] ? " is-active" : ""}`;
                btn.textContent = labelText;
                btn.setAttribute("aria-pressed", trackState[key] ? "true" : "false");
                btn.title = `Toggle ${labelText} on A${track + 1}`;
                btn.onpointerdown = (event) => event.stopPropagation();
                btn.onclick = (event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    state.selectedMixer = { type: "track", track };
                    trackState[key] = !trackState[key];
                    btn.classList.toggle("is-active", Boolean(trackState[key]));
                    btn.setAttribute("aria-pressed", trackState[key] ? "true" : "false");
                    noteTrackMixLocalState(track, key, trackState[key] ? 1 : 0, "toggle");
                    addEdit(`A${track + 1} ${labelText} ${trackState[key] ? "on" : "off"}.`);
                    const pushed = pushArrangerTrackToggle(track, key, trackState[key]);
                    writeState(`track_${key}`);
                    if (!pushed) restartPlaybackIfNeeded(`track_${key}_toggle`);
                    draw();
                };
                controls.appendChild(btn);
            };
            addTrackToggle("M", "mute");
            addTrackToggle("S", "solo");
            addTrackToggle("N", "normalize");
            addTrackToggle("DRY", "bypassEffects");
            addTrackToggle("RV", "reverb");
            addTrackToggle("L", "lock");
            const colorButton = document.createElement("button");
            colorButton.type = "button";
            colorButton.className = "iamccs-track-color-chip";
            colorButton.textContent = "C";
            colorButton.title = `A${track + 1} lane color`;
            colorButton.style.setProperty("--iamccs-track-color", trackState.color);
            colorButton.onclick = (event) => {
                event.stopPropagation();
                state.selectedMixer = { type: "track", track };
                openTrackColorPicker(track, event);
            };
            controls.appendChild(colorButton);
            const knobRow = document.createElement("div");
            knobRow.className = "iamccs-track-effect-row";
            const effectLabels = buildTrackEffectLabels(trackState);
            if (!effectLabels.length) {
                const emptyChip = document.createElement("span");
                emptyChip.className = "iamccs-track-effect-chip is-empty";
                emptyChip.textContent = "NO ACTIVE FX";
                knobRow.appendChild(emptyChip);
            } else {
                effectLabels.forEach((labelText) => {
                    const chip = document.createElement("span");
                    chip.className = "iamccs-track-effect-chip is-active";
                    chip.textContent = labelText;
                    knobRow.appendChild(chip);
                });
            }
            const bottom = document.createElement("div");
            bottom.className = "iamccs-track-strip-bottom";
            const fxSelect = document.createElement("select");
            fxSelect.className = "iamccs-track-fx-select";
            fxSelect.innerHTML = `<option value="">+ insert FX</option>`;
            for (const [value, labelText] of EFFECT_CHOICES) {
                const option = document.createElement("option");
                option.value = value;
                option.textContent = labelText;
                fxSelect.appendChild(option);
            }
            fxSelect.onchange = (event) => {
                event.stopPropagation();
                if (!fxSelect.value) return;
                state.selectedMixer = { type: "track", track };
                addEffectToChain(fxSelect.value, { type: "track", track });
                fxSelect.value = "";
            };
            fxSelect.onpointerdown = (event) => event.stopPropagation();
            fxSelect.onclick = (event) => event.stopPropagation();
            bottom.appendChild(fxSelect);
            trackLabel.append(top, controls, knobRow, bottom);
            lane.appendChild(trackLabel);
            for (const seg of trackSegs) {
                const clip = document.createElement("div");
                clip.className = `iamccs-audio-clip${seg.id === selectedId ? " is-selected" : ""}${seg.mute ? " is-muted" : ""}`;
                clip.dataset.segId = String(seg.id || "");
                clip.style.borderColor = trackState.color;
                clip.title = "Drag to move. Drag edges to trim. Double-click to move playhead.";
                const leftHandle = document.createElement("div");
                leftHandle.className = "iamccs-clip-handle left";
                const rightHandle = document.createElement("div");
                rightHandle.className = "iamccs-clip-handle right";
                const name = document.createElement("div");
                name.className = "iamccs-audio-clip-name";
                name.textContent = String(seg.name || seg.fileName || "audio");
                const timeLabel = document.createElement("div");
                timeLabel.className = "iamccs-audio-clip-time";
                timeLabel.textContent = `${fmtTime(seg.start)} - ${fmtTime(Number(seg.start || 0) + Number(seg.length || 1))}`;
                const trimLabel = document.createElement("div");
                trimLabel.className = "iamccs-audio-clip-trim";
                trimLabel.textContent = `src +${fmtTime(seg.trimStart || 0)}`;
                const sourceMarker = document.createElement("div");
                sourceMarker.className = "iamccs-clip-source-marker";
                clip.append(name, timeLabel, trimLabel, sourceMarker);
                updateClipStyle(clip, seg);
                clip.append(leftHandle, rightHandle);
                attachClipPointer(clip, seg);
                lane.appendChild(clip);
            }
            const playhead = document.createElement("div");
            playhead.className = "iamccs-playhead";
            playhead.style.left = `${frameToX(transport.playhead)}px`;
            lane.appendChild(playhead);
            timeline.appendChild(lane);
        }
        dynamic.appendChild(timeline);

        if (selected && !selectedId) selectedId = selected.id;

        root.appendChild(dynamic);
        collectTransportDom();
        requestAnimationFrame(() => refreshFixedAudioBoardSize());
    };
    let bootError = null;
    try {
        writeState("init", false);
        draw();
    } catch (err) {
        bootError = err;
        console.error("[IAMCCS AudioBoardArranger] draw failed before DOM mount", {
            nodeId: node?.id,
            error: err?.message || String(err),
            stack: err?.stack || "",
            widgets: (node?.widgets || []).map((w) => ({ name: w?.name, type: w?.type, hidden: w?.hidden, valueLength: String(w?.value || "").length })),
        });
        root.innerHTML = "";
        const errorPanel = document.createElement("div");
        errorPanel.className = "iamccs-audio-board-dynamic";
        errorPanel.style.cssText = [
            "min-height:420px",
            "padding:14px",
            "border:1px solid rgba(255,120,120,.45)",
            "border-radius:8px",
            "background:linear-gradient(180deg,#1a1112,#0b0f11)",
            "color:#f3d7d7",
            "font:12px/1.35 ui-monospace, SFMono-Regular, Consolas, monospace",
            "white-space:pre-wrap",
            "overflow:auto",
        ].join(";");
        const title = document.createElement("div");
        title.style.cssText = "font:800 14px/1.2 Inter,system-ui,sans-serif;color:#ffd9a1;margin-bottom:10px";
        title.textContent = "IAMCCS AudioBoardArranger render failed";
        const body = document.createElement("div");
        body.textContent = `${err?.message || String(err)}\n\n${err?.stack || ""}`;
        errorPanel.append(title, body);
        root.appendChild(errorPanel);
    }
    if (typeof node.addDOMWidget !== "function") {
        console.error("[IAMCCS AudioBoardArranger] addDOMWidget is not available", {
            nodeId: node?.id,
            type: node?.type,
            comfyClass: node?.comfyClass,
            widgets: (node?.widgets || []).map((w) => w?.name),
        });
        return;
    }
    const domWidget = node.addDOMWidget("AudioBoard Arranger", "iamccs_audio_board_arranger", root, { serialize: false });
    hideWidget(dataWidget);
    const applyFixedAudioBoardSize = () => {
        const fixed = iamccsAudioBoardFixedSizeFromDom(root, dataWidget?.value || "");
        node._iamccsAudioBoardFixedSize = fixed;
        node.resizable = false;
        node.resizeable = false;
        node.flags = { ...(node.flags || {}), resizable: false };
        node.min_size = fixed.slice();
        const current = Array.isArray(node.size) ? node.size : [0, 0];
        if (Number(current[0] || 0) !== fixed[0] || Number(current[1] || 0) !== fixed[1]) {
            if (typeof node.setSize === "function") node.setSize(fixed.slice());
            else node.size = fixed.slice();
        }
    };
    refreshFixedAudioBoardSize = applyFixedAudioBoardSize;
    if (!node._iamccsAudioBoardResizeLocked) {
        const previousOnResize = node.onResize;
        node.onResize = function (size) {
            const fixed = this._iamccsAudioBoardFixedSize || [AUDIO_BOARD_FIXED_WIDTH, AUDIO_BOARD_FIXED_HEIGHT];
            if (Array.isArray(size)) {
                size[0] = fixed[0];
                size[1] = fixed[1];
            }
            const result = typeof previousOnResize === "function" ? previousOnResize.apply(this, arguments) : undefined;
            this.resizable = false;
            this.resizeable = false;
            this.flags = { ...(this.flags || {}), resizable: false };
            this.min_size = fixed.slice();
            if (Number(this.size?.[0] || 0) !== fixed[0] || Number(this.size?.[1] || 0) !== fixed[1]) {
                if (typeof this.setSize === "function") this.setSize(fixed.slice());
                else this.size = fixed.slice();
            }
            return result;
        };
        node._iamccsAudioBoardResizeLocked = true;
    }
    if (domWidget) {
        domWidget.computeSize = () => root._iamccsAudioFullscreenState
            ? [AUDIO_BOARD_FIXED_WIDTH, 24]
            : [node._iamccsAudioBoardFixedSize?.[0] || AUDIO_BOARD_FIXED_WIDTH, Math.max(24, Number(node._iamccsAudioBoardFixedSize?.[1] || AUDIO_BOARD_FIXED_HEIGHT) - 70)];
    }
    requestAnimationFrame(() => applyFixedAudioBoardSize());
    if (!node._iamccsAudioBoardSizeGuard) {
        node._iamccsAudioBoardSizeGuard = window.setInterval(() => {
            if (!node.graph) {
                window.clearInterval(node._iamccsAudioBoardSizeGuard);
                node._iamccsAudioBoardSizeGuard = 0;
                return;
            }
            applyFixedAudioBoardSize();
        }, 2400);
    }
    node._iamccsAudioBoardReady = true;
    console.info("[IAMCCS AudioBoardArranger] render complete", {
        nodeId: node?.id,
        bootError: bootError?.message || "",
        domWidget: Boolean(domWidget),
        widgetCount: node?.widgets?.length,
        rootChildren: root.childElementCount,
    });
}

function iamccsAudioDialogueType(node) {
    const raw = [
        node?.comfyClass,
        node?.type,
        node?.title,
        node?.constructor?.comfyClass,
        node?.constructor?.type,
        node?.constructor?.nodeData?.name,
    ].map((item) => String(item || "")).filter(Boolean);
    const widgets = (node?.widgets || []).map((w) => String(w?.name || ""));
    const inputs = (node?.inputs || []).map((item) => String(item?.name || ""));
    const outputs = (node?.outputs || []).map((item) => String(item?.name || ""));
    if (raw.includes("IAMCCS_AudioBoardArranger") || raw.includes("IAMCCS AudioBoard Arranger") || widgets.includes("arranger_data")) return "IAMCCS_AudioBoardArranger";
    if (raw.includes("IAMCCS_AudioBoardMixer") || raw.includes("IAMCCS AudioBoard Mixer") || widgets.includes("mixer_data")) return "IAMCCS_AudioBoardMixer";
    if (
        raw.includes("IAMCCS_ControlAudEfx")
        || raw.includes("IAMCCS ControlAudEfx")
        || raw.includes("IAMCCS_ControlAudEfxPanel")
        || raw.includes("IAMCCS ControlAudEfx Panel")
        || widgets.includes("control_data")
        || inputs.includes("cine_linx_from_arranger")
        || outputs.includes("effect_graph_json")
        || outputs.includes("selected_clip_json")
    ) return "IAMCCS_ControlAudEfx";
    if (raw.includes("IAMCCS_CineEmotionButtons") || widgets.includes("selected_emotions")) return "IAMCCS_CineEmotionButtons";
    if (raw.includes("IAMCCS_BoardMaker_DialogueFoley") || widgets.includes("board_json")) return "IAMCCS_BoardMaker_DialogueFoley";
    if (raw.includes("IAMCCS_CineSpeech1PromptCompiler") || widgets.includes("speech1_prompt")) return "IAMCCS_CineSpeech1PromptCompiler";
    return "";
}

function renderIamccsAudioDialogueNode(node, source = "unknown") {
    const type = iamccsAudioDialogueType(node);
    if (!type) return;
    if (![
        "IAMCCS_CineEmotionButtons",
        "IAMCCS_BoardMaker_DialogueFoley",
        "IAMCCS_AudioBoardArranger",
        "IAMCCS_ControlAudEfx",
        "IAMCCS_ControlAudEfxPanel",
        "IAMCCS_AudioBoardMixer",
        "IAMCCS_CineSpeech1PromptCompiler",
    ].includes(type)) return;
    console.info("[IAMCCS AudioDialogueUI] render hook", {
        source,
        type,
        id: node?.id,
        rawType: node?.type,
        comfyClass: node?.comfyClass,
        title: node?.title,
        ready: node?._iamccsAudioBoardReady,
        widgets: (node?.widgets || []).map((w) => w?.name),
    });
    try {
        if (type === "IAMCCS_CineEmotionButtons") renderEmotionPanel(node);
        if (type === "IAMCCS_BoardMaker_DialogueFoley") renderDialogueFoleyBoardMaker(node);
        if (type === "IAMCCS_CineSpeech1PromptCompiler") renderSpeech1Compiler(node);
        if (type === "IAMCCS_AudioBoardArranger") renderAudioBoardArranger(node);
        if (type === "IAMCCS_ControlAudEfx") renderControlAudEfx(node);
        if (type === "IAMCCS_AudioBoardMixer") renderAudioBoardMixer(node);
    } catch (err) {
        if (type === "IAMCCS_AudioBoardArranger") node._iamccsAudioBoardReady = false;
        if (type === "IAMCCS_ControlAudEfx") {
            node._iamccsControlAudEfxReady = false;
            node._iamccsControlAudEfxMounting = false;
        }
        if (type === "IAMCCS_AudioBoardMixer") {
            node._iamccsAudioBoardMixerReady = false;
            node._iamccsAudioBoardMixerMounting = false;
        }
        console.error(`[IAMCCS AudioDialogueUI] render hook failed ${type}: ${err?.message || String(err)}\n${err?.stack || ""}`, {
            source,
            type,
            id: node?.id,
            widgets: (node?.widgets || []).map((w) => ({ name: w?.name, type: w?.type, hidden: w?.hidden, valueLength: String(w?.value || "").length })),
        });
    }
}

function ensureControlAudEfxStyles() {
    if (document.getElementById("iamccs-control-aud-efx-style")) return;
    const style = document.createElement("style");
    style.id = "iamccs-control-aud-efx-style";
    style.textContent = `
        .iamccs-audefx {
            box-sizing: border-box;
            width: 100%;
            max-width: 100%;
            min-height: 520px;
            padding: 10px;
            overflow: hidden;
            color: #dce7ea;
            background: linear-gradient(180deg, #14191c, #0d1114);
            border: 1px solid rgba(150,174,184,.22);
            border-radius: 8px;
            font: 11px/1.25 Inter, ui-sans-serif, system-ui, sans-serif;
        }
        .iamccs-audefx * { box-sizing: border-box; letter-spacing: 0; }
        .iamccs-audefx-head,
        .iamccs-audefx-tools,
        .iamccs-audefx-strip {
            display: flex;
            align-items: center;
            gap: 7px;
            flex-wrap: wrap;
        }
        .iamccs-audefx-head { justify-content: space-between; margin-bottom: 8px; }
        .iamccs-audefx-title { color: #fff; font-weight: 950; font-size: 13px; }
        .iamccs-audefx-sub { color: #8fa5ac; font-weight: 750; font-size: 10px; }
        .iamccs-audefx-target {
            display: inline-flex;
            align-items: center;
            min-height: 26px;
            padding: 0 9px;
            border-radius: 999px;
            border: 1px solid rgba(255,226,168,.24);
            background: linear-gradient(180deg, rgba(52,63,65,.94), rgba(18,23,25,.94));
            color: #ffe7b6;
            font: 900 10px/1 ui-monospace, SFMono-Regular, Consolas, monospace;
        }
        .iamccs-audefx button {
            min-height: 25px;
            padding: 0 8px;
            color: #ecffff;
            background: linear-gradient(180deg, #284a4e, #1a3034);
            border: 1px solid rgba(143,208,204,.35);
            border-radius: 5px;
            font-size: 10px;
            font-weight: 850;
            cursor: pointer;
        }
        .iamccs-audefx button.is-active {
            color: #101315;
            background: linear-gradient(180deg, #f2d79a, #c79e59);
            border-color: #ffe6ae;
        }
        .iamccs-audefx-select {
            min-width: 190px;
            height: 28px;
            min-height: 28px;
            padding: 0 8px;
            border: 1px solid rgba(143,208,204,.38);
            border-radius: 6px;
            background: #071012;
            color: #e9ffff;
            font: 900 10px/1 ui-monospace, SFMono-Regular, Consolas, monospace;
        }
        .iamccs-audefx-strip {
            margin: 8px 0 10px;
            padding: 7px;
            background: rgba(0,0,0,.18);
            border: 1px solid rgba(255,255,255,.10);
            border-radius: 7px;
        }
        .iamccs-audefx-summary {
            color: #9fb1b8;
            font: 900 10px/1.35 ui-monospace, SFMono-Regular, Consolas, monospace;
        }
        .iamccs-audefx-empty {
            min-height: 180px;
            display: grid;
            place-items: center;
            color: #9fb1b8;
            border: 1px dashed rgba(244,212,158,.20);
            border-radius: 8px;
            background: repeating-linear-gradient(135deg, rgba(255,255,255,.025) 0 1px, transparent 1px 9px);
            font: 900 10px/1.4 ui-monospace, SFMono-Regular, Consolas, monospace;
            text-align: center;
        }
    `;
    document.head.appendChild(style);
}

function renderControlAudEfx(node) {
    if (node._iamccsControlAudEfxReady || node._iamccsControlAudEfxMounting) return;
    node._iamccsControlAudEfxMounting = true;
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    if (typeof node.addDOMWidget !== "function") {
        // addDOMWidget not yet available — let the delayed retry handle it
        node._iamccsControlAudEfxMounting = false;
        return;
    }
    ensureAudioBoardArrangerStyles();
    ensureControlAudEfxStyles();
    const dataWidget = findWidget(node, "control_data");
    if (dataWidget) hideWidget(dataWidget);
    const root = document.createElement("div");
    root.className = "iamccs-audefx";
    root.tabIndex = 0;
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com

    const parseControl = () => {
        try { return JSON.parse(String(dataWidget?.value || "{}")); } catch { return {}; }
    };
    const writeControl = (patch = {}) => {
        const data = { schema: "iamccs.control_aud_efx", schema_version: 1, ...parseControl(), ...patch };
        if (dataWidget) {
            dataWidget.value = JSON.stringify(data, null, 2);
            dataWidget.callback?.(dataWidget.value);
        }
        node.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas?.(true, true);
    };
    if (!dataWidget) {
        root.innerHTML = `<div class="iamccs-audefx-head"><div><div class="iamccs-audefx-title">IAMCCS ControlAudEfx</div><div class="iamccs-audefx-sub">UI placeholder mounted, but control_data widget is missing on this node instance.</div></div></div><div class="iamccs-audefx-empty">This node instance was created without the expected hidden widget \"control_data\". Recreate the node from the current IAMCCS-nodes build.</div>`;
        const domWidget = node.addDOMWidget("ControlAudEfx", "iamccs_control_aud_efx", root, { serialize: false });
        domWidget.computeSize = (width) => [Math.max(1320, Number(width || 1320)), 260];
        requestAnimationFrame(() => {
            const w = Math.max(1340, node.size?.[0] || 1340);
            if (typeof node.setSize === "function") node.setSize([w, 320]);
            else node.size = [w, 320];
            node.setDirtyCanvas?.(true, true);
            app.graph?.setDirtyCanvas?.(true, true);
        });
        node._iamccsControlAudEfxReady = true;
        node._iamccsControlAudEfxMounting = false;
        return;
    }
    const EFFECT_SPECS = {
        eq: [["low", "LOW", -24, 24, .5, 0, "dB"], ["mid", "MID", -24, 24, .5, 0, "dB"], ["high", "HIGH", -24, 24, .5, 0, "dB"], ["q", "Q", .2, 8, .1, 1.2, ""], ["lowFreq", "LOW HZ", 40, 800, 10, 140, "Hz"], ["midFreq", "MID HZ", 200, 6000, 50, 1200, "Hz"], ["highFreq", "HIGH HZ", 1800, 16000, 100, 6200, "Hz"]],
        compressor: [["threshold", "THR", -60, 0, 1, -18, "dB"], ["ratio", "RATIO", 1, 20, .5, 4, ":1"], ["attack", "ATK", 1, 100, 1, 6, "ms"], ["release", "REL", 20, 800, 10, 180, "ms"], ["knee", "KNEE", 0, 40, 1, 12, "dB"], ["makeup", "MAKEUP", -12, 24, .5, 0, "dB"]],
        limiter: [["input", "IN", -12, 12, .5, 0, "dB"], ["ceiling", "CEIL", -12, 0, .5, -1, "dB"], ["lookahead", "LOOK", 0, 10, .5, 3, "ms"], ["release", "REL", 20, 800, 10, 120, "ms"], ["output", "OUT", -12, 6, .5, 0, "dB"], ["softclip", "SOFT", 0, 1, .05, .25, ""]],
        gate: [["threshold", "THR", -80, 0, 1, -45, "dB"], ["attack", "ATK", 1, 80, 1, 8, "ms"], ["hold", "HOLD", 0, 400, 10, 80, "ms"], ["release", "REL", 20, 1200, 10, 260, "ms"]],
        reverb: [["size", "SIZE", 0, 1, .01, .45, ""], ["decay", "DECAY", .1, 8, .1, 1.8, "s"], ["damp", "DAMP", 0, 1, .01, .35, ""], ["mix", "MIX", 0, 1, .01, .18, ""], ["predelay", "PRE", 0, 250, 5, 20, "ms"], ["width", "WIDTH", 0, 2, .05, 1, ""]],
        delay: [["time", "TIME", .03, 1.5, .01, .18, "s"], ["feedback", "FDBK", 0, .9, .01, .28, ""], ["filter", "FILT", 200, 12000, 100, 4200, "Hz"], ["mix", "MIX", 0, 1, .01, .2, ""], ["spread", "SPREAD", 0, 1, .05, .25, ""], ["drive", "DRIVE", 0, 18, .5, 0, "dB"]],
        saturator: [["drive", "DRIVE", 0, 24, .5, 4, "dB"], ["color", "COLOR", 0, 1, .01, .45, ""], ["tone", "TONE", 200, 12000, 100, 2600, "Hz"], ["mix", "MIX", 0, 1, .01, .5, ""]],
        utility: [["gain", "GAIN", -24, 24, .5, 0, "dB"], ["width", "WIDTH", 0, 2, .05, 1, ""], ["pan", "PAN", -1, 1, .05, 0, ""], ["mono", "MONO", 0, 1, 1, 0, ""]],
        stereo: [["width", "WIDTH", 0, 2, .05, 1.15, ""], ["angle", "ANGLE", -45, 45, 1, 0, "deg"], ["bassMono", "BASS", 0, 300, 10, 120, "Hz"], ["mix", "MIX", 0, 1, .01, .5, ""]],
        deesser: [["freq", "FREQ", 2500, 10000, 100, 6200, "Hz"], ["threshold", "THR", -60, 0, 1, -26, "dB"], ["range", "RANGE", 0, 24, .5, 8, "dB"], ["mix", "MIX", 0, 1, .01, .75, ""]],
        transient: [["attack", "ATK", -1, 1, .05, .15, ""], ["sustain", "SUS", -1, 1, .05, 0, ""], ["drive", "DRIVE", 0, 12, .5, 0, "dB"], ["mix", "MIX", 0, 1, .01, .6, ""]],
        tape: [["bias", "BIAS", 0, 1, .01, .45, ""], ["wow", "WOW", 0, 1, .01, .08, ""], ["sat", "SAT", 0, 1, .01, .35, ""], ["hiss", "HISS", 0, 1, .01, .04, ""]],
        chorus: [["rate", "RATE", .05, 8, .05, .8, "Hz"], ["depth", "DEPTH", 0, 1, .01, .35, ""], ["phase", "PHASE", 0, 180, 1, 90, "deg"], ["mix", "MIX", 0, 1, .01, .22, ""]],
    };
    const effectParamSpecs = (type) => EFFECT_SPECS[String(type || "")] || [["amount", "AMT", 0, 1, .01, .5, ""], ["mix", "MIX", 0, 1, .01, .5, ""]];
    const normalizeEffect = (fx) => {
        const out = fx && typeof fx === "object" ? fx : {};
        out.id = out.id || `fx_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
        out.type = String(out.type || "utility");
        out.enabled = out.enabled !== false;
        out.params = out.params && typeof out.params === "object" ? out.params : {};
        effectParamSpecs(out.type).forEach(([key, , , , , fallback]) => {
            if (out.params[key] == null || Number.isNaN(Number(out.params[key]))) out.params[key] = fallback;
        });
        out.amount = Math.max(0, Math.min(1, Number(out.amount ?? .5)));
        return out;
    };
    const upstreamArranger = () => {
        const candidates = [];
        const seen = new Set();
        const consider = (maybeNode) => {
            const id = Number(maybeNode?.id || 0);
            const type = String(maybeNode?.comfyClass || maybeNode?.type || "");
            if (!id || seen.has(id) || type !== "IAMCCS_AudioBoardArranger") return;
            seen.add(id);
            candidates.push(maybeNode);
        };
        for (const input of node.inputs || []) {
            const link = input.link != null ? app.graph?.links?.[input.link] : null;
            const origin = link ? app.graph?.getNodeById?.(link.origin_id) : null;
            consider(origin);
        }
        if (candidates.length) return candidates[0];
        return (app.graph?._nodes || []).find((item) => String(item?.comfyClass || item?.type || "") === "IAMCCS_AudioBoardArranger") || null;
    };
    const arrangerState = () => {
        const arranger = upstreamArranger();
        const widget = arranger ? findWidget(arranger, "arranger_data") : null;
        try {
            const data = JSON.parse(String(widget?.value || "{}"));
            return data && typeof data === "object" ? data : {};
        } catch { return {}; }
    };
    const deviceColors = {
        eq: ["#6a5532", "#29251c", "#0b0d0c"],
        compressor: ["#5a3f2c", "#2d241b", "#0b0d0c"],
        limiter: ["#673631", "#2c1e1b", "#0b0d0c"],
        gate: ["#394d3f", "#1d2922", "#0b0d0c"],
        reverb: ["#35445f", "#1b2232", "#090b12"],
        delay: ["#5b4a2d", "#2a2418", "#0d0b08"],
        saturator: ["#6b4026", "#322116", "#0d0907"],
        utility: ["#3e514d", "#1c2927", "#071010"],
        stereo: ["#423c60", "#211f32", "#0a0912"],
        deesser: ["#4b5a35", "#242d1c", "#090d08"],
        transient: ["#614c34", "#2c251a", "#0c0a08"],
        tape: ["#57412b", "#2b2118", "#0d0906"],
        chorus: ["#34565d", "#1a2b30", "#071012"],
    };
    const trackState = (data, index) => {
        data.trackSettings = Array.isArray(data.trackSettings) ? data.trackSettings : [];
        const safeIndex = Math.max(0, Number(index || 0));
        data.trackSettings[safeIndex] = {
            mute: false,
            solo: false,
            normalize: false,
            reverb: false,
            reverbSend: 0,
            volume: 1,
            gainDb: 0,
            pan: 0,
            bypassEffects: false,
            color: normalizeTrackColor("", safeIndex),
            lock: false,
            noAutoEq: true,
            effectChain: [],
            ...(data.trackSettings[safeIndex] && typeof data.trackSettings[safeIndex] === "object" ? data.trackSettings[safeIndex] : {}),
        };
        data.trackSettings[safeIndex].effectChain = Array.isArray(data.trackSettings[safeIndex].effectChain) ? data.trackSettings[safeIndex].effectChain : [];
        return data.trackSettings[safeIndex];
    };
    const selectedTarget = (data) => {
        data.audioTrackCount = Math.max(1, Number(data.audioTrackCount || 5));
        data.selectedMixer = data.selectedMixer && typeof data.selectedMixer === "object" ? data.selectedMixer : { type: "track", track: 0 };
        data.selectedMixer.type = data.selectedMixer.type === "track" ? "track" : "master";
        data.selectedMixer.track = Math.max(0, Math.min(data.audioTrackCount - 1, Number(data.selectedMixer.track || 0)));
        return data.selectedMixer;
    };
    const chainForTarget = (data, target = selectedTarget(data)) => {
        if (target.type === "track") {
            const chain = trackState(data, target.track).effectChain;
            chain.forEach(normalizeEffect);
            return chain;
        }
        data.masterBus = data.masterBus && typeof data.masterBus === "object" ? data.masterBus : {};
        data.masterBus.effectChain = Array.isArray(data.masterBus.effectChain) ? data.masterBus.effectChain : [];
        data.masterBus.effectChain.forEach(normalizeEffect);
        return data.masterBus.effectChain;
    };
    const persistArranger = (data, reason = "control_efx") => {
        const arranger = upstreamArranger();
        const widget = arranger ? findWidget(arranger, "arranger_data") : null;
        if (!widget) return false;
        widget.value = JSON.stringify(data, null, 2);
        widget.callback?.(widget.value);
        arranger._iamccsAudioBoardTransport?.applyExternalState?.(data, `control_efx_${reason}`);
        arranger.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas?.(true, true);
        writeControl({ last_reason: reason, selectedMixer: data.selectedMixer });
        document.dispatchEvent(new CustomEvent("iamccs:audio_arranger_state_changed", {
            detail: { node_id: arranger.id, reason, selectedMixer: JSON.parse(JSON.stringify(data.selectedMixer || { type: "track", track: 0 })) },
        }));
        return true;
    };
    const targetSummary = (data, target) => target.type === "track"
        ? `A${target.track + 1} / clips ${(Array.isArray(data.audioSegments) ? data.audioSegments : []).filter((seg) => Number(seg?.track || 0) === target.track).length}`
        : "MASTER OUT";
    const controlMeterPeak = (target) => {
        const snap = upstreamArranger()?._iamccsAudioBoardTransport?.snapshot?.();
        if (!snap?.playing) return 0;
        return target.type === "track"
            ? Number(snap?.meters?.tracks?.[target.track]?.peak || 0)
            : Number(snap?.meters?.master?.peak || 0);
    };
    const paintControlPreview = (canvas, fx, target) => {
        const rect = canvas.getBoundingClientRect();
        const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
        const w = Math.max(320, Math.round(rect.width || 435));
        const h = Math.max(220, Math.round(rect.height || 318));
        if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
            canvas.width = Math.round(w * dpr);
            canvas.height = Math.round(h * dpr);
        }
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = "#071012";
        ctx.fillRect(0, 0, w, h);
        ctx.strokeStyle = "rgba(255,255,255,.07)";
        ctx.lineWidth = 1;
        for (let x = 0; x <= w; x += w / 8) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
        for (let y = 0; y <= h; y += h / 5) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
        const p = fx.params || {};
        const type = String(fx.type || "utility");
        const point = (x, y, color, label) => {
            ctx.fillStyle = color;
            ctx.beginPath(); ctx.arc(x, y, 7, 0, Math.PI * 2); ctx.fill();
            ctx.strokeStyle = "rgba(0,0,0,.75)"; ctx.stroke();
            ctx.fillStyle = "#fff2cf"; ctx.font = "900 9px ui-monospace,Consolas,monospace"; ctx.fillText(label, x + 10, Math.max(12, y - 9));
        };
        const dbY = (db) => h * .5 - Math.max(-24, Math.min(24, Number(db || 0))) / 48 * h * .78;
        if (type === "eq") {
            const freqX = (hz) => Math.max(0, Math.min(w, (Math.log10(Math.max(20, Number(hz || 20))) - Math.log10(20)) / (Math.log10(20000) - Math.log10(20)) * w));
            const pts = [[0, h*.5], [freqX(p.lowFreq || 140), dbY(p.low)], [freqX(p.midFreq || 1200), dbY(p.mid)], [freqX(p.highFreq || 6200), dbY(p.high)], [w, h*.5]];
            ctx.strokeStyle = "#f5d08e"; ctx.lineWidth = 2.5; ctx.beginPath();
            pts.forEach(([x,y], i) => i ? ctx.lineTo(x,y) : ctx.moveTo(x,y)); ctx.stroke();
            point(pts[1][0], pts[1][1], "#d6a75e", "L");
            point(pts[2][0], pts[2][1], "#55c7b9", "M");
            point(pts[3][0], pts[3][1], "#d08963", "H");
        } else if (type === "compressor" || type === "limiter" || type === "gate" || type === "deesser") {
            const threshold = Math.max(-60, Math.min(0, Number(p.threshold ?? -18)));
            const ratio = Math.max(1, Math.min(20, Number(p.ratio ?? 4)));
            const px = ((threshold + 60) / 60) * w;
            const py = h - ((ratio - 1) / 19) * h;
            ctx.strokeStyle = "#78c5c8"; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(0, h);
            ctx.lineTo(px, h - px / Math.max(1,w) * h);
            ctx.lineTo(w, Math.max(12, h - px / Math.max(1,w) * h - (w-px)/Math.max(1,ratio))); ctx.stroke();
            point(px, py, "#d6a75e", type === "limiter" ? "CEIL" : "THR/R");
        } else if (type === "reverb") {
            const decay = Math.max(.1, Math.min(8, Number(p.decay ?? 1.8)));
            const size = Math.max(0, Math.min(1, Number(p.size ?? .45)));
            ctx.strokeStyle = "#55c7b9"; ctx.lineWidth = 2.2; ctx.beginPath();
            for (let x=0; x<w; x+=1) {
                const t=x/Math.max(1,w-1), env=Math.exp(-t*(1.1+decay*.8));
                const y=h*.5-Math.sin(t*(12+size*28)*Math.PI)*env*h*.34;
                x ? ctx.lineTo(x,y) : ctx.moveTo(x,y);
            }
            ctx.stroke(); point(size*w, (1-Math.min(1,decay/8))*h, "#55c7b9", "SPACE");
        } else if (type === "delay") {
            const time = Math.max(.03, Math.min(1.5, Number(p.time ?? .18)));
            const feedback = Math.max(0, Math.min(.9, Number(p.feedback ?? .28)));
            for (let i=0;i<7;i+=1) {
                const x=18+(i+1)/7*(time/1.5)*(w-36), bar=Math.max(7,(1-i/8)*(feedback+.15)*h*.7);
                ctx.fillStyle=`rgba(240,184,87,${Math.max(.18,.85-i*.1)})`; ctx.fillRect(x-4,h-bar,8,bar);
            }
            point(time/1.5*w, (1-feedback/.9)*h, "#f0b857", "TIME/FB");
        } else {
            const amount = Math.max(0, Math.min(1, Number(p.mix ?? p.amount ?? fx.amount ?? .5)));
            ctx.strokeStyle="#f0b857"; ctx.lineWidth=2.2; ctx.beginPath();
            for(let x=0;x<w;x+=1){const n=x/Math.max(1,w-1)*2-1;const y=h*.5-Math.tanh(n*(1+amount*5))*h*.3;x?ctx.lineTo(x,y):ctx.moveTo(x,y);} ctx.stroke();
            point(amount*w, h*.5, "#55c7b9", "AMOUNT");
        }
        const peak = Math.max(0, Math.min(1, controlMeterPeak(target)));
        ctx.fillStyle = "rgba(85,199,185,.92)"; ctx.fillRect(7, h - Math.max(2, peak*h), 8, Math.max(2, peak*h));
    };
    const bindControlPreview = (canvas, fx, state, target) => {
        canvas.onpointerdown = (event) => {
            event.preventDefault();
            root._iamccsControlActiveFxId = String(fx.id || "");
            const rect = canvas.getBoundingClientRect();
            const update = (ev, commit = false) => {
                const x = Math.max(0, Math.min(1, (ev.clientX - rect.left) / Math.max(1, rect.width)));
                const y = Math.max(0, Math.min(1, (ev.clientY - rect.top) / Math.max(1, rect.height)));
                const p = fx.params || (fx.params = {});
                if (fx.type === "eq") {
                    const bands = [["low","lowFreq",.18],["mid","midFreq",.5],["high","highFreq",.82]];
                    const [gainKey,freqKey] = bands.reduce((best,item)=>Math.abs(item[2]-x)<Math.abs(best[2]-x)?item:best,bands[0]);
                    p[gainKey] = Number(((.5-y)*48).toFixed(2));
                    p[freqKey] = Math.round(20 * Math.pow(1000, x));
                } else if (fx.type === "compressor" || fx.type === "gate" || fx.type === "deesser") {
                    p.threshold = Number((-60+x*60).toFixed(1)); p.ratio = Number((1+(1-y)*19).toFixed(1));
                } else if (fx.type === "limiter") {
                    p.ceiling = Number((-12+x*12).toFixed(1)); p.output = Number((-12+(1-y)*18).toFixed(1));
                } else if (fx.type === "reverb") {
                    p.size = Number(x.toFixed(2)); p.decay = Number((.1+(1-y)*7.9).toFixed(2));
                } else if (fx.type === "delay") {
                    p.time = Number((.03+x*1.47).toFixed(2)); p.feedback = Number(((1-y)*.9).toFixed(2));
                } else {
                    fx.amount = Number(x.toFixed(2)); p.mix = Number((1-y).toFixed(2));
                }
                paintControlPreview(canvas, fx, target);
                if (commit) persistArranger(state, `panel_preview_${fx.type}`);
            };
            update(event, false);
            const move = (ev) => { update(ev, false); ev.preventDefault(); };
            const up = (ev) => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); update(ev, true); draw(); };
            window.addEventListener("pointermove", move, {passive:false});
            window.addEventListener("pointerup", up, {once:true});
        };
    };
    const draw = () => {
        const previousRack = root.querySelector(".iamccs-audefx-dynamic .iamccs-inline-efx-grid");
        if (previousRack) root._iamccsControlRackScrollLeft = Number(previousRack.scrollLeft || 0);
        root.querySelectorAll(".iamccs-audefx-dynamic").forEach((el) => el.remove());
        const state = arrangerState();
        const target = selectedTarget(state);
        const chain = chainForTarget(state, target);
        const dynamic = document.createElement("div");
        dynamic.className = "iamccs-audefx-dynamic";
        const head = document.createElement("div");
        head.className = "iamccs-audefx-head";
        const title = document.createElement("div");
        title.innerHTML = `<div class="iamccs-audefx-title">IAMCCS ControlAudEfx Panel</div><div class="iamccs-audefx-sub">Interactive device rack synced to Arranger selected track</div>`;
        const tools = document.createElement("div");
        tools.className = "iamccs-audefx-tools";
        const targetBadge = document.createElement("span");
        targetBadge.className = "iamccs-audefx-target";
        targetBadge.textContent = target.type === "track" ? `TRACK A${target.track + 1}` : "MASTER OUT";
        const insert = document.createElement("select");
        insert.className = "iamccs-audefx-select";
        insert.innerHTML = `<option value="">+ insert FX</option>`;
        for (const [value, labelText] of EFFECT_CHOICES) {
            const option = document.createElement("option");
            option.value = value;
            option.textContent = labelText;
            insert.appendChild(option);
        }
        insert.onchange = () => {
            const nextType = String(insert.value || "");
            if (!nextType) return;
            const data = arrangerState();
            const currentTarget = selectedTarget(data);
            const nextChain = chainForTarget(data, currentTarget);
            const nextFx = normalizeEffect({ id: `fx_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`, type: nextType, enabled: true, amount: nextType === "limiter" ? 1 : .5 });
            nextChain.push(nextFx);
            root._iamccsControlActiveFxId = String(nextFx.id || "");
            insert.value = "";
            persistArranger(data, `insert_${nextType}`);
            draw();
        };
        const refresh = document.createElement("button");
        refresh.textContent = "Refresh";
        refresh.onclick = () => draw();
        tools.append(targetBadge, insert, refresh);
        head.append(title, tools);
        dynamic.appendChild(head);
        const strip = document.createElement("div");
        strip.className = "iamccs-audefx-strip";
        strip.innerHTML = `<span class="iamccs-audefx-summary">${targetSummary(state, target)} / devices ${chain.length} / sync ${upstreamArranger() ? "Arranger" : "missing arranger"}</span>`;
        dynamic.appendChild(strip);
        if (!upstreamArranger()) {
            const empty = document.createElement("div");
            empty.className = "iamccs-audefx-empty";
            empty.textContent = "Connect cine_linx from IAMCCS_AudioBoardArranger to mirror the selected track effect rack here.";
            dynamic.appendChild(empty);
            root.appendChild(dynamic);
            return;
        }
        const rack = document.createElement("div");
        rack.className = "iamccs-inline-efx";
        rack.innerHTML = `<div class="iamccs-inline-efx-head"><span>DEVICE RACK / AUDIO CONTROL EFX</span><span>${target.type === "track" ? `A${target.track + 1}` : "MASTER OUT"}</span></div>`;
        const grid = document.createElement("div");
        grid.className = "iamccs-inline-efx-grid";
        grid.onscroll = () => { root._iamccsControlRackScrollLeft = Number(grid.scrollLeft || 0); };
        const chainEl = document.createElement("div");
        chainEl.className = "iamccs-device-chain";
        if (!chain.length) {
            const module = document.createElement("div");
            module.className = "iamccs-device-module";
            module.dataset.fxId = String(fx.id || "");
            const emptyDevice = document.createElement("div");
            emptyDevice.className = "iamccs-audio-device";
            emptyDevice.innerHTML = `<div class="iamccs-device-head"><span>NO DEVICE</span></div><div style="color:#b3a37e;font:800 9px/1.35 ui-monospace,SFMono-Regular,Consolas,monospace;">Selected ${target.type === "track" ? `track A${target.track + 1}` : "master"} has no inserted effects.</div>`;
            const ph = document.createElement("div");
            ph.className = "iamccs-inline-efx-panel";
            ph.innerHTML = `<label>Realtime Preview</label><div class="iamccs-efx-placeholder">Insert FX from this node or from the arranger track editor.</div>`;
            module.append(emptyDevice, ph);
            chainEl.appendChild(module);
        }
        chain.forEach((fx, index) => {
            normalizeEffect(fx);
            const module = document.createElement("div");
            module.className = "iamccs-device-module";
            const device = document.createElement("div");
            device.className = `iamccs-audio-device iamccs-device-${String(fx.type || "device").replace(/[^a-z0-9_-]/gi, "")}`;
            const palette = deviceColors[fx.type] || ["#3a3328", "#171714", "#0b0d0c"];
            device.style.setProperty("--device-hi", palette[0]);
            device.style.setProperty("--device-mid", palette[1]);
            device.style.setProperty("--device-lo", palette[2]);
            const name = EFFECT_CHOICES.find(([value]) => value === fx.type)?.[1] || String(fx.type || "FX");
            const headEl = document.createElement("div");
            headEl.className = "iamccs-device-head";
            const titleEl = document.createElement("span");
            titleEl.textContent = `${index + 1}. ${name}`;
            const actions = document.createElement("div");
            actions.className = "iamccs-device-head-actions";
            const lamp = document.createElement("span");
            lamp.className = `iamccs-device-lamp${fx.enabled === false ? " is-off" : ""}`;
            const power = document.createElement("button");
            power.type = "button";
            power.className = `iamccs-device-power${fx.enabled === false ? " is-off" : ""}`;
            power.title = "Enable / bypass";
            power.onclick = () => {
                const data = arrangerState();
                const currentTarget = selectedTarget(data);
                const currentFx = chainForTarget(data, currentTarget).find((item) => String(item?.id || "") === String(fx.id || ""));
                if (!currentFx) return;
                currentFx.enabled = currentFx.enabled === false;
                persistArranger(data, `toggle_${currentFx.type}`);
                draw();
            };
            const remove = document.createElement("button");
            remove.type = "button";
            remove.className = "iamccs-device-remove";
            remove.textContent = "x";
            remove.title = "Remove device";
            remove.onclick = () => {
                const data = arrangerState();
                const currentTarget = selectedTarget(data);
                const currentChain = chainForTarget(data, currentTarget);
                const indexToRemove = currentChain.findIndex((item) => String(item?.id || "") === String(fx.id || ""));
                if (indexToRemove < 0) return;
                currentChain.splice(indexToRemove, 1);
                persistArranger(data, `remove_${fx.type}`);
                draw();
            };
            actions.append(lamp, power, remove);
            headEl.append(titleEl, actions);
            const knobs = document.createElement("div");
            knobs.className = "iamccs-device-knobs";
            effectParamSpecs(fx.type).forEach(([key, knobName, min, max, step, fallback, unit]) => {
                if (fx.params[key] == null) fx.params[key] = fallback;
                const knob = document.createElement("div");
                knob.className = "iamccs-device-knob";
                const dial = document.createElement("i");
                const label = document.createElement("span");
                label.textContent = knobName;
                const readout = document.createElement("em");
                const range = document.createElement("input");
                range.type = "range";
                range.min = String(min);
                range.max = String(max);
                range.step = String(step);
                range.value = String(fx.params[key]);
                const syncUi = () => {
                    const liveValue = Math.max(Number(min), Math.min(Number(max), Number(fx.params[key] ?? fallback)));
                    const t = (liveValue - Number(min)) / Math.max(.0001, Number(max) - Number(min));
                    dial.style.setProperty("--knob-angle", `${-135 + t * 270}deg`);
                    readout.textContent = `${Number(liveValue.toFixed(2))}${unit || ""}`;
                    range.value = String(liveValue);
                };
                range.oninput = () => {
                    root._iamccsControlActiveFxId = String(fx.id || "");
                    fx.params[key] = Number(range.value || fallback);
                    syncUi();
                };
                range.onchange = () => {
                    root._iamccsControlActiveFxId = String(fx.id || "");
                    const data = arrangerState();
                    const currentTarget = selectedTarget(data);
                    const currentFx = chainForTarget(data, currentTarget).find((item) => String(item?.id || "") === String(fx.id || ""));
                    if (!currentFx) return;
                    normalizeEffect(currentFx);
                    currentFx.params[key] = Number(range.value || fallback);
                    persistArranger(data, `${currentFx.type}_${key}`);
                    draw();
                };
                dial.title = `Drag horizontally: ${knobName}`;
                dial.style.cursor = "ew-resize";
                dial.onpointerdown = (event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    root._iamccsControlActiveFxId = String(fx.id || "");
                    const startX = event.clientX;
                    const startValue = Number(fx.params[key] ?? fallback);
                    const move = (moveEvent) => {
                        const multiplier = moveEvent.shiftKey ? 10 : moveEvent.altKey ? .1 : 1;
                        const next = Math.max(Number(min), Math.min(Number(max), startValue + ((moveEvent.clientX - startX) / 5) * Number(step) * multiplier));
                        fx.params[key] = Number(next.toFixed(4));
                        syncUi();
                        module.querySelectorAll(".iamccs-inline-efx-canvas").forEach((preview) => paintControlPreview(preview, fx, target));
                        moveEvent.preventDefault();
                    };
                    const up = () => {
                        window.removeEventListener("pointermove", move);
                        window.removeEventListener("pointerup", up);
                        persistArranger(state, `panel_dial_${fx.type}_${key}`);
                        draw();
                    };
                    window.addEventListener("pointermove", move, {passive:false});
                    window.addEventListener("pointerup", up, {once:true});
                };
                knob.append(dial, label, readout, range);
                knobs.appendChild(knob);
                syncUi();
            });
            device.append(headEl, knobs);
            const panel = document.createElement("div");
            panel.className = "iamccs-inline-efx-panel";
            const previewLabel = document.createElement("label");
            previewLabel.textContent = `${name} realtime editor`;
            const preview = document.createElement("canvas");
            preview.className = "iamccs-inline-efx-canvas";
            preview.dataset.kind = String(fx.type || "fx");
            panel.append(previewLabel, preview);
            bindControlPreview(preview, fx, state, target);
            module.append(device, panel);
            requestAnimationFrame(() => paintControlPreview(preview, fx, target));
            chainEl.appendChild(module);
        });
        grid.appendChild(chainEl);
        rack.appendChild(grid);
        dynamic.appendChild(rack);
        root.appendChild(dynamic);
        requestAnimationFrame(() => {
            const activeId = String(root._iamccsControlActiveFxId || "");
            const activeModule = activeId ? chainEl.querySelector(`[data-fx-id="${CSS.escape(activeId)}"]`) : null;
            grid.scrollLeft = activeModule
                ? Math.max(0, Number(activeModule.offsetLeft || 0) - 8)
                : Math.max(0, Number(root._iamccsControlRackScrollLeft || 0));
            root._iamccsControlRackScrollLeft = Number(grid.scrollLeft || 0);
        });
    };
    document.addEventListener("iamccs:audio_arranger_state_changed", (event) => {
        const arranger = upstreamArranger();
        if (!arranger) return;
        if (Number(event?.detail?.node_id || 0) !== Number(arranger.id || 0)) return;
        draw();
    });
    node._iamccsControlAudEfxPoll = window.setInterval(() => {
        const arranger = upstreamArranger();
        const widget = arranger ? findWidget(arranger, "arranger_data") : null;
        const nextSnapshot = String(widget?.value || "");
        if (node._iamccsControlAudEfxLastSnapshot === nextSnapshot) return;
        node._iamccsControlAudEfxLastSnapshot = nextSnapshot;
        draw();
    }, 500);
    draw();
    const domWidget = node.addDOMWidget("ControlAudEfx", "iamccs_control_aud_efx", root, { serialize: false });
    domWidget.computeSize = (width) => [Math.max(1320, Number(width || 1320)), 560];
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    if (!node._iamccsControlAudEfxResizeLocked) {
        const _prevOnResizeCtrl = node.onResize;
        node.onResize = function(size) {
            if (Array.isArray(size)) { size[0] = Math.max(1340, size[0]); size[1] = Math.max(630, size[1]); }
            const result = typeof _prevOnResizeCtrl === "function" ? _prevOnResizeCtrl.apply(this, arguments) : undefined;
            this.resizable = false;
            this.resizeable = false;
            if (Number(this.size?.[1] || 0) < 630) {
                const nw = Math.max(1340, this.size?.[0] || 1340);
                if (typeof this.setSize === "function") this.setSize([nw, 630]);
                else this.size = [nw, 630];
            }
            return result;
        };
        node._iamccsControlAudEfxResizeLocked = true;
    }
    node.resizable = false;
    node.resizeable = false;
    const _enforceCtrlSize = () => {
        const w = Math.max(1340, node.size?.[0] || 1340);
        if (typeof node.setSize === "function") node.setSize([w, 630]);
        else node.size = [w, 630];
        node.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas?.(true, true);
    };
    requestAnimationFrame(_enforceCtrlSize);
    setTimeout(_enforceCtrlSize, 250);
    setTimeout(_enforceCtrlSize, 1200);
    node._iamccsControlAudEfxReady = true;
    node._iamccsControlAudEfxMounting = false;
}


function ensureAudioBoardMixerStyles() {
    if (document.getElementById("iamccs-audio-board-mixer-style")) return;
    const style = document.createElement("style");
    style.id = "iamccs-audio-board-mixer-style";
    style.textContent = `
        .iamccs-audio-mixer {
            box-sizing: border-box;
            width: 100%;
            min-height: 650px;
            padding: 10px 10px 20px;
            overflow: hidden;
            color: #dce7ea;
            background: linear-gradient(180deg, #222322, #111314 42%, #0b0d0e);
            border: 1px solid rgba(170,184,184,.24);
            border-radius: 8px;
            font: 11px/1.25 Inter, ui-sans-serif, system-ui, sans-serif;
        }
        .iamccs-audio-mixer * { box-sizing: border-box; letter-spacing: 0; }
        .iamccs-mixer-head,
        .iamccs-mixer-transport,
        .iamccs-mixer-strips {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .iamccs-mixer-head { justify-content: space-between; margin-bottom: 8px; }
        .iamccs-mixer-title { color: #f1f3f2; font-weight: 950; font-size: 13px; }
        .iamccs-mixer-sub { color: #a7adae; font-weight: 750; font-size: 10px; }
        .iamccs-audio-mixer button {
            min-height: 24px;
            padding: 0 8px;
            color: #eaf4f2;
            background: linear-gradient(180deg, #45494a, #272a2b);
            border: 1px solid rgba(210,218,216,.24);
            border-radius: 5px;
            font: 850 10px/1 ui-monospace, SFMono-Regular, Consolas, monospace;
            cursor: pointer;
        }
        .iamccs-audio-mixer button.is-active {
            color: #101612;
            border-color: #f4d98e;
            background: linear-gradient(180deg, #fff0b2, #d39b4a);
            box-shadow: 0 0 0 2px rgba(244,217,142,.28), 0 0 14px rgba(244,196,104,.30), inset 0 1px 0 rgba(255,255,255,.75);
            transform: translateY(1px);
        }
        .iamccs-audio-mixer button[aria-pressed="true"]::after {
            content:"";
            display:inline-block;
            width:6px;
            height:6px;
            margin-left:5px;
            border-radius:50%;
            background:#43ef80;
            box-shadow:0 0 8px rgba(67,239,128,.95);
        }
        .iamccs-mixer-transport {
            padding: 8px;
            margin-bottom: 8px;
            background: #303031;
            border: 1px solid rgba(255,255,255,.09);
            border-radius: 6px;
        }
        .iamccs-mixer-time {
            min-width: 190px;
            color: #b9bec4;
            font: 950 23px/1 ui-monospace, SFMono-Regular, Consolas, monospace;
        }
        .iamccs-mixer-state { margin-left: auto; color: #9fa1a5; font: 850 16px/1 ui-monospace, SFMono-Regular, Consolas, monospace; }
        .iamccs-mixer-strips {
            align-items: stretch;
            overflow-x: auto;
            overflow-y: hidden;
            padding-bottom: 18px;
            scrollbar-color: #596164 #1b1d1e;
        }
        .iamccs-mixer-strip {
            flex: 0 0 158px;
            min-height: 490px;
            display: grid;
            grid-template-rows: 40px 42px 38px 1fr 32px 34px;
            gap: 7px;
            padding: 8px;
            color: #d6dbdc;
            background: linear-gradient(180deg, #8e9896 0 24%, #2c2e2f 24% 100%);
            border: 1px solid rgba(255,255,255,.12);
            border-radius: 2px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.10), 0 8px 18px rgba(0,0,0,.18);
        }
        .iamccs-mixer-strip.is-master {
            flex-basis: 210px;
            background: linear-gradient(180deg, #2b2c2d, #17191a);
        }
        .iamccs-mixer-pan {
            display:grid;
            place-items:center;
            gap:2px;
            color:#313437;
            font:800 9px/1 ui-monospace,SFMono-Regular,Consolas,monospace;
        }
        .iamccs-mixer-knob {
            width:34px;
            height:34px;
            min-height:34px;
            padding:0;
            border-radius:50%;
            border:1px solid rgba(0,0,0,.35);
            background:
                conic-gradient(from 225deg, rgba(255,255,255,.12) 0 18%, #cfd6d4 var(--knob-fill, 38%), rgba(0,0,0,.28) var(--knob-fill, 38%) 78%, transparent 78% 100%),
                radial-gradient(circle at 35% 30%, #f2f1ed, #aaa9a4 54%, #3c3e3f 100%);
            box-shadow: inset 0 1px 2px rgba(255,255,255,.7), inset 0 -4px 7px rgba(0,0,0,.36);
            position:relative;
            cursor:ew-resize;
        }
        .iamccs-mixer-knob::after {
            content:"";
            position:absolute;
            left:50%;
            top:5px;
            width:2px;
            height:11px;
            transform-origin:50% 12px;
            transform:translateX(-50%) rotate(var(--knob-angle, 0deg));
            background:#313437;
            border-radius:999px;
        }
        .iamccs-mixer-actions {
            display:grid;
            grid-template-columns: repeat(3, 1fr);
            gap:5px;
        }
        .iamccs-mixer-actions button {
            min-height:28px;
            color:#a85b58;
            background:linear-gradient(180deg,#333536,#202223);
            border-color:rgba(255,255,255,.10);
            font-size:11px;
        }
        .iamccs-mixer-actions button.solo { color:#cbd45b; }
        .iamccs-mixer-actions button.dry { color:#81c9d0; }
        .iamccs-mixer-actions button.is-active { color:#101612; }
        .iamccs-mixer-gain {
            display:grid;
            grid-template-columns:auto 1fr;
            gap:5px;
            align-items:center;
            color:#2d3132;
            font:900 9px/1 ui-monospace,SFMono-Regular,Consolas,monospace;
        }
        .iamccs-mixer-gain button {
            width:100%;
            min-height:28px;
            padding:0 5px;
            color:#182020;
            background:linear-gradient(180deg,#d6dedb,#8f9b98);
            border-color:rgba(0,0,0,.28);
            cursor:ew-resize;
        }
        .iamccs-mixer-gain button:not(:disabled) {
            box-shadow:inset 0 1px 0 rgba(255,255,255,.62), 0 2px 5px rgba(0,0,0,.24);
        }
        .iamccs-mixer-route {
            display:grid;
            place-items:center;
            width:36px;
            height:28px;
            margin:0 auto;
            border-radius:3px;
            background:linear-gradient(135deg,#29b4a7 0 34%,#374349 34% 52%,#c6b15e 52% 100%);
            box-shadow:inset 0 0 0 1px rgba(0,0,0,.35);
        }
        .iamccs-mixer-body {
            display:grid;
            grid-template-columns: 36px 1fr 32px;
            gap:7px;
            align-items:end;
            min-height:230px;
        }
        .iamccs-mixer-scale {
            align-self:stretch;
            display:flex;
            flex-direction:column;
            justify-content:space-between;
            color:#85898c;
            font:900 10px/1 ui-monospace,SFMono-Regular,Consolas,monospace;
        }
        .iamccs-mixer-fader {
            align-self:stretch;
            display:grid;
            place-items:center;
            position:relative;
        }
        .iamccs-mixer-fader input[type="range"] {
            writing-mode: bt-lr;
            appearance: slider-vertical;
            width:32px;
            height:218px;
            accent-color:#c7cbca;
            cursor:ns-resize;
        }
        .iamccs-mixer-meter {
            align-self:stretch;
            position:relative;
            width:18px;
            overflow:hidden;
            background:#101413;
            border:1px solid rgba(0,0,0,.58);
            box-shadow:inset 0 0 0 1px rgba(255,255,255,.06);
        }
        .iamccs-mixer-meter i {
            position:absolute;
            left:0;
            right:0;
            bottom:0;
            height:0%;
            background:linear-gradient(180deg,#e25145 0 10%,#f0c755 10% 28%,#39f077 28% 100%);
            transition:height .08s linear;
        }
        .iamccs-mixer-label {
            display:flex;
            align-items:center;
            justify-content:center;
            min-height:30px;
            color:#2b2e31;
            background:#9aa3a1;
            border-top:1px solid rgba(255,255,255,.18);
            font:950 16px/1 ui-monospace,SFMono-Regular,Consolas,monospace;
        }
        .iamccs-mixer-strip.is-master .iamccs-mixer-label {
            color:#f0f2ef;
            background:#2b2e31;
        }
    `;
    document.head.appendChild(style);
}

function renderAudioBoardMixer(node) {
    if (node._iamccsAudioBoardMixerReady) return;
    node._iamccsAudioBoardMixerReady = true;
    ensureAudioBoardMixerStyles();
    const dataWidget = findWidget(node, "mixer_data");
    hideWidget(dataWidget);
    const root = document.createElement("div");
    root.className = "iamccs-audio-mixer";
    let mixerMeterRaf = 0;

    const parseLocal = () => {
        try { return JSON.parse(String(dataWidget?.value || "{}")); } catch { return {}; }
    };
    const writeLocal = (patch = {}) => {
        const data = { schema: "iamccs.audio_board_mixer", schema_version: 1, ...parseLocal(), ...patch };
        if (dataWidget) {
            dataWidget.value = JSON.stringify(data, null, 2);
            dataWidget.callback?.(dataWidget.value);
        }
        node.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas?.(true, true);
    };
    const upstreamArranger = () => {
        const queue = [node];
        const seen = new Set();
        while (queue.length) {
            const current = queue.shift();
            if (!current || seen.has(current.id)) continue;
            seen.add(current.id);
            if (current !== node && String(current?.comfyClass || current?.type || "") === "IAMCCS_AudioBoardArranger") return current;
            for (const input of current.inputs || []) {
                const link = input.link != null ? app.graph?.links?.[input.link] : null;
                const origin = link ? app.graph?.getNodeById?.(link.origin_id) : null;
                if (origin && !seen.has(origin.id)) queue.push(origin);
            }
        }
        return null;
    };
    const readArranger = () => {
        const arranger = upstreamArranger();
        const widget = arranger ? findWidget(arranger, "arranger_data") : null;
        try {
            const data = JSON.parse(String(widget?.value || "{}"));
            if (data && typeof data === "object") return { arranger, widget, data };
        } catch {}
        return { arranger: null, widget: null, data: {} };
    };
    const arrangerTransport = () => upstreamArranger()?._iamccsAudioBoardTransport || null;
    const fmtMixerTime = (frames, fpsValue = 24) => {
        const safeFps = Math.max(1, Number(fpsValue || 24) || 24);
        const total = Math.max(0, Number(frames || 0) / safeFps);
        const minutes = Math.floor(total / 60);
        const seconds = Math.floor(total % 60);
        const millis = Math.floor((total - Math.floor(total)) * 1000);
        return `${minutes}:${String(seconds).padStart(2, "0")}.${String(millis).padStart(3, "0")}`;
    };
    const ensureTrackSettings = (data) => {
        data.trackSettings = Array.isArray(data.trackSettings) ? data.trackSettings : [];
        const count = Math.max(1, Number(data.audioTrackCount || 1));
        for (let i = 0; i < count; i += 1) {
            data.trackSettings[i] = {
                name: `A${i + 1}`,
                mute: false,
                solo: false,
                volume: 1,
                gainDb: 0,
                pan: 0,
                bypassEffects: false,
                effectChain: [],
                ...(data.trackSettings[i] && typeof data.trackSettings[i] === "object" ? data.trackSettings[i] : {}),
            };
        }
        return data.trackSettings;
    };
    const syncArranger = (data, reason) => {
        const { arranger, widget } = readArranger();
        if (widget) {
            widget.value = JSON.stringify(data, null, 2);
            widget.callback?.(widget.value);
            const transportApi = arranger?._iamccsAudioBoardTransport || null;
            const isContinuousLiveControl = /^mixer_(pan|gain|volume|master_gain)/.test(String(reason || ""));
            const isPlaying = Boolean(transportApi?.snapshot?.()?.playing);
            if (!(isContinuousLiveControl && isPlaying)) {
                transportApi?.applyExternalState?.(data, `mixer_${reason}`);
            }
            arranger?.setDirtyCanvas?.(true, true);
            document.dispatchEvent(new CustomEvent("iamccs:audio_arranger_state_changed", {
                detail: { node_id: arranger?.id, reason: `mixer_${reason}`, selectedMixer: data.selectedMixer || null },
            }));
        }
        writeLocal({
            audioSegments: data.audioSegments || [],
            audioTrackCount: data.audioTrackCount || 1,
            trackSettings: data.trackSettings || [],
            mirrorTrackSettings: data.trackSettings || [],
            masterBus: data.masterBus || {},
            masterAudioGain: data.masterAudioGain ?? 1,
            loopEnabled: Boolean(data.loopEnabled),
            last_sync_reason: reason,
        });
    };
    const peakForTrack = (data, track) => {
        const segs = Array.isArray(data.audioSegments) ? data.audioSegments.filter((seg) => Number(seg.track || 0) === track && !seg.mute) : [];
        if (!segs.length) return 0;
        const raw = segs.reduce((sum, seg) => sum + Math.max(.08, Math.min(1, Number(seg.peak || seg.audioPeak || .38))), 0) / Math.max(1, segs.length);
        const settings = ensureTrackSettings(data)[track] || {};
        const gainLinear = Math.pow(10, Math.max(-24, Math.min(24, Number(settings.gainDb ?? 0) || 0)) / 20);
        return Math.max(0, Math.min(1, raw * Number(settings.volume ?? 1) * gainLinear));
    };
    const setKnob = (button, value, min, max) => {
        const t = (Math.max(min, Math.min(max, Number(value || 0))) - min) / Math.max(.0001, max - min);
        button.style.setProperty("--knob-angle", `${-135 + t * 270}deg`);
        button.style.setProperty("--knob-fill", `${Math.round(t * 78)}%`);
    };
    const attachDrag = (button, getter, setter, min, max, scale) => {
        button.onpointerdown = (event) => {
            event.preventDefault();
            const startX = event.clientX;
            const startY = event.clientY;
            const start = Number(getter() || 0);
            try { button.setPointerCapture?.(event.pointerId); } catch {}
            const move = (moveEvent) => {
                const value = Math.max(min, Math.min(max, start + ((moveEvent.clientX - startX) + (startY - moveEvent.clientY)) * scale));
                setter(value, false);
                moveEvent.preventDefault();
            };
            const up = (upEvent) => {
                window.removeEventListener("pointermove", move);
                window.removeEventListener("pointerup", up);
                try { button.releasePointerCapture?.(upEvent.pointerId); } catch {}
                setter(Number(getter() || 0), true);
            };
            window.addEventListener("pointermove", move, { passive: false });
            window.addEventListener("pointerup", up, { passive: false, once: true });
        };
    };
    const applyTrackLive = (track, key, value) => {
        const api = arrangerTransport();
        if (!api) return false;
        if (key === "volume") return api.setTrackVolume?.(track, value) === true;
        if (key === "gainDb") return api.setTrackGainDb?.(track, value) === true;
        if (key === "pan") return api.setTrackPan?.(track, value) === true;
        if (["mute", "solo", "bypassEffects", "reverb", "normalize", "lock"].includes(String(key || ""))) {
            return api.setTrackToggle?.(track, key, value) === true;
        }
        return false;
    };
    const applyMasterLive = (key, value) => {
        const api = arrangerTransport();
        if (!api) return false;
        if (key === "masterAudioGain") return api.setMasterGain?.(value) === true;
        if (["postFaderMeter", "meterMode"].includes(String(key || ""))) return true;
        return false;
    };
    function updateMixerLiveMeters() {
        const api = arrangerTransport();
        const snap = api?.snapshot?.();
        const meters = snap?.meters || {};
        const playing = Boolean(snap?.playing);
        const source = readArranger();
        const liveData = source.arranger ? source.data : { ...parseLocal() };
        const masterState = liveData.masterBus && typeof liveData.masterBus === "object" ? liveData.masterBus : {};
        const timeEl = root.querySelector(".iamccs-mixer-time");
        if (timeEl && snap) timeEl.textContent = `${fmtMixerTime(Math.max(0, Number(snap.playhead || 0)), snap.fps)} / ${fmtMixerTime(Math.max(1, Number(snap.totalFrames || 1)), snap.fps)}`;
        const stateEl = root.querySelector(".iamccs-mixer-state");
        if (stateEl) stateEl.textContent = api ? (playing ? "[playing]" : "[stopped]") : "[no arranger]";
        const playBtn = root.querySelector("[data-mixer-transport='play']");
        if (playBtn) {
            playBtn.classList.toggle("is-active", playing);
            playBtn.textContent = playing ? "Pause" : "Play";
        }
        root.querySelectorAll(".iamccs-mixer-meter i").forEach((fill) => {
            const key = fill.dataset.mixerMeter || "";
            let peak = Number(fill.dataset.staticPeak || 0);
            if (playing && key === "master") {
                peak = masterState.meterMode === "rms" ? Number(meters.master?.rms || 0) : Number(meters.master?.peak || 0);
                if (masterState.postFaderMeter !== false) peak *= Number(liveData.masterAudioGain ?? 1);
            }
            else if (playing && key !== "") peak = Number(meters.tracks?.[key]?.peak || 0);
            fill.style.height = `${Math.max(0, Math.min(100, Math.round(peak * 100)))}%`;
        });
        mixerMeterRaf = requestAnimationFrame(updateMixerLiveMeters);
    }
    const draw = () => {
        const source = readArranger();
        const data = source.data;
        const local = parseLocal();
        if (!source.arranger) {
            Object.assign(data, local);
            data.trackSettings = Array.isArray(local.trackSettings)
                ? local.trackSettings
                : (Array.isArray(local.mirrorTrackSettings) ? local.mirrorTrackSettings : data.trackSettings);
        } else if (!Array.isArray(data.audioSegments) && Array.isArray(local.audioSegments)) {
            Object.assign(data, local);
        }
        data.audioTrackCount = Math.max(1, Number(data.audioTrackCount || (Array.isArray(data.trackSettings) ? data.trackSettings.length : 4) || 4));
        data.masterBus = data.masterBus && typeof data.masterBus === "object" ? data.masterBus : {};
        data.masterBus.postFaderMeter = data.masterBus.postFaderMeter !== false;
        data.masterBus.meterMode = String(data.masterBus.meterMode || "peak").toLowerCase() === "rms" ? "rms" : "peak";
        const settings = ensureTrackSettings(data);
        root.innerHTML = "";
        const head = document.createElement("div");
        head.className = "iamccs-mixer-head";
        const title = document.createElement("div");
        title.innerHTML = `<div class="iamccs-mixer-title">IAMCCS AudioBoard Mixer</div><div class="iamccs-mixer-sub">${data.audioSegments?.length || 0} clips / ${data.audioTrackCount} channel strips / sync ${upstreamArranger() ? "Arranger" : "cine_linx"}</div>`;
        const refresh = document.createElement("button");
        refresh.textContent = "Sync From Arranger";
        refresh.onclick = () => draw();
        head.append(title, refresh);
        if (mixerMeterRaf) {
            cancelAnimationFrame(mixerMeterRaf);
            mixerMeterRaf = 0;
        }
        const transport = document.createElement("div");
        transport.className = "iamccs-mixer-transport";
        const makeTransportButton = (label, action, attr = "") => {
            const button = document.createElement("button");
            button.type = "button";
            button.textContent = label;
            if (attr) button.dataset.mixerTransport = attr;
            button.onclick = (event) => {
                event.preventDefault();
                event.stopPropagation();
                action();
                setTimeout(updateMixerLiveMeters, 0);
            };
            return button;
        };
        const rewind = makeTransportButton("|<", () => arrangerTransport()?.rewind?.(), "rewind");
        const play = makeTransportButton("Play", () => arrangerTransport()?.toggle?.(), "play");
        const loop = makeTransportButton("Loop", () => {
            data.loopEnabled = !data.loopEnabled;
            syncArranger(data, "mixer_loop_toggle");
            draw();
        }, "loop");
        loop.classList.toggle("is-active", Boolean(data.loopEnabled));
        const stop = makeTransportButton("Stop", () => arrangerTransport()?.stop?.(true), "stop");
        const time = document.createElement("span");
        time.className = "iamccs-mixer-time";
        time.textContent = `${fmtMixerTime(0, data.frame_rate)} / ${fmtMixerTime(Math.max(1, Math.round(Number(data.duration_seconds || 0) * Number(data.frame_rate || 24))), data.frame_rate)}`;
        const stateBadge = document.createElement("span");
        stateBadge.className = "iamccs-mixer-state";
        stateBadge.textContent = upstreamArranger() ? "[Mixer Sync]" : "[no arranger]";
        transport.append(rewind, play, loop, stop, time, stateBadge);
        const strips = document.createElement("div");
        strips.className = "iamccs-mixer-strips";
        const makeStrip = (index, isMaster = false) => {
            const strip = document.createElement("div");
            strip.className = `iamccs-mixer-strip${isMaster ? " is-master" : ""}`;
            const st = isMaster ? data.masterBus : settings[index];
            const label = isMaster ? "MASTER" : String(st.name || `A${index + 1}`);
            const pan = document.createElement("div");
            pan.className = "iamccs-mixer-pan";
            const panKnob = document.createElement("button");
            panKnob.className = "iamccs-mixer-knob";
            panKnob.type = "button";
            setKnob(panKnob, isMaster ? 0 : Number(st.pan || 0), -1, 1);
            const panText = document.createElement("span");
            const syncPanText = () => { const v = isMaster ? 0 : Number(st.pan || 0); panText.textContent = isMaster ? "stereo" : (Math.abs(v) < .02 ? "C" : `${v < 0 ? "L" : "R"}${Math.round(Math.abs(v) * 100)}`); setKnob(panKnob, v, -1, 1); };
            syncPanText();
            if (!isMaster) attachDrag(panKnob, () => st.pan || 0, (value, commit) => {
                st.pan = value;
                syncPanText();
                applyTrackLive(index, "pan", st.pan);
                if (commit) syncArranger(data, `mixer_pan_${index}`);
            }, -1, 1, .006);
            pan.append(panKnob, panText);
            const gainControl = document.createElement("div");
            gainControl.className = "iamccs-mixer-gain";
            const gainLabel = document.createElement("span");
            gainLabel.textContent = "GAIN";
            const gainValue = document.createElement("button");
            gainValue.type = "button";
            const syncGainText = () => {
                const value = isMaster
                    ? Math.max(-24, Math.min(6, 20 * Math.log10(Math.max(.063, Number(data.masterAudioGain ?? 1) || 1))))
                    : Math.max(-24, Math.min(24, Number(st.gainDb ?? 0) || 0));
                gainValue.textContent = `${value >= 0 ? "+" : ""}${value.toFixed(1)} dB`;
            };
            syncGainText();
            if (!isMaster) {
                attachDrag(gainValue, () => st.gainDb || 0, (value, commit) => {
                    st.gainDb = Number(value.toFixed(1));
                    syncGainText();
                    applyTrackLive(index, "gainDb", st.gainDb);
                    if (commit) syncArranger(data, `mixer_gain_${index}`);
                }, -24, 24, .08);
            } else {
                attachDrag(gainValue, () => 20 * Math.log10(Math.max(.063, Number(data.masterAudioGain ?? 1) || 1)), (value, commit) => {
                    data.masterAudioGain = Math.max(.063, Math.min(2, Math.pow(10, value / 20)));
                    syncGainText();
                    applyMasterLive("masterAudioGain", data.masterAudioGain);
                    if (commit) syncArranger(data, "mixer_master_gain_db");
                }, -24, 6, .08);
            }
            gainControl.append(gainLabel, gainValue);
            const actions = document.createElement("div");
            actions.className = "iamccs-mixer-actions";
            if (isMaster) {
                const post = document.createElement("button");
                post.textContent = "POST";
                post.title = "Post-fader master meter";
                post.classList.toggle("is-active", Boolean(st.postFaderMeter));
                post.setAttribute("aria-pressed", String(Boolean(st.postFaderMeter)));
                post.onclick = () => {
                    st.postFaderMeter = !st.postFaderMeter;
                    applyMasterLive("postFaderMeter", st.postFaderMeter);
                    syncArranger(data, "mixer_master_post_fader");
                    draw();
                };
                const rms = document.createElement("button");
                rms.textContent = "RMS";
                rms.classList.toggle("is-active", st.meterMode === "rms");
                rms.setAttribute("aria-pressed", String(st.meterMode === "rms"));
                rms.onclick = () => {
                    st.meterMode = "rms";
                    applyMasterLive("meterMode", st.meterMode);
                    syncArranger(data, "mixer_master_meter_rms");
                    draw();
                };
                const peak = document.createElement("button");
                peak.textContent = "PK";
                peak.classList.toggle("is-active", st.meterMode === "peak");
                peak.setAttribute("aria-pressed", String(st.meterMode === "peak"));
                peak.onclick = () => {
                    st.meterMode = "peak";
                    applyMasterLive("meterMode", st.meterMode);
                    syncArranger(data, "mixer_master_meter_peak");
                    draw();
                };
                actions.append(post, rms, peak);
            } else {
                const mute = document.createElement("button");
                mute.textContent = "M";
                mute.className = st.mute ? "is-active" : "";
                mute.setAttribute("aria-pressed", String(Boolean(st.mute)));
                mute.onclick = () => { st.mute = !st.mute; applyTrackLive(index, "mute", st.mute); syncArranger(data, `mixer_mute_${index}`); draw(); };
                const solo = document.createElement("button");
                solo.textContent = "S";
                solo.className = `solo${st.solo ? " is-active" : ""}`;
                solo.setAttribute("aria-pressed", String(Boolean(st.solo)));
                solo.onclick = () => { st.solo = !st.solo; applyTrackLive(index, "solo", st.solo); syncArranger(data, `mixer_solo_${index}`); draw(); };
                const dry = document.createElement("button");
                dry.textContent = "DRY";
                dry.title = "Bypass channel effects. Gain, fader, pan and normalize remain active.";
                dry.className = `dry${st.bypassEffects ? " is-active" : ""}`;
                dry.setAttribute("aria-pressed", String(Boolean(st.bypassEffects)));
                dry.onclick = () => { st.bypassEffects = !st.bypassEffects; applyTrackLive(index, "bypassEffects", st.bypassEffects); syncArranger(data, `mixer_dry_${index}`); draw(); };
                actions.append(mute, solo, dry);
            }
            const body = document.createElement("div");
            body.className = "iamccs-mixer-body";
            const scale = document.createElement("div");
            scale.className = "iamccs-mixer-scale";
            ["0", "-6", "-18", "-30", "-42", "-54"].forEach((item) => {
                const tick = document.createElement("span");
                tick.textContent = item;
                scale.appendChild(tick);
            });
            const fader = document.createElement("div");
            fader.className = "iamccs-mixer-fader";
            const slider = document.createElement("input");
            slider.type = "range";
            slider.min = "0";
            slider.max = "2";
            slider.step = "0.01";
            slider.value = String(isMaster ? Number(data.masterAudioGain ?? 1) : Number(st.volume ?? 1));
            slider.oninput = () => {
                if (isMaster) data.masterAudioGain = Number(slider.value || 1);
                else st.volume = Number(slider.value || 1);
                if (isMaster) applyMasterLive("masterAudioGain", data.masterAudioGain);
                else applyTrackLive(index, "volume", st.volume);
                meterFill.style.height = `${Math.round((isMaster ? .45 : peakForTrack(data, index)) * 100)}%`;
            };
            slider.onchange = () => syncArranger(data, isMaster ? "mixer_master_gain" : `mixer_volume_${index}`);
            fader.appendChild(slider);
            const meter = document.createElement("div");
            meter.className = "iamccs-mixer-meter";
            const meterFill = document.createElement("i");
            const staticPeak = isMaster ? Math.min(1, Array.from({ length: data.audioTrackCount }, (_, i) => peakForTrack(data, i)).reduce((a, b) => a + b, 0) / Math.max(1, data.audioTrackCount)) : peakForTrack(data, index);
            meterFill.dataset.mixerMeter = isMaster ? "master" : String(index);
            const displayPeak = isMaster && st.postFaderMeter ? Math.min(1, staticPeak * Number(data.masterAudioGain ?? 1)) : staticPeak;
            meterFill.dataset.staticPeak = String(displayPeak);
            meterFill.style.height = `${Math.round(displayPeak * 100)}%`;
            meter.appendChild(meterFill);
            body.append(scale, fader, meter);
            const route = document.createElement("div");
            route.className = "iamccs-mixer-route";
            route.title = "Route";
            const name = document.createElement("div");
            name.className = "iamccs-mixer-label";
            name.textContent = label.replace(/^A/, "");
            strip.append(gainControl, pan, actions, body, route, name);
            return strip;
        };
        strips.appendChild(makeStrip(0, true));
        for (let i = 0; i < data.audioTrackCount; i += 1) strips.appendChild(makeStrip(i, false));
        root.append(head, transport, strips);
        updateMixerLiveMeters();
        writeLocal({ audioSegments: data.audioSegments || [], audioTrackCount: data.audioTrackCount, mirrorTrackSettings: data.trackSettings || [], masterBus: data.masterBus || {}, masterAudioGain: data.masterAudioGain ?? 1 });
    };
    draw();
    const domWidget = node.addDOMWidget("AudioBoardMixer", "iamccs_audio_board_mixer", root, { serialize: false });
    domWidget.computeSize = (width) => [Math.max(1320, Number(width || 1320)), 700];
    const enforceMixerSize = () => {
        const next = [Math.max(1340, Number(node.size?.[0] || 0)), Math.max(780, Number(node.size?.[1] || 0))];
        node.min_size = [1340, 780];
        if (Number(node.size?.[0] || 0) < next[0] || Number(node.size?.[1] || 0) < next[1]) {
            if (typeof node.setSize === "function") node.setSize(next);
            else node.size = next;
        }
    };
    if (!node._iamccsAudioMixerResizeGuard) {
        const previousMixerResize = node.onResize;
        node.onResize = function(size) {
            if (Array.isArray(size)) {
                size[0] = Math.max(1340, Number(size[0] || 0));
                size[1] = Math.max(780, Number(size[1] || 0));
            }
            const result = typeof previousMixerResize === "function" ? previousMixerResize.apply(this, arguments) : undefined;
            enforceMixerSize();
            return result;
        };
        node._iamccsAudioMixerResizeGuard = true;
    }
    requestAnimationFrame(enforceMixerSize);
    setTimeout(enforceMixerSize, 600);
}

app.registerExtension({
    name: "IAMCCS.CineAudioDialogueUI",
    setup() {
        [500, 1500, 3500].forEach((delay) => {
            setTimeout(() => {
                const nodes = Array.isArray(app?.graph?._nodes) ? app.graph._nodes : [];
                console.info("[IAMCCS AudioDialogueUI] graph scan", {
                    delay,
                    nodes: nodes.length,
                    iamccsAudioCandidates: nodes
                        .filter((node) => iamccsAudioDialogueType(node))
                        .map((node) => ({ id: node?.id, type: node?.type, detected: iamccsAudioDialogueType(node), widgets: (node?.widgets || []).map((w) => w?.name) })),
                });
                nodes.forEach((node) => renderIamccsAudioDialogueNode(node, `extension.graphScan+${delay}`));
            }, delay);
        });
    },
    nodeCreated(node) {
        [0, 120, 600].forEach((delay) => setTimeout(() => renderIamccsAudioDialogueNode(node, `extension.nodeCreated+${delay}`), delay));
    },
    loadedGraphNode(node) {
        [0, 120, 600].forEach((delay) => setTimeout(() => renderIamccsAudioDialogueNode(node, `extension.loadedGraphNode+${delay}`), delay));
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (
            nodeData.name !== "IAMCCS_CineEmotionButtons" &&
            nodeData.name !== "IAMCCS_BoardMaker_DialogueFoley" &&
            nodeData.name !== "IAMCCS_AudioBoardArranger" &&
            nodeData.name !== "IAMCCS_ControlAudEfx" &&
            nodeData.name !== "IAMCCS_ControlAudEfxPanel" &&
            nodeData.name !== "IAMCCS_AudioBoardMixer" &&
            nodeData.name !== "IAMCCS_CineSpeech1PromptCompiler"
        ) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            console.info("[IAMCCS AudioDialogueUI] node created", { name: nodeData.name, id: this?.id, type: this?.type });
            renderIamccsAudioDialogueNode(this, "prototype.onNodeCreated");
        };

        if (nodeData.name === "IAMCCS_AudioBoardArranger") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                try {
                    const audioBoardJson = firstUiValue(message, "iamccs_audio_board");
                    if (!audioBoardJson) return;
                    const applied = typeof this._iamccsAudioBoardApplyRuntimeTimeline === "function"
                        ? this._iamccsAudioBoardApplyRuntimeTimeline(audioBoardJson, "onExecuted")
                        : setWidgetValue(this, "arranger_data", audioBoardJson);
                    if (applied) {
                        this.properties = this.properties || {};
                        this.properties.iamccsAudioBoardRuntimeSyncAt = new Date().toISOString();
                        renderIamccsAudioDialogueNode(this, "prototype.onExecuted");
                    }
                } catch (err) {
                    console.warn("[IAMCCS AudioBoardArranger] runtime UI sync failed", err);
                }
            };
        }
    },
});
