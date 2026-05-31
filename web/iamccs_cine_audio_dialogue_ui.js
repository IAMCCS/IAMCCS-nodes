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
            height: 1150px;
            min-height: 1150px;
            max-height: 1150px;
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
        .iamccs-audio-board.is-fullscreen {
            min-height: calc(100vh - 96px);
            height: calc(100vh - 96px);
            border: 0;
            box-shadow: none;
        }
        .iamccs-audio-board.is-fullscreen .iamccs-audio-board-dynamic {
            height: calc(100vh - 118px);
            display: flex;
            flex-direction: column;
            min-width: 0;
            overflow: hidden;
        }
        .iamccs-audio-board-dynamic {
            height: 100%;
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
        .iamccs-audio-board-head { justify-content: space-between; margin-bottom: 7px; }
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
            height: 360px;
            overflow: auto;
            min-width: 0;
            contain: strict;
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
            flex: 0 0 min(38vh, 410px);
            height: min(38vh, 410px);
        }
        .iamccs-audio-board-track {
            position: relative;
            height: var(--iamccs-track-height, 72px);
            border-bottom: 2px solid rgba(255,255,255,.10);
            background-image:
                linear-gradient(180deg, rgba(255,255,255,.018), rgba(0,0,0,.10)),
                linear-gradient(90deg, rgba(244,212,158,.10) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px);
            background-size: 100% 100%, calc(var(--iamccs-px-per-frame, 2px) * 24) 100%, calc(var(--iamccs-px-per-frame, 2px) * 6) 100%;
            background-position: 0 0, 224px 0, 224px 0;
        }
        .iamccs-audio-board-track:nth-child(even) { background-color: rgba(255,255,255,.012); }
        .iamccs-audio-board-track-label {
            position: sticky;
            left: 0;
            z-index: 7;
            width: 224px;
            height: var(--iamccs-track-height, 144px);
            display: grid;
            grid-template-rows: 21px 26px 40px 25px;
            align-content: start;
            gap: 4px;
            padding: 8px 11px;
            color: #f3e2c0;
            background:
                linear-gradient(90deg, rgba(244,212,158,.12), transparent 34%),
                linear-gradient(180deg, rgba(31,39,37,.99), rgba(9,13,15,.99));
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
            outline: 1px solid rgba(255,218,151,.30);
        }
        .iamccs-track-strip-top,
        .iamccs-track-strip-controls,
        .iamccs-track-strip-bottom {
            display: flex;
            align-items: center;
            gap: 5px;
            min-width: 0;
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
            grid-template-columns: repeat(6, 1fr);
            gap: 4px;
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
        }
        .iamccs-track-mini.is-active {
            color: #161109;
            border-color: #ffe0a4;
            background: linear-gradient(180deg, #f3d99b, #b98343);
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
            height: 25px;
            min-height: 25px;
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
            height: 9px;
            border: 1px solid rgba(255,255,255,.18);
            border-radius: 999px;
            overflow: hidden;
            background: linear-gradient(90deg, rgba(255,255,255,.06), rgba(0,0,0,.45));
        }
        .iamccs-audio-board-meter { width: 50px; }
        .iamccs-master-meter { width: 180px; height: 12px; }
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
            height: 100%;
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
            flex: 0 0 390px;
            height: 390px;
            min-height: 0;
            min-width: 0;
            overflow: hidden;
        }
        .iamccs-audio-board-lower.no-monitor { grid-template-columns: 1fr; }
        .iamccs-audio-board.is-fullscreen .iamccs-audio-board-lower {
            flex: 1 1 auto;
            height: auto;
            min-height: 0;
            overflow: hidden;
        }
        .iamccs-event-console,
        .iamccs-inline-efx {
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
            height: calc(100% - 24px);
            overflow: hidden;
        }
        .iamccs-device-chain {
            min-width: 0;
            width: 100%;
            height: 100%;
            display: flex;
            gap: 7px;
            overflow-x: auto;
            overflow-y: hidden;
            padding-bottom: 8px;
            align-items: stretch;
            contain: layout paint;
        }
        .iamccs-device-module {
            flex: 0 0 780px;
            min-width: 780px;
            display: flex;
            gap: 9px;
            align-items: stretch;
        }
        .iamccs-audio-device {
            flex: 0 0 330px;
            min-height: 340px;
            padding: 9px;
            border: 1px solid rgba(244,212,158,.24);
            border-radius: 7px;
            background:
                radial-gradient(circle at 26px 20px, rgba(255,224,164,.20), transparent 22px),
                linear-gradient(180deg, var(--device-hi, #5a3c29), var(--device-mid, #2c261f) 56%, var(--device-lo, #0b0d0c));
            box-shadow: inset 0 1px 0 rgba(255,255,255,.08), 0 5px 14px rgba(0,0,0,.24);
            color: #f7e6be;
        }
        .iamccs-audio-device.iamccs-device-eq {
            flex-basis: 350px;
        }
        .iamccs-audio-device.iamccs-device-eq .iamccs-device-knobs {
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 6px;
        }
        .iamccs-audio-device.iamccs-device-eq .iamccs-device-knob i {
            width: 34px;
            height: 34px;
        }
        .iamccs-audio-device.iamccs-device-eq .iamccs-device-knob input[type="range"] {
            width: 58px;
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
            gap: 7px 9px;
        }
        .iamccs-device-knob {
            display: grid;
            justify-items: center;
            gap: 3px;
            color: #c9b890;
            font: 800 8px/1 ui-monospace, SFMono-Regular, Consolas, monospace;
            text-transform: uppercase;
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
            flex: 0 0 435px;
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
            box-shadow: inset 0 0 0 1px rgba(244,212,158,.38), 0 0 16px rgba(244,212,158,.12);
            background: linear-gradient(180deg, rgba(65,52,35,.99), rgba(13,15,15,.99));
        }
        @media (max-width: 980px) {
            .iamccs-audio-board-lower { grid-template-columns: 1fr; }
            .iamccs-inline-efx-grid { grid-template-columns: 1fr; }
        }
    `;
    document.head.appendChild(style);
}

const IAMCCS_AUDIO_BOARD_FIXED_WIDTH = 1880;
const IAMCCS_AUDIO_BOARD_FIXED_HEIGHT = 1290;

function renderAudioBoardArranger(node) {
    ensureAudioBoardArrangerStyles();
    if (node._iamccsAudioBoardReady) {
        const fixed = [IAMCCS_AUDIO_BOARD_FIXED_WIDTH, IAMCCS_AUDIO_BOARD_FIXED_HEIGHT];
        node._iamccsAudioBoardFixedSize = fixed;
        node.resizable = false;
        node.resizeable = false;
        node.flags = { ...(node.flags || {}), resizable: false };
        node.min_size = fixed.slice();
        const domWidget = (node.widgets || []).find((w) => w?.type === "iamccs_audio_board_arranger" || w?.name === "AudioBoard Arranger");
        const existingRoot = domWidget?.element || domWidget?.inputEl || null;
        if (domWidget) {
            domWidget.computeSize = () => existingRoot?._iamccsAudioFullscreenState
                ? [IAMCCS_AUDIO_BOARD_FIXED_WIDTH, 24]
                : [IAMCCS_AUDIO_BOARD_FIXED_WIDTH, IAMCCS_AUDIO_BOARD_FIXED_HEIGHT - 70];
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
    fileInput.accept = "audio/*";
    fileInput.multiple = true;
    fileInput.style.display = "none";
    root.appendChild(fileInput);

    const LABEL_W = 224;
    const VIEWPORT_SECONDS = 26;
    const DEFAULT_SECONDS = VIEWPORT_SECONDS;
    const AUDIO_BOARD_FIXED_WIDTH = IAMCCS_AUDIO_BOARD_FIXED_WIDTH;
    const AUDIO_BOARD_FIXED_HEIGHT = IAMCCS_AUDIO_BOARD_FIXED_HEIGHT;
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
    const TRACK_COLORS = ["#315f8f", "#2f7f71", "#7a5a34", "#6b4c80", "#8a4b45", "#556b36", "#3c6478", "#7a6738"];
    const normalizeTrackColor = (color, index = 0) => /^#[0-9a-f]{6}$/i.test(String(color || "")) ? String(color) : TRACK_COLORS[index % TRACK_COLORS.length];
    const nextTrackColor = (color, index = 0) => {
        const current = normalizeTrackColor(color, index);
        const pos = TRACK_COLORS.findIndex((item) => item.toLowerCase() === current.toLowerCase());
        return TRACK_COLORS[((pos >= 0 ? pos : index) + 1) % TRACK_COLORS.length];
    };
    const DEFAULT_MASTER_CHAIN = [
        { id: "master_eq", type: "eq", enabled: true, amount: .5, params: { low: 0, mid: 0, high: 0, q: 1.2 } },
        { id: "master_comp", type: "compressor", enabled: true, amount: .45, params: { threshold: -18, ratio: 4, attack: 6, release: 180, knee: 12, makeup: 0 } },
        { id: "master_limit", type: "limiter", enabled: true, amount: 1, params: { input: 0, ceiling: -1, lookahead: 3, release: 120, output: 0, softclip: .25 } },
    ];
    const audioBuffers = new Map();
    const audioUrls = new Map();
    const waveformLoading = new Set();
    let selectedId = "";
    let audioContext = null;
    let shotboardSyncTimer = 0;
    let liveStateWriteTimer = 0;
    const shotboardSyncSignatures = new Map();
    let transport = {
        playing: false,
        playhead: 0,
        startedAt: 0,
        sources: [],
        analysers: new Map(),
        masterAnalyser: null,
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
    const parseState = () => {
        const fallback = {
            schema: "iamccs.audio_board_arranger",
            schema_version: 1,
            audioSegments: [],
            audioTrackCount: 4,
            masterAudioGain: 1,
            masterAudioNormalize: false,
            masterBus: { limiter: true, ceilingDb: -1, compressor: .45, width: 1, reverbSend: 0, delaySend: 0, effectChain: JSON.parse(JSON.stringify(DEFAULT_MASTER_CHAIN)) },
            duration_seconds: DEFAULT_SECONDS,
            frame_rate: Number(fpsWidget?.value || 24),
            status: { edits: [] },
            view: { timeZoom: 1, trackHeight: 144, tool: "cursor", visibleSeconds: VIEWPORT_SECONDS },
            showEventMonitor: false,
            showClipValues: false,
            showMultiGeneration: false,
            trackSettings: [],
            audioBusMode: "all_tracks",
            onlyFirstTrack: false,
            loopEnabled: false,
            loopInFrame: 0,
            loopOutFrame: 0,
            selectedMixer: { type: "master", track: 0 },
        };
        try {
            const data = JSON.parse(String(dataWidget?.value || ""));
            const out = { ...fallback, ...(data && typeof data === "object" ? data : {}) };
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
            out.audioTrackCount = Math.max(1, Number(out.audioTrackCount || 4));
            out.masterBus = { ...fallback.masterBus, ...(out.masterBus && typeof out.masterBus === "object" ? out.masterBus : {}) };
            out.masterBus.effectChain = Array.isArray(out.masterBus.effectChain)
                ? out.masterBus.effectChain
                : JSON.parse(JSON.stringify(DEFAULT_MASTER_CHAIN));
            out.trackSettings = Array.isArray(out.trackSettings) ? out.trackSettings : [];
            out.audioBusMode = (out.audioBusMode === "only_first" || out.onlyFirstTrack) ? "only_first" : "all_tracks";
            out.onlyFirstTrack = out.audioBusMode === "only_first";
            out.loopEnabled = Boolean(out.loopEnabled);
            out.loopInFrame = Math.max(0, Math.round(Number(out.loopInFrame || 0)));
            out.loopOutFrame = Math.max(0, Math.round(Number(out.loopOutFrame || 0)));
            out.selectedMixer = out.selectedMixer && typeof out.selectedMixer === "object" ? out.selectedMixer : fallback.selectedMixer;
            out.showEventMonitor = Boolean(out.showEventMonitor);
            out.showClipValues = Boolean(out.showClipValues);
            out.showMultiGeneration = Boolean(out.showMultiGeneration);
            return out;
        } catch {
            return fallback;
        }
    };
    let state = parseState();
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
            pan: 0,
            color: normalizeTrackColor("", index),
            lock: false,
            effectChain: [{ id: `track_${index}_eq`, type: "eq", enabled: true, amount: .5, params: { low: 0, mid: 0, high: 0, q: 1.2 } }],
            ...(state.trackSettings[index] && typeof state.trackSettings[index] === "object" ? state.trackSettings[index] : {}),
        };
        state.trackSettings[index].effectChain = Array.isArray(state.trackSettings[index].effectChain)
            ? state.trackSettings[index].effectChain
            : [{ id: `track_${index}_eq`, type: "eq", enabled: true, amount: .5, params: { low: 0, mid: 0, high: 0, q: 1.2 } }];
        return state.trackSettings[index];
    };
    const selectedMixer = () => {
        state.selectedMixer = state.selectedMixer && typeof state.selectedMixer === "object" ? state.selectedMixer : { type: "master", track: 0 };
        state.selectedMixer.type = state.selectedMixer.type === "track" ? "track" : "master";
        state.selectedMixer.track = Math.max(0, Number(state.selectedMixer.track || 0));
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
            out.params.lowCutFreq = Math.max(20, Math.min(400, Number(out.params.lowCutFreq || 80)));
            out.params.highCutFreq = Math.max(2000, Math.min(22000, Number(out.params.highCutFreq || 12000)));
            out.params.lowCutLevel = Math.max(-48, Math.min(0, Number(out.params.lowCutLevel ?? -30)));
            out.params.highCutLevel = Math.max(-48, Math.min(0, Number(out.params.highCutLevel ?? -30)));
        }
        out.amount = Math.max(0, Math.min(1, Number(out.amount ?? .5)));
        return out;
    };
    const chainForTarget = () => {
        const target = selectedMixer();
        if (target.type === "track") {
            const chain = trackSettings(target.track).effectChain;
            if (!trackSettings(target.track).noAutoEq && !chain.some((fx) => fx.type === "eq")) chain.unshift({ id: `track_${target.track}_eq`, type: "eq", enabled: true, amount: .5, params: { low: 0, mid: 0, high: 0, q: 1.2, lowCut: false, highCut: false } });
            chain.forEach(normalizeEffect);
            return chain;
        }
        state.masterBus = state.masterBus && typeof state.masterBus === "object" ? state.masterBus : {};
        state.masterBus.effectChain = Array.isArray(state.masterBus.effectChain)
            ? state.masterBus.effectChain
            : JSON.parse(JSON.stringify(DEFAULT_MASTER_CHAIN));
        if (!state.masterBus.noAutoEq && !state.masterBus.effectChain.some((fx) => fx.type === "eq")) state.masterBus.effectChain.unshift({ id: "master_eq", type: "eq", enabled: true, amount: .5, params: { low: 0, mid: 0, high: 0, q: 1.2, lowCut: false, highCut: false } });
        state.masterBus.effectChain.forEach(normalizeEffect);
        return state.masterBus.effectChain;
    };
    const addEffectToChain = (type, target = selectedMixer()) => {
        const cleanType = String(type || "").trim();
        if (!cleanType) return;
        const chain = target.type === "track" ? trackSettings(target.track).effectChain : chainForTarget();
        chain.push(normalizeEffect({ id: newId(`fx_${cleanType}`), type: cleanType, enabled: true, amount: cleanType === "limiter" ? 1 : .5 }));
        if (target.type === "master") {
            if (cleanType === "compressor") state.masterBus.compressor = Math.max(.45, Number(state.masterBus.compressor || 0));
            if (cleanType === "limiter") state.masterBus.limiter = true;
        }
        addEdit(`Inserted ${cleanType} on ${target.type === "track" ? `A${target.track + 1}` : "Master"}.`);
        writeState("insert_effect");
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
        addEdit(`Removed ${String(removed?.type || "device")} from ${target.type === "track" ? `A${target.track + 1}` : "Master"}.`);
        writeState("remove_effect");
        draw();
    };
    const hasMedia = (seg) => Boolean(seg && (String(seg.audioFile || "").trim() || String(seg.audioB64 || "").trim()));
    const effectiveGain = (seg) => Math.max(0, Math.min(4, Number(seg?.gain ?? 1) || 1));
    const peakFor = (seg) => {
        const peaks = Array.isArray(seg?.waveformPeaks) ? seg.waveformPeaks.map((item) => {
            if (item && typeof item === "object") return Math.max(Math.abs(Number(item.min) || 0), Math.abs(Number(item.max) || 0));
            return Math.abs(Number(item) || 0);
        }) : [];
        return peaks.length ? Math.min(1, Math.max(...peaks) * effectiveGain(seg)) : 0;
    };
    const linkedShotboardNodes = () => {
        const out = [];
        for (const output of node.outputs || []) {
            for (const linkId of output.links || []) {
                const link = app.graph?.links?.[linkId];
                const target = link ? app.graph?.getNodeById?.(link.target_id) : null;
                const type = String(target?.comfyClass || target?.type || "");
                if (target && type === "IAMCCS_CineShotboardPlannerV3") out.push(target);
            }
        }
        return out;
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
        return Math.max(secondsToFrames(VIEWPORT_SECONDS), shotboardDurationFrames(), Math.ceil(end));
    };
    const view = () => {
        state.view = state.view && typeof state.view === "object" ? state.view : {};
        state.view.timeZoom = Math.max(0.35, Math.min(8, Number(state.view.timeZoom || 1)));
        state.view.trackHeight = Math.max(144, Math.min(200, Number(state.view.trackHeight || 144)));
        state.view.tool = String(state.view.tool || "cursor");
        if (!["cursor", "move", "trim", "cut"].includes(state.view.tool)) state.view.tool = "cursor";
        state.view.visibleSeconds = VIEWPORT_SECONDS;
        return state.view;
    };
    const visibleTimelineWidth = () => Math.max(820, Math.round((root.clientWidth || 1280) - LABEL_W - 48));
    const pxPerFrame = () => Math.max(0.08, (visibleTimelineWidth() / Math.max(1, secondsToFrames(VIEWPORT_SECONDS))) * view().timeZoom);
    const contentWidth = () => LABEL_W + Math.max(visibleTimelineWidth(), Math.ceil(totalFrames() * pxPerFrame())) + 8;
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
            const fill = meter.querySelector("i");
            if (!fill) return;
            if (!trackBars.has(track)) trackBars.set(track, []);
            trackBars.get(track).push(fill);
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
        for (const board of boards) {
            const widget = findWidget(board, "timeline_data");
            if (!widget) continue;
            let data = {};
            try { data = JSON.parse(String(widget.value || "{}")); } catch { data = {}; }
            if (!data || typeof data !== "object") data = {};
            data.schema = data.schema || "iamccs.cine.filmmaker_timeline";
            data.schema_version = Math.max(2, Number(data.schema_version || 2));
            const allAudioSegments = JSON.parse(JSON.stringify(segments()));
            const shotboardOnlyFirst = state.audioBusMode === "only_first" || state.onlyFirstTrack;
            let nextShotboardAudioSegments = shotboardOnlyFirst
                ? allAudioSegments.filter((seg) => Number(seg.track || 0) === 0)
                : allAudioSegments;
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
            if (shotboardSyncSignatures.get(cacheKey) === syncSignature && reason !== "manual_sync") continue;
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
        if (dataWidget) {
            dataWidget.value = JSON.stringify(state, null, 2);
            if (!options.quiet) dataWidget.callback?.(dataWidget.value);
        }
        if (sync) syncToShotboard(reason);
        if (!options.quiet) {
            markCanvasDirty(false);
        }
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
    const ensureAudioContext = async () => {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextClass) throw new Error("WebAudio unavailable");
        if (!audioContext) audioContext = new AudioContextClass();
        if (audioContext.state === "suspended") await audioContext.resume();
        return audioContext;
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
    const getBuffer = async (seg) => {
        if (!seg) return null;
        if (audioBuffers.has(seg.id)) return audioBuffers.get(seg.id);
        const ctx = await ensureAudioContext();
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
        if (!seg || !hasMedia(seg) || seg.waveformReal === true || waveformLoading.has(seg.id)) return;
        waveformLoading.add(seg.id);
        getBuffer(seg)
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
        return { durationFrames: Math.max(1, Math.round(decoded.duration * fps())), peaks, buffer: decoded };
    };
    const importFiles = async (files) => {
        const list = Array.from(files || []).filter((file) => String(file.type || "").startsWith("audio/"));
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
                addEdit(`Import failed: ${file.name}`);
                console.warn("[IAMCCS AudioBoardArranger] import failed", err);
            }
        }
        writeState("import");
        draw();
    };
    fileInput.onchange = async (event) => {
        await importFiles(event.target.files || []);
        fileInput.value = "";
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
        transport.masterAnalyser = null;
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
            const freqToX = (hz) => {
                const t = (Math.log10(Math.max(20, Math.min(22000, Number(hz || 20)))) - Math.log10(20)) / (Math.log10(22000) - Math.log10(20));
                return Math.max(0, Math.min(w, t * w));
            };
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
            const lowX = freqToX(140);
            const midX = freqToX(1200);
            const highX = freqToX(6200);
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
            ctx.fillStyle = "#ffe2a8";
            ctx.font = "900 9px ui-monospace, Consolas, monospace";
            ctx.fillText(`LOW ${low.toFixed(1)}  MID ${midDb.toFixed(1)}  HIGH ${high.toFixed(1)}`, 8, 14);
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
        const soloed = segments().some((seg) => seg.solo);
        const trackSoloed = Array.from({ length: Math.max(1, Number(state.audioTrackCount || 4)) }, (_, index) => trackSettings(index)).some((item) => item.solo);
        const range = loopRange();
        if (state.loopEnabled && range && (transport.playhead < range.start || transport.playhead >= range.end)) {
            setPlayhead(range.start, false);
        }
        const masterGain = ctx.createGain();
        masterGain.gain.value = Math.max(0, Math.min(2, Number(state.masterAudioGain ?? 1) || 1));
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
            if (masterEqFx.params?.lowCut) addMasterBiquad("highpass", masterEqFx.params.lowCutFreq || 80);
            if (Number(masterEqFx.params?.low || 0)) addMasterBiquad("lowshelf", 140, masterEqFx.params.low, .7);
            if (Number(masterEqFx.params?.mid || 0)) addMasterBiquad("peaking", 1200, masterEqFx.params.mid, Math.max(.2, Number(masterEqFx.params?.q || 1.2)));
            if (Number(masterEqFx.params?.high || 0)) addMasterBiquad("highshelf", 6200, masterEqFx.params.high, .7);
            if (masterEqFx.params?.highCut) addMasterBiquad("lowpass", masterEqFx.params.highCutFreq || 12000);
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
        masterLast.connect(masterAnalyser);
        masterAnalyser.connect(ctx.destination);
        const trackNodes = new Map();
        const getTrack = (track) => {
            if (trackNodes.has(track)) return trackNodes.get(track);
            const trackState = trackSettings(track);
            const gain = ctx.createGain();
            gain.gain.value = Math.max(0, Math.min(2, Number(trackState.volume ?? 1)));
            let last = gain;
            if (ctx.createStereoPanner) {
                const trackPan = ctx.createStereoPanner();
                trackPan.pan.value = Math.max(-1, Math.min(1, Number(trackState.pan || 0)));
                last.connect(trackPan);
                last = trackPan;
            }
            const analyser = ctx.createAnalyser();
            analyser.fftSize = 256;
            last.connect(analyser);
            analyser.connect(masterGain);
            trackNodes.set(track, gain);
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
            if (trackState.mute || (trackSoloed && !trackState.solo) || trackState.lock) continue;
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
                if (Number(seg.hpfHz || 0) > 10) addBiquad("highpass", seg.hpfHz, 0, .7);
                if (Number(seg.lpfHz || 22000) < 21950) addBiquad("lowpass", seg.lpfHz, 0, .7);
                if (Number(seg.eqLowDb || 0)) addBiquad("lowshelf", 140, seg.eqLowDb, .7);
                if (Number(seg.eqMidDb || 0)) addBiquad("peaking", 1200, seg.eqMidDb, 1.1);
                if (Number(seg.eqHighDb || 0)) addBiquad("highshelf", 6200, seg.eqHighDb, .7);
                const trackEq = trackState.effectChain.find((fx) => fx.enabled !== false && fx.type === "eq");
                if (trackEq) {
                    if (trackEq.params?.lowCut) addBiquad("highpass", trackEq.params.lowCutFreq || 80, 0, .7);
                    if (Number(trackEq.params?.low || 0)) addBiquad("lowshelf", 140, trackEq.params.low, .7);
                    if (Number(trackEq.params?.mid || 0)) addBiquad("peaking", 1200, trackEq.params.mid, Math.max(.2, Number(trackEq.params?.q || 1.2)));
                    if (Number(trackEq.params?.high || 0)) addBiquad("highshelf", 6200, trackEq.params.high, .7);
                    if (trackEq.params?.highCut) addBiquad("lowpass", trackEq.params.highCutFreq || 12000, 0, .7);
                }
                if (Number(seg.compressor || 0) > 0 && ctx.createDynamicsCompressor) {
                    const comp = ctx.createDynamicsCompressor();
                    const amount = Math.max(0, Math.min(1, Number(seg.compressor || 0)));
                    comp.threshold.value = -18 - amount * 24;
                    comp.knee.value = 18;
                    comp.ratio.value = 1 + amount * 11;
                    comp.attack.value = .006;
                    comp.release.value = .18;
                    nodes.push(comp);
                }
                const trackComp = trackState.effectChain.find((fx) => fx.enabled !== false && fx.type === "compressor");
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
                if (ctx.createStereoPanner) {
                    const pan = ctx.createStereoPanner();
                    pan.pan.value = Math.max(-1, Math.min(1, Number(seg.pan || 0)));
                    last.connect(pan);
                    last = pan;
                }
                if (Number(seg.delaySend || 0) > 0) {
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
                const laneX = event.clientX - rect.left + Number(seg.start || 0) * pxPerFrame();
                const frame = Math.max(Number(seg.start || 0), Math.min(Number(seg.start || 0) + Number(seg.length || 1), Math.round(laneX / pxPerFrame())));
                setPlayhead(frame, false);
                splitSelectedAtPlayhead();
                return;
            }
            const mode = pickedMode;
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
                const marker = clip.querySelector(".iamccs-clip-source-marker");
                if (marker) marker.style.display = "none";
                if (shotboardSyncTimer) {
                    window.clearTimeout(shotboardSyncTimer);
                    shotboardSyncTimer = 0;
                }
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
        input.title = `${input.title || ""} Drag horizontally to change value.`.trim();
        let startX = 0;
        let startValue = 0;
        let dragging = false;
        let changed = false;
        input.addEventListener("pointerdown", (event) => {
            if (event.button !== 0 || event.altKey || event.ctrlKey || event.metaKey) return;
            startX = event.clientX;
            startValue = Number(input.value || 0);
            dragging = false;
            changed = false;
            const step = Number(options.step ?? input.step ?? 1) || 1;
            input.setPointerCapture?.(event.pointerId);
            const move = (moveEvent) => {
                const dx = moveEvent.clientX - startX;
                if (Math.abs(dx) < 3 && !dragging) return;
                dragging = true;
                changed = true;
                const multiplier = moveEvent.shiftKey ? 10 : moveEvent.altKey ? .1 : 1;
                let next = startValue + Math.round(dx / 6) * step * multiplier;
                if (options.min != null) next = Math.max(Number(options.min), next);
                if (options.max != null) next = Math.min(Number(options.max), next);
                input.value = String(Number(next.toFixed(4)));
                input.dispatchEvent(new Event("input", { bubbles: true }));
                moveEvent.preventDefault();
            };
            const up = (upEvent) => {
                window.removeEventListener("pointermove", move);
                window.removeEventListener("pointerup", up);
                try { input.releasePointerCapture?.(upEvent.pointerId); } catch {}
                if (changed) input.dispatchEvent(new Event("change", { bubbles: true }));
            };
            window.addEventListener("pointermove", move);
            window.addEventListener("pointerup", up, { once: true });
            event.preventDefault();
        });
        input.addEventListener("click", (event) => {
            if (dragging) event.preventDefault();
        });
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
            splitSelectedAtPlayhead();
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
        const source = makeMultiSelect("Source", Array.from({ length: Math.max(5, Number(state.audioTrackCount || 5)) }, (_, i) => [String(i), `A${i + 1}`]), sourceTrack, (value) => { sourceTrack = Number(value || 0); });
        const dest = makeMultiSelect("T1 lane", Array.from({ length: Math.max(5, Number(state.audioTrackCount || 5)) }, (_, i) => [String(i), `A${i + 1}`]), destinationStartTrack, (value) => { destinationStartTrack = Number(value || 0); });
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
            const freqToX = (hz) => {
                const t = (Math.log10(Math.max(20, Math.min(22000, Number(hz || 20)))) - Math.log10(20)) / (Math.log10(22000) - Math.log10(20));
                return Math.max(0, Math.min(w, t * w));
            };
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
            const lowX = freqToX(140);
            const midX = freqToX(1200);
            const highX = freqToX(6200);
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
                [lowX / Math.max(1, w), low, "#d6a75e", "L"],
                [midX / Math.max(1, w), midDb, "#55c7b9", "M"],
                [highX / Math.max(1, w), high, "#d08963", "H"],
            ];
            ctx.font = "900 10px ui-monospace, Consolas, monospace";
            for (const [tx, db, color, label] of points) {
                const x = tx * w;
                const y = yFromDb(db);
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(x, y, 6, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = "rgba(0,0,0,.72)";
                ctx.stroke();
                ctx.fillStyle = "#fff2cf";
                ctx.fillText(label, x + 8, y - 8);
            }
            ctx.fillStyle = "#ffc67f";
            ctx.font = "900 9px ui-monospace, Consolas, monospace";
            if (hpf > 20) ctx.fillText(`LC ${Math.round(hpf)}Hz ${Math.round(lowCutLevel)}dB`, 8, h - 9);
            if (lpf < 22000) ctx.fillText(`HC ${(lpf / 1000).toFixed(1)}k ${Math.round(highCutLevel)}dB`, Math.max(8, w - 116), h - 9);
            return;
        }
        const currentFx = canvas._iamccsFx || null;
        const activeFx = currentFx || chainForTarget().find((fx) => fx.enabled !== false && (fx.type === "compressor" || fx.type === "limiter"));
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
        const chainEl = document.createElement("div");
        chainEl.className = "iamccs-device-chain";
        const previewForFx = (type) => {
            if (type === "eq") return ["eq", "EQ 3-Point Editor"];
            if (type === "compressor" || type === "limiter" || type === "gate" || type === "deesser" || type === "transient") return ["bus", `${String(type || "Dynamics").toUpperCase()} Editor`];
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
            const module = document.createElement("div");
            module.className = "iamccs-device-module";
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
                const applyParam = (nextValue, commit = false) => {
                    const next = Math.max(Number(min), Math.min(Number(max), Number(nextValue ?? range.value ?? fallback)));
                    range.value = String(next);
                    fx.params[key] = next;
                    const t = (next - Number(min)) / Math.max(.0001, Number(max) - Number(min));
                    dial.style.setProperty("--knob-angle", `${-135 + t * 270}deg`);
                    valueText.textContent = `${Number(next.toFixed(2))}${unit || ""}`;
                    if (fx.type === "compressor" && target.type === "master") state.masterBus.compressor = Math.max(.01, Math.min(1, (Math.abs(Number(fx.params.threshold || -18)) / 60) + .15));
                    if (fx.type === "limiter" && target.type === "master") {
                        state.masterBus.limiter = true;
                        state.masterBus.ceilingDb = Number(fx.params.ceiling ?? state.masterBus.ceilingDb ?? -1);
                    }
                    module.querySelectorAll(".iamccs-inline-efx-canvas").forEach((canvas) => paintInlineCanvas(canvas, canvas.dataset.kind || "", selected, staticPeak));
                    if (commit) {
                        addEdit(`${targetName} ${name}: ${knobName} ${valueText.textContent}.`);
                        writeState(`fx_${fx.type}_${key}`);
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
                        const points = [
                            { key: "low", x: .27 },
                            { key: "mid", x: .58 },
                            { key: "high", x: .82 },
                        ];
                        const nearest = points.reduce((best, item) => Math.abs(item.x - x) < Math.abs(best.x - x) ? item : best, points[0]);
                        const key = nearest.key;
                        eqFx.params[key] = Number(((.5 - y) * 48).toFixed(2));
                        paintInlineCanvas(canvas, "eq", selected, staticPeak);
                        if (commit) writeState(`eq_point_${key}`);
                        else scheduleSilentStateWrite(`eq_point_${key}`);
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
                    const activeFx = canvas._iamccsFx || chainForTarget().find((fx) => fx.type === "compressor" || fx.type === "limiter");
                    if (!activeFx) return;
                    normalizeEffect(activeFx);
                    const rect = canvas.getBoundingClientRect();
                    const update = (ev, commit = false) => {
                        const x = Math.max(0, Math.min(1, (ev.clientX - rect.left) / Math.max(1, rect.width)));
                        const y = Math.max(0, Math.min(1, (ev.clientY - rect.top) / Math.max(1, rect.height)));
                        if (activeFx.type === "limiter") {
                            activeFx.params.ceiling = Number((-12 + x * 12).toFixed(1));
                            activeFx.params.output = Number((-12 + (1 - y) * 18).toFixed(1));
                        } else {
                            activeFx.params.threshold = Number((-60 + x * 60).toFixed(1));
                            activeFx.params.ratio = Number((1 + (1 - y) * 19).toFixed(1));
                        }
                        paintInlineCanvas(canvas, "bus", selected, staticPeak);
                        if (commit) writeState(`${activeFx.type}_point`);
                        else scheduleSilentStateWrite(`${activeFx.type}_point`);
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
            if (canvas) requestAnimationFrame(() => paintInlineCanvas(canvas, kind, selected, staticPeak));
        });
        grid.appendChild(chainEl);
        efx.appendChild(grid);
        parent.appendChild(efx);
    };
    const draw = () => {
        root.querySelectorAll(".iamccs-audio-board-dynamic").forEach((el) => el.remove());
        const dynamic = document.createElement("div");
        dynamic.className = "iamccs-audio-board-dynamic";
        const head = document.createElement("div");
        head.className = "iamccs-audio-board-head";
        const left = document.createElement("div");
        left.innerHTML = `<div class="iamccs-audio-board-title">IAMCCS AudioBoard Arranger</div><div class="iamccs-audio-board-sub">${segments().length} clips / ${Math.max(1, Number(state.audioTrackCount || 4))} tracks / ${fmtTime(totalFrames())} / sync V3: ${linkedShotboardNodes().length}</div>`;
        const tools = document.createElement("div");
        tools.className = "iamccs-audio-board-tools";
        addButton(tools, root._iamccsAudioFullscreenState ? "Close Editor" : "Open Editor", () => toggleAudioFullscreen(), root._iamccsAudioFullscreenState ? "is-active" : "");
        addButton(tools, "MULTI", () => { state.showMultiGeneration = !state.showMultiGeneration; addEdit(`Multigeneration panel ${state.showMultiGeneration ? "shown" : "hidden"}.`); writeState("toggle_multi", false); draw(); }, state.showMultiGeneration ? "is-active" : "");
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
            addButton(tools, tool.toUpperCase(), () => { view().tool = tool; addEdit(`Tool: ${tool}.`); writeState("tool", false); draw(); }, view().tool === tool ? "is-active" : "");
        }
        addButton(tools, "Zoom +", () => { view().timeZoom = Math.min(8, view().timeZoom * 1.25); addEdit("Zoomed timeline in."); writeState("zoom", false); draw(); });
        addButton(tools, "Zoom -", () => { view().timeZoom = Math.max(.35, view().timeZoom / 1.25); addEdit("Zoomed timeline out."); writeState("zoom", false); draw(); });
        addButton(tools, "Tall +", () => { view().trackHeight = Math.min(200, view().trackHeight + 10); addEdit("Increased track height."); writeState("track_height", false); draw(); });
        addButton(tools, "Tall -", () => { view().trackHeight = Math.max(144, view().trackHeight - 10); addEdit("Reduced track height."); writeState("track_height", false); draw(); });
        addButton(tools, "Event Monitor", () => { state.showEventMonitor = !state.showEventMonitor; addEdit(`Event monitor ${state.showEventMonitor ? "shown" : "hidden"}.`); writeState("event_monitor", false); draw(); }, state.showEventMonitor ? "is-active" : "");
        addButton(tools, "Sync Now", () => { writeState("manual_sync"); draw(); });
        addButton(tools, "Pull V3", () => pullFromShotboard());
        addButton(tools, "Clear", () => { stopPlayback(false); state.audioSegments = []; selectedId = ""; addEdit("Cleared arranger clips."); writeState("clear"); draw(); }, "danger");
        head.append(left, tools);
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
        master.innerHTML = `<div class="iamccs-master-title">MASTER OUT</div><span class="iamccs-master-meter"><i style="width:0%"></i></span>`;
        const masterControls = document.createElement("div");
        masterControls.className = "iamccs-master-controls";
        masterControls.onclick = (event) => event.stopPropagation();
        const limiterToggle = document.createElement("button");
        limiterToggle.type = "button";
        limiterToggle.className = `iamccs-master-toggle${state.masterBus?.limiter ? " is-active" : ""}`;
        limiterToggle.textContent = "LIMITER";
        limiterToggle.onclick = () => {
            state.masterBus.limiter = !state.masterBus.limiter;
            addEdit(`Master limiter ${state.masterBus.limiter ? "on" : "off"}.`);
            writeState("master_limiter");
            draw();
        };
        const compToggle = document.createElement("button");
        compToggle.type = "button";
        compToggle.className = `iamccs-master-toggle${Number(state.masterBus?.compressor || 0) > 0 ? " is-active" : ""}`;
        compToggle.textContent = "COMP";
        compToggle.onclick = () => {
            state.masterBus.compressor = Number(state.masterBus?.compressor || 0) > 0 ? 0 : .45;
            addEdit(`Master compressor ${Number(state.masterBus.compressor || 0) > 0 ? "on" : "off"}.`);
            writeState("master_compressor");
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
        normInput.onchange = () => { state.masterAudioNormalize = Boolean(normInput.checked); addEdit("Toggled master normalize."); writeState("master_normalize"); draw(); };
        norm.append(normInput, "Normalize");
        masterControls.appendChild(norm);
        master.appendChild(masterControls);
        const masterMonitor = document.createElement("span");
        masterMonitor.className = "iamccs-master-crt iamccs-master-readout";
        masterMonitor.textContent = "PK 000 RMS 000";
        master.appendChild(masterMonitor);
        dynamic.appendChild(master);

        const timeline = document.createElement("div");
        timeline.className = "iamccs-audio-board-timeline";
        timeline.style.setProperty("--iamccs-track-height", `${view().trackHeight}px`);
        timeline.style.setProperty("--iamccs-px-per-frame", `${pxPerFrame()}px`);
        const ruler = document.createElement("div");
        ruler.className = "iamccs-audio-board-ruler";
        ruler.style.minWidth = `${contentWidth()}px`;
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
        for (let s = 0; s <= framesToSeconds(totalFrames()) + .01; s += 1) {
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
        const trackCount = Math.max(1, Number(state.audioTrackCount || 4));
        for (let track = 0; track < trackCount; track += 1) {
            const lane = document.createElement("div");
            lane.className = "iamccs-audio-board-track";
            lane.style.minWidth = `${contentWidth()}px`;
            addLoopRange(lane, "iamccs-loop-range");
            const trackSegs = segments().filter((seg) => Number(seg.track || 0) === track);
            const trackState = trackSettings(track);
            trackState.volume = Math.max(0, Math.min(2, Number(trackState.volume ?? 1)));
            trackState.pan = Math.max(-1, Math.min(1, Number(trackState.pan || 0)));
            trackState.color = normalizeTrackColor(trackState.color, track);
            lane.style.setProperty("--iamccs-track-color", trackState.color);
            const trackPeak = trackState.mute ? 0 : Math.min(1, trackSegs.reduce((sum, seg) => sum + peakFor(seg), 0) * trackState.volume);
            const trackLabel = document.createElement("div");
            trackLabel.className = `iamccs-audio-board-track-label${selectedMixer().type === "track" && selectedMixer().track === track ? " is-selected" : ""}`;
            trackLabel.style.borderLeft = `4px solid ${trackState.color}`;
            trackLabel.onclick = () => {
                state.selectedMixer = { type: "track", track };
                transport.helper = `Selected A${track + 1} device chain.`;
                writeState("select_track", false);
                draw();
            };
            const top = document.createElement("div");
            top.className = "iamccs-track-strip-top";
            top.innerHTML = `<div class="iamccs-track-name"><span>A${track + 1}</span><small>${trackSegs.length} clips</small></div><span class="iamccs-audio-board-meter iamccs-track-meter" data-track="${track}"><i style="width:${transport.playing ? Math.round(trackPeak * 100) : 0}%"></i></span>`;
            const controls = document.createElement("div");
            controls.className = "iamccs-track-strip-controls";
            const addTrackToggle = (labelText, key) => {
                const btn = document.createElement("button");
                btn.type = "button";
                btn.className = `iamccs-track-mini${trackState[key] ? " is-active" : ""}`;
                btn.textContent = labelText;
                btn.onclick = (event) => {
                    event.stopPropagation();
                    state.selectedMixer = { type: "track", track };
                    trackState[key] = !trackState[key];
                    addEdit(`A${track + 1} ${labelText} ${trackState[key] ? "on" : "off"}.`);
                    writeState(`track_${key}`);
                    draw();
                };
                controls.appendChild(btn);
            };
            addTrackToggle("M", "mute");
            addTrackToggle("S", "solo");
            addTrackToggle("N", "normalize");
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
                trackState.color = nextTrackColor(trackState.color, track);
                addEdit(`A${track + 1} color changed.`);
                writeState("track_color");
                draw();
            };
            controls.appendChild(colorButton);
            const knobRow = document.createElement("div");
            knobRow.className = "iamccs-track-knob-row";
            const makeTrackKnob = (labelText, key, min, max, step, fallback, formatter) => {
                const wrap = document.createElement("div");
                wrap.className = "iamccs-track-knob-wrap";
                const knob = document.createElement("button");
                knob.type = "button";
                knob.className = "iamccs-track-knob";
                const readout = document.createElement("span");
                readout.className = "iamccs-track-knob-readout";
                const sync = () => {
                    const value = Math.max(min, Math.min(max, Number(trackState[key] ?? fallback)));
                    trackState[key] = value;
                    const t = (value - min) / Math.max(.0001, max - min);
                    knob.style.setProperty("--iamccs-knob-angle", `${-135 + t * 270}deg`);
                    knob.style.setProperty("--iamccs-knob-fill", `${Math.round(t * 78)}%`);
                    readout.textContent = formatter(value);
                    knob.title = `A${track + 1} ${labelText}: ${formatter(value)}`;
                };
                sync();
                knob.onpointerdown = (event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    state.selectedMixer = { type: "track", track };
                    const startX = event.clientX;
                    const startY = event.clientY;
                    const startValue = Number(trackState[key] ?? fallback);
                    try { knob.setPointerCapture?.(event.pointerId); } catch {}
                    const move = (moveEvent) => {
                        const delta = ((moveEvent.clientX - startX) + (startY - moveEvent.clientY)) * Number(step || .01);
                        trackState[key] = Math.max(min, Math.min(max, startValue + delta));
                        sync();
                        transport.helper = `A${track + 1} ${labelText}: ${formatter(trackState[key])}`;
                        scheduleSilentStateWrite(`track_${key}_live`);
                        scheduleShotboardSync(`track_${key}_live`);
                        moveEvent.preventDefault();
                    };
                    const up = (upEvent) => {
                        window.removeEventListener("pointermove", move);
                        window.removeEventListener("pointerup", up);
                        try { knob.releasePointerCapture?.(upEvent.pointerId); } catch {}
                        addEdit(`A${track + 1} ${labelText} ${formatter(trackState[key])}.`);
                        writeState(`track_${key}`, false);
                        draw();
                    };
                    window.addEventListener("pointermove", move, { passive: false });
                    window.addEventListener("pointerup", up, { passive: false, once: true });
                };
                const labelEl = document.createElement("span");
                labelEl.className = "iamccs-track-knob-label";
                labelEl.textContent = labelText;
                wrap.append(labelEl, knob, readout);
                return wrap;
            };
            knobRow.append(
                makeTrackKnob("Volume", "volume", 0, 2, .006, 1, (v) => `${Math.round(v * 100)}%`),
                makeTrackKnob("Pan", "pan", -1, 1, .006, 0, (v) => Math.abs(v) < .02 ? "C" : `${v < 0 ? "L" : "R"}${Math.round(Math.abs(v) * 100)}`)
            );
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

        const selected = selectedClip();
        if (selected && !selectedId) selectedId = selected.id;
        const clipActions = document.createElement("div");
        clipActions.className = "iamccs-clip-action-bar";
        clipActions.innerHTML = `<strong>CLIP EDIT</strong>`;
        addButton(clipActions, "Clip Values", () => {
            state.showClipValues = !state.showClipValues;
            addEdit(`Clip values ${state.showClipValues ? "shown" : "hidden"}.`);
            writeState("clip_values", false);
            draw();
        }, state.showClipValues ? "is-active" : "is-values");
        addButton(clipActions, "Track Up", () => selected && setSelectedExternal("track", Math.max(0, Number(selected.track || 0) - 1)));
        addButton(clipActions, "Track Down", () => selected && setSelectedExternal("track", Math.min(Math.max(1, Number(state.audioTrackCount || 4)) - 1, Number(selected.track || 0) + 1)));
        addButton(clipActions, "Trim In", () => trimStartToPlayhead());
        addButton(clipActions, "Trim Out", () => trimEndToPlayhead());
        addButton(clipActions, "Split", () => splitSelectedAtPlayhead());
        addButton(clipActions, "Delete Clip", () => {
            if (!selected) return;
            state.audioSegments = segments().filter((seg) => seg.id !== selected.id);
            selectedId = "";
            addEdit("Deleted clip.");
            writeState("delete");
            draw();
        }, "danger");
        dynamic.appendChild(clipActions);
        const editor = document.createElement("div");
        editor.className = "iamccs-audio-board-editor";
        if (!state.showClipValues) editor.style.display = "none";
        const setSelected = (key, value) => {
            if (!selected) return;
            selected[key] = value;
            addEdit(`Edited ${String(selected.name || selected.fileName || "clip")}: ${key}`);
            writeState(`edit_${key}`);
            draw();
        };
        const setSelectedExternal = (key, value) => {
            if (!selected) return;
            selected[key] = value;
            addEdit(`Edited ${String(selected.name || selected.fileName || "clip")}: ${key}`);
            writeState(`edit_${key}`);
            draw();
        };
        const numberControl = (label, key, value, step = "1", transform = (v) => Number(v || 0), min = null, max = null) => {
            const wrap = document.createElement("label");
            wrap.textContent = label;
            const input = document.createElement("input");
            input.type = "number";
            input.step = step;
            if (min != null) input.min = String(min);
            if (max != null) input.max = String(max);
            input.value = String(value);
            input.onchange = () => setSelected(key, transform(input.value));
            makeDraggableNumber(input, { step, min, max });
            wrap.appendChild(input);
            return wrap;
        };
        if (selected) {
            editor.append(
                numberControl("Start", "start", framesToSeconds(selected.start).toFixed(2), "0.01", secondsToFrames),
                numberControl("End", "length", framesToSeconds(Number(selected.start || 0) + Number(selected.length || 1)).toFixed(2), "0.01", (v) => Math.max(1, secondsToFrames(v) - Number(selected.start || 0))),
                numberControl("Len", "length", framesToSeconds(selected.length).toFixed(2), "0.01", (v) => Math.max(1, secondsToFrames(v))),
                numberControl("Trim In", "trimStart", framesToSeconds(selected.trimStart || 0).toFixed(2), "0.01", secondsToFrames),
                numberControl("Gain", "gain", Number(selected.gain ?? 1).toFixed(2), "0.05", (v) => Math.max(0, Math.min(4, Number(v || 0))), 0, 4),
                numberControl("Pan", "pan", Number(selected.pan ?? 0).toFixed(2), "0.05", (v) => Math.max(-1, Math.min(1, Number(v || 0))), -1, 1),
                numberControl("Fade In", "fadeInFrames", Number(selected.fadeInFrames || 0), "1"),
                numberControl("Fade Out", "fadeOutFrames", Number(selected.fadeOutFrames || 0), "1"),
                numberControl("Pitch", "pitchSemitones", Number(selected.pitchSemitones || 0), "0.5", Number, -24, 24),
                numberControl("Stretch", "timeStretch", Number(selected.timeStretch || 1), "0.05", (v) => Math.max(.25, Math.min(4, Number(v || 1))), .25, 4),
                numberControl("HPF", "hpfHz", Number(selected.hpfHz || 0), "10", Number, 0, 20000),
                numberControl("LPF", "lpfHz", Number(selected.lpfHz || 22000), "10", Number, 20, 22000),
                numberControl("EQ Low", "eqLowDb", Number(selected.eqLowDb || 0), "0.5", Number, -24, 24),
                numberControl("EQ Mid", "eqMidDb", Number(selected.eqMidDb || 0), "0.5", Number, -24, 24),
                numberControl("EQ High", "eqHighDb", Number(selected.eqHighDb || 0), "0.5", Number, -24, 24),
                numberControl("Comp", "compressor", Number(selected.compressor || 0), "0.05", Number, 0, 1),
                numberControl("Gate dB", "noiseGateDb", Number(selected.noiseGateDb ?? -60), "1", Number, -80, 0),
                numberControl("Duck", "ducking", Number(selected.ducking || 0), "0.05", Number, 0, 1),
                numberControl("Rev", "reverbSend", Number(selected.reverbSend || 0), "0.05", Number, 0, 1),
                numberControl("Delay", "delaySend", Number(selected.delaySend || 0), "0.05", Number, 0, 1),
                numberControl("Width", "stereoWidth", Number(selected.stereoWidth || 1), "0.05", Number, 0, 2),
                numberControl("Transient", "transient", Number(selected.transient || 0), "0.05", Number, -1, 1),
                numberControl("Denoise", "denoise", Number(selected.denoise || 0), "0.05", Number, 0, 1),
            );
            const purpose = document.createElement("label");
            purpose.textContent = "Purpose";
            const purposeSelect = document.createElement("select");
            ["dialogue", "music", "ambience", "foley", "whoosh", "guide_audio", "dialogue_or_music"].forEach((item) => {
                const option = document.createElement("option");
                option.value = item;
                option.textContent = item;
                option.selected = String(selected.purpose || "") === item;
                purposeSelect.appendChild(option);
            });
            purposeSelect.onchange = () => setSelected("purpose", purposeSelect.value);
            purpose.appendChild(purposeSelect);
            editor.appendChild(purpose);
            const link = document.createElement("label");
            link.textContent = "Panel";
            const linkSelect = document.createElement("select");
            const empty = document.createElement("option");
            empty.value = "";
            empty.textContent = "timeline";
            linkSelect.appendChild(empty);
            for (const item of visualOptions()) {
                const option = document.createElement("option");
                option.value = item.id;
                option.textContent = item.label;
                option.selected = String(selected.linkedVisualId || "") === item.id;
                linkSelect.appendChild(option);
            }
            linkSelect.onchange = () => {
                const picked = visualOptions().find((item) => item.id === linkSelect.value);
                selected.linkedVisualId = linkSelect.value;
                if (picked) selected.start = Math.max(0, Math.round(Number(picked.start || 0)));
                addEdit(picked ? `Linked clip to panel ${picked.label} and aligned start.` : "Unlinked clip from panel.");
                writeState("link_panel");
                draw();
            };
            link.appendChild(linkSelect);
            editor.appendChild(link);
            [["Reverse", "reverse"]].forEach(([labelText, key]) => {
                const wrap = document.createElement("label");
                wrap.textContent = labelText;
                const input = document.createElement("input");
                input.type = "checkbox";
                input.checked = Boolean(selected[key]);
                input.onchange = () => setSelected(key, Boolean(input.checked));
                wrap.appendChild(input);
                editor.appendChild(wrap);
            });
        } else {
            const empty = document.createElement("div");
            empty.style.gridColumn = "1 / -1";
            empty.textContent = "Import audio to create editable DAW-style Shotboard audio lanes. Default timeline is 26 seconds.";
            editor.appendChild(empty);
        }
        dynamic.appendChild(editor);

        const staticPeak = Math.min(1, segments().filter((seg) => !seg.mute).reduce((sum, seg) => sum + peakFor(seg), 0) * Math.max(0, Math.min(2, Number(state.masterAudioGain ?? 1) || 1)));
        const lower = document.createElement("div");
        lower.className = "iamccs-audio-board-lower";
        if (!state.showEventMonitor) lower.classList.add("no-monitor");
        if (state.showEventMonitor) {
            const consoleBox = document.createElement("div");
            consoleBox.className = "iamccs-event-console";
            const edits = (state.status?.edits || []).slice(0, 8);
            consoleBox.innerHTML = `
                <div class="iamccs-event-console-head">
                    <span>EVENT MONITOR</span>
                    <span>${fmtTime(transport.playhead)} / ${fmtTime(totalFrames())}</span>
                </div>
                <div class="iamccs-event-console-body"><b>V3</b> ${linkedShotboardNodes().length} | <b>CUSTOM AUDIO</b> ${segments().some(hasMedia) ? "YES" : "NO"} | <b>PEAK</b> ${Math.round(staticPeak * 100)}%
<b>HELPER</b> ${transport.helper || "Ready."}
<b>SHORTCUTS</b> Space play/stop | C cut | M/S/N/L | [ ] trim | arrows nudge | +/- zoom

${edits.length ? edits.join("\n") : "No edits yet."}</div>`;
            lower.appendChild(consoleBox);
        }
        appendInlineEfxPreview(lower, selected, staticPeak);
        dynamic.appendChild(lower);
        root.appendChild(dynamic);
        collectTransportDom();
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
        const fixed = [AUDIO_BOARD_FIXED_WIDTH, AUDIO_BOARD_FIXED_HEIGHT];
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
            : [AUDIO_BOARD_FIXED_WIDTH, AUDIO_BOARD_FIXED_HEIGHT - 70];
    }
    applyFixedAudioBoardSize();
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
    if (raw.includes("IAMCCS_AudioBoardArranger") || raw.includes("IAMCCS AudioBoard Arranger") || widgets.includes("arranger_data")) return "IAMCCS_AudioBoardArranger";
    if (raw.includes("IAMCCS_AudioBoardMixer") || raw.includes("IAMCCS AudioBoard Mixer") || widgets.includes("mixer_data")) return "IAMCCS_AudioBoardMixer";
    if (raw.includes("IAMCCS_ControlAudEfx") || raw.includes("IAMCCS ControlAudEfx") || widgets.includes("control_data")) return "IAMCCS_ControlAudEfx";
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
        console.error("[IAMCCS AudioDialogueUI] render hook failed", {
            source,
            type,
            id: node?.id,
            error: err?.message || String(err),
            stack: err?.stack || "",
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
            min-height: 460px;
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
        .iamccs-audefx-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 8px;
            width: 100%;
            max-width: 100%;
            overflow: hidden;
        }
        .iamccs-audefx-panel {
            min-width: 0;
            padding: 7px;
            background: rgba(0,0,0,.22);
            border: 1px solid rgba(255,255,255,.10);
            border-radius: 7px;
        }
        .iamccs-audefx-label {
            margin-bottom: 5px;
            color: #ffe2a8;
            font-size: 9px;
            font-weight: 950;
            text-transform: uppercase;
        }
        .iamccs-audefx-canvas {
            width: 100%;
            max-width: 100%;
            height: 108px;
            display: block;
            background: #080c0f;
            border: 1px solid rgba(255,255,255,.08);
            border-radius: 5px;
        }
        .iamccs-audefx-strip {
            margin: 8px 0;
            padding: 7px;
            background: rgba(0,0,0,.18);
            border: 1px solid rgba(255,255,255,.10);
            border-radius: 7px;
        }
        .iamccs-audefx-meter {
            position: relative;
            width: 190px;
            height: 12px;
            overflow: hidden;
            background: #070b0d;
            border: 1px solid rgba(255,255,255,.16);
            border-radius: 999px;
        }
        .iamccs-audefx-meter i {
            display: block;
            width: 0%;
            height: 100%;
            background: linear-gradient(90deg, #55c7b9 0%, #cfe37c 58%, #f0b857 78%, #dc5c42 100%);
            transition: width .045s linear;
        }
        .iamccs-audefx-readout {
            color: #a9bec4;
            font: 900 10px/1 ui-monospace, SFMono-Regular, Consolas, monospace;
        }
        .iamccs-audefx-list {
            max-height: 86px;
            overflow: auto;
            white-space: pre-wrap;
            color: #9fb1b8;
            font: 10px/1.35 ui-monospace, SFMono-Regular, Consolas, monospace;
        }
    `;
}

function renderControlAudEfx(node) {
    if (node._iamccsControlAudEfxReady) return;
    node._iamccsControlAudEfxReady = true;
    ensureControlAudEfxStyles();
    const dataWidget = findWidget(node, "control_data");
    hideWidget(dataWidget);
    const root = document.createElement("div");
    root.className = "iamccs-audefx";
    root.tabIndex = 0;
    let audioContext = null;
    let analyser = null;
    let source = null;
    let raf = 0;
    let playing = false;

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
    const upstreamArranger = () => {
        for (const input of node.inputs || []) {
            const link = input.link != null ? app.graph?.links?.[input.link] : null;
            const origin = link ? app.graph?.getNodeById?.(link.origin_id) : null;
            const type = String(origin?.comfyClass || origin?.type || "");
            if (origin && type === "IAMCCS_AudioBoardArranger") return origin;
            if (origin) {
                for (const other of app.graph?._nodes || []) {
                    const otherType = String(other?.comfyClass || other?.type || "");
                    if (otherType !== "IAMCCS_AudioBoardArranger") continue;
                    const linkedToOrigin = (other.outputs || []).some((output) => (output.links || []).some((linkId) => app.graph?.links?.[linkId]?.target_id === origin.id));
                    if (linkedToOrigin) return other;
                }
            }
        }
        return null;
    };
    const arrangerState = () => {
        const fallbackFromTimeline = () => {
            for (const input of node.inputs || []) {
                const link = input.link != null ? app.graph?.links?.[input.link] : null;
                const origin = link ? app.graph?.getNodeById?.(link.origin_id) : null;
                const timelineWidget = origin ? findWidget(origin, "timeline_data") : null;
                try {
                    const timeline = JSON.parse(String(timelineWidget?.value || "{}"));
                    if (Array.isArray(timeline.audioSegments)) {
                        return {
                            audioSegments: timeline.audioSegments,
                            audioTrackCount: timeline.audioTrackCount || 4,
                            masterAudioGain: timeline.masterAudioGain ?? 1,
                            masterAudioNormalize: Boolean(timeline.masterAudioNormalize),
                            masterBus: timeline.masterBus || {},
                            frame_rate: timeline.frame_rate || 24,
                            selectedClipId: timeline.selectedClipId || "",
                        };
                    }
                } catch {}
            }
            return {};
        };
        const arranger = upstreamArranger();
        const widget = arranger ? findWidget(arranger, "arranger_data") : null;
        try {
            const data = JSON.parse(String(widget?.value || "{}"));
            if (data && typeof data === "object" && Array.isArray(data.audioSegments)) return data;
            return fallbackFromTimeline();
        } catch {
            return fallbackFromTimeline();
        }
    };
    const clips = (state) => Array.isArray(state.audioSegments) ? state.audioSegments : [];
    const selectedClip = (state) => {
        const data = parseControl();
        const selectedId = String(data.selectedClipId || state.selectedClipId || "");
        return clips(state).find((clip) => clip.id === selectedId) || clips(state)[0] || null;
    };
    const audioViewUrl = (seg) => {
        const file = String(seg?.audioFile || "").trim();
        if (!file) return "";
        const parts = file.split(/[\\/]+/).filter(Boolean);
        const filename = parts.pop() || file;
        const subfolder = parts.join("/");
        return `/view?filename=${encodeURIComponent(filename)}&type=${encodeURIComponent(seg.audioUploadType || "input")}&subfolder=${encodeURIComponent(subfolder)}`;
    };
    const drawGrid = (ctx, w, h) => {
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = "#080c0f";
        ctx.fillRect(0, 0, w, h);
        ctx.strokeStyle = "rgba(255,255,255,.07)";
        ctx.lineWidth = 1;
        for (let x = 0; x <= w; x += w / 8) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, h);
            ctx.stroke();
        }
        for (let y = 0; y <= h; y += h / 4) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }
    };
    const normalizedPeak = (raw) => {
        if (raw && typeof raw === "object") return { min: Number(raw.min || 0), max: Number(raw.max || 0), rms: Number(raw.rms || 0) };
        const p = Math.abs(Number(raw || 0));
        return { min: -p, max: p, rms: p * .65 };
    };
    const drawStaticWave = (canvas, clip) => {
        const ctx = canvas.getContext("2d");
        const dpr = window.devicePixelRatio || 1;
        const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
        const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
        if (canvas.width !== w) canvas.width = w;
        if (canvas.height !== h) canvas.height = h;
        drawGrid(ctx, w, h);
        const peaks = Array.isArray(clip?.waveformPeaks) ? clip.waveformPeaks : [];
        if (!peaks.length) return;
        ctx.beginPath();
        const top = [];
        const bottom = [];
        for (let x = 0; x < w; x += 1) {
            const raw = normalizedPeak(peaks[Math.min(peaks.length - 1, Math.floor((x / Math.max(1, w - 1)) * peaks.length))]);
            top.push([x, h * .5 - raw.max * h * .46]);
            bottom.unshift([x, h * .5 - raw.min * h * .46]);
        }
        ctx.fillStyle = "rgba(91,190,181,.50)";
        ctx.strokeStyle = "rgba(244,212,158,.95)";
        ctx.lineWidth = Math.max(1, dpr);
        ctx.beginPath();
        top.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
        bottom.forEach(([x, y]) => ctx.lineTo(x, y));
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        ctx.strokeStyle = "rgba(255,255,255,.22)";
        ctx.beginPath();
        ctx.moveTo(0, h * .5);
        ctx.lineTo(w, h * .5);
        ctx.stroke();
    };
    const drawEq = (canvas, clip, masterBus = {}) => {
        const ctx = canvas.getContext("2d");
        const dpr = window.devicePixelRatio || 1;
        const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
        const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
        if (canvas.width !== w) canvas.width = w;
        if (canvas.height !== h) canvas.height = h;
        drawGrid(ctx, w, h);
        const low = Number(clip?.eqLowDb || 0);
        const mid = Number(clip?.eqMidDb || 0);
        const high = Number(clip?.eqHighDb || 0);
        const hpf = Number(clip?.hpfHz || 0);
        const lpf = Number(clip?.lpfHz || 22000);
        const gainAt = (freq) => {
            const log = (Math.log10(freq) - Math.log10(20)) / (Math.log10(22000) - Math.log10(20));
            let db = low * Math.max(0, 1 - Math.abs(log - .22) / .32);
            db += mid * Math.max(0, 1 - Math.abs(log - .55) / .22);
            db += high * Math.max(0, 1 - Math.abs(log - .82) / .30);
            if (hpf > 20 && freq < hpf) db -= Math.min(36, 24 * (1 - freq / hpf));
            if (lpf < 21950 && freq > lpf) db -= Math.min(36, 24 * (freq / lpf - 1));
            db -= Number(masterBus?.compressor || 0) * 2;
            return Math.max(-24, Math.min(24, db));
        };
        ctx.strokeStyle = "#f4d49e";
        ctx.lineWidth = Math.max(2, 2 * dpr);
        ctx.beginPath();
        for (let x = 0; x < w; x += 1) {
            const t = x / Math.max(1, w - 1);
            const freq = 20 * Math.pow(22000 / 20, t);
            const y = h * .5 - (gainAt(freq) / 24) * h * .42;
            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.fillStyle = "#8fa5ac";
        ctx.font = `${10 * dpr}px ui-monospace`;
        ["20", "100", "1k", "10k"].forEach((label, i) => ctx.fillText(label, 6 + i * (w / 4), h - 7 * dpr));
    };
    const ensureAudioContext = async () => {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextClass) throw new Error("WebAudio unavailable");
        if (!audioContext) audioContext = new AudioContextClass();
        if (audioContext.state === "suspended") await audioContext.resume();
        return audioContext;
    };
    const stop = () => {
        try { source?.stop(); } catch {}
        source = null;
        playing = false;
        if (raf) cancelAnimationFrame(raf);
        raf = 0;
        draw();
    };
    const play = async () => {
        const state = arrangerState();
        const clip = selectedClip(state);
        const url = audioViewUrl(clip);
        if (!clip || !url) return;
        stop();
        const ctx = await ensureAudioContext();
        const resp = await fetch(url);
        const decoded = await ctx.decodeAudioData((await resp.arrayBuffer()).slice(0));
        const src = ctx.createBufferSource();
        src.buffer = decoded;
        src.playbackRate.value = Math.max(.25, Math.min(4, Number(clip.timeStretch || 1))) * Math.pow(2, Number(clip.pitchSemitones || 0) / 12);
        const gain = ctx.createGain();
        gain.gain.value = Math.max(0, Math.min(4, Number(clip.gain || 1)));
        let last = gain;
        const addFilter = (type, freq, db = 0, q = .707) => {
            const f = ctx.createBiquadFilter();
            f.type = type;
            f.frequency.value = Math.max(10, Math.min(22000, Number(freq || 0)));
            f.Q.value = q;
            f.gain.value = Math.max(-36, Math.min(36, Number(db || 0)));
            last.connect(f);
            last = f;
        };
        if (Number(clip.hpfHz || 0) > 10) addFilter("highpass", clip.hpfHz);
        if (Number(clip.lpfHz || 22000) < 21950) addFilter("lowpass", clip.lpfHz);
        if (Number(clip.eqLowDb || 0)) addFilter("lowshelf", 140, clip.eqLowDb);
        if (Number(clip.eqMidDb || 0)) addFilter("peaking", 1200, clip.eqMidDb, 1.1);
        if (Number(clip.eqHighDb || 0)) addFilter("highshelf", 6200, clip.eqHighDb);
        if (ctx.createStereoPanner) {
            const pan = ctx.createStereoPanner();
            pan.pan.value = Math.max(-1, Math.min(1, Number(clip.pan || 0)));
            last.connect(pan);
            last = pan;
        }
        analyser = ctx.createAnalyser();
        analyser.fftSize = 2048;
        analyser.smoothingTimeConstant = .72;
        last.connect(analyser);
        analyser.connect(ctx.destination);
        src.connect(gain);
        const offset = Math.max(0, Number(clip.trimStart || 0) / Math.max(1, Number(state.frame_rate || 24)));
        src.start(0, offset, Math.max(.05, Math.min(decoded.duration - offset, Number(clip.length || 1) / Math.max(1, Number(state.frame_rate || 24)))));
        source = src;
        playing = true;
        src.onended = () => stop();
        drawRealtime();
    };
    const drawRealtime = () => {
        const spectrum = root.querySelector(".iamccs-audefx-spectrum");
        const scope = root.querySelector(".iamccs-audefx-scope");
        const meter = root.querySelector(".iamccs-audefx-meter i");
        const readout = root.querySelector(".iamccs-audefx-readout");
        if (!analyser || !playing) return;
        const freq = new Uint8Array(analyser.frequencyBinCount);
        const wave = new Uint8Array(analyser.fftSize);
        analyser.getByteFrequencyData(freq);
        analyser.getByteTimeDomainData(wave);
        const drawBars = (canvas) => {
            if (!canvas) return;
            const ctx = canvas.getContext("2d");
            const dpr = window.devicePixelRatio || 1;
            const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
            const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
            if (canvas.width !== w) canvas.width = w;
            if (canvas.height !== h) canvas.height = h;
            drawGrid(ctx, w, h);
            const bars = 96;
            for (let i = 0; i < bars; i += 1) {
                const v = freq[Math.floor((i / bars) * freq.length)] / 255;
                ctx.fillStyle = v > .85 ? "#dc5c42" : v > .58 ? "#f0b857" : "#55c7b9";
                ctx.fillRect((i / bars) * w, h - v * h, Math.max(1, w / bars - 1), v * h);
            }
        };
        const drawScope = (canvas) => {
            if (!canvas) return;
            const ctx = canvas.getContext("2d");
            const dpr = window.devicePixelRatio || 1;
            const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
            const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
            if (canvas.width !== w) canvas.width = w;
            if (canvas.height !== h) canvas.height = h;
            drawGrid(ctx, w, h);
            ctx.strokeStyle = "#f4d49e";
            ctx.lineWidth = Math.max(1.5, 1.5 * dpr);
            ctx.beginPath();
            wave.forEach((value, i) => {
                const x = (i / Math.max(1, wave.length - 1)) * w;
                const y = (value / 255) * h;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
        };
        drawBars(spectrum);
        drawScope(scope);
        let peak = 0;
        let sum = 0;
        for (const value of wave) {
            const v = (value - 128) / 128;
            peak = Math.max(peak, Math.abs(v));
            sum += v * v;
        }
        const rms = Math.sqrt(sum / Math.max(1, wave.length));
        if (meter) meter.style.width = `${Math.round(peak * 100)}%`;
        if (readout) readout.textContent = `PK ${Math.round(peak * 100)} RMS ${Math.round(rms * 100)}`;
        raf = requestAnimationFrame(drawRealtime);
    };
    const draw = () => {
        root.querySelectorAll(".iamccs-audefx-dynamic").forEach((el) => el.remove());
        const state = arrangerState();
        const clip = selectedClip(state);
        const control = parseControl();
        const dynamic = document.createElement("div");
        dynamic.className = "iamccs-audefx-dynamic";
        const head = document.createElement("div");
        head.className = "iamccs-audefx-head";
        const title = document.createElement("div");
        title.innerHTML = `<div class="iamccs-audefx-title">IAMCCS ControlAudEfx</div><div class="iamccs-audefx-sub">${clip ? String(clip.name || clip.fileName || clip.id) : "connect Arranger cine_linx output"} / clips ${clips(state).length}</div>`;
        const tools = document.createElement("div");
        tools.className = "iamccs-audefx-tools";
        const playBtn = document.createElement("button");
        playBtn.textContent = playing ? "Stop" : "Play FX";
        playBtn.className = playing ? "is-active" : "";
        playBtn.onclick = () => playing ? stop() : play();
        const refresh = document.createElement("button");
        refresh.textContent = "Refresh";
        refresh.onclick = () => draw();
        tools.append(playBtn, refresh);
        head.append(title, tools);
        dynamic.appendChild(head);
        const strip = document.createElement("div");
        strip.className = "iamccs-audefx-strip";
        strip.innerHTML = `<strong>REALTIME</strong><span class="iamccs-audefx-meter"><i></i></span><span class="iamccs-audefx-readout">PK 0 RMS 0</span><span>${clip ? `HPF ${clip.hpfHz || 0} / LPF ${clip.lpfHz || 22000} / EQ ${clip.eqLowDb || 0}, ${clip.eqMidDb || 0}, ${clip.eqHighDb || 0}` : "no clip"}</span>`;
        dynamic.appendChild(strip);
        const grid = document.createElement("div");
        grid.className = "iamccs-audefx-grid";
        const panels = [
            ["Waveform", "iamccs-audefx-wave"],
            ["EQ Response", "iamccs-audefx-eq"],
            ["Spectrum", "iamccs-audefx-spectrum"],
            ["Oscilloscope", "iamccs-audefx-scope"],
        ];
        const canvases = {};
        for (const [label, klass] of panels) {
            const panel = document.createElement("div");
            panel.className = "iamccs-audefx-panel";
            const lab = document.createElement("div");
            lab.className = "iamccs-audefx-label";
            lab.textContent = label;
            const canvas = document.createElement("canvas");
            canvas.className = `iamccs-audefx-canvas ${klass}`;
            canvases[klass] = canvas;
            panel.append(lab, canvas);
            grid.appendChild(panel);
        }
        dynamic.appendChild(grid);
        const list = document.createElement("div");
        list.className = "iamccs-audefx-list";
        list.textContent = clip ? JSON.stringify({
            id: clip.id,
            linkedVisualId: clip.linkedVisualId || "",
            gain: clip.gain,
            pan: clip.pan,
            hpfHz: clip.hpfHz,
            lpfHz: clip.lpfHz,
            eqLowDb: clip.eqLowDb,
            eqMidDb: clip.eqMidDb,
            eqHighDb: clip.eqHighDb,
            compressor: clip.compressor,
            delaySend: clip.delaySend,
            reverbSend: clip.reverbSend,
            masterBus: state.masterBus || {},
        }, null, 2) : "No AudioBoardArranger connected.";
        dynamic.appendChild(list);
        root.appendChild(dynamic);
        if (clip) {
            requestAnimationFrame(() => {
                drawStaticWave(canvases["iamccs-audefx-wave"], clip);
                drawEq(canvases["iamccs-audefx-eq"], clip, state.masterBus || {});
                drawGrid(canvases["iamccs-audefx-spectrum"].getContext("2d"), canvases["iamccs-audefx-spectrum"].clientWidth, canvases["iamccs-audefx-spectrum"].clientHeight);
                drawGrid(canvases["iamccs-audefx-scope"].getContext("2d"), canvases["iamccs-audefx-scope"].clientWidth, canvases["iamccs-audefx-scope"].clientHeight);
            });
        }
    };
    draw();
    const domWidget = node.addDOMWidget("ControlAudEfx", "iamccs_control_aud_efx", root, { serialize: false });
    domWidget.computeSize = (width) => [width, 500];
}


function ensureAudioBoardMixerStyles() {
    if (document.getElementById("iamccs-audio-board-mixer-style")) return;
    const style = document.createElement("style");
    style.id = "iamccs-audio-board-mixer-style";
    style.textContent = `
        .iamccs-audio-mixer {
            box-sizing: border-box;
            width: 100%;
            min-height: 560px;
            padding: 10px;
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
            color: #1b1712;
            border-color: #e8c58d;
            background: linear-gradient(180deg, #f0d49b, #b8784b);
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
            padding-bottom: 8px;
            scrollbar-color: #596164 #1b1d1e;
        }
        .iamccs-mixer-strip {
            flex: 0 0 158px;
            min-height: 430px;
            display: grid;
            grid-template-rows: 34px 42px 58px 1fr 32px 34px;
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
        .iamccs-mixer-fx {
            display:flex;
            align-items:center;
            justify-content:space-between;
            height:28px;
            padding:0 7px;
            color:#bec6c8;
            background:linear-gradient(90deg, #454848 0 73%, #87918f 73% 100%);
            border:1px solid rgba(255,255,255,.13);
            border-radius:2px;
            font:900 10px/1 ui-monospace,SFMono-Regular,Consolas,monospace;
        }
        .iamccs-mixer-strip.is-master .iamccs-mixer-fx { background:linear-gradient(90deg, #353637 0 74%, #6f7977 74% 100%); }
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
            grid-template-columns: 1fr 1fr;
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
        for (const input of node.inputs || []) {
            const link = input.link != null ? app.graph?.links?.[input.link] : null;
            const origin = link ? app.graph?.getNodeById?.(link.origin_id) : null;
            const type = String(origin?.comfyClass || origin?.type || "");
            if (origin && type === "IAMCCS_AudioBoardArranger") return origin;
            if (origin) {
                for (const other of app.graph?._nodes || []) {
                    const otherType = String(other?.comfyClass || other?.type || "");
                    if (otherType !== "IAMCCS_AudioBoardArranger") continue;
                    const linkedToOrigin = (other.outputs || []).some((output) => (output.links || []).some((linkId) => app.graph?.links?.[linkId]?.target_id === origin.id));
                    if (linkedToOrigin) return other;
                }
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
                pan: 0,
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
            arranger?.setDirtyCanvas?.(true, true);
        }
        writeLocal({ mirrorTrackSettings: data.trackSettings || [], masterBus: data.masterBus || {}, last_sync_reason: reason });
    };
    const peakForTrack = (data, track) => {
        const segs = Array.isArray(data.audioSegments) ? data.audioSegments.filter((seg) => Number(seg.track || 0) === track && !seg.mute) : [];
        if (!segs.length) return 0;
        const raw = segs.reduce((sum, seg) => sum + Math.max(.08, Math.min(1, Number(seg.peak || seg.audioPeak || .38))), 0) / Math.max(1, segs.length);
        const settings = ensureTrackSettings(data)[track] || {};
        return Math.max(0, Math.min(1, raw * Number(settings.volume ?? 1)));
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
    function updateMixerLiveMeters() {
        const api = arrangerTransport();
        const snap = api?.snapshot?.();
        const meters = snap?.meters || {};
        const playing = Boolean(snap?.playing);
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
            if (playing && key === "master") peak = Number(meters.master?.peak || 0);
            else if (playing && key !== "") peak = Number(meters.tracks?.[key]?.peak || 0);
            fill.style.height = `${Math.max(0, Math.min(100, Math.round(peak * 100)))}%`;
        });
        mixerMeterRaf = requestAnimationFrame(updateMixerLiveMeters);
    }
    const draw = () => {
        const { data } = readArranger();
        const local = parseLocal();
        if (!Array.isArray(data.audioSegments) && Array.isArray(local.audioSegments)) Object.assign(data, local);
        data.audioTrackCount = Math.max(1, Number(data.audioTrackCount || (Array.isArray(data.trackSettings) ? data.trackSettings.length : 4) || 4));
        data.masterBus = data.masterBus && typeof data.masterBus === "object" ? data.masterBus : {};
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
            const fx = document.createElement("div");
            fx.className = "iamccs-mixer-fx";
            fx.innerHTML = `<span>FX</span><button title="Effect power">${(st.effectChain || []).some((item) => item.enabled !== false) ? "on" : "off"}</button>`;
            const pan = document.createElement("div");
            pan.className = "iamccs-mixer-pan";
            const panKnob = document.createElement("button");
            panKnob.className = "iamccs-mixer-knob";
            panKnob.type = "button";
            setKnob(panKnob, isMaster ? 0 : Number(st.pan || 0), -1, 1);
            const panText = document.createElement("span");
            const syncPanText = () => { const v = isMaster ? 0 : Number(st.pan || 0); panText.textContent = isMaster ? "stereo" : (Math.abs(v) < .02 ? "C" : `${v < 0 ? "L" : "R"}${Math.round(Math.abs(v) * 100)}`); setKnob(panKnob, v, -1, 1); };
            syncPanText();
            if (!isMaster) attachDrag(panKnob, () => st.pan || 0, (value, commit) => { st.pan = value; syncPanText(); if (commit) syncArranger(data, `mixer_pan_${index}`); }, -1, 1, .006);
            pan.append(panKnob, panText);
            const actions = document.createElement("div");
            actions.className = "iamccs-mixer-actions";
            if (isMaster) {
                actions.innerHTML = `<button>RMS</button><button>PK</button>`;
            } else {
                const mute = document.createElement("button");
                mute.textContent = "M";
                mute.className = st.mute ? "is-active" : "";
                mute.onclick = () => { st.mute = !st.mute; syncArranger(data, `mixer_mute_${index}`); draw(); };
                const solo = document.createElement("button");
                solo.textContent = "S";
                solo.className = `solo${st.solo ? " is-active" : ""}`;
                solo.onclick = () => { st.solo = !st.solo; syncArranger(data, `mixer_solo_${index}`); draw(); };
                actions.append(mute, solo);
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
                meterFill.style.height = `${Math.round((isMaster ? .45 : peakForTrack(data, index)) * 100)}%`;
            };
            slider.onchange = () => syncArranger(data, isMaster ? "mixer_master_gain" : `mixer_volume_${index}`);
            fader.appendChild(slider);
            const meter = document.createElement("div");
            meter.className = "iamccs-mixer-meter";
            const meterFill = document.createElement("i");
            const staticPeak = isMaster ? Math.min(1, Array.from({ length: data.audioTrackCount }, (_, i) => peakForTrack(data, i)).reduce((a, b) => a + b, 0) / Math.max(1, data.audioTrackCount)) : peakForTrack(data, index);
            meterFill.dataset.mixerMeter = isMaster ? "master" : String(index);
            meterFill.dataset.staticPeak = String(staticPeak);
            meterFill.style.height = `${Math.round(staticPeak * 100)}%`;
            meter.appendChild(meterFill);
            body.append(scale, fader, meter);
            const route = document.createElement("div");
            route.className = "iamccs-mixer-route";
            route.title = "Route";
            const name = document.createElement("div");
            name.className = "iamccs-mixer-label";
            name.textContent = label.replace(/^A/, "");
            strip.append(fx, pan, actions, body, route, name);
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
    domWidget.computeSize = (width) => [width, 620];
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
            nodeData.name !== "IAMCCS_AudioBoardMixer" &&
            nodeData.name !== "IAMCCS_CineSpeech1PromptCompiler"
        ) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            console.info("[IAMCCS AudioDialogueUI] node created", { name: nodeData.name, id: this?.id, type: this?.type });
            renderIamccsAudioDialogueNode(this, "prototype.onNodeCreated");
        };
    },
});
