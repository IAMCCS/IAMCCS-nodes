import { app } from "../../scripts/app.js";

const EDITOR_VERSION = "2026-04-25-1";

const CUT_MODES = ["hard_cut", "continuity_cut", "soft_cut", "match_cut"];
const AUDIO_OPTIONS = ["0", "1", "2", "3", "4", "5", "6", "7", "8"];
const REF_OPTIONS = ["1", "2", "3", "4", "5", "6", "7", "8", "1>2", "2>1", "1+2", "2+3", "3+1"];
const SOURCE_OPTIONS = ["0", "1", "2", "3", "4", "5", "6", "7", "8"];
const SOURCE_RANGE_OPTIONS = ["all", "0+96", "24+84", "48+120", "tail120", "0-96", "24-108"];
const V2V_MODES = [
    "v2v_source_plus_reference",
    "v2v_source_context",
    "i2v_from_reference",
    "two_segments_if_long",
    "loop_if_long",
];

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
    IAMCCS_LTX2_CinematicMultiGenPlanner: {
        kind: "multigen",
        widgetName: "shot_lines",
        title: "IAMCCS MultiGen Timeline Editor",
        lineFields: ["seconds", "cut", "ref", "audio", "label", "prompt", "dialogue", "voice"],
    },
    IAMCCS_LTX2_CinematicV2VTimelinePlanner: {
        kind: "v2v",
        widgetName: "timeline_lines",
        title: "IAMCCS V2V Timeline Editor",
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
    out.audio = AUDIO_OPTIONS.includes(String(out.audio)) ? String(out.audio) : "0";
    out.label = out.label || "shot";
    out.prompt = out.prompt || (kind === "v2v" ? "V2V cinematic shot, keep source motion" : "cinematic shot, natural acting");
    out.dialogue = out.dialogue || "";
    out.voice = out.voice || "";
    out.framing = out.framing || "custom";
    if (kind === "v2v") {
        out.source = SOURCE_OPTIONS.includes(String(out.source)) ? String(out.source) : "1";
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
            background: rgba(5, 8, 12, 0.72);
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: Inter, Arial, sans-serif;
            color: #eaf1f8;
        }
        .iamccs-cine-modal {
            width: min(96vw, 1500px);
            height: min(92vh, 920px);
            background: #101820;
            border: 1px solid #375064;
            box-shadow: 0 24px 80px rgba(0,0,0,0.55);
            display: grid;
            grid-template-rows: auto auto 1fr auto;
            overflow: hidden;
        }
        .iamccs-cine-header,
        .iamccs-cine-toolbar,
        .iamccs-cine-footer {
            padding: 12px 16px;
            border-bottom: 1px solid #263847;
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
        }
        .iamccs-cine-footer {
            border-top: 1px solid #263847;
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
            color: #9eb2c4;
            font-size: 12px;
        }
        .iamccs-cine-btn,
        .iamccs-cine-select,
        .iamccs-cine-input,
        .iamccs-cine-textarea {
            background: #172331;
            color: #edf6ff;
            border: 1px solid #3c5368;
            border-radius: 6px;
            font-size: 12px;
            font-family: Inter, Arial, sans-serif;
        }
        .iamccs-cine-btn {
            padding: 8px 10px;
            cursor: pointer;
            white-space: nowrap;
        }
        .iamccs-cine-btn:hover { background: #20344a; }
        .iamccs-cine-btn.primary { background: #2d6f9f; border-color: #4d9bd0; }
        .iamccs-cine-btn.danger { background: #552734; border-color: #81445a; }
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
            background: #172331;
            z-index: 1;
            border-bottom: 1px solid #3c5368;
            color: #bcd0df;
            font-size: 11px;
            text-align: left;
            padding: 7px 6px;
        }
        .iamccs-cine-table td {
            border-bottom: 1px solid #253747;
            padding: 6px;
            vertical-align: top;
        }
        .iamccs-cine-row-index {
            color: #91a9bc;
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
            color: #c9d8e5;
            background: #0b1118;
        }
        .iamccs-cine-status {
            color: #9eb2c4;
            font-size: 12px;
        }
    `;
    document.head.appendChild(style);
}

function makeOption(value, label = value) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    return option;
}

function makeSelect(value, options, onChange) {
    const select = document.createElement("select");
    select.className = "iamccs-cine-select";
    for (const option of options) select.appendChild(makeOption(option));
    select.value = String(value || options[0] || "");
    if (!options.includes(select.value) && options.length) {
        select.appendChild(makeOption(select.value));
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

function openTimelineEditor(node, config) {
    ensureStyles();
    createDatalist("iamccs-cine-ref-options", REF_OPTIONS);
    createDatalist("iamccs-cine-range-options", SOURCE_RANGE_OPTIONS);

    const widget = getWidget(node, config.widgetName);
    const rows = parseRows(widget?.value || "", config);
    let selectedPreset = config.kind === "v2v" ? "v2v_two_camera" : "multigen_dialogue_4";

    const overlay = document.createElement("div");
    overlay.className = "iamccs-cine-overlay";
    const modal = document.createElement("div");
    modal.className = "iamccs-cine-modal";
    overlay.appendChild(modal);
    stopCanvasKeys(overlay);

    const header = document.createElement("div");
    header.className = "iamccs-cine-header";
    const title = document.createElement("div");
    title.className = "iamccs-cine-title";
    title.textContent = config.title;
    const help = document.createElement("div");
    help.className = "iamccs-cine-help";
    help.textContent = config.kind === "v2v"
        ? "Edit V2V source, range, reference, audio and prompt. Apply writes timeline_lines."
        : "Edit shots with dropdowns and prompt boxes. Apply writes shot_lines.";
    header.append(title, help);

    const toolbar = document.createElement("div");
    toolbar.className = "iamccs-cine-toolbar";

    const presetSelect = makeSelect(selectedPreset, config.kind === "v2v"
        ? ["v2v_two_camera", "v2v_single", "v2v_long"]
        : ["multigen_dialogue_4", "multigen_dialogue_2", "multigen_monologue"], (value) => {
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

    const body = document.createElement("div");
    body.className = "iamccs-cine-body";
    const table = document.createElement("table");
    table.className = "iamccs-cine-table";
    body.appendChild(table);

    const footer = document.createElement("div");
    footer.className = "iamccs-cine-footer";
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
    footer.append(preview, status, actions);

    modal.append(header, toolbar, body, footer);
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
            addCell(tr, makeInput(row.ref, (v) => { row.ref = v; updatePreview(); }, { list: "iamccs-cine-ref-options" }), "90px");
            addCell(tr, makeSelect(row.audio, AUDIO_OPTIONS, (v) => { row.audio = v; updatePreview(); }), "70px");
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
        status.textContent = `${rows.length} row(s). ${config.widgetName} will receive ${text.length} characters.`;
    }

    renderRows();
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

    const rawWidget = getWidget(node, config.widgetName);
    if (rawWidget) {
        rawWidget.label = config.kind === "v2v" ? "timeline_lines raw text" : "shot_lines raw text";
    }

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
