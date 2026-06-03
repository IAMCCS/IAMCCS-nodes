import { app } from "../../scripts/app.js";

console.info("[IAMCCS DialogueScriptPlanner UI] module loaded", { ts: new Date().toISOString() });

const STYLE_ID = "iamccs-dialogue-script-planner-style";

const EMOTIONS = [
    "none", "happy", "sad", "angry", "excited", "calm", "fearful", "surprised",
    "disgusted", "confusion", "empathy", "embarrass", "depressed", "coldness",
    "admiration", "whisper", "urgent", "wonder", "resolve"
];

const STYLES = [
    "none", "whisper", "serious", "child", "older", "girl", "pure", "sister",
    "sweet", "exaggerated", "ethereal", "generous", "recite", "act_coy",
    "warm", "shy", "comfort", "authority", "chat", "radio", "soulful",
    "gentle", "story", "vivid", "program", "news", "advertising", "roar",
    "murmur", "shout", "deeply", "loudly", "friendly"
];

const PARA = [
    "none", "Breathing", "Laughter", "Surprise-oh", "Confirmation-en",
    "Uhm", "Surprise-ah", "Surprise-wa", "Sigh", "Question-ei", "Dissatisfaction-hnn"
];

const DEFAULT_DATA = {
    schema: "iamccs.dialogue_script_planner",
    schema_version: 1,
    settings: {
        engine_profile: "stepaudio_editx",
        timeline_mode: "speaker_stems_for_overlap",
        default_line_seconds: 2.6,
        default_gap_seconds: 0.15,
    },
    speakers: [
        { id: "A", name: "Alice", voice: "voices_examples/female/female_02.wav", reference_text: "", emotion_ref: "happy" },
        { id: "B", name: "Bob", voice: "voices_examples/Clint_Eastwood CC3 (enhanced2).wav", reference_text: "", emotion_ref: "serious" },
    ],
    lines: [
        { id: "line_001", speaker: "Alice", start: 0, duration: 2.6, overlap_after: 0.25, emotion: "happy", style: "warm", paralinguistic: "none", text: "I thought the room would be empty by now." },
        { id: "line_002", speaker: "Bob", start: 2.35, duration: 2.7, overlap_after: 0, emotion: "calm", style: "serious", paralinguistic: "none", text: "It is never empty when somebody is still listening." },
    ],
};

function nodeType(node) {
    return String(node?.type || node?.comfyClass || node?.constructor?.type || "");
}

function isPlanner(node) {
    const type = nodeType(node);
    return type === "IAMCCS_DialogueScriptPlanner" || type.includes("DialogueScriptPlanner");
}

function widget(node, name) {
    return (node?.widgets || []).find((item) => item?.name === name);
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

function hideWidget(item) {
    if (!item) return;
    item.hidden = true;
    item.type = "hidden";
    item.computeSize = () => [0, -4];
    item.draw = () => {};
    item.options = { ...(item.options || {}), hidden: true };
    if (item.inputEl) {
        item.inputEl.style.display = "none";
        item.inputEl.style.height = "0";
    }
}

function parseData(node) {
    const raw = widget(node, "dialogue_data")?.value;
    try {
        const parsed = JSON.parse(String(raw || ""));
        if (parsed && typeof parsed === "object") return parsed;
    } catch {}
    return JSON.parse(JSON.stringify(DEFAULT_DATA));
}

function writeData(node, data) {
    data.schema = "iamccs.dialogue_script_planner";
    data.schema_version = 1;
    data.settings = data.settings || {};
    data.settings.engine_profile = widget(node, "engine_profile")?.value || data.settings.engine_profile || "stepaudio_editx";
    data.settings.timeline_mode = widget(node, "timeline_mode")?.value || data.settings.timeline_mode || "speaker_stems_for_overlap";
    data.settings.default_line_seconds = Number(widget(node, "default_line_seconds")?.value || data.settings.default_line_seconds || 2.6);
    data.settings.default_gap_seconds = Number(widget(node, "default_gap_seconds")?.value || data.settings.default_gap_seconds || 0.15);
    setWidget(node, "dialogue_data", JSON.stringify(data, null, 2));
}

function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
        .iamccs-dsp {
            box-sizing: border-box;
            width: 100%;
            padding: 10px;
            color: #dbe7e6;
            background: linear-gradient(180deg, #11191b, #0b1012);
            border: 1px solid rgba(150,185,185,.22);
            border-radius: 8px;
            font: 12px/1.35 Inter, ui-sans-serif, system-ui, sans-serif;
        }
        .iamccs-dsp * { box-sizing: border-box; }
        .iamccs-dsp-head {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 10px;
            margin-bottom: 10px;
        }
        .iamccs-dsp-title {
            color: #f4f3df;
            font-size: 14px;
            font-weight: 800;
        }
        .iamccs-dsp-sub {
            color: #8facaa;
            font-size: 10px;
            margin-top: 2px;
        }
        .iamccs-dsp-actions,
        .iamccs-dsp-toolbar {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 6px;
        }
        .iamccs-dsp button,
        .iamccs-dsp select,
        .iamccs-dsp input,
        .iamccs-dsp textarea {
            color: #e7f4ef;
            background: #0b2022;
            border: 1px solid rgba(103,178,183,.42);
            border-radius: 6px;
            font: inherit;
        }
        .iamccs-dsp button {
            padding: 6px 9px;
            cursor: pointer;
            font-weight: 800;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.06);
        }
        .iamccs-dsp button:hover {
            border-color: rgba(238,215,146,.8);
            color: #fff4cf;
        }
        .iamccs-dsp .primary {
            color: #1f1906;
            background: linear-gradient(180deg, #f2d88c, #c99a45);
            border-color: #f4d28a;
        }
        .iamccs-dsp .danger {
            background: #4f1717;
            border-color: #9a5550;
        }
        .iamccs-dsp select,
        .iamccs-dsp input {
            height: 28px;
            padding: 3px 6px;
        }
        .iamccs-dsp textarea {
            width: 100%;
            min-height: 52px;
            padding: 7px;
            resize: vertical;
            background: #081316;
        }
        .iamccs-dsp-grid {
            display: grid;
            grid-template-columns: 220px 1fr;
            gap: 10px;
        }
        .iamccs-dsp-speakers,
        .iamccs-dsp-lines {
            min-width: 0;
        }
        .iamccs-dsp-section-title {
            color: #f3d58a;
            font-size: 10px;
            font-weight: 900;
            text-transform: uppercase;
            margin: 0 0 6px;
        }
        .iamccs-dsp-speaker {
            padding: 8px;
            margin-bottom: 7px;
            background: linear-gradient(180deg, #182524, #111917);
            border: 1px solid rgba(161,135,72,.45);
            border-radius: 7px;
        }
        .iamccs-dsp-speaker input {
            width: 100%;
            margin-top: 5px;
        }
        .iamccs-dsp-line {
            padding: 8px;
            margin-bottom: 8px;
            background: linear-gradient(180deg, #132024, #0d1518);
            border: 1px solid rgba(102,144,150,.32);
            border-left: 4px solid #c79a45;
            border-radius: 7px;
        }
        .iamccs-dsp-line-top {
            display: grid;
            grid-template-columns: 110px repeat(3, 72px) 115px 115px 120px auto;
            gap: 6px;
            align-items: center;
            margin-bottom: 6px;
        }
        .iamccs-dsp-line-top input,
        .iamccs-dsp-line-top select { width: 100%; }
        .iamccs-dsp-mini {
            display: flex;
            gap: 5px;
            justify-content: flex-end;
        }
        .iamccs-dsp-help {
            margin-top: 9px;
            padding: 7px 8px;
            color: #9ec2be;
            background: #071012;
            border: 1px solid rgba(127,165,164,.18);
            border-radius: 7px;
            font-size: 10px;
        }
        @media (max-width: 980px) {
            .iamccs-dsp-grid { grid-template-columns: 1fr; }
            .iamccs-dsp-line-top { grid-template-columns: 1fr 1fr; }
        }
    `;
    document.head.appendChild(style);
}

function optionList(values, current) {
    return values.map((value) => {
        const opt = document.createElement("option");
        opt.value = value;
        opt.textContent = value;
        opt.selected = value === current;
        return opt;
    });
}

function fieldInput(value, type = "text", step = "0.05") {
    const input = document.createElement("input");
    input.type = type;
    input.value = value ?? "";
    if (type === "number") input.step = step;
    return input;
}

function fieldSelect(values, current) {
    const select = document.createElement("select");
    select.append(...optionList(values, current));
    return select;
}

function installPlannerUI(node, reason = "install") {
    if (!isPlanner(node) || node._iamccsDialoguePlannerReady) return;
    ensureStyle();
    node._iamccsDialoguePlannerReady = true;

    [
        "dialogue_data",
        "engine_profile",
        "timeline_mode",
        "default_line_seconds",
        "default_gap_seconds",
    ].forEach((name) => hideWidget(widget(node, name)));

    const root = document.createElement("div");
    root.className = "iamccs-dsp";
    const data = parseData(node);
    data.speakers = Array.isArray(data.speakers) && data.speakers.length ? data.speakers : JSON.parse(JSON.stringify(DEFAULT_DATA.speakers));
    data.lines = Array.isArray(data.lines) && data.lines.length ? data.lines : JSON.parse(JSON.stringify(DEFAULT_DATA.lines));

    const render = () => {
        root.replaceChildren();

        const head = document.createElement("div");
        head.className = "iamccs-dsp-head";
        const titleWrap = document.createElement("div");
        const title = document.createElement("div");
        title.className = "iamccs-dsp-title";
        title.textContent = "IAMCCS DialogueScript Planner";
        const sub = document.createElement("div");
        sub.className = "iamccs-dsp-sub";
        sub.textContent = `${data.lines.length} lines / ${data.speakers.length} speakers / overlap-ready stems`;
        titleWrap.append(title, sub);

        const actions = document.createElement("div");
        actions.className = "iamccs-dsp-actions";
        const addSpeaker = document.createElement("button");
        addSpeaker.type = "button";
        addSpeaker.textContent = "Add Speaker";
        addSpeaker.onclick = () => {
            const id = String.fromCharCode(65 + data.speakers.length);
            data.speakers.push({ id, name: `Speaker ${id}`, voice: "", reference_text: "", emotion_ref: "calm" });
            writeData(node, data);
            render();
        };
        const addLine = document.createElement("button");
        addLine.type = "button";
        addLine.className = "primary";
        addLine.textContent = "Add Line";
        addLine.onclick = () => {
            const last = data.lines[data.lines.length - 1] || { start: 0, duration: 2.6, overlap_after: 0 };
            const start = Math.max(0, Number(last.start || 0) + Number(last.duration || 2.6) - Number(last.overlap_after || 0) + 0.15);
            data.lines.push({
                id: `line_${String(data.lines.length + 1).padStart(3, "0")}`,
                speaker: data.speakers[0]?.name || "Speaker",
                start: Number(start.toFixed(2)),
                duration: 2.6,
                overlap_after: 0,
                emotion: "calm",
                style: "none",
                paralinguistic: "none",
                text: "New dialogue line.",
            });
            writeData(node, data);
            render();
        };
        const reflow = document.createElement("button");
        reflow.type = "button";
        reflow.textContent = "Reflow Timing";
        reflow.onclick = () => {
            let cursor = 0;
            const gap = Number(widget(node, "default_gap_seconds")?.value || data.settings?.default_gap_seconds || 0.15);
            data.lines.forEach((line) => {
                line.start = Number(cursor.toFixed(2));
                const dur = Number(line.duration || 2.6);
                cursor += dur + gap - Number(line.overlap_after || 0);
            });
            writeData(node, data);
            render();
        };
        actions.append(addSpeaker, addLine, reflow);
        head.append(titleWrap, actions);

        const toolbar = document.createElement("div");
        toolbar.className = "iamccs-dsp-toolbar";
        const engine = fieldSelect(["stepaudio_editx", "chatterbox", "indextts2", "plain"], widget(node, "engine_profile")?.value || data.settings?.engine_profile || "stepaudio_editx");
        engine.onchange = () => {
            setWidget(node, "engine_profile", engine.value);
            writeData(node, data);
        };
        const mode = fieldSelect(["speaker_stems_for_overlap", "flatten_for_single_tts", "preserve_overlap_in_master_srt"], widget(node, "timeline_mode")?.value || data.settings?.timeline_mode || "speaker_stems_for_overlap");
        mode.onchange = () => {
            setWidget(node, "timeline_mode", mode.value);
            writeData(node, data);
        };
        const lineSec = fieldInput(widget(node, "default_line_seconds")?.value || 2.6, "number");
        lineSec.onchange = () => setWidget(node, "default_line_seconds", Number(lineSec.value || 2.6));
        const gapSec = fieldInput(widget(node, "default_gap_seconds")?.value || 0.15, "number");
        gapSec.onchange = () => setWidget(node, "default_gap_seconds", Number(gapSec.value || 0.15));
        toolbar.append(engine, mode, lineSec, gapSec);

        const grid = document.createElement("div");
        grid.className = "iamccs-dsp-grid";

        const speakers = document.createElement("div");
        speakers.className = "iamccs-dsp-speakers";
        const speakerTitle = document.createElement("div");
        speakerTitle.className = "iamccs-dsp-section-title";
        speakerTitle.textContent = "Speakers / Clones";
        speakers.appendChild(speakerTitle);

        data.speakers.forEach((speaker, index) => {
            const card = document.createElement("div");
            card.className = "iamccs-dsp-speaker";
            const label = document.createElement("div");
            label.textContent = `${speaker.id || index + 1}. ${speaker.name || "Speaker"}`;
            label.style.color = "#f0d88d";
            label.style.fontWeight = "900";
            const name = fieldInput(speaker.name || "");
            name.placeholder = "Speaker name";
            name.onchange = () => {
                const oldName = speaker.name;
                speaker.name = name.value || `Speaker ${index + 1}`;
                data.lines.forEach((line) => {
                    if (line.speaker === oldName) line.speaker = speaker.name;
                });
                writeData(node, data);
                render();
            };
            const voice = fieldInput(speaker.voice || "");
            voice.placeholder = "voice path / alias";
            voice.onchange = () => {
                speaker.voice = voice.value;
                writeData(node, data);
            };
            const ref = fieldInput(speaker.reference_text || "");
            ref.placeholder = "reference text";
            ref.onchange = () => {
                speaker.reference_text = ref.value;
                writeData(node, data);
            };
            const emotionRef = fieldInput(speaker.emotion_ref || "");
            emotionRef.placeholder = "IndexTTS emotion ref / alias";
            emotionRef.onchange = () => {
                speaker.emotion_ref = emotionRef.value;
                writeData(node, data);
            };
            card.append(label, name, voice, ref, emotionRef);
            speakers.appendChild(card);
        });

        const lines = document.createElement("div");
        lines.className = "iamccs-dsp-lines";
        const lineTitle = document.createElement("div");
        lineTitle.className = "iamccs-dsp-section-title";
        lineTitle.textContent = "Dialogue Lines";
        lines.appendChild(lineTitle);

        const speakerNames = data.speakers.map((speaker) => speaker.name || speaker.id || "Speaker");
        data.lines.forEach((line, index) => {
            const card = document.createElement("div");
            card.className = "iamccs-dsp-line";
            const top = document.createElement("div");
            top.className = "iamccs-dsp-line-top";
            const speaker = fieldSelect(speakerNames, line.speaker || speakerNames[0]);
            speaker.onchange = () => {
                line.speaker = speaker.value;
                writeData(node, data);
            };
            const start = fieldInput(line.start ?? 0, "number");
            start.onchange = () => {
                line.start = Number(start.value || 0);
                writeData(node, data);
            };
            const duration = fieldInput(line.duration ?? 2.6, "number");
            duration.onchange = () => {
                line.duration = Math.max(0.08, Number(duration.value || 2.6));
                writeData(node, data);
            };
            const overlap = fieldInput(line.overlap_after ?? 0, "number");
            overlap.onchange = () => {
                line.overlap_after = Math.max(0, Number(overlap.value || 0));
                writeData(node, data);
            };
            const emotion = fieldSelect(EMOTIONS, line.emotion || "none");
            emotion.onchange = () => {
                line.emotion = emotion.value;
                writeData(node, data);
            };
            const style = fieldSelect(STYLES, line.style || "none");
            style.onchange = () => {
                line.style = style.value;
                writeData(node, data);
            };
            const para = fieldSelect(PARA, line.paralinguistic || "none");
            para.onchange = () => {
                line.paralinguistic = para.value;
                writeData(node, data);
            };
            const mini = document.createElement("div");
            mini.className = "iamccs-dsp-mini";
            const up = document.createElement("button");
            up.type = "button";
            up.textContent = "Up";
            up.onclick = () => {
                if (index <= 0) return;
                const tmp = data.lines[index - 1];
                data.lines[index - 1] = data.lines[index];
                data.lines[index] = tmp;
                writeData(node, data);
                render();
            };
            const del = document.createElement("button");
            del.type = "button";
            del.className = "danger";
            del.textContent = "Del";
            del.onclick = () => {
                data.lines.splice(index, 1);
                writeData(node, data);
                render();
            };
            mini.append(up, del);
            top.append(speaker, start, duration, overlap, emotion, style, para, mini);
            const text = document.createElement("textarea");
            text.value = line.text || "";
            text.placeholder = "Dialogue text";
            text.onchange = () => {
                line.text = text.value;
                writeData(node, data);
            };
            card.append(top, text);
            lines.appendChild(card);
        });

        grid.append(speakers, lines);

        const help = document.createElement("div");
        help.className = "iamccs-dsp-help";
        help.textContent = "For true conversation overlap: route speaker_1_srt and speaker_2_srt to separate Unified TTS SRT branches, then mix stems in AudioBoardArranger. Master SRT/tagged text is for serial dialogue or tests.";

        root.append(head, toolbar, grid, help);
        writeData(node, data);
    };

    render();
    const uiWidget = node.addDOMWidget("IAMCCS DialogueScript Planner UI", "iamccs_dialogue_script_planner_ui", root, { serialize: false });
    uiWidget.computeSize = (width) => [width, 620];
    node.size = [Math.max(Number(node.size?.[0] || 0), 900), Math.max(Number(node.size?.[1] || 0), 760)];
    console.info("[IAMCCS DialogueScriptPlanner UI] installed", { nodeId: node?.id, reason });
}

app.registerExtension({
    name: "IAMCCS.DialogueScriptPlannerUI",
    setup() {
        [600, 1600, 3600].forEach((delay) => setTimeout(() => {
            const nodes = Array.isArray(app?.graph?._nodes) ? app.graph._nodes : [];
            nodes.forEach((node) => installPlannerUI(node, `scan+${delay}`));
        }, delay));
    },
    nodeCreated(node) {
        [0, 200, 700].forEach((delay) => setTimeout(() => installPlannerUI(node, `nodeCreated+${delay}`), delay));
    },
    loadedGraphNode(node) {
        [0, 200, 700].forEach((delay) => setTimeout(() => installPlannerUI(node, `loadedGraphNode+${delay}`), delay));
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "IAMCCS_DialogueScriptPlanner") return;
        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            originalOnNodeCreated?.apply(this, arguments);
            setTimeout(() => installPlannerUI(this, "prototype.onNodeCreated"), 0);
        };
    },
});
