// SPDX-License-Identifier: GPL-3.0-or-later

const PICTURE_ROLES = [
    ["off", "OFF"], ["identity", "IDENTITY"], ["opening_frame", "OPENING FRAME"],
    ["closing_frame", "CLOSING FRAME"], ["environment", "ENVIRONMENT"],
    ["composition", "COMPOSITION"], ["style", "STYLE"], ["wardrobe_prop", "WARDROBE / PROP"],
];
const VIDEO_ROLES = [
    ["off", "OFF"], ["motion_camera_timing", "MOTION + CAMERA + TIMING"],
    ["performance", "PERFORMANCE"], ["camera", "CAMERA"], ["continuation", "CONTINUATION"],
    ["source_plate", "SOURCE PLATE"],
];
const AUDIO_ROLES = [
    ["off", "OFF"], ["exact_dialogue", "EXACT DIALOGUE + TIMING"],
    ["exact_soundtrack", "EXACT SOUNDTRACK"], ["voice_timbre", "VOICE TIMBRE"],
    ["timing_reference", "TIMING REFERENCE"],
];

function make(tag, className = "", text = "") {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
}

function parseTimeline(board, getWidget) {
    let timeline = {};
    try { timeline = JSON.parse(String(getWidget(board, "timeline_data")?.value || "{}")); } catch {}
    const visual = (timeline.segments || timeline.rows || timeline.keyframes || [])
        .filter((row) => !row?.placeholder && !["audio", "motion", "video"].includes(String(row?.type || "image").toLowerCase()))
        .sort((a, b) => Number(a?.start ?? a?.start_seconds ?? 0) - Number(b?.start ?? b?.start_seconds ?? 0));
    const audio = (timeline.audioSegments || timeline.audio_segments || [])
        .filter((row) => !row?.placeholder)
        .sort((a, b) => Number(a?.start ?? a?.start_seconds ?? 0) - Number(b?.start ?? b?.start_seconds ?? 0));
    return { timeline, visual, audio };
}

function countToken(text, token) {
    return (String(text || "").match(new RegExp(token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "g")) || []).length;
}

export function inspectH3Prompt({ project, prompt, board, getWidget }) {
    const issues = [];
    const add = (level, title, detail) => issues.push({ level, title, detail });
    const locals = (project.local_prompts || []).filter((row) => row?.enabled !== false && String(row?.prompt || "").trim());
    const text = [prompt, ...locals.map((row) => row.prompt)].join("\n");
    if (!String(prompt || "").trim() && !locals.length) add("error", "No authored prompt", "Write a global field or at least one enabled local slot prompt.");
    if (String(prompt || "").length > 7000) add("error", "Prompt is over 7000 characters", "Reduce repetition before injecting. This check does not truncate your text.");
    else if (String(prompt || "").length > 6200) add("warn", "Prompt is close to the authoring budget", "Review repeated locks and sound descriptions.");
    if (/(?:\.\.\.|\[(?:write|describe|state|define|list|identify|insert|language)\b[^\]]*\])/i.test(text)) add("warn", "Template placeholders remain", "Replace ellipses and bracketed instructions with the real shot facts.");
    [["<d>", "</d>"], ["<lyrics_start>", "<lyrics_end>"], ["<caption_start>", "<caption_end>"]].forEach(([open, close]) => {
        if (countToken(text, open) !== countToken(text, close)) add("error", `Unbalanced ${open}`, `Every ${open} requires one ${close}.`);
    });
    if (/<d>(?!\s*\[[^\]\r\n]+\])/i.test(text)) add("warn", "Dialogue language is missing", "Use <d>[English] exact words</d> (or the actual spoken language). Dialogue text is not inferred from audio.");
    if (/<(?:Picture|Video|Audio|Subject)\s*\d+\s*>/i.test(text) && !/<(?:Picture|Video|Audio|Subject)\s+\d+>/i.test(text)) add("warn", "Non-canonical reference spacing", "Use tags such as <Picture 1> and <Subject 1>.");

    if (!board) {
        add("info", "No unique connected Shotboard", "Input checks are limited. Connect CineLinX to one MiniMax H3 Shotboard; no queue will be started.");
        return issues;
    }
    const { visual, audio } = parseTimeline(board, getWidget);
    const mode = String(project.task_mode || "t2va");
    if (mode === "i2va" && visual.length < 1) add("error", "I2VA needs an opening image", "Add at least one real image slot to the Shotboard timeline.");
    if (mode === "fl2va" && visual.length < 2) add("error", "FL2VA needs two boundary images", "Add an opening and closing image slot.");
    if (mode === "v2va_object_swap" && !/<Video 1>/i.test(text)) add("warn", "V2VA source authority is not named", "Name <Video 1> in the visible source-authority box and ensure the reference module is connected.");
    if (["audio_driven", "multi_shot_lipsync"].includes(mode) && !audio.length) add("error", "No audible Shotboard segment", "Publish or place AudioBoard audio in the Shotboard lanes. H3-generated audio does not require an input segment.");
    if (mode === "multi_shot_lipsync" && visual.length < 2) add("warn", "Only one guided shot", "Multi-Shot LipSync is valid, but two or more image slots are needed to exercise editorial cuts.");
    const maxLocal = Math.max(0, ...locals.map((row) => Number(row.slot) || 0));
    if (visual.length && maxLocal > visual.length) add("warn", "A local prompt targets a missing visual slot", `Highest enabled local slot is ${maxLocal}; the Shotboard currently exposes ${visual.length}.`);
    if (!issues.some((item) => item.level === "error")) add("ok", "Authoring contract is ready", "Visible Prompter fields can be injected. The Shotboard boxes remain the final queue-time truth.");
    return issues;
}

function roleSentence(name, role) {
    const label = name.replace(/([A-Za-z]+)(\d+)/, "$1 $2");
    const map = {
        identity: `${label} is identity authority; retain face, body proportions and stable identifying traits.`,
        opening_frame: `${label} is the complete opening-frame authority for composition, perspective and visible state.`,
        closing_frame: `${label} is the final-frame authority; arrive there near the authored ending without an early morph.`,
        environment: `${label} defines environment, geography, materials and lighting logic, not subject identity.`,
        composition: `${label} defines composition and spatial relationships, not an unrequested redesign.`,
        style: `${label} defines image treatment and texture only; preserve identities and scene geometry from their assigned authorities.`,
        wardrobe_prop: `${label} defines the named wardrobe or prop and its construction, scale and material.`,
        motion_camera_timing: `${label} owns motion, camera path, action timing, occlusion order and edit rhythm.`,
        performance: `${label} owns performance timing and body mechanics, not identity or background.`,
        camera: `${label} owns framing and camera movement only.`,
        continuation: `${label} is a continuity source; retain its last valid state into the requested continuation.`,
        source_plate: `${label} is the source plate; preserve duration, environment, camera and unselected subjects.`,
        exact_dialogue: `${label} owns exact dialogue, phoneme timing, pauses and breaths; do not invent or reorder words.`,
        exact_soundtrack: `${label} is the exact soundtrack authority; synchronize visible contacts and performance to it.`,
        voice_timbre: `${label} defines voice timbre only; dialogue content must remain explicitly authored.`,
        timing_reference: `${label} is timing reference for visible performance and contacts.`,
    };
    return map[role] || "";
}

export function createPrompterInspector({ getProject, composePrompt, getShotboard, getWidget, onProjectChanged, snapshotProject }) {
    const root = make("div", "iamccs-pr-inspector");
    const style = make("style");
    style.textContent = `
      .iamccs-pr-inspector{display:flex;min-height:0;flex:1;flex-direction:column}.iamccs-pr-tabs{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:3px;margin-bottom:8px}.iamccs-pr-tab{height:25px!important;min-width:0!important;padding:0 3px!important;font-size:8px!important}.iamccs-pr-tab.active{border-color:#d9ad58!important;background:#3a3020!important;color:#ffe5aa!important}.iamccs-pr-inspector-panel{display:none;min-height:0;flex:1;overflow:auto}.iamccs-pr-inspector-panel.active{display:block}.iamccs-pr-inspector-panel[data-panel="prompt"].active{display:flex;flex-direction:column}.iamccs-pr-check{display:grid;gap:6px}.iamccs-pr-checkitem{padding:8px;border:1px solid #38414d;border-left-width:4px;border-radius:6px;background:#151b23}.iamccs-pr-checkitem strong{display:block;font-size:10px}.iamccs-pr-checkitem span{display:block;margin-top:3px;color:#9ca8b4;font-size:9px;line-height:1.4}.iamccs-pr-checkitem.error{border-left-color:#dd7268}.iamccs-pr-checkitem.warn{border-left-color:#e0ad54}.iamccs-pr-checkitem.ok{border-left-color:#65b887}.iamccs-pr-checkitem.info{border-left-color:#6da7dc}.iamccs-pr-map{display:grid;gap:7px}.iamccs-pr-mapcard{padding:8px;border:1px solid #34404c;border-radius:6px;background:#151c24;font-size:9px;line-height:1.45}.iamccs-pr-mapcard b{color:#f0cf87}.iamccs-pr-mapempty{padding:10px;color:#93a0ad;border:1px dashed #44505e;border-radius:6px;font-size:9px;line-height:1.5}.iamccs-pr-authrow{display:grid;grid-template-columns:64px minmax(0,1fr);gap:6px;align-items:center;margin:5px 0}.iamccs-pr-authrow label{font:800 8px 'Courier New',monospace;color:#c9d7e4}.iamccs-pr-authrow select{min-width:0;width:100%;height:27px;border:1px solid #3b4b5d;border-radius:5px;background:#101821;color:#e5edf4;font-size:8px}.iamccs-pr-help{display:grid;gap:8px}.iamccs-pr-help section{padding:9px;border:1px solid #38424e;border-radius:6px;background:#151c24}.iamccs-pr-help h4{margin:0 0 5px;color:#efd18e;font-size:10px}.iamccs-pr-help p,.iamccs-pr-help li{margin:0;color:#a9b4bf;font-size:9px;line-height:1.45}.iamccs-pr-help ol{margin:0;padding-left:17px}.iamccs-pr-minihelp{margin:0 0 8px;padding:7px;border-left:2px solid #6c9bc2;background:#14202b;color:#9eb2c3;font-size:9px;line-height:1.4}.iamccs-pr-apply{width:100%;margin-top:8px}
    `;
    root.appendChild(style);
    const tabs = make("div", "iamccs-pr-tabs");
    const panels = new Map();
    [["prompt", "PROMPT"], ["check", "CHECK"], ["inputs", "INPUTS"], ["authority", "AUTH"], ["help", "HELP"]].forEach(([key, label]) => {
        const button = make("button", "iamccs-pr-btn iamccs-pr-tab", label); button.type = "button";
        button.onclick = () => select(key);
        tabs.appendChild(button);
        const panel = make("div", "iamccs-pr-inspector-panel"); panel.dataset.panel = key; panels.set(key, panel); root.appendChild(panel);
    });
    root.insertBefore(tabs, root.children[1]);
    const promptMount = panels.get("prompt");

    function select(key) {
        tabs.querySelectorAll(".iamccs-pr-tab").forEach((item, index) => item.classList.toggle("active", ["prompt", "check", "inputs", "authority", "help"][index] === key));
        panels.forEach((panel, name) => panel.classList.toggle("active", name === key));
        if (key !== "prompt") render(key);
    }
    function render(key) {
        const project = getProject();
        const board = getShotboard();
        const panel = panels.get(key); panel.replaceChildren();
        if (key === "check") {
            panel.appendChild(make("div", "iamccs-pr-minihelp", "Prompt Check validates visible authoring and connected timeline facts. It never rewrites a field and never starts Queue."));
            const list = make("div", "iamccs-pr-check");
            inspectH3Prompt({ project, prompt: composePrompt(project), board, getWidget }).forEach((issue) => {
                const item = make("div", `iamccs-pr-checkitem ${issue.level}`); item.append(make("strong", "", issue.title), make("span", "", issue.detail)); list.appendChild(item);
            });
            panel.appendChild(list);
        } else if (key === "inputs") {
            panel.appendChild(make("div", "iamccs-pr-minihelp", "Live read-only map of the connected Shotboard. It helps you address local slots and audio lanes; files are still owned by Shotboard/CineLinX."));
            if (!board) { panel.appendChild(make("div", "iamccs-pr-mapempty", "No unique connected MiniMax H3 Shotboard found.")); return; }
            const { visual, audio } = parseTimeline(board, getWidget); const map = make("div", "iamccs-pr-map");
            const mode = String(getWidget(board, "shotboard_mode")?.value || getWidget(board, "mode")?.value || project.task_mode);
            map.appendChild(make("div", "iamccs-pr-mapcard", `MODE · ${mode}  ·  ${visual.length} visual slot(s)  ·  ${audio.length} audio segment(s)`));
            visual.forEach((row, index) => {
                const start = Number(row.start ?? row.start_seconds ?? 0); const end = Number(row.end ?? row.end_seconds ?? (start + Number(row.duration || 0)));
                const name = String(row.filename || row.file || row.image || row.id || "visual guide").split(/[\\/]/).pop();
                const local = (project.local_prompts || []).find((item) => Number(item.slot) === index + 1);
                const card = make("div", "iamccs-pr-mapcard"); card.innerHTML = `<b>VISUAL ${index + 1}</b> · ${start.toFixed(2)}–${end.toFixed(2)}s<br>${name}<br>Local prompt: ${local && String(local.prompt || "").trim() ? "READY" : "EMPTY"}`; map.appendChild(card);
            });
            audio.forEach((row, index) => {
                const start = Number(row.start ?? row.start_seconds ?? 0); const end = Number(row.end ?? row.end_seconds ?? (start + Number(row.duration || 0)));
                const lane = Number(row.lane ?? row.track ?? 0) + (row.lane == null && row.track == null ? 1 : 0);
                const card = make("div", "iamccs-pr-mapcard"); card.innerHTML = `<b>AUDIO ${index + 1}</b> · lane ${lane} · ${start.toFixed(2)}–${end.toFixed(2)}s<br>${String(row.filename || row.file || row.segment_id || row.id || "audio segment").split(/[\\/]/).pop()}`; map.appendChild(card);
            });
            if (!visual.length && !audio.length) map.appendChild(make("div", "iamccs-pr-mapempty", "The timeline has no authored visual or audio segments."));
            panel.appendChild(map);
        } else if (key === "authority") {
            panel.appendChild(make("div", "iamccs-pr-minihelp", "Assign one job to each reference. APPLY writes plain, editable sentences into visible mode fields. It does not connect files, hide prompts or queue the workflow."));
            project.authority_map = project.authority_map && typeof project.authority_map === "object" ? project.authority_map : {};
            [["Picture", 4, PICTURE_ROLES], ["Video", 2, VIDEO_ROLES], ["Audio", 2, AUDIO_ROLES]].forEach(([kind, count, roles]) => {
                for (let index = 1; index <= count; index++) {
                    const keyName = `${kind}${index}`; const row = make("div", "iamccs-pr-authrow"); row.appendChild(make("label", "", `${kind.toUpperCase()} ${index}`));
                    const select = make("select"); roles.forEach(([value, label]) => select.appendChild(new Option(label, value))); select.value = String(project.authority_map[keyName] || "off");
                    select.onchange = () => { project.authority_map[keyName] = select.value; onProjectChanged(); };
                    row.appendChild(select); panel.appendChild(row);
                }
            });
            const apply = make("button", "iamccs-pr-btn primary iamccs-pr-apply", "APPLY AUTHORITY TO VISIBLE BOXES"); apply.type = "button";
            apply.onclick = () => {
                snapshotProject?.("authority apply");
                const lines = Object.entries(project.authority_map).map(([name, role]) => roleSentence(name, role)).filter(Boolean);
                if (!lines.length) return;
                const mode = String(project.task_mode || "t2va");
                const target = mode === "ref2va" ? "retention_analysis" : mode === "v2va_object_swap" ? "v2va_replacement_retention" : mode === "fl2va" || mode === "i2va" ? "reference_use" : mode === "multi_shot_lipsync" ? "multishot_audio_contract" : mode === "audio_driven" ? "audio_drive_contract" : "scene";
                project.sections[target] = lines.join("\n"); onProjectChanged({ rerender: true }); select("prompt");
            };
            panel.appendChild(apply);
        } else if (key === "help") {
            const mode = String(project.task_mode || "t2va");
            panel.innerHTML = `<div class="iamccs-pr-help">
              <section><h4>Safe authoring order</h4><ol><li>Choose the generation mode.</li><li>Open INPUTS and verify visual slots/audio lanes.</li><li>Write REQUEST or the global boxes.</li><li>Write one LOCAL action per generated Shotboard slot.</li><li>Optionally use AI on the active box.</li><li>Use AUTH only to clarify reference jobs.</li><li>Run CHECK, then INJECT. Queue remains manual.</li></ol></section>
              <section><h4>Current mode · ${mode}</h4><p>${mode === "multi_shot_lipsync" ? "Each Shotboard image is an editorial shot opening. One AudioBoard track owns continuous lip-sync; hard cuts are expected between authored shots." : mode === "ref2va" ? "References define identity, motion, composition or audio. Assign each role explicitly; one source should not silently control everything." : mode === "v2va_object_swap" ? "Video 1 owns time, camera and motion. Pictures define only the requested replacement. This is not ControlNet or masking." : mode === "fl2va" ? "Picture 1 and Picture 2 are boundary frames connected within one shot. Do not describe them as editorial cuts." : mode === "i2va" ? "Picture 1 is the opening-frame truth. Describe the action that grows from it without redesign." : mode === "audio_driven" ? "Connected audio owns words, pauses and timing. The visible prompt maps those sounds to readable performance." : "Text defines the complete audiovisual shot. Use explicit subjects, chronology, one coherent camera idea and separated sound roles."}</p></section>
              <section><h4>Existing controls</h4><p>Writing Mode changes UI assistance only. Replace/Append controls injection into the visible Shotboard target. Temporary AI images help the LLM rewrite but never become H3 references. Tag buttons insert syntax only in the active box.</p></section>
              <section><h4>Truth and safety</h4><p>Prompter text is advisory until INJECT is pressed. After injection, the visible MiniMax Shotboard global/local boxes and Shotboard settings are queue-time truth. No inspector action starts generation.</p></section>
            </div>`;
        }
    }
    select("prompt");
    return { root, promptMount, refresh: () => { const active = [...panels].find(([, panel]) => panel.classList.contains("active"))?.[0]; if (active && active !== "prompt") render(active); }, select };
}
