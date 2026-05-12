import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const CINE_VERSION = "2026-05-10-shotboard-lite-v1";
const SHOTBOARD_NODE_MIN_SIZE = [1500, 760];
const SHOTBOARD_NODE_DEFAULT_SIZE = [1500, 780];
const SHOTBOARD_ROW_GRID = "24px 92px 182px 188px 44px 44px minmax(350px,410px) minmax(360px,1fr) 28px";
const SHOTBOARD_LITE_NODE_MIN_SIZE = [1120, 610];
const SHOTBOARD_LITE_NODE_DEFAULT_SIZE = [1180, 660];
const SHOTBOARD_LITE_ROW_GRID = "24px 92px minmax(152px,180px) minmax(138px,170px) 44px minmax(130px,170px) minmax(300px,1fr) 28px";
const FLF_SEQUENCER_NODE_MIN_SIZE = [760, 730];
const FLF_SEQUENCER_TOP_CLEARANCE = 32;
const CINE_FILM_LAB = {
    header: "#123E3A",
    nodeBg: "#07191D",
    panel: "#0D2324",
    panelDark: "#071417",
    field: "#061115",
    border: "#2C6057",
    borderSoft: "#1C4248",
    text: "#E6F3EC",
    muted: "#A8C8BE",
    guide: "#8FBE72",
    relay: "#6FB6D2",
    active: "#7CC99A",
    danger: "#B84A4A",
    dangerDark: "#6E2D31",
    button: "#143039",
    buttonHover: "#1D4852",
};
const CINE_NODE_CHROME = {
    shotboard: {
        header: "#164A43",
        nodeBg: "#071A1E",
        box: "#89C782",
        border: "#2E7567",
        glow: "rgba(112,190,156,.22)",
    },
    shotboardV2: {
        header: "#0F3E55",
        nodeBg: "#061923",
        box: "#6FB6D2",
        border: "#2D7A92",
        glow: "rgba(111,182,210,.24)",
    },
    shotboardLite: {
        header: "#4A3720",
        nodeBg: "#16110B",
        box: "#D7A84D",
        border: "#8B6A32",
        glow: "rgba(215,168,77,.18)",
    },
    shapeSync: {
        header: "#273E78",
        nodeBg: "#0D172A",
        box: "#82AFFF",
        border: "#4F75C8",
        glow: "rgba(130,175,255,.22)",
    },
    flfEngine: {
        header: "#3F4A24",
        nodeBg: "#151A0E",
        box: "#C7D36A",
        border: "#7D8B39",
        glow: "rgba(199,211,106,.18)",
    },
    flfSimple: {
        header: "#6A3D14",
        nodeBg: "#231306",
        box: "#F3B34B",
        border: "#B77428",
        glow: "rgba(243,179,75,.22)",
    },
    info: {
        header: "#2E5B48",
        nodeBg: "#0D2119",
        box: "#8BCF9B",
        border: "#4C9468",
        glow: "rgba(139,207,155,.22)",
    },
};

function applyCineChrome(node, key) {
    const chrome = CINE_NODE_CHROME[key];
    if (!node || !chrome) return chrome || {};
    node.color = chrome.header;
    node.bgcolor = chrome.nodeBg;
    node.boxcolor = chrome.box;
    return chrome;
}

const CAMERA_OPTIONS = [
    "locked-off camera",
    "slow push-in",
    "easy-in push",
    "continuous dolly-in",
    "slow pull-back",
    "macro push-in",
    "tracking shot",
    "lateral tracking",
    "orbit move",
    "subtle handheld",
    "pan left",
    "pan right",
    "tilt up",
    "tilt down",
    "crane descent",
    "aerial descent",
    "reverse angle",
    "over-the-shoulder",
    "wide reveal",
];

function normalizeCameraOption(value) {
    const raw = String(value || "").trim();
    if (!raw) return "continuous dolly-in";
    if (CAMERA_OPTIONS.includes(raw)) return raw;
    const lowered = raw.toLowerCase();
    if (/(macro|iris|pupil|eye|detail|close detail)/.test(lowered)) return "macro push-in";
    if (/(aerial|drone)/.test(lowered)) return "aerial descent";
    if (/(crane|descend|downward)/.test(lowered)) return "crane descent";
    if (/(reverse)/.test(lowered)) return "reverse angle";
    if (/(over.?the.?shoulder|ots|behind)/.test(lowered)) return "over-the-shoulder";
    if (/(close-up|close up|medium close|medium shot|portrait)/.test(lowered)) return "slow push-in";
    if (/(wide|establish|reveal)/.test(lowered)) return "wide reveal";
    if (/(pull|back|zoom out|dolly out)/.test(lowered)) return "slow pull-back";
    if (/(handheld|drift)/.test(lowered)) return "subtle handheld";
    if (/(locked|lock|static|still)/.test(lowered)) return "locked-off camera";
    if (/(orbit|arc)/.test(lowered)) return "orbit move";
    if (/(pan left)/.test(lowered)) return "pan left";
    if (/(pan right)/.test(lowered)) return "pan right";
    if (/(tilt up)/.test(lowered)) return "tilt up";
    if (/(tilt down)/.test(lowered)) return "tilt down";
    if (/(track|follow|lateral|side)/.test(lowered)) return "tracking shot";
    if (/(easy|ease)/.test(lowered)) return "easy-in push";
    if (/(push|dolly|forward|travel|approach|ocean|sea|coast|wave|surf|water)/.test(lowered)) return "continuous dolly-in";
    return "continuous dolly-in";
}

function nodeClassName(node) {
    return String(node?.comfyClass || node?.type || node?.constructor?.comfyClass || "");
}

function getWidget(node, name) {
    return node?.widgets?.find((widget) => widget?.name === name || widget?.label === name) || null;
}

function hideWidget(widget) {
    if (!widget) return;
    if (widget._iamccsCineHidden) {
        widget.type = "hidden";
        widget.hidden = true;
        widget.computeSize = () => [0, 0];
        widget.draw = () => {};
        return;
    }
    widget._iamccsCineOrigType = widget.type;
    widget._iamccsCineOrigCompute = widget.computeSize;
    widget._iamccsCineOrigDraw = widget.draw;
    widget.type = "hidden";
    widget.hidden = true;
    widget.computeSize = () => [0, 0];
    widget.draw = () => {};
    widget._iamccsCineHidden = true;
}

function showWidget(widget) {
    if (!widget) return;
    if (widget._iamccsCineOrigType) widget.type = widget._iamccsCineOrigType;
    if (widget._iamccsCineOrigCompute) widget.computeSize = widget._iamccsCineOrigCompute;
    if (widget._iamccsCineOrigDraw) widget.draw = widget._iamccsCineOrigDraw;
    widget.hidden = false;
    widget._iamccsCineHidden = false;
}

function hideWidgetAsLinkedSlot(widget, label = "linked timeline input") {
    if (!widget) return;
    widget.type = "iamccs_linked_slot";
    widget.hidden = false;
    widget.computeSize = (width) => [width || 0, 24];
    widget.draw = (ctx, node, width, y) => {
        if (!ctx || typeof y !== "number") return;
        ctx.save();
        ctx.fillStyle = CINE_FILM_LAB.muted;
        ctx.font = "10px sans-serif";
        ctx.textBaseline = "middle";
        ctx.fillText(label, 32, y + 12);
        ctx.restore();
    };
    widget._iamccsCineHidden = true;
}

function setWidgetValue(node, name, value) {
    const widget = getWidget(node, name);
    if (!widget) return false;
    widget.value = value;
    try { widget.callback?.(value, app.canvas, node); } catch {}
    try { node.setDirtyCanvas?.(true, true); app.graph?.setDirtyCanvas?.(true, true); } catch {}
    try { node.graph?.change?.(); app.graph?.change?.(); } catch {}
    return true;
}

function lockNodeMinimumSize(node, minSize, options = {}) {
    if (!node || !Array.isArray(minSize)) return;
    const minWidth = Number(minSize[0]) || 0;
    const minHeight = Number(minSize[1]) || 0;
    node._iamccsCineMinSize = [minWidth, minHeight];
    if (options.lockResize) {
        node.resizable = false;
        node.resizeable = false;
    }

    const applySize = () => {
        const currentWidth = Number(node.size?.[0] || 0);
        const currentHeight = Number(node.size?.[1] || 0);
        const preferred = Array.isArray(options.preferredSize) ? options.preferredSize : null;
        const preferredWidth = Number(preferred?.[0] || 0);
        const preferredHeight = Number(preferred?.[1] || 0);
        const next = preferred
            ? [Math.max(minWidth, preferredWidth), Math.max(minHeight, preferredHeight)]
            : [Math.max(minWidth, currentWidth), Math.max(minHeight, currentHeight)];
        if (currentWidth !== next[0] || currentHeight !== next[1]) {
            if (typeof node.setSize === "function") node.setSize(next);
            else node.size = next;
        }
    };

    applySize();
    if (node._iamccsCineMinSizeWrapped) return;
    const originalOnResize = node.onResize;
    node.onResize = function (size) {
        const min = this._iamccsCineMinSize || minSize;
        if (Array.isArray(size)) {
            size[0] = Math.max(Number(min?.[0] || 0), Number(size[0] || 0));
            size[1] = Math.max(Number(min?.[1] || 0), Number(size[1] || 0));
        }
        const result = originalOnResize ? originalOnResize.apply(this, arguments) : undefined;
        const width = Math.max(Number(min?.[0] || 0), Number(this.size?.[0] || 0));
        const height = Math.max(Number(min?.[1] || 0), Number(this.size?.[1] || 0));
        if (this.size?.[0] !== width || this.size?.[1] !== height) {
            if (typeof this.setSize === "function") this.setSize([width, height]);
            else this.size = [width, height];
        }
        return result;
    };
    node._iamccsCineMinSizeWrapped = true;
}

function parseJsonWidget(node, fallback) {
    const widget = getWidget(node, "timeline_data");
    const text = String(widget?.value || "").trim();
    if (!text) return fallback();
    try {
        const data = JSON.parse(text);
        if (Array.isArray(data)) return data;
        if (Array.isArray(data.keyframes)) return data.keyframes;
        if (Array.isArray(data.segments)) return data.segments;
        if (Array.isArray(data.rows)) return data.rows;
    } catch {
        // Existing pipe-line workflows still run in Python; the UI starts from a fresh preset.
    }
    return fallback();
}

function writeKeyframes(node, rows, metadata = null) {
    const cleanMetadata = metadata && typeof metadata === "object" ? metadata : {};
    setWidgetValue(node, "timeline_data", JSON.stringify({ ...cleanMetadata, keyframes: rows }, null, 2));
}

function getOriginNodeFromLink(linkId) {
    if (!linkId || !app?.graph?.links) return null;
    const link = app.graph.links[linkId];
    if (!link) return null;
    const originId = link.origin_id ?? link.source_id;
    if (originId == null || typeof app.graph.getNodeById !== "function") return null;
    return app.graph.getNodeById(originId) || null;
}

function getLinkedOriginNode(node, inputName) {
    const input = node?.inputs?.find((item) => item?.name === inputName);
    return getOriginNodeFromLink(input?.link);
}

function traceLinkedCinePlanner(node) {
    const startInputs = ["timeline_data", "multi_input", "duration_seconds"];
    const accepted = new Set(["IAMCCS_CineShotboardLite", "IAMCCS_CineShotboardPlannerPro", "IAMCCS_CineShotboardPlannerProV2", "IAMCCS_CineShotboardPlannerProLegacy", "IAMCCS_CineShotboardTimelinePro", "IAMCCS_CineReferenceBoard"]);
    for (const inputName of startInputs) {
        let current = getLinkedOriginNode(node, inputName);
        const visited = new Set();
        for (let depth = 0; current && depth < 8; depth += 1) {
            if (visited.has(current.id)) break;
            visited.add(current.id);
            const cls = nodeClassName(current);
            if (accepted.has(cls)) return current;
            if (cls === "IAMCCS_CineInfo" || cls === "IAMCCS_CinePromptModeSwitch" || cls === "IAMCCS_CinePromptRelayImageBridge") {
                current = getLinkedOriginNode(current, "cine_linx");
                continue;
            }
            if (cls === "Reroute" || current.type === "Reroute") {
                current = getOriginNodeFromLink(current.inputs?.[0]?.link);
                continue;
            }
            break;
        }
    }
    return null;
}

function readPlannerLinkedFLFState(node) {
    const planner = traceLinkedCinePlanner(node);
    if (!planner) return null;
    const timelineText = String(getWidget(planner, "timeline_data")?.value || "").trim();
    let plannerRows = [];
    if (timelineText) {
        try {
            const parsed = JSON.parse(timelineText);
            plannerRows = parsed.rows || parsed.keyframes || parsed.timeline || [];
        } catch {
            plannerRows = [];
        }
    }
    const normalized = Array.isArray(plannerRows) ? plannerRows.map(normalizeShotboardRow) : [];
    const guideRows = normalized.filter((row) => row.use_guide !== false && Number(row.force || 0) > 0);
    const rowsForFLF = (guideRows.length ? guideRows : normalized).map((row, index) => ({
        second: Number(row.second || 0),
        ref: Math.max(1, Number(row.ref || index + 1)),
        strength: Math.max(0, Math.min(1, Number(row.force ?? row.strength ?? 0.18))),
        label: row.label || `guide_${index + 1}`,
        camera: [row.camera, row.transition, row.note].filter(Boolean).join(" | "),
    }));
    const duration = Number(getWidget(planner, "duration_seconds")?.value || 0);
    const fps = Number(getWidget(planner, "frame_rate")?.value || 0);
    const refs = splitReferencePaths(getWidget(planner, "image_paths")?.value).length;
    return {
        planner,
        rows: rowsForFLF,
        duration: Number.isFinite(duration) && duration > 0 ? duration : null,
        fps: Number.isFinite(fps) && fps > 0 ? fps : null,
        refs,
    };
}

function writeSegments(node, rows) {
    setWidgetValue(node, "timeline_data", JSON.stringify({ segments: rows }, null, 2));
}

function inputBase() {
    return `
        width: 100%;
        box-sizing: border-box;
        border: 1px solid ${CINE_FILM_LAB.border};
        background: ${CINE_FILM_LAB.field};
        color: ${CINE_FILM_LAB.text};
        border-radius: 4px;
        padding: 6px 7px;
        font-size: 12px;
        outline: none;
    `;
}

function button(label, tone = "neutral", palette = null) {
    const btn = document.createElement("button");
    btn.textContent = label;
    const bg = tone === "primary"
        ? (palette?.primaryBg || "#174F63")
        : tone === "danger"
            ? (palette?.dangerBg || CINE_FILM_LAB.dangerDark)
            : (palette?.neutralBg || CINE_FILM_LAB.button);
    const border = tone === "primary"
        ? (palette?.primaryBorder || CINE_FILM_LAB.guide)
        : tone === "danger"
            ? (palette?.dangerBorder || CINE_FILM_LAB.danger)
            : (palette?.neutralBorder || CINE_FILM_LAB.border);
    const color = palette?.text || CINE_FILM_LAB.text;
    btn.style.cssText = `
        background: ${bg};
        border: 1px solid ${border};
        color: ${color};
        border-radius: 4px;
        padding: 6px 9px;
        cursor: pointer;
        font-size: 12px;
        white-space: nowrap;
    `;
    return btn;
}

function protectControlDrag(element) {
    if (!element) return element;
    element.draggable = false;
    element.setAttribute?.("draggable", "false");
    for (const eventName of ["pointerdown", "pointermove", "mousedown", "mousemove", "touchstart", "touchmove", "wheel", "drag", "dragover", "drop"]) {
        element.addEventListener(eventName, (event) => {
            event.stopPropagation();
        }, { passive: false, capture: true });
    }
    element.addEventListener("dragstart", (event) => {
        event.preventDefault();
        event.stopPropagation();
    }, { capture: true });
    return element;
}

function protectDragHandle(element) {
    if (!element) return element;
    for (const eventName of ["pointerdown", "mousedown", "touchstart"]) {
        element.addEventListener(eventName, (event) => {
            event.stopPropagation();
        }, { passive: false, capture: true });
    }
    return element;
}

function tableShell(title, subtitle) {
    const root = document.createElement("div");
    root.style.cssText = `
        width: 100%;
        box-sizing: border-box;
        background: ${CINE_FILM_LAB.panel};
        border: 1px solid ${CINE_FILM_LAB.border};
        border-radius: 6px;
        padding: 8px;
        color: ${CINE_FILM_LAB.text};
        font-family: Arial, sans-serif;
        pointer-events: auto;
    `;

    const head = document.createElement("div");
    head.style.cssText = "display:flex; flex-direction:column; gap:2px; margin-bottom:7px;";
    const h = document.createElement("div");
    h.textContent = title;
    h.style.cssText = `font-weight:700; font-size:14px; color:${CINE_FILM_LAB.text};`;
    const s = document.createElement("div");
    s.textContent = subtitle;
    s.style.cssText = `font-size:11px; color:${CINE_FILM_LAB.muted}; line-height:1.35;`;
    head.appendChild(h);
    if (String(subtitle || "").trim()) head.appendChild(s);
    root.appendChild(head);

    const toolbar = document.createElement("div");
    toolbar.style.cssText = "display:flex; flex-wrap:wrap; gap:6px; margin-bottom:7px;";
    root.appendChild(toolbar);

    const table = document.createElement("div");
    table.style.cssText = "display:flex; flex-direction:column; gap:7px; width:100%; max-width:100%; min-width:0; overflow:hidden;";
    root.appendChild(table);

    return { root, toolbar, table };
}

function defaultKeyframes() {
    return [
        { second: 0.0, ref: 1, strength: 0.78, label: "opening_anchor", camera: "first reference starts alive; avoid a static slide" },
        { second: 2.2, ref: 2, strength: 0.08, label: "micro_motion", camera: "subtle motion only; avoid hard still-frame anchor" },
        { second: 4.8, ref: 3, strength: 0.24, label: "approach_anchor", camera: "smooth camera approach before the next visual beat" },
        { second: 6.9, ref: 4, strength: 0.24, label: "detail_anchor", camera: "macro detail stays sharp, no jitter" },
        { second: 8.9, ref: 5, strength: 0.26, label: "transition_anchor", camera: "visual transition opens gradually through physical travel" },
        { second: 10.9, ref: 6, strength: 0.18, label: "environment_entry", camera: "new space enters as physical travel, no overlay" },
        { second: 12.9, ref: 7, strength: 0.18, label: "environment_motion", camera: "descending or tracking motion, no cross dissolve" },
        { second: 14.7, ref: 8, strength: 0.18, label: "texture_anchor", camera: "surface detail grows through parallax" },
        { second: 16.6, ref: 10, strength: 0.26, label: "final_motion", camera: "final subject or environment stays moving" },
        { second: 18.7, ref: 11, strength: 0.18, label: "end_anchor", camera: "camera keeps moving until the end" },
    ];
}

function defaultSegments() {
    return [
        { seconds: 4.0, prompt: "opening subject or place stays alive with subtle natural motion, not frozen", camera: "slow push-in" },
        { seconds: 3.0, prompt: "camera approaches the next detail with parallax and stable focus", camera: "easy-in push" },
        { seconds: 3.0, prompt: "visual transition develops gradually through physical travel, no dissolve", camera: "macro push-in" },
        { seconds: 3.0, prompt: "new environment or object emerges coherently through motion", camera: "continuous dolly-in" },
        { seconds: 3.0, prompt: "camera continues through space, texture and scale grow naturally", camera: "tracking shot" },
        { seconds: 2.5, prompt: "final beat keeps real motion until the end", camera: "continuous dolly-in" },
    ];
}

function normalizeKeyframe(row, index) {
    return {
        second: Number.isFinite(Number(row.second)) ? Number(row.second) : index * 2,
        ref: Number.isFinite(Number(row.ref ?? row.reference_index)) ? Number(row.ref ?? row.reference_index) : index + 1,
        strength: Number.isFinite(Number(row.strength)) ? Number(row.strength) : 0.82,
        label: String(row.label || `key_${index + 1}`),
        camera: normalizeCameraOption(row.camera || row.camera_note || "continuous dolly-in"),
    };
}

function normalizeSegment(row, index) {
    return {
        seconds: Number.isFinite(Number(row.seconds ?? row.duration)) ? Number(row.seconds ?? row.duration) : 3,
        prompt: String(row.prompt || row.local_prompt || `cinematic beat ${index + 1}, natural motion`),
        camera: normalizeCameraOption(row.camera || "slow push-in"),
    };
}

function makeSelect(value, options, onChange) {
    const select = document.createElement("select");
    select.style.cssText = inputBase();
    for (const option of options) {
        const opt = document.createElement("option");
        opt.value = option;
        opt.textContent = option;
        select.appendChild(opt);
    }
    if (!options.includes(value)) {
        const custom = document.createElement("option");
        custom.value = value;
        custom.textContent = value || "custom";
        select.appendChild(custom);
    }
    select.value = value;
    select.onchange = () => onChange(select.value);
    return protectControlDrag(select);
}

function makeChoiceSelect(value, choices, onChange) {
    const select = document.createElement("select");
    select.style.cssText = inputBase();
    const values = choices.map((choice) => choice.value);
    for (const choice of choices) {
        const opt = document.createElement("option");
        opt.value = choice.value;
        opt.textContent = choice.label;
        select.appendChild(opt);
    }
    if (!values.includes(value)) {
        const custom = document.createElement("option");
        custom.value = value;
        custom.textContent = value || "custom";
        select.appendChild(custom);
    }
    select.value = value;
    select.onchange = () => onChange(select.value);
    return protectControlDrag(select);
}

function renderKeyframeEditor(node) {
    if (node._iamccsCineTimelineReady) return;
    node._iamccsCineTimelineReady = true;
    lockNodeMinimumSize(node, FLF_SEQUENCER_NODE_MIN_SIZE, { preferredSize: FLF_SEQUENCER_NODE_MIN_SIZE });
    node.size = [
        Math.max(node.size?.[0] || 620, FLF_SEQUENCER_NODE_MIN_SIZE[0]),
        Math.max(node.size?.[1] || 520, FLF_SEQUENCER_NODE_MIN_SIZE[1]),
    ];
    const chrome = applyCineChrome(node, "flfEngine");

    const raw = getWidget(node, "timeline_data");
    hideWidgetAsLinkedSlot(raw, "timeline_data from ShotPlanner / CineInfo");

    let rows = parseJsonWidget(node, defaultKeyframes).map(normalizeKeyframe);
    const { root, toolbar, table } = tableShell(
        "Cine FLF Timeline",
        "Single generation. Add reference keyframes by seconds: face, eye, object, environment, waves."
    );
    root.style.marginTop = `${FLF_SEQUENCER_TOP_CLEARANCE}px`;

    const addBtn = button("Add Key", "primary");
    const presetFaceBtn = button("Preset Continuous Path");
    const presetDialogBtn = button("Preset Shot/Reverse");
    const syncPlannerBtn = button("Sync ShotPlanner");
    const clearBtn = button("Clear", "danger");
    toolbar.append(addBtn, presetFaceBtn, presetDialogBtn, syncPlannerBtn, clearBtn);

    const linkedStatus = document.createElement("div");
    linkedStatus.style.cssText = `margin:6px 0 8px 0;padding:7px 9px;border:1px solid ${chrome.border};border-radius:5px;background:${CINE_FILM_LAB.panelDark};color:${CINE_FILM_LAB.muted};font-size:11px;box-shadow:inset 0 1px 0 ${chrome.glow};`;
    linkedStatus.textContent = "Manual FLF timeline. If linked from CineInfo/ShotPlanner, use Sync ShotPlanner or leave auto-sync on.";
    root.insertBefore(linkedStatus, table);

    let linkedSignature = "";
    let linkedTimelineMetadata = {};

    function applyLinkedPlannerState(redraw = true) {
        const state = readPlannerLinkedFLFState(node);
        if (!state || !state.rows.length) {
            linkedTimelineMetadata = {};
            linkedStatus.textContent = "Manual FLF timeline. No linked ShotPlanner rows detected.";
            return false;
        }
        const signature = JSON.stringify({
            rows: state.rows,
            duration: state.duration,
            fps: state.fps,
            refs: state.refs,
        });
        if (signature === linkedSignature) {
            return true;
        }
        linkedSignature = signature;
        rows = state.rows.map(normalizeKeyframe);
        linkedTimelineMetadata = {
            duration_seconds: state.duration,
            frame_rate: state.fps,
            reference_count: state.refs,
            source: "shotplanner_linked_auto_sync",
        };
        if (state.duration != null) setWidgetValue(node, "duration_seconds", state.duration);
        if (state.fps != null) setWidgetValue(node, "frame_rate", Math.round(state.fps));
        if (state.refs > 0) setWidgetValue(node, "fallback_num_images", state.refs);
        sync();
        linkedStatus.textContent = `Synced from ShotPlanner: ${rows.length} FLF guides, ${state.refs || rows.length} refs, ${state.duration ?? "?"}s @ ${state.fps ?? "?"}fps.`;
        if (redraw) draw();
        return true;
    }

    function sync() {
        rows = rows.map(normalizeKeyframe);
        writeKeyframes(node, rows, linkedTimelineMetadata);
    }

    function draw() {
        table.innerHTML = "";
        const header = document.createElement("div");
        header.style.cssText = "display:grid; grid-template-columns:88px 46px 64px 92px 1fr 28px; gap:6px; color:#9fb0bd; font-size:10px; padding:0 2px;";
        header.innerHTML = "<div>Sec</div><div>Ref</div><div>Force</div><div>Label</div><div>Camera / note</div><div></div>";
        table.appendChild(header);

        rows.forEach((row, index) => {
            const r = normalizeKeyframe(row, index);
            const line = document.createElement("div");
            line.style.cssText = "display:grid; grid-template-columns:88px 46px 64px 92px 1fr 28px; gap:6px; align-items:center;";

            const sec = timeControl(r.second, (value) => { rows[index].second = value; sync(); });

            const ref = document.createElement("input");
            ref.type = "number"; ref.min = "1"; ref.max = "50"; ref.step = "1"; ref.value = r.ref; ref.style.cssText = inputBase();
            ref.oninput = () => { rows[index].ref = Number(ref.value); sync(); };

            const strength = document.createElement("input");
            strength.type = "number"; strength.min = "0"; strength.max = "1"; strength.step = "0.01"; strength.value = r.strength; strength.style.cssText = inputBase();
            strength.oninput = () => { rows[index].strength = Number(strength.value); sync(); };

            const label = document.createElement("input");
            label.value = r.label; label.style.cssText = inputBase();
            label.oninput = () => { rows[index].label = label.value; sync(); };

            const camera = document.createElement("input");
            camera.value = r.camera; camera.style.cssText = inputBase();
            camera.setAttribute("list", `iamccs_cine_camera_${node.id || "x"}`);
            camera.oninput = () => { rows[index].camera = camera.value; sync(); };

            const del = button("x", "danger");
            del.onclick = () => {
                if (rows.length <= 1) return;
                rows.splice(index, 1);
                sync();
                draw();
            };

            line.append(sec, ref, strength, label, camera, del);
            table.appendChild(line);
        });

        const list = document.createElement("datalist");
        list.id = `iamccs_cine_camera_${node.id || "x"}`;
        CAMERA_OPTIONS.forEach((opt) => {
            const item = document.createElement("option");
            item.value = opt;
            list.appendChild(item);
        });
        table.appendChild(list);
    }

    addBtn.onclick = () => { rows.push({ second: rows.length ? Number(rows[rows.length - 1].second) + 2 : 0, ref: rows.length + 1, strength: 0.78, label: `key_${rows.length + 1}`, camera: "continuous dolly-in" }); sync(); draw(); };
    presetFaceBtn.onclick = () => { rows = defaultKeyframes(); sync(); draw(); };
    presetDialogBtn.onclick = () => { rows = [
        { second: 0.0, ref: 1, strength: 0.94, label: "field_A", camera: "medium close-up, person A speaks" },
        { second: 4.0, ref: 2, strength: 0.88, label: "reverse_B", camera: "reverse close-up, person B answers" },
        { second: 7.5, ref: 1, strength: 0.84, label: "return_A", camera: "return to person A reaction" },
        { second: 11.0, ref: 3, strength: 0.74, label: "wide", camera: "wide reveal of room or city environment" },
    ]; sync(); draw(); };
    syncPlannerBtn.onclick = () => { applyLinkedPlannerState(true); };
    clearBtn.onclick = () => { rows = [defaultKeyframes()[0]]; sync(); draw(); };

    applyLinkedPlannerState(false);
    draw();
    sync();
    if (!node._iamccsCineFLFLinkedSyncTimer) {
        node._iamccsCineFLFLinkedSyncTimer = setInterval(() => {
            if (!node.graph) {
                clearInterval(node._iamccsCineFLFLinkedSyncTimer);
                node._iamccsCineFLFLinkedSyncTimer = null;
                return;
            }
            applyLinkedPlannerState(true);
        }, 750);
    }
    const widget = node.addDOMWidget("Cine FLF Timeline", "iamccs_cine_flf_timeline", root, { serialize: false });
    widget.computeSize = (width) => [width, Math.max(320, FLF_SEQUENCER_TOP_CLEARANCE + 104 + rows.length * 38)];
}

function renderPromptRelayEditor(node) {
    if (node._iamccsCinePromptReady) return;
    node._iamccsCinePromptReady = true;
    node.size = [Math.max(node.size?.[0] || 640, 700), Math.max(node.size?.[1] || 520, 640)];
    node.color = CINE_FILM_LAB.header;
    node.bgcolor = CINE_FILM_LAB.nodeBg;
    node.boxcolor = CINE_FILM_LAB.relay;

    const raw = getWidget(node, "timeline_data");
    hideWidget(raw);

    let rows = parseJsonWidget(node, defaultSegments).map(normalizeSegment);
    const { root, toolbar, table } = tableShell(
        "Cine PromptRelay Timeline",
        "Temporal prompt beats. Connect outputs to PromptRelayEncodeTimeline or PromptRelayEncode."
    );

    const addBtn = button("Add Beat", "primary");
    const facePreset = button("Preset Continuous");
    const dialoguePreset = button("Preset Dialogue");
    const actionPreset = button("Preset Action");
    const clearBtn = button("Clear", "danger");
    toolbar.append(addBtn, facePreset, dialoguePreset, actionPreset, clearBtn);

    function sync() {
        rows = rows.map(normalizeSegment);
        writeSegments(node, rows);
    }

    function draw() {
        table.innerHTML = "";
        const header = document.createElement("div");
        header.style.cssText = "display:grid; grid-template-columns:66px 1fr 150px 28px; gap:6px; color:#9fb0bd; font-size:10px; padding:0 2px;";
        header.innerHTML = "<div>Seconds</div><div>Local prompt</div><div>Camera</div><div></div>";
        table.appendChild(header);

        rows.forEach((row, index) => {
            const r = normalizeSegment(row, index);
            const line = document.createElement("div");
            line.style.cssText = "display:grid; grid-template-columns:66px 1fr 150px 28px; gap:6px; align-items:start;";

            const seconds = document.createElement("input");
            seconds.type = "number"; seconds.step = "0.1"; seconds.min = "0.1"; seconds.value = r.seconds; seconds.style.cssText = inputBase();
            seconds.oninput = () => { rows[index].seconds = Number(seconds.value); sync(); };

            const prompt = document.createElement("textarea");
            prompt.value = r.prompt;
            prompt.rows = 2;
            prompt.style.cssText = inputBase() + "resize: vertical; min-height: 44px;";
            prompt.oninput = () => { rows[index].prompt = prompt.value; sync(); };

            const camera = makeSelect(r.camera, CAMERA_OPTIONS, (value) => { rows[index].camera = value; sync(); });
            const del = button("x", "danger");
            del.onclick = () => {
                if (rows.length <= 1) return;
                rows.splice(index, 1);
                sync();
                draw();
            };

            line.append(seconds, prompt, camera, del);
            table.appendChild(line);
        });
    }

    addBtn.onclick = () => { rows.push({ seconds: 3, prompt: `cinematic beat ${rows.length + 1}, natural motion`, camera: "slow push-in" }); sync(); draw(); };
    facePreset.onclick = () => { rows = defaultSegments(); sync(); draw(); };
    dialoguePreset.onclick = () => { rows = [
        { seconds: 4, prompt: "person A speaks in close-up, natural mouth and eyes, subtle emotion", camera: "medium close-up" },
        { seconds: 3.5, prompt: "person B answers in reverse angle, controlled reaction, no identity drift", camera: "reverse angle" },
        { seconds: 3, prompt: "return to person A listening, micro expression, cinematic pause", camera: "slow push-in" },
        { seconds: 4, prompt: "wide shot reveals both people and the environment", camera: "wide reveal" },
    ]; sync(); draw(); };
    actionPreset.onclick = () => { rows = [
        { seconds: 3, prompt: "subject prepares the action, breathing and small anticipatory movement", camera: "medium close-up" },
        { seconds: 2.5, prompt: "the action begins with clear body movement and readable direction", camera: "slow pull-back" },
        { seconds: 4, prompt: "the action resolves from a new angle while preserving screen direction", camera: "over-the-shoulder" },
    ]; sync(); draw(); };
    clearBtn.onclick = () => { rows = [defaultSegments()[0]]; sync(); draw(); };

    draw();
    sync();
    const widget = node.addDOMWidget("Cine PromptRelay Timeline", "iamccs_cine_promptrelay_timeline", root, { serialize: false });
    widget.computeSize = (width) => [width, Math.max(310, 116 + rows.length * 62)];
}


const TRANSITION_OPTIONS = [
    "continuous_motion",
    "soft_morph",
    "match_cut",
    "hard_cut",
];

const CAMERA_RELAY_OPTIONS = ["off", "before", "after"];
const TRANSITION_RELAY_OPTIONS = ["off", "safe_only", "append"];
const RELAY_ADDON_POSITION_OPTIONS = ["after", "before"];

function normalizeOption(value, options, fallback) {
    const raw = String(value || "").trim();
    return options.includes(raw) ? raw : fallback;
}

function cameraRelayText(row) {
    const camera = normalizeCameraOption(row.camera || "").toLowerCase().replace("_", " ");
    const phrases = {
        "locked-off camera": "locked-off camera, stable framing, subtle subject motion",
        "slow push-in": "slow push-in, gentle forward camera movement",
        "easy-in push": "easy-in push, gradual acceleration into the move",
        "continuous dolly-in": "continuous forward dolly, physical travel and growing parallax",
        "slow pull-back": "slow pull-back, widening space and revealing context",
        "macro push-in": "macro push-in, sharp detail, micro-parallax, stable focus",
        "tracking shot": "tracking shot, smooth camera follow-through",
        "lateral tracking": "lateral tracking movement, side parallax",
        "orbit move": "orbiting camera move, curved parallax around the subject",
        "subtle handheld": "subtle handheld energy, natural micro-movement",
        "pan left": "slow pan left, continuous horizontal camera motion",
        "pan right": "slow pan right, continuous horizontal camera motion",
        "tilt up": "tilt up, continuous vertical camera movement",
        "tilt down": "tilt down, continuous vertical camera movement",
        "crane descent": "crane descent, smooth downward camera travel",
        "aerial descent": "aerial descent, smooth downward travel from above",
        "reverse angle": "reverse angle staging, coherent screen direction",
        "over-the-shoulder": "over-the-shoulder staging, foreground and background depth",
        "wide reveal": "wide reveal, expanding composition and environmental context",
    };
    return phrases[camera] || "";
}

function transitionRelayParts(row, nextRow, mode = "safe_only") {
    const transition = String(row.transition || "continuous_motion").trim();
    const parts = [];
    if (transition === "hard_cut") {
        if (mode === "append") parts.push("hard cut staging; do not morph identities inside this single shot");
        return parts;
    }
    if (transition === "match_cut") {
        parts.push("match movement continuity through shape and camera direction");
    } else if (transition === "soft_morph") {
        parts.push("single continuous transformation, avoid visible cross dissolve");
    } else {
        parts.push("continuous physical camera movement, no slideshow, no cross dissolve");
    }
    if (nextRow && transition !== "hard_cut") {
        const nextLabel = String(nextRow.label || "next target").replace(/_/g, " ");
        parts.push(`move toward ${nextLabel} without flashing reference frames`);
    }
    return parts;
}

function composeRelayPromptPreview(row, nextRow) {
    const normalized = normalizeShotboardRow(row, 0);
    const base = String(normalized.relay_prompt || "").trim();
    const before = [];
    const after = [];
    const cameraMode = normalizeOption(normalized.camera_relay_mode, CAMERA_RELAY_OPTIONS, "off");
    const transitionMode = normalizeOption(normalized.transition_relay_mode, TRANSITION_RELAY_OPTIONS, "off");
    const addonPosition = normalizeOption(normalized.relay_addon_position, RELAY_ADDON_POSITION_OPTIONS, "after");
    const cameraText = cameraRelayText(normalized);
    if (cameraText && cameraMode === "before") before.push(cameraText);
    if (cameraText && cameraMode === "after") after.push(cameraText);
    if (transitionMode !== "off") after.push(...transitionRelayParts(normalized, nextRow, transitionMode));
    const addon = String(normalized.relay_modifier_text || "").trim();
    if (addon && addonPosition === "before") before.push(addon);
    if (addon && addonPosition === "after") after.push(addon);
    return [...before, base, ...after].filter(Boolean).join(", ");
}

function defaultShotboardRows() {
    return [
        { second: 0.0, ref: 1, force: 0.62, use_guide: true, use_prompt: true, label: "opening_anchor", camera: "slow push-in", transition: "continuous_motion", camera_relay_mode: "before", transition_relay_mode: "off", relay_addon_position: "after", note: "", relay_prompt: "opening subject or place stays alive with subtle natural motion" },
        { second: 4.0, ref: 2, force: 0.0, use_guide: false, use_prompt: true, label: "approach_motion", camera: "easy-in push", transition: "continuous_motion", camera_relay_mode: "before", transition_relay_mode: "off", relay_addon_position: "after", note: "", relay_prompt: "camera advances through the same space with growing parallax" },
        { second: 7.0, ref: 3, force: 0.22, use_guide: true, use_prompt: true, label: "midpoint_anchor", camera: "macro push-in", transition: "soft_morph", camera_relay_mode: "before", transition_relay_mode: "off", relay_addon_position: "after", note: "", relay_prompt: "soft visual checkpoint, keep focus stable and avoid a still-frame flash" },
        { second: 10.0, ref: 4, force: 0.0, use_guide: false, use_prompt: true, label: "environment_shift", camera: "continuous dolly-in", transition: "continuous_motion", camera_relay_mode: "before", transition_relay_mode: "off", relay_addon_position: "after", note: "", relay_prompt: "the next space appears through physical travel, not a slideshow transition" },
        { second: 13.0, ref: 5, force: 0.0, use_guide: false, use_prompt: false, label: "prompt_continuity", camera: "tracking shot", transition: "continuous_motion", camera_relay_mode: "before", transition_relay_mode: "off", relay_addon_position: "after", note: "", relay_prompt: "optional prompt-only continuity row; keep previous visual direction" },
        { second: 16.0, ref: 6, force: 0.18, use_guide: true, use_prompt: true, label: "final_anchor", camera: "continuous dolly-in", transition: "continuous_motion", camera_relay_mode: "before", transition_relay_mode: "off", relay_addon_position: "after", note: "", relay_prompt: "final soft anchor, keep motion alive until the end" },
    ];
}

function firstNonEmpty(...values) {
    for (const value of values) {
        const text = String(value ?? "").trim();
        if (text) return text;
    }
    return "";
}

function rowHasCanonicalRelayPrompt(row) {
    return Boolean(firstNonEmpty(
        row?.relay_prompt,
        row?.local_prompt,
        row?.prompt_beat,
        row?.beat_prompt,
        row?.video_prompt,
        row?.action_prompt,
        row?.prompt,
        row?.localPrompt,
        row?.relayPrompt,
        row?.promptLocal,
    ));
}

function relayPromptCountInRows(sourceRows, options = {}) {
    return (Array.isArray(sourceRows) ? sourceRows : [])
        .map((row, index) => normalizeShotboardRow(row, index, options))
        .filter((row) => String(row.relay_prompt || "").trim())
        .length;
}

function parseShotboardTimelineString(value) {
    if (typeof value !== "string" || !value.trim()) return [];
    try {
        const parsed = JSON.parse(value);
        if (Array.isArray(parsed)) return parsed;
        return parsed.rows || parsed.keyframes || parsed.shotboard || parsed.timeline || [];
    } catch {
        return [];
    }
}

function boardFromWorkflowJson(data) {
    if (!Array.isArray(data?.nodes)) return null;
    const node = data.nodes.find((item) => {
        const type = String(item?.type || item?.class_type || "");
        return type === "IAMCCS_CineShotboardLite" || type === "IAMCCS_CineShotboardPlannerPro" || type === "IAMCCS_CineShotboardPlannerProV2" || type === "IAMCCS_CineShotboardPlannerProLegacy" || type === "IAMCCS_CineShotboardTimelinePro";
    });
    if (!node) return null;
    const widgets = Array.isArray(node.widgets_values) ? node.widgets_values : [];
    const isLite = String(node?.type || node?.class_type || "") === "IAMCCS_CineShotboardLite";
    return {
        metadata: {
            schema: "iamccs.cine.shotboard.board",
            schema_version: 0,
            imported_from: "comfy_workflow_shotplanner_node",
            source_node_id: node.id,
        },
        global_prompt: String(widgets[0] || ""),
        timeline_data: String(widgets[1] || ""),
        duration_seconds: widgets[2],
        frame_rate: widgets[3],
        guide_policy: widgets[4],
        min_guide_gap_seconds: widgets[5],
        max_guides: widgets[6],
        default_force: widgets[7],
        promptrelay_epsilon: isLite ? 0.65 : widgets[8],
        ltx_round_mode: isLite ? "up_8n_plus_1" : widgets[9],
        image_paths: String(widgets[isLite ? 8 : 10] || "").split(/\r?\n/).map((line) => line.trim()).filter(Boolean),
        image_width: widgets[isLite ? 9 : 11],
        image_height: widgets[isLite ? 10 : 12],
    };
}

function normalizeShotboardRow(row, index, options = {}) {
    const force = Number(row.force ?? row.strength);
    const legacyModifiers = row.use_relay_modifiers ?? row.use_camera_transition_in_relay ?? row.relay_modifiers ?? false;
    const legacyOn = String(legacyModifiers).toLowerCase() === "true" || legacyModifiers === true;
    const legacyRelayPrompt = firstNonEmpty(
        row.relay_prompt,
        row.local_prompt,
        row.prompt_beat,
        row.beat_prompt,
        row.video_prompt,
        row.action_prompt,
        row.prompt,
        row.localPrompt,
        row.relayPrompt,
        row.promptLocal,
    );
    return {
        _ui_id: String(row._ui_id || row.ui_id || `row_${Date.now()}_${Math.random().toString(16).slice(2)}`),
        second: Number.isFinite(Number(row.second ?? row.time ?? row.seconds)) ? Number(row.second ?? row.time ?? row.seconds) : index * 3,
        ref: Number.isFinite(Number(row.ref ?? row.image_ref ?? row.reference_index)) ? Number(row.ref ?? row.image_ref ?? row.reference_index) : index + 1,
        force: Number.isFinite(force) ? Math.max(0, Math.min(1, force)) : 0.22,
        use_guide: row.use_guide ?? row.guide ?? true,
        use_prompt: row.use_prompt ?? row.use_relay ?? row.relay ?? row.prompt_relay ?? true,
        label: String(row.label || row.shot_label || `shot_${index + 1}`),
        camera: normalizeCameraOption(row.camera || row.camera_move || "continuous dolly-in"),
        transition: String(row.transition || row.transition_intent || "continuous_motion"),
        note: String(row.note || row.camera_note || ""),
        relay_prompt: legacyRelayPrompt,
        use_relay_modifiers: legacyModifiers,
        camera_relay_mode: normalizeOption(row.camera_relay_mode || row.camera_prompt_mode || (legacyOn ? "before" : "off"), CAMERA_RELAY_OPTIONS, "off"),
        transition_relay_mode: normalizeOption(row.transition_relay_mode || row.transition_prompt_mode || (legacyOn ? "safe_only" : "off"), TRANSITION_RELAY_OPTIONS, "off"),
        relay_addon_position: normalizeOption(row.relay_addon_position || row.addon_position || "after", RELAY_ADDON_POSITION_OPTIONS, "after"),
        relay_modifier_text: String(row.relay_modifier_text || row.modifier_text || row.relay_addon || ""),
    };
}

function writeShotboard(node, rows) {
    setWidgetValue(node, "timeline_data", JSON.stringify({ rows: rows.map(normalizeShotboardRow) }, null, 2));
}

function downloadJsonFile(data, filename) {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1200);
}

function readJsonFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            try {
                resolve(JSON.parse(String(reader.result || "{}").replace(/^\uFEFF/, "").trim()));
            } catch (err) {
                reject(err);
            }
        };
        reader.onerror = () => reject(reader.error || new Error("Could not read board file"));
        reader.readAsText(file);
    });
}

function safeBoardFilename(label) {
    const clean = String(label || "cine_shotboard")
        .trim()
        .replace(/[^\w.-]+/g, "_")
        .replace(/^_+|_+$/g, "")
        .slice(0, 80);
    return `${clean || "cine_shotboard"}_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.json`;
}


function isAbsoluteLocalPath(path) {
    return /^[a-zA-Z]:[\\/]/.test(path) || String(path || "").startsWith("\\\\") || String(path || "").startsWith("/");
}

function previewUrlForPath(path) {
    const clean = String(path || "").trim();
    if (!clean) return "";
    if (isAbsoluteLocalPath(clean)) return `/api/iamccs/cine/view_image?path=${encodeURIComponent(clean)}`;
    const normalized = clean.replace(/\\/g, "/").replace(/^\/+/, "");
    const parts = normalized.split("/").filter(Boolean);
    const filename = parts.pop() || normalized;
    const subfolder = parts.join("/");
    const query = new URLSearchParams({ filename, type: "input" });
    if (subfolder) query.set("subfolder", subfolder);
    return `/api/view?${query.toString()}`;
}

function splitReferencePaths(value) {
    return String(value || "").split(/\n|,/).map((item) => item.trim()).filter(Boolean);
}

function ellipsize(text, maxLength = 14) {
    const value = String(text || "");
    if (value.length <= maxLength) return value;
    return `${value.slice(0, Math.max(1, maxLength - 3))}...`;
}

function moveItem(items, from, to) {
    const next = items.slice();
    if (from < 0 || from >= next.length || to < 0 || to >= next.length || from === to) return next;
    const [moved] = next.splice(from, 1);
    next.splice(to, 0, moved);
    return next;
}

function moveRowsKeepingTimelineSlots(rows, from, to, normalizeRow) {
    const normalized = rows.map((row, index) => normalizeRow(row, index));
    if (from < 0 || from >= normalized.length || to < 0 || to >= normalized.length || from === to) return normalized;
    const timeSlots = normalized.map((row) => Number(row.second || 0)).sort((a, b) => a - b);
    return moveItem(normalized, from, to).map((row, index) => normalizeRow({
        ...row,
        second: Number.isFinite(timeSlots[index]) ? timeSlots[index] : row.second,
    }, index));
}

function getOwnReferencePaths(node) {
    return splitReferencePaths(getWidget(node, "image_paths")?.value);
}

function setOwnReferencePaths(node, paths) {
    return setWidgetValue(node, "image_paths", paths.join("\n"));
}

function replaceReferencePathAt(node, index, newPath) {
    const cleanPath = String(newPath || "").trim();
    if (!cleanPath) return getConnectedReferencePaths(node);
    const current = getConnectedReferencePaths(node).slice();
    const next = current.length ? current : getOwnReferencePaths(node).slice();
    const safeIndex = Number.isFinite(Number(index)) ? Math.max(0, Number(index)) : 0;
    if (!next.length) {
        next.push(cleanPath);
    } else if (safeIndex < next.length) {
        next[safeIndex] = cleanPath;
    } else {
        next.push(cleanPath);
    }
    const compact = next.map((item) => String(item || "").trim()).filter(Boolean);
    setOwnReferencePaths(node, compact);
    try { node?.setDirtyCanvas?.(true, true); app?.graph?.setDirtyCanvas?.(true, true); } catch {}
    return compact;
}

function isShotboardV2Node(node) {
    return nodeClassName(node) === "IAMCCS_CineShotboardPlannerProV2";
}

function isShotboardLiteNode(node) {
    return nodeClassName(node) === "IAMCCS_CineShotboardLite";
}

function getConnectedReferencePaths(node) {
    const own = getOwnReferencePaths(node);
    if (own.length) return own;
    const input = node?.inputs?.find((item) => item?.name === "multi_input");
    if (!input?.link || !app?.graph?.links) return [];
    const link = app.graph.links[input.link];
    const originId = link?.origin_id ?? link?.source_id;
    if (originId == null) return [];
    const source = app.graph.getNodeById(originId);
    if (!source) return [];
    const pathsWidget = getWidget(source, "image_paths");
    return splitReferencePaths(pathsWidget?.value);
}

function getBoardReferencePaths(node) {
    const input = node?.inputs?.find((item) => item?.name === "multi_input");
    if (!input?.link || !app?.graph?.links) return [];
    const link = app.graph.links[input.link];
    const originId = link?.origin_id ?? link?.source_id;
    if (originId == null) return [];
    const source = app.graph.getNodeById(originId);
    if (!source) return [];
    return splitReferencePaths(getWidget(source, "image_paths")?.value);
}

async function uploadShotboardImages(files) {
    const uploaded = [];
    for (const file of files || []) {
        const body = new FormData();
        body.append("image", file);
        try {
            const resp = await api.fetchApi("/upload/image", { method: "POST", body });
            if (resp.status === 200) {
                const data = await resp.json();
                let name = data.name;
                if (data.subfolder) name = `${data.subfolder}/${name}`;
                uploaded.push(name);
            }
        } catch (err) {
            console.error("[IAMCCS Cine Shotboard] image upload failed", err);
        }
    }
    return uploaded;
}

async function transformShotboardReference(payload) {
    const resp = await api.fetchApi("/api/iamccs/cine/transform_reference", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok || !data?.ok) {
        throw new Error(data?.error || `Reference transform failed (${resp.status})`);
    }
    return data;
}

function openReferenceFrameEditor(node, index, path, onApply) {
    const existing = document.querySelector(".iamccs-cine-ref-editor-overlay");
    existing?.remove();

    const targetWidth = Number(getWidget(node, "image_width")?.value || 768);
    const targetHeight = Number(getWidget(node, "image_height")?.value || 432);
    const state = { fit_mode: "cover", zoom: 1, pan_x: 0, pan_y: 0, rotation: 0, resample: "lanczos" };
    let previewCropBox = null;

    const overlay = document.createElement("div");
    overlay.className = "iamccs-cine-ref-editor-overlay";
    overlay.style.cssText = [
        "position:fixed",
        "inset:0",
        "z-index:100000",
        "background:rgba(5,7,9,.82)",
        "display:flex",
        "align-items:center",
        "justify-content:center",
        "padding:28px",
        "box-sizing:border-box",
        "pointer-events:auto",
    ].join(";");

    const panel = document.createElement("div");
    panel.style.cssText = [
        "width:min(1180px,96vw)",
        "height:min(780px,92vh)",
        `background:${CINE_FILM_LAB.nodeBg}`,
        `border:1px solid ${CINE_NODE_CHROME.shotboard.border}`,
        "border-radius:8px",
        `box-shadow:0 18px 80px rgba(0,0,0,.55), inset 0 1px 0 ${CINE_NODE_CHROME.shotboard.glow}`,
        "display:grid",
        "grid-template-columns:minmax(520px,1fr) 320px",
        "gap:14px",
        "padding:14px",
        "box-sizing:border-box",
        `color:${CINE_FILM_LAB.text}`,
        "font:12px Arial,sans-serif",
    ].join(";");

    const previewWrap = document.createElement("div");
    previewWrap.style.cssText = [
        "position:relative",
        "overflow:hidden",
        `background:${CINE_FILM_LAB.field}`,
        `border:1px solid ${CINE_FILM_LAB.border}`,
        "border-radius:7px",
        "display:flex",
        "align-items:center",
        "justify-content:center",
    ].join(";");

    const frame = document.createElement("div");
    frame.style.cssText = [
        "position:relative",
        "width:92%",
        `aspect-ratio:${Math.max(1, targetWidth)} / ${Math.max(1, targetHeight)}`,
        "overflow:hidden",
        "border:1px solid rgba(255,255,255,.30)",
        "box-shadow:0 0 0 999px rgba(0,0,0,.32)",
        "background:#050505",
    ].join(";");

    const img = document.createElement("img");
    img.src = previewUrlForPath(path);
    img.draggable = false;
    img.style.cssText = [
        "position:absolute",
        "max-width:none",
        "max-height:none",
        "display:block",
        "transform-origin:center center",
        "will-change:transform",
    ].join(";");
    frame.appendChild(img);

    const label = document.createElement("div");
    label.textContent = `${targetWidth} x ${targetHeight} reference frame`;
    label.style.cssText = "position:absolute;left:10px;bottom:8px;background:rgba(0,0,0,.72);color:#fff;border-radius:4px;padding:4px 7px;font-size:11px;";
    previewWrap.append(frame, label);

    const side = document.createElement("div");
    side.style.cssText = "display:flex;flex-direction:column;gap:10px;min-width:0;";
    const title = document.createElement("div");
    title.textContent = `Reference ${index + 1} Frame Editor`;
    title.style.cssText = "font-size:16px;font-weight:700;color:#fff;margin-bottom:2px;";
    const subtitle = document.createElement("div");
    subtitle.textContent = "Create a new project-sized reference. Use Fill for crops or Fit to keep the full source image.";
    subtitle.style.cssText = `font-size:11px;color:${CINE_FILM_LAB.muted};line-height:1.35;margin-bottom:8px;`;

    const makeControl = (labelText, key, min, max, step) => {
        const wrap = document.createElement("label");
        wrap.style.cssText = "display:grid;grid-template-columns:74px 1fr 54px;gap:7px;align-items:center;";
        const text = document.createElement("span");
        text.textContent = labelText;
        text.style.cssText = `color:${CINE_FILM_LAB.muted};`;
        const range = document.createElement("input");
        range.type = "range";
        range.min = String(min);
        range.max = String(max);
        range.step = String(step);
        range.value = String(state[key]);
        const num = document.createElement("input");
        num.type = "number";
        num.min = String(min);
        num.max = String(max);
        num.step = String(step);
        num.value = String(state[key]);
        num.style.cssText = inputBase();
        num.style.height = "26px";
        const setValue = (value) => {
            const numeric = Math.max(Number(min), Math.min(Number(max), Number(value) || 0));
            state[key] = numeric;
            if (state.fit_mode === "contain" && (
                (key === "zoom" && numeric > 1.0001) ||
                ((key === "pan_x" || key === "pan_y") && Math.abs(numeric) > 0.0001)
            )) {
                state.fit_mode = "cover";
                try { updateFitButtons?.(); } catch {}
                try { status.textContent = "Zoom/pan switched to Fill Frame so the saved crop matches this preview."; } catch {}
            }
            range.value = String(state[key]);
            num.value = String(state[key]);
            updatePreview();
        };
        range.oninput = () => setValue(range.value);
        num.oninput = () => setValue(num.value);
        wrap.append(text, range, num);
        return wrap;
    };

    function updatePreview() {
        const frameW = Math.max(1, frame.clientWidth || targetWidth);
        const frameH = Math.max(1, frame.clientHeight || targetHeight);
        const naturalW = Math.max(1, img.naturalWidth || targetWidth);
        const naturalH = Math.max(1, img.naturalHeight || targetHeight);
        const fitScale = state.fit_mode === "contain"
            ? Math.min(frameW / naturalW, frameH / naturalH)
            : Math.max(frameW / naturalW, frameH / naturalH);
        const scale = fitScale * Math.max(1, Number(state.zoom) || 1);
        const displayW = naturalW * scale;
        const displayH = naturalH * scale;
        const maxShiftX = Math.max(0, (displayW - frameW) / 2);
        const maxShiftY = Math.max(0, (displayH - frameH) / 2);
        const left = (frameW - displayW) / 2 + state.pan_x * maxShiftX;
        const top = (frameH - displayH) / 2 + state.pan_y * maxShiftY;
        previewCropBox = null;
        if (state.fit_mode !== "contain" && Math.abs(Number(state.rotation) || 0) < 0.001 && scale > 0) {
            const srcLeft = Math.max(0, Math.min(naturalW - 1, -left / scale));
            const srcTop = Math.max(0, Math.min(naturalH - 1, -top / scale));
            const srcRight = Math.max(srcLeft + 1, Math.min(naturalW, (frameW - left) / scale));
            const srcBottom = Math.max(srcTop + 1, Math.min(naturalH, (frameH - top) / scale));
            previewCropBox = [srcLeft, srcTop, srcRight, srcBottom];
        }
        img.style.width = `${displayW}px`;
        img.style.height = `${displayH}px`;
        img.style.left = `${left}px`;
        img.style.top = `${top}px`;
        img.style.transform = `rotate(${state.rotation}deg)`;
    }
    img.onload = updatePreview;

    const resampleRow = document.createElement("label");
    resampleRow.style.cssText = "display:grid;grid-template-columns:74px 1fr;gap:7px;align-items:center;";
    const resampleLabel = document.createElement("span");
    resampleLabel.textContent = "Resize";
    resampleLabel.style.cssText = `color:${CINE_FILM_LAB.muted};`;
    const resample = makeSelect("lanczos", ["lanczos", "bicubic", "bilinear", "nearest"], (value) => {
        state.resample = value;
    });
    resampleRow.append(resampleLabel, resample);

    const status = document.createElement("div");
    status.style.cssText = `min-height:34px;color:${CINE_FILM_LAB.muted};font-size:11px;line-height:1.35;`;
    status.textContent = "Fill crops to the project ratio. Fit keeps the full imported image and pads to the project size.";

    const actions = document.createElement("div");
    actions.style.cssText = "display:flex;gap:8px;justify-content:flex-end;margin-top:auto;";
    const fitActions = document.createElement("div");
    fitActions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const fillFrame = button("Fill Frame", "primary");
    const fitProject = button("Fit to Project", "primary");
    const updateFitButtons = () => {
        fillFrame.style.opacity = state.fit_mode === "cover" ? "1" : ".62";
        fitProject.style.opacity = state.fit_mode === "contain" ? "1" : ".62";
    };
    fillFrame.title = "Fill the project frame. This may crop the source image if the aspect ratio is different.";
    fitProject.title = "Fit the entire source image inside the selected project resolution without cropping.";
    fillFrame.onclick = () => {
        state.fit_mode = "cover";
        updateFitButtons();
        updatePreview();
        status.textContent = "Fill Frame: project frame is fully covered; edges may be cropped.";
    };
    fitProject.onclick = () => {
        state.fit_mode = "contain";
        state.zoom = 1;
        state.pan_x = 0;
        state.pan_y = 0;
        updateFitButtons();
        updatePreview();
        status.textContent = "Fit to Project: full image preserved, padded to the project resolution.";
    };
    fitActions.append(fillFrame, fitProject);
    const reset = button("Reset");
    const cancel = button("Cancel");
    const apply = button("Apply as New Reference", "primary");
    reset.onclick = () => {
        state.fit_mode = "cover"; state.zoom = 1; state.pan_x = 0; state.pan_y = 0; state.rotation = 0; state.resample = "lanczos";
        overlay.remove();
        openReferenceFrameEditor(node, index, path, onApply);
    };
    cancel.onclick = () => overlay.remove();
    apply.onclick = async () => {
        apply.disabled = true;
        status.textContent = "Saving new reference in IAMCCS_newimages...";
        try {
            const data = await transformShotboardReference({
                path,
                width: targetWidth,
                height: targetHeight,
                fit_mode: state.fit_mode,
                zoom: state.zoom,
                pan_x: state.pan_x,
                pan_y: state.pan_y,
                rotation: state.rotation,
                resample: state.resample,
                crop_box: previewCropBox,
                crop_box_source: previewCropBox ? "ui_preview" : "backend_formula",
            });
            const appliedPath = data?.path || data?.relative_path || data?.absolute_path || data?.filename || data?.name;
            try {
                onApply?.(appliedPath, data);
            } catch (applyErr) {
                console.error("[IAMCCS Cine Shotboard V2] reference applied but UI refresh failed", applyErr);
            }
            overlay.remove();
        } catch (err) {
            console.error("[IAMCCS Cine Shotboard V2] reference transform failed", err);
            status.textContent = err?.message || String(err);
            apply.disabled = false;
        }
    };
    actions.append(reset, cancel, apply);

    side.append(
        title,
        subtitle,
        fitActions,
        makeControl("Zoom", "zoom", 1, 8, 0.05),
        makeControl("Pan X", "pan_x", -1, 1, 0.01),
        makeControl("Pan Y", "pan_y", -1, 1, 0.01),
        makeControl("Rotate", "rotation", -20, 20, 0.1),
        resampleRow,
        status,
        actions
    );
    panel.append(previewWrap, side);
    overlay.appendChild(panel);
    overlay.addEventListener("click", (event) => {
        if (event.target === overlay) overlay.remove();
    });
    document.addEventListener("keydown", function closeOnEsc(event) {
        if (event.key !== "Escape") return;
        overlay.remove();
        document.removeEventListener("keydown", closeOnEsc);
    });
    document.body.appendChild(overlay);
    updateFitButtons();
    updatePreview();
}

function getEditorZoom(root) {
    const zoom = Number(root?._iamccsEditorZoom);
    return Number.isFinite(zoom) ? Math.max(0.75, Math.min(1.6, zoom)) : 1;
}

function applyEditorZoom(root) {
    if (!root) return;
    const zoom = getEditorZoom(root);
    if (root._iamccsFullscreenState) {
        root.style.zoom = String(zoom);
        root.style.transformOrigin = "top left";
    } else {
        root.style.zoom = "";
        root.style.transformOrigin = "";
    }
    root.dispatchEvent(new CustomEvent("iamccs:cine-editor-zoom", { detail: { zoom } }));
}

function setEditorZoom(root, node, zoom) {
    if (!root) return;
    root._iamccsEditorZoom = Math.max(0.75, Math.min(1.6, Math.round((Number(zoom) || 1) * 10) / 10));
    applyEditorZoom(root);
    try { node?.setDirtyCanvas?.(true, true); app.graph?.setDirtyCanvas?.(true, true); } catch {}
}

function restoreFullscreenEditor(root, node) {
    const state = root._iamccsFullscreenState;
    if (!state) return;
    root.style.cssText = state.rootCss;
    if (state.placeholder?.parentNode) {
        state.placeholder.parentNode.insertBefore(root, state.placeholder);
        state.placeholder.remove();
    }
    state.overlay?.remove();
    document.removeEventListener("keydown", state.keyHandler);
    root._iamccsFullscreenState = null;
    applyEditorZoom(root);
    root.dispatchEvent(new CustomEvent("iamccs:cine-fullscreen", { detail: { open: false } }));
    try { node.setDirtyCanvas?.(true, true); app.graph?.setDirtyCanvas?.(true, true); } catch {}
}

function toggleFullscreenEditor(root, node) {
    if (root._iamccsFullscreenState) {
        restoreFullscreenEditor(root, node);
        return;
    }

    const parent = root.parentNode;
    if (!parent) return;
    const placeholder = document.createComment("iamccs-cine-shotboard-home");
    parent.insertBefore(placeholder, root);

    const overlay = document.createElement("div");
    overlay.style.cssText = `
        position: fixed;
        inset: 0;
        z-index: 999999;
        background: rgba(4, 8, 11, .82);
        display: flex;
        flex-direction: column;
        padding: 18px;
        box-sizing: border-box;
        pointer-events: auto;
    `;

    const bar = document.createElement("div");
    bar.style.cssText = `
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        padding: 0 0 10px 0;
        color: #e8eef2;
        font: 12px Arial, sans-serif;
    `;
    const title = document.createElement("div");
    title.textContent = "Cine Shotboard Timeline Pro";
    title.style.cssText = "font-weight:700;letter-spacing:0;color:#ffffff;";
    const close = button("Close Editor", "primary");
    bar.append(title, close);

    const panel = document.createElement("div");
    panel.style.cssText = `
        flex: 1;
        min-height: 0;
        overflow-y: auto;
        overflow-x: hidden;
        border: 1px solid #557062;
        border-radius: 8px;
        background: #12191f;
        box-shadow: 0 22px 80px rgba(0,0,0,.65);
        padding: 10px;
        box-sizing: border-box;
    `;

    const keyHandler = (event) => {
        if (event.key === "Escape") restoreFullscreenEditor(root, node);
    };
    document.addEventListener("keydown", keyHandler);
    close.onclick = () => restoreFullscreenEditor(root, node);

    root._iamccsFullscreenState = { overlay, placeholder, rootCss: root.style.cssText, keyHandler, panel };
    root.style.cssText += `
        max-height: none !important;
        height: auto !important;
        min-height: calc(100vh - 92px) !important;
        overflow: visible !important;
        border: 0 !important;
        box-shadow: none !important;
    `;
    panel.appendChild(root);
    overlay.append(bar, panel);
    document.body.appendChild(overlay);
    applyEditorZoom(root);
    root.dispatchEvent(new CustomEvent("iamccs:cine-fullscreen", { detail: { open: true } }));
    try { node.setDirtyCanvas?.(true, true); app.graph?.setDirtyCanvas?.(true, true); } catch {}
}

function cineReferenceActionTone(label) {
    const tones = {
        E: { bg: "#0F5F78", border: "#79D7F0", color: "#F2FCFF" },
        R: { bg: "#7A5420", border: "#F0C36A", color: "#FFF4D8" },
        D: { bg: "#513887", border: "#C9B4FF", color: "#F6F1FF" },
    };
    return tones[label] || { bg: CINE_NODE_CHROME.shotboard.header, border: CINE_FILM_LAB.borderSoft, color: "#fff" };
}

function refPicker(value, referencePaths, onChange, options = {}) {
    const thumbWidth = Math.max(150, Number(options.thumbWidth) || 168);
    const thumbHeight = Math.max(116, Number(options.thumbHeight) || 116);
    const hasSideActions = Boolean(options.onEdit || options.onReplace || options.onDuplicate);
    const sideActionCount = [options.onEdit, options.onReplace, options.onDuplicate].filter(Boolean).length;
    const gutterWidth = hasSideActions ? 28 : 0;
    const imageWidth = hasSideActions ? Math.max(72, thumbWidth - gutterWidth) : Math.max(120, thumbWidth);
    const wrap = document.createElement("div");
    wrap.style.cssText = `display:grid;grid-template-rows:${thumbHeight}px 26px;gap:5px;align-items:center;`;

    const frameWrap = document.createElement("div");
    frameWrap.style.cssText = hasSideActions
        ? `display:grid;grid-template-columns:${gutterWidth}px 1fr;width:${thumbWidth}px;height:${thumbHeight}px;min-width:0;`
        : `display:block;width:${thumbWidth}px;height:${thumbHeight}px;`;

    const thumb = document.createElement("div");
    thumb.style.cssText = `
        position: relative;
        width: ${imageWidth}px;
        height: ${thumbHeight}px;
        border: 1px solid #40505a;
        background: #070b0e;
        border-radius: 4px;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #8fa3af;
        font-size: 11px;
    `;

    const index = Math.max(1, Math.min(50, Number(value) || 1));
    const path = referencePaths[index - 1];
    if (path) {
        const img = document.createElement("img");
        img.src = previewUrlForPath(path);
        img.title = path;
        img.style.cssText = "width:100%;height:100%;object-fit:cover;display:block;";
        img.onerror = () => { thumb.textContent = `ref ${index}`; };
        thumb.appendChild(img);
    } else {
        thumb.textContent = `ref ${index}`;
    }

    if (hasSideActions) {
        const actionRail = document.createElement("div");
        actionRail.style.cssText = [
            `width:${gutterWidth}px`,
            `height:${thumbHeight}px`,
            "display:grid",
            "grid-template-columns:1fr",
            `grid-template-rows:repeat(${Math.max(1, sideActionCount)},1fr)`,
            `border:1px solid ${CINE_FILM_LAB.borderSoft}`,
            "border-right:0",
            "border-radius:4px 0 0 4px",
            "overflow:hidden",
            "box-sizing:border-box",
        ].join(";");
        const sideAction = (label, title, action) => {
            const tone = cineReferenceActionTone(label);
            const btn = document.createElement("button");
            btn.textContent = label;
            btn.title = title;
            btn.style.cssText = [
                "width:100%",
                "height:100%",
                "padding:0",
                "border:0",
                `border-bottom:1px solid ${tone.border}`,
                `background:${tone.bg}`,
                `color:${tone.color}`,
                "display:flex",
                "align-items:center",
                "justify-content:center",
                "font-size:11px",
                "font-weight:900",
                "line-height:1",
                "letter-spacing:0",
                "writing-mode:horizontal-tb",
                "transform:none",
                "cursor:pointer",
            ].join(";");
            btn.onclick = (event) => {
                event.preventDefault();
                event.stopPropagation();
                action();
            };
            return protectControlDrag(btn);
        };
        if (options.onEdit) {
            actionRail.appendChild(sideAction("E", "Edit this row reference: crop, pan, zoom and save a new reference", () => {
                if (!path) return;
                options.onEdit(index - 1, path);
            }));
        }
        if (options.onReplace) {
            actionRail.appendChild(sideAction("R", "Replace this reference image without changing the row timing", () => {
                options.onReplace(index - 1, path);
            }));
        }
        if (options.onDuplicate) {
            actionRail.appendChild(sideAction("D", "Duplicate this reference into the next slot and create a new keyframe", () => {
                options.onDuplicate(index - 1, path);
            }));
        }
        frameWrap.appendChild(actionRail);
    }

    const badge = document.createElement("div");
    badge.textContent = String(index);
    badge.style.cssText = "position:absolute;left:4px;bottom:3px;background:rgba(0,0,0,.72);color:#fff;font-size:11px;padding:1px 5px;border-radius:3px;";
    thumb.appendChild(badge);
    frameWrap.appendChild(thumb);

    const select = document.createElement("select");
    select.style.cssText = inputBase() + "height:26px;padding:2px 5px;";
    const maxOptions = Math.max(50, referencePaths.length || 0);
    for (let i = 1; i <= maxOptions; i++) {
        const opt = document.createElement("option");
        opt.value = String(i);
        const filename = referencePaths[i - 1] ? referencePaths[i - 1].split(/[\\/]/).pop() : "";
        opt.textContent = `ref ${i}`;
        opt.title = filename || `ref ${i}`;
        select.appendChild(opt);
    }
    select.value = String(index);
    select.title = path ? `ref ${index} - ${path.split(/[\\/]/).pop()}` : `ref ${index}`;
    select.onchange = () => onChange(Number(select.value));
    protectControlDrag(select);

    wrap.append(frameWrap, select);
    return protectControlDrag(wrap);
}

function refSelect(value, onChange) {
    const select = document.createElement("select");
    select.style.cssText = inputBase();
    for (let i = 1; i <= 50; i++) {
        const opt = document.createElement("option");
        opt.value = String(i);
        opt.textContent = `ref ${i}`;
        select.appendChild(opt);
    }
    select.value = String(Math.max(1, Math.min(50, Number(value) || 1)));
    select.onchange = () => onChange(Number(select.value));
    return protectControlDrag(select);
}

function checkbox(value, onChange) {
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = value === true || String(value).toLowerCase() === "true";
    box.style.cssText = "width:16px;height:16px;margin:6px auto 0;display:block;accent-color:#59a9cf;";
    box.onchange = () => onChange(box.checked);
    return protectControlDrag(box);
}

function formatTimeValue(value) {
    const n = Math.max(0, Number(value) || 0);
    return Number.isInteger(n) ? String(n) : n.toFixed(1).replace(/\.0$/, "");
}

function stepPrecision(step) {
    const text = String(step ?? "1");
    const dot = text.indexOf(".");
    return dot < 0 ? 0 : Math.min(4, text.length - dot - 1);
}

function formatStepperValue(value, precision) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "";
    if (precision <= 0) return String(Math.round(n));
    return n.toFixed(precision).replace(/\.?0+$/, "");
}

function numberStepperControl(value, step, min, max, onChange) {
    const stepValue = Math.max(0.0001, Number(step) || 1);
    const minValue = Number.isFinite(Number(min)) ? Number(min) : -Infinity;
    const hasMax = max !== null && max !== undefined && String(max).trim() !== "";
    const maxValue = hasMax && Number.isFinite(Number(max)) ? Number(max) : Infinity;
    const precision = stepPrecision(step);
    const wrap = document.createElement("div");
    wrap.style.cssText = "display:grid;grid-template-columns:22px minmax(48px,1fr) 22px;gap:3px;align-items:center;min-width:96px;";

    const input = document.createElement("input");
    input.type = "text";
    input.inputMode = precision > 0 ? "decimal" : "numeric";
    input.value = formatStepperValue(value, precision);
    input.style.cssText = inputBase() + "height:28px;padding:4px 5px;text-align:center;font-variant-numeric:tabular-nums;";

    const clamp = (raw) => Math.min(maxValue, Math.max(minValue, Number(raw)));
    const apply = (raw) => {
        const n = clamp(raw);
        if (!Number.isFinite(n)) return;
        input.value = formatStepperValue(n, precision);
        onChange(n);
    };
    const makeStep = (label, delta) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = label;
        btn.style.cssText = "height:28px;padding:0;border:1px solid #405664;border-radius:4px;background:#0b1116;color:#e8eef2;font-size:13px;line-height:1;cursor:pointer;";
        btn.onclick = (event) => {
            event.preventDefault();
            const current = Number(String(input.value).replace(",", ".")) || 0;
            apply(Math.round((current + delta) * 10000) / 10000);
        };
        return protectControlDrag(btn);
    };

    input.oninput = () => {
        const raw = String(input.value).replace(",", ".").trim();
        if (!raw) return;
        const n = Number(raw);
        if (Number.isFinite(n)) onChange(clamp(n));
    };
    input.onblur = () => {
        const raw = String(input.value).replace(",", ".").trim();
        if (!raw) {
            input.value = formatStepperValue(value, precision);
            return;
        }
        apply(Number(raw));
    };
    input.onkeydown = (event) => {
        if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
        event.preventDefault();
        const current = Number(String(input.value).replace(",", ".")) || 0;
        apply(current + (event.key === "ArrowUp" ? stepValue : -stepValue));
    };
    input.onfocus = () => input.select();
    protectControlDrag(input);

    wrap.append(makeStep("-", -stepValue), input, makeStep("+", stepValue));
    wrap._iamccsSetValue = (next) => {
        input.value = formatStepperValue(next, precision);
    };
    return protectControlDrag(wrap);
}

function timeControl(value, onChange) {
    const wrap = document.createElement("div");
    wrap.style.cssText = "display:grid;grid-template-columns:22px minmax(42px,1fr) 22px;gap:3px;align-items:center;min-width:86px;";

    const makeStep = (label, delta) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = label;
        btn.title = delta < 0 ? "Decrease time by 0.1s" : "Increase time by 0.1s";
        btn.style.cssText = "height:28px;padding:0;border:1px solid #405664;border-radius:4px;background:#0b1116;color:#e8eef2;font-size:13px;line-height:1;cursor:pointer;";
        btn.onclick = (event) => {
            event.preventDefault();
            const current = Math.max(0, Number(String(input.value).replace(",", ".")) || 0);
            const next = Math.max(0, Math.round((current + delta) * 10) / 10);
            input.value = formatTimeValue(next);
            onChange(next);
        };
        return protectControlDrag(btn);
    };

    const input = document.createElement("input");
    input.type = "text";
    input.inputMode = "decimal";
    input.value = formatTimeValue(value);
    input.style.cssText = inputBase() + "height:28px;padding:4px 5px;text-align:center;font-variant-numeric:tabular-nums;";
    input.oninput = () => {
        const raw = String(input.value).replace(",", ".").trim();
        if (!raw) return;
        const n = Number(raw);
        if (Number.isFinite(n)) onChange(Math.max(0, n));
    };
    input.onblur = () => {
        input.value = formatTimeValue(input.value);
    };
    input.onkeydown = (event) => {
        if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
        event.preventDefault();
        const delta = event.key === "ArrowUp" ? 0.1 : -0.1;
        const current = Math.max(0, Number(String(input.value).replace(",", ".")) || 0);
        const next = Math.max(0, Math.round((current + delta) * 10) / 10);
        input.value = formatTimeValue(next);
        onChange(next);
    };
    input.onfocus = () => input.select();
    protectControlDrag(input);

    wrap.append(makeStep("-", -0.1), input, makeStep("+", 0.1));
    return protectControlDrag(wrap);
}

function forceControl(value, onChange) {
    const wrap = document.createElement("div");
    wrap.style.cssText = "display:grid;grid-template-columns:minmax(64px,1fr) 50px;gap:7px;align-items:center;min-width:124px;";
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "0";
    slider.max = "1";
    slider.step = "0.01";
    slider.value = Number(value).toFixed(2);
    slider.style.cssText = "width:100%;accent-color:#5aa9cf;";
    const out = document.createElement("input");
    out.type = "text";
    out.inputMode = "decimal";
    out.value = Number(value).toFixed(2);
    out.style.cssText = inputBase() + "padding:4px 6px;text-align:center;font-variant-numeric:tabular-nums;";
    const sync = (v) => {
        const n = Math.max(0, Math.min(1, Number(v) || 0));
        slider.value = n.toFixed(2);
        out.value = n.toFixed(2);
        onChange(n);
    };
    slider.oninput = () => sync(slider.value);
    out.oninput = () => sync(out.value);
    out.onfocus = () => out.select();
    protectControlDrag(slider);
    protectControlDrag(out);
    wrap.append(slider, out);
    return protectControlDrag(wrap);
}

function editorZoomControl(root, node) {
    const wrap = document.createElement("div");
    wrap.style.cssText = "display:grid;grid-template-columns:58px 1fr 58px;gap:5px;align-items:center;min-width:160px;";
    const minus = button("Zoom -");
    const plus = button("Zoom +");
    const value = document.createElement("div");
    value.style.cssText = `
        height: 28px;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px solid #405664;
        border-radius: 4px;
        background: #0b1116;
        color: #dbe7ee;
        font-size: 11px;
        font-variant-numeric: tabular-nums;
    `;

    const updateLabel = () => {
        value.textContent = `${Math.round(getEditorZoom(root) * 100)}%`;
    };
    minus.onclick = (event) => {
        event.preventDefault();
        setEditorZoom(root, node, getEditorZoom(root) - 0.1);
        updateLabel();
    };
    plus.onclick = (event) => {
        event.preventDefault();
        setEditorZoom(root, node, getEditorZoom(root) + 0.1);
        updateLabel();
    };
    root?.addEventListener?.("iamccs:cine-editor-zoom", updateLabel);
    updateLabel();
    wrap.append(protectControlDrag(minus), value, protectControlDrag(plus));
    return protectControlDrag(wrap);
}

function shotboardWarnings(rows) {
    const warnings = [];
    const clean = rows.map(normalizeShotboardRow).sort((a, b) => a.second - b.second);
    const activeGuides = clean.filter((row) => row.use_guide && Number(row.force || 0) > 0);
    const relayIsOn = (row) => row.use_prompt !== false && String(row.use_prompt).toLowerCase() !== "false";
    const relayHasPrompt = (row) => relayIsOn(row) && String(row.relay_prompt || "").trim().length > 0;
    const activeRelay = clean.filter(relayHasPrompt);
    for (let i = 0; i < clean.length; i++) {
        const row = clean[i];
        if (row.use_guide && Number(row.force || 0) === 0) warnings.push(`${row.label}: Guide is ON but force is 0, so this row will not anchor the image.`);
        if (row.use_guide && row.force > 0.55) warnings.push(`${row.label}: high FLF force can pin the frame or create a still-slide. Use strong anchors only for start/end or deliberate locks.`);
        if (relayIsOn(row) && !String(row.relay_prompt || "").trim()) warnings.push(`${row.label}: Relay is ON but Local prompt is empty, so this row will be skipped by PromptRelay.`);
        if (relayHasPrompt(row) && String(row.relay_prompt || "").length > 260) warnings.push(`${row.label}: local PromptRelay beat is long. Keep beat prompts concise and focused on motion/action.`);
        if (row.transition === "hard_cut") warnings.push(`${row.label}: hard cuts inside one generation usually morph. Use multigen + concat for a true editorial cut.`);
        if (row.transition === "hard_cut" && row.transition_relay_mode === "append") warnings.push(`${row.label}: avoid appending hard_cut wording into PromptRelay; it can cause semantic jumps or morphs.`);
        if (i > 0) {
            const gap = row.second - clean[i - 1].second;
            if (row.use_guide && clean[i - 1].use_guide && gap < 1.5) warnings.push(`${clean[i - 1].label} -> ${row.label}: FLF guides are very close; expect flashes or visible joints. Space anchors out or make one row prompt-only.`);
            if (relayHasPrompt(row) && relayHasPrompt(clean[i - 1]) && gap < 0.75) warnings.push(`${clean[i - 1].label} -> ${row.label}: PromptRelay beat is very short; the model may rush the transition.`);
        }
    }
    if (activeGuides.length > 5) {
        warnings.push("Many FLF guides are active. For continuous camera moves, use a few soft image anchors and let the prompt carry the in-between motion.");
    }
    if (clean.length && !activeRelay.length) {
        warnings.push("Relay is fully OFF. Use an FLF-only/basic text path, or enable at least one concise local beat before routing through PromptRelay.");
    }
    return warnings;
}

function shotboardStatusTip(rows) {
    const clean = rows.map(normalizeShotboardRow);
    if (!clean.length) return "Add rows, then choose image anchors with Guide and timed text beats with Relay.";
    const guides = clean.filter((row) => row.use_guide && Number(row.force || 0) > 0).length;
    const relay = clean.filter((row) => row.use_prompt !== false && String(row.use_prompt).toLowerCase() !== "false" && String(row.relay_prompt || "").trim()).length;
    if (!guides && relay) return "PromptRelay-only: useful for timed text beats, but images will not act as FLF anchors.";
    if (guides && !relay) return "FLF-only: image guides steer structure; the global prompt should describe motion, not the static image.";
    return `${guides} FLF guides and ${relay} Relay beats active. Keep guide forces soft and local beats short.`;
}

function defaultLiteRows() {
    return [
        { second: 0.0, ref: 1, force: 0.82, use_guide: true, use_prompt: false, label: "opening_anchor", camera: "slow push-in", transition: "continuous_motion", note: "Opening visual anchor. Keep the global prompt focused on one continuous motion path.", relay_prompt: "" },
        { second: 3.5, ref: 2, force: 0.24, use_guide: true, use_prompt: false, label: "middle_waypoint", camera: "continuous dolly-in", transition: "continuous_motion", note: "Soft waypoint. Lower force helps avoid a still-frame or slideshow feel.", relay_prompt: "" },
        { second: 7.0, ref: 3, force: 0.28, use_guide: true, use_prompt: false, label: "final_anchor", camera: "continuous dolly-in", transition: "continuous_motion", note: "Final visual target. Keep the ending guide inside the video duration.", relay_prompt: "" },
    ];
}

function normalizeLiteRow(row, index) {
    const base = normalizeShotboardRow(row || {}, index);
    return {
        ...base,
        use_prompt: false,
        relay_prompt: "",
        use_relay_modifiers: false,
        camera_relay_mode: "off",
        transition_relay_mode: "off",
        relay_addon_position: "after",
        relay_modifier_text: "",
    };
}

function renderShotboardLite(node) {
    if (node._iamccsCineShotboardLiteReady) return;
    node._iamccsCineShotboardLiteReady = true;
    const chrome = CINE_NODE_CHROME.shotboardLite;
    const palette = {
        primaryBg: "#6C4D1F",
        primaryBorder: "#E2B75B",
        neutralBg: "#2B2115",
        neutralBorder: "#7C6135",
        dangerBg: "#683331",
        dangerBorder: "#C16D63",
        text: "#FFF2D8",
    };
    const liteButton = (label, tone = "neutral") => button(label, tone, palette);

    lockNodeMinimumSize(node, SHOTBOARD_LITE_NODE_MIN_SIZE, { lockResize: true, preferredSize: SHOTBOARD_LITE_NODE_DEFAULT_SIZE });
    node.color = chrome.header;
    node.bgcolor = chrome.nodeBg;
    node.boxcolor = chrome.box;

    [
        "timeline_data",
        "global_prompt",
        "duration_seconds",
        "frame_rate",
        "guide_policy",
        "min_guide_gap_seconds",
        "max_guides",
        "default_force",
        "image_paths",
        "image_width",
        "image_height",
    ].forEach((name) => hideWidget(getWidget(node, name)));

    let rows = parseJsonWidget(node, defaultLiteRows).map(normalizeLiteRow);
    if (!rows.length) rows = defaultLiteRows().map(normalizeLiteRow);

    const { root, toolbar, table } = tableShell(
        "Cine Shotboard Lite",
        "FLF-only public board. Image guides + one global prompt. No PromptRelay, no local prompt routing."
    );
    root.style.borderColor = chrome.border;
    root.style.boxShadow = `inset 0 1px 0 ${chrome.glow}, 0 0 0 1px rgba(0,0,0,.35)`;
    root.style.background = "#151008";
    root.style.maxHeight = "680px";
    root.style.overflowY = "auto";
    root.style.overflowX = "hidden";

    const globalPromptWidget = getWidget(node, "global_prompt");
    const promptPanel = document.createElement("div");
    promptPanel.style.cssText = `margin:6px 0 8px 0;padding:8px;border:1px solid ${chrome.border};background:#0F0B07;border-radius:6px;`;
    const promptLabel = document.createElement("div");
    promptLabel.textContent = "Global FLF prompt";
    promptLabel.style.cssText = `font-size:11px;color:#D8BC80;margin-bottom:5px;font-weight:700;`;
    const promptArea = document.createElement("textarea");
    promptArea.value = String(globalPromptWidget?.value || "");
    promptArea.rows = 3;
    promptArea.style.cssText = inputBase() + "resize:vertical;min-height:62px;line-height:1.35;padding:9px 18px 9px 10px;scrollbar-gutter:stable;background:#0A0907;border-color:#70572E;color:#FFF2D8;";
    const syncLitePromptWidget = () => setWidgetValue(node, "global_prompt", promptArea.value);
    promptArea.oninput = syncLitePromptWidget;
    promptArea.onchange = syncLitePromptWidget;
    promptPanel.append(promptLabel, promptArea);
    root.insertBefore(promptPanel, toolbar);

    const settingsPanel = document.createElement("div");
    settingsPanel.style.cssText = `margin:0 0 8px 0;padding:8px;border:1px solid ${chrome.border};background:#0F0B07;border-radius:6px;`;
    const settingsGrid = document.createElement("div");
    settingsGrid.style.cssText = "display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:8px;";
    const addSetting = (labelText, control) => {
        const wrap = document.createElement("label");
        wrap.style.cssText = "display:flex;flex-direction:column;gap:4px;color:#D8BC80;font-size:11px;font-weight:700;";
        const label = document.createElement("span");
        label.textContent = labelText;
        wrap.append(label, control);
        settingsGrid.appendChild(wrap);
    };
    const numberSetting = (label, name, step = "1", min = "0") => {
        const widget = getWidget(node, name);
        addSetting(label, numberStepperControl(widget?.value ?? "", step, min, null, (value) => setWidgetValue(node, name, value)));
    };
    const selectSetting = (label, name, options) => {
        const widget = getWidget(node, name);
        addSetting(label, makeSelect(String(widget?.value || options[0]), options, (value) => setWidgetValue(node, name, value)));
    };
    numberSetting("Duration", "duration_seconds", "0.1", "0.1");
    numberSetting("FPS", "frame_rate", "1", "1");
    selectSetting("Guide mode", "guide_policy", ["every_checked_row", "safe_core_guides"]);
    numberSetting("Max guides", "max_guides", "1", "1");
    numberSetting("Guide gap", "min_guide_gap_seconds", "0.05", "0");
    numberSetting("Default force", "default_force", "0.01", "0");
    numberSetting("Ref width", "image_width", "32", "64");
    numberSetting("Ref height", "image_height", "32", "64");
    settingsPanel.appendChild(settingsGrid);
    root.insertBefore(settingsPanel, toolbar);

    const referencesPanel = document.createElement("div");
    referencesPanel.style.cssText = `margin-bottom:7px;padding:8px;border:1px solid ${chrome.border};background:#0F0B07;border-radius:6px;`;
    const referencesHead = document.createElement("div");
    referencesHead.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:6px;";
    const referencesTitle = document.createElement("div");
    referencesTitle.textContent = "Reference images";
    referencesTitle.style.cssText = "font-size:11px;color:#D8BC80;font-weight:700;";
    const referencesActions = document.createElement("div");
    referencesActions.style.cssText = "display:flex;gap:6px;flex-wrap:wrap;";
    const addImagesBtn = liteButton("Add Images", "primary");
    const importBoardBtn = liteButton("Import Board");
    const saveBoardBtn = liteButton("Save Board");
    const importRefsBtn = liteButton("Import Refs");
    const clearImagesBtn = liteButton("Clear Images", "danger");
    referencesActions.append(addImagesBtn, importBoardBtn, saveBoardBtn, importRefsBtn, clearImagesBtn);
    referencesHead.append(referencesTitle, referencesActions);
    const referencesGrid = document.createElement("div");
    referencesGrid.style.cssText = "display:flex;gap:8px;overflow-x:auto;padding-bottom:2px;min-height:74px;";
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.multiple = true;
    fileInput.accept = "image/*";
    fileInput.style.display = "none";
    const replaceFileInput = document.createElement("input");
    replaceFileInput.type = "file";
    replaceFileInput.accept = "image/*";
    replaceFileInput.style.display = "none";
    const boardFileInput = document.createElement("input");
    boardFileInput.type = "file";
    boardFileInput.accept = "application/json,.json";
    boardFileInput.style.display = "none";
    referencesPanel.append(referencesHead, referencesGrid, fileInput, replaceFileInput, boardFileInput);
    root.insertBefore(referencesPanel, toolbar);
    let pendingReplaceReferenceIndex = null;

    function referenceSecond(index, total) {
        const duration = Number(getWidget(node, "duration_seconds")?.value || 0);
        const safeDuration = Math.max(0, duration - 0.1);
        if (total <= 1 || safeDuration <= 0) return 0;
        return Math.round((safeDuration * (index / Math.max(1, total - 1))) * 10) / 10;
    }

    function makeLiteReferenceRow(refNumber, total, seed = {}) {
        const index = Math.max(0, Number(refNumber || 1) - 1);
        const defaultForce = Number(getWidget(node, "default_force")?.value || 0.22);
        return normalizeLiteRow({
            second: referenceSecond(index, Math.max(1, total || 1)),
            ref: Math.max(1, Number(refNumber) || 1),
            force: index === 0 ? Math.max(0.62, Math.min(0.88, defaultForce || 0.82)) : Math.max(0.14, Math.min(0.32, defaultForce || 0.22)),
            use_guide: true,
            use_prompt: false,
            label: `ref_${Math.max(1, Number(refNumber) || 1)}`,
            camera: "continuous dolly-in",
            transition: "continuous_motion",
            note: "",
            relay_prompt: "",
            ...seed,
        }, index);
    }

    function sync() {
        const clean = rows.map(normalizeLiteRow).sort((a, b) => a.second - b.second);
        setWidgetValue(node, "timeline_data", JSON.stringify({ rows: clean }, null, 2));
    }

    function syncRowsToReferencePaths(paths, forceOneToOne = false) {
        const total = Array.isArray(paths) ? paths.length : 0;
        if (forceOneToOne || !rows.length) {
            rows = Array.from({ length: Math.max(1, total || 1) }, (_, index) => makeLiteReferenceRow(index + 1, Math.max(1, total || 1)));
        } else {
            for (let ref = 1; ref <= total; ref += 1) {
                if (!rows.some((row, index) => Number(normalizeLiteRow(row, index).ref) === ref)) {
                    rows.push(makeLiteReferenceRow(ref, total));
                }
            }
        }
        sync();
    }

    function liteRowsAreOneToOneWithReferences(referenceCount) {
        const count = Number(referenceCount) || 0;
        if (!count || !rows.length || rows.length > count) return false;
        const seen = new Set();
        for (let index = 0; index < rows.length; index += 1) {
            const ref = Number(normalizeLiteRow(rows[index], index).ref);
            if (!Number.isFinite(ref) || ref < 1 || ref > count || seen.has(ref)) return false;
            seen.add(ref);
        }
        return true;
    }

    function renumberLiteRowsForReferenceOrder(referenceCount) {
        const count = Math.max(1, Number(referenceCount) || rows.length || 1);
        rows = rows.map((row, index) => normalizeLiteRow({
            ...row,
            ref: Math.min(index + 1, count),
        }, index));
        if (!rows.length) rows = [makeLiteReferenceRow(1, count)];
        sync();
    }

    function moveLiteReferenceAndLinkedRows(from, to) {
        const current = getConnectedReferencePaths(node);
        if (from < 0 || to < 0 || from >= current.length || to >= current.length || from === to) return;
        const paired = liteRowsAreOneToOneWithReferences(current.length);
        const nextPaths = moveItem(current, from, to);
        setOwnReferencePaths(node, nextPaths);
        if (paired) {
            rows = moveRowsKeepingTimelineSlots(rows, from, to, normalizeLiteRow);
            renumberLiteRowsForReferenceOrder(nextPaths.length);
        } else {
            rows = rows.map((row, index) => {
                const normalized = normalizeLiteRow(row, index);
                let ref = Number(normalized.ref);
                if (ref === from + 1) ref = to + 1;
                else if (from < to && ref > from + 1 && ref <= to + 1) ref -= 1;
                else if (from > to && ref >= to + 1 && ref < from + 1) ref += 1;
                return { ...normalized, ref };
            });
            sync();
        }
        drawReferenceStrip();
        draw();
    }

    function moveLiteRowAndLinkedReference(from, to) {
        const paths = getConnectedReferencePaths(node);
        const paired = liteRowsAreOneToOneWithReferences(paths.length);
        rows = moveRowsKeepingTimelineSlots(rows, from, to, normalizeLiteRow);
        if (paired && paths.length >= rows.length) {
            setOwnReferencePaths(node, moveItem(paths, from, to));
            renumberLiteRowsForReferenceOrder(paths.length);
            drawReferenceStrip();
        } else {
            sync();
        }
        draw();
    }

    function openReplaceReferencePicker(index) {
        const current = getConnectedReferencePaths(node);
        if (index < 0 || index >= current.length) return;
        pendingReplaceReferenceIndex = index;
        replaceFileInput.value = "";
        replaceFileInput.click();
    }

    function duplicateReferenceAndLinkedRow(index) {
        const current = getConnectedReferencePaths(node);
        if (index < 0 || index >= current.length) return;
        const nextPaths = current.slice();
        nextPaths.splice(index + 1, 0, current[index]);
        setOwnReferencePaths(node, nextPaths);

        const paired = rowsAreOneToOneWithReferences(current.length);
        const normalizedRows = rows.map((row, rowIndex) => normalizeShotboardRow(row, rowIndex));
        const duration = Math.max(0, Number(getWidget(node, "duration_seconds")?.value || 0));
        const roundedSecond = (value) => Math.round(Math.max(0, Number(value) || 0) * 10) / 10;
        const midpointSecond = (source, following) => {
            const sourceSecond = Number(source?.second || 0);
            const nextSecond = following ? Number(following.second || sourceSecond + 0.75) : Math.min(Math.max(0, duration - 0.1), sourceSecond + 0.75);
            if (Number.isFinite(nextSecond) && nextSecond > sourceSecond) return roundedSecond((sourceSecond + nextSecond) / 2);
            return roundedSecond(Math.min(Math.max(0, duration - 0.1), sourceSecond + 0.75));
        };
        const cloneFrom = (source, following, refNumber, rowIndex) => normalizeShotboardRow({
            ...source,
            ref: refNumber,
            second: midpointSecond(source, following),
            label: `${String(source?.label || `ref_${Math.max(1, refNumber - 1)}`)}_hold`,
            use_guide: true,
        }, rowIndex);

        if (paired) {
            const adjusted = normalizedRows.map((row, rowIndex) => rowIndex > index ? { ...row, ref: Number(row.ref || 1) + 1 } : row);
            const source = adjusted[index] || makeReferenceRow(index + 1, nextPaths.length);
            const clone = cloneFrom(source, adjusted[index + 1] || null, index + 2, index + 1);
            adjusted.splice(index + 1, 0, clone);
            rows = adjusted;
            sync();
        } else {
            const adjusted = normalizedRows.map((row) => {
                const ref = Number(row.ref || 1);
                return ref > index + 1 ? { ...row, ref: ref + 1 } : row;
            });
            const sourceRowIndex = adjusted.findIndex((row) => Number(row.ref || 1) === index + 1);
            const safeSourceIndex = sourceRowIndex >= 0 ? sourceRowIndex : Math.min(index, Math.max(0, adjusted.length - 1));
            const source = adjusted[safeSourceIndex] || makeReferenceRow(index + 1, nextPaths.length);
            const insertAt = sourceRowIndex >= 0 ? sourceRowIndex + 1 : adjusted.length;
            const clone = cloneFrom(source, adjusted[insertAt] || null, index + 2, insertAt);
            adjusted.splice(insertAt, 0, clone);
            rows = adjusted;
            sync();
        }
        drawReferenceStrip();
        draw();
    }

    function drawReferenceStrip() {
        referencesGrid.innerHTML = "";
        const paths = getConnectedReferencePaths(node);
        if (!paths.length) {
            const empty = document.createElement("div");
            empty.textContent = "No internal images. Add Images or Import Board.";
            empty.style.cssText = "color:#CBB17B;font-size:11px;padding:20px 4px;";
            referencesGrid.appendChild(empty);
            return;
        }
        paths.forEach((path, index) => {
            const card = document.createElement("div");
            card.style.cssText = "position:relative;flex:0 0 102px;height:70px;border:1px solid #6F5529;background:#080705;border-radius:5px;overflow:hidden;";
            card.draggable = true;
            card.ondragstart = (event) => {
                event.dataTransfer?.setData("text/iamccs-lite-ref", String(index));
                event.dataTransfer?.setData("text/plain", `ref:${index}`);
            };
            card.ondragover = (event) => {
                if (!Array.from(event.dataTransfer?.types || []).includes("text/iamccs-lite-ref")) return;
                event.preventDefault();
                card.style.borderColor = "#E2B75B";
            };
            card.ondragleave = () => { card.style.borderColor = "#6F5529"; };
            card.ondrop = (event) => {
                const raw = event.dataTransfer?.getData("text/iamccs-lite-ref");
                if (raw == null || raw === "") return;
                event.preventDefault();
                card.style.borderColor = "#6F5529";
                const from = Number(raw);
                if (!Number.isFinite(from) || from === index) return;
                moveLiteReferenceAndLinkedRows(from, index);
            };
            const img = document.createElement("img");
            img.src = previewUrlForPath(path);
            img.title = path;
            img.draggable = false;
            img.style.cssText = "width:100%;height:100%;object-fit:cover;display:block;";
            const swapBtn = document.createElement("button");
            swapBtn.textContent = "R";
            swapBtn.title = "Replace this reference image";
            swapBtn.style.cssText = "position:absolute;right:3px;top:3px;width:20px;height:18px;padding:0;border:1px solid #E2B75B;border-radius:3px;background:rgba(42,30,14,.92);color:#FFF2D8;font-size:10px;font-weight:800;cursor:pointer;";
            swapBtn.onclick = (event) => {
                event.preventDefault();
                event.stopPropagation();
                openReplaceReferencePicker(index);
            };
            const badge = document.createElement("div");
            badge.textContent = `ref ${index + 1}`;
            badge.style.cssText = "position:absolute;left:4px;bottom:3px;background:rgba(0,0,0,.75);color:#fff;font-size:11px;padding:1px 5px;border-radius:3px;";
            card.append(img, swapBtn, badge);
            referencesGrid.appendChild(card);
        });
    }

    const addBtn = liteButton("Add Row", "primary");
    const templateBtn = liteButton("3-Point FLF Template");
    const firstLastBtn = liteButton("First/Last Only");
    const smoothBtn = liteButton("Smooth Forces");
    const clearBtn = liteButton("Clear", "danger");
    toolbar.append(addBtn, templateBtn, firstLastBtn, smoothBtn, clearBtn);

    const statusBar = document.createElement("div");
    statusBar.style.cssText = `margin:0 0 7px 0;padding:7px 9px;border:1px solid ${chrome.border};border-radius:5px;background:#0F0B07;color:#EBD8AA;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;`;
    root.insertBefore(statusBar, table);

    function draw() {
        table.innerHTML = "";
        rows = rows.map(normalizeLiteRow).sort((a, b) => a.second - b.second);
        const paths = getConnectedReferencePaths(node);
        const activeGuides = rows.filter((row) => row.use_guide && Number(row.force || 0) > 0).length;
        statusBar.textContent = `FLF-only: ${activeGuides} guide rows active. Notes are private and are not sent to PromptRelay.`;

        const header = document.createElement("div");
        header.style.cssText = `display:grid;grid-template-columns:${SHOTBOARD_LITE_ROW_GRID};gap:8px;color:#D8BC80;font-size:10px;font-weight:700;padding:0 4px;`;
        header.innerHTML = "<div></div><div>Time</div><div>Image Ref</div><div>Force</div><div>Guide</div><div>Label</div><div>Notes</div><div></div>";
        table.appendChild(header);

        rows.forEach((row, index) => {
            const r = normalizeLiteRow(row, index);
            rows[index] = r;
            const updateRow = (patch) => {
                rows[index] = normalizeLiteRow({ ...rows[index], ...patch }, index);
                sync();
            };
            const card = document.createElement("div");
            card.style.cssText = `
                display:grid;
                grid-template-columns:${SHOTBOARD_LITE_ROW_GRID};
                gap:8px;
                align-items:start;
                padding:8px 6px;
                border:1px solid ${chrome.border};
                background:#130E08;
                border-radius:6px;
                box-sizing:border-box;
                width:100%;
                overflow:hidden;
            `;

            const handle = document.createElement("button");
            handle.textContent = "::";
            handle.title = "Drag shot row";
            handle.draggable = true;
            handle.style.cssText = "width:22px;height:28px;padding:0;border:1px solid #73572A;border-radius:4px;background:#090705;color:#D8BC80;cursor:grab;font-size:12px;";
            handle.ondragstart = (event) => {
                event.stopPropagation();
                event.dataTransfer?.setData("text/iamccs-lite-row", String(index));
            };
            protectDragHandle(handle);
            card.ondragover = (event) => {
                if (!Array.from(event.dataTransfer?.types || []).includes("text/iamccs-lite-row")) return;
                event.preventDefault();
                card.style.borderColor = "#E2B75B";
            };
            card.ondragleave = () => { card.style.borderColor = chrome.border; };
            card.ondrop = (event) => {
                const raw = event.dataTransfer?.getData("text/iamccs-lite-row");
                if (raw == null || raw === "") return;
                event.preventDefault();
                card.style.borderColor = chrome.border;
                const from = Number(raw);
                if (!Number.isFinite(from) || from === index) return;
                moveLiteRowAndLinkedReference(from, index);
            };

            const sec = timeControl(r.second, (value) => updateRow({ second: value }));
            const ref = refPicker(r.ref, paths, (value) => { updateRow({ ref: value }); draw(); }, {
                thumbWidth: 152,
                thumbHeight: 92,
                onReplace: (referenceIndex) => openReplaceReferencePicker(referenceIndex),
            });
            const force = forceControl(r.force, (value) => updateRow({ force: value }));
            const guide = checkbox(r.use_guide, (value) => updateRow({ use_guide: value }));
            guide.title = "Use this row as an FLF image guide";

            const label = document.createElement("input");
            label.value = r.label;
            label.style.cssText = inputBase() + "background:#0A0907;border-color:#70572E;color:#FFF2D8;";
            label.oninput = () => updateRow({ label: label.value });
            protectControlDrag(label);

            const notes = document.createElement("textarea");
            notes.value = r.note || "";
            notes.rows = 4;
            notes.placeholder = "Private note for this FLF guide. Not sent to PromptRelay.";
            notes.style.cssText = inputBase() + "resize:vertical;min-height:84px;line-height:1.32;font-size:11px;padding:8px 15px 8px 9px;scrollbar-gutter:stable;background:#0A0907;border-color:#70572E;color:#FFF2D8;";
            notes.oninput = () => updateRow({ note: notes.value });
            protectControlDrag(notes);

            const del = liteButton("x", "danger");
            del.style.width = "26px";
            del.style.height = "30px";
            del.style.padding = "0";
            del.onclick = () => {
                if (rows.length <= 1) {
                    statusBar.textContent = "At least one FLF row must remain.";
                    return;
                }
                rows.splice(index, 1);
                sync();
                draw();
            };
            protectControlDrag(del);

            card.append(handle, sec, ref, force, guide, label, notes, del);
            table.appendChild(card);
        });
        sync();
    }

    addImagesBtn.onclick = () => fileInput.click();
    importRefsBtn.onclick = () => fileInput.click();
    fileInput.onchange = async (event) => {
        const uploaded = await uploadShotboardImages(event.target.files);
        if (uploaded.length) {
            const current = getOwnReferencePaths(node);
            const next = [...current, ...uploaded];
            setOwnReferencePaths(node, next);
            syncRowsToReferencePaths(next);
            drawReferenceStrip();
            draw();
        }
        fileInput.value = "";
    };
    replaceFileInput.onchange = async (event) => {
        const replaceIndex = pendingReplaceReferenceIndex;
        pendingReplaceReferenceIndex = null;
        const file = event.target.files?.[0];
        if (file && Number.isFinite(Number(replaceIndex))) {
            const uploaded = await uploadShotboardImages([file]);
            if (uploaded[0]) {
                replaceReferencePathAt(node, Number(replaceIndex), uploaded[0]);
                drawReferenceStrip();
                draw();
            }
        }
        replaceFileInput.value = "";
    };
    clearImagesBtn.onclick = () => {
        setOwnReferencePaths(node, []);
        rows = [makeLiteReferenceRow(1, 1)];
        drawReferenceStrip();
        draw();
    };
    saveBoardBtn.onclick = () => {
        syncLitePromptWidget();
        const currentPrompt = String(promptArea.value || getWidget(node, "global_prompt")?.value || "");
        const board = {
            metadata: {
                schema: "iamccs.cine.shotboard.lite.board",
                schema_version: 1,
                node: "IAMCCS_CineShotboardLite",
                promptrelay: "disabled",
            },
            global_prompt: currentPrompt,
            prompt: currentPrompt,
            timeline_data: JSON.stringify({ rows: rows.map(normalizeLiteRow) }, null, 2),
            rows: rows.map(normalizeLiteRow),
            duration_seconds: Number(getWidget(node, "duration_seconds")?.value || 8),
            frame_rate: Number(getWidget(node, "frame_rate")?.value || 24),
            guide_policy: String(getWidget(node, "guide_policy")?.value || "every_checked_row"),
            min_guide_gap_seconds: Number(getWidget(node, "min_guide_gap_seconds")?.value || 0),
            max_guides: Number(getWidget(node, "max_guides")?.value || 12),
            default_force: Number(getWidget(node, "default_force")?.value || 0.22),
            image_paths: getConnectedReferencePaths(node),
            image_width: Number(getWidget(node, "image_width")?.value || 768),
            image_height: Number(getWidget(node, "image_height")?.value || 432),
        };
        downloadJsonFile(board, safeBoardFilename("iamccs_cine_shotboard_lite"));
    };
    importBoardBtn.onclick = () => boardFileInput.click();
    boardFileInput.onchange = async (event) => {
        const file = event.target.files?.[0];
        if (!file) return;
        try {
            const raw = await readJsonFile(file);
            const data = boardFromWorkflowJson(raw) || raw;
            const settings = data.settings && typeof data.settings === "object" ? data.settings : {};
            const boardValue = (name) => Object.prototype.hasOwnProperty.call(data, name) ? data[name] : settings[name];
            const importedPrompt = data.global_prompt ?? data.prompt ?? data.globalPrompt ?? "";
            if (importedPrompt != null) {
                promptArea.value = String(importedPrompt || "");
                syncLitePromptWidget();
            }
            if (boardValue("duration_seconds") != null) setWidgetValue(node, "duration_seconds", Number(boardValue("duration_seconds")));
            if (boardValue("frame_rate") != null) setWidgetValue(node, "frame_rate", Number(boardValue("frame_rate")));
            if (boardValue("guide_policy") != null) setWidgetValue(node, "guide_policy", String(boardValue("guide_policy")));
            if (boardValue("min_guide_gap_seconds") != null) setWidgetValue(node, "min_guide_gap_seconds", Number(boardValue("min_guide_gap_seconds")));
            if (boardValue("max_guides") != null) setWidgetValue(node, "max_guides", Number(boardValue("max_guides")));
            if (boardValue("default_force") != null) setWidgetValue(node, "default_force", Number(boardValue("default_force")));
            if (boardValue("image_width") != null) setWidgetValue(node, "image_width", Number(boardValue("image_width")));
            if (boardValue("image_height") != null) setWidgetValue(node, "image_height", Number(boardValue("image_height")));
            const paths = Array.isArray(data.image_paths) ? data.image_paths : splitReferencePaths(data.image_paths);
            if (paths.length) setOwnReferencePaths(node, paths);
            const importedRows = Array.isArray(data.rows) ? data.rows : parseShotboardTimelineString(String(data.timeline_data || ""));
            rows = (importedRows.length ? importedRows : defaultLiteRows()).map(normalizeLiteRow);
            if (!rows.length) rows = [makeLiteReferenceRow(1, Math.max(1, paths.length))];
            sync();
            drawReferenceStrip();
            draw();
        } catch (err) {
            console.error("[IAMCCS Cine Shotboard Lite] board load failed", err);
            statusBar.textContent = `Board import failed: ${err?.message || err}`;
        }
        boardFileInput.value = "";
    };

    addBtn.onclick = () => {
        rows.push(makeLiteReferenceRow(rows.length + 1, Math.max(rows.length + 1, getConnectedReferencePaths(node).length || rows.length + 1), {
            second: rows.length ? Number(rows[rows.length - 1].second) + 2.5 : 0,
            label: `lite_shot_${rows.length + 1}`,
            note: "",
        }));
        draw();
    };
    templateBtn.onclick = () => {
        const paths = getConnectedReferencePaths(node);
        const count = Math.max(3, Math.min(paths.length || 3, 6));
        rows = Array.from({ length: count }, (_, index) => makeLiteReferenceRow(index + 1, count));
        draw();
    };
    firstLastBtn.onclick = () => {
        const paths = getConnectedReferencePaths(node);
        const count = Math.max(2, paths.length || 2);
        rows = [
            makeLiteReferenceRow(1, count, { force: 0.82, label: "first_anchor", second: 0 }),
            makeLiteReferenceRow(count, count, { force: 0.32, label: "last_anchor", second: referenceSecond(count - 1, count) }),
        ];
        draw();
    };
    smoothBtn.onclick = () => {
        rows = rows.map((row, index) => {
            const r = normalizeLiteRow(row, index);
            return { ...r, force: index === 0 ? Math.min(Math.max(r.force, 0.62), 0.86) : Math.min(r.force || 0.22, 0.28) };
        });
        draw();
    };
    clearBtn.onclick = () => { rows = [makeLiteReferenceRow(1, Math.max(1, getConnectedReferencePaths(node).length))]; draw(); };

    drawReferenceStrip();
    draw();
    const widget = node.addDOMWidget("Cine Shotboard Lite", "iamccs_cine_shotboard_lite", root, { serialize: false });
    widget.computeSize = (width) => {
        const rowCount = Math.max(1, rows.length);
        return [width, Math.min(700, Math.max(560, 300 + rowCount * 132))];
    };
}

function renderShotboardPro(node) {
    if (node._iamccsCineShotboardReady) return;
    node._iamccsCineShotboardReady = true;
    const shotboardV2 = isShotboardV2Node(node);
    const shotChrome = shotboardV2 ? CINE_NODE_CHROME.shotboardV2 : CINE_NODE_CHROME.shotboard;
    const shotButtonPalette = shotboardV2
        ? {
            primaryBg: "#11617A",
            primaryBorder: "#82D8EF",
            neutralBg: "#0E2D3A",
            neutralBorder: "#2D7A92",
            dangerBg: "#663344",
            dangerBorder: "#C66878",
            text: "#E9F8FF",
        }
        : {
            primaryBg: "#1D5A3F",
            primaryBorder: "#9ED784",
            neutralBg: "#16352A",
            neutralBorder: "#3D755E",
            dangerBg: "#613030",
            dangerBorder: "#B86969",
            text: "#EEF7E8",
        };
    const shotButton = (label, tone = "neutral") => button(label, tone, shotButtonPalette);
    lockNodeMinimumSize(node, SHOTBOARD_NODE_MIN_SIZE, { lockResize: true, preferredSize: SHOTBOARD_NODE_DEFAULT_SIZE });
    node.color = shotChrome.header;
    node.bgcolor = shotChrome.nodeBg;
    node.boxcolor = shotChrome.box;

    hideWidget(getWidget(node, "timeline_data"));
    hideWidget(getWidget(node, "global_prompt"));
    hideWidget(getWidget(node, "duration_seconds"));
    hideWidget(getWidget(node, "frame_rate"));
    hideWidget(getWidget(node, "guide_policy"));
    hideWidget(getWidget(node, "min_guide_gap_seconds"));
    hideWidget(getWidget(node, "max_guides"));
    hideWidget(getWidget(node, "default_force"));
    hideWidget(getWidget(node, "promptrelay_epsilon"));
    hideWidget(getWidget(node, "ltx_round_mode"));
    hideWidget(getWidget(node, "tail_safety_frames"));
    hideWidget(getWidget(node, "image_paths"));
    hideWidget(getWidget(node, "image_width"));
    hideWidget(getWidget(node, "image_height"));

    const initialSourceRows = parseJsonWidget(node, defaultShotboardRows);
    const initialHasCanonicalRelay = initialSourceRows.some(rowHasCanonicalRelayPrompt);
    const initialHasLegacyNotes = initialSourceRows.some((row) => firstNonEmpty(row?.note, row?.camera_note));
    let rows = initialSourceRows.map((row, index) => normalizeShotboardRow(row, index, {
        useNoteAsRelayFallback: !initialHasCanonicalRelay && initialHasLegacyNotes,
    }));
    const { root, toolbar, table } = tableShell(
        shotboardV2 ? "Cine Shotboard Timeline Pro V2" : "Cine Shotboard Timeline Pro",
        ""
    );
    root.style.borderColor = shotChrome.border;
    root.style.boxShadow = `inset 0 1px 0 ${shotChrome.glow}, 0 0 0 1px rgba(0,0,0,.35)`;
    root.style.maxHeight = "790px";
    root.style.overflowY = "auto";
    root.style.overflowX = "hidden";

    const globalPromptWidget = getWidget(node, "global_prompt");
    const promptPanel = document.createElement("div");
    promptPanel.style.cssText = `margin:6px 0 8px 0;padding:9px;border:1px solid ${CINE_FILM_LAB.border};background:${CINE_FILM_LAB.panelDark};border-radius:6px;`;
    const promptLabel = document.createElement("div");
    promptLabel.textContent = "Global prompt";
    promptLabel.style.cssText = `font-size:11px;color:${CINE_FILM_LAB.muted};margin-bottom:5px;font-weight:600;`;
    const promptArea = document.createElement("textarea");
    promptArea.value = String(globalPromptWidget?.value || "");
    promptArea.rows = 3;
    promptArea.style.cssText = inputBase() + "resize:vertical;min-height:66px;line-height:1.38;padding:9px 18px 9px 10px;scrollbar-gutter:stable;";
    const syncProPromptWidget = () => setWidgetValue(node, "global_prompt", promptArea.value);
    promptArea.oninput = syncProPromptWidget;
    promptArea.onchange = syncProPromptWidget;
    promptPanel.append(promptLabel, promptArea);
    root.insertBefore(promptPanel, toolbar);

    const settingsPanel = document.createElement("div");
    settingsPanel.style.cssText = `margin:0 0 8px 0;padding:9px;border:1px solid ${CINE_FILM_LAB.border};background:${CINE_FILM_LAB.panelDark};border-radius:6px;`;
    const settingsGrid = document.createElement("div");
    settingsGrid.style.cssText = "display:grid;grid-template-columns:repeat(4,minmax(132px,1fr));gap:8px;";
    const settingControls = new Map();
    const settingNames = [
        "duration_seconds",
        "frame_rate",
        "guide_policy",
        "min_guide_gap_seconds",
        "max_guides",
        "default_force",
        "promptrelay_epsilon",
        "ltx_round_mode",
        "tail_safety_frames",
        "image_width",
        "image_height",
    ];
    const addSetting = (labelText, control) => {
        const wrap = document.createElement("label");
        wrap.style.cssText = `display:flex;flex-direction:column;gap:4px;color:${CINE_FILM_LAB.muted};font-size:11px;font-weight:600;`;
        const label = document.createElement("span");
        label.textContent = labelText;
        wrap.append(label, control);
        settingsGrid.appendChild(wrap);
    };
    const numberSetting = (label, name, step = "1", min = "0") => {
        const widget = getWidget(node, name);
        const control = numberStepperControl(widget?.value ?? "", step, min, null, (value) => setWidgetValue(node, name, value));
        settingControls.set(name, control);
        addSetting(label, control);
    };
    const selectSetting = (label, name, options, afterChange = null) => {
        const widget = getWidget(node, name);
        const select = makeSelect(String(widget?.value || options[0]), options, (value) => {
            setWidgetValue(node, name, value);
            if (afterChange) afterChange(value);
        });
        settingControls.set(name, select);
        addSetting(label, select);
    };

    const numericSettingValue = (name, fallback = 0) => {
        const value = Number(getWidget(node, name)?.value);
        return Number.isFinite(value) ? value : fallback;
    };
    const guideIndexesForPolicy = (sourceRows, policyValue = null) => {
        const policy = String(policyValue || getWidget(node, "guide_policy")?.value || "safe_core_guides");
        const maxGuides = Math.max(0, Math.floor(numericSettingValue("max_guides", 5)));
        const minGap = Math.max(0, numericSettingValue("min_guide_gap_seconds", 0));
        if (policy === "prompt_only" || maxGuides <= 0) return new Set();
        const normalized = sourceRows.map(normalizeShotboardRow);
        const candidates = normalized
            .map((row, index) => ({ ...row, __rowIndex: index }))
            .filter((row) => Number(row.force || 0) > 0);
        const chosen = [];
        const farEnough = (row) => !chosen.some((existing) => Math.abs(Number(row.second || 0) - Number(existing.second || 0)) < minGap);
        if (policy === "safe_core_guides") {
            const scored = candidates.map((row, index) => {
                let score = Number(row.force || 0);
                if (index === 0) score += 0.35;
                if (index === candidates.length - 1) score += 0.25;
                if (String(row.transition || "") === "hard_cut") score -= 0.25;
                return { row, index, score };
            }).sort((a, b) => (b.score - a.score) || (a.index - b.index));
            for (const item of scored) {
                if (chosen.length >= maxGuides) break;
                if (farEnough(item.row)) chosen.push(item.row);
            }
        } else {
            for (const row of candidates) {
                if (chosen.length >= maxGuides) break;
                if (farEnough(row)) chosen.push(row);
            }
        }
        return new Set(chosen.map((row) => row.__rowIndex));
    };
    const applyGuidePolicyToRows = (policyValue = null) => {
        const selected = guideIndexesForPolicy(rows, policyValue);
        rows = rows.map((row, index) => ({ ...normalizeShotboardRow(row, index), use_guide: selected.has(index) }));
        draw();
    };

    numberSetting("Duration", "duration_seconds", "0.1", "0.1");
    numberSetting("FPS", "frame_rate", "1", "1");
    selectSetting("Guide policy", "guide_policy", ["safe_core_guides", "prompt_only", "every_checked_row"], applyGuidePolicyToRows);
    numberSetting("Max guides", "max_guides", "1", "0");
    numberSetting("Guide gap", "min_guide_gap_seconds", "0.05", "0");
    numberSetting("Default force", "default_force", "0.01", "0");
    numberSetting("Relay softness", "promptrelay_epsilon", "0.01", "0");
    selectSetting("LTX frames", "ltx_round_mode", ["up_8n_plus_1", "nearest_8n_plus_1", "none"]);
    numberSetting("Ref width", "image_width", "32", "64");
    numberSetting("Ref height", "image_height", "32", "64");
    addSetting("Editor zoom", editorZoomControl(root, node));
    settingsPanel.appendChild(settingsGrid);
    root.insertBefore(settingsPanel, toolbar);

    const referencesPanel = document.createElement("div");
    referencesPanel.style.cssText = `margin-bottom:7px;padding:9px;border:1px solid ${CINE_FILM_LAB.border};background:${CINE_FILM_LAB.panelDark};border-radius:6px;`;
    const referencesHead = document.createElement("div");
    referencesHead.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:6px;";
    const referencesTitle = document.createElement("div");
    referencesTitle.textContent = shotboardV2 ? "References inside this node - frame editor enabled" : "References inside this node";
    referencesTitle.style.cssText = `font-size:11px;color:${CINE_FILM_LAB.muted};font-weight:600;`;
    const referencesActions = document.createElement("div");
    referencesActions.style.cssText = "display:flex;gap:6px;flex-wrap:wrap;";
    const addImagesBtn = shotButton("Add Images", "primary");
    const importBoardBtn = shotButton("Import Board");
    const saveBoardBtn = shotButton("Save Board");
    const importRefsBtn = shotButton("Import Refs");
    const clearImagesBtn = shotButton("Clear Images", "danger");
    referencesActions.append(addImagesBtn, importBoardBtn, saveBoardBtn, importRefsBtn, clearImagesBtn);
    referencesHead.append(referencesTitle, referencesActions);
    const referencesGrid = document.createElement("div");
    referencesGrid.style.cssText = "display:flex;gap:8px;overflow-x:auto;padding-bottom:2px;min-height:76px;";
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.multiple = true;
    fileInput.accept = "image/*";
    fileInput.style.display = "none";
    const replaceFileInput = document.createElement("input");
    replaceFileInput.type = "file";
    replaceFileInput.accept = "image/*";
    replaceFileInput.style.display = "none";
    const boardFileInput = document.createElement("input");
    boardFileInput.type = "file";
    boardFileInput.accept = "application/json,.json";
    boardFileInput.style.display = "none";
    referencesPanel.append(referencesHead, referencesGrid, fileInput, replaceFileInput, boardFileInput);
    root.insertBefore(referencesPanel, toolbar);
    let pendingReplaceReferenceIndex = null;

    function referenceSecond(index, total) {
        const duration = Number(getWidget(node, "duration_seconds")?.value || 0);
        const safeDuration = Math.max(0, duration - 0.1);
        if (total <= 1 || safeDuration <= 0) return 0;
        return Math.round((safeDuration * (index / Math.max(1, total - 1))) * 10) / 10;
    }

    function makeReferenceRow(refNumber, total, seed = {}) {
        const index = Math.max(0, Number(refNumber || 1) - 1);
        const defaultForce = Number(getWidget(node, "default_force")?.value || 0.18);
        return normalizeShotboardRow({
            second: referenceSecond(index, Math.max(1, total || 1)),
            ref: Math.max(1, Number(refNumber) || 1),
            force: index === 0 ? Math.max(0.5, Math.min(0.72, defaultForce || 0.62)) : Math.max(0.12, Math.min(0.28, defaultForce || 0.18)),
            use_guide: true,
            use_prompt: false,
            label: `ref_${Math.max(1, Number(refNumber) || 1)}`,
            camera: "continuous dolly-in",
            transition: "continuous_motion",
            note: "",
            relay_prompt: "",
            camera_relay_mode: "off",
            transition_relay_mode: "off",
            relay_addon_position: "after",
            ...seed,
        }, index);
    }

    function isFactoryDefaultRows() {
        const defaults = defaultShotboardRows();
        if (!rows.length || rows.length !== defaults.length) return false;
        return rows.every((row, index) => {
            const current = normalizeShotboardRow(row, index);
            const expected = normalizeShotboardRow(defaults[index], index);
            return current.label === expected.label
                && Number(current.ref) === Number(expected.ref)
                && String(current.relay_prompt || "") === String(expected.relay_prompt || "");
        });
    }

    function rowsAreOneToOneWithReferences(referenceCount) {
        const count = Number(referenceCount) || 0;
        if (!count || !rows.length || rows.length > count) return false;
        const seen = new Set();
        for (let index = 0; index < rows.length; index += 1) {
            const ref = Number(normalizeShotboardRow(rows[index], index).ref);
            if (!Number.isFinite(ref) || ref < 1 || ref > count || seen.has(ref)) return false;
            seen.add(ref);
        }
        return true;
    }

    function renumberRowsForReferenceOrder(referenceCount, keepSeconds = true) {
        const count = Math.max(1, Number(referenceCount) || rows.length || 1);
        rows = rows.map((row, index) => {
            const normalized = normalizeShotboardRow(row, index);
            return {
                ...normalized,
                ref: Math.min(index + 1, count),
                second: keepSeconds ? normalized.second : referenceSecond(index, count),
            };
        });
        if (!rows.length) rows = [makeReferenceRow(1, count)];
        sync();
    }

    function syncRowsToReferencePaths(paths, options = {}) {
        const total = Array.isArray(paths) ? paths.length : 0;
        const forceOneToOne = Boolean(options.forceOneToOne);
        if (forceOneToOne) {
            const count = Math.max(1, total);
            rows = Array.from({ length: count }, (_, index) => makeReferenceRow(index + 1, count));
            sync();
            return;
        }

        if (!rows.length) {
            rows = [makeReferenceRow(1, Math.max(1, total))];
        }
        for (let ref = 1; ref <= total; ref += 1) {
            if (!rows.some((row, index) => Number(normalizeShotboardRow(row, index).ref) === ref)) {
                rows.push(makeReferenceRow(ref, total));
            }
        }
        sync();
    }

    function moveReferenceAndLinkedRows(from, to) {
        const current = getConnectedReferencePaths(node);
        if (from < 0 || to < 0 || from >= current.length || to >= current.length || from === to) return;
        const paired = rowsAreOneToOneWithReferences(current.length);
        const nextPaths = moveItem(current, from, to);
        setOwnReferencePaths(node, nextPaths);

        if (paired) {
            rows = moveRowsKeepingTimelineSlots(rows, from, to, normalizeShotboardRow);
            renumberRowsForReferenceOrder(nextPaths.length, true);
        } else {
            const refOrder = current.map((_, refIndex) => refIndex + 1);
            const movedRefOrder = moveItem(refOrder, from, to);
            const oldRefToNewRef = new Map(movedRefOrder.map((oldRef, newIndex) => [oldRef, newIndex + 1]));
            const normalized = rows.map((row, index) => normalizeShotboardRow(row, index));
            const timeSlots = normalized.map((row) => Number(row.second || 0)).sort((a, b) => a - b);
            const groups = new Map(refOrder.map((ref) => [ref, []]));
            const extras = [];
            for (const row of normalized) {
                const ref = Number(row.ref || 1);
                if (groups.has(ref)) groups.get(ref).push(row);
                else extras.push(row);
            }
            const reordered = [];
            for (const oldRef of movedRefOrder) {
                const newRef = oldRefToNewRef.get(oldRef) || oldRef;
                for (const row of groups.get(oldRef) || []) {
                    reordered.push({ ...row, ref: newRef });
                }
            }
            reordered.push(...extras);
            rows = reordered.map((row, index) => normalizeShotboardRow({
                ...row,
                second: Number.isFinite(timeSlots[index]) ? timeSlots[index] : row.second,
            }, index));
            sync();
        }
        drawReferenceStrip();
        draw();
    }

    function removeReferenceAndLinkedRows(index) {
        const current = getConnectedReferencePaths(node);
        if (index < 0 || index >= current.length) return;
        const paired = rowsAreOneToOneWithReferences(current.length);
        const next = current.slice();
        next.splice(index, 1);
        setOwnReferencePaths(node, next);

        if (paired) {
            rows.splice(index, 1);
            if (!rows.length) rows = [makeReferenceRow(1, Math.max(1, next.length))];
            renumberRowsForReferenceOrder(next.length || 1, true);
        } else {
            rows = rows.map((row, rowIndex) => {
                const normalized = normalizeShotboardRow(row, rowIndex);
                const ref = Number(normalized.ref);
                if (ref === index + 1) return next.length ? { ...normalized, ref: Math.min(index + 1, next.length) } : normalized;
                if (ref > index + 1) return { ...normalized, ref: ref - 1 };
                return normalized;
            });
            if (!rows.length) rows = [makeReferenceRow(1, Math.max(1, next.length))];
            sync();
        }
        drawReferenceStrip();
        draw();
    }

    function openReplaceReferencePicker(index) {
        const current = getConnectedReferencePaths(node);
        if (index < 0 || index >= current.length) return;
        pendingReplaceReferenceIndex = index;
        replaceFileInput.value = "";
        replaceFileInput.click();
    }

    function duplicateReferenceAndLinkedRow(referenceIndex, rowIndexHint = null) {
        const current = getConnectedReferencePaths(node);
        const index = Math.floor(Number(referenceIndex));
        if (!Number.isFinite(index) || index < 0 || index >= current.length) return;

        const nextPaths = current.slice();
        nextPaths.splice(index + 1, 0, current[index]);
        setOwnReferencePaths(node, nextPaths);

        const normalizedRows = rows.map((row, rowIndex) => normalizeShotboardRow(row, rowIndex));
        const paired = rowsAreOneToOneWithReferences(current.length);
        const duration = Math.max(0, Number(getWidget(node, "duration_seconds")?.value || 0));
        const safeEnd = Math.max(0, duration - 0.1);
        const roundedSecond = (value) => Math.round(Math.max(0, Number(value) || 0) * 10) / 10;
        const midpointSecond = (source, following) => {
            const sourceSecond = Number(source?.second || 0);
            const nextSecond = following ? Number(following.second || sourceSecond + 0.75) : Math.min(safeEnd, sourceSecond + 0.75);
            if (Number.isFinite(nextSecond) && nextSecond > sourceSecond) return roundedSecond((sourceSecond + nextSecond) / 2);
            return roundedSecond(Math.min(safeEnd, sourceSecond + 0.75));
        };
        const duplicateLabel = (source, refNumber) => {
            const base = String(source?.label || `ref_${Math.max(1, refNumber - 1)}`).replace(/_dup\d*$/i, "");
            return `${base}_dup`;
        };
        const cloneFrom = (source, following, refNumber, rowIndex) => normalizeShotboardRow({
            ...source,
            ref: refNumber,
            second: midpointSecond(source, following),
            label: duplicateLabel(source, refNumber),
            use_guide: true,
        }, rowIndex);

        if (paired) {
            const adjusted = normalizedRows.map((row, rowIndex) => rowIndex > index ? { ...row, ref: Number(row.ref || 1) + 1 } : row);
            const source = adjusted[index] || makeReferenceRow(index + 1, nextPaths.length);
            const clone = cloneFrom(source, adjusted[index + 1] || null, index + 2, index + 1);
            adjusted.splice(index + 1, 0, clone);
            rows = adjusted;
        } else {
            const adjusted = normalizedRows.map((row) => {
                const ref = Number(row.ref || 1);
                return ref > index + 1 ? { ...row, ref: ref + 1 } : row;
            });
            const hinted = Number.isFinite(Number(rowIndexHint)) ? Math.floor(Number(rowIndexHint)) : -1;
            const hintedMatches = hinted >= 0 && hinted < adjusted.length && Number(adjusted[hinted]?.ref || 1) === index + 1;
            const sourceRowIndex = hintedMatches ? hinted : adjusted.findIndex((row) => Number(row.ref || 1) === index + 1);
            const safeSourceIndex = sourceRowIndex >= 0 ? sourceRowIndex : Math.min(index, Math.max(0, adjusted.length - 1));
            const source = adjusted[safeSourceIndex] || makeReferenceRow(index + 1, nextPaths.length);
            const insertAt = safeSourceIndex >= 0 ? safeSourceIndex + 1 : adjusted.length;
            const clone = cloneFrom(source, adjusted[insertAt] || null, index + 2, insertAt);
            adjusted.splice(insertAt, 0, clone);
            rows = adjusted.map((row, rowIndex) => normalizeShotboardRow(row, rowIndex));
        }

        sync();
        drawReferenceStrip();
        draw();
    }

    function drawReferenceStrip() {
        referencesGrid.innerHTML = "";
        const paths = getConnectedReferencePaths(node);
        if (!paths.length) {
            const empty = document.createElement("div");
            empty.textContent = "No internal images. Add Images or Import Board.";
            empty.style.cssText = `color:${CINE_FILM_LAB.muted};font-size:11px;padding:20px 4px;`;
            referencesGrid.appendChild(empty);
            return;
        }
        paths.forEach((path, index) => {
            const card = document.createElement("div");
            card.style.cssText = `position:relative;flex:0 0 ${shotboardV2 ? "152px" : "104px"};height:72px;border:1px solid ${CINE_FILM_LAB.borderSoft};background:${CINE_FILM_LAB.field};border-radius:5px;overflow:hidden;`;
            card.draggable = true;
            card.ondragstart = (event) => {
                event.dataTransfer?.setData("text/iamccs-ref", String(index));
                event.dataTransfer?.setData("text/plain", `ref:${index}`);
            };
            card.ondragover = (event) => {
                if (!Array.from(event.dataTransfer?.types || []).includes("text/iamccs-ref")) return;
                event.preventDefault();
            };
            card.ondrop = (event) => {
                const raw = event.dataTransfer?.getData("text/iamccs-ref");
                if (raw == null || raw === "") return;
                event.preventDefault();
                const from = Number(raw);
                if (!Number.isFinite(from) || from === index) return;
                moveReferenceAndLinkedRows(from, index);
            };
            const img = document.createElement("img");
            img.src = previewUrlForPath(path);
            img.title = path;
            img.draggable = false;
            const stripeActionWidth = shotboardV2 ? 28 : 0;
            img.style.cssText = shotboardV2
                ? `width:calc(100% - ${stripeActionWidth}px);height:100%;margin-left:${stripeActionWidth}px;object-fit:cover;display:block;`
                : "width:100%;height:100%;object-fit:cover;display:block;";
            if (shotboardV2) {
                const stripeRail = document.createElement("div");
                stripeRail.style.cssText = [
                    "position:absolute",
                    "left:0",
                    "top:0",
                    "bottom:0",
                    `width:${stripeActionWidth}px`,
                    "display:grid",
                    "grid-template-columns:1fr",
                    "grid-template-rows:repeat(3,1fr)",
                    `border-right:1px solid ${CINE_FILM_LAB.borderSoft}`,
                    "overflow:hidden",
                ].join(";");
                const makeRailButton = (label, titleText, action) => {
                    const tone = cineReferenceActionTone(label);
                    const btn = document.createElement("button");
                    btn.textContent = label;
                    btn.title = titleText;
                    btn.style.cssText = [
                        "width:100%",
                        "height:100%",
                        "padding:0",
                        "border:0",
                        `border-bottom:1px solid ${tone.border}`,
                        `background:${tone.bg}`,
                        `color:${tone.color}`,
                        "display:flex",
                        "align-items:center",
                        "justify-content:center",
                        "font-size:11px",
                        "font-weight:900",
                        "line-height:1",
                        "letter-spacing:0",
                        "writing-mode:horizontal-tb",
                        "transform:none",
                        "cursor:pointer",
                    ].join(";");
                    btn.onclick = (event) => {
                        event.preventDefault();
                        event.stopPropagation();
                        action();
                    };
                    return protectControlDrag(btn);
                };
                stripeRail.append(
                    makeRailButton("E", "Open frame editor: crop, pan, zoom and save a new reference", () => {
                        openReferenceFrameEditor(node, index, path, (newPath) => {
                            replaceReferencePathAt(node, index, newPath);
                            drawReferenceStrip();
                            draw();
                        });
                    }),
                    makeRailButton("R", "Replace this reference image without changing row timing", () => {
                        openReplaceReferencePicker(index);
                    }),
                    makeRailButton("D", "Duplicate this reference into the next slot and create a new keyframe", () => {
                        duplicateReferenceAndLinkedRow(index);
                    })
                );
                card.appendChild(stripeRail);
            }
            const badge = document.createElement("div");
            badge.textContent = `ref ${index + 1}`;
            badge.style.cssText = `position:absolute;left:${shotboardV2 ? `${stripeActionWidth + 4}px` : "4px"};bottom:3px;background:rgba(0,0,0,.75);color:#fff;font-size:11px;padding:1px 5px;border-radius:3px;`;
            const controls = document.createElement("div");
            controls.style.cssText = "position:absolute;top:2px;right:2px;display:flex;gap:2px;";
            const makeMini = (label, titleText, action) => {
                const mini = document.createElement("button");
                mini.textContent = label;
                mini.title = titleText;
                mini.style.cssText = "width:19px;height:19px;padding:0;border:1px solid rgba(255,255,255,.35);border-radius:3px;background:rgba(10,15,18,.84);color:#fff;font-size:10px;line-height:16px;cursor:pointer;";
                mini.onclick = (event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    action();
                };
                return mini;
            };
            const miniButtons = [
                makeMini("<", "Move left", () => {
                    if (index <= 0) return;
                    moveReferenceAndLinkedRows(index, index - 1);
                }),
                makeMini(">", "Move right", () => {
                    const next = getConnectedReferencePaths(node);
                    if (index >= next.length - 1) return;
                    moveReferenceAndLinkedRows(index, index + 1);
                })
            ];
            if (!shotboardV2) {
                miniButtons.push(makeMini("R", "Replace this reference image", () => {
                    openReplaceReferencePicker(index);
                }));
            }
            miniButtons.push(
                makeMini("x", "Remove reference", () => {
                    removeReferenceAndLinkedRows(index);
                })
            );
            controls.append(...miniButtons);
            card.append(img, badge, controls);
            referencesGrid.appendChild(card);
        });
    }

    addImagesBtn.onclick = () => fileInput.click();
    fileInput.onchange = async (event) => {
        const uploaded = await uploadShotboardImages(event.target.files);
        if (uploaded.length) {
            const current = getOwnReferencePaths(node);
            const next = current.concat(uploaded);
            const resetToOneToOne = !current.length && (rows.length === 0 || isFactoryDefaultRows());
            setOwnReferencePaths(node, next);
            syncRowsToReferencePaths(next, { forceOneToOne: resetToOneToOne });
            drawReferenceStrip();
            draw();
        }
        fileInput.value = "";
    };
    replaceFileInput.onchange = async (event) => {
        const replaceIndex = pendingReplaceReferenceIndex;
        pendingReplaceReferenceIndex = null;
        const file = event.target.files?.[0];
        if (file && Number.isFinite(Number(replaceIndex))) {
            const uploaded = await uploadShotboardImages([file]);
            if (uploaded[0]) {
                replaceReferencePathAt(node, Number(replaceIndex), uploaded[0]);
                drawReferenceStrip();
                draw();
            }
        }
        replaceFileInput.value = "";
    };
    importBoardBtn.onclick = () => boardFileInput.click();
    importRefsBtn.onclick = () => {
        const boardPaths = getBoardReferencePaths(node);
        if (boardPaths.length) {
            setOwnReferencePaths(node, boardPaths);
            syncRowsToReferencePaths(boardPaths, { forceOneToOne: rows.length === 0 || isFactoryDefaultRows() });
            drawReferenceStrip();
            draw();
        }
    };
    const collectBoard = () => {
        syncProPromptWidget();
        sync();
        const imagePaths = getConnectedReferencePaths(node);
        const settings = {};
        for (const name of settingNames) {
            settings[name] = getWidget(node, name)?.value ?? null;
        }
        return {
            metadata: {
                schema: "iamccs.cine.shotboard.board",
                schema_version: 1,
                cine_ui_version: CINE_VERSION,
                saved_at: new Date().toISOString(),
                node_type: nodeClassName(node),
                image_storage: "paths_or_comfy_input_names",
            },
            global_prompt: String(promptArea.value || getWidget(node, "global_prompt")?.value || ""),
            prompt: String(promptArea.value || getWidget(node, "global_prompt")?.value || ""),
            timeline_data: String(getWidget(node, "timeline_data")?.value || ""),
            rows: rows.map(normalizeShotboardRow),
            settings,
            duration_seconds: settings.duration_seconds,
            frame_rate: settings.frame_rate,
            guide_policy: settings.guide_policy,
            min_guide_gap_seconds: settings.min_guide_gap_seconds,
            max_guides: settings.max_guides,
            default_force: settings.default_force,
            promptrelay_epsilon: settings.promptrelay_epsilon,
            ltx_round_mode: settings.ltx_round_mode,
            image_width: settings.image_width,
            image_height: settings.image_height,
            image_paths: imagePaths,
            images: imagePaths.map((path, index) => ({
                ref: index + 1,
                path,
                name: String(path).split(/[\\/]/).pop() || `ref_${index + 1}`,
            })),
        };
    };
    const refreshBoardControls = () => {
        promptArea.value = String(getWidget(node, "global_prompt")?.value || "");
        for (const [name, control] of settingControls.entries()) {
            const value = getWidget(node, name)?.value;
            if (typeof control?._iamccsSetValue === "function") {
                control._iamccsSetValue(value);
            } else if (control && "value" in control) {
                control.value = String(value ?? "");
            }
        }
    };
    const applyBoard = (data) => {
        const workflowBoard = boardFromWorkflowJson(data);
        const nestedBoard = data?.board && typeof data.board === "object" ? data.board : null;
        const board = workflowBoard || nestedBoard || data || {};
        const metadata = board.metadata || data?.metadata || {};
        const settings = { ...(board.settings || {}) };
        for (const name of settingNames) {
            if (!Object.prototype.hasOwnProperty.call(settings, name) && Object.prototype.hasOwnProperty.call(board, name)) {
                settings[name] = board[name];
            }
        }
        const importedPrompt = board.global_prompt ?? board.prompt ?? board.globalPrompt;
        if (typeof importedPrompt === "string") {
            promptArea.value = importedPrompt;
            syncProPromptWidget();
        }
        for (const name of settingNames) {
            if (Object.prototype.hasOwnProperty.call(settings, name)) {
                setWidgetValue(node, name, settings[name]);
            }
        }
        const refs = Array.isArray(board.image_paths)
            ? board.image_paths
            : Array.isArray(board.images)
                ? board.images.map((item) => item?.path || item?.filename || item?.name).filter(Boolean)
                : [];
        if (refs.length) setOwnReferencePaths(node, refs);

        const rowCandidates = [];
        if (Array.isArray(board.rows)) rowCandidates.push({ source: "board.rows", rows: board.rows });
        const parsedBoardTimeline = parseShotboardTimelineString(board.timeline_data);
        if (parsedBoardTimeline.length) rowCandidates.push({ source: "board.timeline_data", rows: parsedBoardTimeline });
        const parsedRootTimeline = parseShotboardTimelineString(data?.timeline_data);
        if (parsedRootTimeline.length) rowCandidates.push({ source: "root.timeline_data", rows: parsedRootTimeline });
        const workflowRows = workflowBoard ? parseShotboardTimelineString(workflowBoard.timeline_data) : [];
        if (workflowRows.length) rowCandidates.push({ source: "workflow.shotplanner.timeline_data", rows: workflowRows });
        let loadedRows = [];
        let loadedRowsSource = "";
        if (rowCandidates.length) {
            const ranked = rowCandidates.map((candidate) => {
                return {
                    ...candidate,
                    noteFallback: false,
                    promptCount: relayPromptCountInRows(candidate.rows),
                };
            }).sort((a, b) => (b.promptCount - a.promptCount) || (b.rows.length - a.rows.length));
            loadedRows = ranked[0].rows;
            loadedRowsSource = ranked[0].source;
        }
        if (loadedRows.length) {
            const hasCanonicalRelay = loadedRows.some(rowHasCanonicalRelayPrompt);
            const schemaVersion = Number(metadata.schema_version || board.schema_version || 0);
            const useNoteAsRelayFallback = false;
            rows = loadedRows.map((row, index) => normalizeShotboardRow(row, index, { useNoteAsRelayFallback }));
            writeShotboard(node, rows);
            const localCount = rows.filter((row) => String(row.relay_prompt || "").trim()).length;
            console.info("[IAMCCS Cine Shotboard] board imported", {
                source: loadedRowsSource,
                rows: rows.length,
                local_prompts: localCount,
                canonical_relay_fields_found: hasCanonicalRelay,
                legacy_note_fallback: false,
                schema_version: schemaVersion,
            });
            if (warnText) {
                warnText.textContent = `Imported board: ${rows.length} rows, ${localCount} Local prompts from ${loadedRowsSource}${useNoteAsRelayFallback ? " (legacy notes migrated)" : ""}.`;
            }
        } else if (refs.length) {
            syncRowsToReferencePaths(refs, { forceOneToOne: rows.length === 0 || isFactoryDefaultRows() });
        }
        refreshBoardControls();
        drawReferenceStrip();
        draw();
    };
    saveBoardBtn.onclick = () => {
        const board = collectBoard();
        const firstLabel = rows[0]?.label || "cine_shotboard";
        downloadJsonFile(board, safeBoardFilename(firstLabel));
    };
    boardFileInput.onchange = async (event) => {
        const file = event.target.files?.[0];
        if (!file) return;
        try {
            const data = await readJsonFile(file);
            applyBoard(data);
        } catch (err) {
            console.error("[IAMCCS Cine Shotboard] board load failed", err);
            warnText.textContent = `Board load failed: ${err?.message || err}`;
        } finally {
            boardFileInput.value = "";
        }
    };
    clearImagesBtn.onclick = () => {
        setOwnReferencePaths(node, []);
        if (!rows.length) rows = [makeReferenceRow(1, 1)];
        sync();
        drawReferenceStrip();
        draw();
    };

    const addBtn = shotButton("Add Row", "primary");
    const presetSafe = shotButton("Continuous Template");
    presetSafe.title = "Create a generic continuous FLF board with a few soft image anchors and prompt-only motion rows.";
    const promptOnly = shotButton("Prompt-Only Mode");
    promptOnly.title = "Keep only the first row as an image guide and turn the other rows into PromptRelay/text beats.";
    const smoothBtn = shotButton("Smooth Forces");
    smoothBtn.title = "Lower strong guide forces and replace hard_cut with match_cut so the board is safer for one continuous generation.";
    const coreBtn = shotButton("Core Guides");
    coreBtn.title = "Keep only opening, middle and final rows as active FLF guides; other rows become inactive guides.";
    const thumbsBtn = shotButton("Refresh Thumbs");
    thumbsBtn.title = "Refresh image thumbnails after changing or importing reference paths.";
    const bakeRelayBtn = shotButton("Apply Controls to Prompts");
    bakeRelayBtn.title = "Write Camera, Transition and Add-on controls into every Local prompt box, then turn those controls off.";
    const openEditorBtn = shotButton("Open Editor", "primary");
    const dialogueBtn = shotButton("Shot/Reverse Template");
    dialogueBtn.title = "Create a generic shot/reverse-shot board. Use multigen for true editorial cuts.";
    const clearBtn = shotButton("Clear", "danger");
    toolbar.append(addBtn, presetSafe, promptOnly, smoothBtn, coreBtn, thumbsBtn, bakeRelayBtn, openEditorBtn, dialogueBtn, clearBtn);

    const warnBox = document.createElement("div");
    warnBox.style.cssText = `margin-bottom:7px;padding:7px 9px;border:1px solid ${CINE_FILM_LAB.border};background:${CINE_FILM_LAB.panelDark};color:${CINE_FILM_LAB.text};font-size:11px;line-height:1.35;border-radius:5px;min-height:30px;display:flex;align-items:center;gap:10px;box-sizing:border-box;width:100%;`;
    const warnControls = document.createElement("div");
    warnControls.style.cssText = "display:flex;align-items:center;gap:6px;flex:0 0 auto;";
    const warnText = document.createElement("div");
    warnText.style.cssText = "flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
    const makeBulkToggle = (label, titleText, apply) => {
        const toggle = document.createElement("button");
        toggle.type = "button";
        toggle.title = titleText;
        toggle.style.cssText = `min-width:86px;height:25px;padding:0 9px;border:1px solid ${CINE_FILM_LAB.border};border-radius:5px;background:${CINE_FILM_LAB.button};color:${CINE_FILM_LAB.text};font-size:11px;font-weight:700;cursor:pointer;`;
        toggle.onclick = (event) => {
            event.preventDefault();
            event.stopPropagation();
            apply();
        };
        protectControlDrag(toggle);
        return toggle;
    };
    const guideBulkToggle = makeBulkToggle("Guides", "Toggle the Guide column for every row", () => {
        const normalized = rows.map(normalizeShotboardRow);
        const allOn = normalized.length > 0 && normalized.every((row) => row.use_guide !== false);
        rows = normalized.map((row) => ({ ...row, use_guide: !allOn }));
        draw();
    });
    const relayBulkToggle = makeBulkToggle("Relay", "Toggle the Relay column for every row", () => {
        const normalized = rows.map(normalizeShotboardRow);
        const allOn = normalized.length > 0 && normalized.every((row) => row.use_prompt !== false && String(row.use_prompt).toLowerCase() !== "false");
        rows = normalized.map((row) => ({ ...row, use_prompt: !allOn }));
        draw();
    });
    warnControls.append(guideBulkToggle, relayBulkToggle);
    warnBox.append(warnControls, warnText);
    root.insertBefore(warnBox, table);

    const updateBulkToggle = (button, label, activeCount, total) => {
        const all = total > 0 && activeCount === total;
        const some = activeCount > 0 && activeCount < total;
        const state = all ? "ON" : some ? "MIX" : "OFF";
        button.textContent = `${label}: ${state}`;
        button.style.borderColor = all ? CINE_FILM_LAB.active : some ? CINE_FILM_LAB.guide : CINE_FILM_LAB.border;
        button.style.background = all ? "#243722" : some ? "#3C301B" : CINE_FILM_LAB.button;
        button.style.color = all ? "#E5F3DB" : some ? "#F6D796" : CINE_FILM_LAB.text;
    };

    function sync() {
        rows = rows.map(normalizeShotboardRow).sort((a, b) => a.second - b.second);
        writeShotboard(node, rows);
        const warnings = shotboardWarnings(rows);
        const normalized = rows.map(normalizeShotboardRow);
        const guideCount = normalized.filter((row) => row.use_guide !== false).length;
        const relayCount = normalized.filter((row) => row.use_prompt !== false && String(row.use_prompt).toLowerCase() !== "false").length;
        updateBulkToggle(guideBulkToggle, "Guides", guideCount, normalized.length);
        updateBulkToggle(relayBulkToggle, "Relay", relayCount, normalized.length);
        warnText.textContent = warnings.length ? `Warnings: ${warnings.join("  |  ")}` : `Ready: ${shotboardStatusTip(rows)}`;
        warnText.title = warnText.textContent;
    }

    function draw() {
        table.innerHTML = "";
        const header = document.createElement("div");
        header.style.cssText = `display:grid;grid-template-columns:${SHOTBOARD_ROW_GRID};gap:8px;color:${CINE_FILM_LAB.muted};font-size:11px;font-weight:600;padding:0 6px;box-sizing:border-box;width:100%;max-width:100%;min-width:0;overflow:hidden;`;
        header.innerHTML = "<div></div><div>Time</div><div>Image Ref</div><div>Force / Notes</div><div>Guide</div><div>Relay</div><div>Shot controls</div><div>Local prompt</div><div></div>";
        table.appendChild(header);

        const referencePaths = getConnectedReferencePaths(node);
        const rowCount = Math.max(1, rows.length);
        const expandedRows = rowCount <= 2;
        const rowThumbHeight = rowCount === 1 ? 190 : rowCount === 2 ? 156 : 116;
        const rowThumbWidth = rowCount === 1 ? 238 : rowCount === 2 ? 206 : 168;
        const notesMinHeight = rowCount === 1 ? 132 : rowCount === 2 ? 104 : 68;
        const localPromptMinHeight = rowCount === 1 ? 214 : rowCount === 2 ? 170 : 128;
        rows.forEach((row, index) => {
            const r = normalizeShotboardRow(row, index);
            rows[index] = r;
            const rowId = r._ui_id;
            const updateRow = (patch) => {
                const target = rows.find((item) => item?._ui_id === rowId);
                if (!target) return;
                Object.assign(target, patch);
                sync();
            };
            const card = document.createElement("div");
            card.style.cssText = `
                display:grid;
                grid-template-columns:${SHOTBOARD_ROW_GRID};
                gap:8px;
                align-items:start;
                padding:9px 6px;
                border:1px solid ${CINE_FILM_LAB.border};
                background:${CINE_FILM_LAB.panel};
                border-radius:6px;
                box-shadow: inset 0 1px 0 rgba(255,255,255,.03);
                box-sizing:border-box;
                width:100%;
                max-width:100%;
                min-width:0;
                overflow:hidden;
            `;
            card.draggable = false;
            card.ondragover = (event) => {
                if (!Array.from(event.dataTransfer?.types || []).includes("text/iamccs-row")) return;
                event.preventDefault();
                card.style.borderColor = CINE_FILM_LAB.guide;
            };
            card.ondragleave = () => {
                card.style.borderColor = CINE_FILM_LAB.border;
            };
            card.ondrop = (event) => {
                const raw = event.dataTransfer?.getData("text/iamccs-row");
                if (raw == null || raw === "") return;
                event.preventDefault();
                card.style.borderColor = CINE_FILM_LAB.border;
                const from = Number(raw);
                if (!Number.isFinite(from)) return;
                const referencePathsBeforeMove = getConnectedReferencePaths(node);
                const paired = rowsAreOneToOneWithReferences(referencePathsBeforeMove.length);
                rows = moveRowsKeepingTimelineSlots(rows, from, index, normalizeShotboardRow);
                if (paired && referencePathsBeforeMove.length >= rows.length) {
                    setOwnReferencePaths(node, moveItem(referencePathsBeforeMove, from, index));
                    renumberRowsForReferenceOrder(referencePathsBeforeMove.length, true);
                    drawReferenceStrip();
                } else {
                    sync();
                }
                draw();
            };

            const handle = document.createElement("button");
            handle.textContent = "::";
            handle.title = "Drag shot row";
            handle.draggable = true;
            handle.style.cssText = `
                width:22px;
                height:28px;
                padding:0;
                border:1px solid #405664;
                border-radius:4px;
                background:#0b1116;
                color:#9fb0bd;
                cursor:grab;
                font-size:12px;
            `;
            handle.ondragstart = (event) => {
                event.stopPropagation();
                event.dataTransfer?.setData("text/iamccs-row", String(index));
                event.dataTransfer?.setData("text/plain", `row:${index}`);
                handle.style.cursor = "grabbing";
            };
            handle.ondragend = () => {
                handle.style.cursor = "grab";
            };
            protectDragHandle(handle);
            const sec = timeControl(r.second, (value) => updateRow({ second: value }));

            const ref = refPicker(r.ref, referencePaths, (value) => { updateRow({ ref: value }); draw(); }, {
                thumbWidth: rowThumbWidth,
                thumbHeight: rowThumbHeight,
                onEdit: shotboardV2 ? (referenceIndex, path) => {
                    openReferenceFrameEditor(node, referenceIndex, path, (newPath) => {
                        replaceReferencePathAt(node, referenceIndex, newPath);
                        drawReferenceStrip();
                        draw();
                    });
                } : null,
                onReplace: (referenceIndex) => {
                    openReplaceReferencePicker(referenceIndex);
                },
                onDuplicate: shotboardV2 ? (referenceIndex) => {
                    duplicateReferenceAndLinkedRow(referenceIndex, index);
                } : null,
            });
            const force = forceControl(r.force, (value) => updateRow({ force: value }));
            const guide = checkbox(r.use_guide, (value) => updateRow({ use_guide: value }));
            guide.title = "Use as FLF image guide";
            const relay = checkbox(r.use_prompt, (value) => updateRow({ use_prompt: value }));
            relay.title = "Use this row as PromptRelay local prompt segment";

            const label = document.createElement("input");
            label.value = r.label; label.style.cssText = inputBase();
            label.oninput = () => updateRow({ label: label.value });
            protectControlDrag(label);

            const camera = makeSelect(r.camera, CAMERA_OPTIONS, (value) => { updateRow({ camera: value }); });
            const transition = makeSelect(r.transition, TRANSITION_OPTIONS, (value) => { updateRow({ transition: value }); });

            const forceNotesCell = document.createElement("div");
            forceNotesCell.style.cssText = "display:flex;flex-direction:column;gap:8px;min-width:0;";
            forceNotesCell.appendChild(force);
            const notesBox = document.createElement("textarea");
            notesBox.value = r.note || "";
            notesBox.rows = 3;
            notesBox.placeholder = "Notes";
            notesBox.title = "Private shot notes. Notes are never sent to PromptRelay.";
            notesBox.style.cssText = inputBase() + `resize:vertical;min-height:${notesMinHeight}px;line-height:1.32;font-size:11px;padding:8px 15px 8px 9px;scrollbar-gutter:stable;`;
            notesBox.oninput = () => { updateRow({ note: notesBox.value }); };
            protectControlDrag(notesBox);
            forceNotesCell.appendChild(notesBox);

            const promptCell = document.createElement("div");
            promptCell.style.cssText = "display:flex;flex-direction:column;gap:5px;min-width:0;height:100%;";
            const promptActions = document.createElement("div");
            promptActions.style.cssText = "display:flex;gap:6px;align-items:center;justify-content:flex-end;min-width:0;";
            const applyRowControls = shotButton("Apply Shot Controls");
            applyRowControls.title = "Insert this row's Camera Relay, Trans Relay and custom add-on text into the Local prompt box.";
            applyRowControls.style.minHeight = "24px";
            applyRowControls.style.fontSize = "10px";
            applyRowControls.onclick = (event) => {
                event.preventDefault();
                event.stopPropagation();
                const current = normalizeShotboardRow({ ...r, ...rows[index] }, index);
                const normalizedRows = rows.map(normalizeShotboardRow);
                const nextRow = normalizedRows[index + 1] || null;
                updateRow({
                    relay_prompt: composeRelayPromptPreview(current, nextRow),
                    camera_relay_mode: "off",
                    transition_relay_mode: "off",
                    relay_modifier_text: "",
                });
                draw();
            };
            protectControlDrag(applyRowControls);
            promptActions.appendChild(applyRowControls);
            const localPrompt = document.createElement("textarea");
            localPrompt.value = r.relay_prompt || "";
            localPrompt.rows = 6;
            localPrompt.placeholder = "Local PromptRelay beat";
            localPrompt.title = "Only this text becomes one PromptRelay local prompt segment when Relay is checked. Notes and labels are not sent.";
            localPrompt.style.cssText = inputBase() + `resize:vertical;min-height:${localPromptMinHeight}px;line-height:1.35;font-size:${expandedRows ? "13px" : "12px"};padding:10px 20px 10px 10px;scrollbar-gutter:stable;`;
            localPrompt.oninput = () => { updateRow({ relay_prompt: localPrompt.value }); };
            protectControlDrag(localPrompt);
            promptCell.append(promptActions, localPrompt);

            const controlsCell = document.createElement("div");
            controlsCell.style.cssText = "display:flex;flex-direction:column;gap:6px;min-width:0;";
            const cameraRelay = makeChoiceSelect(r.camera_relay_mode, [
                { value: "off", label: "off" },
                { value: "before", label: "before local" },
                { value: "after", label: "after local" },
            ], (value) => { updateRow({ camera_relay_mode: value }); });
            cameraRelay.title = "Insert camera movement into this PromptRelay local prompt";
            const transitionRelay = makeChoiceSelect(r.transition_relay_mode, [
                { value: "off", label: "off" },
                { value: "safe_only", label: "safe only" },
                { value: "append", label: "append all" },
            ], (value) => { updateRow({ transition_relay_mode: value }); });
            transitionRelay.title = "Insert transition language into this PromptRelay local prompt. safe_only avoids hard_cut.";
            const shotControls = document.createElement("div");
            shotControls.style.cssText = "display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;min-width:0;";
            const namedControl = (labelText, control, relayControl = null) => {
                const wrap = document.createElement("label");
                wrap.style.cssText = "display:flex;flex-direction:column;gap:3px;color:#a9bac5;font-size:10px;min-width:0;";
                const span = document.createElement("span");
                span.textContent = labelText;
                wrap.append(span, control);
                if (relayControl) wrap.append(relayControl);
                return wrap;
            };
            shotControls.append(
                namedControl("Label", label),
                namedControl("Camera", camera, cameraRelay),
                namedControl("Transition", transition, transitionRelay)
            );

            const addonControls = document.createElement("div");
            addonControls.style.cssText = "display:grid;grid-template-columns:minmax(110px,160px) 1fr;gap:6px;align-items:end;";
            const miniControl = (labelText, control) => {
                const wrap = document.createElement("label");
                wrap.style.cssText = "display:flex;flex-direction:column;gap:2px;color:#a9bac5;font-size:10px;min-width:0;";
                const span = document.createElement("span");
                span.textContent = labelText;
                wrap.append(span, control);
                return wrap;
            };
            const addonPosition = makeChoiceSelect(r.relay_addon_position, [
                { value: "after", label: "add-on after" },
                { value: "before", label: "add-on before" },
            ], (value) => { updateRow({ relay_addon_position: value }); });
            addonPosition.title = "Where to place the custom add-on text";
            const modifierBox = document.createElement("textarea");
            modifierBox.value = r.relay_modifier_text || "";
            modifierBox.placeholder = "Custom Relay add-on text, optional";
            modifierBox.rows = 1;
            modifierBox.style.cssText = inputBase() + "resize:vertical;min-height:30px;font-size:11px;padding:7px 15px 7px 9px;";
            modifierBox.oninput = () => { updateRow({ relay_modifier_text: modifierBox.value }); };
            protectControlDrag(modifierBox);
            addonControls.append(miniControl("Add-on pos", addonPosition), modifierBox);
            controlsCell.append(shotControls, addonControls);

            const del = shotButton("x", "danger");
            del.style.width = "26px";
            del.style.height = "30px";
            del.style.padding = "0";
            del.style.justifySelf = "end";
            del.onclick = () => {
                if (rows.length <= 1) {
                    if (warnText) warnText.textContent = "At least one shot row must remain.";
                    return;
                }
                rows.splice(index, 1);
                sync();
                draw();
            };
            protectControlDrag(del);

            card.append(handle, sec, ref, forceNotesCell, guide, relay, controlsCell, promptCell, del);
            table.appendChild(card);
        });
        sync();
    }

    addBtn.onclick = () => { rows.push({ second: rows.length ? Number(rows[rows.length - 1].second) + 2.5 : 0, ref: rows.length + 1, force: 0.18, use_guide: false, use_prompt: true, label: `shot_${rows.length + 1}`, camera: "continuous dolly-in", transition: "continuous_motion", camera_relay_mode: "before", transition_relay_mode: "off", relay_addon_position: "after", note: "", relay_prompt: "describe the motion beat" }); draw(); };
    presetSafe.onclick = () => { rows = defaultShotboardRows(); draw(); };
    promptOnly.onclick = () => { rows = defaultShotboardRows().map((row, i) => ({ ...row, force: i === 0 ? 0.65 : 0, use_guide: i === 0, use_prompt: true })); draw(); };
    smoothBtn.onclick = () => { rows = rows.map((row, i) => ({ ...row, force: i === 0 ? Math.min(row.force, 0.72) : Math.min(row.force, 0.24), transition: row.transition === "hard_cut" ? "match_cut" : row.transition })); draw(); };
    coreBtn.onclick = () => { rows = rows.map((row, i) => ({ ...row, use_guide: i === 0 || i === Math.floor(rows.length / 2) || i === rows.length - 1, use_prompt: i === 0 || i === Math.floor(rows.length / 2) || i === rows.length - 1, force: (i === 0 || i === Math.floor(rows.length / 2) || i === rows.length - 1) ? Math.max(0.18, Math.min(row.force || 0.22, 0.32)) : 0 })); draw(); };
    thumbsBtn.onclick = () => { drawReferenceStrip(); draw(); };
    bakeRelayBtn.onclick = () => {
        const normalized = rows.map(normalizeShotboardRow);
        rows = normalized.map((row, i) => ({
            ...row,
            relay_prompt: composeRelayPromptPreview(row, normalized[i + 1]),
            camera_relay_mode: "off",
            transition_relay_mode: "off",
            relay_modifier_text: "",
        }));
        draw();
    };
    root.addEventListener("iamccs:cine-fullscreen", (event) => {
        openEditorBtn.textContent = event.detail?.open ? "Close Editor" : "Open Editor";
    });
    openEditorBtn.onclick = () => { toggleFullscreenEditor(root, node); };
    dialogueBtn.onclick = () => { rows = [
        { second: 0.0, ref: 1, force: 0.82, use_guide: true, use_prompt: true, label: "field_A", camera: "medium close-up", transition: "hard_cut", note: "", relay_prompt: "person A speaks; use multigen if you want a real cut" },
        { second: 4.0, ref: 2, force: 0.82, use_guide: true, use_prompt: true, label: "reverse_B", camera: "reverse angle", transition: "hard_cut", note: "", relay_prompt: "person B answers; better as second generation" },
        { second: 8.0, ref: 1, force: 0.72, use_guide: true, use_prompt: true, label: "return_A", camera: "slow push-in", transition: "hard_cut", note: "", relay_prompt: "return to person A reaction" },
        { second: 11.0, ref: 3, force: 0.58, use_guide: true, use_prompt: true, label: "wide_reveal", camera: "wide reveal", transition: "match_cut", note: "", relay_prompt: "environment reveals both people" },
    ]; draw(); };
    clearBtn.onclick = () => { rows = [makeReferenceRow(1, Math.max(1, getConnectedReferencePaths(node).length))]; draw(); };

    drawReferenceStrip();
    draw();
    const widget = node.addDOMWidget("Cine Shotboard Pro", "iamccs_cine_shotboard_pro", root, { serialize: false });
    widget.computeSize = (width) => {
        if (root._iamccsFullscreenState) return [width, 24];
        const rowCount = Math.max(1, rows.length);
        const rowHeight = rowCount === 1 ? 246 : rowCount === 2 ? 196 : 156;
        return [width, Math.min(840, Math.max(660, 330 + rowCount * rowHeight))];
    };
}

function renderCineFLFEngineSimple(node) {
    if (node._iamccsCineEngineSimpleReady) return;
    node._iamccsCineEngineSimpleReady = true;
    const chrome = applyCineChrome(node, "flfSimple");

    [
        "num_images",
        "insert_mode",
        "timeline_data",
        "manual_keyframes",
    ].forEach((name) => hideWidget(getWidget(node, name)));
    showWidget(getWidget(node, "images_loaded"));
    showWidget(getWidget(node, "frame_rate"));
    for (let i = 1; i <= 50; i += 1) {
        hideWidget(getWidget(node, `insert_frame_${i}`));
        hideWidget(getWidget(node, `insert_second_${i}`));
        hideWidget(getWidget(node, `strength_${i}`));
    }

    node.size = [Math.max(node.size?.[0] || 420, 460), 150];
    try { node.setDirtyCanvas?.(true, true); app.graph?.setDirtyCanvas?.(true, true); } catch {}

    if (typeof node.addDOMWidget !== "function") return;
    const panel = document.createElement("div");
    panel.style.cssText = [
        "width:100%",
        "box-sizing:border-box",
        "padding:9px 10px",
        `border:1px solid ${chrome.border}`,
        "border-radius:6px",
        `background:${CINE_FILM_LAB.panelDark}`,
        `color:${CINE_FILM_LAB.text}`,
        `box-shadow:inset 0 1px 0 ${chrome.glow}`,
        "font:11px Arial,sans-serif",
        "line-height:1.35",
        "pointer-events:auto",
    ].join(";");
    const title = document.createElement("div");
    title.textContent = "Technical FLF engine";
    title.style.cssText = `font-weight:700;color:${CINE_FILM_LAB.text};`;
    panel.append(title);
    const widget = node.addDOMWidget("Cine FLF Engine Simple", "iamccs_cine_flf_engine_simple", panel, { serialize: false });
    widget.computeSize = (width) => [width, 36];
}

function renderPromptRelayShapeSync(node) {
    if (node._iamccsCineShapeSyncReady) return;
    node._iamccsCineShapeSyncReady = true;
    applyCineChrome(node, "shapeSync");
    node.size = [Math.max(node.size?.[0] || 360, 380), Math.max(node.size?.[1] || 160, 170)];
    try { node.setDirtyCanvas?.(true, true); app.graph?.setDirtyCanvas?.(true, true); } catch {}
}

function renderCineInfo(node) {
    if (node._iamccsCineInfoReady) return;
    node._iamccsCineInfoReady = true;
    node.size = [Math.max(node.size?.[0] || 380, 420), Math.max(node.size?.[1] || 320, 340)];
    applyCineChrome(node, "info");
    try { node.setDirtyCanvas?.(true, true); app.graph?.setDirtyCanvas?.(true, true); } catch {}
}

function renderCineMusicVideoPlanner(node) {
    if (node._iamccsCineMusicVideoReady) return;
    node._iamccsCineMusicVideoReady = true;
    node.size = [Math.max(node.size?.[0] || 420, 520), Math.max(node.size?.[1] || 260, 300)];
    node.color = CINE_FILM_LAB.header;
    node.bgcolor = CINE_FILM_LAB.nodeBg;
    node.boxcolor = CINE_FILM_LAB.relay;
    try { node.setDirtyCanvas?.(true, true); app.graph?.setDirtyCanvas?.(true, true); } catch {}
    if (typeof node.addDOMWidget !== "function") return;

    const panel = document.createElement("div");
    panel.style.cssText = [
        "width:100%",
        "box-sizing:border-box",
        "padding:9px 10px",
        `border:1px solid ${CINE_FILM_LAB.border}`,
        "border-radius:6px",
        `background:${CINE_FILM_LAB.panelDark}`,
        `color:${CINE_FILM_LAB.text}`,
        "font:11px Arial,sans-serif",
        "line-height:1.35",
        "pointer-events:auto",
    ].join(";");
    const title = document.createElement("div");
    title.textContent = "Cine Videoclip Maker";
    title.style.cssText = `font-weight:700;color:${CINE_FILM_LAB.text};margin-bottom:4px;font-size:13px;`;
    const body = document.createElement("div");
    body.textContent = "Audio/shot sequencer planner. It creates one music-video shot: image prompt for Z-Image/Flux, PromptRelay local beats, LTX frame counts and music_linx for the future CINE_VIDEOCLIP_1 backend.";
    body.style.cssText = `color:${CINE_FILM_LAB.muted};white-space:normal;margin-bottom:7px;`;
    const flow = document.createElement("div");
    flow.textContent = "Use: shot_index -> image generator -> LTX I2V+A. For a full videoclip, iterate shot_index and concatenate rendered clips.";
    flow.style.cssText = `color:${CINE_FILM_LAB.relay};white-space:normal;`;
    panel.append(title, body, flow);
    const widget = node.addDOMWidget("Videoclip Maker", "iamccs_cine_music_video", panel, { serialize: false });
    widget.computeSize = (width) => [width, 92];
}

function showRenderError(node, err) {
    console.error("[IAMCCS Cine UI]", err);
    if (node._iamccsCineUiErrorShown || typeof node.addDOMWidget !== "function") return;
    node._iamccsCineUiErrorShown = true;
    const box = document.createElement("div");
    box.style.cssText = `
        width: 100%; box-sizing: border-box; padding: 10px;
        border: 1px solid ${CINE_FILM_LAB.danger}; border-radius: 6px;
        background: ${CINE_FILM_LAB.dangerDark}; color: #FFE4DD; font: 11px Arial, sans-serif;
        white-space: normal; pointer-events: auto;
    `;
    box.textContent = `Cine UI failed to render: ${err?.message || err}`;
    const widget = node.addDOMWidget("Cine UI status", "iamccs_cine_ui_error", box, { serialize: false });
    widget.computeSize = (width) => [width, 64];
}

function renderForNode(node) {
    const klass = nodeClassName(node);
    try {
        const title = String(node?.title || "");
        if (klass === "IAMCCS_CineShotboardLite") {
            lockNodeMinimumSize(node, SHOTBOARD_LITE_NODE_MIN_SIZE, { lockResize: true });
        }
        if (klass === "IAMCCS_CineShotboardTimelinePro" || klass === "IAMCCS_CineShotboardPlannerPro" || klass === "IAMCCS_CineShotboardPlannerProV2" || klass === "IAMCCS_CineShotboardPlannerProLegacy") {
            lockNodeMinimumSize(node, SHOTBOARD_NODE_MIN_SIZE, { lockResize: true });
        }
        if (klass === "IAMCCS_CineLTXSequencer") applyCineChrome(node, "flfEngine");
        if (klass === "IAMCCS_CinePromptRelayLatentShapeSync") renderPromptRelayShapeSync(node);
        if (klass === "IAMCCS_CineLTXSequencer") renderKeyframeEditor(node);
        if (klass === "IAMCCS_CinePromptRelayTimeline") renderPromptRelayEditor(node);
        if (klass === "IAMCCS_CineShotboardLite") renderShotboardLite(node);
        if (klass === "IAMCCS_CineShotboardTimelinePro" || klass === "IAMCCS_CineShotboardPlannerPro" || klass === "IAMCCS_CineShotboardPlannerProV2" || klass === "IAMCCS_CineShotboardPlannerProLegacy") renderShotboardPro(node);
        if (klass === "IAMCCS_CineFLFEngineSimple") renderCineFLFEngineSimple(node);
        if (klass === "IAMCCS_CineInfo") renderCineInfo(node);
        if (klass === "IAMCCS_CineMusicVideoPlanner") renderCineMusicVideoPlanner(node);
    } catch (err) {
        showRenderError(node, err);
    }
}

function scheduleRender(node) {
    const klass = nodeClassName(node);
    if (klass !== "IAMCCS_CineLTXSequencer" && klass !== "IAMCCS_CinePromptRelayLatentShapeSync" && klass !== "IAMCCS_CinePromptRelayTimeline" && klass !== "IAMCCS_CineShotboardLite" && klass !== "IAMCCS_CineShotboardTimelinePro" && klass !== "IAMCCS_CineShotboardPlannerPro" && klass !== "IAMCCS_CineShotboardPlannerProV2" && klass !== "IAMCCS_CineShotboardPlannerProLegacy" && klass !== "IAMCCS_CineFLFEngineSimple" && klass !== "IAMCCS_CineInfo" && klass !== "IAMCCS_CineMusicVideoPlanner") return;
    setTimeout(() => renderForNode(node), 80);
    setTimeout(() => renderForNode(node), 450);
}

app.registerExtension({
    name: "iamccs.cine.timeline.ui",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const name = String(nodeData?.name || nodeData?.class_type || "");
        if (name !== "IAMCCS_CineLTXSequencer" && name !== "IAMCCS_CinePromptRelayLatentShapeSync" && name !== "IAMCCS_CinePromptRelayTimeline" && name !== "IAMCCS_CineShotboardLite" && name !== "IAMCCS_CineShotboardTimelinePro" && name !== "IAMCCS_CineShotboardPlannerPro" && name !== "IAMCCS_CineShotboardPlannerProV2" && name !== "IAMCCS_CineShotboardPlannerProLegacy" && name !== "IAMCCS_CineFLFEngineSimple" && name !== "IAMCCS_CineInfo" && name !== "IAMCCS_CineMusicVideoPlanner") return;
        if (nodeType.prototype._iamccsCineTimelineWrapped) return;
        nodeType.prototype._iamccsCineTimelineWrapped = true;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = onNodeCreated?.apply(this, args);
            scheduleRender(this);
            return result;
        };
    },
    async nodeCreated(node) {
        scheduleRender(node);
    },
});

// â”€â”€ NarrativePlanner push listener â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// When the NarrativePlanner "â†’ Push to PlannerPro" button fires, re-render
// the shotboard so the new rows appear immediately without a page reload.
// By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
document.addEventListener("iamccs:planner_rows_updated", (ev) => {
    const nodeId = ev?.detail?.node_id;
    if (!nodeId) return;
    const plannerNode = app.graph?.getNodeById(nodeId);
    if (!plannerNode) return;
    const klass = String(plannerNode?.comfyClass || plannerNode?.type || "");
    if (klass !== "IAMCCS_CineShotboardLite" && klass !== "IAMCCS_CineShotboardPlannerPro" && klass !== "IAMCCS_CineShotboardPlannerProV2" && klass !== "IAMCCS_CineShotboardPlannerProLegacy" && klass !== "IAMCCS_CineShotboardTimelinePro") return;
    // Clear render guard so renderShotboardPro rebuilds the table
    plannerNode._iamccsCineShotboardReady = false;
    plannerNode._iamccsCineShotboardLiteReady = false;
    scheduleRender(plannerNode);
});

