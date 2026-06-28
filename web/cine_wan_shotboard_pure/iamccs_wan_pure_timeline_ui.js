import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.info("[IAMCCS WAN V3] Isolated WAN Shotboard UI mode active.");
const WAN_NODE_CLASS = "IAMCCS_WanShotboardPlannerPure";
const CINE_VERSION = "2026-06-08-wan-v3-isolated-v2-palette-meter-scrub-reliable-plus";
const SHOTBOARD_V3_RIGID_WIDTH = 1920;
const SHOTBOARD_V3_OPEN_HEIGHT = 900;
const SHOTBOARD_V3_COLLAPSED_HEIGHT = 660; // increased to accommodate global prompt always visible in collapsed mode
const SHOTBOARD_NODE_MIN_SIZE = [1500, 760];
const SHOTBOARD_NODE_DEFAULT_SIZE = [1500, 780];
const SHOTBOARD_ROW_GRID = "24px 108px 164px 188px 42px 42px minmax(430px,500px) minmax(280px,1fr) 28px";
const SHOTBOARD_ROW_GRID_V2 = "24px 108px 164px 188px 42px 42px minmax(520px,640px) minmax(260px,1fr) 28px";
const SHOTBOARD_LITE_NODE_MIN_SIZE = [1120, 610];
const SHOTBOARD_LITE_NODE_DEFAULT_SIZE = [1180, 660];
const SHOTBOARD_LITE_ROW_GRID = "24px 108px minmax(152px,180px) minmax(138px,170px) 44px minmax(130px,170px) minmax(280px,1fr) 28px";
const FLF_SEQUENCER_NODE_MIN_SIZE = [760, 980];
const FLF_SEQUENCER_TOP_CLEARANCE = 32;
const FLF_SEQUENCER_ROW_HEIGHT = 68;
const FLF_SEQUENCER_BASE_HEIGHT = 168;
const CINE_FULLSCREEN_Z_INDEX = 2147483000;
const CINE_REF_EDITOR_Z_INDEX = 2147483647;
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
    shotboardV3: {
        header: "#3A342D",
        nodeBg: "#1E2022",
        box: "#D89B45",
        border: "#6B6258",
        glow: "rgba(216,155,69,.20)",
    },
    wanShotboardV3: {
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

function isShotboardV3Class(klass) {
    return klass === WAN_NODE_CLASS;
}

function isWanShotboardV3Class(klass) {
    return klass === WAN_NODE_CLASS;
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
    syncWidgetSerializedValue(node, widget, value);
    try { widget.callback?.(value, app.canvas, node); } catch {}
    try { node.setDirtyCanvas?.(true, true); app.graph?.setDirtyCanvas?.(true, true); } catch {}
    try { node.graph?.change?.(); app.graph?.change?.(); } catch {}
    return true;
}

function syncWidgetSerializedValue(node, widget, value) {
    if (!node || !widget) return false;
    const index = Array.isArray(node.widgets) ? node.widgets.indexOf(widget) : -1;
    if (index < 0) return false;
    if (!Array.isArray(node.widgets_values)) node.widgets_values = [];
    node.widgets_values[index] = value;
    return true;
}

function syncSerializedWidgetValue(node, serialized, name, value) {
    const widget = getWidget(node, name);
    if (!widget || !serialized) return false;
    const index = Array.isArray(node.widgets) ? node.widgets.indexOf(widget) : -1;
    if (index < 0) return false;
    if (!Array.isArray(serialized.widgets_values)) serialized.widgets_values = [];
    serialized.widgets_values[index] = value;
    widget.value = value;
    syncWidgetSerializedValue(node, widget, value);
    return true;
}

function lockNodeMinimumSize(node, minSize, options = {}) {
    if (!node || !Array.isArray(minSize)) return;
    const minWidth = Number(minSize[0]) || 0;
    const minHeight = Number(minSize[1]) || 0;
    node._iamccsCineMinSize = [minWidth, minHeight];
    node._iamccsCineMinOptions = { ...options };
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
        const nextWidth = options.lockWidth
            ? Math.max(minWidth, preferredWidth || minWidth)
            : Math.max(minWidth, preferredWidth || currentWidth);
        const next = preferred
            ? [nextWidth, Math.max(minHeight, preferredHeight)]
            : [nextWidth, Math.max(minHeight, currentHeight)];
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
        const activeOptions = this._iamccsCineMinOptions || options || {};
        if (Array.isArray(size)) {
            size[0] = activeOptions.lockWidth ? Number(min?.[0] || 0) : Math.max(Number(min?.[0] || 0), Number(size[0] || 0));
            size[1] = Math.max(Number(min?.[1] || 0), Number(size[1] || 0));
        }
        const result = originalOnResize ? originalOnResize.apply(this, arguments) : undefined;
        const width = activeOptions.lockWidth ? Number(min?.[0] || 0) : Math.max(Number(min?.[0] || 0), Number(this.size?.[0] || 0));
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

function getLinkedInputSourceInfo(node, inputName) {
    const input = node?.inputs?.find((item) => item?.name === inputName);
    const link = input?.link && app?.graph?.links ? app.graph.links[input.link] : null;
    if (!link) return null;
    const originId = link.origin_id ?? link.source_id;
    const origin = originId != null ? app.graph?.getNodeById(originId) : null;
    const originSlot = Number(link.origin_slot ?? link.source_slot ?? -1);
    const output = origin?.outputs?.[originSlot] || null;
    return {
        node: origin,
        nodeClass: nodeClassName(origin),
        outputName: String(output?.name || output?.label || ""),
        outputType: String(output?.type || ""),
        originSlot,
    };
}

function findOutputSlotByName(node, names) {
    const wanted = new Set((Array.isArray(names) ? names : [names]).map((item) => String(item || "").toLowerCase()));
    return (node?.outputs || []).findIndex((output) => {
        const label = String(output?.name || output?.label || "").toLowerCase();
        return wanted.has(label);
    });
}

function findInputSlotByName(node, names) {
    const wanted = new Set((Array.isArray(names) ? names : [names]).map((item) => String(item || "").toLowerCase()));
    return (node?.inputs || []).findIndex((input) => wanted.has(String(input?.name || input?.label || "").toLowerCase()));
}

function reconnectInputToOutput(sourceNode, outputSlot, targetNode, inputSlot) {
    if (!sourceNode || !targetNode || outputSlot < 0 || inputSlot < 0) return false;
    try {
        targetNode.disconnectInput?.(inputSlot);
    } catch {}
    try {
        sourceNode.connect(outputSlot, targetNode, inputSlot);
        try { targetNode.setDirtyCanvas?.(true, true); app.graph?.setDirtyCanvas?.(true, true); } catch {}
        return true;
    } catch (err) {
        console.warn("[IAMCCS Cine UI] could not reconnect image batch output", err);
        return false;
    }
}

function healCineInfoImageBatchLink(node, state) {
    const source = state?.multiInputSource;
    if (!source?.node || source.nodeClass !== "IAMCCS_CineInfo") return false;
    if (Number(state?.refs || 0) <= 1) return false;
    const outputName = String(source.outputName || "").replace(/\s+/g, "_").toLowerCase();
    if (outputName !== "image_1" && outputName !== "image1") return false;
    const multiOutputSlot = findOutputSlotByName(source.node, ["multi_output"]);
    const multiInputSlot = findInputSlotByName(node, ["multi_input"]);
    return reconnectInputToOutput(source.node, multiOutputSlot, node, multiInputSlot);
}

function traceLinkedCinePlanner(node) {
    const startInputs = ["timeline_data", "multi_input", "duration_seconds"];
    const accepted = new Set([WAN_NODE_CLASS]);
    for (const inputName of startInputs) {
        let current = getLinkedOriginNode(node, inputName);
        const visited = new Set();
        for (let depth = 0; current && depth < 8; depth += 1) {
            if (visited.has(current.id)) break;
            visited.add(current.id);
            const cls = nodeClassName(current);
            if (accepted.has(cls)) return current;
            if (cls === "IAMCCS_CineInfo" || cls === "IAMCCS_CinePromptModeSwitch" || cls === "IAMCCS_CinePromptRelayImageBridge") {
                current = getLinkedOriginNode(current, "wan_shotboard");
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
    const multiInputSource = getLinkedInputSourceInfo(node, "multi_input");
    return {
        planner,
        rows: rowsForFLF,
        duration: Number.isFinite(duration) && duration > 0 ? duration : null,
        fps: Number.isFinite(fps) && fps > 0 ? fps : null,
        refs,
        multiInputSource,
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

function styleValueControls(element) {
    if (element?.matches?.("input,select")) {
        element.style.background = CINE_FILM_LAB.valueBg;
        element.style.color = CINE_FILM_LAB.valueText;
        element.style.borderColor = CINE_FILM_LAB.border;
    }
    element?.querySelectorAll?.("input,select").forEach((child) => {
        child.style.background = CINE_FILM_LAB.valueBg;
        child.style.color = CINE_FILM_LAB.valueText;
        child.style.borderColor = CINE_FILM_LAB.border;
    });
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
            if ((eventName === "dragover" || eventName === "drop") && hasFileDrag(event)) return;
            if (isNumericStepDragTarget(event) && /^(pointer|mouse|touch)/.test(eventName)) return;
            event.stopPropagation();
        }, { passive: false, capture: true });
    }
    element.addEventListener("dragstart", (event) => {
        event.preventDefault();
        event.stopPropagation();
    }, { capture: true });
    return element;
}

function bindReliableTimelineButton(button, action) {
    if (!button || typeof action !== "function") return button;
    let armed = false;
    let lastFire = 0;
    const fire = (event) => {
        event?.stopPropagation?.();
        const now = performance.now();
        if (now - lastFire < 260) return;
        lastFire = now;
        action(event);
    };
    button.draggable = false;
    button.setAttribute?.("draggable", "false");
    button.addEventListener("pointerdown", (event) => {
        armed = true;
        event.stopPropagation();
        try { button.setPointerCapture?.(event.pointerId); } catch {}
        button._iamccsReliableFilter = button.style.filter;
        button._iamccsReliableBoxShadow = button.style.boxShadow;
        button.style.filter = "brightness(1.24) saturate(1.2)";
        button.style.boxShadow = `${button._iamccsReliableBoxShadow || "0 4px 14px rgba(0,0,0,.35)"}, 0 0 0 3px rgba(244,239,230,.32)`;
        button.classList.add("iamccs-pressing");
    }, { capture: true });
    button.addEventListener("pointerup", (event) => {
        event.stopPropagation();
        try { button.releasePointerCapture?.(event.pointerId); } catch {}
        button.style.filter = button._iamccsReliableFilter || "";
        button.style.boxShadow = button._iamccsReliableBoxShadow || "";
        button.classList.remove("iamccs-pressing");
        if (!armed) return;
        armed = false;
        fire(event);
    }, { capture: true });
    button.addEventListener("pointercancel", (event) => {
        event.stopPropagation();
        armed = false;
        button.style.filter = button._iamccsReliableFilter || "";
        button.style.boxShadow = button._iamccsReliableBoxShadow || "";
        button.classList.remove("iamccs-pressing");
    }, { capture: true });
    button.addEventListener("click", (event) => {
        event.preventDefault();
        fire(event);
    }, { capture: true });
    button.addEventListener("dblclick", (event) => {
        event.preventDefault();
        fire(event);
    }, { capture: true });
    return button;
}

function addPressPreview(button, options = {}) {
    if (!button || button._iamccsPressPreview) return button;
    button._iamccsPressPreview = true;
    const pressedBg = options.pressedBg || "linear-gradient(180deg,#FFE08A 0%,#E08B3E 100%)";
    const pressedColor = options.pressedColor || "#120D08";
    const pressedBorder = options.pressedBorder || "#FFE08A";
    const pressedShadow = options.pressedShadow || "inset 0 3px 8px rgba(7,12,17,.75),0 0 0 2px rgba(255,224,138,.42),0 0 16px rgba(255,162,84,.28)";
    const press = () => {
        button.dataset.iamccsPressedPreview = "true";
        button.style.transform = "";
        button.style.background = pressedBg;
        button.style.color = pressedColor;
        button.style.borderColor = pressedBorder;
        button.style.boxShadow = pressedShadow;
        button.style.filter = "brightness(1.18) saturate(1.16)";
        window.setTimeout(() => {
            if (button.dataset.iamccsPressedPreview === "true") release();
        }, Number(options.duration || 280));
    };
    const release = () => {
        button.dataset.iamccsPressedPreview = "false";
        button.style.transform = "";
        if (button._iamccsPressBg) button.style.background = button._iamccsPressBg;
        if (button._iamccsPressColor) button.style.color = button._iamccsPressColor;
        if (button._iamccsPressBorder) button.style.borderColor = button._iamccsPressBorder;
        button.style.boxShadow = button._iamccsPressShadow || "";
        button.style.filter = button._iamccsPressFilter || "";
    };
    button.addEventListener("pointerdown", () => {
        button._iamccsPressBg = button.style.background;
        button._iamccsPressColor = button.style.color;
        button._iamccsPressBorder = button.style.borderColor;
        button._iamccsPressShadow = button.style.boxShadow;
        button._iamccsPressFilter = button.style.filter;
        press();
    }, { capture: true });
    button.addEventListener("click", () => {
        button._iamccsPressBg = button._iamccsPressBg || button.style.background;
        button._iamccsPressColor = button._iamccsPressColor || button.style.color;
        button._iamccsPressBorder = button._iamccsPressBorder || button.style.borderColor;
        button._iamccsPressShadow = button._iamccsPressShadow || button.style.boxShadow;
        button._iamccsPressFilter = button._iamccsPressFilter || button.style.filter;
        press();
    }, { capture: true });
    button.addEventListener("pointerup", release, { capture: true });
    button.addEventListener("pointercancel", release, { capture: true });
    button.addEventListener("mouseleave", () => {
        if (button.dataset.iamccsPressedPreview === "true") release();
    });
    return button;
}

function installRootPressFeedback(root, palette) {
    if (!root || root._iamccsRootPressFeedback) return;
    root._iamccsRootPressFeedback = true;
    const pressTarget = (event) => {
        const button = event.target?.closest?.("button");
        if (!button || !root.contains(button)) return;
        if (button.dataset.iamccsRootPressed !== "true") {
            button._iamccsRootPressTransform = button.style.transform;
            button._iamccsRootPressFilter = button.style.filter;
            button._iamccsRootPressBorder = button.style.borderColor;
        button._iamccsRootPressShadow = button.style.boxShadow;
        }
        button.dataset.iamccsRootPressed = "true";
        button.style.transform = button._iamccsRootPressTransform || "";
        button.style.filter = "brightness(1.22) saturate(1.22)";
        button.style.borderColor = palette?.accent || "#FFE08A";
        button.style.boxShadow = "inset 0 0 0 2px rgba(255,255,255,.32),0 0 0 3px rgba(255,224,138,.45),0 0 18px rgba(111,182,210,.34)";
        window.setTimeout(() => {
            if (button.dataset.iamccsRootPressed === "true") {
                button.dataset.iamccsRootPressed = "false";
                button.style.transform = button._iamccsRootPressTransform || "";
                button.style.filter = button._iamccsRootPressFilter || "";
                button.style.borderColor = button._iamccsRootPressBorder || button.style.borderColor;
                button.style.boxShadow = button._iamccsRootPressShadow || button.style.boxShadow;
            }
        }, 300);
    };
    root.addEventListener("pointerdown", pressTarget, { capture: true });
    root.addEventListener("click", pressTarget, { capture: true });
}

function isNumericStepDragTarget(event) {
    return Boolean(event?.target?.closest?.("[data-iamccs-step-drag='true']"));
}

function hasJsonBoardDrag(event) {
    const transfer = event?.dataTransfer;
    if (!transfer) return false;
    const files = Array.from(transfer.files || []);
    if (files.some((file) => /\.json$/i.test(String(file?.name || "")) || String(file?.type || "").includes("json"))) return true;
    return Array.from(transfer.items || []).some((item) => {
        const name = String(item?.name || "");
        const type = String(item?.type || "");
        return item?.kind === "file" && (/\.json$/i.test(name) || type.includes("json"));
    });
}

function hasFileDrag(event) {
    const transfer = event?.dataTransfer;
    if (!transfer) return false;
    if (Array.from(transfer.files || []).length) return true;
    if (Array.from(transfer.items || []).some((item) => item?.kind === "file")) return true;
    return Array.from(transfer.types || []).includes("Files");
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
        contain: layout paint style;
        content-visibility: auto;
        contain-intrinsic-size: 1500px 900px;
        transform: translateZ(0);
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
        { second: 0.0, ref: 1, strength: 0.78, label: "opening_anchor", camera: "first reference starts alive with subtle natural motion" },
        { second: 2.2, ref: 2, strength: 0.08, label: "micro_motion", camera: "subtle motion only with a soft anchor" },
        { second: 4.8, ref: 3, strength: 0.24, label: "approach_anchor", camera: "smooth camera approach before the next visual beat" },
        { second: 6.9, ref: 4, strength: 0.24, label: "detail_anchor", camera: "macro detail stays sharp with stable focus" },
        { second: 8.9, ref: 5, strength: 0.26, label: "transition_anchor", camera: "visual transition opens gradually through physical travel" },
        { second: 10.9, ref: 6, strength: 0.18, label: "environment_entry", camera: "new space enters as physical travel and parallax" },
        { second: 12.9, ref: 7, strength: 0.18, label: "environment_motion", camera: "descending or tracking motion with connected parallax" },
        { second: 14.7, ref: 8, strength: 0.18, label: "texture_anchor", camera: "surface detail grows through parallax" },
        { second: 16.6, ref: 10, strength: 0.26, label: "final_motion", camera: "final subject or environment stays moving" },
        { second: 18.7, ref: 11, strength: 0.18, label: "end_anchor", camera: "camera keeps moving until the end" },
    ];
}

function defaultSegments() {
    return [
        { seconds: 4.0, prompt: "opening subject or place stays alive with subtle natural motion", camera: "slow push-in" },
        { seconds: 3.0, prompt: "camera approaches the next detail with parallax and stable focus", camera: "easy-in push" },
        { seconds: 3.0, prompt: "visual transition develops gradually through physical travel", camera: "macro push-in" },
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
    root.style.overflow = "visible";
    table.style.overflow = "visible";

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
        let imageSource = state.multiInputSource;
        let outputName = String(imageSource?.outputName || "");
        let imageSourceText = imageSource ? `${imageSource.nodeClass || "node"}.${outputName || `slot_${imageSource.originSlot}`}` : "unlinked multi_input";
        const fixedSingleImageOutput = healCineInfoImageBatchLink(node, state);
        if (fixedSingleImageOutput) {
            const refreshedState = readPlannerLinkedFLFState(node);
            imageSource = refreshedState?.multiInputSource || imageSource;
            outputName = String(imageSource?.outputName || outputName);
            imageSourceText = imageSource ? `${imageSource.nodeClass || "node"}.${outputName || `slot_${imageSource.originSlot}`}` : imageSourceText;
        }
        const wrongSingleImageOutput = state.refs > 1 && /(^|_)image_?1$/i.test(outputName.replace(/\s+/g, "_"));
        linkedStatus.textContent = fixedSingleImageOutput
            ? `Fixed image link: multi_input now uses ${imageSourceText}; all ${state.refs} references can reach FLF.`
            : wrongSingleImageOutput
                ? `Warning: synced ${rows.length} guides / ${state.refs} refs, but multi_input is linked to ${imageSourceText}. Connect CineInfo.multi_output to use all images.`
                : `Synced from ShotPlanner: ${rows.length} FLF guides, ${state.refs || rows.length} refs, ${state.duration ?? "?"}s @ ${state.fps ?? "?"}fps. Images: ${imageSourceText}.`;
        linkedStatus.style.borderColor = (wrongSingleImageOutput && !fixedSingleImageOutput) ? CINE_FILM_LAB.warn : chrome.border;
        linkedStatus.style.color = (wrongSingleImageOutput && !fixedSingleImageOutput) ? "#FFE8A3" : CINE_FILM_LAB.muted;
        if (redraw) draw();
        return true;
    }

    function sync() {
        rows = rows.map(normalizeKeyframe);
        writeKeyframes(node, rows, linkedTimelineMetadata);
    }

    function draw() {
        table.innerHTML = "";
        const desiredHeight = FLF_SEQUENCER_TOP_CLEARANCE + FLF_SEQUENCER_BASE_HEIGHT + Math.max(1, rows.length) * FLF_SEQUENCER_ROW_HEIGHT;
        node.size = [
            Math.max(node.size?.[0] || FLF_SEQUENCER_NODE_MIN_SIZE[0], FLF_SEQUENCER_NODE_MIN_SIZE[0]),
            Math.max(node.size?.[1] || FLF_SEQUENCER_NODE_MIN_SIZE[1], desiredHeight),
        ];
        const header = document.createElement("div");
        header.style.cssText = "display:grid; grid-template-columns:94px 46px 64px 92px minmax(220px,1fr) 28px; gap:6px; color:#9fb0bd; font-size:10px; padding:0 2px;";
        header.innerHTML = "<div>Sec</div><div>Ref</div><div>Force</div><div>Label</div><div>Camera / note</div><div></div>";
        table.appendChild(header);

        rows.forEach((row, index) => {
            const r = normalizeKeyframe(row, index);
            const line = document.createElement("div");
            line.style.cssText = `display:grid; grid-template-columns:94px 46px 64px 92px minmax(220px,1fr) 28px; gap:6px; align-items:start; min-height:${FLF_SEQUENCER_ROW_HEIGHT - 8}px;`;

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
        try { node.setDirtyCanvas?.(true, true); app.graph?.setDirtyCanvas?.(true, true); } catch {}
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
    widget.computeSize = (width) => [
        width,
        Math.max(520, FLF_SEQUENCER_TOP_CLEARANCE + FLF_SEQUENCER_BASE_HEIGHT + Math.max(1, rows.length) * FLF_SEQUENCER_ROW_HEIGHT),
    ];
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
    const values = (Array.isArray(options) ? options : []).map((option) => (
        option && typeof option === "object" ? String(option.value ?? "") : String(option ?? "")
    ));
    return values.includes(raw) ? raw : fallback;
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
        if (mode === "append") parts.push("hard cut staging with a clean identity handoff inside this single shot");
        return parts;
    }
    if (transition === "match_cut") {
        parts.push("match movement continuity through shape and camera direction");
    } else if (transition === "soft_morph") {
        parts.push("single continuous transformation with a feathered optical blend");
    } else {
        parts.push("continuous physical camera movement with stable parallax and connected spatial motion");
    }
    if (nextRow && transition !== "hard_cut") {
        const nextLabel = String(nextRow.label || "next target").replace(/_/g, " ");
        parts.push(`move toward ${nextLabel} through one steady camera path`);
    }
    return parts;
}

const STEP_TRANSITION_OPTIONS = [
    { value: "off", label: "Off" },
    { value: "action_beat", label: "Action beat" },
    { value: "slow_dolly_in", label: "Slow dolly" },
    { value: "hold_then_push", label: "Hold + push" },
    { value: "orbit_bridge", label: "Orbit bridge" },
    { value: "match_move", label: "Match move" },
    { value: "rack_focus", label: "Rack focus" },
    { value: "soft_push", label: "Soft push" },
];

const STEP_TRANSITION_ARRIVAL_OPTIONS = [
    { value: "auto", label: "Auto" },
    { value: "early", label: "Early" },
    { value: "middle", label: "Middle" },
    { value: "late", label: "Late" },
    { value: "very_late", label: "Very late" },
];

function stepTransitionLabel(value) {
    const found = STEP_TRANSITION_OPTIONS.find((item) => item.value === String(value || "off"));
    return found ? found.label : "Slow dolly";
}

function stepTransitionArrivalLabel(value) {
    const found = STEP_TRANSITION_ARRIVAL_OPTIONS.find((item) => item.value === String(value || "auto"));
    return found ? found.label : "Auto";
}

function defaultStepTransitionSeconds(type, availableSeconds = 0) {
    const available = Math.max(0, Number(availableSeconds || 0));
    const fallback = {
        action_beat: 2.5,
        slow_dolly_in: 3.2,
        hold_then_push: 3.8,
        orbit_bridge: 3.6,
        match_move: 2.6,
        rack_focus: 2.4,
        soft_push: 2.8,
    }[String(type || "slow_dolly_in")] || 3.0;
    if (available >= 2.5 && available <= 5.0) return Number(available.toFixed(1));
    return fallback;
}

function defaultStepTransitionArrival(type) {
    return {
        action_beat: "middle",
        slow_dolly_in: "late",
        hold_then_push: "very_late",
        orbit_bridge: "late",
        match_move: "middle",
        rack_focus: "middle",
        soft_push: "late",
    }[String(type || "slow_dolly_in")] || "late";
}

function makeStepTransitionHeader(active, labelText = "Step transition", detailText = "to next shot") {
    const header = document.createElement("div");
    header.style.cssText = [
        "display:grid",
        "grid-template-columns:42px 1fr",
        "gap:8px",
        "align-items:center",
        "min-width:0",
        "padding:3px 2px 5px",
    ].join(";");
    const icon = document.createElement("div");
    icon.style.cssText = [
        "width:42px",
        "height:24px",
        "border-radius:999px",
        `background:${active ? "linear-gradient(135deg, rgba(223,164,81,.24), rgba(123,45,36,.28))" : "rgba(255,255,255,.045)"}`,
        `border:1px solid ${active ? "rgba(223,164,81,.72)" : "rgba(139,128,110,.36)"}`,
        "display:grid",
        "place-items:center",
        "box-shadow:inset 0 1px 0 rgba(255,255,255,.10), 0 3px 10px rgba(0,0,0,.18)",
    ].join(";");
    icon.innerHTML = `
        <svg viewBox="0 0 42 24" width="36" height="20" aria-hidden="true">
            <path d="M4 13 C12 4 26 4 35 12" fill="none" stroke="${active ? "#F1C77A" : "#8C8374"}" stroke-width="4.4" stroke-linecap="round" opacity=".28"/>
            <path d="M4 13 C12 4 26 4 35 12" fill="none" stroke="${active ? "#DFA451" : "#B5AA98"}" stroke-width="1.8" stroke-linecap="round"/>
            <path d="M33 7 L40 12 L33 17 Z" fill="${active ? "#DFA451" : "#B5AA98"}"/>
            <circle cx="5" cy="13" r="2.6" fill="${active ? "#F4E5C4" : "#D7D0C4"}" stroke="#1B1612" stroke-width="1"/>
        </svg>`;
    const copy = document.createElement("div");
    copy.style.cssText = "display:flex;flex-direction:column;gap:1px;min-width:0;text-align:left;";
    const title = document.createElement("span");
    title.textContent = labelText;
    title.style.cssText = `color:${active ? "#F4E5C4" : "#CFC6B6"};font-size:10px;font-weight:900;line-height:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;`;
    const detail = document.createElement("span");
    detail.textContent = detailText;
    detail.style.cssText = `color:${active ? "#DFA451" : "#928879"};font-size:8px;font-weight:800;line-height:1;text-transform:uppercase;letter-spacing:.03em;`;
    copy.append(title, detail);
    header.append(icon, copy);
    return header;
}

function makeStepBridgeMarker({ active = true, label = "", compact = false } = {}) {
    const marker = document.createElement("div");
    marker.style.cssText = [
        "position:relative",
        `width:${compact ? 44 : 72}px`,
        `height:${compact ? 56 : 46}px`,
        "flex:0 0 auto",
        "display:flex",
        "align-items:center",
        "justify-content:center",
        "pointer-events:none",
        "filter:drop-shadow(0 2px 5px rgba(0,0,0,.48))",
    ].join(";");
    marker.innerHTML = `
        <svg viewBox="0 0 72 38" width="${compact ? 44 : 72}" height="${compact ? 28 : 38}" aria-hidden="true">
            <path d="M7 20 C22 4 47 4 63 20" fill="none" stroke="${active ? "#F0C477" : "#8E8679"}" stroke-width="7" stroke-linecap="round" opacity=".22"/>
            <path d="M7 20 C22 4 47 4 63 20" fill="none" stroke="${active ? "#DFA451" : "#B8AD9D"}" stroke-width="2.4" stroke-linecap="round"/>
            <path d="M61 12 L71 20 L61 28 Z" fill="${active ? "#DFA451" : "#B8AD9D"}"/>
            <circle cx="8" cy="20" r="3.5" fill="${active ? "#F4E5C4" : "#D7D0C4"}" stroke="#1B1612" stroke-width="1.2"/>
        </svg>`;
    if (label && !compact) {
        const chip = document.createElement("div");
        chip.textContent = label;
        chip.style.cssText = "position:absolute;left:50%;bottom:0;transform:translateX(-50%);max-width:70px;padding:2px 6px;border:1px solid rgba(223,164,81,.65);border-radius:999px;background:rgba(30,25,20,.94);color:#F4E5C4;font:8px/1.1 monospace;font-weight:900;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
        marker.appendChild(chip);
    }
    return marker;
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

function shotboardTemplateRows() {
    return [
        { second: 0.0, ref: 1, force: 0.78, label: "opening_anchor", camera: "continuous dolly-in", transition: "continuous_motion", note: "Opening image breathes with subtle natural motion while the camera begins one connected move.", use_guide: true, use_prompt: false, relay_prompt: "opening image breathes with subtle natural motion while the camera begins one connected move" },
        { second: 2.5, ref: 2, force: 0.24, label: "path_checkpoint", camera: "continuous dolly-in", transition: "continuous_motion", note: "Soft spatial checkpoint; keep parallax and focus stable through the same camera path.", use_guide: true, use_prompt: false, relay_prompt: "soft spatial checkpoint with stable focus and connected parallax" },
        { second: 5.0, ref: 3, force: 0.26, label: "macro_detail", camera: "macro push-in", transition: "continuous_motion", note: "Macro detail grows through forward motion with crisp texture and living highlights.", use_guide: true, use_prompt: false, relay_prompt: "macro detail grows through forward motion, crisp texture, living highlights" },
        { second: 7.5, ref: 4, force: 0.22, label: "space_transfer", camera: "macro push-in", transition: "soft_morph", note: "The next space appears as an optical travel event with feathered texture continuity.", use_guide: true, use_prompt: false, relay_prompt: "the next space appears as an optical travel event with feathered texture continuity" },
        { second: 10.0, ref: 5, force: 0.20, label: "wide_motion", camera: "aerial descent", transition: "continuous_motion", note: "Wide environment opens with moving light, atmospheric depth and a steady camera drift.", use_guide: true, use_prompt: false, relay_prompt: "wide environment opens with moving light, atmospheric depth, steady camera drift" },
        { second: 13.5, ref: 6, force: 0.18, label: "final_push", camera: "continuous dolly-in", transition: "continuous_motion", note: "Final push continues the same movement with coherent scale, texture and direction.", use_guide: true, use_prompt: false, relay_prompt: "final push continues the same movement with coherent scale, texture and direction" },
    ];
}

function defaultShotboardRows() {
    return [
        { second: 0.0, ref: 1, force: 0.22, label: "ref_1", camera: "continuous dolly-in", transition: "continuous_motion", note: "", use_guide: true, use_prompt: false, relay_prompt: "", camera_relay_mode: "off", transition_relay_mode: "off", relay_addon_position: "after", relay_modifier_text: "" },
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

function parseTimelinePayloadForImport(value) {
    if (typeof value !== "string" || !value.trim()) return null;
    try {
        const parsed = JSON.parse(value);
        if (!parsed || typeof parsed !== "object") return null;
        if (Array.isArray(parsed.segments) || Array.isArray(parsed.rows) || Array.isArray(parsed.audioSegments)) return parsed;
        if (parsed.timeline && typeof parsed.timeline === "object") return parsed.timeline;
        if (String(parsed.schema || "").includes("timeline") || String(parsed.schema || "").includes("shotboard")) return parsed;
    } catch {}
    return null;
}

function looksLikeReferencePathList(value) {
    if (typeof value !== "string" || !value.trim()) return false;
    if (parseTimelinePayloadForImport(value)) return false;
    const lines = value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    if (!lines.length) return false;
    return lines.some((line) => /\.(png|jpe?g|webp|bmp|gif|tiff?|avif)(\?|$)/i.test(line) || line.includes("/") || line.includes("\\"));
}

function addReferencePathsFromValue(target, value) {
    const seen = new Set(target.map((item) => String(item || "").trim()).filter(Boolean));
    const add = (item) => {
        const clean = String(item || "").trim();
        if (!clean || seen.has(clean)) return;
        seen.add(clean);
        target.push(clean);
    };
    if (Array.isArray(value)) {
        value.forEach(add);
        return;
    }
    splitReferencePaths(value).forEach(add);
}

function workflowWidgetImagePaths(widgets, preferredIndex) {
    const paths = [];
    if (looksLikeReferencePathList(String(widgets?.[preferredIndex] || ""))) addReferencePathsFromValue(paths, widgets[preferredIndex]);
    if (looksLikeReferencePathList(String(widgets?.[1] || ""))) addReferencePathsFromValue(paths, widgets[1]);
    for (const value of widgets || []) {
        if (looksLikeReferencePathList(String(value || ""))) addReferencePathsFromValue(paths, value);
    }
    return paths;
}

function boardFromWorkflowJson(data) {
    if (!Array.isArray(data?.nodes)) return null;
    const node = data.nodes.find((item) => {
        const type = String(item?.type || item?.class_type || "");
        return type === "IAMCCS_CineShotboardLite" || type === "IAMCCS_CineShotboardPlannerPro" || type === "IAMCCS_CineShotboardPlannerProV2" || isShotboardV3Class(type) || type === "IAMCCS_CineShotboardPlannerProLegacy" || type === "IAMCCS_CineShotboardTimelinePro";
    });
    if (!node) return null;
    const widgets = Array.isArray(node.widgets_values) ? node.widgets_values : [];
    const nodeType = String(node?.type || node?.class_type || "");
    const isLite = nodeType === "IAMCCS_CineShotboardLite";
    const backupTimelineText = String(node?.properties?.iamccs_v3_timeline_data_backup || "");
    const widgetTimelineText = widgets.map((value) => String(value || "")).find((value) => parseTimelinePayloadForImport(value)) || "";
    const timelineText = parseTimelinePayloadForImport(backupTimelineText) ? backupTimelineText : widgetTimelineText;
    const timelinePayload = parseTimelinePayloadForImport(timelineText);
    const imagePaths = [];
    if (timelinePayload?.image_paths) addReferencePathsFromValue(imagePaths, timelinePayload.image_paths);
    addReferencePathsFromValue(imagePaths, workflowWidgetImagePaths(widgets, isLite ? 8 : 10));
    const settings = {
        duration_seconds: widgets[2],
        frame_rate: widgets[3],
        guide_policy: widgets[4],
        min_guide_gap_seconds: widgets[5],
        max_guides: widgets[6],
        default_force: widgets[7],
        promptrelay_epsilon: isLite ? 0.65 : widgets[8],
        wan_frame_round_mode: isLite ? "up_8n_plus_1" : widgets[9],
        image_width: widgets[isLite ? 9 : 11],
        image_height: widgets[isLite ? 10 : 12],
        image_resize_method: isLite ? "crop" : cineResizeMethodValue(widgets[13]),
        image_multiple_of: isLite ? 32 : widgets[14],
        img_compression: isLite ? 0 : widgets[15],
    };
    return {
        metadata: {
            schema: "iamccs.cine.shotboard.board",
            schema_version: 0,
            imported_from: "comfy_workflow_shotplanner_node",
            source_node_id: node.id,
            source_node_type: nodeType,
        },
        global_prompt: String(widgets[0] || ""),
        timeline_data: timelineText,
        image_paths: imagePaths,
        images: Array.isArray(data?.images) ? cloneJsonData(data.images) : [],
        settings,
        ...settings,
    };
}

function normalizeShotboardRow(row, index, options = {}) {
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    const motionForce = Number(row.motion_force ?? row.force ?? row.strength);
    const guideStrength = Number(firstNonEmpty(
        row.guide_strength,
        row.guideStrength,
        row.strength,
        row.force,
        row.motion_force,
        row.image_lock_strength,
        row.imageLockStrength,
    ));
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
    const stepEnabled = Boolean(row.step_transition_enabled || row.stepTransitionEnabled);
    const stepType = normalizeOption(
        row.step_transition_type || row.stepTransitionType || (stepEnabled ? "slow_dolly_in" : "off"),
        STEP_TRANSITION_OPTIONS,
        "off",
    );
    const stepDuration = Number(row.step_transition_duration ?? row.stepTransitionDuration ?? row.step_seconds ?? 0);
    const relayFlag = row.use_prompt ?? row.use_relay ?? row.relay ?? row.prompt_relay;
    const relayFlagText = String(relayFlag).trim().toLowerCase();
    const relayRequested = relayFlag === undefined
        ? String(legacyRelayPrompt || "").trim().length > 0
        : !(relayFlag === false || relayFlagText === "false" || relayFlagText === "0" || relayFlagText === "off" || relayFlagText === "no");
    return {
        _ui_id: String(row._ui_id || row.ui_id || `row_${Date.now()}_${Math.random().toString(16).slice(2)}`),
        second: Number.isFinite(Number(row.second ?? row.time ?? row.seconds)) ? Number(row.second ?? row.time ?? row.seconds) : index * 3,
        ref: Number.isFinite(Number(row.ref ?? row.image_ref ?? row.reference_index)) ? Number(row.ref ?? row.image_ref ?? row.reference_index) : index + 1,
        force: Number.isFinite(motionForce) ? Math.max(0, Math.min(1, motionForce)) : 0.22,
        motion_force: Number.isFinite(motionForce) ? Math.max(0, Math.min(1, motionForce)) : 0.22,
        image_lock_strength: Number.isFinite(guideStrength) ? Math.max(0, Math.min(1, guideStrength)) : (Number.isFinite(motionForce) ? Math.max(0, Math.min(1, motionForce)) : 0.22),
        guide_strength: Number.isFinite(guideStrength) ? Math.max(0, Math.min(1, guideStrength)) : (Number.isFinite(motionForce) ? Math.max(0, Math.min(1, motionForce)) : 0.22),
        strength: Number.isFinite(guideStrength) ? Math.max(0, Math.min(1, guideStrength)) : (Number.isFinite(motionForce) ? Math.max(0, Math.min(1, motionForce)) : 0.22),
        use_guide: row.use_guide ?? row.guide ?? true,
        use_prompt: Boolean(relayRequested || (stepEnabled && stepType !== "off")),
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
        step_transition_enabled: Boolean(stepEnabled && stepType !== "off"),
        step_transition_type: stepType,
        step_transition_prompt: String(row.step_transition_prompt || row.stepTransitionPrompt || ""),
        step_transition_easing: normalizeOption(row.step_transition_easing || row.stepTransitionEasing || "ease_in_out", [
            { value: "linear" },
            { value: "ease_in" },
            { value: "ease_out" },
            { value: "ease_in_out" },
        ], "ease_in_out"),
        step_transition_force_curve: normalizeOption(row.step_transition_force_curve || row.stepTransitionForceCurve || "late_target", [
            { value: "balanced" },
            { value: "late_target" },
            { value: "early_source" },
            { value: "free_motion" },
        ], "late_target"),
        step_transition_duration: Number.isFinite(stepDuration) ? Math.max(0, stepDuration) : 0,
        step_transition_arrival: normalizeOption(row.step_transition_arrival || row.stepTransitionArrival || "auto", STEP_TRANSITION_ARRIVAL_OPTIONS, "auto"),
        step_transition_auto_fit: (row.step_transition_auto_fit ?? row.stepTransitionAutoFit ?? true) !== false,
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

function sanitizePackageComponent(value, fallback = "cine_shotboard_package") {
    const clean = String(value || fallback)
        .trim()
        .replace(/[<>:"/\\|?*\x00-\x1F]+/g, "_")
        .replace(/\s+/g, "_")
        .replace(/^_+|_+$/g, "")
        .slice(0, 90);
    return clean || fallback;
}

function timestampForPackageName() {
    return new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
}

function askShotboardPackageName(defaultName) {
    return new Promise((resolve) => {
        const initial = sanitizePackageComponent(defaultName || `cine_filmmaker_v3_${timestampForPackageName()}`);
        const overlay = document.createElement("div");
        overlay.style.cssText = `
            position: fixed;
            inset: 0;
            z-index: ${CINE_REF_EDITOR_Z_INDEX};
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(0,0,0,.42);
            font-family: system-ui, -apple-system, Segoe UI, sans-serif;
        `;
        const panel = document.createElement("div");
        panel.style.cssText = `
            width: min(520px, calc(100vw - 40px));
            border: 1px solid #8B6A32;
            border-radius: 8px;
            background: #24211E;
            box-shadow: 0 18px 46px rgba(0,0,0,.48);
            color: #F3E8D3;
            padding: 16px;
        `;
        const title = document.createElement("div");
        title.textContent = "Save Package";
        title.style.cssText = "font-weight:800;font-size:15px;margin-bottom:6px;";
        const note = document.createElement("div");
        note.textContent = "Choose the package folder name. Then select an existing parent folder; IAMCCS will create this folder inside it.";
        note.style.cssText = "font-size:12px;line-height:1.35;color:#CBB99C;margin-bottom:12px;";
        const input = document.createElement("input");
        input.value = initial;
        input.style.cssText = `
            width: 100%;
            box-sizing: border-box;
            border: 1px solid #D89B45;
            border-radius: 5px;
            background: #0F1112;
            color: #FFF7EA;
            padding: 9px 10px;
            font-size: 13px;
            outline: none;
        `;
        const actions = document.createElement("div");
        actions.style.cssText = "display:flex;justify-content:flex-end;gap:8px;margin-top:14px;";
        const cancel = document.createElement("button");
        cancel.textContent = "Cancel";
        cancel.style.cssText = "border:1px solid #6B6258;background:#302D29;color:#F3E8D3;border-radius:5px;padding:7px 12px;cursor:pointer;";
        const confirm = document.createElement("button");
        confirm.textContent = "Choose Parent Folder";
        confirm.style.cssText = "border:1px solid #D89B45;background:#6A431B;color:#FFF7EA;border-radius:5px;padding:7px 12px;cursor:pointer;font-weight:700;";
        actions.append(cancel, confirm);
        panel.append(title, note, input, actions);
        overlay.append(panel);

        const cleanup = (value) => {
            window.removeEventListener("keydown", onKeyDown, true);
            overlay.remove();
            resolve(value);
        };
        const submit = () => {
            const clean = sanitizePackageComponent(input.value, initial);
            cleanup(clean);
        };
        const onKeyDown = (event) => {
            if (event.key === "Escape") {
                event.preventDefault();
                cleanup(null);
            } else if (event.key === "Enter") {
                event.preventDefault();
                submit();
            }
        };
        cancel.onclick = () => cleanup(null);
        confirm.onclick = submit;
        overlay.addEventListener("pointerdown", (event) => {
            if (event.target === overlay) cleanup(null);
        });
        window.addEventListener("keydown", onKeyDown, true);
        document.body.appendChild(overlay);
        setTimeout(() => {
            input.focus();
            input.select();
        }, 0);
    });
}

function imageExtensionForPackage(path, contentType = "") {
    const fromPath = String(path || "").split(/[?#]/)[0].match(/\.([a-z0-9]{2,5})$/i)?.[1];
    if (fromPath) return fromPath.toLowerCase() === "jpeg" ? "jpg" : fromPath.toLowerCase();
    const type = String(contentType || "").toLowerCase();
    if (type.includes("jpeg")) return "jpg";
    if (type.includes("png")) return "png";
    if (type.includes("webp")) return "webp";
    if (type.includes("gif")) return "gif";
    return "png";
}

function cineResizeMethodValue(value) {
    const method = String(value || "").trim();
    return ["crop", "pad", "keep proportion", "stretch"].includes(method) ? method : "crop";
}

function parsedTimelineSourcesForBoard(board) {
    const sources = [];
    const add = (source) => {
        if (!source) return;
        if (typeof source === "string") {
            try {
                const parsed = JSON.parse(source);
                if (parsed && typeof parsed === "object") sources.push(parsed);
            } catch {}
            return;
        }
        if (typeof source === "object") sources.push(source);
    };
    add(board?.timeline);
    add(board?.timeline_data);
    add(Array.isArray(board?.segments) ? { segments: board.segments } : null);
    add(Array.isArray(board?.rows) ? { rows: board.rows } : null);
    return sources;
}

const IAMCCS_IMAGE_TRUTH_KEYS = ["imageTruthPath", "image_truth_path", "imageFile", "image_file", "path"];

function iamccsPathLookup(pathMap, value) {
    const raw = String(value || "").trim();
    if (!raw || !pathMap) return "";
    if (pathMap[raw]) return pathMap[raw];
    const normalized = raw.replace(/\\/g, "/").replace(/^\/+/, "");
    if (pathMap[normalized]) return pathMap[normalized];
    const basename = normalized.split("/").filter(Boolean).pop() || "";
    if (basename && pathMap[basename]) return pathMap[basename];
    const entries = Object.entries(pathMap);
    const hit = entries.find(([key]) => {
        const cleanKey = String(key || "").replace(/\\/g, "/").replace(/^\/+/, "");
        return cleanKey === normalized || (basename && cleanKey.split("/").filter(Boolean).pop() === basename);
    });
    return hit ? hit[1] : "";
}

function iamccsPathBasename(value) {
    return String(value || "").replace(/\\/g, "/").split("/").filter(Boolean).pop() || "";
}

function iamccsNameLookup(nameMap, value) {
    const raw = String(value || "").trim();
    if (!raw || !nameMap) return "";
    if (nameMap[raw]) return nameMap[raw];
    const normalized = raw.replace(/\\/g, "/").replace(/^\/+/, "");
    if (nameMap[normalized]) return nameMap[normalized];
    const basename = iamccsPathBasename(normalized);
    if (basename && nameMap[basename]) return nameMap[basename];
    const hit = Object.entries(nameMap).find(([key]) => {
        const cleanKey = String(key || "").replace(/\\/g, "/").replace(/^\/+/, "");
        return cleanKey === normalized || (basename && iamccsPathBasename(cleanKey) === basename);
    });
    return hit ? hit[1] : "";
}

function collectActivePackageImagePaths(board) {
    const referencePaths = splitReferencePaths(board?.image_paths);
    const seen = new Set();
    const paths = [];
    const add = (value) => {
        const clean = String(value || "").trim();
        if (!clean || seen.has(clean)) return;
        seen.add(clean);
        paths.push(clean);
    };
    const addFromRef = (ref) => {
        const index = Math.round(Number(ref || 0)) - 1;
        if (index >= 0 && index < referencePaths.length) add(referencePaths[index]);
    };
    for (const data of parsedTimelineSourcesForBoard(board)) {
        const segments = Array.isArray(data?.segments) ? data.segments : [];
        for (const seg of segments) {
            if (!seg || typeof seg !== "object") continue;
            const type = String(seg.type || "image");
            if (type === "text" || type === "audio" || seg.textPlaceholder || seg.placeholder) continue;
            const segPath = seg.imageTruthPath || seg.image_truth_path || seg.imageFile || seg.image_file || seg.path;
            add(segPath);
            if (!segPath) addFromRef(seg.ref);
        }
        const rows = Array.isArray(data?.rows) ? data.rows : [];
        for (const row of rows) {
            if (!row || typeof row !== "object") continue;
            if (row.use_guide === false || Number(row.force ?? row.strength ?? row.guideStrength ?? 0) <= 0) continue;
            const rowPath = row.imageTruthPath || row.image_truth_path || row.imageFile || row.image_file || row.path;
            add(rowPath);
            if (!rowPath) addFromRef(row.ref);
        }
    }
    return paths;
}

function collectPackageImagePaths(board) {
    const seen = new Set();
    const paths = [];
    const add = (value) => {
        const clean = String(value || "").trim();
        if (!clean || seen.has(clean)) return;
        seen.add(clean);
        paths.push(clean);
    };
    for (const path of collectActivePackageImagePaths(board)) add(path);
    if (paths.length) return paths;
    for (const path of splitReferencePaths(board?.image_paths)) add(path);
    if (Array.isArray(board?.images)) {
        for (const image of board.images) add(image?.path || image?.original_path || image?.filename || image?.name);
    }
    for (const data of parsedTimelineSourcesForBoard(board)) {
        const segments = Array.isArray(data?.segments) ? data.segments : [];
        for (const seg of segments) add(seg?.imageTruthPath || seg?.image_truth_path || seg?.imageFile || seg?.image_file || seg?.path);
    }
    if (Array.isArray(board?.segments)) {
        for (const seg of board.segments) add(seg?.imageTruthPath || seg?.image_truth_path || seg?.imageFile || seg?.image_file || seg?.path);
    }
    return paths;
}

function cloneJsonData(value) {
    try { return JSON.parse(JSON.stringify(value ?? {})); }
    catch { return {}; }
}

function packagedImageEntries(board) {
    const entries = [];
    const addEntries = (items) => {
        if (!Array.isArray(items)) return;
        for (const item of items) {
            if (item && typeof item === "object") entries.push(item);
        }
    };
    addEntries(board?.package?.images);
    addEntries(board?.images);
    return entries;
}

function packagedReferencePaths(board) {
    const seen = new Set();
    const paths = [];
    const packageName = sanitizePackageComponent(board?.package?.name || board?.metadata?.package_name || "", "");
    const add = (value) => {
        const clean = String(value || "").trim();
        if (!clean || seen.has(clean)) return;
        seen.add(clean);
        paths.push(clean);
    };
    const entries = packagedImageEntries(board);
    const packageEntries = entries.filter((item) => item?.path || item?.comfy_input_path || item?.package_path || item?.filename || item?.name);
    for (const item of packageEntries) {
        let path = item.path || item.comfy_input_path || item.package_path || item.filename || item.name;
        if (packageName && !item.path && !item.comfy_input_path && String(path || "").replace(/\\/g, "/").startsWith("images/")) {
            path = `${packageName}/${String(path).replace(/\\/g, "/")}`;
        }
        add(path);
    }
    if (!paths.length) {
        for (const path of splitReferencePaths(board?.image_paths)) add(path);
    }
    return paths;
}

function packageHintFromReferencePaths(paths) {
    const list = Array.isArray(paths) ? paths : splitReferencePaths(paths);
    for (const path of list) {
        const clean = String(path || "").replace(/\\/g, "/").replace(/^\/+/, "");
        const match = clean.match(/^([^/]+)\/images\/[^/]+$/i);
        if (match?.[1]) {
            return {
                packageName: sanitizePackageComponent(match[1], ""),
                imagesDir: "images",
            };
        }
    }
    return { packageName: "", imagesDir: "" };
}

function packageHintFromBoard(board) {
    const explicitName = sanitizePackageComponent(board?.package?.name || board?.metadata?.package_name || "", "");
    if (explicitName) return { packageName: explicitName, imagesDir: String(board?.package?.images_dir || "images") || "images" };
    return packageHintFromReferencePaths(splitReferencePaths(board?.image_paths));
}

function boardLooksPackageLocal(board, paths) {
    if (board?.package || board?.metadata?.package_name) return true;
    if (Array.isArray(board?.images) && board.images.some((entry) => entry?.package_path || entry?.comfy_input_path)) return true;
    return (paths || []).some((path) => /(^|\/)images\/[^/]+\.(png|jpe?g|webp|bmp|gif|tiff?|avif)$/i.test(String(path || "").replace(/\\/g, "/")));
}

function applyPackageLocalImageFallbacks(board) {
    const paths = collectPackageImagePaths(board);
    if (!boardLooksPackageLocal(board, paths)) return { changed: false, paths: [] };
    const hint = packageHintFromBoard(board);
    const pathMap = {};
    for (const originalPath of paths) {
        const normalized = String(originalPath || "").replace(/\\/g, "/").replace(/^\/+/, "").trim();
        if (!normalized) continue;
        const basename = normalized.split("/").filter(Boolean).pop() || "";
        if (!basename || !looksLikeCompleteReferencePath(basename)) continue;
        const target = hint.packageName
            ? `${hint.packageName}/images/${basename}`
            : `images/${basename}`;
        if (target && target !== originalPath) pathMap[originalPath] = target;
    }
    if (!Object.keys(pathMap).length) return { changed: false, paths: [] };
    rewriteBoardPackagedImagePaths(board, pathMap, {});
    return { changed: true, paths: collectPackageImagePaths(board) };
}

function attachPackageHintToBoard(board) {
    if (!board || typeof board !== "object") return board;
    const hint = packageHintFromReferencePaths(splitReferencePaths(board.image_paths));
    if (!hint.packageName) return board;
    board.metadata = {
        ...(board.metadata || {}),
        package_name: board.metadata?.package_name || hint.packageName,
    };
    board.package = {
        ...(board.package || {}),
        name: board.package?.name || hint.packageName,
        images_dir: board.package?.images_dir || "images",
    };
    return board;
}

function sourcePathForRef(ref, originalReferencePaths) {
    const index = Math.round(Number(ref || 0)) - 1;
    return index >= 0 && index < originalReferencePaths.length ? String(originalReferencePaths[index] || "").trim() : "";
}

function rewritePackagedSegments(segments, pathMap, refMap = {}, originalReferencePaths = [], nameMap = {}) {
    if (!Array.isArray(segments)) return;
    for (const seg of segments) {
        if (!seg || typeof seg !== "object") continue;
        const type = String(seg.type || "image");
        if (type === "text" || type === "audio") continue;
        const explicitSource = String(seg.imageTruthPath || seg.image_truth_path || seg.imageFile || seg.image_file || seg.path || "").trim();
        const refSource = sourcePathForRef(seg.ref, originalReferencePaths);
        const mappedExplicit = iamccsPathLookup(pathMap, explicitSource);
        const mappedRef = iamccsPathLookup(pathMap, refSource);
        const nextRef = refMap[explicitSource] || refMap[refSource] || refMap[mappedExplicit] || refMap[mappedRef];
        const nextPath = mappedExplicit || mappedRef;
        const nextName = iamccsNameLookup(nameMap, explicitSource) || iamccsNameLookup(nameMap, refSource) || iamccsPathBasename(nextPath);
        if (nextRef) {
            seg.ref = nextRef;
            seg.imageTruthRef = nextRef;
        }
        if (nextPath) {
            seg.imageFile = nextPath;
            seg.path = nextPath;
            seg.imageTruthPath = nextPath;
            seg.imageTruthPinned = true;
            seg.imageTruthSource = "package_import_remap";
            if (nextName) {
                seg.imageTruthName = nextName;
                seg.imageName = nextName;
                if (!String(seg.label || "").trim() || /^ref_?\d+$/i.test(String(seg.label || ""))) {
                    seg.label = nextName.replace(/\.[^.]+$/, "");
                }
            }
            delete seg.image_file;
            delete seg.image_truth_path;
            continue;
        }
        for (const key of IAMCCS_IMAGE_TRUTH_KEYS) {
            const value = String(seg[key] || "").trim();
            const mapped = iamccsPathLookup(pathMap, value);
            if (mapped) {
                seg[key] = mapped;
                const mappedName = iamccsNameLookup(nameMap, value) || iamccsPathBasename(mapped);
                if (mappedName) {
                    seg.imageTruthName = mappedName;
                    seg.imageName = mappedName;
                }
            }
        }
    }
}

function rewritePackagedRows(rows, refMap = {}, originalReferencePaths = []) {
    if (!Array.isArray(rows)) return;
    for (const row of rows) {
        if (!row || typeof row !== "object") continue;
        if (row.use_guide === false) continue;
        const explicitSource = String(row.imageTruthPath || row.image_truth_path || row.imageFile || row.image_file || row.path || "").trim();
        const refSource = sourcePathForRef(row.ref, originalReferencePaths);
        const nextRef = refMap[explicitSource] || refMap[refSource];
        if (nextRef) row.ref = nextRef;
    }
}

function rewriteBoardForPackage(board, orderedPaths, pathMap, manifestImages) {
    const packagedBoard = cloneJsonData(board);
    const packagedPaths = orderedPaths.map((path) => pathMap[path]).filter(Boolean);
    const originalReferencePaths = splitReferencePaths(board?.image_paths);
    const refMap = {};
    orderedPaths.forEach((path, index) => {
        const clean = String(path || "").trim();
        if (clean) refMap[clean] = index + 1;
    });
    if (packagedPaths.length) packagedBoard.image_paths = packagedPaths;
    packagedBoard.images = (manifestImages || []).map((entry, index) => ({
        ref: entry.ref || index + 1,
        path: entry.path || entry.comfy_input_path || entry.package_path || entry.original_path || "",
        package_path: entry.package_path || "",
        comfy_input_path: entry.comfy_input_path || "",
        original_path: entry.original_path || "",
        name: entry.filename || entry.original_name || `ref_${index + 1}`,
        filename: entry.filename || "",
        content_type: entry.content_type || "",
        data_url: entry.data_url || "",
        bytes: entry.bytes || 0,
        error: entry.error || undefined,
    }));
    rewritePackagedSegments(packagedBoard.segments, pathMap, refMap, originalReferencePaths);
    rewritePackagedRows(packagedBoard.rows, refMap, originalReferencePaths);
    if (packagedBoard.timeline && typeof packagedBoard.timeline === "object") {
        if (packagedPaths.length) packagedBoard.timeline.image_paths = packagedPaths;
        rewritePackagedSegments(packagedBoard.timeline.segments, pathMap, refMap, originalReferencePaths);
        rewritePackagedRows(packagedBoard.timeline.rows, refMap, originalReferencePaths);
    }
    if (typeof packagedBoard.timeline_data === "string" && packagedBoard.timeline_data.trim()) {
        try {
            const parsed = JSON.parse(packagedBoard.timeline_data);
            if (parsed && typeof parsed === "object") {
                if (packagedPaths.length) parsed.image_paths = packagedPaths;
                rewritePackagedSegments(parsed.segments, pathMap, refMap, originalReferencePaths);
                rewritePackagedRows(parsed.rows, refMap, originalReferencePaths);
                packagedBoard.timeline_data = JSON.stringify(parsed, null, 2);
            }
        } catch {}
    }
    return packagedBoard;
}

async function writePackageTextFile(directoryHandle, filename, text) {
    const fileHandle = await directoryHandle.getFileHandle(filename, { create: true });
    const writable = await fileHandle.createWritable();
    await writable.write(text);
    await writable.close();
}

async function writePackageBlobFile(directoryHandle, filename, blob) {
    const fileHandle = await directoryHandle.getFileHandle(filename, { create: true });
    const writable = await fileHandle.createWritable();
    await writable.write(blob);
    await writable.close();
}

async function clearPackageImageFolder(directoryHandle) {
    if (!directoryHandle || typeof directoryHandle.entries !== "function" || typeof directoryHandle.removeEntry !== "function") return;
    try {
        for await (const [name, handle] of directoryHandle.entries()) {
            if (handle?.kind !== "file") continue;
            if (!/^ref_\d+\.(png|jpe?g|webp|gif|bmp|tiff?|avif)$/i.test(String(name || ""))) continue;
            try {
                await directoryHandle.removeEntry(name);
            } catch (err) {
                console.warn("[IAMCCS Cine Shotboard] package stale image cleanup failed", name, err);
            }
        }
    } catch (err) {
        console.warn("[IAMCCS Cine Shotboard] package image cleanup unavailable", err);
    }
}

async function fetchReferenceBlobForPackage(path) {
    const url = previewUrlForPath(path);
    if (!url) throw new Error("Missing reference image path");
    const resp = await fetch(url, { cache: "no-store" });
    if (!resp.ok) throw new Error(`Image fetch failed (${resp.status})`);
    return await resp.blob();
}

function blobToDataUrl(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ""));
        reader.onerror = () => reject(reader.error || new Error("Blob read failed"));
        reader.readAsDataURL(blob);
    });
}

async function ensureDirectoryWritable(directoryHandle) {
    if (!directoryHandle || typeof directoryHandle.queryPermission !== "function") return true;
    const options = { mode: "readwrite" };
    try {
        if ((await directoryHandle.queryPermission(options)) === "granted") return true;
        if (typeof directoryHandle.requestPermission === "function") {
            return (await directoryHandle.requestPermission(options)) === "granted";
        }
    } catch {}
    return false;
}

async function buildShotboardPackageJson(board, label, reason = "", statusCallback = null) {
    const imagePaths = collectPackageImagePaths(board);
    const packageName = `${sanitizePackageComponent(label)}_${timestampForPackageName()}`;
    const images = [];
    for (let index = 0; index < imagePaths.length; index += 1) {
        const originalPath = imagePaths[index];
        statusCallback?.(`Building fallback package image ${index + 1}/${imagePaths.length}...`);
        const entry = {
            ref: index + 1,
            original_path: originalPath,
            original_name: String(originalPath).split(/[\\/]/).pop() || `ref_${index + 1}`,
        };
        try {
            const blob = await fetchReferenceBlobForPackage(originalPath);
            entry.content_type = blob.type || "";
            entry.bytes = blob.size || 0;
            entry.ext = imageExtensionForPackage(originalPath, blob.type);
            entry.data_url = await blobToDataUrl(blob);
        } catch (err) {
            entry.error = String(err?.message || err);
        }
        images.push(entry);
    }
    return {
        filename: `${packageName}.json`,
        packageName,
        payload: {
        metadata: {
            schema: "iamccs.cine.shotboard.package.fallback",
            schema_version: 2,
            cine_ui_version: CINE_VERSION,
            saved_at: new Date().toISOString(),
            package_name: packageName,
            reason: String(reason || "Folder package export unavailable"),
            note: "Self-contained fallback package. Images are embedded as data_url when they were readable from ComfyUI.",
        },
        board,
        images,
        },
    };
}

async function downloadShotboardPackageJsonFallback(board, label, reason = "", statusCallback = null) {
    const built = await buildShotboardPackageJson(board, label, reason, statusCallback);
    downloadJsonFile(built.payload, built.filename);
    statusCallback?.(`Downloaded fallback package JSON: ${built.filename}`);
}

async function saveShotboardPackageViaBackend(board, label, packageName, statusCallback = null) {
    statusCallback?.("Folder picker unavailable; saving package folder in ComfyUI input...");
    const resp = await api.fetchApi("/api/iamccs/cine/save_shotboard_package", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ board, label, package_name: packageName }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok || !data?.ok) {
        throw new Error(data?.error || `Backend package save failed (${resp.status})`);
    }
    const failed = Number(data.failed_images || 0);
    statusCallback?.(`Saved package folder: ${data.package_name || packageName}${failed ? ` (${failed} image errors in manifest)` : ""}`);
    return data;
}

async function saveShotboardPackageFolder(board, label, statusCallback = null, packageNameOverride = "", options = {}) {
    const imagePaths = collectPackageImagePaths(board);
    const useSelectedFolderAsPackage = Boolean(options?.useSelectedFolderAsPackage);
    let packageName = sanitizePackageComponent(packageNameOverride || `${sanitizePackageComponent(label)}_${timestampForPackageName()}`);
    if (typeof window.showDirectoryPicker !== "function") {
        try {
            await saveShotboardPackageViaBackend(board, label, packageName, statusCallback);
        } catch (err) {
            console.warn("[IAMCCS Cine Shotboard] backend package save failed; using JSON fallback", err);
            await downloadShotboardPackageJsonFallback(board, label, err?.message || "Browser folder writing is unavailable.", statusCallback);
        }
        return;
    }

    try {
        statusCallback?.(useSelectedFolderAsPackage
            ? "Choose an existing package folder, or create one with the New folder button..."
            : "Choose the parent folder; IAMCCS will create the package folder inside it...");
        const rootHandle = await window.showDirectoryPicker({
            mode: "readwrite",
            id: "iamccs-shotboard-package",
            startIn: "documents",
        });
        const packageHandle = useSelectedFolderAsPackage ? rootHandle : await rootHandle.getDirectoryHandle(packageName, { create: true });
        if (useSelectedFolderAsPackage) {
            packageName = sanitizePackageComponent(rootHandle?.name || packageName);
        }
        const imagesHandle = await packageHandle.getDirectoryHandle("images", { create: true });
        const pathMap = {};
        const pendingImageWrites = [];
        const manifest = {
            metadata: {
                schema: "iamccs.cine.shotboard.package",
                schema_version: 1,
                cine_ui_version: CINE_VERSION,
                saved_at: new Date().toISOString(),
                package_name: packageName,
            },
            board_file: "board.json",
            images_dir: "images",
            image_count: imagePaths.length,
            images: [],
        };

        for (let index = 0; index < imagePaths.length; index += 1) {
            const originalPath = imagePaths[index];
            statusCallback?.(`Reading package image ${index + 1}/${imagePaths.length}...`);
            const entry = {
                ref: index + 1,
                original_path: originalPath,
                original_name: String(originalPath).split(/[\\/]/).pop() || `ref_${index + 1}`,
            };
            try {
                const blob = await fetchReferenceBlobForPackage(originalPath);
                const ext = imageExtensionForPackage(originalPath, blob.type);
                const filename = `ref_${String(index + 1).padStart(3, "0")}.${ext}`;
                entry.package_path = `images/${filename}`;
                entry.comfy_input_path = `${packageName}/images/${filename}`;
                entry.path = entry.comfy_input_path;
                entry.content_type = blob.type || "";
                entry.bytes = blob.size || 0;
                entry.data_url = await blobToDataUrl(blob);
                pathMap[originalPath] = entry.path;
                pendingImageWrites.push({ filename, blob });
            } catch (err) {
                entry.error = String(err?.message || err);
                console.warn("[IAMCCS Cine Shotboard] package image export failed", originalPath, err);
            }
            manifest.images.push(entry);
        }

        await clearPackageImageFolder(imagesHandle);
        for (let index = 0; index < pendingImageWrites.length; index += 1) {
            const item = pendingImageWrites[index];
            statusCallback?.(`Writing package image ${index + 1}/${pendingImageWrites.length}...`);
            await writePackageBlobFile(imagesHandle, item.filename, item.blob);
        }

        const packagedBoard = {
            ...rewriteBoardForPackage(board, imagePaths, pathMap, manifest.images),
            metadata: {
                ...(board?.metadata || {}),
                packaged_at: manifest.metadata.saved_at,
                package_schema: manifest.metadata.schema,
            },
            package: {
                name: packageName,
                images_dir: "images",
                images: manifest.images,
            },
        };
        await writePackageTextFile(packageHandle, "board.json", JSON.stringify(packagedBoard, null, 2));
        await writePackageTextFile(packageHandle, "manifest.json", JSON.stringify(manifest, null, 2));
        const failed = manifest.images.filter((entry) => entry.error).length;
        statusCallback?.(`Saved package folder: ${packageName}${failed ? ` (${failed} image errors in manifest)` : ""}`);
    } catch (err) {
        if (String(err?.name || "") === "AbortError") {
            statusCallback?.("Save Package cancelled.");
            return;
        }
        console.warn("[IAMCCS Cine Shotboard] folder package failed; using JSON fallback", err);
        try {
            await saveShotboardPackageViaBackend(board, label, packageName, statusCallback);
        } catch (backendErr) {
            console.warn("[IAMCCS Cine Shotboard] backend package fallback failed; using JSON fallback", backendErr);
            await downloadShotboardPackageJsonFallback(board, label, backendErr?.message || err?.message || err, statusCallback);
        }
    }
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

function sanitizeSaveFilename(filename, extension = ".json") {
    let value = String(filename || "iamccs_board")
        .replace(/[<>:"/\\|?*\x00-\x1F]+/g, "_")
        .replace(/^_+|_+$/g, "")
        .slice(0, 140) || "iamccs_board";
    return extension && !value.toLowerCase().endsWith(extension.toLowerCase()) ? `${value}${extension}` : value;
}

async function saveJsonAsFile(payload, suggestedName, statusCallback = null, description = "IAMCCS JSON") {
    const safeName = sanitizeSaveFilename(suggestedName || safeBoardFilename("iamccs_board"), ".json");
    if (typeof window.showSaveFilePicker === "function") {
        try {
            const fileHandle = await window.showSaveFilePicker({
                suggestedName: safeName,
                types: [{ description, accept: { "application/json": [".json"] } }],
            });
            const writable = await fileHandle.createWritable();
            await writable.write(JSON.stringify(payload, null, 2));
            await writable.close();
            statusCallback?.(`Saved: ${safeName}`);
            return { mode: "save_file_picker", filename: safeName };
        } catch (err) {
            if (String(err?.name || "") === "AbortError") {
                statusCallback?.("Save cancelled.");
                return { mode: "cancelled", filename: safeName };
            }
            console.warn("[IAMCCS Cine Shotboard] Save As file picker failed; using download fallback", err);
        }
    }
    downloadJsonFile(payload, safeName);
    statusCallback?.(`Downloaded: ${safeName}`);
    return { mode: "download", filename: safeName };
}

async function saveBoardJsonAs(board, suggestedName, statusCallback = null) {
    return saveJsonAsFile(board, suggestedName || safeBoardFilename("iamccs_board"), statusCallback, "IAMCCS Shotboard Board");
}

async function saveBoardJsonToChosenFolder(board, filename, statusCallback = null) {
    const safeName = String(filename || safeBoardFilename("iamccs_board"))
        .replace(/[<>:"/\\|?*\x00-\x1F]+/g, "_")
        .replace(/\.json$/i, "") + ".json";
    if (typeof window.showDirectoryPicker !== "function") {
        downloadJsonFile(board, safeName);
        statusCallback?.("Folder picker unavailable; board downloaded instead.");
        return { mode: "download", filename: safeName };
    }
    const directoryHandle = await window.showDirectoryPicker({ mode: "readwrite" });
    const fileHandle = await directoryHandle.getFileHandle(safeName, { create: true });
    const writable = await fileHandle.createWritable();
    await writable.write(JSON.stringify(board, null, 2));
    await writable.close();
    statusCallback?.(`Board exported: ${safeName}`);
    return { mode: "folder", filename: safeName };
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

function looksLikeCompleteReferencePath(value) {
    const clean = String(value || "").trim();
    if (!clean) return false;
    if (isAbsoluteLocalPath(clean)) return true;
    return /\.(png|jpe?g|webp|bmp|gif|tiff?|avif)$/i.test(clean);
}

function splitReferencePaths(value) {
    if (Array.isArray(value)) {
        return value.map((item) => String(item || "").trim()).filter(Boolean);
    }
    const raw = String(value || "").trim();
    if (!raw) return [];
    try {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) {
            return parsed.map((item) => String(item || "").trim()).filter(Boolean);
        }
    } catch {}
    const newlineParts = raw.split(/\r?\n/).map((item) => item.trim()).filter(Boolean);
    if (newlineParts.length > 1) return newlineParts;
    const commaParts = raw.split(",").map((item) => item.trim()).filter(Boolean);
    if (commaParts.length > 1 && commaParts.every(looksLikeCompleteReferencePath)) return commaParts;
    return [raw];
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

function roundToStep(value, step = 0.1) {
    const scale = Math.max(1, Math.round(1 / Math.max(0.0001, Number(step) || 0.1)));
    return Math.round((Number(value) || 0) * scale) / scale;
}

function shiftFollowingRowsForTimeChange(rows, rowId, value, normalizeRow, minGapSeconds = 0.1) {
    const list = (Array.isArray(rows) ? rows : []).map((row, index) => normalizeRow(row, index));
    const index = list.findIndex((row) => row?._ui_id === rowId);
    if (index < 0) return list;
    const before = Number(list[index]?.second || 0);
    let nextValue = Math.max(0, Number(value) || 0);
    if (index > 0) nextValue = Math.max(nextValue, Number(list[index - 1]?.second || 0) + minGapSeconds);
    nextValue = roundToStep(nextValue, 0.1);
    const delta = nextValue - before;
    list[index] = normalizeRow({ ...list[index], second: nextValue }, index);
    for (let i = index + 1; i < list.length; i += 1) {
        const shifted = Number(list[i]?.second || 0) + delta;
        const minAllowed = Number(list[i - 1]?.second || 0) + minGapSeconds;
        list[i] = normalizeRow({ ...list[i], second: roundToStep(Math.max(shifted, minAllowed), 0.1) }, i);
    }
    return list;
}

function insertDuplicateRowAfter(rows, sourceIndex, normalizeRow, makeFallbackRow, newRef) {
    const list = (Array.isArray(rows) ? rows : []).map((row, index) => normalizeRow(row, index));
    const safeIndex = Math.max(0, Math.min(list.length - 1, Number(sourceIndex) || 0));
    const source = list[safeIndex] || makeFallbackRow();
    const insertSecond = roundToStep(Number(source?.second || 0) + 1, 0.1);
    const shifted = list.map((row, index) => index > safeIndex ? normalizeRow({
        ...row,
        second: roundToStep(Number(row.second || 0) + 1, 0.1),
    }, index) : row);
    const clone = normalizeRow({
        ...source,
        _ui_id: undefined,
        ref: newRef,
        second: insertSecond,
        label: `${String(source?.label || `ref_${Math.max(1, Number(newRef) - 1)}`).replace(/_dup\d*$/i, "")}_dup`,
        use_guide: true,
    }, safeIndex + 1);
    shifted.splice(safeIndex + 1, 0, clone);
    return shifted.map((row, index) => normalizeRow(row, index));
}

function getOwnReferencePaths(node) {
    return splitReferencePaths(getWidget(node, "image_paths")?.value);
}

function setOwnReferencePaths(node, paths) {
    const clean = (Array.isArray(paths) ? paths : []).map((p) => String(p || "").trim()).filter(Boolean);
    if (clean.length && node) {
        node.properties = node.properties || {};
        node.properties.iamccs_refs_cleared = false;
    }
    return setWidgetValue(node, "image_paths", clean.join("\n"));
}

function clearOwnReferencePaths(node) {
    if (node) {
        node.properties = node.properties || {};
        node.properties.iamccs_refs_cleared = true;
    }
    return setWidgetValue(node, "image_paths", "");
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

function appendReferencePath(node, newPath) {
    const cleanPath = String(newPath || "").trim();
    const current = getConnectedReferencePaths(node).slice();
    const next = (current.length ? current : getOwnReferencePaths(node).slice())
        .map((item) => String(item || "").trim())
        .filter(Boolean);
    if (!cleanPath) return { paths: next, refNumber: Math.max(1, next.length), appended: false };
    const normalize = (value) => String(value || "").replace(/\\/g, "/").trim();
    const existingIndex = next.findIndex((item) => normalize(item) === normalize(cleanPath));
    if (existingIndex >= 0) {
        setOwnReferencePaths(node, next);
        try { node?.setDirtyCanvas?.(true, true); app?.graph?.setDirtyCanvas?.(true, true); } catch {}
        return { paths: next, refNumber: existingIndex + 1, appended: false };
    }
    next.push(cleanPath);
    setOwnReferencePaths(node, next);
    try { node?.setDirtyCanvas?.(true, true); app?.graph?.setDirtyCanvas?.(true, true); } catch {}
    return { paths: next, refNumber: next.length, appended: true };
}

function syncSegmentImageFileForReference(segments, refIndex, imagePath) {
    const refNumber = Math.max(1, Math.round(Number(refIndex || 0) + 1));
    const cleanPath = String(imagePath || "").trim();
    if (!Array.isArray(segments) || !cleanPath) return false;
    let changed = false;
    for (const seg of segments) {
        if (!seg || typeof seg !== "object") continue;
        if (String(seg.type || "image") === "text" || String(seg.type || "image") === "audio") continue;
        if (Math.round(Number(seg.ref || 0)) !== refNumber) continue;
        seg.imageFile = cleanPath;
        seg.path = cleanPath;
        delete seg.image_file;
        changed = true;
    }
    return changed;
}

function isShotboardV2Node(node) {
    return nodeClassName(node) === "IAMCCS_CineShotboardPlannerProV2";
}

function isShotboardLiteNode(node) {
    return nodeClassName(node) === "IAMCCS_CineShotboardLite";
}

function getConnectedReferencePaths(node) {
    if (node?.properties?.iamccs_refs_cleared === true) return [];
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
        body.append("type", "input");
        body.append("overwrite", "false");
        try {
            const resp = await api.fetchApi("/upload/image", { method: "POST", body });
            if (resp.status === 200) {
                const data = await resp.json();
                let name = String(data.name || file.name || "").trim();
                if (data.subfolder) name = `${data.subfolder}/${name}`;
                if (name) uploaded.push(name);
            }
        } catch (err) {
            console.error("[IAMCCS Cine Shotboard] image upload failed", err);
        }
    }
    return uploaded;
}

async function fileFromPackageDataUrl(dataUrl, filename, contentType = "") {
    const resp = await fetch(String(dataUrl || ""));
    if (!resp.ok) throw new Error(`Packaged image decode failed (${resp.status})`);
    const blob = await resp.blob();
    return new File([blob], filename || "iamccs_packaged_ref.png", { type: contentType || blob.type || "image/png" });
}

async function restorePackagedImagesToComfyInput(board, statusCallback = null) {
    const seen = new Set();
    const entries = [];
    for (const entry of packagedImageEntries(board)) {
        const dataUrl = String(entry?.data_url || "").trim();
        if (!dataUrl) continue;
        const key = `${String(entry?.path || entry?.comfy_input_path || entry?.package_path || entry?.original_path || entry?.filename || entry?.name || "")}|${dataUrl.slice(0, 96)}`;
        if (seen.has(key)) continue;
        seen.add(key);
        entries.push(entry);
    }
    if (!entries.length) return { paths: [], pathMap: {}, nameMap: {} };
    const packageName = sanitizePackageComponent(board?.package?.name || board?.metadata?.package_name || "", "");
    const files = [];
    for (let index = 0; index < entries.length; index += 1) {
        const entry = entries[index];
        const fallbackExt = imageExtensionForPackage(entry.filename || entry.original_name || entry.name || "", entry.content_type || "");
        const refFallback = `ref_${String(index + 1).padStart(3, "0")}.${fallbackExt}`;
        // Prefer package_path basename (ref_001.png) over original names so imported images keep the clean package naming
        // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        const packageBasename = iamccsPathBasename(String(entry.package_path || "").replace(/\\/g, "/"));
        const rawName = packageBasename || entry.filename || refFallback;
        const filename = sanitizePackageComponent(rawName, refFallback);
        statusCallback?.(`Restoring packaged image ${index + 1}/${entries.length}...`);
        try {
            files.push(await fileFromPackageDataUrl(entry.data_url, filename, entry.content_type || ""));
        } catch (err) {
            console.warn("[IAMCCS Cine Shotboard] packaged image restore failed", entry, err);
        }
    }
    const uploaded = await uploadShotboardImages(files);
    const pathMap = {};
    const nameMap = {};
    entries.forEach((entry, index) => {
        const uploadedPath = uploaded[index];
        if (!uploadedPath) return;
        const uploadedName = iamccsPathBasename(uploadedPath) || String(entry?.filename || entry?.original_name || entry?.name || "").trim();
        const mapAlias = (value) => {
            const clean = String(value || "").trim();
            if (!clean) return;
            pathMap[clean] = uploadedPath;
            nameMap[clean] = uploadedName;
            const normalized = clean.replace(/\\/g, "/").replace(/^\/+/, "");
            if (normalized) {
                pathMap[normalized] = uploadedPath;
                nameMap[normalized] = uploadedName;
            }
            const base = iamccsPathBasename(normalized);
            if (base) {
                pathMap[base] = uploadedPath;
                nameMap[base] = uploadedName;
            }
        };
        for (const key of ["path", "comfy_input_path", "package_path", "original_path", "filename", "name", "original_name"]) mapAlias(entry?.[key]);
        mapAlias(uploadedPath);
        const packagePath = String(entry.package_path || "").replace(/\\/g, "/").replace(/^\/+/, "");
        if (packageName && packagePath) mapAlias(`${packageName}/${packagePath}`);
    });
    return { paths: uploaded.filter(Boolean), pathMap, nameMap };
}

function rewriteReferencePathList(value, pathMap) {
    const paths = splitReferencePaths(value);
    if (!paths.length) return value;
    return paths.map((path) => iamccsPathLookup(pathMap, path) || path);
}

function rewriteBoardPackagedImagePaths(board, pathMap, nameMap = {}) {
    if (!board || !pathMap || !Object.keys(pathMap).length) return board;
    const originalReferencePaths = splitReferencePaths(board.image_paths);
    if (Object.prototype.hasOwnProperty.call(board, "image_paths")) {
        board.image_paths = rewriteReferencePathList(board.image_paths, pathMap);
    }
    rewritePackagedSegments(board.segments, pathMap, {}, originalReferencePaths, nameMap);
    if (board.timeline && typeof board.timeline === "object") {
        const timelineOriginalReferencePaths = splitReferencePaths(board.timeline.image_paths).length ? splitReferencePaths(board.timeline.image_paths) : originalReferencePaths;
        if (Object.prototype.hasOwnProperty.call(board.timeline, "image_paths")) {
            board.timeline.image_paths = rewriteReferencePathList(board.timeline.image_paths, pathMap);
        }
        rewritePackagedSegments(board.timeline.segments, pathMap, {}, timelineOriginalReferencePaths, nameMap);
    }
    if (typeof board.timeline_data === "string" && board.timeline_data.trim()) {
        try {
            const parsed = JSON.parse(board.timeline_data);
            if (parsed && typeof parsed === "object") {
                const parsedOriginalReferencePaths = splitReferencePaths(parsed.image_paths).length ? splitReferencePaths(parsed.image_paths) : originalReferencePaths;
                if (Object.prototype.hasOwnProperty.call(parsed, "image_paths")) {
                    parsed.image_paths = rewriteReferencePathList(parsed.image_paths, pathMap);
                }
                rewritePackagedSegments(parsed.segments, pathMap, {}, parsedOriginalReferencePaths, nameMap);
                board.timeline_data = JSON.stringify(parsed, null, 2);
            }
        } catch {}
    }
    return board;
}

async function packagedReferencePathsForImport(board, statusCallback = null) {
    const restored = await restorePackagedImagesToComfyInput(board, statusCallback);
    if (restored.paths.length) {
        rewriteBoardPackagedImagePaths(board, restored.pathMap, restored.nameMap || {});
        return restored.paths;
    }
    const packageLocal = applyPackageLocalImageFallbacks(board);
    if (packageLocal.changed) {
        statusCallback?.("Package images fallback applied: using images/ beside the board when names match.");
        return packagedReferencePaths(board);
    }
    return packagedReferencePaths(board);
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

async function grabShotboardReferenceProjectCopy(node, path) {
    if (!path) return null;
    const targetWidth = Number(getWidget(node, "image_width")?.value || 768);
    const targetHeight = Number(getWidget(node, "image_height")?.value || 432);
    const data = await transformShotboardReference({
        path,
        width: targetWidth,
        height: targetHeight,
        fit_mode: "contain",
        zoom: 1,
        pan_x: 0,
        pan_y: 0,
        rotation: 0,
        resample: "lanczos",
        crop_box: null,
        crop_box_source: "grab_project_size",
    });
    node._iamccsLastGrabbedReferencePath = data?.absolute_path || data?.path || "";
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
        `z-index:${CINE_REF_EDITOR_Z_INDEX}`,
        "background:rgba(5,7,9,.82)",
        "display:flex",
        "align-items:center",
        "justify-content:center",
        "padding:28px",
        "box-sizing:border-box",
        "pointer-events:auto",
        "isolation:isolate",
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
        status.textContent = "Saving edited frame as a new project reference...";
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
            const appliedPath = data?.absolute_path || data?.path || data?.relative_path || data?.filename || data?.name;
            console.info("[IAMCCS REF EDITOR] transform response", {
                source: path,
                appliedPath,
                absolutePath: data?.absolute_path || "",
                transform: data?.metadata?.transform || null,
            });
            // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
            // Bug #6 fix: removed strict same-path throw — the backend saves a new _cineedit_ file,
            // but if timestamps collide or the user re-edits the same slot, we should still apply.
            if (!appliedPath) {
                throw new Error("Reference editor: transform API returned no output path.");
            }
            if (String(appliedPath) === String(path)) {
                console.warn("[IAMCCS REF EDITOR] output path equals source — applying anyway (same-edit idempotent)");
            }
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
        z-index: ${CINE_FULLSCREEN_Z_INDEX};
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
    const klass = nodeClassName(node);
    title.textContent = klass === "IAMCCS_BoardMaker"
        ? "IAMCCS BoardMaker Editor"
        : isWanShotboardV3Class(klass)
            ? "IAMCCS Shotboard WAN PURE"
            : klass === "IAMCCS_WanLegacyPlannerDisabled"
                ? "IAMCCS Shotboard V3 Full Frame"
            : "IAMCCS Shotboard Full Frame";
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
        G: { bg: "#2E6B3F", border: "#8CEAA3", color: "#F2FFF5" },
    };
    return tones[label] || { bg: CINE_NODE_CHROME.shotboard.header, border: CINE_FILM_LAB.borderSoft, color: "#fff" };
}

function refPicker(value, referencePaths, onChange, options = {}) {
    const thumbWidth = Math.max(132, Number(options.thumbWidth) || 150);
    const thumbHeight = Math.max(86, Number(options.thumbHeight) || 96);
    const hasSideActions = Boolean(options.onEdit || options.onReplace || options.onDuplicate || options.onGrab);
    const sideActionCount = [options.onEdit, options.onReplace, options.onDuplicate, options.onGrab].filter(Boolean).length;
    const gutterWidth = hasSideActions ? 28 : 0;
    const imageWidth = hasSideActions ? Math.max(72, thumbWidth - gutterWidth) : Math.max(120, thumbWidth);
    const wrap = document.createElement("div");
    wrap.style.cssText = `display:grid;grid-template-rows:${thumbHeight}px 26px;gap:5px;align-items:center;width:${thumbWidth}px;max-width:${thumbWidth}px;min-width:0;box-sizing:border-box;`;

    const frameWrap = document.createElement("div");
    frameWrap.style.cssText = hasSideActions
        ? `display:grid;grid-template-columns:${gutterWidth}px ${imageWidth}px;width:${thumbWidth}px;height:${thumbHeight}px;min-width:0;box-sizing:border-box;`
        : `display:block;width:${thumbWidth}px;height:${thumbHeight}px;min-width:0;box-sizing:border-box;`;

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
        box-sizing: border-box;
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
        if (options.onGrab) {
            actionRail.appendChild(sideAction("G", "Grab/export a project-sized copy into IAMCCS_newimages", () => {
                if (!path) return;
                options.onGrab(index - 1, path);
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
    select.style.cssText = inputBase() + `height:26px;padding:2px 5px;width:${thumbWidth}px;max-width:${thumbWidth}px;`;
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

function attachVerticalStepDrag(element, getValue, applyValue, stepValue, options = {}) {
    const threshold = Number(options.threshold || 8);
    const minValue = Number.isFinite(Number(options.min)) ? Number(options.min) : -Infinity;
    const maxValue = Number.isFinite(Number(options.max)) ? Number(options.max) : Infinity;
    const clamp = (value) => Math.min(maxValue, Math.max(minValue, Number(value) || 0));
    element.dataset.iamccsStepDrag = "true";
    element.style.cursor = element.style.cursor || "ew-resize";
    element.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;
        if (event.detail > 1 || element.dataset.iamccsNumberEditing === "true") return;
        event.preventDefault();
        event.stopPropagation();
        const startX = Number(event.clientX || 0);
        const startY = Number(event.clientY || 0);
        let stepsApplied = 0;
        let dragging = true;
        element.setPointerCapture?.(event.pointerId);
        const finish = (finishEvent) => {
            dragging = false;
            finishEvent?.stopPropagation?.();
            try { element.releasePointerCapture?.(event.pointerId); } catch {}
            window.removeEventListener("pointermove", move, true);
            window.removeEventListener("pointerup", finish, true);
            window.removeEventListener("pointercancel", finish, true);
        };
        const move = (moveEvent) => {
            if (!dragging || !(moveEvent.buttons & 1)) return;
            moveEvent.preventDefault();
            moveEvent.stopPropagation();
            const dx = Number(moveEvent.clientX || 0) - startX;
            const dy = startY - Number(moveEvent.clientY || 0);
            const dominantDelta = Math.abs(dx) >= Math.abs(dy) ? dx : dy;
            const nextSteps = Math.trunc(dominantDelta / threshold);
            const deltaSteps = nextSteps - stepsApplied;
            if (!deltaSteps) return;
            stepsApplied = nextSteps;
            const current = Number(getValue());
            const next = clamp((Number.isFinite(current) ? current : 0) + deltaSteps * stepValue);
            applyValue(next);
        };
        window.addEventListener("pointermove", move, { passive: false, capture: true });
        window.addEventListener("pointerup", finish, { passive: false, capture: true });
        window.addEventListener("pointercancel", finish, { passive: false, capture: true });
    }, { passive: false, capture: true });
}

function numberStepperControl(value, step, min, max, onChange, options = {}) {
    const stepValue = Math.max(0.0001, Number(step) || 1);
    const minValue = Number.isFinite(Number(min)) ? Number(min) : -Infinity;
    const hasMax = max !== null && max !== undefined && String(max).trim() !== "";
    const maxValue = hasMax && Number.isFinite(Number(max)) ? Number(max) : Infinity;
    const precision = Math.max(1, stepPrecision(step));
    const liveInput = options.liveInput !== false;
    const wrap = document.createElement("div");
    wrap.style.cssText = "display:grid;grid-template-columns:32px minmax(58px,1fr) 32px;gap:12px;align-items:center;min-width:148px;";

    const input = document.createElement("input");
    input.type = "text";
    input.inputMode = "decimal";
    input.readOnly = true;
    input.value = formatStepperValue(value, precision);
    input.style.cssText = inputBase() + "height:30px;padding:4px 6px;text-align:center;font-variant-numeric:tabular-nums;cursor:ew-resize;";

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
        btn.style.cssText = "height:30px;min-width:32px;padding:0;border:1px solid #8DE7FF;border-radius:5px;background:linear-gradient(180deg,#216077,#0B2E42);color:#EAF8FF;font-size:15px;font-weight:900;line-height:1;cursor:pointer;margin:0 2px;box-shadow:inset 0 1px 0 rgba(255,255,255,.18),0 0 0 1px rgba(0,0,0,.35);";
        btn.onclick = (event) => {
            event.preventDefault();
            const current = Number(String(input.value).replace(",", ".")) || 0;
            apply(Math.round((current + delta) * 10000) / 10000);
        };
        return protectControlDrag(addPressPreview(btn));
    };

    input.oninput = () => {
        if (!liveInput) return;
        const raw = String(input.value).replace(",", ".").trim();
        if (!raw) return;
        const n = Number(raw);
        if (Number.isFinite(n)) onChange(clamp(n));
    };
    const setEditing = (editing) => {
        input.dataset.iamccsNumberEditing = editing ? "true" : "false";
        input.readOnly = !editing;
        input.style.cursor = editing ? "text" : "ew-resize";
    };
    input.ondblclick = (event) => {
        event.preventDefault();
        event.stopPropagation();
        setEditing(true);
        input.focus();
        input.select();
    };
    input.onblur = () => {
        const raw = String(input.value).replace(",", ".").trim();
        if (!raw) {
            input.value = formatStepperValue(value, precision);
            setEditing(false);
            return;
        }
        apply(Number(raw));
        setEditing(false);
    };
    input.onkeydown = (event) => {
        if (event.key === "Enter") {
            event.preventDefault();
            const raw = String(input.value).replace(",", ".").trim();
            if (raw) apply(Number(raw));
            input.blur();
            return;
        }
        if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
        event.preventDefault();
        const current = Number(String(input.value).replace(",", ".")) || 0;
        apply(current + (event.key === "ArrowUp" ? stepValue : -stepValue));
    };
    input.onfocus = () => {
        if (input.dataset.iamccsNumberEditing === "true") input.select();
    };
    attachVerticalStepDrag(
        input,
        () => Number(String(input.value).replace(",", ".")) || 0,
        (next) => apply(next),
        stepValue,
        { min: minValue, max: maxValue }
    );
    protectControlDrag(input);

    wrap.append(makeStep("-", -stepValue), input, makeStep("+", stepValue));
    wrap._iamccsSetValue = (next) => {
        input.value = formatStepperValue(next, precision);
    };
    return protectControlDrag(wrap);
}

function timeControl(value, onChange, frameRate = null, opts = {}) {
    const validFps = Number.isFinite(Number(frameRate)) && Number(frameRate) > 0 ? Number(frameRate) : null;
    const totalDuration = Number(opts.totalDuration || opts.duration || 0);
    const nextSecond = Number(opts.nextSecond);
    const liveInput = opts.liveInput !== false;
    const wrap = document.createElement("div");
    wrap.style.cssText = "display:grid;grid-template-columns:1fr;gap:4px;min-width:104px;";

    const makeMiniStepper = (initial, step, min, apply, inputMode = "decimal", formatter = formatTimeValue, title = "") => {
        const row = document.createElement("div");
        row.style.cssText = "display:grid;grid-template-columns:20px minmax(48px,1fr) 20px;gap:3px;align-items:center;";
        const input = document.createElement("input");
        input.type = "text";
        input.inputMode = inputMode;
        input.value = formatter(initial);
        input.title = title;
        input.style.cssText = inputBase() + "height:25px;padding:3px 4px;text-align:center;font-variant-numeric:tabular-nums;";
        const commit = (raw) => {
            const n = Math.max(min, Number(String(raw).replace(",", ".")) || 0);
            input.value = formatter(n);
            apply(n);
            updateInfo();
        };
        const makeStep = (label, delta) => {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.textContent = label;
            btn.title = delta < 0 ? "Decrease" : "Increase";
            btn.style.cssText = "height:25px;padding:0;border:1px solid #405664;border-radius:4px;background:#0b1116;color:#e8eef2;font-size:12px;line-height:1;cursor:pointer;";
            btn.onclick = (event) => {
                event.preventDefault();
                const current = Number(String(input.value).replace(",", ".")) || 0;
                commit(current + delta);
            };
            return protectControlDrag(btn);
        };
        input.oninput = () => {
            if (!liveInput) return;
            const raw = String(input.value).replace(",", ".").trim();
            if (!raw) return;
            const n = Number(raw);
            if (Number.isFinite(n)) {
                apply(Math.max(min, n));
                updateInfo();
            }
        };
        input.onblur = () => commit(input.value);
        input.onkeydown = (event) => {
            if (event.key === "Enter") {
                event.preventDefault();
                commit(input.value);
                return;
            }
            if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
            event.preventDefault();
            const current = Number(String(input.value).replace(",", ".")) || 0;
            commit(current + (event.key === "ArrowUp" ? step : -step));
        };
        input.onfocus = () => input.select();
        attachVerticalStepDrag(input, () => Number(String(input.value).replace(",", ".")) || 0, commit, step, { min });
        protectControlDrag(input);
        row._iamccsSetValue = (next) => { input.value = formatter(next); };
        row.append(makeStep("-", -step), input, makeStep("+", step));
        return row;
    };

    let currentSeconds = Math.max(0, Number(value) || 0);
    const secondsRow = makeMiniStepper(currentSeconds, 0.1, 0, (next) => {
        currentSeconds = Math.max(0, Number(next) || 0);
        if (frameRow && validFps) frameRow._iamccsSetValue(Math.round(currentSeconds * validFps));
        onChange(currentSeconds);
    }, "decimal", formatTimeValue, "Seconds");

    let frameRow = null;
    if (validFps) {
        frameRow = makeMiniStepper(Math.round(currentSeconds * validFps), 1, 0, (frame) => {
            currentSeconds = Math.max(0, Number(frame) || 0) / validFps;
            secondsRow._iamccsSetValue(currentSeconds);
            onChange(currentSeconds);
        }, "numeric", (frame) => String(Math.max(0, Math.round(Number(frame) || 0))), "Absolute frame");
    }

    const durationLine = document.createElement("div");
    const rangeLine = document.createElement("div");
    const infoCss = "font-size:10px;line-height:1.15;color:#9fb9c7;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-variant-numeric:tabular-nums;border:1px solid rgba(104,138,154,.35);border-radius:3px;background:rgba(5,16,22,.72);padding:2px 4px;";
    durationLine.style.cssText = infoCss;
    rangeLine.style.cssText = infoCss + "color:#7f98a6;";

    function updateInfo() {
        const start = Math.max(0, Number(currentSeconds) || 0);
        const endCandidate = Number.isFinite(nextSecond) ? nextSecond : totalDuration;
        const end = Number.isFinite(endCandidate) && endCandidate > 0 ? Math.max(start, endCandidate) : start;
        const dur = Math.max(0, end - start);
        const durText = formatTimeValue(dur);
        const frameText = validFps ? ` (${Math.round(dur * validFps)}f)` : "";
        durationLine.textContent = `â± ${durText}s${frameText} - segment duration`;
        rangeLine.textContent = `${formatTimeValue(start)}s â†’ ${formatTimeValue(end)}s`;
    }

    updateInfo();
    wrap.append(secondsRow);
    if (frameRow) wrap.append(frameRow);
    wrap.append(durationLine, rangeLine);
    return protectControlDrag(wrap);
}

function oldTimeControlRemoved(value, onChange) {
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
    attachVerticalStepDrag(out, () => Number(String(out.value).replace(",", ".")) || 0, sync, 0.1, { min: 0, max: 1 });
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
        if (row.use_guide && row.force > 0.70 && i > 0 && i < clean.length - 1) warnings.push(`${row.label}: force > 0.70 on a middle guide can pin the frame. Keep strong anchors for first/last unless you want a deliberate lock.`);
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
        { second: 0.0, ref: 1, force: 0.22, label: "ref_1", camera: "continuous dolly-in", transition: "continuous_motion", note: "", use_guide: true },
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
        "image_resize_method",
        "image_multiple_of",
        "img_compression",
        "debug_verbose",
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
        addSetting(label, numberStepperControl(widget?.value ?? "", step, min, null, (value) => {
            setWidgetValue(node, name, value);
            if (name === "duration_seconds" || name === "frame_rate") draw();
        }));
    };
    const selectSetting = (label, name, options) => {
        const widget = getWidget(node, name);
        addSetting(label, makeSelect(String(widget?.value || options[0]), options, (value) => setWidgetValue(node, name, value)));
    };
    numberSetting("Duration", "duration_seconds", "1", "1");
    numberSetting("FPS", "frame_rate", "1", "1");
    selectSetting("Guide mode", "guide_policy", ["every_checked_row", "safe_core_guides"]);
    numberSetting("Max guides", "max_guides", "1", "1");
    numberSetting("Guide gap", "min_guide_gap_seconds", "0.1", "0");
    numberSetting("Default motion", "default_force", "0.1", "1");
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
    const savePackageBtn = liteButton("Save Package");
    const importRefsBtn = liteButton("Import Refs");
    const clearImagesBtn = liteButton("Clear Images", "danger");
    referencesActions.append(addImagesBtn, importBoardBtn, saveBoardBtn, savePackageBtn, importRefsBtn, clearImagesBtn);
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
        const safeDuration = Math.max(0, duration - 1);
        if (total <= 1 || safeDuration <= 0) return 0;
        return Math.round(safeDuration * (index / Math.max(1, total - 1)));
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

        const normalizedRows = rows.map((row, rowIndex) => normalizeLiteRow(row, rowIndex)).map((row) => {
            const ref = Number(row.ref || 1);
            return ref > index + 1 ? { ...row, ref: ref + 1 } : row;
        });
        rows = insertDuplicateRowAfter(
            normalizedRows,
            index,
            normalizeLiteRow,
            () => makeLiteReferenceRow(index + 1, nextPaths.length),
            index + 2
        );
        sync();
        drawReferenceStrip();
        draw();
    }

    function deleteLiteRowAndLinkedReference(rowIndex) {
        const normalized = rows.map((row, index) => normalizeLiteRow(row, index));
        if (rowIndex < 0 || rowIndex >= normalized.length) return;
        const paths = getConnectedReferencePaths(node);
        const removedRef = Number(normalized[rowIndex]?.ref || rowIndex + 1);
        const refUseCount = normalized.filter((row) => Number(row.ref || 1) === removedRef).length;
        normalized.splice(rowIndex, 1);
        rows = normalized;

        if (paths.length && removedRef >= 1 && removedRef <= paths.length && refUseCount <= 1) {
            const nextPaths = paths.slice();
            nextPaths.splice(removedRef - 1, 1);
            setOwnReferencePaths(node, nextPaths);
            rows = rows.map((row, index) => {
                const item = normalizeLiteRow(row, index);
                const ref = Number(item.ref || 1);
                return normalizeLiteRow({
                    ...item,
                    ref: ref > removedRef ? ref - 1 : Math.min(ref, Math.max(1, nextPaths.length || 1)),
                }, index);
            });
            drawReferenceStrip();
        }

        if (!rows.length) rows = [makeLiteReferenceRow(1, Math.max(1, getConnectedReferencePaths(node).length || 1))];
        sync();
        draw();
    }

    function drawReferenceStrip() {
        referencesGrid.innerHTML = "";
        const paths = getConnectedReferencePaths(node);
        const normalizedRows = rows.map((row, rowIndex) => normalizeLiteRow(row, rowIndex));
        const rowByReference = new Map();
        normalizedRows.forEach((row, rowIndex) => {
            const ref = Math.max(1, Math.round(Number(row.ref || rowIndex + 1))) - 1;
            if (!rowByReference.has(ref)) rowByReference.set(ref, row);
        });
        if (!paths.length) {
            const card = document.createElement("div");
            card.style.width = "104px";
            card.style.height = "58px";
            card.style.border = "1px dashed rgba(125, 211, 252, 0.45)";
            card.style.borderRadius = "4px";
            card.style.background = "rgba(3, 20, 31, 0.48)";
            card.style.display = "flex";
            card.style.alignItems = "flex-end";
            card.style.justifyContent = "flex-start";
            card.style.padding = "4px";
            card.style.color = "#9cc7dc";
            card.style.fontSize = "11px";
            card.textContent = "ref 1";
            referencesGrid.appendChild(card);
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
        const highForceMiddle = rows.map(normalizeLiteRow).filter((row, i, list) => i > 0 && i < list.length - 1 && row.use_guide && Number(row.force || 0) > 0.70);
        statusBar.textContent = highForceMiddle.length
            ? `Warnings: ${highForceMiddle.map((row) => `${row.label}: force > 0.70 on a middle guide`).join("  |  ")}`
            : `FLF-only: ${activeGuides} guide rows active. Notes are private and are not sent to PromptRelay.`;

        const header = document.createElement("div");
        header.style.cssText = `display:grid;grid-template-columns:${SHOTBOARD_LITE_ROW_GRID};gap:8px;color:#D8BC80;font-size:10px;font-weight:700;padding:0 4px;`;
        header.innerHTML = "<div></div><div>Time</div><div>Image Ref</div><div>Motion / Lock</div><div>Guide</div><div>Label</div><div>Notes</div><div></div>";
        table.appendChild(header);

        rows.forEach((row, index) => {
            const r = normalizeLiteRow(row, index);
            rows[index] = r;
            const rowId = r._ui_id;
            const updateRow = (patch) => {
                const targetIndex = rows.findIndex((item) => item?._ui_id === rowId);
                if (targetIndex < 0) return;
                rows[targetIndex] = normalizeLiteRow({ ...rows[targetIndex], ...patch }, targetIndex);
                sync();
            };
            const updateRowTime = (value) => {
                rows = shiftFollowingRowsForTimeChange(rows, rowId, value, normalizeLiteRow, 0.1);
                sync();
                draw();
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

            const liteFps = Number(getWidget(node, "frame_rate")?.value || 0);
            const liteDuration = Number(getWidget(node, "duration_seconds")?.value || 0);
            const sec = timeControl(r.second, updateRowTime, liteFps, {
                nextSecond: rows[index + 1]?.second,
                totalDuration: liteDuration,
            });
            const ref = refPicker(r.ref, paths, (value) => { updateRow({ ref: value }); draw(); }, {
                thumbWidth: 152,
                thumbHeight: 92,
                onReplace: (referenceIndex) => openReplaceReferencePicker(referenceIndex),
            });
            // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
            const motion = forceControl(r.force, (value) => updateRow({ force: value, motion_force: value, guide_strength: value, image_lock_strength: value, strength: value }));
            const forceLockCell = document.createElement("div");
            forceLockCell.style.cssText = "display:grid;grid-template-rows:auto auto;gap:6px;min-width:0;";
            const motionLabel = document.createElement("div");
            motionLabel.textContent = "Motion";
            motionLabel.style.cssText = "text-align:center;color:#D8BC80;font-size:9px;font-weight:800;";
            const motionGroup = document.createElement("div");
            motionGroup.style.cssText = "display:grid;gap:2px;";
            motionGroup.append(motionLabel, motion);
            forceLockCell.append(motionGroup);
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
                deleteLiteRowAndLinkedReference(index);
            };
            protectControlDrag(del);

            card.append(handle, sec, ref, forceLockCell, guide, label, notes, del);
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
        pendingReplaceReferenceIndex = null;
        fileInput.value = "";
        replaceFileInput.value = "";
        boardFileInput.value = "";
        clearOwnReferencePaths(node);
        rows = [makeLiteReferenceRow(1, 1, { second: 0, ref: 1, label: "ref_1", note: "", relay_prompt: "", use_prompt: false })];
        sync();
        drawReferenceStrip();
        draw();
    };
    const collectLiteBoard = () => {
        syncLitePromptWidget();
        const currentPrompt = String(promptArea.value || getWidget(node, "global_prompt")?.value || "");
        return {
            metadata: {
                schema: "iamccs.cine.shotboard.lite.board",
                schema_version: 1,
                node: "IAMCCS_CineShotboardLite",
                promptrelay: "disabled",
                cine_ui_version: CINE_VERSION,
                saved_at: new Date().toISOString(),
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
            images: getConnectedReferencePaths(node).map((path, index) => ({
                ref: index + 1,
                path,
                name: String(path).split(/[\\/]/).pop() || `ref_${index + 1}`,
            })),
        };
    };
    saveBoardBtn.onclick = () => {
        const board = collectLiteBoard();
        downloadJsonFile(board, safeBoardFilename("iamccs_cine_shotboard_lite"));
    };
    savePackageBtn.onclick = async () => {
        try {
            savePackageBtn.disabled = true;
            const board = collectLiteBoard();
            await saveShotboardPackageFolder(board, "iamccs_cine_shotboard_lite", (message) => {
                statusBar.textContent = message;
            });
        } catch (err) {
            console.error("[IAMCCS Cine Shotboard Lite] package save failed", err);
            statusBar.textContent = `Save Package failed: ${err?.message || err}`;
        } finally {
            savePackageBtn.disabled = false;
        }
    };
    importBoardBtn.onclick = () => boardFileInput.click();
    const importLiteBoardFile = async (file) => {
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
            if (boardValue("image_resize_method") != null) setWidgetValue(node, "image_resize_method", cineResizeMethodValue(boardValue("image_resize_method")));
            if (boardValue("image_multiple_of") != null) setWidgetValue(node, "image_multiple_of", Number(boardValue("image_multiple_of")));
            if (boardValue("img_compression") != null) setWidgetValue(node, "img_compression", Number(boardValue("img_compression")));
            const paths = packagedReferencePaths(data);
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
    };
    boardFileInput.onchange = async (event) => {
        await importLiteBoardFile(event.target.files?.[0]);
        boardFileInput.value = "";
    };
    const handleLiteBoardDrop = async (event) => {
        const file = Array.from(event.dataTransfer?.files || []).find((item) => /\.json$/i.test(item.name || ""));
        if (!file) return;
        event.preventDefault();
        await importLiteBoardFile(file);
    };
    [root, referencesPanel].forEach((target) => {
        target.addEventListener("dragover", (event) => {
            if (Array.from(event.dataTransfer?.items || []).some((item) => item.kind === "file")) event.preventDefault();
        });
        target.addEventListener("drop", handleLiteBoardDrop);
    });

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

function disposeShotboardProWidget(node) {
    const widget = node?._iamccsCineShotboardWidget;
    const root = node?._iamccsCineShotboardRoot;
    try { root?.remove?.(); } catch {}
    try { widget?.element?.remove?.(); } catch {}
    if (widget && Array.isArray(node?.widgets)) {
        const idx = node.widgets.indexOf(widget);
        if (idx >= 0) node.widgets.splice(idx, 1);
    }
    node._iamccsCineShotboardWidget = null;
    node._iamccsCineShotboardRoot = null;
    node._iamccsCineShotboardReady = false;
    node._iamccsCineShotboardVersion = "";
}

function renderShotboardPro(node) {
    if (node._iamccsCineShotboardReady && node._iamccsCineShotboardVersion === CINE_VERSION) return;
    disposeShotboardProWidget(node);
    node._iamccsCineShotboardReady = true;
    node._iamccsCineShotboardVersion = CINE_VERSION;
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
    hideWidget(getWidget(node, "wan_frame_round_mode"));
    hideWidget(getWidget(node, "tail_safety_frames"));
    hideWidget(getWidget(node, "image_paths"));
    hideWidget(getWidget(node, "image_width"));
    hideWidget(getWidget(node, "image_height"));
    hideWidget(getWidget(node, "image_resize_method"));
    hideWidget(getWidget(node, "image_multiple_of"));
    hideWidget(getWidget(node, "img_compression"));

    const initialSourceRows = parseJsonWidget(node, defaultShotboardRows);
    const initialHasCanonicalRelay = initialSourceRows.some(rowHasCanonicalRelayPrompt);
    const initialHasLegacyNotes = initialSourceRows.some((row) => firstNonEmpty(row?.note, row?.camera_note));
    let rows = initialSourceRows.map((row, index) => normalizeShotboardRow(row, index, {
        useNoteAsRelayFallback: !initialHasCanonicalRelay && initialHasLegacyNotes,
    }));
    if (shotboardV2) {
        rows = rows.map((row) => {
            const transition = String(row.transition || "continuous_motion");
            const activeTransition = transition && transition !== "continuous_motion" && transition !== "off";
            return {
                ...row,
                camera_relay_mode: "off",
                transition_relay_mode: activeTransition ? "safe_only" : "off",
                use_prompt: activeTransition ? true : row.use_prompt,
            };
        });
    }
    const { root, toolbar, table } = tableShell(
        shotboardV2 ? "Cine Shotboard Timeline Pro V2" : "Cine Shotboard Timeline Pro",
        ""
    );
    root.dataset.iamccsCineVersion = CINE_VERSION;
    node._iamccsCineShotboardRoot = root;
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
        "wan_frame_round_mode",
        "tail_safety_frames",
        "image_width",
        "image_height",
        "image_resize_method",
        "image_multiple_of",
        "img_compression",
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
        const control = numberStepperControl(widget?.value ?? "", step, min, null, (value) => {
            setWidgetValue(node, name, value);
            if (name === "duration_seconds" || name === "frame_rate") draw();
        });
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

    numberSetting("Duration", "duration_seconds", "1", "1");
    numberSetting("FPS", "frame_rate", "1", "1");
    selectSetting("Guide policy", "guide_policy", ["safe_core_guides", "prompt_only", "every_checked_row"], applyGuidePolicyToRows);
    numberSetting("Max guides", "max_guides", "1", "0");
    numberSetting("Guide gap", "min_guide_gap_seconds", "0.1", "0");
    numberSetting("Default motion", "default_force", "0.1", "1");
    numberSetting("Relay softness", "promptrelay_epsilon", "0.0001", "0.0001");
    selectSetting("WAN frames", "wan_frame_round_mode", ["up_8n_plus_1", "nearest_8n_plus_1", "none"]);
    numberSetting("Ref width", "image_width", "32", "64");
    numberSetting("Ref height", "image_height", "32", "64");
    selectSetting("Resize", "image_resize_method", ["crop", "pad", "keep proportion", "stretch"]);
    numberSetting("Multiple", "image_multiple_of", "1", "1");
    numberSetting("Compression", "img_compression", "1", "0");
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
    const savePackageBtn = shotButton("Save Package");
    const importRefsBtn = shotButton("Import Refs");
    const clearImagesBtn = shotButton("Clear Images", "danger");
    referencesActions.append(addImagesBtn, importBoardBtn, saveBoardBtn, savePackageBtn, importRefsBtn, clearImagesBtn);
    referencesHead.append(referencesTitle, referencesActions);
    const referencesGrid = document.createElement("div");
    referencesGrid.style.cssText = `display:flex;gap:8px;overflow-x:auto;padding-bottom:2px;min-height:${shotboardV2 ? "104px" : "76px"};`;
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
        const safeDuration = Math.max(0, duration - 1);
        if (total <= 1 || safeDuration <= 0) return 0;
        return Math.round(safeDuration * (index / Math.max(1, total - 1)));
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

        const normalizedRows = rows.map((row, rowIndex) => normalizeShotboardRow(row, rowIndex)).map((row) => {
            const ref = Number(row.ref || 1);
            return ref > index + 1 ? { ...row, ref: ref + 1 } : row;
        });
        const hinted = Number.isFinite(Number(rowIndexHint)) ? Math.floor(Number(rowIndexHint)) : -1;
        const sourceIndex = hinted >= 0 && hinted < normalizedRows.length && Number(normalizedRows[hinted]?.ref || 1) === index + 1
            ? hinted
            : Math.max(0, normalizedRows.findIndex((row) => Number(row.ref || 1) === index + 1));
        rows = insertDuplicateRowAfter(
            normalizedRows,
            sourceIndex,
            normalizeShotboardRow,
            () => makeReferenceRow(index + 1, nextPaths.length),
            index + 2
        );

        sync();
        drawReferenceStrip();
        draw();
    }

    function deleteRowAndLinkedReference(rowIndex) {
        const normalized = rows.map((row, index) => normalizeShotboardRow(row, index));
        if (rowIndex < 0 || rowIndex >= normalized.length) return;
        const paths = getConnectedReferencePaths(node);
        const removedRef = Number(normalized[rowIndex]?.ref || rowIndex + 1);
        const refUseCount = normalized.filter((row) => Number(row.ref || 1) === removedRef).length;
        normalized.splice(rowIndex, 1);
        rows = normalized;

        if (paths.length && removedRef >= 1 && removedRef <= paths.length && refUseCount <= 1) {
            const nextPaths = paths.slice();
            nextPaths.splice(removedRef - 1, 1);
            setOwnReferencePaths(node, nextPaths);
            rows = rows.map((row, index) => {
                const item = normalizeShotboardRow(row, index);
                const ref = Number(item.ref || 1);
                return normalizeShotboardRow({
                    ...item,
                    ref: ref > removedRef ? ref - 1 : Math.min(ref, Math.max(1, nextPaths.length || 1)),
                }, index);
            });
            drawReferenceStrip();
        }

        if (!rows.length) rows = [makeReferenceRow(1, Math.max(1, getConnectedReferencePaths(node).length || 1))];
        sync();
        draw();
    }

    function compactReferencesToUsedRows(referencePaths = getConnectedReferencePaths(node), options = {}) {
        const paths = (Array.isArray(referencePaths) ? referencePaths : []).map((path) => String(path || "").trim()).filter(Boolean);
        const normalized = rows.map((row, index) => normalizeShotboardRow(row, index));
        if (!paths.length || !normalized.length) {
            if (options.applyToNode && !paths.length) clearOwnReferencePaths(node);
            return { paths, rows: normalized, changed: false };
        }

        const oldRefToNewRef = new Map();
        const nextPaths = [];
        for (const row of normalized) {
            const oldRef = Math.round(Number(row.ref || 0));
            if (!Number.isFinite(oldRef) || oldRef < 1 || oldRef > paths.length) continue;
            if (!oldRefToNewRef.has(oldRef)) {
                oldRefToNewRef.set(oldRef, nextPaths.length + 1);
                nextPaths.push(paths[oldRef - 1]);
            }
        }
        if (!nextPaths.length) {
            return { paths, rows: normalized, changed: false };
        }

        const nextRows = normalized.map((row, index) => {
            const oldRef = Math.round(Number(row.ref || 0));
            const nextRef = oldRefToNewRef.get(oldRef);
            return normalizeShotboardRow({
                ...row,
                ref: nextRef || Math.min(Math.max(1, oldRef || 1), nextPaths.length),
            }, index);
        });
        const changed = nextPaths.length !== paths.length
            || nextRows.some((row, index) => Number(row.ref || 1) !== Number(normalized[index]?.ref || 1));
        if (changed && options.applyToNode) {
            rows = nextRows;
            setOwnReferencePaths(node, nextPaths);
            sync();
            drawReferenceStrip();
        }
        return { paths: nextPaths, rows: nextRows, changed };
    }

    function drawReferenceStrip() {
        referencesGrid.innerHTML = "";
        const paths = getConnectedReferencePaths(node);
        const normalizedRows = rows.map((row, rowIndex) => normalizeShotboardRow(row, rowIndex));
        const rowByReference = new Map();
        normalizedRows.forEach((row, rowIndex) => {
            const ref = Math.max(1, Math.round(Number(row.ref || rowIndex + 1))) - 1;
            if (!rowByReference.has(ref)) rowByReference.set(ref, { row, rowIndex });
        });
        const isRelayBridgeRowForStrip = (row) => {
            if (!shotboardV2 || !row) return false;
            const text = String(row.relay_prompt || row.note || "").trim();
            const transition = String(row.transition || "").trim();
            const guideOff = row.use_guide === false || Number(row.force || 0) <= 0;
            return transition === "prompt_relay_text" && guideOff && Boolean(text);
        };
        const relayBridgeAfterReference = (refIndex, sourceRowIndex) => {
            if (!shotboardV2 || !Number.isFinite(Number(sourceRowIndex))) return null;
            const sourceRef = Math.max(1, Number(refIndex || 0) + 1);
            for (let i = Number(sourceRowIndex) + 1; i < normalizedRows.length; i += 1) {
                const candidate = normalizedRows[i];
                if (!candidate) continue;
                if (isRelayBridgeRowForStrip(candidate)) return { row: candidate, rowIndex: i };
                const candidateRef = Math.max(1, Math.round(Number(candidate.ref || 1)));
                const hasImageGuide = candidate.use_guide !== false && Number(candidate.force || 0) > 0;
                const isImageRow = String(candidate.transition || "") !== "prompt_relay_text" && hasImageGuide;
                if (isImageRow && candidateRef !== sourceRef) break;
            }
            return null;
        };
        if (!paths.length) {
            const card = document.createElement("div");
            card.style.width = shotboardV2 ? "152px" : "104px";
            card.style.height = shotboardV2 ? "88px" : "58px";
            card.style.border = "1px dashed rgba(214, 166, 90, 0.48)";
            card.style.borderRadius = "4px";
            card.style.background = "rgba(25, 17, 10, 0.58)";
            card.style.display = "flex";
            card.style.alignItems = "flex-end";
            card.style.justifyContent = "flex-start";
            card.style.padding = "4px";
            card.style.color = "#e8c98f";
            card.style.fontSize = "11px";
            card.textContent = "ref 1";
            referencesGrid.appendChild(card);
            return;
        }
        paths.forEach((path, index) => {
            const card = document.createElement("div");
            card.style.cssText = `position:relative;flex:0 0 ${shotboardV2 ? "164px" : "104px"};height:${shotboardV2 ? "96px" : "72px"};border:1px solid ${CINE_FILM_LAB.borderSoft};background:${CINE_FILM_LAB.field};border-radius:5px;overflow:${shotboardV2 ? "visible" : "hidden"};`;
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
                    "grid-template-rows:repeat(4,1fr)",
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
                        openReferenceFrameEditor(node, index, path, (newPath, data) => {
                            const appended = appendReferencePath(node, newPath);
                            const nextRef = Math.max(1, Number(appended?.refNumber || index + 1));
                            refPreviewBusters.set(String(newPath), String(data?.cache_bust || Date.now()));
                            console.info("[IAMCCS REF STRIP EDIT] saved candidate reference only; timeline truth unchanged", {
                                oldRef: index + 1,
                                newRef: nextRef,
                                candidatePath: newPath,
                                timelineTruthUnchanged: true,
                                savedTo: data?.absolute_path || data?.path || "",
                                appended: Boolean(appended?.appended),
                            });
                            showTimelineNotice(`Reference candidate saved as ref ${nextRef}. Timeline panels unchanged until edited/applied from the panel.`, "warn");
                            writeTimeline({ force: true });
                            drawReferenceStrip();
                            draw();
                        });
                    }),
                    makeRailButton("R", "Replace this reference image without changing row timing", () => {
                        openReplaceReferencePicker(index);
                    }),
                    makeRailButton("D", "Duplicate this reference into the next slot and create a new keyframe", () => {
                        duplicateReferenceAndLinkedRow(index);
                    }),
                    makeRailButton("G", "Grab/export a project-sized copy into IAMCCS_newimages", async () => {
                        try {
                            await grabShotboardReferenceProjectCopy(node, path);
                        } catch (err) {
                            console.error("[IAMCCS Cine Shotboard V2] grab reference failed", err);
                        }
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
            const rowEntryForStripe = rowByReference.get(index) || { row: normalizedRows[index], rowIndex: index };
            const rowForStripe = rowEntryForStripe.row;
            const rowIndexForStripe = Number.isFinite(Number(rowEntryForStripe.rowIndex)) ? Number(rowEntryForStripe.rowIndex) : index;
            const classicTransition = String(rowForStripe?.transition || "continuous_motion");
            const relayTransitionMode = String(rowForStripe?.transition_relay_mode || "off");
            const hasClassicTransition = classicTransition && classicTransition !== "continuous_motion" && classicTransition !== "off";
            const hasRelayTransition = relayTransitionMode !== "off";
            const bridgeText = String(rowForStripe?.step_transition_prompt || rowForStripe?.note || "");
            const stripeRelayBridge = relayBridgeAfterReference(index, rowIndexForStripe);
            const stripeRelayText = String(stripeRelayBridge?.row?.relay_prompt || stripeRelayBridge?.row?.note || "").trim();
            const stripeRelaySeconds = stripeRelayBridge
                ? Math.max(0.1, Number(normalizedRows[(stripeRelayBridge.rowIndex || 0) + 1]?.second || 0) > Number(stripeRelayBridge.row.second || 0)
                    ? Number(normalizedRows[(stripeRelayBridge.rowIndex || 0) + 1].second || 0) - Number(stripeRelayBridge.row.second || 0)
                    : 0)
                : 0;
            const hasStepTransition = Boolean((rowForStripe?.step_transition_enabled || bridgeText.trim()) && String(rowForStripe?.step_transition_type || "off") !== "off");
            const hasStripeBridge = shotboardV2 && index < paths.length - 1;
            const stripeBridgeActive = hasStepTransition || hasClassicTransition || hasRelayTransition || Boolean(bridgeText.trim()) || Boolean(stripeRelayText);
            card.style.zIndex = hasStripeBridge ? "5" : "1";
            if (false && hasStripeBridge) {
                const bridge = document.createElement("div");
                const bridgeLabel = hasStepTransition ? stepTransitionLabel(rowForStripe.step_transition_type) : (hasClassicTransition ? classicTransition.replace(/_/g, " ") : relayTransitionMode.replace(/_/g, " "));
                bridge.title = `${bridgeLabel} to next reference`;
                bridge.innerHTML = `
                    <svg viewBox="0 0 76 34" width="76" height="34" aria-hidden="true">
                        <path d="M5 22 C22 8 48 8 66 21" fill="none" stroke="rgba(0,0,0,.42)" stroke-width="8" stroke-linecap="round"/>
                        <path d="M5 22 C22 8 48 8 66 21" fill="none" stroke="#F0C247" stroke-width="4" stroke-linecap="round"/>
                        <path d="M61 10 L74 23 L55 28" fill="none" stroke="rgba(0,0,0,.42)" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M61 10 L74 23 L55 28" fill="none" stroke="#F0C247" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>`;
                bridge.style.cssText = "position:absolute;right:-48px;top:18px;width:76px;height:34px;display:flex;align-items:center;justify-content:center;z-index:30;pointer-events:none;filter:drop-shadow(0 2px 4px rgba(0,0,0,.52));";
                card.appendChild(bridge);
            }
            card.append(img, badge, controls);
            referencesGrid.appendChild(card);
            if (hasStripeBridge && stripeRelayBridge) {
                const bridgeBlock = document.createElement("div");
                bridgeBlock.dataset.iamccsRole = "relay-bridge-placeholder";
                bridgeBlock.title = `Relay Bridge${stripeRelaySeconds ? ` ${stripeRelaySeconds.toFixed(2)}s` : ""}: ${stripeRelayText}`;
                bridgeBlock.style.cssText = [
                    "flex:0 0 236px",
                    "height:96px",
                    "border:1px solid rgba(143,208,204,.58)",
                    "border-radius:6px",
                    "background:linear-gradient(135deg, rgba(11,39,43,.96), rgba(34,35,30,.96))",
                    "box-shadow:inset 0 0 0 1px rgba(255,255,255,.05)",
                    "display:grid",
                    "grid-template-rows:22px minmax(0,1fr)",
                    "gap:4px",
                    "padding:5px",
                    "box-sizing:border-box",
                    "min-width:0",
                    "overflow:hidden",
                ].join(";");
                const bridgeHead = document.createElement("div");
                bridgeHead.innerHTML = `<span>Relay Bridge</span><span>${stripeRelaySeconds ? `${stripeRelaySeconds.toFixed(1)}s` : "text"}</span>`;
                bridgeHead.style.cssText = "display:grid;grid-template-columns:1fr auto;align-items:center;gap:6px;border-radius:999px;background:rgba(41,132,142,.24);border:1px solid rgba(143,208,204,.48);color:#CFF2EE;font:8px/1 monospace;font-weight:900;letter-spacing:0;text-transform:uppercase;padding:0 7px;";
                const bridgePreview = document.createElement("div");
                bridgePreview.textContent = stripeRelayText || "Relay bridge text";
                bridgePreview.style.cssText = "min-width:0;overflow:hidden;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;white-space:normal;border:1px solid rgba(244,239,230,.28);border-radius:5px;background:#F4EFE6;color:#15130F;font:10px/1.25 'Courier New',monospace;font-weight:700;padding:7px 8px;";
                bridgeBlock.onclick = (event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    const row = table?.children?.[Number(stripeRelayBridge.rowIndex || 0) + 1];
                    try { row?.scrollIntoView?.({ block: "center", behavior: "smooth" }); } catch {}
                };
                referencesGrid.appendChild(bridgeBlock);
                bridgeBlock.append(bridgeHead, bridgePreview);
            }
            if (false && hasStripeBridge) {
                const bridgeBlock = document.createElement("div");
                bridgeBlock.dataset.iamccsRole = "action-bridge-block";
                bridgeBlock.style.cssText = [
                    "flex:0 0 236px",
                    "height:96px",
                    `border:1px solid ${stripeBridgeActive ? "rgba(223,164,81,.72)" : "rgba(118,103,83,.42)"}`,
                    "border-radius:6px",
                    `background:${stripeBridgeActive ? "linear-gradient(135deg, rgba(42,34,25,.95), rgba(22,38,41,.95))" : "linear-gradient(135deg, rgba(30,29,27,.92), rgba(22,26,27,.92))"}`,
                    "box-shadow:inset 0 0 0 1px rgba(255,255,255,.05)",
                    "display:grid",
                    "grid-template-rows:20px 30px minmax(34px,1fr)",
                    "gap:4px",
                    "padding:4px",
                    "box-sizing:border-box",
                    "min-width:0",
                    "overflow:hidden",
                ].join(";");
                const bridgeHead = document.createElement("div");
                bridgeHead.innerHTML = `<span>Action</span><span style="display:flex;align-items:center;gap:2px;min-width:48px;justify-content:center;"><i></i><b></b></span><span>next</span>`;
                bridgeHead.style.cssText = "display:grid;grid-template-columns:1fr 54px 1fr;align-items:center;gap:5px;border-radius:999px;background:rgba(223,164,81,.16);border:1px solid rgba(223,164,81,.38);color:#F4E5C4;font:8px/1 monospace;font-weight:900;letter-spacing:0;text-transform:uppercase;";
                const bridgeLine = bridgeHead.querySelector("i");
                const bridgeArrow = bridgeHead.querySelector("b");
                if (bridgeLine) bridgeLine.style.cssText = `display:block;height:4px;flex:1;border-radius:999px;background:${stripeBridgeActive ? "#F0C247" : "rgba(244,229,196,.42)"};box-shadow:0 1px 2px rgba(0,0,0,.45);`;
                if (bridgeArrow) bridgeArrow.style.cssText = `display:block;width:0;height:0;border-top:6px solid transparent;border-bottom:6px solid transparent;border-left:10px solid ${stripeBridgeActive ? "#F0C247" : "rgba(244,229,196,.42)"};filter:drop-shadow(0 1px 1px rgba(0,0,0,.45));`;
                const bridgeSelect = makeChoiceSelect(rowForStripe.step_transition_type || (stripeBridgeActive ? "action_beat" : "off"), STEP_TRANSITION_OPTIONS, (value) => {
                    const source = normalizeShotboardRow(rows[rowIndexForStripe] || rowForStripe, rowIndexForStripe);
                    rows[rowIndexForStripe] = normalizeShotboardRow({
                        ...source,
                        step_transition_type: value,
                        step_transition_enabled: value !== "off",
                        use_prompt: value !== "off" ? true : source.use_prompt,
                        step_transition_duration: value !== "off" && Number(source.step_transition_duration || 0) <= 0
                            ? defaultStepTransitionSeconds(value, Math.max(0, Number(rows[rowIndexForStripe + 1]?.second || 0) - Number(source.second || 0)))
                            : source.step_transition_duration,
                        step_transition_arrival: value !== "off" ? (source.step_transition_arrival || defaultStepTransitionArrival(value)) : source.step_transition_arrival,
                    }, rowIndexForStripe);
                    sync();
                    drawReferenceStrip();
                    draw();
                });
                bridgeSelect.style.cssText += "height:30px;line-height:30px;padding-top:0;padding-bottom:0;font-size:10px;font-weight:900;text-align:center;";
                const bridgePrompt = document.createElement("textarea");
                bridgePrompt.value = bridgeText;
                bridgePrompt.placeholder = "what happens before next ref...";
                bridgePrompt.rows = 1;
                bridgePrompt.style.cssText = inputBase() + "resize:none;min-height:34px;height:34px;font-size:9px;line-height:1.18;padding:5px 7px;background:#F4EFE6;color:#111;";
                bridgePrompt.oninput = () => {
                    const source = normalizeShotboardRow(rows[rowIndexForStripe] || rowForStripe, rowIndexForStripe);
                    const hasText = Boolean(String(bridgePrompt.value || "").trim());
                    rows[rowIndexForStripe] = normalizeShotboardRow({
                        ...source,
                        note: bridgePrompt.value,
                        step_transition_prompt: bridgePrompt.value,
                        step_transition_enabled: hasText || source.step_transition_enabled,
                        step_transition_type: hasText && source.step_transition_type === "off" ? "action_beat" : source.step_transition_type,
                        use_prompt: hasText || source.use_prompt,
                    }, rowIndexForStripe);
                    sync();
                };
                protectControlDrag(bridgeSelect);
                protectControlDrag(bridgePrompt);
                bridgeBlock.append(bridgeHead, bridgeSelect, bridgePrompt);
                referencesGrid.appendChild(bridgeBlock);
            }
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
        const compact = compactReferencesToUsedRows(getConnectedReferencePaths(node), { applyToNode: true });
        const imagePaths = compact.paths;
        const boardRows = compact.rows;
        const settings = {};
        for (const name of settingNames) {
            settings[name] = getWidget(node, name)?.value ?? null;
        }
        return attachPackageHintToBoard({
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
            rows: boardRows,
            settings,
            duration_seconds: settings.duration_seconds,
            frame_rate: settings.frame_rate,
            guide_policy: settings.guide_policy,
            min_guide_gap_seconds: settings.min_guide_gap_seconds,
            max_guides: settings.max_guides,
            default_force: settings.default_force,
            promptrelay_epsilon: settings.promptrelay_epsilon,
            wan_frame_round_mode: settings.wan_frame_round_mode,
            image_width: settings.image_width,
            image_height: settings.image_height,
            image_paths: imagePaths,
            images: imagePaths.map((path, index) => ({
                ref: index + 1,
                path,
                name: String(path).split(/[\\/]/).pop() || `ref_${index + 1}`,
            })),
        });
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
    const applyBoard = async (data) => {
        const workflowBoard = boardFromWorkflowJson(data);
        const nestedBoard = data?.board && typeof data.board === "object" ? data.board : null;
        const rootLooksLikeBoard = Boolean(
            data?.timeline ||
            data?.timeline_data ||
            data?.image_paths ||
            data?.package ||
            Array.isArray(data?.images) ||
            String(data?.metadata?.schema || data?.schema || "").includes("shotboard")
        );
        const board = (rootLooksLikeBoard ? (nestedBoard || data) : null) || workflowBoard || nestedBoard || data || {};
        if ((!Array.isArray(board.images) || !board.images.length) && Array.isArray(data?.images)) board.images = data.images;
        console.log("[IAMCCS V3 BOARD IMPORT] board source selected", {
            nodeId: node?.id,
            source: rootLooksLikeBoard ? (nestedBoard ? "nested_package_board" : "root_package_board") : (workflowBoard ? "workflow_node" : "root_fallback"),
            workflowNode: Boolean(workflowBoard),
            hasPackage: Boolean(board?.package || data?.package),
            images: Array.isArray(board?.images) ? board.images.length : 0,
            imagePaths: splitReferencePaths(board?.image_paths).length,
        });
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
        const refs = await packagedReferencePathsForImport(board, (message) => {
            if (warnText) {
                warnText.textContent = message;
                warnText.title = message;
            }
        });
        if (refs.length) {
            setOwnReferencePaths(node, refs);
        } else {
            clearOwnReferencePaths(node);
        }

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
            if (refs.length) {
                compactReferencesToUsedRows(refs, { applyToNode: true });
            }
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
    savePackageBtn.onclick = async () => {
        try {
            savePackageBtn.disabled = true;
            const board = collectBoard();
            const firstLabel = rows[0]?.label || "cine_shotboard";
            await saveShotboardPackageFolder(board, firstLabel, (message) => {
                if (warnText) {
                    warnText.textContent = message;
                    warnText.title = message;
                }
            });
        } catch (err) {
            console.error("[IAMCCS Cine Shotboard] package save failed", err);
            if (warnText) {
                warnText.textContent = `Save Package failed: ${err?.message || err}`;
                warnText.title = warnText.textContent;
            }
        } finally {
            savePackageBtn.disabled = false;
        }
    };
    boardFileInput.onchange = async (event) => {
        const file = event.target.files?.[0];
        if (!file) return;
        try {
            const data = await readJsonFile(file);
            await applyBoard(data);
        } catch (err) {
            console.error("[IAMCCS Cine Shotboard] board load failed", err);
            warnText.textContent = `Board load failed: ${err?.message || err}`;
        } finally {
            boardFileInput.value = "";
        }
    };
    const handleBoardDrop = async (event) => {
        const file = Array.from(event.dataTransfer?.files || []).find((item) => /\.json$/i.test(item.name || ""));
        if (!file) return;
        event.preventDefault();
        try {
            const data = await readJsonFile(file);
            await applyBoard(data);
        } catch (err) {
            console.error("[IAMCCS Cine Shotboard] board drop import failed", err);
            warnText.textContent = `Board drop import failed: ${err?.message || err}`;
        }
    };
    [root, referencesPanel].forEach((target) => {
        target.addEventListener("dragover", (event) => {
            if (Array.from(event.dataTransfer?.items || []).some((item) => item.kind === "file")) event.preventDefault();
        });
        target.addEventListener("drop", handleBoardDrop);
    });
    clearImagesBtn.onclick = () => {
        pendingReplaceReferenceIndex = null;
        fileInput.value = "";
        replaceFileInput.value = "";
        boardFileInput.value = "";
        clearOwnReferencePaths(node);
        rows = [makeReferenceRow(1, 1, { second: 0, ref: 1, label: "ref_1", note: "", relay_prompt: "", use_prompt: false })];
        sync();
        drawReferenceStrip();
        draw();
    };

    const addBtn = shotButton("Add Row", "primary");
    const addTextSlotBtn = shotButton(shotboardV2 ? "Add Relay Bridge" : "Add Text Slot", "primary");
    addTextSlotBtn.title = shotboardV2
        ? "Add a dedicated relay bridge row: no image guide, only timed PromptRelay text between two frames."
        : "Add a prompt-only row: no image guide, only a timed PromptRelay text beat.";
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
    toolbar.append(addBtn, addTextSlotBtn, presetSafe, promptOnly, smoothBtn, coreBtn, thumbsBtn, bakeRelayBtn, openEditorBtn);
    toolbar.append(dialogueBtn, clearBtn);

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
        sync();
        draw();
    });
    const relayBulkToggle = makeBulkToggle("Relay", "Toggle the Relay column for every row", () => {
        const normalized = rows.map(normalizeShotboardRow);
        const allOn = normalized.length > 0 && normalized.every((row) => row.use_prompt !== false && String(row.use_prompt).toLowerCase() !== "false");
        rows = normalized.map((row) => ({ ...row, use_prompt: !allOn }));
        sync();
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

    function insertRelayBridgeAfterRow(sourceIndex) {
        const normalized = rows.map(normalizeShotboardRow).sort((a, b) => Number(a.second || 0) - Number(b.second || 0));
        const source = normalized[sourceIndex];
        if (!source) return;
        const next = normalized[sourceIndex + 1] || null;
        const sourceSecond = Math.max(0, Number(source.second || 0));
        const nextSecond = next ? Math.max(sourceSecond + 0.1, Number(next.second || sourceSecond + 2.5)) : sourceSecond + 2.5;
        const gap = Math.max(0.2, nextSecond - sourceSecond);
        const bridgeSeconds = next
            ? Math.max(0.25, Math.min(1.5, gap * 0.4))
            : Math.max(0.5, Math.min(1.5, gap));
        let bridgeStart = next ? Number((nextSecond - bridgeSeconds).toFixed(3)) : Number((sourceSecond + gap).toFixed(3));
        if (next && bridgeStart <= sourceSecond + 0.05) {
            bridgeStart = Number((sourceSecond + Math.max(0.1, gap * 0.5)).toFixed(3));
        }
        const seedText = String(source.step_transition_prompt || source.note || "").trim();
        normalized.splice(sourceIndex + 1, 0, normalizeShotboardRow({
            second: bridgeStart,
            ref: Math.max(1, Number(source.ref || 1)),
            force: 0,
            image_lock_strength: 0,
            use_guide: false,
            use_prompt: true,
            label: `relay_${sourceIndex + 1}_to_${sourceIndex + 2}`,
            camera: "prompt relay text",
            transition: "prompt_relay_text",
            camera_relay_mode: "off",
            transition_relay_mode: "off",
            relay_addon_position: "after",
            note: seedText || "continue the same forward motion toward the next frame",
            relay_prompt: seedText || "continue the same forward motion toward the next frame",
            step_transition_enabled: false,
            step_transition_type: "off",
            step_transition_prompt: "",
        }, sourceIndex + 1));
        rows = normalized.map((row, index) => normalizeShotboardRow(row, index));
        sync();
        drawReferenceStrip();
        draw();
    }

    function draw() {
        table.innerHTML = "";
        const rowGrid = shotboardV2 ? SHOTBOARD_ROW_GRID_V2 : SHOTBOARD_ROW_GRID;
        const header = document.createElement("div");
        header.style.cssText = `display:grid;grid-template-columns:${rowGrid};gap:8px;color:${CINE_FILM_LAB.muted};font-size:11px;font-weight:600;padding:0 6px;box-sizing:border-box;width:100%;max-width:100%;min-width:0;overflow:hidden;`;
        header.innerHTML = `<div></div><div>Time</div><div>Image Ref</div><div>${shotboardV2 ? "Guide force" : "Force / Notes"}</div><div>Guide</div><div>Relay</div><div>Shot controls</div><div>Local prompt</div><div></div>`;
        table.appendChild(header);

        const referencePaths = getConnectedReferencePaths(node);
        const rowCount = Math.max(1, rows.length);
        const expandedRows = rowCount <= 2;
        const rowThumbHeight = 96;
        const rowThumbWidth = 150;
        const notesMinHeight = rowCount === 1 ? 96 : rowCount === 2 ? 84 : 72;
        const localPromptMinHeight = rowCount === 1 ? 104 : rowCount === 2 ? 96 : 88;
        const isV2RelayBridgeRow = (row) => {
            if (!shotboardV2) return false;
            const transition = String(row.transition || "").trim();
            const hasRelayText = Boolean(String(row.relay_prompt || row.note || "").trim());
            const guideOff = row.use_guide === false || Number(row.force || 0) <= 0;
            return transition === "prompt_relay_text" && guideOff && hasRelayText;
        };
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
            const updateRowTime = (value) => {
                rows = shiftFollowingRowsForTimeChange(rows, rowId, value, normalizeShotboardRow, 0.1);
                sync();
                draw();
            };
            const card = document.createElement("div");
            card.style.cssText = `
                display:grid;
                grid-template-columns:${rowGrid};
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
            const proFps = Number(getWidget(node, "frame_rate")?.value || 0);
            const proDuration = Number(getWidget(node, "duration_seconds")?.value || 0);
            const updateRowDuration = (value) => {
                const nextDuration = Math.max(0.1, Number(value) || 0.1);
                const list = rows.map((item, itemIndex) => normalizeShotboardRow(item, itemIndex)).sort((a, b) => Number(a.second || 0) - Number(b.second || 0));
                const targetIndex = list.findIndex((item) => item?._ui_id === rowId);
                if (targetIndex < 0) return;
                const start = Math.max(0, Number(list[targetIndex].second || 0));
                const oldEnd = targetIndex < list.length - 1
                    ? Math.max(start + 0.1, Number(list[targetIndex + 1].second || 0))
                    : Math.max(start + 0.1, Number(getWidget(node, "duration_seconds")?.value || proDuration || start + nextDuration));
                const nextEnd = Number((start + nextDuration).toFixed(3));
                const delta = Number((nextEnd - oldEnd).toFixed(3));
                if (targetIndex < list.length - 1) {
                    for (let i = targetIndex + 1; i < list.length; i += 1) {
                        list[i] = normalizeShotboardRow({
                            ...list[i],
                            second: Number((Number(list[i].second || 0) + delta).toFixed(3)),
                        }, i);
                    }
                    const durationWidget = getWidget(node, "duration_seconds");
                    if (durationWidget && Math.abs(delta) > 0.0005) {
                        const oldDuration = Number(durationWidget.value || proDuration || 0);
                        const lastStart = Math.max(nextEnd, ...list.map((item) => Number(item.second || 0)));
                        const nextTotal = Math.max(lastStart + 0.1, oldDuration + delta);
                        setWidgetValue(node, "duration_seconds", Number(nextTotal.toFixed(3)));
                    }
                } else {
                    const durationWidget = getWidget(node, "duration_seconds");
                    if (durationWidget) setWidgetValue(node, "duration_seconds", Number(nextEnd.toFixed(3)));
                }
                rows = list;
                sync();
                draw();
            };
            const sec = timeControl(r.second, updateRowTime, proFps, {
                nextSecond: rows[index + 1]?.second,
                totalDuration: proDuration,
                liveInput: false,
            });
            const segmentEnd = Number.isFinite(Number(rows[index + 1]?.second)) ? Number(rows[index + 1].second) : proDuration;
            const segmentSeconds = Math.max(0.1, (Number.isFinite(segmentEnd) && segmentEnd > Number(r.second || 0) ? segmentEnd : Number(r.second || 0) + 0.1) - Number(r.second || 0));
            const lenControl = numberStepperControl(segmentSeconds, "0.1", "0.1", null, updateRowDuration, { liveInput: false });
            lenControl.title = "Set this shot/box duration in seconds. Following rows ripple to preserve their spacing.";
            lenControl.style.cssText += "min-width:0;grid-template-columns:20px minmax(42px,1fr) 20px;gap:3px;";
            lenControl.querySelectorAll("button").forEach((button) => {
                button.style.width = "20px";
                button.style.minWidth = "20px";
                button.style.height = "24px";
            });
            const timeCell = document.createElement("div");
            timeCell.style.cssText = "display:flex;flex-direction:column;gap:6px;min-width:0;";
            const lenWrap = document.createElement("label");
            lenWrap.style.cssText = "display:flex;flex-direction:column;gap:3px;color:#9fb9c7;font-size:9px;font-weight:800;min-width:0;";
            const lenLabel = document.createElement("span");
            lenLabel.textContent = "Len seconds";
            lenLabel.style.cssText = "text-align:center;line-height:1;";
            lenWrap.append(lenLabel, lenControl);
            timeCell.append(sec, lenWrap);

            if (isV2RelayBridgeRow(r)) {
                card.style.cssText = `
                    display:grid;
                    grid-template-columns:24px minmax(108px,128px) minmax(170px,220px) minmax(0,1fr) 46px 28px;
                    gap:8px;
                    align-items:stretch;
                    padding:8px 8px 8px 34px;
                    margin-left:28px;
                    border:1px solid rgba(143,208,204,.46);
                    background:linear-gradient(180deg,rgba(12,31,34,.92),rgba(34,36,33,.92));
                    border-radius:6px;
                    box-shadow:inset 0 1px 0 rgba(255,255,255,.06),0 4px 10px rgba(0,0,0,.12);
                    box-sizing:border-box;
                    width:calc(100% - 28px);
                    max-width:calc(100% - 28px);
                    min-width:0;
                    overflow:hidden;
                `;
                const bridgeMeta = document.createElement("div");
                bridgeMeta.style.cssText = "display:flex;flex-direction:column;gap:5px;justify-content:center;min-width:0;";
                const bridgeTitle = document.createElement("div");
                bridgeTitle.textContent = "Relay Bridge";
                bridgeTitle.style.cssText = "display:flex;align-items:center;justify-content:center;height:20px;border-radius:999px;border:1px solid rgba(143,208,204,.58);background:rgba(41,132,142,.22);color:#CFF2EE;font-size:8px;font-weight:900;text-transform:uppercase;";
                const bridgeLabel = document.createElement("input");
                bridgeLabel.value = r.label || `relay_${index + 1}`;
                bridgeLabel.title = "Relay bridge label";
                bridgeLabel.style.cssText = inputBase() + "height:24px;text-align:center;font-size:10px;font-weight:800;";
                bridgeLabel.oninput = () => updateRow({ label: bridgeLabel.value || `relay_${index + 1}` });
                protectControlDrag(bridgeLabel);
                bridgeMeta.append(bridgeTitle, bridgeLabel);

                const bridgePrompt = document.createElement("textarea");
                bridgePrompt.value = r.relay_prompt || r.note || "";
                bridgePrompt.placeholder = "PromptRelay bridge between surrounding frames...";
                bridgePrompt.title = "This text is the actual PromptRelay local prompt for this bridge row.";
                bridgePrompt.style.cssText = inputBase() + "resize:vertical;min-height:62px;line-height:1.32;font-size:12px;padding:8px 16px 8px 10px;scrollbar-gutter:stable;";
                bridgePrompt.oninput = () => {
                    const text = bridgePrompt.value;
                    updateRow({
                        relay_prompt: text,
                        note: text,
                        transition: "prompt_relay_text",
                        force: 0,
                        image_lock_strength: 0,
                        use_guide: false,
                        use_prompt: Boolean(String(text || "").trim()),
                    });
                    drawReferenceStrip();
                };
                protectControlDrag(bridgePrompt);

                const bridgeRelay = checkbox(r.use_prompt !== false, (value) => updateRow({ use_prompt: value }));
                bridgeRelay.title = "Use this relay bridge as a PromptRelay segment";
                const relayWrap = document.createElement("label");
                relayWrap.style.cssText = "display:flex;flex-direction:column;gap:6px;align-items:center;justify-content:center;color:#9fb9c7;font-size:9px;font-weight:900;text-transform:uppercase;";
                const relayText = document.createElement("span");
                relayText.textContent = "Relay";
                relayWrap.append(relayText, bridgeRelay);

                const removeBridge = shotButton("x", "danger");
                removeBridge.style.width = "26px";
                removeBridge.style.height = "30px";
                removeBridge.style.padding = "0";
                removeBridge.style.alignSelf = "center";
                removeBridge.onclick = () => {
                    if (rows.length <= 1) {
                        if (warnText) warnText.textContent = "At least one shot row must remain.";
                        return;
                    }
                    rows.splice(index, 1);
                    sync();
                    drawReferenceStrip();
                    draw();
                };
                protectControlDrag(removeBridge);

                card.append(handle, timeCell, bridgeMeta, bridgePrompt, relayWrap, removeBridge);
                table.appendChild(card);
                return;
            }

            const ref = refPicker(r.ref, referencePaths, (value) => { updateRow({ ref: value }); draw(); }, {
                thumbWidth: rowThumbWidth,
                thumbHeight: rowThumbHeight,
                onEdit: shotboardV2 ? (referenceIndex, path) => {
                    openReferenceFrameEditor(node, referenceIndex, path, (newPath, data) => {
                        const appended = appendReferencePath(node, newPath);
                        const nextRef = Math.max(1, Number(appended?.refNumber || referenceIndex + 1));
                        updateRow({ ref: nextRef });
                        console.info("[IAMCCS ROW REF EDIT] applied edited reference", {
                            row: index,
                            oldRef: referenceIndex + 1,
                            newRef: nextRef,
                            path: newPath,
                            savedTo: data?.absolute_path || data?.path || "",
                            appended: Boolean(appended?.appended),
                        });
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
                onGrab: shotboardV2 ? async (_referenceIndex, path) => {
                    try {
                        await grabShotboardReferenceProjectCopy(node, path);
                    } catch (err) {
                        console.error("[IAMCCS Cine Shotboard V2] grab reference failed", err);
                    }
                } : null,
            });
            // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
            const motion = forceControl(r.force, (value) => updateRow({ force: value, motion_force: value, guide_strength: value, image_lock_strength: value, strength: value }));
            const guide = checkbox(r.use_guide, (value) => updateRow({ use_guide: value }));
            guide.title = "Use as FLF image guide";
            const relay = checkbox(r.use_prompt, (value) => updateRow({ use_prompt: value }));
            relay.title = "Use this row as PromptRelay local prompt segment";

            const label = document.createElement("input");
            label.value = r.label; label.style.cssText = inputBase();
            label.oninput = () => updateRow({ label: label.value });
            protectControlDrag(label);

            const camera = makeSelect(r.camera, CAMERA_OPTIONS, (value) => { updateRow({ camera: value }); });
            const transition = makeSelect(r.transition, TRANSITION_OPTIONS, (value) => {
                const activeTransition = value && value !== "continuous_motion" && value !== "off";
                updateRow({
                    transition: value,
                    use_prompt: activeTransition ? true : r.use_prompt,
                    transition_relay_mode: activeTransition ? "safe_only" : "off",
                });
                drawReferenceStrip();
            });
            const stepSelect = makeChoiceSelect(r.step_transition_type, STEP_TRANSITION_OPTIONS, (value) => {
                const nextPatch = {
                    step_transition_type: value,
                    step_transition_enabled: value !== "off",
                    use_prompt: value !== "off" ? true : r.use_prompt,
                };
                if (value !== "off" && Number(r.step_transition_duration || 0) <= 0) {
                    const nextSecond = Number(rows[index + 1]?.second ?? 0);
                    const gap = nextSecond > Number(r.second || 0) ? nextSecond - Number(r.second || 0) : 0;
                    nextPatch.step_transition_duration = defaultStepTransitionSeconds(value, gap);
                    nextPatch.step_transition_auto_fit = true;
                    nextPatch.step_transition_arrival = defaultStepTransitionArrival(value);
                }
                updateRow({
                    ...nextPatch,
                });
                drawReferenceStrip();
                draw();
            });
            stepSelect.title = shotboardV2 ? "Relay Bridge timing/motion instruction from this row to the next row" : "Step Transition: cinematic instruction from this row to the next row";

            const forceNotesCell = document.createElement("div");
            forceNotesCell.style.cssText = "display:flex;flex-direction:column;gap:8px;min-width:0;";
            const forceWrap = document.createElement("label");
            forceWrap.style.cssText = "display:grid;gap:4px;color:#b6cac3;font-size:10px;font-weight:800;";
            const motionLabel = document.createElement("span");
            motionLabel.textContent = "Motion";
            motionLabel.style.textAlign = "center";
            forceWrap.append(motionLabel, motion);
            forceNotesCell.append(forceWrap);
            const notesBox = document.createElement("textarea");
            notesBox.value = shotboardV2 ? (r.step_transition_prompt || r.note || "") : (r.note || "");
            notesBox.rows = 3;
            notesBox.placeholder = shotboardV2 ? "Relay Bridge to next frame" : "Notes";
            notesBox.title = shotboardV2
                ? "Relay Bridge text: what happens between this row/reference and the next. This is sent to PromptRelay when text is present."
                : "Private shot notes. Notes are never sent to PromptRelay.";
            notesBox.style.cssText = inputBase() + `resize:vertical;min-height:${notesMinHeight}px;line-height:1.32;font-size:11px;padding:8px 15px 8px 9px;scrollbar-gutter:stable;`;
            notesBox.oninput = () => {
                if (shotboardV2) {
                    const hasText = Boolean(String(notesBox.value || "").trim());
                    updateRow({
                        note: notesBox.value,
                        step_transition_prompt: notesBox.value,
                        step_transition_enabled: hasText ? true : r.step_transition_enabled,
                        step_transition_type: hasText && String(r.step_transition_type || "off") === "off" ? "action_beat" : r.step_transition_type,
                        use_prompt: hasText ? true : r.use_prompt,
                    });
                    drawReferenceStrip();
                    return;
                }
                updateRow({ note: notesBox.value });
            };
            protectControlDrag(notesBox);
            if (!shotboardV2) forceNotesCell.appendChild(notesBox);

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
                const transitionValue = String(current.transition || "continuous_motion");
                const bakeSource = normalizeShotboardRow({
                    ...current,
                    camera_relay_mode: "off",
                    transition_relay_mode: transitionValue !== "continuous_motion" && transitionValue !== "off" ? "safe_only" : "off",
                }, index);
                const bakedPrompt = composeRelayPromptPreview(bakeSource, nextRow);
                updateRow({
                    relay_prompt: bakedPrompt,
                    use_prompt: Boolean(String(bakedPrompt || "").trim()),
                    camera_relay_mode: "off",
                    transition_relay_mode: transitionValue !== "continuous_motion" && transitionValue !== "off" ? "safe_only" : "off",
                    relay_modifier_text: "",
                });
                drawReferenceStrip();
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
            localPrompt.oninput = () => {
                updateRow({
                    relay_prompt: localPrompt.value,
                    use_prompt: Boolean(String(localPrompt.value || "").trim()),
                });
            };
            protectControlDrag(localPrompt);
            promptCell.append(promptActions, localPrompt);

            const controlsCell = document.createElement("div");
            controlsCell.style.cssText = "display:grid;grid-template-rows:auto auto;gap:7px;min-width:0;align-content:start;overflow:hidden;";
            const shotControls = document.createElement("div");
            shotControls.style.cssText = `display:grid;grid-template-columns:${shotboardV2 ? "minmax(0,.9fr) minmax(0,1.2fr) minmax(0,1.25fr)" : "minmax(0,.95fr) minmax(0,1fr) minmax(0,1fr)"};gap:6px;min-width:0;`;
            const namedControl = (labelText, control) => {
                const wrap = document.createElement("label");
                wrap.style.cssText = "display:flex;flex-direction:column;gap:3px;color:#a9bac5;font-size:10px;min-width:0;overflow:hidden;";
                const span = document.createElement("span");
                span.textContent = labelText;
                wrap.append(span, control);
                return wrap;
            };
            shotControls.append(
                namedControl("Label", label),
                namedControl("Camera", camera),
                namedControl("Transition", transition)
            );

            const stepEnabled = Boolean(r.step_transition_enabled && String(r.step_transition_type || "off") !== "off");
            const stepPanel = document.createElement("div");
            stepPanel.style.cssText = [
                "display:grid",
                `grid-template-columns:${stepEnabled ? "minmax(250px,.7fr) minmax(240px,1.2fr)" : "minmax(0,1fr)"}`,
                `gap:${stepEnabled ? "9px" : "6px"}`,
                "align-items:stretch",
                `border:1px solid ${stepEnabled ? "rgba(223,164,81,.72)" : "rgba(118,103,83,.36)"}`,
                "border-radius:7px",
                `background:${stepEnabled ? "rgba(73,50,28,.35)" : "rgba(255,255,255,.025)"}`,
                `padding:${stepEnabled ? "9px" : "7px"}`,
                "min-width:0",
                `${shotboardV2 ? "min-height:178px" : "min-height:142px"}`,
            ].join(";");
            const stepLeft = document.createElement("div");
            stepLeft.style.cssText = "display:flex;flex-direction:column;gap:6px;min-width:0;";
            stepSelect.style.height = "32px";
            stepSelect.style.lineHeight = "32px";
            stepSelect.style.paddingTop = "0";
            stepSelect.style.paddingBottom = "0";
            stepSelect.style.borderColor = r.step_transition_enabled ? "rgba(223,164,81,.78)" : "rgba(118,103,83,.42)";
            stepSelect.style.fontWeight = "800";
            const stepPrompt = document.createElement("textarea");
            stepPrompt.value = r.step_transition_prompt || "";
            stepPrompt.placeholder = shotboardV2 ? "Relay bridge toward next row..." : "Bridge note toward next row...";
            stepPrompt.rows = shotboardV2 ? 3 : 1;
            stepPrompt.style.cssText = inputBase() + (shotboardV2
                ? "resize:vertical;min-height:88px;font-size:12px;line-height:1.32;padding:9px 15px 9px 10px;scrollbar-gutter:stable;"
                : "resize:vertical;min-height:42px;font-size:11px;line-height:1.25;padding:7px 15px 7px 9px;");
            stepPrompt.oninput = () => {
                const hasText = Boolean(String(stepPrompt.value || "").trim());
                updateRow({
                    note: shotboardV2 ? stepPrompt.value : r.note,
                    step_transition_prompt: stepPrompt.value,
                    step_transition_type: hasText && r.step_transition_type === "off" ? "action_beat" : r.step_transition_type,
                    step_transition_enabled: hasText ? true : r.step_transition_enabled,
                    use_prompt: hasText ? true : r.use_prompt,
                });
                if (shotboardV2) drawReferenceStrip();
            };
            protectControlDrag(stepPrompt);

            const addonControls = document.createElement("div");
            addonControls.style.cssText = shotboardV2
                ? "display:grid;grid-template-columns:minmax(122px,150px) minmax(0,1fr);gap:7px;align-items:stretch;min-width:0;"
                : "display:grid;grid-template-columns:minmax(118px,140px) minmax(0,1fr);gap:7px;align-items:stretch;min-width:0;";
            const miniControl = (labelText, control) => {
                const wrap = document.createElement("label");
                wrap.style.cssText = "display:flex;flex-direction:column;gap:3px;color:#a9bac5;font-size:9px;line-height:1.1;min-width:0;overflow:hidden;";
                const span = document.createElement("span");
                span.textContent = labelText;
                wrap.append(span, control);
                return wrap;
            };
            const fitProStepDuration = (seconds) => {
                const nextRow = rows[index + 1];
                if (!nextRow || seconds <= 0) return;
                const requiredSecond = Number(r.second || 0) + Number(seconds || 0);
                const nextSecond = Number(nextRow.second || 0);
                if (nextSecond >= requiredSecond) return;
                const delta = Number((requiredSecond - nextSecond).toFixed(3));
                for (let j = index + 1; j < rows.length; j += 1) {
                    rows[j].second = Number((Number(rows[j].second || 0) + delta).toFixed(3));
                }
                const lastSecond = Math.max(...rows.map((item) => Number(item.second || 0)));
                const durationCtl = getWidget(node, "duration_seconds");
                if (durationCtl && Number(durationCtl.value || 0) < lastSecond + 0.1) {
                    durationCtl.value = Number((lastSecond + 0.1).toFixed(3));
                    setWidgetValue(node, "duration_seconds", durationCtl.value);
                }
            };
            const stepSeconds = numberStepperControl(Math.max(0, Number(r.step_transition_duration || 0) || 0), "0.1", "0", null, (value) => {
                const seconds = Math.max(0, Number(value || 0));
                updateRow({
                    step_transition_duration: seconds,
                    step_transition_enabled: seconds > 0 ? true : r.step_transition_enabled,
                    step_transition_type: seconds > 0 && r.step_transition_type === "off" ? "slow_dolly_in" : r.step_transition_type,
                    use_prompt: seconds > 0 ? true : r.use_prompt,
                });
                if ((r.step_transition_auto_fit ?? true) !== false) fitProStepDuration(seconds);
                sync();
                draw();
            }, { liveInput: false });
            stepSeconds.style.gridTemplateColumns = "22px minmax(34px,1fr) 22px";
            stepSeconds.style.gap = "4px";
            stepSeconds.style.minWidth = "0";
            stepSeconds.querySelectorAll("button").forEach((button) => {
                button.style.width = "22px";
                button.style.minWidth = "22px";
            });
            styleValueControls(stepSeconds);
            const stepArrival = makeChoiceSelect(r.step_transition_arrival || "auto", STEP_TRANSITION_ARRIVAL_OPTIONS, (value) => {
                updateRow({ step_transition_arrival: value });
            });
            stepArrival.style.height = "32px";
            stepArrival.style.lineHeight = "32px";
            stepArrival.style.paddingTop = "0";
            stepArrival.style.paddingBottom = "0";
            stepArrival.title = "Where the arrival toward the next row should happen inside the transition";
            const stepAuto = checkbox((r.step_transition_auto_fit ?? true) !== false, (value) => {
                updateRow({ step_transition_auto_fit: value });
                if (value && Number(r.step_transition_duration || 0) > 0) {
                    fitProStepDuration(Number(r.step_transition_duration || 0));
                    sync();
                    draw();
                }
            });
            stepAuto.title = "When transition seconds exceed the current gap, shift following rows later";
            const stepTiming = document.createElement("div");
            stepTiming.style.cssText = "display:grid;grid-template-columns:minmax(78px,1fr) minmax(88px,1fr) 28px;gap:5px;align-items:end;min-width:0;";
            stepTiming.append(miniControl("Seconds", stepSeconds), miniControl("Arrival", stepArrival), miniControl("Fit", stepAuto));
            const addonPosition = makeChoiceSelect(r.relay_addon_position, [
                { value: "after", label: "add-on after" },
                { value: "before", label: "add-on before" },
            ], (value) => { updateRow({ relay_addon_position: value }); });
            addonPosition.title = "Where to place the custom add-on text";
            const modifierBox = document.createElement("textarea");
            modifierBox.value = r.relay_modifier_text || "";
            modifierBox.placeholder = "Custom Relay add-on text, optional";
            modifierBox.rows = 1;
            modifierBox.style.cssText = inputBase() + (shotboardV2
                ? "resize:vertical;min-height:52px;font-size:11px;line-height:1.25;padding:7px 15px 7px 9px;scrollbar-gutter:stable;"
                : "resize:vertical;min-height:66px;font-size:11px;line-height:1.28;padding:8px 15px 8px 9px;");
            modifierBox.oninput = () => { updateRow({ relay_modifier_text: modifierBox.value }); };
            protectControlDrag(modifierBox);
            const bridgeNoteControl = miniControl(shotboardV2 ? "Relay Bridge" : "Bridge note", stepPrompt);
            const addonPositionControl = miniControl("Add-on pos", addonPosition);
            const customRelayControl = miniControl("Custom Relay", modifierBox);
            if (shotboardV2) addonControls.append(addonPositionControl, customRelayControl);
            else addonControls.append(addonPositionControl, bridgeNoteControl);
            stepLeft.append(
                makeStepTransitionHeader(stepEnabled, stepTransitionLabel(r.step_transition_type), "to next row"),
                miniControl("Mode", stepSelect)
            );
            if (stepEnabled) stepLeft.append(stepTiming);
            const stepRight = document.createElement("div");
            stepRight.style.cssText = shotboardV2
                ? "display:grid;grid-template-rows:minmax(88px,auto) auto;gap:8px;min-width:0;align-content:stretch;"
                : "display:grid;grid-template-rows:auto auto;gap:8px;min-width:0;align-content:stretch;";
            if (shotboardV2) stepRight.append(bridgeNoteControl, addonControls);
            else stepRight.append(addonControls, customRelayControl);
            if (stepEnabled) stepPanel.append(stepLeft, stepRight);
            else stepPanel.append(stepLeft);
            if (shotboardV2) {
                const insertRelayAfter = shotButton("Insert Relay After", "primary");
                insertRelayAfter.title = "Create a real PromptRelay bridge row after this frame. The bridge appears as its own block between this frame and the next one.";
                insertRelayAfter.style.minHeight = "32px";
                insertRelayAfter.style.fontSize = "10px";
                insertRelayAfter.onclick = (event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    insertRelayBridgeAfterRow(index);
                };
                protectControlDrag(insertRelayAfter);
                const v2Tools = document.createElement("div");
                v2Tools.style.cssText = "display:grid;grid-template-rows:auto auto;gap:8px;min-width:0;";
                v2Tools.append(shotControls, insertRelayAfter);
                controlsCell.append(v2Tools);
            } else {
                controlsCell.append(shotControls, addonControls);
            }

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
                deleteRowAndLinkedReference(index);
            };
            protectControlDrag(del);

            card.append(handle, timeCell, ref, forceNotesCell, guide, relay, controlsCell, promptCell, del);
            table.appendChild(card);
        });
        sync();
    }

    addBtn.onclick = () => { rows.push({ second: rows.length ? Number(rows[rows.length - 1].second) + 2.5 : 0, ref: rows.length + 1, force: 0.18, use_guide: false, use_prompt: true, label: `shot_${rows.length + 1}`, camera: "continuous dolly-in", transition: "continuous_motion", camera_relay_mode: "off", transition_relay_mode: "off", relay_addon_position: "after", note: "", relay_prompt: "describe the motion beat" }); draw(); };
    addTextSlotBtn.onclick = () => {
        const currentRefs = getConnectedReferencePaths(node);
        const last = rows.length ? normalizeShotboardRow(rows[rows.length - 1], rows.length - 1) : null;
        rows.push({
            second: last ? Number(last.second || 0) + 2.5 : 0,
            ref: Math.max(1, Math.min(Number(last?.ref || 1), Math.max(1, currentRefs.length || 1))),
            force: 0,
            image_lock_strength: 0,
            use_guide: false,
            use_prompt: true,
            label: `text_${rows.length + 1}`,
            camera: "prompt relay text",
            transition: "prompt_relay_text",
            camera_relay_mode: "off",
            transition_relay_mode: "off",
            relay_addon_position: "after",
            note: "",
            relay_prompt: "describe the timed action beat",
            step_transition_enabled: false,
            step_transition_type: "off",
            step_transition_prompt: "",
        });
        sync();
        draw();
    };
    presetSafe.onclick = () => { rows = shotboardTemplateRows(); draw(); };
    promptOnly.onclick = () => {
        const sourceRows = rows.length ? rows : shotboardTemplateRows();
        rows = sourceRows.map((row, i) => normalizeShotboardRow({
            ...row,
            force: i === 0 ? 0.65 : 0,
            use_guide: i === 0,
            use_prompt: true,
        }, i));
        draw();
    };
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
    clearBtn.onclick = () => {
        promptArea.value = "";
        syncProPromptWidget();
        rows = [makeReferenceRow(1, Math.max(1, getConnectedReferencePaths(node).length))];
        draw();
    };

    drawReferenceStrip();
    draw();
    const widget = node.addDOMWidget("Cine Shotboard Pro", "iamccs_cine_shotboard_pro", root, { serialize: false });
    node._iamccsCineShotboardWidget = widget;
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
    body.textContent = "Audio/shot sequencer planner. It creates one music-video shot: image prompt for Z-Image/Flux, PromptRelay local beats, WAN frame counts and music_linx for the future CINE_VIDEOCLIP_1 backend.";
    body.style.cssText = `color:${CINE_FILM_LAB.muted};white-space:normal;margin-bottom:7px;`;
    const flow = document.createElement("div");
    flow.textContent = "Use: shot_index -> image generator -> WAN I2V. For a full videoclip, iterate shot_index and concatenate rendered clips.";
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

function disposeShotboardV3Widget(node) {
    const widget = node?._iamccsCineShotboardV3Widget;
    const root = node?._iamccsCineShotboardV3Root;
    try { root?.remove?.(); } catch {}
    try { widget?.element?.remove?.(); } catch {}
    if (widget && Array.isArray(node?.widgets)) {
        const idx = node.widgets.indexOf(widget);
        if (idx >= 0) node.widgets.splice(idx, 1);
    }
    node._iamccsCineShotboardV3Widget = null;
    node._iamccsCineShotboardV3Root = null;
}

function renderShotboardV3(node) {
    if (node._iamccsCineShotboardV3Ready && node._iamccsCineShotboardV3Version === CINE_VERSION) return;
    disposeShotboardV3Widget(node);
    node._iamccsCineShotboardV3Ready = true;
    node._iamccsCineShotboardV3Version = CINE_VERSION;
    const isWanEdition = isWanShotboardV3Class(nodeClassName(node));
    const chrome = applyCineChrome(node, isWanEdition ? "wanShotboardV3" : "shotboardV3");
    [
        "timeline_data",
        "global_prompt",
        "duration_seconds",
        "frame_rate",
        "guide_policy",
        "min_guide_gap_seconds",
        "max_guides",
        "default_force",
        "promptrelay_epsilon",
        "wan_frame_round_mode",
        "image_paths",
        "image_width",
        "image_height",
        "image_resize_method",
        "image_multiple_of",
        "img_compression",
    ].forEach((name) => hideWidget(getWidget(node, name)));

    const purple = isWanEdition ? {
        bg: "#061923",
        panel: "linear-gradient(180deg,#0F3140 0%,#071D28 100%)",
        panel2: "linear-gradient(180deg,#143F52 0%,#0A2634 100%)",
        border: "#2D7A92",
        borderSoft: "#1B4D63",
        text: "#EAF8FF",
        muted: "#B4D4DF",
        image: "#6FB6D2",
        image2: "linear-gradient(180deg,#6FB6D2 0%,#244F67 100%)",
        textBlock: "linear-gradient(180deg,#111111 0%,#000000 100%)",
        audio: "linear-gradient(180deg,#D7A84D 0%,#75582A 100%)",
        danger: "#D56B5C",
        button: "linear-gradient(180deg,#17445C 0%,#0B2838 100%)",
        buttonHover: "linear-gradient(180deg,#215B75 0%,#10384D 100%)",
        buttonPress: "linear-gradient(180deg,#FFE08A 0%,#E08B3E 100%)",
        play: "#FFE08A",
        accent: "#8DE7FF",
        warm: "#F3B34B",
        valueBg: "#F4EFE7",
        valueText: "#181512",
        relay: "#000000",
        guide: "#92C46E",
        imageAccent: "#6FB6D2",
        audioAccent: "#D7A84D",
    } : {
        bg: "#1E2022",
        panel: "#2A2D30",
        panel2: "#34383C",
        border: "#6B6258",
        borderSoft: "#4A4742",
        text: "#F7F2EA",
        muted: "#D2C7BA",
        image: "#3F7781",
        image2: "#2E515A",
        textBlock: "#4B4037",
        audio: "#40614F",
        danger: "#B55245",
        button: "#3B3936",
        buttonHover: "#4B4640",
        play: "#D89B45",
        accent: "#55B8B2",
        warm: "#C96F32",
        valueBg: "#F4EFE7",
        valueText: "#181512",
    };

    const styleValueControls = (element) => {
        if (element?.matches?.("input,select")) {
            element.style.background = purple.valueBg;
            element.style.color = purple.valueText;
            element.style.borderColor = purple.border;
            element.style.textAlign = "center";
            element.style.fontWeight = "800";
        }
        element?.querySelectorAll?.("input,select")?.forEach((input) => {
            input.style.background = purple.valueBg;
            input.style.color = purple.valueText;
            input.style.borderColor = purple.border;
            input.style.textAlign = "center";
            input.style.fontWeight = "800";
        });
        element?.querySelectorAll?.("button")?.forEach((button) => {
            button.style.background = purple.button;
            button.style.color = purple.text;
            button.style.borderColor = purple.border;
            addPressPreview(button, {
                pressedBg: purple.buttonPress || "linear-gradient(180deg,#FFE08A,#E08B3E)",
                pressedColor: purple.valueText || "#102018",
                pressedBorder: purple.accent || "#D6E879",
            });
        });
        return element;
    };

    const root = document.createElement("div");
    root.dataset.iamccsCineVersion = CINE_VERSION;
    node._iamccsCineShotboardV3Root = root;
    installRootPressFeedback(root, purple);
    root.style.cssText = [
        "width:100%",
        "box-sizing:border-box",
        "padding:9px",
        `border:1px solid ${purple.border}`,
        "border-radius:7px",
        `background:${purple.bg}`,
        `color:${purple.text}`,
        "font:12px Arial,sans-serif",
        `max-height:${SHOTBOARD_V3_OPEN_HEIGHT}px`,
        "overflow-y:auto",
        "overflow-x:hidden",
        "scrollbar-gutter:stable",
        `box-shadow:inset 0 1px 0 ${chrome.glow},0 0 0 1px rgba(0,0,0,.45)`,
        "pointer-events:auto",
        "contain:layout paint style",
        "content-visibility:auto",
        "contain-intrinsic-size:1920px 900px",
        "transform:translateZ(0)",
    ].join(";");

    const promptWidget = getWidget(node, "global_prompt");
    const timelineWidget = getWidget(node, "timeline_data");
    const durationWidget = getWidget(node, "duration_seconds");
    const fpsWidget = getWidget(node, "frame_rate");
    const defaultForceWidget = getWidget(node, "default_force");
    const debugVerboseWidget = getWidget(node, "debug_verbose");
    const imageWidthWidget = getWidget(node, "image_width");
    const imageHeightWidget = getWidget(node, "image_height");
    const removeStaleV3MultiInput = () => {
        const slot = findInputSlotByName(node, ["multi_input"]);
        if (slot < 0 || !Array.isArray(node.inputs)) return;
        try { node.disconnectInput?.(slot); } catch {}
        node.inputs.splice(slot, 1);
        try { node.setDirtyCanvas?.(true, true); app.graph?.setDirtyCanvas?.(true, true); } catch {}
    };
    removeStaleV3MultiInput();
    const v3SettingNames = [
        "duration_seconds",
        "frame_rate",
        "default_force",
        "promptrelay_epsilon",
        "wan_frame_round_mode",
        "image_width",
        "image_height",
        "image_resize_method",
        "image_multiple_of",
        "img_compression",
        "guide_policy",
        "min_guide_gap_seconds",
        "max_guides",
    ];
    const clearV3BoardTransientState = () => {
        node.properties = node.properties || {};
        [
            "iamccs_v3_imported_board",
            "iamccs_v3_imported_board_backup",
            "iamccs_v3_imported_timeline_data",
            "iamccs_v3_imported_settings",
            "iamccs_v3_board",
            "iamccs_v3_board_settings",
            "iamccs_v3_board_data",
        ].forEach((key) => {
            if (Object.prototype.hasOwnProperty.call(node.properties, key)) delete node.properties[key];
        });
    };
    const v3SettingsSnapshot = () => Object.fromEntries(v3SettingNames
        .filter((name) => getWidget(node, name))
        .map((name) => {
            const value = getWidget(node, name)?.value;
            return [name, name === "image_resize_method" ? cineResizeMethodValue(value) : value];
        }));
    clearV3BoardTransientState();
    if (defaultForceWidget) defaultForceWidget.value = clampGuideStrength(defaultForceWidget.value || 1);
    let collapsed = Boolean(node.properties?.iamccs_v3_collapsed);
    let promptTextScale = Math.max(0.85, Math.min(1.55, Number(node.properties?.iamccs_v3_prompt_text_scale || 1)));
    const promptFontSize = (base) => `${Math.max(8, Math.round(Number(base || 10) * promptTextScale * 10) / 10)}px`;
    const storedTimelineMeter = Number(node.properties?.iamccs_v3_timeline_meter_seconds);
    const timelineMeterWasUserSet = Boolean(node.properties?.iamccs_v3_timeline_meter_user_set);
    let timelineMeterSeconds = timelineMeterWasUserSet && Number.isFinite(storedTimelineMeter)
        ? Math.max(0.5, storedTimelineMeter)
        : Math.max(0.5, Number(durationWidget?.value || 20) || 20);
    let selectedId = null;
    let timelineExtraH = 0; // extra px added by user timeline-height resize drag
    let _tlResizeDragStartY = null; // pointer Y at drag start
    let _tlResizeDragStartExtra = 0; // timelineExtraH value at drag start
    let pendingImageInsertFrame = null;
    let pendingImageTargetId = null;
    let timelineNotice = null;
    let timelineNoticeUntil = 0;
    let lastDefaultForce = clampGuideStrength(defaultForceWidget?.value || 1);
    let durationValueControl = null;

    const positiveNumberOrNull = (value) => {
        const parsed = Number(value);
        return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
    };
    const objectDurationTruth = (source) => {
        if (!source || typeof source !== "object") return null;
        const settings = source.settings && typeof source.settings === "object" ? source.settings : {};
        return positiveNumberOrNull(
            source.duration_seconds ??
            source.durationSeconds ??
            source.duration ??
            settings.duration_seconds ??
            settings.durationSeconds ??
            settings.duration
        );
    };
    const objectFpsTruth = (source) => {
        if (!source || typeof source !== "object") return null;
        const settings = source.settings && typeof source.settings === "object" ? source.settings : {};
        return positiveNumberOrNull(
            source.frame_rate ??
            source.frameRate ??
            source.fps ??
            settings.frame_rate ??
            settings.frameRate ??
            settings.fps
        );
    };
    const getDuration = () => Math.max(0.1, Number(durationWidget?.value || 20));
    const getFps = () => Math.max(1, Math.round(Number(fpsWidget?.value || 24)));
    const getTotalFrames = () => Math.max(1, Math.round(getDuration() * getFps()));
    function clampGuideStrength(value, fallback = 1) {
        const parsed = Number(value);
        return Math.max(1, Math.min(2, Number.isFinite(parsed) ? parsed : Number(fallback) || 1));
    }
    const clampTimelineMeterSeconds = (value = timelineMeterSeconds) => {
        const duration = Math.max(0.5, getDuration());
        const rounded = Math.round((Number(value) || duration) * 2) / 2;
        return Math.max(duration, Math.min(Math.max(duration * 3, duration + 0.5), rounded));
    };
    const defaultLen = () => Math.max(1, Math.round(getFps() * 3));
    const newId = (prefix) => `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;

    const timelinePromptStats = (data) => {
        const prompts = [];
        if (data && typeof data === "object") {
            if (Array.isArray(data.segments)) {
                data.segments.forEach((seg) => {
                    const prompt = String(seg?.prompt || "").trim();
                    if (prompt) prompts.push(prompt);
                });
            }
            if (Array.isArray(data.rows)) {
                data.rows.forEach((row) => {
                    const prompt = String(row?.relay_prompt ?? row?.local_prompt ?? row?.prompt ?? "").trim();
                    if (prompt) prompts.push(prompt);
                });
            }
            for (const key of ["director_local_prompts", "local_prompts"]) {
                const value = String(data[key] || "").trim();
                if (value) value.split("|").forEach((part) => {
                    const prompt = String(part || "").trim();
                    if (prompt) prompts.push(prompt);
                });
            }
        }
        const joined = prompts.join(" | ");
        return {
            promptCount: prompts.length,
            containsCoastline: /\bcoastline\b/i.test(joined),
            firstPrompt: prompts[0] || "",
        };
    };

    function readTimeline() {
        try {
            const candidateRevision = (value) => Math.max(0, Number(
                value?.truth_revision ??
                value?._iamccs_v3_truth_revision ??
                value?.metadata?.truth_revision ??
                0
            ) || 0);
            const candidates = [];
            const widgetText = String(timelineWidget?.value || "").trim();
            const backupText = String(node.properties?.iamccs_v3_timeline_data_backup || "").trim();
            let widgetCandidate = null;
            let backupCandidate = null;
            if (widgetText) {
                try {
                    const value = JSON.parse(widgetText);
                    widgetCandidate = { source: "widget.timeline_data", textLength: widgetText.length, value, revision: candidateRevision(value), stats: timelinePromptStats(value) };
                    candidates.push(widgetCandidate);
                } catch {}
            }
            if (backupText) {
                try {
                    const value = JSON.parse(backupText);
                    backupCandidate = { source: "properties.iamccs_v3_timeline_data_backup", textLength: backupText.length, value, revision: candidateRevision(value), stats: timelinePromptStats(value) };
                    candidates.push(backupCandidate);
                } catch {}
            }
            // The visible workflow widget is the current board truth.
            // The property backup is only a recovery fallback for empty/invalid widgets.
            const selected = widgetCandidate || backupCandidate || { source: "empty", value: {}, revision: 0 };
            const data = selected.value;
            console.log("[IAMCCS V3 TIMELINE LOAD]", {
                selected: selected.source,
                truth: selected.source === "widget.timeline_data" ? "node.timeline_data_current_widget" : selected.source === "properties.iamccs_v3_timeline_data_backup" ? "backup_fallback_widget_empty_or_invalid" : "empty",
                revision: Number(selected.revision || 0),
                nodeId: node?.id,
                candidates: candidates.map((candidate) => ({
                    source: candidate.source,
                    textLength: candidate.textLength,
                    revision: Number(candidate.revision || 0),
                    hasSegments: Array.isArray(candidate.value?.segments),
                    segmentCount: Array.isArray(candidate.value?.segments) ? candidate.value.segments.length : 0,
                    rowCount: Array.isArray(candidate.value?.rows) ? candidate.value.rows.length : 0,
                    promptCount: candidate.stats?.promptCount || 0,
                    containsCoastline: Boolean(candidate.stats?.containsCoastline),
                    firstPrompt: String(candidate.stats?.firstPrompt || "").slice(0, 220),
                })),
            });
            if (data && typeof data === "object") {
                return {
                    schema: data.schema || "iamccs.cine.filmmaker_timeline",
                    schema_version: Number(data.schema_version || 1),
                    segments: Array.isArray(data.segments) ? data.segments : [],
                    rows: Array.isArray(data.rows) ? data.rows : [],
                    audioSegments: Array.isArray(data.audioSegments) ? data.audioSegments : [],
                    audioTrackCount: Math.max(1, Number(data.audioTrackCount || 1)),
                    masterAudioGain: Math.max(0, Math.min(2, Number(data.masterAudioGain ?? data.master_audio_gain ?? 1) || 1)),
                    masterAudioNormalize: Boolean(data.masterAudioNormalize || data.master_audio_normalize),
                    duration_seconds: objectDurationTruth(data),
                    frame_rate: objectFpsTruth(data),
                    audioSyncMode: String(data.audioSyncMode || "timeline_audio"),
                    generationStrategy: String(data.generationStrategy || "single_timeline"),
                    flfrealMode: String(data.flfrealMode || data.flfreal_mode || "iamccs_enhanced"),
                    globalPromptOnly: Boolean(data.global_prompt_only || data.use_global_prompt_only),
                    verboseLog: data.verbose_log !== undefined ? Boolean(data.verbose_log) : data.verboseLog !== undefined ? Boolean(data.verboseLog) : true,
                    multiGeneration: data.multiGeneration && typeof data.multiGeneration === "object" ? data.multiGeneration : {},
                    truthRevision: candidateRevision(data),
                };
            }
        } catch {}
        return { schema: "iamccs.cine.filmmaker_timeline", schema_version: 1, segments: [], rows: [], audioSegments: [], audioTrackCount: 1, duration_seconds: null, frame_rate: null, audioSyncMode: "timeline_audio", generationStrategy: "single_timeline", flfrealMode: "iamccs_enhanced", globalPromptOnly: false, verboseLog: true };
    }

    let timeline = readTimeline();
    node._iamccsCineShotboardV3LastTimelineText = String(timelineWidget?.value || "");
    const refPreviewBusters = new Map();
    const refPaths = () => getConnectedReferencePaths(node);
    const normalizeReferencePathKey = (value) => String(value || "").replace(/\\/g, "/").trim();
    const referencePathBasename = (value) => normalizeReferencePathKey(value).split("/").pop();
    const sameReferencePath = (a, b) => {
        const left = normalizeReferencePathKey(a);
        const right = normalizeReferencePathKey(b);
        if (!left || !right) return false;
        return left === right || referencePathBasename(left) === referencePathBasename(right);
    };
    const referenceIndexForPath = (path, paths = refPaths()) => {
        const index = (paths || []).findIndex((item) => sameReferencePath(item, path));
        return index >= 0 ? index + 1 : 0;
    };
    const setSegmentReference = (seg, refNumber, options = {}) => {
        if (!isTimelineImageSegment(seg)) return;
        const paths = refPaths();
        const nextRef = Math.max(1, Math.round(Number(refNumber || 1)));
        seg.ref = nextRef;
        const nextPath = String(paths[nextRef - 1] || "").trim();
        if (nextPath) {
            seg.imageFile = nextPath;
            seg.path = nextPath;
            seg.imageTruthPath = nextPath;
            seg.imageTruthRef = nextRef;
            seg.imageTruthPinned = options.pinned !== false;
            seg.imageTruthSource = options.truthSource || "set_segment_reference";
            delete seg.image_file;
            delete seg.image_truth_path;
            if (options.updateAutoLabel !== false && /^ref_\d+$/i.test(String(seg.label || ""))) seg.label = `ref_${nextRef}`;
        } else {
            delete seg.imageFile;
            delete seg.image_file;
            delete seg.path;
            delete seg.imageTruthPath;
            delete seg.image_truth_path;
            delete seg.imageTruthRef;
            delete seg.imageTruthPinned;
        }
    };
    const applyEditedReferenceTruth = (seg, refNumber, newPath, options = {}) => {
        if (!seg || !isTimelineImageSegment(seg)) return false;
        setSegmentReference(seg, refNumber, { ...options, pinned: true, truthSource: options.truthSource || "timeline_panel_edit" });
        const cleanPath = String(newPath || "").trim();
        if (cleanPath) {
            seg.imageFile = cleanPath;
            seg.path = cleanPath;
            seg.imageTruthPath = cleanPath;
            seg.imageTruthRef = Math.max(1, Math.round(Number(refNumber || seg.ref || 1)));
            seg.imageTruthPinned = true;
            seg.imageTruthSource = options.truthSource || "timeline_panel_edit";
            seg.imageTruthUpdatedAt = new Date().toISOString();
            delete seg.image_file;
            delete seg.image_truth_path;
        }
        return true;
    };
    const applyEditedReferenceToMatches = (oldRef, oldPath, newRef, newPath, options = {}) => {
        const oldRefNumber = Math.max(1, Math.round(Number(oldRef || 1)));
        const cleanOldPath = String(oldPath || "").trim();
        let count = 0;
        for (const item of (timeline.segments || [])) {
            if (!isTimelineImageSegment(item)) continue;
            const itemPath = segmentReferencePath(item);
            const refMatches = Math.round(Number(item.ref || 0)) === oldRefNumber;
            const pathMatches = cleanOldPath && sameReferencePath(itemPath, cleanOldPath);
            if (refMatches || pathMatches) {
                if (applyEditedReferenceTruth(item, newRef, newPath, options)) count += 1;
            }
        }
        return count;
    };
    // Fix 2 (auto-fill): removed Math.max(1,...) clamping — ref:0 (new empty slots) now correctly returns "" instead of the first board image — By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    const segmentReferencePath = (seg) => {
        const truthPath = String(seg?.imageTruthPath || seg?.image_truth_path || "").trim();
        if (truthPath) return truthPath;
        const explicitPath = String(seg?.imageFile || seg?.image_file || seg?.path || "").trim();
        if (explicitPath) return explicitPath;
        const refNumber = Math.round(Number(seg?.ref || 0));
        return refNumber >= 1 ? String(refPaths()[refNumber - 1] || "") : "";
    };
    const normalizeTimelineSegmentReferences = () => {
        const paths = refPaths();
        let changed = false;
        for (const seg of (timeline.segments || [])) {
            if (!isTimelineImageSegment(seg)) continue;
            const explicitPath = String(seg.imageTruthPath || seg.image_truth_path || seg.imageFile || seg.image_file || seg.path || "").trim();
            if (explicitPath) {
                const refFromPath = referenceIndexForPath(explicitPath, paths);
                if (refFromPath && Math.round(Number(seg.ref || 0)) !== refFromPath) {
                    console.warn("[IAMCCS V3 REF SYNC] image truth/ref mismatch fixed", {
                        segmentId: seg.id,
                        label: seg.label,
                        oldRef: seg.ref,
                        newRef: refFromPath,
                        imageTruthPath: explicitPath,
                        oldRefPath: paths[Math.max(0, Math.round(Number(seg.ref || 1)) - 1)] || "",
                    });
                    seg.ref = refFromPath;
                    if (/^ref_\d+$/i.test(String(seg.label || ""))) seg.label = `ref_${refFromPath}`;
                    changed = true;
                }
                if (seg.imageFile !== explicitPath || seg.path !== explicitPath || seg.imageTruthPath !== explicitPath || !seg.imageTruthPinned) {
                    seg.imageFile = explicitPath;
                    seg.path = explicitPath;
                    seg.imageTruthPath = explicitPath;
                    seg.imageTruthRef = refFromPath || Math.max(1, Math.round(Number(seg.ref || 1)));
                    seg.imageTruthPinned = true;
                    seg.imageTruthSource = seg.imageTruthSource || "normalized_existing_segment";
                    delete seg.image_file;
                    delete seg.image_truth_path;
                    changed = true;
                }
                continue;
            }
            const refNumber = Math.max(1, Math.round(Number(seg.ref || 1)));
            const path = String(paths[refNumber - 1] || "").trim();
            if (path) {
                seg.imageFile = path;
                seg.path = path;
                seg.imageTruthPath = path;
                seg.imageTruthRef = refNumber;
                seg.imageTruthPinned = true;
                seg.imageTruthSource = seg.imageTruthSource || "normalized_ref_path";
                delete seg.image_file;
                delete seg.image_truth_path;
                changed = true;
            }
        }
        return changed;
    };
    const endOfSegments = (items) => (items || []).reduce((max, item) => Math.max(max, Number(item.start || 0) + Number(item.length || 1)), 0);
    const endOfVisualSegments = () => endOfSegments((timeline.segments || []).filter((seg) => String(seg.type || "image") !== "audio"));
    const showTimelineNotice = (message, tone = "warn") => {
        if (!timelineNotice) return;
        if (!message && Date.now() < timelineNoticeUntil) return;
        if (message) timelineNoticeUntil = Date.now() + 3500;
        timelineNotice.textContent = message || "";
        timelineNotice.style.display = message ? "block" : "none";
        timelineNotice.style.borderColor = tone === "error" ? purple.danger : purple.play;
        timelineNotice.style.color = tone === "error" ? "#FFE3DD" : "#FFF1BE";
    };
    const setDurationSeconds = (seconds, reason = "manual") => {
        const next = Math.max(0.1, Number(seconds) || 0.1);
        if (durationWidget) durationWidget.value = next;
        setWidgetValue(node, "duration_seconds", next);
        timeline.duration_seconds = next;
        timelineMeterSeconds = clampTimelineMeterSeconds(timelineMeterSeconds);
        durationValueControl?._iamccsSetValue?.(next);
        console.log("[IAMCCS V3 DURATION TRUTH]", { nodeId: node?.id, reason, duration_seconds: next, fps: getFps() });
    };
    const setFrameRateValue = (fps, reason = "manual") => {
        const next = Math.max(1, Math.round(Number(fps) || 24));
        if (fpsWidget) fpsWidget.value = next;
        setWidgetValue(node, "frame_rate", next);
        timeline.frame_rate = next;
        timelineMeterSeconds = clampTimelineMeterSeconds(timelineMeterSeconds);
        console.log("[IAMCCS V3 FPS TRUTH]", { nodeId: node?.id, reason, frame_rate: next, duration_seconds: getDuration() });
    };
    const syncTimingWidgetsFromTimelineTruth = (reason = "timeline_truth") => {
        const durationTruth = objectDurationTruth(timeline);
        const fpsTruth = objectFpsTruth(timeline);
        if (durationTruth !== null) setDurationSeconds(durationTruth, reason);
        if (fpsTruth !== null) setFrameRateValue(fpsTruth, reason);
    };
    syncTimingWidgetsFromTimelineTruth("initial_timeline_load");
    const enforceDurationMinimum = () => {
        const fps = getFps();
        const minFrames = endOfVisualSegments();
        if (!minFrames) {
            showTimelineNotice("");
            return false;
        }
        const currentFrames = Math.round(getDuration() * fps);
        if (currentFrames >= minFrames) {
            showTimelineNotice("");
            return false;
        }
        const minSeconds = Number((minFrames / fps).toFixed(3));
        showTimelineNotice(`Timeline content reaches ${minSeconds}s, but board duration remains ${getDuration().toFixed(3)}s. Change Duration explicitly or shorten/ripple-delete the extra space.`);
        return false;
    };
    const ensureDurationForFrames = (requiredFrames) => {
        const fps = getFps();
        const need = Math.max(1, Math.round(Number(requiredFrames || 1)));
        if (Math.round(getDuration() * fps) >= need) return false;
        const needSeconds = Number((need / fps).toFixed(3));
        setDurationSeconds(needSeconds, "auto_expand_timeline_content");
        showTimelineNotice(`Timeline duration auto-extended to ${needSeconds}s to keep the new slot inside the visible board.`);
        return true;
    };
    const followsDefaultForce = (seg, previousDefault) => {
        if (!seg || String(seg.type || "image") === "text") return false;
        const current = Number(seg.guideStrength);
        if (!Number.isFinite(current)) return true;
        const source = Number(seg.defaultForceSource);
        if (seg.forceCustom === true) {
            if (Number.isFinite(source) && Math.abs(current - source) < 0.0005) return true;
            if (!Number.isFinite(source) && Math.abs(current - Number(previousDefault || 0)) < 0.0005) return true;
            return false;
        }
        if (Number.isFinite(source) && Math.abs(current - source) < 0.0005) return true;
        return Math.abs(current - Number(previousDefault || 0)) < 0.0005;
    };
    const applyDefaultForceToLinkedSegments = (value) => {
        const next = clampGuideStrength(value);
        (timeline.segments || []).forEach((seg) => {
            if (!seg || String(seg.type || "image") === "text") return;
            seg.guideStrength = next;
            seg.guide_strength = next;
            seg.force = next;
            seg.strength = next;
            seg.imageLockStrength = next;
            seg.image_lock_strength = next;
            seg.defaultForceSource = next;
            seg.forceCustom = false;
        });
        lastDefaultForce = next;
    };
    const clampSegment = (seg) => {
        const total = getTotalFrames();
        seg.length = Math.max(1, Math.round(Number(seg.length || defaultLen())));
        seg.start = Math.max(0, Math.min(Math.round(Number(seg.start || 0)), Math.max(0, total - 1)));
        if (seg.start + seg.length > total) seg.length = Math.max(1, total - seg.start);
        if (String(seg.type || "image") !== "text" && String(seg.type || "image") !== "audio") {
            const guideStrength = clampGuideStrength(seg.motion ?? seg.guideStrength ?? seg.guide_strength ?? seg.strength ?? seg.force ?? seg.motion_force ?? seg.imageLockStrength ?? seg.image_lock_strength ?? defaultForceWidget?.value ?? 1);
            seg.motion = guideStrength;
            seg.guideStrength = guideStrength;
            seg.guide_strength = guideStrength;
            seg.force = guideStrength;
            seg.strength = guideStrength;
            seg.imageLockStrength = guideStrength;
            seg.image_lock_strength = guideStrength;
            seg.linkGuideLock = true;
            seg.link_guide_lock = true;
            if (seg.defaultForceSource !== undefined) seg.defaultForceSource = clampGuideStrength(seg.defaultForceSource, guideStrength);
        }
        return seg;
    };
    const audioSegmentHasMedia = (seg) => Boolean(seg && (String(seg.audioFile || "").trim() || String(seg.audioB64 || "").trim()));
    const stripEmptyAudioForBoardExport = (payload) => {
        const clean = cloneJsonData(payload || {});
        const audioSegments = Array.isArray(clean.audioSegments) ? clean.audioSegments : [];
        const hasAudioMedia = audioSegments.some((seg) => audioSegmentHasMedia(seg));
        if (hasAudioMedia) return clean;
        delete clean.audioSegments;
        delete clean.audioTrackCount;
        delete clean.masterAudioGain;
        delete clean.masterAudioNormalize;
        delete clean.audioSyncMode;
        delete clean.audio_data;
        delete clean.use_custom_audio;
        return clean;
    };
    const boardTimelineForExport = (timelineText) => {
        let payload = null;
        try { payload = JSON.parse(String(timelineText || "{}")); } catch {}
        const cleanPayload = stripEmptyAudioForBoardExport(payload || timeline || {});
        if (Array.isArray(cleanPayload.segments)) {
            for (const seg of cleanPayload.segments) {
                if (!isTimelineImageSegment(seg)) continue;
                const truth = String(seg.imageTruthPath || seg.image_truth_path || seg.imageFile || seg.image_file || seg.path || "").trim();
                if (!truth) continue;
                seg.imageFile = truth;
                seg.path = truth;
                seg.imageTruthPath = truth;
                seg.imageTruthRef = Math.max(1, Math.round(Number(seg.ref || seg.imageTruthRef || 1)));
                seg.imageTruthPinned = true;
                seg.imageTruthSource = seg.imageTruthSource || "board_export";
                delete seg.image_file;
                delete seg.image_truth_path;
            }
        }
        const settings = v3SettingsSnapshot();
        Object.entries(settings).forEach(([key, value]) => {
            if (cleanPayload[key] === undefined || cleanPayload[key] === null || cleanPayload[key] === "") cleanPayload[key] = value;
        });
        return {
            payload: cleanPayload,
            text: JSON.stringify(cleanPayload, null, 2),
        };
    };
    const compactV3BoardForPackageExport = (board) => {
        const compact = cloneJsonData(board || {});
        const sourceRefs = splitReferencePaths(board?.image_paths);
        const activePaths = [];
        const seen = new Set();
        const addPath = (value) => {
            const clean = String(value || "").trim();
            if (!clean || seen.has(clean)) return;
            seen.add(clean);
            activePaths.push(clean);
        };
        const pathForRef = (ref) => {
            const index = Math.round(Number(ref || 0)) - 1;
            return index >= 0 && index < sourceRefs.length ? String(sourceRefs[index] || "").trim() : "";
        };
        const payloads = [];
        if (compact.timeline && typeof compact.timeline === "object") payloads.push(compact.timeline);
        if (typeof compact.timeline_data === "string" && compact.timeline_data.trim()) {
            try {
                const parsed = JSON.parse(compact.timeline_data);
                if (parsed && typeof parsed === "object") payloads.push(parsed);
            } catch {}
        }
        if (!payloads.length) payloads.push(timeline);
        for (const payload of payloads) {
            const segments = Array.isArray(payload?.segments) ? payload.segments : [];
            for (const seg of segments) {
                if (!isTimelineImageSegment(seg)) continue;
                addPath(seg.imageTruthPath || seg.image_truth_path || seg.imageFile || seg.image_file || seg.path || pathForRef(seg.ref));
            }
            if (activePaths.length) continue;
            const rows = Array.isArray(payload?.rows) ? payload.rows : [];
            for (const row of rows) {
                if (!row || row.use_guide === false || Number(row.force ?? row.strength ?? row.guideStrength ?? 0) <= 0) continue;
                addPath(row.imageTruthPath || row.image_truth_path || row.imageFile || row.image_file || row.path || pathForRef(row.ref));
            }
        }
        if (!activePaths.length) return compact;
        const refMap = new Map(activePaths.map((path, index) => [path, index + 1]));
        const rewriteSegmentsForCompact = (segments) => {
            if (!Array.isArray(segments)) return;
            for (const seg of segments) {
                if (!isTimelineImageSegment(seg)) continue;
                const source = String(seg.imageTruthPath || seg.image_truth_path || seg.imageFile || seg.image_file || seg.path || pathForRef(seg.ref) || "").trim();
                const nextRef = refMap.get(source);
                if (!nextRef) continue;
                seg.ref = nextRef;
                seg.imageFile = source;
                seg.path = source;
                seg.imageTruthPath = source;
                seg.imageTruthRef = nextRef;
                seg.imageTruthPinned = true;
                seg.imageTruthSource = seg.imageTruthSource || "package_export";
                delete seg.image_file;
                delete seg.image_truth_path;
                if (/^ref_\d+$/i.test(String(seg.label || ""))) seg.label = `ref_${nextRef}`;
            }
        };
        const rewriteRowsForCompact = (rows) => {
            if (!Array.isArray(rows)) return;
            for (const row of rows) {
                if (!row || row.use_guide === false) continue;
                const source = String(row.imageTruthPath || row.image_truth_path || row.imageFile || row.image_file || row.path || pathForRef(row.ref) || "").trim();
                const nextRef = refMap.get(source);
                if (nextRef) row.ref = nextRef;
            }
        };
        compact.image_paths = activePaths;
        compact.images = activePaths.map((path, index) => ({
            ref: index + 1,
            path,
            name: String(path).split(/[\\/]/).pop() || `ref_${index + 1}`,
        }));
        if (compact.timeline && typeof compact.timeline === "object") {
            compact.timeline.image_paths = activePaths;
            rewriteSegmentsForCompact(compact.timeline.segments);
            rewriteRowsForCompact(compact.timeline.rows);
        }
        if (typeof compact.timeline_data === "string" && compact.timeline_data.trim()) {
            try {
                const parsed = JSON.parse(compact.timeline_data);
                if (parsed && typeof parsed === "object") {
                    parsed.image_paths = activePaths;
                    rewriteSegmentsForCompact(parsed.segments);
                    rewriteRowsForCompact(parsed.rows);
                    compact.timeline_data = JSON.stringify(parsed, null, 2);
                }
            } catch {}
        }
        console.log("[IAMCCS V3 PACKAGE EXPORT]", {
            nodeId: node?.id,
            activeImagePaths: activePaths,
            oldImagePaths: sourceRefs,
        });
        return compact;
    };
    const segmentRangesOverlap = (aStart, aLength, bStart, bLength) => {
        const a0 = Math.max(0, Math.round(Number(aStart || 0)));
        const b0 = Math.max(0, Math.round(Number(bStart || 0)));
        const a1 = a0 + Math.max(1, Math.round(Number(aLength || 1)));
        const b1 = b0 + Math.max(1, Math.round(Number(bLength || 1)));
        return a0 < b1 && b0 < a1;
    };
    const removeEmptyAudioPlaceholdersInRange = (track, start, length) => {
        const trackIndex = Math.max(0, Math.round(Number(track || 0)));
        const before = timeline.audioSegments || [];
        timeline.audioSegments = before.filter((seg) => {
            if (Number(seg?.track || 0) !== trackIndex) return true;
            if (audioSegmentHasMedia(seg) && !seg.placeholder) return true;
            return !segmentRangesOverlap(seg?.start, seg?.length, start, length);
        });
        return timeline.audioSegments.length !== before.length;
    };
    const cleanupAudioPlaceholdersOverlappingMedia = () => {
        const audio = timeline.audioSegments || [];
        const media = audio.filter((seg) => audioSegmentHasMedia(seg) && !seg.placeholder);
        if (!media.length) return false;
        const before = audio.length;
        timeline.audioSegments = audio.filter((seg) => {
            if (audioSegmentHasMedia(seg) && !seg.placeholder) return true;
            return !media.some((clip) =>
                Number(clip.track || 0) === Number(seg?.track || 0)
                && segmentRangesOverlap(clip.start, clip.length, seg?.start, seg?.length)
            );
        });
        return timeline.audioSegments.length !== before;
    };
    const magnetize = () => {
        const total = getTotalFrames();
        timeline.segments = (timeline.segments || []).map((seg) => clampSegment(seg)).sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
        let cursor = 0;
        for (const seg of timeline.segments) {
            if (Number(seg.start || 0) < cursor) seg.start = cursor;
            if (seg.start + seg.length > total) seg.length = Math.max(1, total - seg.start);
            cursor = seg.start + seg.length;
        }
        timeline.audioSegments = (timeline.audioSegments || []).map((seg) => clampSegment(seg)).sort((a, b) => (Number(a.track || 0) - Number(b.track || 0)) || (Number(a.start || 0) - Number(b.start || 0)));
    };
    const isGuideLockLinked = (_seg) => true;
    const segmentToRow = (seg, index) => {
        const fps = getFps();
        const isText = String(seg.type || "image") === "text";
        const truthPath = isText ? "" : String(seg.imageTruthPath || seg.image_truth_path || seg.imageFile || seg.image_file || seg.path || "").trim();
        const singleStrength = isText ? 0 : clampGuideStrength(seg.motion ?? seg.guideStrength ?? seg.guide_strength ?? seg.force ?? seg.strength ?? defaultForceWidget?.value ?? 1);
        return {
            second: Number((Number(seg.start || 0) / fps).toFixed(3)),
            frame: Math.round(Number(seg.start || 0)),
            ref: Math.max(1, Number(seg.ref || index + 1)),
            imageFile: truthPath,
            path: truthPath,
            imageTruthPath: truthPath,
            imageTruthRef: Math.max(1, Number(seg.imageTruthRef || seg.ref || index + 1)),
            imageTruthPinned: !isText,
            imageTruthSource: String(seg.imageTruthSource || "segment_to_row"),
            force: singleStrength,
            strength: singleStrength,
            guide_strength: singleStrength,
            motion: singleStrength,
            guideStrength: singleStrength,
            motion_force: singleStrength,
            image_lock_strength: singleStrength,
            imageLockStrength: singleStrength,
            linkGuideLock: true,
            link_guide_lock: true,
            label: String(seg.label || `${isText ? "text" : "shot"}_${index + 1}`),
            camera: String(seg.camera || "cinematic motion"),
            transition: String(seg.transition || "continuous_motion"),
            note: String(seg.note || ""),
            use_guide: !isText && seg.use_guide !== false,
            use_prompt: Boolean((String(seg.prompt ?? seg.local_prompt ?? seg.relay_prompt ?? "").trim() && seg.relay_manual_off !== true && seg.promptrelay_manual_off !== true) || seg.dialogue_pin || seg.image_lock || seg.motion_boost),
            relay_prompt: String(seg.prompt ?? seg.local_prompt ?? seg.relay_prompt ?? ""),
            camera_relay_mode: String(seg.camera_relay_mode || "off"),
            transition_relay_mode: String(seg.transition_relay_mode || "off"),
            relay_addon_position: String(seg.relay_addon_position || "after"),
            relay_modifier_text: String(seg.relay_modifier_text || ""),
            dialogue_pin: Boolean(seg.dialogue_pin || seg.dialoguePin),
            image_lock: Boolean(seg.image_lock || seg.imageLock),
            motion_boost: Boolean(seg.motion_boost || seg.motionBoost),
            clean_relay: Boolean(seg.clean_relay || seg.cleanRelay),
            step_transition_enabled: false,
            step_transition_type: "off",
            step_transition_prompt: "",
            step_transition_easing: String(seg.step_transition_easing || "ease_in_out"),
            step_transition_force_curve: String(seg.step_transition_force_curve || "late_target"),
            step_transition_duration: 0,
            step_transition_arrival: "auto",
            step_transition_auto_fit: true,
        };
    };
    let pendingTimelineCommit = 0;
    let promptEditCounter = 0;
    const markPromptFieldEdited = (field) => {
        if (!field?.dataset) return "";
        const stamp = `${Date.now()}-${++promptEditCounter}`;
        field.dataset.iamccsV3EditedAt = stamp;
        return stamp;
    };
    const fieldEditStamp = (field) => {
        const raw = String(field?.dataset?.iamccsV3EditedAt || "");
        const [time, order] = raw.split("-");
        return (Number(time) || 0) * 100000 + (Number(order) || 0);
    };
    const isTextEditing = () => {
        const active = document.activeElement;
        return Boolean(active && root.contains(active) && /^(TEXTAREA|INPUT)$/i.test(active.tagName || ""));
    };
    const commitTimelineJson = (json, force = false) => {
        node._iamccsCineShotboardV3LastTimelineText = String(json || "");
        if (timelineWidget) {
            timelineWidget.value = json;
            syncWidgetSerializedValue(node, timelineWidget, json);
        }
        node.properties = node.properties || {};
        node.properties.iamccs_v3_timeline_data_backup = json;
        if (pendingTimelineCommit) {
            clearTimeout(pendingTimelineCommit);
            pendingTimelineCommit = 0;
        }
        if (force || !isTextEditing()) {
            setWidgetValue(node, "timeline_data", json);
            return;
        }
        pendingTimelineCommit = setTimeout(() => {
            pendingTimelineCommit = 0;
            setWidgetValue(node, "timeline_data", timelineWidget?.value ?? json);
        }, 220);
    };
    const syncTimelineTextFromDom = () => {
        let changed = 0;
        const active = document.activeElement;
        const fields = Array.from(root.querySelectorAll("[data-iamccs-v3-segment-id][data-iamccs-v3-key]"))
            .sort((a, b) => {
                if (a === active) return -1;
                if (b === active) return 1;
                return fieldEditStamp(b) - fieldEditStamp(a);
            });
        fields.forEach((el) => {
            const key = String(el.dataset.iamccsV3Key || "");
            if (key !== "prompt") return;
            const segmentId = String(el.dataset.iamccsV3SegmentId || "");
            if (!segmentId) return;
            const seg = (timeline.segments || []).find((item) => String(item?.id || "") === segmentId);
            if (!seg) return;
            const value = String(el.value ?? "");
            const currentPrompt = String(seg.prompt ?? seg.local_prompt ?? seg.relay_prompt ?? "");
            if (currentPrompt === value) return;
            if (!value.trim() && currentPrompt.trim()) {
                seg.prompt = currentPrompt;
                seg.use_prompt = Boolean(currentPrompt.trim());
                if (currentPrompt.trim()) {
                    seg.relay_manual_off = false;
                    seg.promptrelay_manual_off = false;
                }
                syncSegmentTextPeers(segmentId, "prompt", currentPrompt, el);
                syncSegmentRelayPeers(segmentId, Boolean(seg.use_prompt), null);
                return;
            }
            seg.prompt = value;
            seg.use_prompt = Boolean(value.trim());
            if (value.trim()) {
                seg.relay_manual_off = false;
                seg.promptrelay_manual_off = false;
            }
            changed += 1;
            syncSegmentTextPeers(segmentId, "prompt", value, el);
            syncSegmentRelayPeers(segmentId, Boolean(seg.use_prompt), null);
        });
        if (changed) {
            console.log("[IAMCCS V3 PERSIST] DOM text authority sync", { changed });
        }
        return changed;
    };
    const flushTimelineWrite = () => {
        syncTimelineTextFromDom();
        if (pendingTimelineCommit) {
            clearTimeout(pendingTimelineCommit);
            pendingTimelineCommit = 0;
            const value = timelineWidget?.value || "";
            if (timelineWidget) syncWidgetSerializedValue(node, timelineWidget, value);
            setWidgetValue(node, "timeline_data", value);
        }
    };
    const syncSegmentTextPeers = (segmentId, key, value, source) => {
        const sourceStamp = String(source?.dataset?.iamccsV3EditedAt || "");
        root.querySelectorAll(`[data-iamccs-v3-segment-id="${String(segmentId)}"][data-iamccs-v3-key="${String(key)}"]`).forEach((el) => {
            if (el === source) return;
            if (el.value !== value) el.value = value;
            if (sourceStamp && el.dataset) el.dataset.iamccsV3EditedAt = sourceStamp;
        });
    };
    const syncSegmentRelayPeers = (segmentId, checked, source) => {
        root.querySelectorAll(`[data-iamccs-v3-segment-id="${String(segmentId)}"][data-iamccs-v3-key="use_prompt"]`).forEach((el) => {
            if (el === source) return;
            if (el.checked !== checked) el.checked = checked;
        });
    };
    const logPromptPersistence = (seg, source) => {
        try {
            const prompt = String(seg?.prompt || "");
            const preview = prompt.replace(/\s+/g, " ").slice(0, 220);
            console.log("[IAMCCS V3 PERSIST]", {
                source,
                segmentId: seg?.id,
                label: seg?.label,
                promptLength: prompt.length,
                promptPreview: preview,
                timelineWidgetLength: String(timelineWidget?.value || "").length,
                widgetIndex: Array.isArray(node.widgets) ? node.widgets.indexOf(timelineWidget) : -1,
                widgetsValueLength: Array.isArray(node.widgets_values) && timelineWidget ? String(node.widgets_values[node.widgets.indexOf(timelineWidget)] || "").length : 0,
            });
        } catch {}
    };
    const writeTimeline = (options = {}) => {
        if (!options.skipDomSync) syncTimelineTextFromDom();
        neutralizeLegacyStepTransitions();
        enforceDurationMinimum();
        cleanupAudioPlaceholdersOverlappingMedia();
        magnetize();
        normalizeTimelineSegmentReferences();
        const fps = getFps();
        const effectiveDurationSeconds = getDuration();
        timeline.duration_seconds = effectiveDurationSeconds;
        timeline.frame_rate = fps;
        const rows = timeline.segments.filter((seg) => !seg.placeholder).map(segmentToRow);
        const promptRelayEnabled = rows.some((row) => {
            const hasPrompt = row.use_prompt && String(row.relay_prompt || "").trim();
            return hasPrompt;
        });
        const audioHasMedia = (timeline.audioSegments || []).some((seg) => audioSegmentHasMedia(seg));
        const durationFrames = getTotalFrames();
        const visual = (timeline.segments || [])
            .filter((seg) => String(seg.type || "image") !== "audio" && !seg.placeholder)
            .slice()
            .sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
        const directorPrompts = [];
        const directorLengths = [];
        let cursor = 0;
        let pendingGap = 0;
        for (const seg of visual) {
            const start = Math.max(0, Math.round(Number(seg.start || 0)));
            if (start >= durationFrames) break;
            if (start > cursor) {
                const gap = Math.min(start, durationFrames) - cursor;
                if (directorLengths.length) directorLengths[directorLengths.length - 1] += Math.max(0, gap);
                else pendingGap += Math.max(0, gap);
            }
            const clippedEnd = Math.min(start + Math.max(1, Math.round(Number(seg.length || 1))), durationFrames);
            const length = Math.max(1, clippedEnd - start + pendingGap);
            pendingGap = 0;
            const prompt = String(seg.prompt ?? seg.local_prompt ?? seg.relay_prompt ?? "").trim();
            const promptActive = Boolean(prompt && seg.relay_manual_off !== true && seg.promptrelay_manual_off !== true);
            if (promptActive) {
                directorPrompts.push(prompt);
                directorLengths.push(length);
            } else if (directorLengths.length) {
                directorLengths[directorLengths.length - 1] += length;
            } else {
                pendingGap += length;
            }
            cursor = start + Math.max(1, Math.round(Number(seg.length || 1)));
        }
        if (directorLengths.length && Math.min(cursor, durationFrames) < durationFrames) {
            directorLengths[directorLengths.length - 1] += durationFrames - Math.min(cursor, durationFrames);
        }
        const guideStrength = visual
            .filter((seg) => String(seg.type || "image") !== "text")
            .map((seg) => Number(seg.guideStrength ?? seg.guide_strength ?? seg.force ?? defaultForceWidget?.value ?? 1.0).toFixed(2))
            .join(",");
        const audioData = JSON.stringify({
            audioSegments: timeline.audioSegments || [],
            use_custom_audio: audioHasMedia,
            masterAudioGain: Math.max(0, Math.min(2, Number(timeline.masterAudioGain ?? 1) || 1)),
            masterAudioNormalize: Boolean(timeline.masterAudioNormalize),
            audioSyncMode: String(timeline.audioSyncMode || "timeline_audio"),
        });
        const flfrealMode = ["flfreal_parity", "iamccs_enhanced"].includes(String(timeline.flfrealMode || "iamccs_enhanced"))
            ? String(timeline.flfrealMode || "iamccs_enhanced")
            : "iamccs_enhanced";
        const globalPromptOnly = Boolean(timeline.globalPromptOnly);
        const effectiveDirectorPrompts = globalPromptOnly ? [] : directorPrompts;
        const effectiveDirectorLengths = globalPromptOnly ? [] : directorLengths;
        const effectivePromptRelayEnabled = globalPromptOnly ? false : promptRelayEnabled;
        timeline.rows = rows;
        const timelineRows = rows;
        const multiGeneration = timeline.multiGeneration && typeof timeline.multiGeneration === "object"
            ? JSON.parse(JSON.stringify(timeline.multiGeneration))
            : {};
        if (multiGeneration.activeTimelineId) {
            const activeTimelineId = String(multiGeneration.activeTimelineId || "T01");
            multiGeneration.visualTimelines = multiGeneration.visualTimelines && typeof multiGeneration.visualTimelines === "object"
                ? multiGeneration.visualTimelines
                : {};
            multiGeneration.visualTimelines[activeTimelineId] = {
                schema: "iamccs.multigeneration.visual_timeline",
                schema_version: 1,
                timeline_id: activeTimelineId,
                saved_at: new Date().toISOString(),
                duration_seconds: effectiveDurationSeconds,
                frame_rate: fps,
                image_paths: refPaths(),
                segments: JSON.parse(JSON.stringify(timeline.segments || [])),
                rows: JSON.parse(JSON.stringify(timelineRows)),
                director_local_prompts: effectiveDirectorPrompts.join(" | "),
                director_segment_lengths: effectiveDirectorLengths.join(","),
                director_guide_strength: guideStrength,
                local_prompts: effectiveDirectorPrompts.join(" | "),
                segment_lengths: effectiveDirectorLengths.join(","),
                guide_strength: guideStrength,
            };
        }
        node.properties = node.properties || {};
        const nextTruthRevision = Math.max(
            Number(node.properties.iamccs_v3_timeline_revision || 0),
            Number(timeline.truthRevision || 0),
            0
        ) + 1;
        const truthUpdatedAt = new Date().toISOString();
        timeline.truthRevision = nextTruthRevision;
        const clean = {
            schema: "iamccs.cine.filmmaker_timeline",
            schema_version: 2,
            truth_revision: nextTruthRevision,
            _iamccs_v3_truth_revision: nextTruthRevision,
            truth_updated_at: truthUpdatedAt,
            global_prompt: String(promptArea?.value || promptWidget?.value || ""),
            prompt: String(promptArea?.value || promptWidget?.value || ""),
            flfrealMode,
            flfreal_mode: flfrealMode,
            verbose_log: timeline.verboseLog !== false,
            verboseLog: timeline.verboseLog !== false,
            global_prompt_only: globalPromptOnly,
            use_global_prompt_only: globalPromptOnly,
            promptrelay_enabled: effectivePromptRelayEnabled,
            use_custom_audio: audioHasMedia,
            audioSyncMode: String(timeline.audioSyncMode || "timeline_audio"),
            generationStrategy: String(timeline.generationStrategy || "single_timeline"),
            director_local_prompts: effectiveDirectorPrompts.join(" | "),
            director_segment_lengths: effectiveDirectorLengths.join(","),
            director_guide_strength: guideStrength,
            local_prompts: effectiveDirectorPrompts.join(" | "),
            segment_lengths: effectiveDirectorLengths.join(","),
            guide_strength: guideStrength,
            audio_data: audioData,
            duration_seconds: effectiveDurationSeconds,
            frame_rate: fps,
            image_paths: refPaths(),
            image_width: Number(imageWidthWidget?.value || 768),
            image_height: Number(imageHeightWidget?.value || 432),
            image_resize_method: cineResizeMethodValue(getWidget(node, "image_resize_method")?.value),
            image_multiple_of: Number(getWidget(node, "image_multiple_of")?.value || 32),
            promptrelay_epsilon: Number(getWidget(node, "promptrelay_epsilon")?.value || 0.001),
            img_compression: Number(getWidget(node, "img_compression")?.value || 0),
            default_force: clampGuideStrength(defaultForceWidget?.value || 1),
            guide_policy: String(getWidget(node, "guide_policy")?.value || "every_checked_row"),
            min_guide_gap_seconds: Number(getWidget(node, "min_guide_gap_seconds")?.value || 0),
            max_guides: Number(getWidget(node, "max_guides")?.value || 50),
            wan_frame_round_mode: String(getWidget(node, "wan_frame_round_mode")?.value || "up_8n_plus_1"),
            audioTrackCount: Math.max(1, Math.round(Number(timeline.audioTrackCount || 1))),
            masterAudioGain: Math.max(0, Math.min(2, Number(timeline.masterAudioGain ?? 1) || 1)),
            masterAudioNormalize: Boolean(timeline.masterAudioNormalize),
            segments: timeline.segments,
            audioSegments: timeline.audioSegments,
            rows: timelineRows,
            multiGeneration,
        };
        node.properties.iamccs_v3_timeline_revision = nextTruthRevision;
        node.properties.iamccs_v3_timeline_updated_at = truthUpdatedAt;
        commitTimelineJson(JSON.stringify(clean, null, 2), Boolean(options.force));
    };

    const applyExternalTimelineData = (payload = {}) => {
        const source = payload && typeof payload === "object" ? JSON.parse(JSON.stringify(payload)) : {};
        const nextPrompt = String(source.global_prompt ?? source.prompt ?? promptArea?.value ?? promptWidget?.value ?? "");
        if (promptArea) promptArea.value = nextPrompt;
        if (promptWidget) {
            promptWidget.value = nextPrompt;
            try { promptWidget.callback?.(nextPrompt); } catch {}
        }
        node._iamccsCineShotboardV3LastPromptText = nextPrompt;
        const sourceRows = Array.isArray(source.rows)
            ? source.rows.map((row, index) => normalizeShotboardRow(row, index))
            : [];
        const sourceSegments = Array.isArray(source.segments)
            ? source.segments.map((seg) => normalizeV3RelayOnlySegment({ ...(seg || {}), id: seg?.id || newId(seg?.type === "text" ? "text" : "seg") }))
            : [];
        const importedSegments = sourceRows.length ? rowsToSegments(sourceRows) : [];
        const reconciledSegments = (sourceSegments.length ? sourceSegments : importedSegments).map((seg) => ({ ...(seg || {}) }));
        const visualIndexes = [];
        reconciledSegments.forEach((seg, index) => {
            if (String(seg?.type || "image") !== "audio" && !seg?.placeholder) visualIndexes.push(index);
        });
        sourceRows.forEach((row, index) => {
            const prompt = String(row?.relay_prompt ?? row?.local_prompt ?? row?.prompt ?? "").trim();
            const targetIndex = visualIndexes[index];
            if (!prompt || targetIndex === undefined || !reconciledSegments[targetIndex]) return;
            const target = reconciledSegments[targetIndex];
            target.prompt = prompt;
            target.local_prompt = prompt;
            target.relay_prompt = prompt;
            target.use_prompt = true;
            target.relay_manual_off = false;
            target.promptrelay_manual_off = false;
        });
        const nextRows = sourceRows.length
            ? sourceRows
            : reconciledSegments.filter((seg) => !seg?.placeholder).map(segmentToRow);
        timeline = {
            ...timeline,
            ...source,
            schema: "iamccs.cine.filmmaker_timeline",
            schema_version: Math.max(2, Number(source.schema_version || timeline.schema_version || 2)),
            segments: reconciledSegments,
            rows: nextRows,
            audioSegments: Array.isArray(source.audioSegments) ? source.audioSegments.map((seg) => ({ ...(seg || {}), id: seg?.id || newId("aud") })) : (timeline.audioSegments || []),
            audioTrackCount: Math.max(1, Number(source.audioTrackCount || timeline.audioTrackCount || 1)),
            audioSyncMode: String(source.audioSyncMode || timeline.audioSyncMode || "timeline_audio"),
            duration_seconds: objectDurationTruth(source) || timeline.duration_seconds,
            frame_rate: objectFpsTruth(source) || timeline.frame_rate,
            generationStrategy: String(source.generationStrategy || timeline.generationStrategy || "single_timeline"),
            flfrealMode: String(source.flfrealMode || source.flfreal_mode || timeline.flfrealMode || "iamccs_enhanced"),
            globalPromptOnly: Boolean(source.globalPromptOnly ?? source.global_prompt_only ?? source.use_global_prompt_only ?? timeline.globalPromptOnly),
            verboseLog: source.verboseLog ?? source.verbose_log ?? timeline.verboseLog,
            multiGeneration: source.multiGeneration && typeof source.multiGeneration === "object" ? cloneForMultiTimeline(source.multiGeneration, {}) : (timeline.multiGeneration || {}),
        };
        node._iamccsCineShotboardV3LastTimelineText = "";
        writeTimeline({ force: true, skipDomSync: true });
        syncTimingWidgetsFromTimelineTruth("external_apply_timeline_data");
        draw();
    };

    if (!Array.isArray(timeline.segments)) timeline.segments = [];
    const collapsedNodeHeight = () => {
        const tracks = Math.max(1, Math.round(Number(timeline.audioTrackCount || 1)));
        return Math.max(SHOTBOARD_V3_COLLAPSED_HEIGHT, 450 + tracks * 90);
    };
    const currentNodeHeight = () => collapsed ? collapsedNodeHeight() : SHOTBOARD_V3_OPEN_HEIGHT;

    const head = document.createElement("div");
    head.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px;";
    const title = document.createElement("div");
    title.textContent = "Cine Shotboard Planner V3";
    title.style.cssText = `font-size:13px;font-weight:800;color:${purple.text};`;
    const topActions = document.createElement("div");
    topActions.style.cssText = "display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end;";
    const v3ButtonColors = {
        normal: { bg: purple.button, hover: purple.buttonHover, border: purple.border, color: purple.text },
        teal: { bg: "linear-gradient(180deg,#267E8E,#12475A)", hover: "linear-gradient(180deg,#3197A8,#195B72)", border: "#8DE7FF", color: "#EAFBFF" },
        amber: { bg: "linear-gradient(180deg,#8D642C,#4F3518)", hover: "linear-gradient(180deg,#A67734,#60421E)", border: "#F3B34B", color: "#FFF4D5" },
        blue: { bg: "linear-gradient(180deg,#205E82,#123B5D)", hover: "linear-gradient(180deg,#2A75A0,#194D75)", border: "#78B8E8", color: "#EFF9FF" },
        violet: { bg: "linear-gradient(180deg,#554478,#352C59)", hover: "linear-gradient(180deg,#67528F,#443772)", border: "#B9A2F0", color: "#FAF5FF" },
        olive: { bg: "linear-gradient(180deg,#5E7534,#394B24)", hover: "linear-gradient(180deg,#708B3D,#4B622D)", border: "#A8D36B", color: "#F7FFE9" },
        slate: { bg: "linear-gradient(180deg,#405362,#273642)", hover: "linear-gradient(180deg,#50687A,#334655)", border: "#91AFC0", color: "#F2FAFF" },
        sand: { bg: "linear-gradient(180deg,#7A6B4E,#4F4330)", hover: "linear-gradient(180deg,#8C7D5A,#60523C)", border: "#D4BE83", color: "#FFF8E6" },
        green: { bg: "linear-gradient(180deg,#3C7B56,#244E3A)", hover: "linear-gradient(180deg,#4A9167,#2F644A)", border: "#92C46E", color: "#F2FFF6" },
        gold: { bg: "linear-gradient(180deg,#99752E,#5A4219)", hover: "linear-gradient(180deg,#B58C39,#6D5321)", border: "#FFE08A", color: "#FFF7D5" },
        danger: { bg: "#6B302A", hover: "#8A3A32", border: purple.danger, color: purple.text },
    };
    const makeBtn = (label, tone = "normal") => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = label;
        const colors = v3ButtonColors[tone] || v3ButtonColors.normal;
        btn.style.cssText = [
            "height:28px",
            "padding:0 10px",
            `border:1px solid ${colors.border}`,
            "border-radius:5px",
            `background:${colors.bg}`,
            `color:${colors.color}`,
            "font-size:11px",
            "font-weight:800",
            "cursor:pointer",
        ].join(";");
        btn.onmouseenter = () => {
            if (btn.dataset.iamccsPressedPreview !== "true") btn.style.background = colors.hover;
        };
        btn.onmouseleave = () => {
            if (btn.dataset.iamccsPressedPreview !== "true") btn.style.background = colors.bg;
        };
        return protectControlDrag(addPressPreview(btn, {
            pressedBg: purple.buttonPress,
            pressedColor: purple.valueText,
            pressedBorder: purple.accent,
        }));
    };
    const markToggleButton = (btn, active) => {
        btn.setAttribute("aria-pressed", active ? "true" : "false");
        btn.style.borderColor = active ? "#F2FFFB" : "";
        btn.style.boxShadow = active ? "inset 0 0 0 1px rgba(255,255,255,.38),0 0 0 1px rgba(143,208,204,.16)" : "none";
        btn.style.opacity = active ? "1" : ".82";
        return btn;
    };
    const bindToolbarToggle = (btn, apply) => {
        let lastFire = 0;
        const fire = (event) => {
            event?.preventDefault?.();
            event?.stopPropagation?.();
            event?.stopImmediatePropagation?.();
            const now = Date.now();
            if (now - lastFire < 180) return;
            lastFire = now;
            apply();
        };
        btn.onpointerdown = (event) => {
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation?.();
        };
        btn.onpointerup = fire;
        btn.onclick = fire;
        btn.onkeydown = (event) => {
            if (event.key === " " || event.key === "Enter") fire(event);
        };
        return btn;
    };
    const cloneForMultiTimeline = (value, fallback) => {
        try {
            return JSON.parse(JSON.stringify(value ?? fallback));
        } catch {
            return fallback;
        }
    };
    const multiTimelineId = (takeIndex) => `T${String(Math.max(1, Math.round(Number(takeIndex) || 1))).padStart(2, "0")}`;
    const multiTimelineTakeFromId = (timelineId) => Math.max(1, Math.round(Number(String(timelineId || "T01").replace(/\D/g, "") || 1)));
    const multiTargetDurationSeconds = () => {
        const multi = timeline.multiGeneration && typeof timeline.multiGeneration === "object" ? timeline.multiGeneration : {};
        const chunk = Math.max(0, Number(multi.chunkSeconds || 0));
        return chunk > 0 ? chunk : getDuration();
    };
    const findMultiTimelineBridge = () => {
        const nodes = Array.isArray(app?.graph?._nodes) ? app.graph._nodes : [];
        return nodes.find((candidate) => {
            const type = String(candidate?.type || candidate?.comfyClass || candidate?.constructor?.type || "");
            return type === "IAMCCS_MultiTimelineBridge" || type.includes("MultiTimelineBridge");
        });
    };
    const setBridgeTake = (takeIndex) => {
        const bridge = findMultiTimelineBridge();
        if (!bridge) return;
        const widget = (bridge.widgets || []).find((item) => item?.name === "active_take");
        if (!widget) return;
        const take = Math.max(1, Math.round(Number(takeIndex) || 1));
        widget.value = take;
        try { widget.callback?.(take); } catch {}
        try {
            window.dispatchEvent(new CustomEvent("iamccs:multigeneration-active-take", {
                detail: { nodeId: bridge.id, activeTake: take },
            }));
        } catch {}
    };
    const snapshotCurrentVisualTimeline = (timelineId) => ({
        schema: "iamccs.multigeneration.visual_timeline",
        schema_version: 1,
        timeline_id: String(timelineId || "T01"),
        saved_at: new Date().toISOString(),
        duration_seconds: getDuration(),
        frame_rate: getFps(),
        image_paths: refPaths(),
        segments: cloneForMultiTimeline(timeline.segments || [], []),
        rows: cloneForMultiTimeline(timeline.rows || [], []),
        guide_strength: Number(defaultForceWidget?.value || 0.3),
    });
    const defaultVisualTimeline = (timelineId) => ({
        schema: "iamccs.multigeneration.visual_timeline",
        schema_version: 1,
        timeline_id: String(timelineId || "T01"),
        created_at: new Date().toISOString(),
        duration_seconds: multiTargetDurationSeconds(),
        frame_rate: getFps(),
        image_paths: [],
        segments: [],
        rows: [],
        guide_strength: Number(defaultForceWidget?.value || 0.3),
    });
    const multiAudioSegmentsForTake = (multi, takeIndex) => {
        const take = Math.max(1, Math.round(Number(takeIndex) || 1));
        const timelineId = multiTimelineId(take);
        const normalizeMultiTimelineId = (value, fallbackTake = take) => {
            const raw = String(value || "").trim();
            const normalizedTake = Math.max(1, Math.round(Number(raw.replace(/\D/g, "")) || Number(fallbackTake) || 1));
            return multiTimelineId(normalizedTake);
        };
        const all = Array.isArray(multi?.audioSegmentsAll)
            ? multi.audioSegmentsAll
            : Array.isArray(multi?.allAudioSegments)
                ? multi.allAudioSegments
                : [];
        if (!all.length) return [];
        const matches = all.filter((seg) => {
            const segTake = Math.max(0, Math.round(Number(seg?.multiTakeIndex || 0)));
            const rawTimelineId = String(seg?.timelineId || "").trim();
            const segTimelineId = rawTimelineId ? normalizeMultiTimelineId(rawTimelineId, segTake || take) : "";
            const isMulti = Boolean(seg?.multiGenerationClip) || /^T\d+/i.test(rawTimelineId) || segTake > 0;
            if (!isMulti) return false;
            return segTimelineId === timelineId || segTake === take;
        });
        return matches.map((seg) => {
            const localStart = Number(seg?.localStart);
            const next = cloneForMultiTimeline(seg, {});
            next.track = 0;
            next.start = Number.isFinite(localStart)
                ? Math.max(0, Math.round(localStart))
                : 0;
            next.timelineId = timelineId;
            next.multiTakeIndex = take;
            next.shotboardActiveTakeAudio = true;
            next.sourceTrackOriginal = Math.max(0, Math.round(Number(seg?.track || seg?.sourceTrackOriginal || 0)));
            return next;
        });
    };
    const applyMultiAudioForTake = (multi, takeIndex) => {
        const sourceAll = Array.isArray(multi?.audioSegmentsAll) || Array.isArray(multi?.allAudioSegments);
        if (!sourceAll) return false;
        timeline.audioSegments = multiAudioSegmentsForTake(multi, takeIndex);
        timeline.audioTrackCount = 1;
        timeline.audioBusMode = "shotboard_only_first";
        timeline.onlyFirstTrack = true;
        return true;
    };
    const switchMultiTimeline = (takeIndex) => {
        const take = Math.max(1, Math.round(Number(takeIndex) || 1));
        const nextId = multiTimelineId(take);
        const multi = timeline.multiGeneration && typeof timeline.multiGeneration === "object"
            ? cloneForMultiTimeline(timeline.multiGeneration, {})
            : {};
        const currentId = String(multi.activeTimelineId || multiTimelineId(multi.activeTake || 1));
        const visualTimelines = multi.visualTimelines && typeof multi.visualTimelines === "object"
            ? cloneForMultiTimeline(multi.visualTimelines, {})
            : {};
        visualTimelines[currentId] = snapshotCurrentVisualTimeline(currentId);
        const selected = visualTimelines[nextId] || defaultVisualTimeline(nextId);
        visualTimelines[nextId] = selected;
        timeline.segments = cloneForMultiTimeline(selected.segments || [], []);
        timeline.rows = cloneForMultiTimeline(selected.rows || [], []);
        const selectedDuration = Math.max(0, Number(selected.duration_seconds || 0)) || multiTargetDurationSeconds();
        if (durationWidget && selectedDuration > 0) durationWidget.value = Number(selectedDuration.toFixed(3));
        if (fpsWidget && Number(selected.frame_rate) > 0) fpsWidget.value = Number(selected.frame_rate);
        timeline.generationStrategy = "multigeneration_manual_take";
        const nextMultiGeneration = {
            ...multi,
            enabled: true,
            activeTake: take,
            activeTimelineId: nextId,
            timelineIds: Array.from(new Set([
                ...(Array.isArray(multi.timelineIds) ? multi.timelineIds : []),
                ...Object.keys(visualTimelines),
                nextId,
            ])).sort(),
            visualTimelines,
            updatedAt: new Date().toISOString(),
        };
        applyMultiAudioForTake(nextMultiGeneration, take);
        timeline.multiGeneration = nextMultiGeneration;
        setBridgeTake(take);
        writeTimeline({ force: true });
        showTimelineNotice(`Loaded ${nextId}. Visual boxes are independent for this generation.`, "info");
        draw();
    };
    const makeMultiTimelineControl = () => {
        const multi = timeline.multiGeneration && typeof timeline.multiGeneration === "object" ? timeline.multiGeneration : {};
        const bridge = findMultiTimelineBridge();
        const maxWidget = (bridge?.widgets || []).find((item) => item?.name === "max_takes");
        const fixedWidget = (bridge?.widgets || []).find((item) => item?.name === "fixed_take_count");
        const known = Math.max(
            Array.isArray(multi.timelineIds) ? multi.timelineIds.length : 0,
            multi.visualTimelines && typeof multi.visualTimelines === "object" ? Object.keys(multi.visualTimelines).length : 0,
            0
        );
        const maxTakes = Math.max(2, Math.min(12, Math.round(Number(maxWidget?.value || fixedWidget?.value || known || 5))));
        const activeTake = Math.max(1, Math.min(maxTakes, Math.round(Number(multi.activeTake || multiTimelineTakeFromId(multi.activeTimelineId) || 1))));
        const wrap = document.createElement("div");
        wrap.title = "Switch real Shotboard visual timelines for staged multigeneration.";
        wrap.style.cssText = `display:flex;align-items:center;gap:4px;height:28px;padding:0 5px;border:1px solid ${purple.border};border-radius:5px;background:${purple.button};`;
        const label = document.createElement("span");
        label.textContent = "MULTI";
        label.style.cssText = `color:#fff1ba;font-size:9px;font-weight:950;`;
        const select = document.createElement("select");
        select.style.cssText = `height:22px;min-width:112px;border:1px solid ${purple.border};border-radius:4px;background:${purple.valueBg};color:${purple.valueText};font-size:10px;font-weight:900;`;
        for (let i = 1; i <= maxTakes; i += 1) {
            const id = multiTimelineId(i);
            const option = document.createElement("option");
            option.value = String(i);
            option.textContent = `${id} / gen ${i}`;
            select.appendChild(option);
        }
        select.value = String(activeTake);
        select.onchange = (event) => {
            event.preventDefault();
            event.stopPropagation();
            switchMultiTimeline(Number(select.value));
        };
        wrap.append(label, select);
        if (String(multi.durationWarning || "").trim()) {
            const warn = document.createElement("span");
            warn.textContent = "DURATION";
            warn.title = String(multi.durationWarning || "");
            warn.style.cssText = "color:#ffdf8a;font-size:8px;font-weight:950;border-left:1px solid rgba(255,255,255,.16);padding-left:5px;";
            wrap.appendChild(warn);
        }
        return protectControlDrag(wrap);
    };
    const setFlfrealMode = (value) => {
        timeline.flfrealMode = ["flfreal_parity", "iamccs_enhanced"].includes(String(value)) ? String(value) : "iamccs_enhanced";
        writeTimeline({ force: true });
        draw();
    };
    const logBtn = makeBtn(timeline.verboseLog === false ? "Log: OFF" : "Log: ON", timeline.verboseLog === false ? "slate" : "blue");
    logBtn.title = "Toggle verbose ComfyUI backend logs for Shotboard V3 values, prompts, motion and image-lock strengths, and audio state.";
    markToggleButton(logBtn, timeline.verboseLog !== false);
    setWidgetValue(node, "debug_verbose", timeline.verboseLog !== false);
    bindToolbarToggle(logBtn, () => {
        timeline.verboseLog = timeline.verboseLog === false;
        setWidgetValue(node, "debug_verbose", timeline.verboseLog !== false);
        logBtn.textContent = timeline.verboseLog === false ? "Log: OFF" : "Log: ON";
        markToggleButton(logBtn, timeline.verboseLog !== false);
        writeTimeline({ force: true });
        draw();
    });
    const globalOnlyBtn = makeBtn(Boolean(timeline.globalPromptOnly) ? "Global only: ON" : "Global only: OFF", Boolean(timeline.globalPromptOnly) ? "green" : "slate");
    globalOnlyBtn.title = "When ON, ignore local PromptRelay text/action lanes and use only the global prompt, while keeping image guides.";
    markToggleButton(globalOnlyBtn, Boolean(timeline.globalPromptOnly));
    bindToolbarToggle(globalOnlyBtn, () => {
        timeline.globalPromptOnly = !Boolean(timeline.globalPromptOnly);
        globalOnlyBtn.textContent = Boolean(timeline.globalPromptOnly) ? "Global only: ON" : "Global only: OFF";
        markToggleButton(globalOnlyBtn, Boolean(timeline.globalPromptOnly));
        writeTimeline({ force: true });
        draw();
    });
    const promptSizeWrap = document.createElement("div");
    promptSizeWrap.title = "Prompt text size";
    promptSizeWrap.style.cssText = `display:flex;align-items:center;gap:3px;height:28px;padding:0 4px;border:1px solid ${purple.border};border-radius:5px;background:${purple.button};`;
    const promptSizeButton = (label, delta) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = label;
        btn.style.cssText = `width:24px;height:22px;padding:0;border:1px solid ${purple.accent};border-radius:4px;background:linear-gradient(180deg,#EAF8FF,#6FB6D2);color:#071923;font-size:15px;font-weight:900;line-height:1;cursor:pointer;box-shadow:inset 0 1px 0 rgba(255,255,255,.62),0 0 0 1px rgba(0,0,0,.35);`;
        btn.onclick = (event) => {
            event.preventDefault();
            event.stopPropagation();
            promptTextScale = Math.max(0.85, Math.min(1.55, Math.round((promptTextScale + delta) * 100) / 100));
            node.properties = node.properties || {};
            node.properties.iamccs_v3_prompt_text_scale = promptTextScale;
            draw();
        };
        return protectControlDrag(addPressPreview(btn, {
            pressedBg: purple.buttonPress,
            pressedColor: purple.valueText,
            pressedBorder: "#FFE08A",
        }));
    };
    const promptSizeReadout = document.createElement("span");
    promptSizeReadout.style.cssText = `min-width:34px;text-align:center;color:${purple.muted};font-size:9px;font-weight:900;`;
    promptSizeWrap.append(promptSizeButton("-", -0.1), promptSizeReadout, promptSizeButton("+", 0.1));
    const timelineMeterWrap = document.createElement("div");
    timelineMeterWrap.title = "Visual timeline meter only. It does not change duration, frame starts, segment lengths or backend values.";
    timelineMeterWrap.style.cssText = `display:flex;align-items:center;gap:3px;height:28px;padding:0 4px;border:1px solid ${purple.border};border-radius:5px;background:${purple.button};`;
    const timelineMeterButton = (label, delta) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = label;
        btn.title = label === "-" ? "Compress timeline visually" : "Expand timeline visually";
        btn.style.cssText = `width:24px;height:22px;border:1px solid ${purple.accent};border-radius:4px;background:linear-gradient(180deg,#EAF8FF,#6FB6D2);color:#071923;font-size:14px;font-weight:900;cursor:pointer;box-shadow:inset 0 1px 0 rgba(255,255,255,.62),0 0 0 1px rgba(0,0,0,.35);`;
        btn.onclick = (event) => {
            event.preventDefault();
            event.stopPropagation();
            timelineMeterSeconds = clampTimelineMeterSeconds(timelineMeterSeconds + delta);
            node.properties = node.properties || {};
            node.properties.iamccs_v3_timeline_meter_seconds = timelineMeterSeconds;
            node.properties.iamccs_v3_timeline_meter_user_set = true;
            draw();
        };
        return protectControlDrag(addPressPreview(btn, {
            pressedBg: purple.buttonPress,
            pressedColor: purple.valueText,
            pressedBorder: "#FFE08A",
        }));
    };
    const timelineMeterReadout = document.createElement("span");
    timelineMeterReadout.style.cssText = `min-width:50px;text-align:center;color:${purple.muted};font-size:9px;font-weight:900;`;
    timelineMeterWrap.append(timelineMeterButton("-", -0.5), timelineMeterReadout, timelineMeterButton("+", 0.5));
    const truthMeterWrap = document.createElement("div");
    truthMeterWrap.title = "WAN timeline truth. Frames and seconds are synced from duration x FPS.";
    truthMeterWrap.style.cssText = [
        "display:grid",
        "grid-template-rows:1fr 1fr",
        "align-items:center",
        "justify-items:center",
        "height:28px",
        "min-width:112px",
        "padding:2px 8px",
        `border:1px solid ${purple.accent}`,
        "border-radius:5px",
        "background:linear-gradient(180deg,rgba(12,55,75,.96),rgba(4,18,28,.98))",
        "box-shadow:inset 0 1px 0 rgba(255,255,255,.20),0 0 0 1px rgba(0,0,0,.55),0 0 16px rgba(111,182,210,.22)",
        "box-sizing:border-box",
        "font:9px/1 monospace",
        "font-weight:950",
        "letter-spacing:0",
    ].join(";");
    const frameMeterReadout = document.createElement("span");
    frameMeterReadout.style.cssText = `color:${purple.play};white-space:nowrap;text-shadow:0 0 8px rgba(255,224,138,.55),0 1px 2px #000;`;
    const secondsMeterReadout = document.createElement("span");
    secondsMeterReadout.style.cssText = `color:#EAF8FF;white-space:nowrap;opacity:1;text-shadow:0 1px 2px #000;`;
    truthMeterWrap.append(frameMeterReadout, secondsMeterReadout);
    const multiTimelineControl = makeMultiTimelineControl();
    const addImageBtn = makeBtn("Add Image", "blue");
    const addTextBtn = makeBtn("Add Text", "violet");
    const addAudioBtn = makeBtn("Add Audio", "olive");
    const addTrackBtn = makeBtn("Add Audio Track", "sand");
    const collapseBtn = makeBtn(collapsed ? "Show Boxes" : "Collapse Boxes", "slate");
    const openEditorBtn = makeBtn("Open Editor", "teal");
    const importBoardBtn = makeBtn("Import Board", "violet");
    const saveBtn = makeBtn("Save Board", "green");
    const savePackageBtn = makeBtn("Save Package", "gold");
    const clearBtn = makeBtn("Clear Board", "danger");
    topActions.append(logBtn, globalOnlyBtn, multiTimelineControl, promptSizeWrap, timelineMeterWrap, addImageBtn, addTextBtn, addAudioBtn, addTrackBtn, collapseBtn, openEditorBtn, importBoardBtn, saveBtn, savePackageBtn, clearBtn);
    head.append(title, topActions);
    root.addEventListener("iamccs:cine-fullscreen", (event) => {
        openEditorBtn.textContent = event.detail?.open ? "Close Editor" : "Open Editor";
    });

    const promptWrap = document.createElement("label");
    promptWrap.style.cssText = `display:block;margin-bottom:8px;padding:8px;border:1px solid ${purple.borderSoft};background:${purple.panel};border-radius:6px;color:${purple.muted};font-size:11px;font-weight:800;`;
    const promptLabel = document.createElement("div");
    promptLabel.textContent = "Global prompt";
    promptLabel.style.marginBottom = "5px";
    const promptArea = document.createElement("textarea");
    promptArea.value = String(promptWidget?.value || "");
    promptArea.rows = 3;
    promptArea.style.cssText = inputBase() + `background:${purple.valueBg};border-color:${purple.border};color:${purple.valueText};resize:vertical;min-height:64px;font-weight:700;font-size:${promptFontSize(12)};`;
    node._iamccsCineShotboardV3LastPromptText = String(promptArea.value || "");
    promptArea.oninput = () => {
        node._iamccsCineShotboardV3LastPromptText = String(promptArea.value || "");
        setWidgetValue(node, "global_prompt", promptArea.value);
        node.properties = node.properties || {};
        node.properties.iamccs_v3_global_prompt_backup = promptArea.value;
    };
    promptArea.onchange = promptArea.oninput;
    promptArea.onblur = promptArea.oninput;
    promptWrap.append(promptLabel, promptArea);

    const settings = document.createElement("div");
    settings.style.cssText = "display:grid;grid-template-columns:repeat(10,minmax(86px,1fr));gap:10px;margin-bottom:14px;";
    const addSetting = (label, name, step, min) => {
        const wrap = document.createElement("label");
        wrap.style.cssText = `display:flex;flex-direction:column;gap:4px;color:${purple.muted};font-size:10px;font-weight:800;text-align:center;`;
        const span = document.createElement("span");
        span.textContent = label;
        const widget = getWidget(node, name);
        const max = name === "default_force" ? "2" : null;
        const ctrl = numberStepperControl(widget?.value ?? "", step, min, max, (value) => {
            const nextValue = name === "default_force" ? clampGuideStrength(value) : value;
            if (name === "default_force") applyDefaultForceToLinkedSegments(nextValue);
            setWidgetValue(node, name, nextValue);
            if (name === "duration_seconds") {
                setDurationSeconds(nextValue, "duration_control");
                enforceDurationMinimum();
            }
            if (name === "frame_rate") {
                setFrameRateValue(nextValue, "fps_control");
            }
            writeTimeline();
            draw();
        }, name === "duration_seconds" ? { liveInput: false } : {});
        if (name === "duration_seconds") durationValueControl = ctrl;
        styleValueControls(ctrl);
        wrap.append(span, ctrl);
        settings.appendChild(wrap);
    };
    addSetting("Duration", "duration_seconds", "1", "1");
    addSetting("FPS", "frame_rate", "1", "1");
    addSetting("Default motion", "default_force", "0.01", "1");
    addSetting("Width", "image_width", "32", "64");
    addSetting("Height", "image_height", "32", "64");
    const addSelectSetting = (label, name, options) => {
        const wrap = document.createElement("label");
        wrap.style.cssText = `display:flex;flex-direction:column;gap:4px;color:${purple.muted};font-size:10px;font-weight:800;text-align:center;`;
        const span = document.createElement("span");
        span.textContent = label;
        const ctrl = makeSelect(String(getWidget(node, name)?.value || options[0]), options, (value) => {
            setWidgetValue(node, name, value);
            writeTimeline();
            draw();
        });
        styleValueControls(ctrl);
        wrap.append(span, ctrl);
        settings.appendChild(wrap);
    };
    addSelectSetting("Resize", "image_resize_method", ["crop", "pad", "keep proportion", "stretch"]);
    addSetting("Multiple", "image_multiple_of", "1", "1");
    addSetting("Relay softness", "promptrelay_epsilon", "0.0001", "0.0001");
    const guidePolicyWrap = document.createElement("label");
    guidePolicyWrap.style.cssText = `display:flex;flex-direction:column;gap:4px;color:${purple.muted};font-size:10px;font-weight:800;text-align:center;`;
    const guidePolicyLabel = document.createElement("span");
    guidePolicyLabel.textContent = "Guides";
    const guidePolicy = makeSelect(String(getWidget(node, "guide_policy")?.value || "every_checked_row"), ["every_checked_row", "safe_core_guides", "prompt_only"], (value) => setWidgetValue(node, "guide_policy", value));
    styleValueControls(guidePolicy);
    guidePolicyWrap.append(guidePolicyLabel, guidePolicy);
    settings.appendChild(guidePolicyWrap);

    const parityWrap = document.createElement("label");
    parityWrap.style.cssText = `display:flex;flex-direction:column;gap:4px;color:${purple.muted};font-size:10px;font-weight:800;text-align:center;`;
    const parityLabel = document.createElement("span");
    parityLabel.textContent = "Mode";
    const parityMode = makeChoiceSelect(String(timeline.flfrealMode || "iamccs_enhanced"), [
        { value: "flfreal_parity", label: "FLFreal parity" },
        { value: "iamccs_enhanced", label: "IAMCCS enhanced" },
    ], setFlfrealMode);
    styleValueControls(parityMode);
    parityWrap.append(parityLabel, parityMode);
    settings.appendChild(parityWrap);

    timelineNotice = document.createElement("div");
    timelineNotice.style.cssText = `display:none;margin:-2px 0 8px 0;padding:7px 9px;border:1px solid ${purple.play};border-radius:6px;background:rgba(0,0,0,.28);color:#FFF1BE;font-size:11px;font-weight:800;`;

    const refsPanel = document.createElement("div");
    refsPanel.style.cssText = `margin-bottom:0;padding:0;border:0;background:transparent;display:none;`;
    const refsTitle = document.createElement("div");
    refsTitle.textContent = "References";
    refsTitle.style.cssText = `color:${purple.muted};font-size:11px;font-weight:800;margin-bottom:6px;`;
    const refsGrid = document.createElement("div");
    refsGrid.style.cssText = "display:flex;gap:7px;overflow-x:auto;min-height:74px;";
    refsPanel.append(refsTitle, refsGrid);

    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = "image/*";
    fileInput.multiple = true;
    fileInput.style.display = "none";
    const audioInput = document.createElement("input");
    audioInput.type = "file";
    audioInput.accept = "audio/*";
    audioInput.multiple = true;
    audioInput.style.display = "none";
    const boardInput = document.createElement("input");
    boardInput.type = "file";
    boardInput.accept = "application/json,.json";
    boardInput.style.display = "none";
    root.append(head, promptWrap, settings, timelineNotice, fileInput, audioInput, boardInput);

    const timelineViewport = document.createElement("div");
    timelineViewport.style.cssText = [
        "width:100%",
        "max-width:100%",
        "overflow-x:auto",
        "overflow-y:hidden",
        "border-radius:6px",
        "box-sizing:border-box",
        "scrollbar-gutter:stable",
    ].join(";");
    const timelineCanvas = document.createElement("div");
    timelineCanvas.style.cssText = [
        "position:relative",
        "min-width:0",
        "box-sizing:border-box",
    ].join(";");
    const frameRuler = document.createElement("div");
    frameRuler.title = "Frame ruler. This is synced to the same duration and FPS used by the WAN backend.";
    frameRuler.style.cssText = `height:34px;position:relative;border:1px solid ${purple.accent};border-bottom:0;background:linear-gradient(180deg,#0F3E55 0%,#061923 100%);border-radius:6px 6px 0 0;overflow:hidden;box-shadow:inset 0 1px 0 rgba(255,255,255,.18),0 0 0 1px rgba(0,0,0,.55);cursor:ew-resize;`;
    const ruler = document.createElement("div");
    ruler.title = "Seconds ruler. It shares the same timeline scale as the frame ruler above.";
    ruler.style.cssText = `height:36px;position:relative;border:1px solid ${purple.border};border-bottom:0;background:linear-gradient(180deg,#12384B 0%,#082231 58%,#051722 100%);border-radius:0;overflow:hidden;box-shadow:inset 0 1px 0 rgba(255,255,255,.12),0 0 0 1px rgba(0,0,0,.50);cursor:ew-resize;`;
    const timelineBox = document.createElement("div");
    timelineBox.title = "Double click in the image timeline to import a reference at that frame.";
    timelineBox.style.cssText = `position:relative;height:344px;border:1px solid ${purple.border};background:#061923;overflow:hidden;border-radius:0 0 6px 6px;margin-bottom:6px;box-shadow:inset 0 0 0 1px rgba(111,182,210,.10);`;
    const imageTrack = document.createElement("div");
    // imageTrack fills full width; endEdge marker overlays the last 4px so there is no dark gap at the right — By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    imageTrack.style.cssText = `position:absolute;left:0;right:0;top:0;height:254px;border-bottom:1px solid ${purple.borderSoft};background:linear-gradient(180deg,rgba(111,182,210,.15),rgba(6,25,35,.28));`;
    const actionTrack = document.createElement("div");
    actionTrack.style.cssText = "display:none;";
    const audioTracks = document.createElement("div");
    // audioTracks also fills full width matching imageTrack — By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    audioTracks.style.cssText = "position:absolute;left:0;right:0;top:254px;bottom:0;";
    timelineBox.append(imageTrack, audioTracks);
    // Timeline start/end edge markers — visible boundaries showing where the timeline begins and ends
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    const timelineStartEdge = document.createElement("div");
    timelineStartEdge.title = "Timeline start (frame 0)";
    timelineStartEdge.style.cssText = "position:absolute;left:0;top:0;bottom:0;width:4px;background:linear-gradient(180deg,#8DE7FF,rgba(141,231,255,.42));pointer-events:none;z-index:25;border-radius:2px 0 0 2px;box-shadow:2px 0 12px rgba(141,231,255,.34),0 0 0 1px rgba(141,231,255,.22);";
    const timelineEndEdge = document.createElement("div");
    timelineEndEdge.title = "Timeline end";
    timelineEndEdge.style.cssText = "position:absolute;right:0;top:0;bottom:0;width:4px;background:linear-gradient(180deg,#F3B34B,rgba(243,179,75,.44));pointer-events:none;z-index:25;border-radius:0 2px 2px 0;box-shadow:-2px 0 12px rgba(243,179,75,.32),0 0 0 1px rgba(243,179,75,.22);";
    timelineBox.append(timelineStartEdge, timelineEndEdge);
    const playbar = document.createElement("div");
    playbar.style.cssText = `display:flex;align-items:center;gap:9px;margin-bottom:8px;padding:8px 9px;border:1px solid ${purple.border};background:linear-gradient(180deg,#102D3D 0%,#071C28 100%);border-radius:7px;box-shadow:inset 0 1px 0 rgba(255,255,255,.12), inset 0 -10px 18px rgba(0,0,0,.22);`;
    const playBtn = makeBtn("Play");
    const loopBtn = makeBtn("Loop");
    const timeReadout = document.createElement("div");
    timeReadout.style.cssText = `width:86px;color:${purple.muted};font-size:11px;font-weight:800;`;
    const audioPlaybarControls = document.createElement("div");
    audioPlaybarControls.style.cssText = [
        "flex:0 0 auto",
        "display:flex",
        "align-items:center",
        "gap:6px",
        "min-height:28px",
        "padding:3px 6px",
        `border:1px solid ${purple.borderSoft}`,
        "border-radius:6px",
        "background:rgba(0,0,0,.16)",
        "box-sizing:border-box",
    ].join(";");
    const scrub = document.createElement("input");
    scrub.type = "range";
    scrub.min = "0";
    scrub.max = String(getTotalFrames());
    scrub.value = "0";
    scrub.step = "1";
    scrub.className = "iamccs-v3-analog-scrub";
    scrub.style.cssText = "flex:1 1 auto;min-width:180px;--iamccs-play-progress:0%;";
    const scrubStyle = document.createElement("style");
    scrubStyle.textContent = `
        .iamccs-v3-analog-scrub {
            appearance: none;
            height: 20px;
            border: 1px solid ${purple.border};
            border-radius: 999px;
            background:
                linear-gradient(90deg, ${purple.play} 0%, ${purple.play} var(--iamccs-play-progress), transparent var(--iamccs-play-progress), transparent 100%),
                repeating-linear-gradient(90deg, rgba(255,255,255,.22) 0, rgba(255,255,255,.22) 1px, transparent 1px, transparent 24px),
                linear-gradient(180deg, #061923 0%, #12384B 48%, #04111A 100%);
            box-shadow: inset 0 2px 5px rgba(0,0,0,.65), 0 0 0 1px rgba(141,231,255,.18);
            cursor: pointer;
        }
        .iamccs-v3-analog-scrub::-webkit-slider-runnable-track {
            height: 20px;
            background: transparent;
            border: 0;
        }
        .iamccs-v3-analog-scrub::-webkit-slider-thumb {
            appearance: none;
            width: 22px;
            height: 22px;
            margin-top: -1px;
            border: 2px solid #EAF8FF;
            border-radius: 999px;
            background: radial-gradient(circle at 35% 30%, #FFF8EC 0%, #FFE08A 42%, #B06B2B 100%);
            box-shadow: 0 2px 8px rgba(0,0,0,.65),0 0 12px rgba(255,224,138,.42);
        }
        .iamccs-v3-analog-scrub::-moz-range-track {
            height: 20px;
            background: transparent;
            border: 0;
        }
        .iamccs-v3-analog-scrub::-moz-range-thumb {
            width: 20px;
            height: 20px;
            border: 2px solid #EAF8FF;
            border-radius: 999px;
            background: #FFE08A;
            box-shadow: 0 2px 8px rgba(0,0,0,.65),0 0 12px rgba(255,224,138,.42);
        }
    `;
    let isPlaying = false;
    let isLooping = false;
    let playFrame = 0;
    let playTimer = null;
    let audioContext = null;
    let activeAudioNodes = [];
    let audioBufferCache = new Map();
    let waveformLoading = new Set();
    let playbackStartFrame = 0;
    let playbackStartTimestamp = 0;
    let dragState = null;
    let previewSegments = null;
    let previewAudioSegments = null;
    let pendingAudioInsertFrame = null;
    let pendingAudioTrack = 0;
    let drawRaf = 0;
    let transitionAppliedStamp = 0;
    playbar.append(scrubStyle, playBtn, loopBtn, timeReadout, audioPlaybarControls, scrub);
    bindMeterScrub(frameRuler);
    bindMeterScrub(ruler);
    timelineCanvas.append(frameRuler, ruler, timelineBox);
    timelineViewport.appendChild(timelineCanvas);
    // Timeline height resize handle — drag to expand/shrink timeline rows (slots + local prompts)
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    const timelineResizeHandle = document.createElement("div");
    timelineResizeHandle.title = "Drag up/down to resize the timeline height (expands slots and local prompts)";
    timelineResizeHandle.style.cssText = [
        "display:flex", "align-items:center", "justify-content:center",
        "height:10px", "margin:0 0 4px 0",
        `border:1px solid ${purple.borderSoft}`,
        "background:linear-gradient(180deg,rgba(108,101,90,.35),rgba(50,46,42,.35))",
        "cursor:row-resize", "user-select:none", "box-sizing:border-box",
        `color:${purple.muted}`, "font-size:9px", "letter-spacing:4px", "flex-shrink:0",
    ].join(";");
    timelineResizeHandle.innerHTML = `<span style="pointer-events:none;opacity:.45;">&bull;&bull;&bull;&bull;&bull;</span>`;
    timelineResizeHandle.addEventListener("pointerdown", (e) => {
        e.preventDefault(); e.stopPropagation();
        _tlResizeDragStartY = e.clientY;
        _tlResizeDragStartExtra = Math.max(0, Number(node.properties?.iamccs_v3_timeline_extra_height || 0));
        timelineResizeHandle.setPointerCapture(e.pointerId);
    });
    timelineResizeHandle.addEventListener("pointermove", (e) => {
        if (_tlResizeDragStartY === null) return;
        e.preventDefault();
        const delta = e.clientY - _tlResizeDragStartY;
        const newExtra = Math.max(0, Math.min(600, Math.round(_tlResizeDragStartExtra + delta)));
        node.properties = node.properties || {};
        node.properties.iamccs_v3_timeline_extra_height = newExtra;
        draw();
    });
    timelineResizeHandle.addEventListener("pointerup", () => {
        if (_tlResizeDragStartY === null) return;
        _tlResizeDragStartY = null;
        writeTimeline({ force: true });
    });
    timelineResizeHandle.addEventListener("pointercancel", () => { _tlResizeDragStartY = null; });
    root.append(timelineViewport, timelineResizeHandle, playbar);

    const inspector = document.createElement("div");
    inspector.style.cssText = `display:${collapsed ? "none" : "block"};width:100%;min-width:0;box-sizing:border-box;`;
    const boxList = document.createElement("div");
    boxList.style.cssText = `border:1px solid ${purple.borderSoft};background:${purple.panel};border-radius:6px;padding:6px;overflow:visible;box-sizing:border-box;width:100%;min-width:0;`;
    const editBox = document.createElement("div");
    editBox.style.cssText = "display:none;";
    inspector.append(boxList, editBox);
    root.append(inspector);

    function drawRefs() {
        refsGrid.innerHTML = "";
        const paths = refPaths();
        paths.forEach((path, index) => {
            const card = document.createElement("div");
            card.style.cssText = `position:relative;flex:0 0 132px;height:72px;border:1px solid ${purple.borderSoft};border-radius:5px;overflow:hidden;background:#050308;`;
            const img = document.createElement("img");
            img.src = previewUrlForPath(path);
            img.style.cssText = "width:100%;height:100%;object-fit:cover;display:block;";
            const badge = document.createElement("div");
            badge.textContent = `ref ${index + 1}`;
            badge.style.cssText = "position:absolute;left:4px;bottom:3px;background:rgba(0,0,0,.72);color:#fff;font-size:10px;padding:1px 5px;border-radius:3px;";
            card.append(img, badge);
            refsGrid.appendChild(card);
        });
    }

    function frameLabel(frame) {
        return `${(frame / getFps()).toFixed(2)}s`;
    }

    function chooseFrameRulerStep(total, viewportWidth) {
        const px = Math.max(1, Number(viewportWidth || timelineViewport?.clientWidth || 0) || 1);
        const targetLabels = Math.max(4, Math.min(14, Math.floor(px / 120)));
        const raw = Math.max(1, total / targetLabels);
        const steps = [1, 2, 4, 5, 8, 10, 12, 16, 20, 24, 30, 32, 40, 48, 60, 64, 80, 96, 120, 121, 160, 192, 240];
        return steps.find((step) => step >= raw) || Math.ceil(raw / 60) * 60;
    }

    function drawFrameRuler() {
        frameRuler.innerHTML = "";
        const total = Math.max(1, getTotalFrames());
        const fps = Math.max(1, getFps());
        const visibleWidth = Math.max(1, Number(timelineViewport?.clientWidth || timelineCanvas?.clientWidth || 0) || 1);
        const majorStep = chooseFrameRulerStep(total, visibleWidth);
        const minorStep = 1;
        const baseLine = document.createElement("div");
        baseLine.style.cssText = `position:absolute;left:0;right:0;bottom:0;height:2px;background:${purple.accent};opacity:.95;pointer-events:none;box-shadow:0 0 7px rgba(141,231,255,.38);`;
        frameRuler.appendChild(baseLine);
        for (let frame = 0; frame <= total; frame += minorStep) {
            const major = frame % majorStep === 0 || frame === 0 || frame === total;
            const pos = (frame / total) * 100;
            const tick = document.createElement("div");
            tick.style.cssText = [
                "position:absolute",
                `left:calc(${pos}% - ${major ? 1 : 0.5}px)`,
                "bottom:0",
                `width:${major ? 2 : 1}px`,
                `height:${major ? 30 : 13}px`,
                `background:${major ? purple.play : "#6FB6D2"}`,
                `opacity:${major ? 1 : 0.70}`,
                major ? "box-shadow:0 0 8px rgba(255,224,138,.45)" : "box-shadow:none",
                "pointer-events:none",
            ].join(";");
            frameRuler.appendChild(tick);
            if (!major) continue;
            const label = document.createElement("div");
            label.style.cssText = [
                "position:absolute",
                `left:${pos}%`,
                "top:4px",
                frame >= total - 1 ? "transform:translateX(calc(-100% - 6px))" : "transform:translateX(6px)",
                `color:${frame % fps === 0 ? purple.play : "#EAF8FF"}`,
                "font-size:11px",
                "font-weight:950",
                "line-height:1",
                "text-shadow:0 1px 2px rgba(0,0,0,.72)",
                "white-space:nowrap",
                "pointer-events:none",
            ].join(";");
            label.textContent = `F${Math.round(frame)}`;
            frameRuler.appendChild(label);
        }
        const playPos = (playFrame / total) * 100;
        const marker = document.createElement("div");
        marker.style.cssText = `position:absolute;left:calc(${playPos}% - 2px);top:0;bottom:0;width:4px;background:${purple.play};box-shadow:0 0 0 1px rgba(0,0,0,.72),0 0 14px rgba(255,224,138,.75);pointer-events:none;z-index:20;`;
        frameRuler.appendChild(marker);
        const info = document.createElement("div");
        info.style.cssText = `position:absolute;right:7px;top:5px;color:#FFFFFF;font-size:10px;font-weight:950;background:rgba(0,0,0,.78);padding:3px 6px;border:1px solid ${purple.accent};border-radius:4px;pointer-events:none;text-shadow:0 1px 2px #000;`;
        info.textContent = `${total}f / ${fps}fps`;
        frameRuler.appendChild(info);
    }

    function drawRuler() {
        ruler.innerHTML = "";
        const total = getTotalFrames();
        const seconds = Math.max(0.001, getDuration());
        const step = 0.5;
        const baseLine = document.createElement("div");
        baseLine.style.cssText = `position:absolute;left:0;right:0;bottom:0;height:2px;background:${purple.accent};opacity:.95;pointer-events:none;box-shadow:0 0 7px rgba(141,231,255,.38);`;
        ruler.appendChild(baseLine);
        for (let s = 0; s <= seconds + 0.001; s += step) {
            const major = Math.abs(s - Math.round(s)) < 0.001;
            const five = Math.abs((s / 5) - Math.round(s / 5)) < 0.001;
            const pos = (s / seconds) * 100;
            const tick = document.createElement("div");
            tick.style.cssText = [
                "position:absolute",
                `left:calc(${pos}% - ${five ? 1 : 0.5}px)`,
                "bottom:0",
                `width:${five ? 2 : 1}px`,
                `height:${five ? 34 : major ? 27 : 12}px`,
                `background:${five ? purple.play : major ? purple.accent : "#6FB6D2"}`,
                `opacity:${five ? 1 : major ? 0.95 : 0.65}`,
                five ? "box-shadow:0 0 8px rgba(255,224,138,.45)" : "box-shadow:none",
                "pointer-events:none",
            ].join(";");
            ruler.appendChild(tick);
            if (!major) continue;
            const label = document.createElement("div");
            label.style.cssText = [
                "position:absolute",
                `left:${pos}%`,
                "top:4px",
                "transform:translateX(5px)",
                `color:${five ? purple.play : "#EAF8FF"}`,
                "font-size:11px",
                "font-weight:950",
                "line-height:1",
                "text-shadow:0 1px 2px #000",
                "pointer-events:none",
            ].join(";");
            label.textContent = `${s.toFixed(0)}s`;
            ruler.appendChild(label);
        }
        const playPos = (playFrame / Math.max(1, total)) * 100;
        const marker = document.createElement("div");
        marker.style.cssText = `position:absolute;left:calc(${playPos}% - 2px);top:0;bottom:0;width:4px;background:${purple.play};box-shadow:0 0 0 1px rgba(0,0,0,.72),0 0 14px rgba(255,224,138,.75);pointer-events:none;z-index:20;`;
        ruler.appendChild(marker);
        const last = document.createElement("div");
        last.style.cssText = `position:absolute;right:6px;top:5px;color:#FFFFFF;font-size:10px;font-weight:950;background:rgba(0,0,0,.78);padding:3px 6px;border:1px solid ${purple.accent};border-radius:4px;pointer-events:none;text-shadow:0 1px 2px #000;`;
        last.textContent = `${seconds.toFixed(2)}s`;
        ruler.appendChild(last);
    }

    function updatePlayUI() {
        const total = getTotalFrames();
        playFrame = Math.max(0, Math.min(total, Math.round(playFrame)));
        scrub.max = String(total);
        scrub.value = String(playFrame);
        scrub.style.setProperty("--iamccs-play-progress", `${total ? (playFrame / total) * 100 : 0}%`);
        timeReadout.textContent = `${(playFrame / getFps()).toFixed(2)}s`;
        playBtn.textContent = isPlaying ? "Pause" : "Play";
        loopBtn.style.borderColor = isLooping ? purple.play : purple.border;
        loopBtn.style.color = isLooping ? "#FFF2B8" : purple.text;
    }

    function setPlayFrameFromMeterEvent(event, meterElement) {
        if (!meterElement) return;
        event?.preventDefault?.();
        event?.stopPropagation?.();
        const rect = meterElement.getBoundingClientRect();
        const ratio = Math.max(0, Math.min(1, (Number(event.clientX || 0) - rect.left) / Math.max(1, rect.width)));
        playFrame = Math.max(0, Math.min(getTotalFrames(), Math.round(ratio * getTotalFrames())));
        if (isPlaying) {
            playbackStartFrame = playFrame;
            playbackStartTimestamp = performance.now();
            scheduleAudioFromFrame(playFrame);
        }
        draw();
    }

    function bindMeterScrub(meterElement) {
        if (!meterElement || meterElement._iamccsMeterScrubBound) return;
        meterElement._iamccsMeterScrubBound = true;
        meterElement.addEventListener("pointerdown", (event) => {
            setPlayFrameFromMeterEvent(event, meterElement);
            const move = (moveEvent) => setPlayFrameFromMeterEvent(moveEvent, meterElement);
            const finish = () => {
                window.removeEventListener("pointermove", move, true);
                window.removeEventListener("pointerup", finish, true);
                window.removeEventListener("pointercancel", finish, true);
            };
            window.addEventListener("pointermove", move, { passive: false, capture: true });
            window.addEventListener("pointerup", finish, { passive: false, capture: true });
            window.addEventListener("pointercancel", finish, { passive: false, capture: true });
        }, { passive: false, capture: true });
    }

    function audioPeakValue(raw) {
        if (raw && typeof raw === "object") {
            const min = Math.abs(Number(raw.min) || 0);
            const max = Math.abs(Number(raw.max) || 0);
            const rms = Math.abs(Number(raw.rms) || 0);
            return Math.max(min, max, rms);
        }
        return Math.abs(Number(raw) || 0);
    }

    function peaksFromBuffer(audioBuffer, peakCount) {
        const channelCount = Math.max(1, Number(audioBuffer?.numberOfChannels || 1));
        const frameCount = Math.max(1, Number(audioBuffer?.length || 1));
        const count = Math.max(64, Math.round(Number(peakCount || 200) || 200));
        const peaks = [];
        for (let i = 0; i < count; i += 1) {
            const start = Math.floor((i / count) * frameCount);
            const end = Math.max(start + 1, Math.floor(((i + 1) / count) * frameCount));
            let min = 0;
            let max = 0;
            let sum = 0;
            let n = 0;
            for (let channelIndex = 0; channelIndex < channelCount; channelIndex += 1) {
                const channel = audioBuffer.getChannelData(channelIndex);
                for (let j = start; j < end; j += 1) {
                    const value = Number(channel[j] || 0);
                    min = Math.min(min, value);
                    max = Math.max(max, value);
                    sum += value * value;
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
    }

    function ensureSegmentWaveform(seg) {
        if (!seg || !audioSegmentHasMedia(seg) || waveformLoading.has(seg.id)) return;
        const sourceKey = String(seg.audioFile || `${seg.id || "audio"}:${String(seg.audioB64 || "").slice(0, 64)}`);
        const hasDecodedPeaks = Array.isArray(seg.waveformPeaks)
            && seg.waveformPeaks.length > 8
            && seg.waveformPeaks.some((item) => item && typeof item === "object" && Number.isFinite(Number(item.min)) && Number.isFinite(Number(item.max)));
        if (seg._iamccsWaveformDecodedKey === sourceKey && seg.waveformReal === true && hasDecodedPeaks) return;
        if (seg._iamccsWaveformFailedKey === sourceKey) return;
        // Never trust serialized/fallback peaks as the visual truth. Decode the actual
        // current media source so the waveform remains useful while trimming.
        seg.waveformReal = false;
        waveformLoading.add(seg.id);
        audioBufferForSegment(seg)
            .then((buffer) => {
                if (!buffer) throw new Error("decoded audio buffer unavailable");
                seg.audioDurationFrames = Math.max(1, Math.round(Number(buffer.duration || 0) * getFps()));
                seg.waveformPeaks = peaksFromBuffer(buffer, Math.max(900, Math.min(2200, Math.round(Number(buffer.duration || 0) * 70))));
                seg.waveformReal = true;
                seg._iamccsWaveformDecodedKey = sourceKey;
                delete seg._iamccsWaveformFailedKey;
                waveformLoading.delete(seg.id);
                draw();
            })
            .catch((err) => {
                waveformLoading.delete(seg.id);
                seg.waveformPeaks = [];
                seg.waveformReal = false;
                seg._iamccsWaveformFailedKey = sourceKey;
                console.warn("[IAMCCS Cine Shotboard V3] waveform decode failed", seg?.audioFile || seg?.fileName || seg?.id, err);
            });
    }

    const effectiveClipGain = (seg) => {
        const peaks = Array.isArray(seg?.waveformPeaks) ? seg.waveformPeaks.map((item) => audioPeakValue(item)) : [];
        const rawPeak = peaks.length ? Math.max(...peaks) : 0;
        const gain = Math.max(0, Math.min(2, Number(seg?.gain ?? seg?.volume ?? 1) || 1));
        const levelGain = seg?.normalizeAudio && rawPeak > 0.0001 ? Math.min(4, 0.92 / rawPeak) : 1;
        return gain * levelGain;
    };

    const mixedAudioLivePeak = () => {
        let peak = 0;
        const fps = getFps();
        const frame = Math.max(0, Math.round(Number(playFrame || 0)));
        for (const seg of timeline.audioSegments || []) {
            if (!audioSegmentHasMedia(seg) || seg.placeholder) continue;
            const start = Math.max(0, Math.round(Number(seg.start || 0)));
            const length = Math.max(1, Math.round(Number(seg.length || 1)));
            if (frame < start || frame > start + length) continue;
            const peaks = Array.isArray(seg.waveformPeaks) ? seg.waveformPeaks.map((item) => audioPeakValue(item)) : [];
            if (!peaks.length) continue;
            const localFrame = frame - start + Math.max(0, Math.round(Number(seg.trimStart || 0)));
            const audioDuration = Math.max(1, Math.round(Number(seg.audioDurationFrames || length)));
            const index = Math.max(0, Math.min(peaks.length - 1, Math.floor((localFrame / audioDuration) * peaks.length)));
            peak += peaks[index] * effectiveClipGain(seg);
        }
        const masterGain = Math.max(0, Math.min(2, Number(timeline.masterAudioGain ?? 1) || 1));
        const masterNormalize = Boolean(timeline.masterAudioNormalize);
        const normalized = peak * masterGain;
        return Math.max(0, Math.min(1, masterNormalize && normalized > 0.0001 ? Math.min(1, normalized * Math.min(4, 0.92 / normalized)) : normalized));
    };

    function drawAudioPlaybarControls() {
        audioPlaybarControls.innerHTML = "";
        const hasAudio = (timeline.audioSegments || []).some((seg) => audioSegmentHasMedia(seg) && !seg.placeholder);
        const miniLabel = document.createElement("span");
        miniLabel.textContent = "MASTER";
        miniLabel.style.cssText = `color:${purple.muted};font-size:9px;font-weight:900;line-height:1;`;
        audioPlaybarControls.appendChild(miniLabel);
        if (!hasAudio) {
            const empty = document.createElement("button");
            empty.type = "button";
            empty.textContent = "+";
            empty.title = "Import audio";
            empty.style.cssText = `width:24px;height:22px;border:1px dashed ${purple.border};border-radius:5px;background:${purple.valueBg};color:${purple.valueText};font-size:15px;font-weight:900;cursor:pointer;line-height:1;`;
            empty.onpointerdown = (event) => { event.preventDefault(); event.stopPropagation(); };
            empty.onclick = (event) => {
                event.preventDefault();
                event.stopPropagation();
                pendingAudioInsertFrame = null;
                pendingAudioTrack = 0;
                audioInput.click();
            };
            audioPlaybarControls.appendChild(protectControlDrag(empty));
            return;
        }
        const makeMasterNumber = (label, key, step = "0.05", width = 42) => {
            const wrap = document.createElement("label");
            wrap.title = label;
            wrap.style.cssText = "display:flex;align-items:center;gap:3px;color:#EDE3D0;font-size:8px;font-weight:900;line-height:1;";
            const text = document.createElement("span");
            text.textContent = label;
            const input = document.createElement("input");
            input.type = "number";
            input.step = step;
            input.value = timeline[key] ?? 1;
            input.style.cssText = `width:${width}px;height:22px;box-sizing:border-box;border:1px solid ${purple.border};border-radius:4px;background:${purple.valueBg};color:${purple.valueText};font-size:10px;font-weight:800;text-align:center;padding:0 3px;`;
            input.onpointerdown = (event) => event.stopPropagation();
            input.onchange = input.oninput = () => {
                const raw = Number(input.value || 0);
                timeline[key] = Math.max(0, Math.min(2, raw));
                writeTimeline({ force: true });
            };
            protectControlDrag(input);
            wrap.append(text, input);
            return wrap;
        };
        const level = document.createElement("label");
        level.title = "Normalize final mixed audio";
        level.style.cssText = "display:flex;align-items:center;gap:3px;color:#EDE3D0;font-size:8px;font-weight:900;line-height:1;";
        const levelCheck = document.createElement("input");
        levelCheck.type = "checkbox";
        levelCheck.checked = Boolean(timeline.masterAudioNormalize);
        levelCheck.style.cssText = `width:15px;height:15px;accent-color:${purple.accent};cursor:pointer;`;
        levelCheck.onpointerdown = (event) => event.stopPropagation();
        levelCheck.onchange = () => {
            timeline.masterAudioNormalize = Boolean(levelCheck.checked);
            writeTimeline({ force: true });
            draw();
        };
        level.append(document.createTextNode("Level"), levelCheck);
        const meterShell = document.createElement("div");
        meterShell.title = "Realtime final audio peak";
        meterShell.style.cssText = "width:42px;height:8px;border:1px solid rgba(255,255,255,.2);border-radius:999px;background:rgba(0,0,0,.35);overflow:hidden;";
        const meterFill = document.createElement("div");
        meterFill.style.cssText = `width:${Math.round(mixedAudioLivePeak() * 100)}%;height:100%;background:linear-gradient(90deg,#5EB9B4,#F4D49E,#D8792B);`;
        meterShell.appendChild(meterFill);
        audioPlaybarControls.append(
            makeMasterNumber("Vol", "masterAudioGain", "0.05", 42),
            protectControlDrag(level),
            meterShell
        );
    }

    function stopAudioNodes() {
        activeAudioNodes.forEach((node) => {
            try { node.stop(); } catch {}
            try { node.disconnect(); } catch {}
        });
        activeAudioNodes = [];
    }

    async function audioBufferForSegment(seg) {
        if (!seg || (!seg.audioFile && !seg.audioB64)) return null;
        const key = seg.audioFile || `${seg.id || "audio"}:${String(seg.audioB64 || "").slice(0, 64)}`;
        if (audioBufferCache.has(key)) return audioBufferCache.get(key);
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextClass) return null;
        if (!audioContext) audioContext = new AudioContextClass();
        let arrayBuffer = null;
        if (seg.audioFile) {
            const parts = String(seg.audioFile || "").split("/");
            const filename = parts.pop() || "";
            const subfolder = parts.join("/");
            const declaredType = String(seg.audioUploadType || seg.uploadType || seg.storageType || "").toLowerCase();
            const storageTypes = [...new Set([
                ["input", "output", "temp"].includes(declaredType) ? declaredType : "",
                "input",
                "output",
                "temp",
            ].filter(Boolean))];
            let lastStatus = "not found";
            for (const storageType of storageTypes) {
                const audioUrl = api.apiURL(`/view?filename=${encodeURIComponent(filename)}&type=${encodeURIComponent(storageType)}&subfolder=${encodeURIComponent(subfolder)}`);
                const resp = await fetch(audioUrl);
                if (!resp.ok) {
                    lastStatus = `${storageType}:${resp.status}`;
                    continue;
                }
                arrayBuffer = await resp.arrayBuffer();
                seg.audioUploadType = storageType;
                break;
            }
            if (!arrayBuffer) throw new Error(`audio fetch failed: ${lastStatus}`);
        } else if (seg.audioB64) {
            let encoded = String(seg.audioB64 || "");
            if (encoded.includes(",")) encoded = encoded.split(",").pop();
            const binary = window.atob(encoded);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
            arrayBuffer = bytes.buffer;
        }
        if (!arrayBuffer) return null;
        const decoded = await audioContext.decodeAudioData(arrayBuffer.slice(0));
        audioBufferCache.set(key, decoded);
        return decoded;
    }

    async function scheduleAudioFromFrame(frame) {
        stopAudioNodes();
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextClass) return;
        if (!audioContext) audioContext = new AudioContextClass();
        try { if (audioContext.state !== "running") await audioContext.resume(); } catch {}
        const fps = getFps();
        const startFrame = Math.max(0, Math.round(Number(frame || 0)));
        const playToken = Symbol("audio_play");
        scheduleAudioFromFrame._token = playToken;
        for (const seg of timeline.audioSegments || []) {
            const segStart = Math.max(0, Math.round(Number(seg.start || 0)));
            const segLength = Math.max(1, Math.round(Number(seg.length || 1)));
            const segEnd = segStart + segLength;
            if (segEnd <= startFrame) continue;
            try {
                const buffer = await audioBufferForSegment(seg);
                if (!buffer || scheduleAudioFromFrame._token !== playToken || !isPlaying) return;
                const source = audioContext.createBufferSource();
                const gain = audioContext.createGain();
                const peaks = Array.isArray(seg.waveformPeaks) ? seg.waveformPeaks.map((item) => audioPeakValue(item)) : [];
                const rawPeak = peaks.length ? Math.max(...peaks) : 0;
                const clipGain = effectiveClipGain(seg);
                const masterGain = Math.max(0, Math.min(2, Number(timeline.masterAudioGain ?? 1) || 1));
                const masterLevelGain = timeline.masterAudioNormalize && rawPeak > 0.0001 ? Math.min(4, 0.92 / rawPeak) : 1;
                gain.gain.value = Math.max(0, Math.min(8, clipGain * masterGain * masterLevelGain));
                source.buffer = buffer;
                source.connect(gain);
                gain.connect(audioContext.destination);
                const skipFrames = Math.max(0, startFrame - segStart);
                const fileOffset = (Math.max(0, Number(seg.trimStart || 0)) + skipFrames) / fps;
                const wait = Math.max(0, segStart - startFrame) / fps;
                const playDuration = Math.max(0, (segLength - skipFrames) / fps);
                if (playDuration <= 0) continue;
                source.start(audioContext.currentTime + wait, Math.max(0, fileOffset), playDuration);
                source.onended = () => {
                    activeAudioNodes = activeAudioNodes.filter((item) => item !== source);
                };
                activeAudioNodes.push(source);
            } catch (err) {
                console.warn("[IAMCCS Cine Shotboard V3] audio preview failed", err);
            }
        }
    }

    function stopPlayback() {
        isPlaying = false;
        if (playTimer) clearInterval(playTimer);
        playTimer = null;
        stopAudioNodes();
        updatePlayUI();
    }

    function startPlayback() {
        if (isPlaying) return;
        isPlaying = true;
        if (playFrame >= getTotalFrames()) playFrame = 0;
        playbackStartFrame = playFrame;
        playbackStartTimestamp = performance.now();
        scheduleAudioFromFrame(playFrame);
        playTimer = setInterval(() => {
            const elapsedFrames = Math.floor(((performance.now() - playbackStartTimestamp) / 1000) * getFps());
            playFrame = playbackStartFrame + elapsedFrames;
            if (playFrame > getTotalFrames()) {
                if (isLooping) {
                    playFrame = 0;
                    playbackStartFrame = 0;
                    playbackStartTimestamp = performance.now();
                    scheduleAudioFromFrame(0);
                }
                else {
                    playFrame = getTotalFrames();
                    stopPlayback();
                }
            }
            draw();
        }, Math.max(16, Math.round(1000 / Math.max(12, getFps()))));
        updatePlayUI();
    }

    const cloneSegments = (items) => JSON.parse(JSON.stringify(items || []));
    const activeVisualSegments = () => previewSegments || timeline.segments || [];
    const activeAudioSegments = () => previewAudioSegments || timeline.audioSegments || [];
    const visibleAudioSegments = () => {
        const items = activeAudioSegments();
        const media = items.filter((seg) => audioSegmentHasMedia(seg) && !seg.placeholder);
        if (!media.length) return items;
        return items.filter((seg) => {
            if (audioSegmentHasMedia(seg) && !seg.placeholder) return true;
            return !media.some((clip) =>
                Number(clip.track || 0) === Number(seg?.track || 0)
                && segmentRangesOverlap(clip.start, clip.length, seg?.start, seg?.length)
            );
        });
    };
    function isTimelineImageSegment(seg) {
        return String(seg?.type || "image") === "image" && !seg?.placeholder;
    }
    function isActionBridgeRelaySegment(seg) {
        return String(seg?.type || "") === "text" && Boolean(seg?.actionBridgeSourceId);
    }
    function relayKindOf(seg) {
        const raw = String(seg?.relay_kind || seg?.relayKind || "").trim().toLowerCase();
        if (raw) return raw;
        if (seg?.slotRelay) return "slot";
        if (seg?.transitionRelay) return "transition";
        return String(seg?.type || "") === "text" ? "transition" : "";
    }
    function isSlotRelaySegment(seg) {
        return String(seg?.type || "") === "text" && relayKindOf(seg) === "slot";
    }
    function isTransitionRelaySegment(seg) {
        return String(seg?.type || "") === "text" && relayKindOf(seg) !== "slot";
    }
    function normalizeV3RelayOnlySegment(seg) {
        const next = { ...(seg || {}) };
        if (isActionBridgeRelaySegment(next)) {
            delete next.actionBridgeSourceId;
            delete next.actionBridgeSourceLabel;
            if (!String(next.label || "").trim() || /^action_bridge/i.test(String(next.label || ""))) next.label = "text_relay_slot";
        }
        next.step_transition_enabled = false;
        next.step_transition_type = "off";
        next.step_transition_prompt = "";
        next.step_transition_duration = 0;
        next.step_transition_arrival = "auto";
        next.step_transition_auto_fit = true;
        return next;
    }
    function neutralizeLegacyStepTransitions() {
        timeline.segments = (timeline.segments || []).map((seg) => normalizeV3RelayOnlySegment(seg));
    }
    function sortedActionBridgeSources(items = timeline.segments || []) {
        return (items || [])
            .filter(isTimelineImageSegment)
            .slice()
            .sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
    }
    const sortedTimelineVisualSegments = () => sortedActionBridgeSources(timeline.segments || []);
    const sortedTimelineDirectorSegments = () => (timeline.segments || [])
        .filter((seg) => String(seg.type || "image") !== "audio" && !seg.placeholder)
        .slice()
        .sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
    function generationWindowForImage(seg) {
        const start = Math.max(0, Math.round(Number(seg?.start || 0)));
        const length = Math.max(1, Math.round(Number(seg?.length || 1)));
        if (!isTimelineImageSegment(seg)) return { start, end: start + length, length };
        const images = (timeline.segments || [])
            .filter(isTimelineImageSegment)
            .slice()
            .sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
        const index = images.findIndex((item) => String(item.id || "") === String(seg.id || ""));
        if (index >= 0 && index + 1 < images.length) {
            const end = Math.max(start + 1, Math.round(Number(images[index + 1].start || (start + length))));
            return { start, end, length: Math.max(1, end - start) };
        }
        return { start, end: start + length, length };
    }
    function generationFrameFor(seg, absoluteFrame) {
        const win = generationWindowForImage(seg);
        return Math.max(0, Math.round(Number(absoluteFrame || 0) - win.start));
    }
    function generationOwnerForFrame(frame) {
        const absolute = Math.max(0, Math.round(Number(frame || 0)));
        const images = (timeline.segments || [])
            .filter(isTimelineImageSegment)
            .slice()
            .sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
        for (const image of images) {
            const win = generationWindowForImage(image);
            if (absolute >= win.start && absolute < win.end) return image;
        }
        return null;
    }
    function setGenerationWindowLength(seg, nextLength) {
        if (!isTimelineImageSegment(seg)) return false;
        const win = generationWindowForImage(seg);
        const desired = Math.max(1, Math.round(Number(nextLength || win.length)));
        const delta = desired - win.length;
        if (!delta) return false;
        const nextImage = (timeline.segments || [])
            .filter(isTimelineImageSegment)
            .slice()
            .sort((a, b) => Number(a.start || 0) - Number(b.start || 0))
            .find((item) => Number(item.start || 0) >= win.end && String(item.id || "") !== String(seg.id || ""));
        if (!nextImage) {
            seg.length = desired;
            return true;
        }
        const boundary = Math.max(win.start + 1, Math.round(Number(nextImage.start || win.end) + delta));
        const shift = boundary - Math.round(Number(nextImage.start || win.end));
        if (!shift) return false;
        (timeline.segments || []).forEach((item) => {
            if (String(item.id || "") === String(seg.id || "")) return;
            if (Number(item.start || 0) >= Number(nextImage.start || 0)) {
                item.start = Math.max(0, Math.round(Number(item.start || 0) + shift));
            }
        });
        return true;
    }
    const stepTransitionAvailableSeconds = (seg) => {
        const fps = getFps();
        const sorted = sortedTimelineVisualSegments();
        const index = sorted.findIndex((item) => item.id === seg?.id);
        if (index < 0 || index >= sorted.length - 1) return 0;
        const next = sorted[index + 1];
        return Math.max(0, (Number(next.start || 0) - Number(seg.start || 0)) / fps);
    };
    const fitStepTransitionDuration = (seg, seconds) => {
        const fps = getFps();
        const desiredFrames = Math.max(1, Math.round(Math.max(0, Number(seconds || 0)) * fps));
        if (!seg || desiredFrames <= 1) return false;
        const sorted = sortedTimelineVisualSegments();
        const index = sorted.findIndex((item) => item.id === seg.id);
        if (index < 0 || index >= sorted.length - 1) return false;
        const next = sorted[index + 1];
        const requiredNextStart = Math.round(Number(seg.start || 0) + desiredFrames);
        const nextStart = Math.round(Number(next.start || 0));
        const delta = requiredNextStart - nextStart;
        const idsToShift = new Set(sorted.slice(index + 1).map((item) => item.id));
        timeline.segments = (timeline.segments || []).map((item) => {
            if (item.id === seg.id) return { ...item, length: Math.max(1, desiredFrames) };
            if (idsToShift.has(item.id) && delta !== 0) return { ...item, start: Math.max(0, Math.round(Number(item.start || 0) + delta)) };
            return item;
        });
        seg.length = Math.max(1, desiredFrames);
        ensureDurationForFrames(endOfSegments(timeline.segments));
        showTimelineNotice(`Applied transition timing: ${(desiredFrames / fps).toFixed(2)}s window.`);
        return delta !== 0 || Number(seg.length || 1) !== desiredFrames;
    };
    const scheduleDraw = () => {
        if (drawRaf) return;
        drawRaf = requestAnimationFrame(() => {
            drawRaf = 0;
            draw();
        });
    };

    function applyCenterDragPhysics(initItems, targetId, targetStart, pointerFrame, durationFrames) {
        const items = cloneSegments(initItems);
        const targetIndex = items.findIndex((item) => item.id === targetId);
        if (targetIndex < 0) return items;
        const dragged = items[targetIndex];
        const maxStart = Math.max(0, durationFrames - Number(dragged.length || 1));
        let clampedStart = Math.max(0, Math.min(Math.round(targetStart), maxStart));
        const base = items.filter((item) => item.id !== dragged.id);

        let insertIndex = base.length;
        for (let i = 0; i < base.length; i += 1) {
            const center = Number(base[i].start || 0) + Number(base[i].length || 1) / 2;
            if (pointerFrame < center) {
                insertIndex = i;
                break;
            }
        }

        const leftBound = insertIndex > 0 ? Number(base[insertIndex - 1].start || 0) + Number(base[insertIndex - 1].length || 1) : 0;
        const rightBound = insertIndex < base.length ? Number(base[insertIndex].start || 0) : durationFrames;
        if (rightBound - leftBound >= Number(dragged.length || 1)) {
            clampedStart = Math.max(leftBound, Math.min(clampedStart, rightBound - Number(dragged.length || 1)));
        } else {
            clampedStart = (leftBound + rightBound) / 2 - Number(dragged.length || 1) / 2;
        }

        const test = [];
        for (let i = 0; i < insertIndex; i += 1) test.push({ ...base[i], original_start: Number(base[i].start || 0) });
        test.push({ ...dragged, start: clampedStart, original_start: clampedStart });
        const draggedIndex = insertIndex;
        for (let i = insertIndex; i < base.length; i += 1) test.push({ ...base[i], original_start: Number(base[i].start || 0) });

        for (let i = draggedIndex + 1; i < test.length; i += 1) {
            const prev = test[i - 1];
            test[i].start = Math.max(Number(test[i].original_start || 0), Number(prev.start || 0) + Number(prev.length || 1));
        }
        for (let i = draggedIndex - 1; i >= 0; i -= 1) {
            const next = test[i + 1];
            test[i].start = Math.min(Number(test[i].original_start || 0), Number(next.start || 0) - Number(test[i].length || 1));
        }

        let rightCursor = durationFrames;
        for (let i = test.length - 1; i >= 0; i -= 1) {
            if (Number(test[i].start || 0) + Number(test[i].length || 1) > rightCursor) {
                test[i].start = rightCursor - Number(test[i].length || 1);
            }
            rightCursor = Number(test[i].start || 0);
        }
        let leftCursor = 0;
        for (let i = 0; i < test.length; i += 1) {
            if (Number(test[i].start || 0) < leftCursor) test[i].start = leftCursor;
            leftCursor = Number(test[i].start || 0) + Number(test[i].length || 1);
        }

        return test.map((item) => {
            const clean = { ...item, start: Math.round(Number(item.start || 0)) };
            delete clean.original_start;
            return clean;
        });
    }

    function edgeDragPreview(initItems, targetId, dragDelta, edge, durationFrames) {
        const items = cloneSegments(initItems).sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
        const index = items.findIndex((item) => item.id === targetId);
        if (index < 0) return items;
        const target = items[index];
        const minLength = 1;
        if (isSlotRelaySegment(target)) {
            const parent = items.find((item) => String(item.id || "") === String(target.parentSegmentId || "") && isTimelineImageSegment(item));
            if (!parent) return items;
            const parentStart = Math.max(0, Math.round(Number(parent.start || 0)));
            const parentEnd = parentStart + Math.max(1, Math.round(Number(parent.length || 1)));
            const oldStart = Math.max(parentStart, Math.round(Number(target.start || parentStart)));
            const oldEnd = Math.min(parentEnd, oldStart + Math.max(minLength, Math.round(Number(target.length || 1))));
            if (edge === "right") {
                const nextEnd = Math.max(oldStart + minLength, Math.min(parentEnd, oldEnd + dragDelta));
                target.start = oldStart;
                target.length = Math.max(minLength, nextEnd - oldStart);
            } else if (edge === "left") {
                const nextStart = Math.max(parentStart, Math.min(oldStart + dragDelta, oldEnd - minLength));
                target.start = nextStart;
                target.length = Math.max(minLength, oldEnd - nextStart);
            }
            target.type = "text";
            target.textPlaceholder = true;
            target.relay_kind = "slot";
            target.relayKind = "slot";
            target.slotRelay = true;
            target.transitionRelay = false;
            return items;
        }
        if (edge === "right") {
            const oldEnd = Number(target.start || 0) + Number(target.length || 1);
            const next = items[index + 1];
            if (next) {
                const nextEnd = Number(next.start || 0) + Number(next.length || 1);
                const boundary = Math.max(Number(target.start || 0) + minLength, Math.min(oldEnd + dragDelta, nextEnd - minLength));
                target.length = Math.round(boundary - Number(target.start || 0));
                next.start = Math.round(boundary);
                next.length = Math.max(minLength, Math.round(nextEnd - boundary));
            } else {
                const maxLength = durationFrames - Number(target.start || 0);
                target.length = Math.max(minLength, Math.min(Number(target.length || 1) + dragDelta, maxLength));
            }
        } else if (edge === "left") {
            const oldStart = Number(target.start || 0);
            const oldLength = Number(target.length || 1);
            const targetEnd = oldStart + oldLength;
            const prev = items[index - 1];
            if (prev) {
                const prevStart = Number(prev.start || 0);
                const boundary = Math.max(prevStart + minLength, Math.min(oldStart + dragDelta, targetEnd - minLength));
                prev.length = Math.round(boundary - prevStart);
                target.start = Math.round(boundary);
                target.length = Math.max(minLength, Math.round(targetEnd - boundary));
            } else {
                const maxStart = targetEnd - minLength;
                const nextStart = Math.max(0, Math.min(oldStart + dragDelta, maxStart));
                target.start = Math.round(nextStart);
                target.length = Math.max(minLength, oldLength - (nextStart - oldStart));
            }
        }
        return items;
    }

    function slotRelayCenterDragPreview(initItems, targetId, dragDelta) {
        const items = cloneSegments(initItems).sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
        const target = items.find((item) => item.id === targetId);
        if (!target || !isSlotRelaySegment(target)) return items;
        const parent = items.find((item) => String(item.id || "") === String(target.parentSegmentId || "") && isTimelineImageSegment(item));
        if (!parent) return items;
        const parentStart = Math.max(0, Math.round(Number(parent.start || 0)));
        const parentLength = Math.max(1, Math.round(Number(parent.length || 1)));
        const parentEnd = parentStart + parentLength;
        const length = Math.max(1, Math.min(parentLength, Math.round(Number(target.length || 1))));
        const oldStart = Math.max(parentStart, Math.round(Number(target.start || parentStart)));
        target.start = Math.max(parentStart, Math.min(oldStart + dragDelta, parentEnd - length));
        target.length = length;
        target.type = "text";
        target.textPlaceholder = true;
        target.relay_kind = "slot";
        target.relayKind = "slot";
        target.slotRelay = true;
        target.transitionRelay = false;
        return items;
    }

    function audioDragPreview(initItems, targetId, dragDelta, edge, durationFrames) {
        const items = cloneSegments(initItems).sort((a, b) => (Number(a.track || 0) - Number(b.track || 0)) || (Number(a.start || 0) - Number(b.start || 0)));
        const target = items.find((item) => item.id === targetId);
        if (!target) return items;
        const total = Math.max(1, Math.round(Number(durationFrames || getTotalFrames())));
        const oldStart = Math.max(0, Math.round(Number(target.start || 0)));
        const oldLength = Math.max(1, Math.round(Number(target.length || 1)));
        const oldEnd = oldStart + oldLength;
        const audioDuration = Math.max(1, Math.round(Number(target.audioDurationFrames || oldLength)));
        const oldTrim = Math.max(0, Math.round(Number(target.trimStart || 0)));
        if (edge === "center") {
            target.start = Math.max(0, Math.min(Math.round(oldStart + dragDelta), Math.max(0, total - oldLength)));
            return items;
        }
        if (edge === "right") {
            const newEnd = Math.max(oldStart + 1, Math.min(total, oldEnd + dragDelta));
            target.length = Math.max(1, Math.min(newEnd - oldStart, Math.max(1, audioDuration - oldTrim)));
            return items;
        }
        if (edge === "left") {
            const newStart = Math.max(0, Math.min(oldStart + dragDelta, oldEnd - 1));
            const trimDelta = newStart - oldStart;
            const newTrim = Math.max(0, Math.min(audioDuration - 1, oldTrim + trimDelta));
            target.start = newStart;
            target.trimStart = newTrim;
            target.length = Math.max(1, Math.min(oldEnd - newStart, audioDuration - newTrim));
            return items;
        }
        return items;
    }

    function startTimelineDrag(event, seg, isAudio = false, edge = "center") {
        event.preventDefault();
        event.stopPropagation();
        selectedId = seg.id;
        stopPlayback();
        const startX = event.clientX;
        const originalStart = Number(seg.start || 0);
        dragState = {
            kind: edge,
            isAudio,
            targetId: seg.id,
            startX,
            originalStart,
            initial: cloneSegments(isAudio ? timeline.audioSegments : timeline.segments),
        };
        event.currentTarget?.setPointerCapture?.(event.pointerId);
        const captureTarget = event.currentTarget;
        const pointerId = event.pointerId;
        let finished = false;

        const onMove = (move) => {
            move.preventDefault();
            if (!dragState) return;
            if (typeof move.buttons === "number" && move.buttons === 0) {
                finishDrag();
                return;
            }
            const rect = timelineBox.getBoundingClientRect();
            const widthPx = rect.width || 1;
            const deltaFrames = Math.round(((move.clientX - startX) / widthPx) * getTotalFrames());
            let next;
            if (isAudio) {
                next = audioDragPreview(dragState.initial, dragState.targetId, deltaFrames, edge, getTotalFrames());
            } else if (edge === "center" && isSlotRelaySegment((dragState.initial || []).find((item) => item.id === dragState.targetId))) {
                next = slotRelayCenterDragPreview(dragState.initial, dragState.targetId, deltaFrames);
            } else if (edge === "center") {
                const pointerFrame = Math.round(((move.clientX - rect.left) / Math.max(1, widthPx)) * getTotalFrames());
                next = applyCenterDragPhysics(dragState.initial, dragState.targetId, dragState.originalStart + deltaFrames, pointerFrame, getTotalFrames());
            } else {
                next = edgeDragPreview(dragState.initial, dragState.targetId, deltaFrames, edge, getTotalFrames());
            }
            if (isAudio) previewAudioSegments = next;
            else previewSegments = next;
            scheduleDraw();
        };

        const finishDrag = () => {
            if (finished) return;
            finished = true;
            window.removeEventListener("pointermove", onMove, true);
            window.removeEventListener("pointerup", finishDrag, true);
            window.removeEventListener("pointercancel", finishDrag, true);
            window.removeEventListener("mouseup", finishDrag, true);
            window.removeEventListener("blur", finishDrag, true);
            captureTarget?.removeEventListener?.("lostpointercapture", finishDrag, true);
            try {
                if (captureTarget?.hasPointerCapture?.(pointerId)) captureTarget.releasePointerCapture(pointerId);
            } catch (_) {}
            if (dragState) {
                if (isAudio && previewAudioSegments) timeline.audioSegments = previewAudioSegments;
                if (!isAudio && previewSegments) timeline.segments = previewSegments;
            }
            if (!isAudio) {
                const moved = (timeline.segments || []).find((item) => item.id === seg.id);
                if (moved && isActionBridgeRelaySegment(moved)) syncActionBridgeSourceFromRelay(moved);
            }
            previewSegments = null;
            previewAudioSegments = null;
            dragState = null;
            writeTimeline();
            draw();
        };

        window.addEventListener("pointermove", onMove, { passive: false, capture: true });
        window.addEventListener("pointerup", finishDrag, { passive: false, capture: true });
        window.addEventListener("pointercancel", finishDrag, { passive: false, capture: true });
        window.addEventListener("mouseup", finishDrag, { passive: false, capture: true });
        window.addEventListener("blur", finishDrag, { passive: false, capture: true });
        captureTarget?.addEventListener?.("lostpointercapture", finishDrag, { passive: false, capture: true });
        scheduleDraw();
    }

    function duplicateVisualSegment(seg) {
        const sorted = (timeline.segments || []).slice().sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
        const index = sorted.findIndex((item) => item.id === seg.id);
        if (index < 0) return;
        const source = sorted[index];
        const oldLength = Math.max(1, Math.round(Number(source.length || defaultLen())));
        let cloneLength = Math.max(1, Math.floor(oldLength / 2));
        if (oldLength > 1) {
            source.length = Math.max(1, oldLength - cloneLength);
        } else {
            cloneLength = 1;
            for (let i = index + 1; i < sorted.length; i += 1) sorted[i].start = Number(sorted[i].start || 0) + 1;
        }
        const clone = {
            ...source,
            id: newId("seg"),
            start: Number(source.start || 0) + Number(source.length || 1),
            length: cloneLength,
            label: `${String(source.label || "shot").replace(/_dup\d*$/i, "")}_dup`,
        };
        sorted.splice(index + 1, 0, clone);
        timeline.segments = sorted;
        selectedId = clone.id;
        writeTimeline();
        draw();
    }


    function splitImageSlotAfterSegment(seg) {
        if (!seg) return;
        const sorted = (timeline.segments || []).slice().sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
        const index = sorted.findIndex((item) => item.id === seg.id);
        if (index < 0) return;
        const source = sorted[index];
        const next = sorted[index + 1] || null;
        const oldLength = Math.max(1, Math.round(Number(source.length || defaultLen())));
        let slotLength = Math.max(1, Math.round(defaultLen()));
        if (next) {
            slotLength = Math.max(1, Math.floor(oldLength / 2));
            if (oldLength > 1) {
                source.length = Math.max(1, oldLength - slotLength);
            } else {
                slotLength = 1;
                for (let i = index + 1; i < sorted.length; i += 1) sorted[i].start = Math.round(Number(sorted[i].start || 0) + 1);
                ensureDurationForFrames(endOfSegments(sorted) + 1);
            }
        } else {
            ensureDurationForFrames(Math.round(Number(source.start || 0) + Number(source.length || 1)) + slotLength);
        }
        const slot = {
            id: newId("slot"),
            type: "image",
            placeholder: true,
            start: Math.round(Number(source.start || 0) + Number(source.length || 1)),
            length: slotLength,
            ref: 0,
            label: "empty_slot",
            prompt: "",
            note: "",
            camera: "cut to",
            transition: "hard_cut",
            motion: clampGuideStrength(defaultForceWidget?.value || 1),
            guideStrength: clampGuideStrength(defaultForceWidget?.value || 1),
            imageLockStrength: clampGuideStrength(defaultForceWidget?.value || 1),
            defaultForceSource: clampGuideStrength(defaultForceWidget?.value || 1),
            forceCustom: false,
            use_guide: false,
            use_prompt: false,
        };
        sorted.splice(index + 1, 0, slot);
        timeline.segments = sorted;
        selectedId = slot.id;
        writeTimeline({ force: true });
        draw();
    }

    function renderAudioWaveform(block, seg) {
        ensureSegmentWaveform(seg);
        const allPeaks = Array.isArray(seg.waveformPeaks) ? seg.waveformPeaks : [];
        const durationFrames = Math.max(1, Number(seg.audioDurationFrames || seg.length || 1));
        const trimStart = Math.max(0, Number(seg.trimStart || 0));
        const trimEnd = Math.max(trimStart + 1, trimStart + Math.max(1, Number(seg.length || 1)));
        const startIndex = allPeaks.length ? Math.max(0, Math.min(allPeaks.length - 1, Math.floor((trimStart / durationFrames) * allPeaks.length))) : 0;
        const endIndex = allPeaks.length ? Math.max(startIndex + 1, Math.min(allPeaks.length, Math.ceil((trimEnd / durationFrames) * allPeaks.length))) : 0;
        const peaks = allPeaks.slice(startIndex, endIndex);
        const shell = document.createElement("div");
        shell.style.cssText = [
            "position:absolute",
            "left:14px",
            "right:14px",
            "top:6px",
            "bottom:6px",
            "border-radius:7px",
            "overflow:hidden",
            "background:linear-gradient(180deg,rgba(9,19,22,.92),rgba(22,15,10,.88))",
            "border:1px solid rgba(244,213,158,.22)",
            "box-shadow:inset 0 1px 0 rgba(255,255,255,.08), inset 0 -10px 18px rgba(0,0,0,.20)",
            "pointer-events:none",
        ].join(";");
        const name = document.createElement("div");
        name.textContent = String(seg.fileName || seg.audioFile || "Audio").split(/[\\/]/).pop();
        name.style.cssText = "position:absolute;left:8px;top:4px;right:8px;color:#F4E5C4;font:9px/1 monospace;font-weight:900;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-shadow:0 1px 2px rgba(0,0,0,.85);z-index:2;";
        shell.appendChild(name);
        const canvas = document.createElement("canvas");
        canvas.style.cssText = "position:absolute;left:0;right:0;bottom:0;width:100%;height:100%;z-index:1;";
        const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
        const cssW = Math.max(64, Math.round(Number(block.offsetWidth || 120)));
        const cssH = Math.max(46, Math.round(Number(block.offsetHeight || 54)));
        const w = Math.max(96, Math.min(4096, Math.round(cssW * dpr)));
        const h = Math.max(50, Math.min(260, Math.round(cssH * dpr)));
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d");
        if (!ctx) {
            shell.appendChild(canvas);
            block.appendChild(shell);
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
        if (!peaks.length) {
            ctx.fillStyle = "rgba(235,248,255,.76)";
            ctx.font = `900 ${Math.max(10, Math.round(12 * dpr))}px ui-monospace, Consolas, monospace`;
            ctx.textAlign = "center";
            ctx.fillText(waveformLoading.has(seg.id) ? "decoding real waveform..." : "no waveform peaks", w * .5, h * .53);
            shell.appendChild(canvas);
            block.appendChild(shell);
            return;
        }
        const peakValue = (raw) => {
            const p = normPeak(raw);
            return Math.max(Math.abs(p.min), Math.abs(p.max), p.rms);
        };
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
        shell.appendChild(canvas);
        block.appendChild(shell);
    }

    function openAppendImagePicker(targetId = null) {
        pendingImageTargetId = targetId;
        pendingImageInsertFrame = null;
        // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        // Bug #5 fix: reset fileInput before triggering click so re-selecting the same file fires onchange.
        try { fileInput.value = ""; } catch {}
        fileInput.click();
    }

    function textRelaySegment(start, length, label = "transition_relay", options = {}) {
        const relayKind = String(options.relayKind || options.relay_kind || "transition").toLowerCase() === "slot" ? "slot" : "transition";
        return {
            id: newId(relayKind === "slot" ? "slotrelay" : "txt"),
            type: "text",
            textPlaceholder: true,
            relay_kind: relayKind,
            relayKind,
            transitionRelay: relayKind === "transition",
            slotRelay: relayKind === "slot",
            parentSegmentId: String(options.parentSegmentId || ""),
            start: Math.max(0, Math.round(Number(start || 0))),
            length: Math.max(1, Math.round(Number(length || defaultLen()))),
            ref: 0,
            label,
            prompt: "",
            note: "",
            camera: relayKind === "slot" ? "in-slot semantic relay" : "continuous cinematic action",
            transition: relayKind === "slot" ? "prompt_relay_slot" : "prompt_relay_transition",
            guideStrength: 0,
            imageLockStrength: 0,
            defaultForceSource: 0,
            forceCustom: true,
            use_guide: false,
            use_prompt: false,
        };
    }

    function syncActionBridgeSourceFromRelay(relay) {
        if (!isActionBridgeRelaySegment(relay)) return false;
        const source = (timeline.segments || []).find((item) => item.id === relay.actionBridgeSourceId);
        if (!source) return false;
        const prompt = String(relay.prompt || "").trim();
        source.step_transition_prompt = String(relay.prompt || "");
        source.step_transition_duration = Math.max(0, Math.round(Number(relay.length || 1)) / getFps());
        source.step_transition_enabled = Boolean(prompt);
        if (prompt && String(source.step_transition_type || "off") === "off") source.step_transition_type = "action_beat";
        source.use_prompt = Boolean(prompt || source.use_prompt);
        return true;
    }

    function syncActionBridgeRelaySegment(source, options = {}) {
        if (!isTimelineImageSegment(source)) return null;
        const forceTiming = Boolean(options.forceTiming);
        const sourceId = String(source.id || "");
        const type = String(source.step_transition_type || "off");
        const prompt = String(source.step_transition_prompt || "").trim();
        const seconds = Math.max(0, Number(source.step_transition_duration || 0) || 0);
        const existing = (timeline.segments || []).find((item) => isActionBridgeRelaySegment(item) && String(item.actionBridgeSourceId) === sourceId);
        const active = Boolean(source.step_transition_enabled && type !== "off" && prompt && seconds > 0);
        if (!active) {
            if (existing) timeline.segments = (timeline.segments || []).filter((item) => item.id !== existing.id);
            return null;
        }
        const fps = getFps();
        const length = Math.max(1, Math.round(seconds * fps));
        const start = Math.max(0, Math.round(Number(source.start || 0) + Number(source.length || 1)));
        let relay = existing;
        if (!relay) {
            (timeline.segments || []).forEach((item) => {
                if (item.id !== source.id && Number(item.start || 0) >= start) {
                    item.start = Math.max(0, Math.round(Number(item.start || 0) + length));
                }
            });
            relay = textRelaySegment(start, length, "action_bridge_relay");
            relay.actionBridgeSourceId = sourceId;
            relay.actionBridgeManaged = true;
            relay.transition = "prompt_relay_action_bridge";
            relay.camera = "continuous relay bridge";
            relay.use_prompt = true;
            relay.prompt = String(source.step_transition_prompt || "");
            relay.note = String(source.step_transition_prompt || "");
            timeline.segments = (timeline.segments || []).concat(relay);
        } else {
            relay.type = "text";
            relay.textPlaceholder = true;
            relay.actionBridgeManaged = true;
            relay.use_prompt = true;
            relay.transition = relay.transition || "prompt_relay_action_bridge";
            relay.camera = relay.camera || "continuous relay bridge";
            relay.prompt = String(source.step_transition_prompt || "");
            relay.note = String(source.step_transition_prompt || "");
            relay.label = relay.label || "action_bridge_relay";
            if (forceTiming) {
                relay.start = start;
                relay.length = length;
            }
        }
        relay.length = forceTiming || !Number(relay.length || 0) ? length : Math.max(1, Math.round(Number(relay.length || length)));
        relay.ref = 0;
        relay.guideStrength = 0;
        relay.imageLockStrength = 0;
        relay.defaultForceSource = 0;
        relay.forceCustom = true;
        relay.use_guide = false;
        ensureDurationForFrames(endOfSegments(timeline.segments));
        return relay;
    }

    function syncAllActionBridgeRelaySegments(options = {}) {
        const sources = sortedActionBridgeSources(timeline.segments || []);
        const sourceIds = new Set(sources.map((item) => String(item.id || "")));
        timeline.segments = (timeline.segments || []).filter((item) => !isActionBridgeRelaySegment(item) || sourceIds.has(String(item.actionBridgeSourceId || "")));
        sources.forEach((source) => syncActionBridgeRelaySegment(source, options));
    }

    function createTailTextPlaceholder(options = {}) {
        const total = getTotalFrames();
        const cursor = Math.max(0, Math.min(endOfSegments(activeVisualSegments()), Math.max(0, total - 1)));
        const length = Math.min(defaultLen(), Math.max(1, total - cursor));
        const seg = textRelaySegment(cursor, length, "transition_relay", { relayKind: "transition", ...options });
        timeline.segments = (timeline.segments || []).concat(seg).sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
        selectedId = seg.id;
        writeTimeline();
        draw();
    }

    function relayoutSlotRelaysForSource(sourceId) {
        const source = (timeline.segments || []).find((item) => item.id === sourceId);
        if (!isTimelineImageSegment(source)) return;
        const sourceStart = Math.max(0, Math.round(Number(source.start || 0)));
        const sourceLength = Math.max(1, Math.round(Number(source.length || 1)));
        const relays = (timeline.segments || [])
            .filter((item) => isSlotRelaySegment(item) && String(item.parentSegmentId || "") === String(sourceId || ""))
            .sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
        if (!relays.length) return;
        const phases = relays.length + 1;
        relays.forEach((relay, index) => {
            const start = sourceStart + Math.round((sourceLength * (index + 1)) / phases);
            const next = index + 1 < relays.length
                ? sourceStart + Math.round((sourceLength * (index + 2)) / phases)
                : sourceStart + sourceLength;
            relay.start = Math.max(sourceStart, Math.min(sourceStart + sourceLength - 1, start));
            relay.length = Math.max(1, next - relay.start);
            relay.type = "text";
            relay.textPlaceholder = true;
            relay.relay_kind = "slot";
            relay.relayKind = "slot";
            relay.slotRelay = true;
            relay.transitionRelay = false;
            relay.parentSegmentId = String(sourceId || "");
            relay.transition = "prompt_relay_slot";
        });
    }

    function migrateSlotRelaySegmentsToNested() {
        const items = timeline.segments || [];
        const slotRelaySegments = items.filter((item) => isSlotRelaySegment(item));
        if (!slotRelaySegments.length) return false;
        const images = items.filter(isTimelineImageSegment);
        let changed = false;
        slotRelaySegments.forEach((relay) => {
            const relayStart = Math.max(0, Math.round(Number(relay.start || 0)));
            const parent = images.find((image) => String(image.id || "") === String(relay.parentSegmentId || ""))
                || images.find((image) => {
                    const start = Math.max(0, Math.round(Number(image.start || 0)));
                    const end = start + Math.max(1, Math.round(Number(image.length || 1)));
                    return relayStart >= start && relayStart < end;
                });
            if (!parent) return;
            if (!Array.isArray(parent.slot_relays)) parent.slot_relays = [];
            const exists = parent.slot_relays.some((item) => String(item.id || "") === String(relay.id || ""));
            if (!exists) {
                parent.slot_relays.push({
                    id: String(relay.id || newId("slotrelay")),
                    start: relayStart,
                    length: Math.max(1, Math.round(Number(relay.length || 1))),
                    prompt: String(relay.prompt || ""),
                    note: String(relay.note || relay.prompt || ""),
                    use_prompt: Boolean(relay.use_prompt !== false && String(relay.prompt || "").trim()),
                    relay_kind: "slot",
                    relayKind: "slot",
                    slotRelay: true,
                    parentSegmentId: String(parent.id || ""),
                });
            }
            changed = true;
        });
        if (changed) {
            timeline.segments = items.filter((item) => !isSlotRelaySegment(item));
            timeline.segments.forEach((image) => {
                if (!Array.isArray(image.slot_relays)) return;
                image.slot_relays.sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
            });
        }
        return changed;
    }

    function createSlotRelayInSegment(source) {
        const sourceId = String(source?.id || selectedId || "");
        const target = (timeline.segments || []).find((item) => String(item.id || "") === sourceId && isTimelineImageSegment(item))
            || (timeline.segments || []).find((item) => String(item.id || "") === String(selectedId || "") && isTimelineImageSegment(item));
        if (!isTimelineImageSegment(target)) {
            showTimelineNotice("Select an image slot first. Slot Relay belongs inside one image slot.");
            return;
        }
        const win = generationWindowForImage(target);
        const relay = {
            id: newId("slotrelay"),
            start: Math.max(win.start, win.end - 1),
            length: 1,
            prompt: "",
            note: "",
            use_prompt: false,
            relay_kind: "slot",
            relayKind: "slot",
            slotRelay: true,
            parentSegmentId: String(target.id || ""),
        };
        if (!Array.isArray(target.slot_relays)) target.slot_relays = [];
        target.slot_relays.push(relay);
        const relays = relayoutNestedSlotRelaysForImage(target);
        selectedId = target.id;
        previewSegments = null;
        console.info("[IAMCCS WAN PURE][SlotRelay] added", {
            segment_id: target.id,
            relay_count: relays.length,
            generation_window: win,
            relays: relays.map((item, index) => ({
                index,
                start: Math.round(Number(item.start || 0)),
                length: Math.max(1, Math.round(Number(item.length || 1))),
                prompt: String(item.prompt || ""),
            })),
        });
        showTimelineNotice("Slot Relay added as a prompt division inside the selected image slot.");
        writeTimeline({ force: true });
        draw();
    }

    function relayoutNestedSlotRelaysForImage(target) {
        if (!isTimelineImageSegment(target)) return [];
        const win = generationWindowForImage(target);
        const relays = Array.isArray(target.slot_relays)
            ? target.slot_relays.filter((relay) => relay && relay.slotRelay !== false)
            : [];
        target.slot_relays = relays;
        if (!relays.length) return relays;
        relays.sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
        const phaseCount = relays.length + 1;
        relays.forEach((relay, index) => {
            const start = win.start + Math.round((win.length * (index + 1)) / phaseCount);
            const next = index + 1 < relays.length
                ? win.start + Math.round((win.length * (index + 2)) / phaseCount)
                : win.end;
            relay.start = Math.max(win.start, Math.min(win.end - 1, start));
            relay.length = Math.max(1, Math.min(win.end, next) - relay.start);
            relay.parentSegmentId = String(target.id || "");
            relay.relay_kind = "slot";
            relay.relayKind = "slot";
            relay.slotRelay = true;
            relay.transitionRelay = false;
            relay.label = relay.label || `slot_relay_${index + 1}`;
            relay.note = String(relay.note ?? relay.prompt ?? "");
            relay.prompt = String(relay.prompt ?? "");
            relay.use_prompt = Boolean(relay.use_prompt !== false && String(relay.prompt || "").trim());
        });
        return relays;
    }

    function deleteNestedSlotRelay(parentSegmentId, relayId) {
        const parent = (timeline.segments || []).find((item) => String(item.id || "") === String(parentSegmentId || "") && isTimelineImageSegment(item));
        if (!parent || !Array.isArray(parent.slot_relays)) return false;
        const before = parent.slot_relays.length;
        parent.slot_relays = parent.slot_relays.filter((item) => String(item?.id || "") !== String(relayId || ""));
        const changed = parent.slot_relays.length !== before;
        if (changed) {
            relayoutNestedSlotRelaysForImage(parent);
            selectedId = parent.id;
            previewSegments = null;
            console.info("[IAMCCS WAN PURE][SlotRelay] deleted", {
                segment_id: parent.id,
                relay_id: relayId,
                relay_count: parent.slot_relays.length,
            });
        }
        return changed;
    }

    function createPlaceholderAfterSegment(seg, kind = "image") {
        if (!seg) return;
        const isTextKind = kind === "text" || kind === "transition_relay";
        const sorted = (timeline.segments || []).slice().sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
        const index = sorted.findIndex((item) => item.id === seg.id);
        if (index < 0) return;
        const source = sorted[index];
        const start = Number(source.start || 0) + Number(source.length || 1);
        const next = sorted[index + 1];
        if (kind === "image" && next?.placeholder && Math.abs(Number(next.start || 0) - start) <= 1) {
            selectedId = next.id;
            draw();
            return;
        }
        if (isTextKind && String(next?.type || "") === "text" && next?.textPlaceholder && isTransitionRelaySegment(next) && Math.abs(Number(next.start || 0) - start) <= 1) {
            selectedId = next.id;
            draw();
            return;
        }
        if (next) {
            const oldLength = Math.max(1, Math.round(Number(source.length || defaultLen())));
            let splitLen = Math.max(1, Math.floor(oldLength / 2));
            if (oldLength > 1) {
                source.length = Math.max(1, oldLength - splitLen);
            } else {
                splitLen = 1;
                for (let i = index + 1; i < sorted.length; i += 1) {
                    sorted[i].start = Math.round(Number(sorted[i].start || 0) + 1);
                }
                ensureDurationForFrames(endOfSegments(sorted) + 1);
            }
            const splitStart = Math.round(Number(source.start || 0) + Number(source.length || 1));
            const placeholder = isTextKind ? textRelaySegment(splitStart, splitLen, "transition_relay", { relayKind: "transition" }) : {
                id: newId("slot"),
                type: "image",
                placeholder: true,
                start: splitStart,
                length: splitLen,
                ref: 0,
                label: "empty_slot",
                prompt: "",
                note: "",
                camera: "continuous dolly-in",
                transition: "continuous_motion",
                motion: clampGuideStrength(defaultForceWidget?.value || 1),
                guideStrength: clampGuideStrength(defaultForceWidget?.value || 1),
                imageLockStrength: clampGuideStrength(defaultForceWidget?.value || 1),
                defaultForceSource: clampGuideStrength(defaultForceWidget?.value || 1),
                forceCustom: false,
                use_guide: false,
                use_prompt: false,
            };
            sorted.splice(index + 1, 0, placeholder);
            timeline.segments = sorted;
            selectedId = placeholder.id;
            writeTimeline({ force: true });
            draw();
            return;
        }
        const length = defaultLen();
        ensureDurationForFrames(Math.round(start) + length);
        const placeholder = isTextKind ? textRelaySegment(start, length, "transition_relay", { relayKind: "transition" }) : {
            id: newId("slot"),
            type: "image",
            placeholder: true,
            start,
            length,
            ref: 0,
            label: "empty_slot",
            prompt: "",
            note: "",
            camera: "continuous dolly-in",
            transition: "continuous_motion",
            motion: clampGuideStrength(defaultForceWidget?.value || 1),
            guideStrength: clampGuideStrength(defaultForceWidget?.value || 1),
            imageLockStrength: clampGuideStrength(defaultForceWidget?.value || 1),
            defaultForceSource: clampGuideStrength(defaultForceWidget?.value || 1),
            forceCustom: false,
            use_guide: false,
            use_prompt: false,
        };
        sorted.splice(index + 1, 0, placeholder);
        timeline.segments = sorted;
        selectedId = placeholder.id;
        writeTimeline();
        draw();
    }

    function openTimelineAddMenu(event, seg = null, targetId = null) {
        event?.preventDefault?.();
        event?.stopPropagation?.();
        root.querySelectorAll(".iamccs-v3-add-menu").forEach((item) => item.remove());
        document.querySelectorAll(".iamccs-v3-add-menu").forEach((item) => item.remove());
        const menu = document.createElement("div");
        menu.className = "iamccs-v3-add-menu";
        const fallbackRect = root.getBoundingClientRect();
        const menuW = 180;
        const menuH = 156;
        const viewportW = Math.max(1, Number(window.innerWidth || document.documentElement?.clientWidth || fallbackRect.right || 1));
        const viewportH = Math.max(1, Number(window.innerHeight || document.documentElement?.clientHeight || fallbackRect.bottom || 1));
        const pointerX = Number.isFinite(Number(event?.clientX)) ? Number(event.clientX) : fallbackRect.left + 24;
        const pointerY = Number.isFinite(Number(event?.clientY)) ? Number(event.clientY) : fallbackRect.top + 24;
        const x = Math.max(8, Math.min(viewportW - menuW - 8, pointerX));
        const y = Math.max(8, Math.min(viewportH - menuH - 8, pointerY));
        menu.style.cssText = [
            "position:fixed",
            `left:${x}px`,
            `top:${y}px`,
            `width:${menuW}px`,
            "box-sizing:border-box",
            `border:1px solid ${purple.border}`,
            "border-radius:6px",
            `background:${purple.valueBg}`,
            `color:${purple.valueText}`,
            "box-shadow:0 12px 28px rgba(0,0,0,.42)",
            "padding:6px",
            "display:grid",
            "gap:5px",
            `z-index:${CINE_FULLSCREEN_Z_INDEX + 20}`,
            "font:10px/1.15 monospace",
            "font-weight:900",
        ].join(";");
        const addChoice = (label, title, action) => {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.textContent = label;
            btn.title = title;
            btn.style.cssText = `height:28px;border:1px solid ${purple.borderSoft};border-radius:4px;background:#F4EFE6;color:#14110E;cursor:pointer;font:10px/1 monospace;font-weight:900;text-align:left;padding:0 9px;`;
            btn.onpointerdown = (pointerEvent) => { pointerEvent.preventDefault(); pointerEvent.stopPropagation(); };
            btn.onclick = (clickEvent) => {
                clickEvent.preventDefault();
                clickEvent.stopPropagation();
                menu.remove();
                action();
            };
            menu.appendChild(btn);
        };
        addChoice("Transition Relay", "Creates a transition relay owned by the previous WAN generation chunk. It shares that chunk's frame budget and semantically guides motion toward the next image.", () => {
            if (seg) createPlaceholderAfterSegment(seg, "transition_relay");
            else createTailTextPlaceholder({ relayKind: "transition" });
        });
        addChoice("Slot Relay", "Creates an in-slot relay inside the selected image slot. It divides that image slot's frame budget into PromptRelay sub-segments.", () => {
            createSlotRelayInSegment(seg);
        });
        addChoice("Image Slot", "Create or fill an image guide slot", () => {
            if (seg) splitImageSlotAfterSegment(seg);
            else openAppendImagePicker(targetId);
        });
        addChoice("Audio", "Import audio at this timeline position", () => {
            const frame = seg ? Number(seg.start || 0) + Number(seg.length || 0) : endOfSegments(activeVisualSegments());
            pendingAudioInsertFrame = Math.max(0, Math.round(frame));
            pendingAudioTrack = 0;
            audioInput.click();
        });
        document.body.appendChild(menu);
        const close = (closeEvent) => {
            if (!menu.contains(closeEvent.target)) {
                menu.remove();
                document.removeEventListener("pointerdown", close, true);
            }
        };
        setTimeout(() => document.addEventListener("pointerdown", close, true), 0);
        return menu;
    }

    function makeImagePlaceholderBlock() {
        const total = getTotalFrames();
        const cursor = endOfSegments(activeVisualSegments());
        const start = Math.max(0, Math.min(cursor, Math.max(0, total - 1)));
        const remaining = Math.max(1, total - start);
        const length = Math.min(defaultLen(), remaining);
        const normalWidth = Math.max(5, (length / Math.max(1, total)) * 100);
        const left = cursor >= total ? Math.max(0, 100 - normalWidth) : (start / Math.max(1, total)) * 100;
        const block = document.createElement("button");
        block.type = "button";
        block.title = "Add text, image or audio to the next empty slot";
        block.style.cssText = [
            "position:absolute",
            `left:${left}%`,
            `width:${normalWidth}%`,
            "top:8px",
            "height:238px",
            "box-sizing:border-box",
            `border:1px dashed ${purple.border}`,
            "border-radius:4px",
            "background:linear-gradient(180deg,rgba(244,239,231,.08),rgba(0,0,0,.12))",
            `color:${purple.valueBg}`,
            "cursor:pointer",
            "display:flex",
            "align-items:center",
            "justify-content:center",
            "font-size:30px",
            "font-weight:900",
            "box-shadow:inset 0 0 0 1px rgba(255,255,255,.05)",
            "z-index:2",
        ].join(";");
        block.textContent = "+";
        bindReliableTimelineButton(block, (event) => {
            openTimelineAddMenu(event, null, null);
        });
        return protectControlDrag(block);
    }

    function computeTimelineCanvasWidth(segments = activeVisualSegments()) {
        const viewportWidth = Math.max(1, Number(timelineViewport?.clientWidth || 0) || (SHOTBOARD_V3_RIGID_WIDTH - 32));
        timelineMeterSeconds = clampTimelineMeterSeconds(timelineMeterSeconds);
        const duration = Math.max(0.5, getDuration());
        const halfSecondSteps = Math.round((timelineMeterSeconds - duration) * 2);
        const visualScale = Math.max(0.2, Math.min(8, Math.pow(1.18, halfSecondSteps)));
        const meterWidth = viewportWidth * visualScale;
        const minUsableWidth = Math.min(viewportWidth, 360);
        return Math.max(minUsableWidth, Math.min(12000, Math.ceil(meterWidth)));
    }

    function makeBlock(seg, isAudio = false) {
        const total = getTotalFrames();
        const block = document.createElement("div");
        const left = (Number(seg.start || 0) / total) * 100;
        const width = Math.max(1, (Number(seg.length || 1) / total) * 100);
        const top = isAudio ? (Number(seg.track || 0) * 90 + 4) : 8;
        const truthRailHeight = isAudio ? 0 : 16;
        const frameShellHeight = isAudio ? 128 : 144;
        const height = isAudio ? 82 : 238 + timelineExtraH; // grows with user timeline resize
        const imageHeight = 108;
        const transitionLaneTop = truthRailHeight + imageHeight;
        const transitionLaneHeight = 20;
        const promptTop = frameShellHeight + 10;
        const promptHeight = isAudio ? 0 : 74 + timelineExtraH; // local prompt textarea grows with timeline height
        const leftRailWidth = isAudio ? 0 : 24;
        const innerLeft = isAudio ? 8 : (leftRailWidth + 6);
        const innerRight = 8;
        const topRightSafe = isAudio ? innerRight : 42;
        const selected = selectedId === seg.id;
        const isSlotRelayBlock = !isAudio && isSlotRelaySegment(seg);
        const isTransitionRelayBlock = !isAudio && isTransitionRelaySegment(seg);
        const showDragStripes = Boolean(dragState && !isAudio && dragState.targetId === seg.id && dragState.kind !== "center");
        const color = isAudio
            ? purple.audio
            : (isSlotRelayBlock
                ? "linear-gradient(180deg,rgba(22,45,38,.86),rgba(8,18,16,.88))"
                : (isTransitionRelayBlock ? "linear-gradient(180deg,rgba(67,53,27,.90),rgba(18,14,8,.90))" : (String(seg.type) === "text" ? purple.textBlock : purple.image2)));
        block.style.cssText = [
            "position:absolute",
            `left:${left}%`,
            `width:${width}%`,
            `min-width:${isAudio ? 20 : 30}px`,
            `top:${top}px`,
            `height:${height}px`,
            `background:${color}`,
            (selected ? "border:2px solid #F9C859" : `border:1px solid ${isSlotRelayBlock ? "rgba(137,238,164,.42)" : (isTransitionRelayBlock ? "rgba(255,207,111,.72)" : purple.borderSoft)}`),
            "border-radius:4px",
            "box-sizing:border-box",
            "overflow:visible",
            "cursor:grab",
            (selected ? "box-shadow:0 0 0 2px rgba(249,200,89,.35),0 6px 16px rgba(0,0,0,.38),inset 0 1px 0 rgba(255,190,120,.08)" : (isSlotRelayBlock ? "box-shadow:0 4px 10px rgba(0,0,0,.30),inset 0 1px 0 rgba(195,255,205,.10)" : "box-shadow:0 6px 16px rgba(0,0,0,.38),inset 0 1px 0 rgba(255,190,120,.08)")),
            "user-select:none",
            `z-index:${isAudio ? 14 : 4}`,
        ].join(";");
        const content = document.createElement("div");
        content.style.cssText = isAudio
            ? "position:absolute;left:8px;right:8px;bottom:2px;height:14px;display:flex;align-items:center;justify-content:center;text-align:center;color:#FFF2E4;font-size:10px;font-weight:800;text-shadow:0 1px 2px rgba(0,0,0,.75);padding:0;box-sizing:border-box;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
            : `position:absolute;left:${innerLeft}px;right:${innerRight}px;top:${truthRailHeight}px;height:${imageHeight}px;display:flex;align-items:center;justify-content:center;text-align:center;color:#fff;font-size:11px;font-weight:800;text-shadow:0 1px 2px rgba(0,0,0,.75);padding:8px;box-sizing:border-box;overflow:hidden;`;
        const appendAddAfterButton = () => {
            const addAfter = document.createElement("button");
            addAfter.type = "button";
            addAfter.textContent = "+";
            addAfter.title = "Add text, image or audio after this slot";
            addAfter.style.cssText = `position:absolute;right:14px;top:${truthRailHeight + 8}px;width:24px;height:24px;border:1px solid ${purple.border};border-radius:999px;background:${purple.valueBg};color:${purple.valueText};font-size:17px;font-weight:900;line-height:1;cursor:pointer;box-shadow:0 3px 10px rgba(0,0,0,.45);z-index:9;`;
            bindReliableTimelineButton(addAfter, (event) => {
                openTimelineAddMenu(event, seg, null);
            });
            block.appendChild(addAfter);
        };
        const appendNestedSlotRelayLines = () => {
            if (isAudio || !isTimelineImageSegment(seg) || !Array.isArray(seg.slot_relays) || !seg.slot_relays.length) return;
            const win = generationWindowForImage(seg);
            const segStart = win.start;
            const segLength = Math.max(1, Math.round(Number(win.length || 1)));
            const segEnd = win.end;
            seg.slot_relays.forEach((relay) => {
                const relayStart = Math.max(segStart, Math.min(segEnd - 1, Math.round(Number(relay.start ?? (segStart + Math.floor(segLength / 2))))));
                const relayLength = Math.max(1, Math.round(Number(relay.length || Math.max(1, segEnd - relayStart))));
                const relayEnd = Math.max(relayStart + 1, Math.min(segEnd, relayStart + relayLength));
                const localRelayStart = generationFrameFor(seg, relayStart);
                relay.start = relayStart;
                relay.length = relayEnd - relayStart;
                relay.parentSegmentId = String(seg.id || "");
                relay.relay_kind = "slot";
                relay.relayKind = "slot";
                relay.slotRelay = true;

                const leftPct = ((relayStart - segStart) / segLength) * 100;
                const widthPct = Math.max(.8, ((relayEnd - relayStart) / segLength) * 100);
                const contentFixed = innerLeft + innerRight;
                const contentLeftOffsetPx = innerLeft - (contentFixed * (leftPct / 100));
                const contentWidthOffsetPx = contentFixed * (widthPct / 100);
                const contentLeftCss = `calc(${leftPct.toFixed(4)}% + ${contentLeftOffsetPx.toFixed(2)}px)`;
                const contentWidthCss = `calc(${widthPct.toFixed(4)}% - ${contentWidthOffsetPx.toFixed(2)}px)`;
                const lineLeftCss = `calc(${leftPct.toFixed(4)}% + ${(contentLeftOffsetPx - 3).toFixed(2)}px)`;
                const badgeLeftCss = `calc(${leftPct.toFixed(4)}% + ${(contentLeftOffsetPx + 5).toFixed(2)}px)`;
                const line = document.createElement("div");
                line.title = `Slot Relay starts at generation frame ${localRelayStart}. Drag to move this prompt division.`;
                line.style.cssText = [
                    "position:absolute",
                    `left:${lineLeftCss}`,
                    `top:${truthRailHeight}px`,
                    `height:${promptTop + promptHeight - truthRailHeight}px`,
                    "width:6px",
                    "border-radius:999px",
                    "background:linear-gradient(180deg,#DDF8D8,#6FD57B,#DDF8D8)",
                    "box-shadow:0 0 0 1px rgba(0,0,0,.62),0 0 8px rgba(137,238,164,.38)",
                    "cursor:ew-resize",
                    "z-index:66",
                ].join(";");
                line.onpointerdown = (event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    selectedId = seg.id;
                    const rect = timelineBox.getBoundingClientRect();
                    const widthPx = rect.width || 1;
                    const startX = event.clientX;
                    const originalStart = relayStart;
                    const originalEnd = relayEnd;
                    const pointerId = event.pointerId;
                    try { line.setPointerCapture?.(pointerId); } catch (_) {}
                    const onMove = (move) => {
                        move.preventDefault();
                        const deltaFrames = Math.round(((move.clientX - startX) / widthPx) * getTotalFrames());
                        const nextStart = Math.max(segStart, Math.min(originalStart + deltaFrames, originalEnd - 1));
                        relay.start = nextStart;
                        relay.length = Math.max(1, originalEnd - nextStart);
                        writeTimeline({ force: true });
                        scheduleDraw();
                    };
                    const finish = () => {
                        window.removeEventListener("pointermove", onMove, true);
                        window.removeEventListener("pointerup", finish, true);
                        window.removeEventListener("pointercancel", finish, true);
                        try { if (line.hasPointerCapture?.(pointerId)) line.releasePointerCapture(pointerId); } catch (_) {}
                        writeTimeline({ force: true });
                        draw();
                    };
                    window.addEventListener("pointermove", onMove, { passive: false, capture: true });
                    window.addEventListener("pointerup", finish, { passive: false, capture: true });
                    window.addEventListener("pointercancel", finish, { passive: false, capture: true });
                };
                const frameBadge = document.createElement("div");
                frameBadge.textContent = `F${localRelayStart}`;
                frameBadge.title = line.title;
                frameBadge.style.cssText = [
                    "position:absolute",
                    `left:${badgeLeftCss}`,
                    `top:${truthRailHeight + 4}px`,
                    "height:15px",
                    "box-sizing:border-box",
                    "padding:1px 4px",
                    "border-radius:4px",
                    "background:rgba(4,12,10,.82)",
                    "border:1px solid rgba(137,238,164,.58)",
                    "color:#DDF8D8",
                    "font:8px/11px monospace",
                    "font-weight:900",
                    "text-shadow:0 1px 2px rgba(0,0,0,.75)",
                    "z-index:67",
                    "pointer-events:none",
                ].join(";");
                const duration = document.createElement("div");
                duration.title = "Slot Relay duration inside this image slot";
                duration.style.cssText = [
                    "position:absolute",
                    `left:${contentLeftCss}`,
                    `width:${contentWidthCss}`,
                    `top:${transitionLaneTop + 7}px`,
                    "height:3px",
                    "border-radius:999px",
                    "background:rgba(137,238,164,.82)",
                    "box-shadow:0 1px 3px rgba(0,0,0,.40)",
                    "z-index:54",
                    "pointer-events:none",
                ].join(";");
                block.append(duration, line, frameBadge);
            });
        };
        if (!isAudio && String(seg.type || "image") !== "text") {
            const path = segmentReferencePath(seg);
            if (path) {
                let previewUrl = previewUrlForPath(path);
                const previewBust = refPreviewBusters.get(String(path)) || "";
                if (previewBust) previewUrl += `${previewUrl.includes("?") ? "&" : "?"}v=${encodeURIComponent(previewBust)}`;
                const stripe = document.createElement("div");
                stripe.style.cssText = [
                    "position:absolute",
                    `left:${innerLeft}px`,
                    `right:${innerRight}px`,
                    `top:${truthRailHeight}px`,
                    `height:${imageHeight}px`,
                    `background-image:url("${previewUrl}")`,
                    "background-size:auto 100%",
                    "background-repeat:repeat-x",
                    "background-position:left center",
                    `opacity:${showDragStripes ? ".96" : ".9"}`,
                    "box-shadow:inset 0 0 0 999px rgba(0,0,0,.08)",
                ].join(";");
                block.appendChild(stripe);
                const replaceImage = document.createElement("button");
                replaceImage.type = "button";
                replaceImage.textContent = "+";
                replaceImage.title = "Replace this image, keeping the same duration, prompt and timing";
                replaceImage.style.cssText = [
                    "position:absolute",
                    "left:calc(50% + 12px)",
                    `top:${truthRailHeight + 54}px`,
                    "transform:translate(-50%,-50%)",
                    "width:28px",
                    "height:28px",
                    `border:1px solid ${purple.border}`,
                    "border-radius:999px",
                    `background:${purple.valueBg}`,
                    `color:${purple.valueText}`,
                    "font-size:18px",
                    "font-weight:900",
                    "line-height:1",
                    "cursor:pointer",
                    "box-shadow:0 3px 10px rgba(0,0,0,.38)",
                    "opacity:.78",
                    "z-index:8",
                ].join(";");
                bindReliableTimelineButton(replaceImage, () => {
                    openAppendImagePicker(seg.id);
                });
                block.appendChild(replaceImage);
            } else {
                const plus = document.createElement("button");
                plus.type = "button";
                plus.textContent = "+";
                plus.title = "Import image into this empty image slot";
                plus.style.cssText = `position:absolute;left:calc(50% + 12px);top:${truthRailHeight + 44}px;transform:translate(-50%,-50%);width:46px;height:46px;border:2px solid ${purple.border};border-radius:999px;background:${purple.valueBg};color:${purple.valueText};font-size:28px;font-weight:900;line-height:1;cursor:pointer;box-shadow:0 5px 18px rgba(0,0,0,.42),0 0 0 3px rgba(244,239,230,.08);z-index:12;touch-action:manipulation;`;
                bindReliableTimelineButton(plus, () => {
                    openAppendImagePicker(seg.id);
                });
                block.appendChild(plus);
            }
        }
        appendNestedSlotRelayLines();
        if (!isAudio) appendAddAfterButton();
        if (isAudio) {
            const removeAudio = document.createElement("button");
            removeAudio.type = "button";
            removeAudio.textContent = "X";
            removeAudio.title = "Remove this audio clip";
            removeAudio.style.cssText = `position:absolute;right:5px;top:5px;width:22px;height:22px;border:1px solid ${purple.danger};border-radius:999px;background:#6B302A;color:#FFF2E4;font-size:10px;font-weight:900;line-height:1;cursor:pointer;box-shadow:0 2px 7px rgba(0,0,0,.40);z-index:20;`;
            removeAudio.onpointerdown = (event) => { event.preventDefault(); event.stopPropagation(); };
            removeAudio.onclick = (event) => {
                event.preventDefault();
                event.stopPropagation();
                timeline.audioSegments = (timeline.audioSegments || []).filter((item) => item.id !== seg.id);
                selectedId = timeline.segments[0]?.id || null;
                writeTimeline({ force: true });
                draw();
            };
            block.appendChild(removeAudio);
            renderAudioWaveform(block, seg);
        }
        content.textContent = isAudio ? String(seg.name || "audio") : (String(seg.type || "image") === "text" ? String(seg.label || (isSlotRelayBlock ? "slot_relay" : "text")) : "");
        if (isSlotRelayBlock) {
            content.style.left = "14px";
            content.style.right = "8px";
            content.style.top = `${truthRailHeight + 8}px`;
            content.style.height = "38px";
            content.style.alignItems = "flex-start";
            content.style.justifyContent = "flex-start";
            content.style.textAlign = "left";
            content.style.color = "#DDF8D8";
            content.style.fontSize = "9px";
            content.style.textShadow = "0 1px 2px rgba(0,0,0,.70)";
            content.style.opacity = ".82";
        }
        if (isAudio || String(seg.type || "image") === "text") block.appendChild(content);
        if (isSlotRelayBlock || isTransitionRelayBlock) {
            const durationLine = document.createElement("div");
            durationLine.title = isSlotRelayBlock
                ? "Slot Relay duration: drag the start line or the edge handle to adjust this prompt segment"
                : "Transition Relay duration";
            durationLine.style.cssText = [
                "position:absolute",
                "left:0",
                "right:0",
                `top:${transitionLaneTop + 6}px`,
                "height:3px",
                "border-radius:999px",
                `background:${isSlotRelayBlock ? "rgba(137,238,164,.82)" : "rgba(255,207,111,.78)"}`,
                "box-shadow:0 1px 3px rgba(0,0,0,.38)",
                "z-index:24",
                "pointer-events:none",
            ].join(";");
            block.appendChild(durationLine);
            if (isSlotRelayBlock) {
                const startLine = document.createElement("div");
                startLine.title = "Drag Slot Relay start";
                startLine.style.cssText = [
                    "position:absolute",
                    "left:0",
                    "top:0",
                    "bottom:0",
                    "width:7px",
                    "cursor:ew-resize",
                    "background:linear-gradient(180deg,#DDF8D8,#6DD47A,#DDF8D8)",
                    "border-right:1px solid rgba(8,18,16,.72)",
                    "box-shadow:0 0 0 1px rgba(0,0,0,.35),0 0 8px rgba(137,238,164,.30)",
                    "z-index:64",
                    "border-radius:4px 0 0 4px",
                ].join(";");
                startLine.onpointerdown = (event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    startTimelineDrag(event, seg, false, "left");
                };
                block.appendChild(startLine);
            }
        }
        if (!isAudio) {
            const transitionLane = document.createElement("div");
            transitionLane.style.cssText = [
                "position:absolute",
                `left:${innerLeft}px`,
                `right:${innerRight}px`,
                `top:${transitionLaneTop}px`,
                `height:${transitionLaneHeight}px`,
                "background:linear-gradient(180deg,rgba(0,0,0,.18),rgba(255,238,205,.12) 48%,rgba(0,0,0,.20))",
                `border-top:1px solid ${purple.borderSoft}`,
                `border-bottom:1px solid ${purple.borderSoft}`,
                "box-sizing:border-box",
                "pointer-events:none",
                "z-index:4",
            ].join(";");
            block.appendChild(transitionLane);
        }
        if (!isAudio && !isSlotRelayBlock) {
            const rail = document.createElement("div");
            rail.style.cssText = `position:absolute;left:0;top:0;width:24px;height:${frameShellHeight}px;display:grid;grid-template-rows:repeat(3,1fr);background:rgba(0,0,0,.45);z-index:15;`;
            const railBtn = (label, titleText, action) => {
                const b = document.createElement("button");
                b.type = "button";
                b.textContent = label;
                b.title = titleText;
                b.style.cssText = "border:0;border-bottom:1px solid rgba(255,255,255,.22);background:transparent;color:#fff;font-size:10px;font-weight:900;cursor:pointer;padding:0;";
                b.onpointerdown = (event) => { event.preventDefault(); event.stopPropagation(); };
                b.onmousedown = (event) => { event.preventDefault(); event.stopPropagation(); };
                b.onclick = (event) => { event.stopPropagation(); action(); };
                return protectControlDrag(b);
            };
            rail.append(
                railBtn("E", "Open frame editor for this reference", () => {
                    selectedId = seg.id;
                    const currentPath = segmentReferencePath(seg);
                    const currentRef = referenceIndexForPath(currentPath) || Math.max(1, Number(seg.ref || 1));
                    const refIndex = Math.max(0, currentRef - 1);
                    const path = currentPath || refPaths()[refIndex];
                    if (path) {
                        openReferenceFrameEditor(node, refIndex, path, (newPath, data) => {
                            const appended = appendReferencePath(node, newPath);
                            const nextRef = Math.max(1, Number(appended?.refNumber || refIndex + 1));
                            const truthSeg = (timeline.segments || []).find((item) => String(item?.id || "") === String(seg.id || "")) || seg;
                            applyEditedReferenceTruth(truthSeg, nextRef, newPath, {
                                updateAutoLabel: false,
                                truthSource: "timeline_frame_editor_slot",
                            });
                            if (truthSeg !== seg) {
                                applyEditedReferenceTruth(seg, nextRef, newPath, {
                                    updateAutoLabel: false,
                                    truthSource: "timeline_frame_editor_slot",
                                });
                            }
                            if (newPath) refPreviewBusters.set(String(newPath), String(data?.cache_bust || Date.now()));
                            console.info("[IAMCCS V3 REF EDIT] applied edited reference to timeline truth", {
                                segmentId: seg.id,
                                truthSegmentFound: truthSeg !== seg,
                                oldRef: refIndex + 1,
                                newRef: nextRef,
                                path: newPath,
                                truthPath: truthSeg?.imageFile || truthSeg?.path || "",
                                savedTo: data?.absolute_path || data?.path || "",
                                appended: Boolean(appended?.appended),
                                propagated: 1,
                                slotOnly: true,
                            });
                            writeTimeline({ force: true });
                            draw();
                        });
                    } else {
                        openAppendImagePicker(seg.id);
                    }
                }),
                railBtn("D", "Duplicate block", () => {
                    duplicateVisualSegment(seg);
                }),
                railBtn("X", "Delete block", () => {
                    rippleDeleteVisualSegment(seg);
                })
            );
            block.appendChild(rail);
        }
        if (!isAudio) {
            const nestedSlotRelays = isTimelineImageSegment(seg) && Array.isArray(seg.slot_relays)
                ? seg.slot_relays.filter((relay) => relay && relay.slotRelay !== false)
                    .sort((a, b) => Number(a.start || 0) - Number(b.start || 0))
                : [];
            const promptWin = isTimelineImageSegment(seg)
                ? generationWindowForImage(seg)
                : {
                    start: Math.max(0, Math.round(Number(seg.start || 0))),
                    length: Math.max(1, Math.round(Number(seg.length || 1))),
                    end: Math.max(0, Math.round(Number(seg.start || 0))) + Math.max(1, Math.round(Number(seg.length || 1))),
                };
            const segStart = promptWin.start;
            const segLength = Math.max(1, Math.round(Number(promptWin.length || 1)));
            const segEnd = promptWin.end;
            const promptShell = document.createElement("div");
            promptShell.style.cssText = [
                "position:absolute",
                `left:${innerLeft}px`,
                `right:${innerRight}px`,
                `top:${promptTop}px`,
                `height:${promptHeight}px`,
                "box-sizing:border-box",
                "display:block",
                "overflow:hidden",
                "border-radius:5px",
                "z-index:18",
            ].join(";");
            const makePromptBox = (value, placeholder, onValue, options = {}) => {
                const wrap = document.createElement("div");
                wrap.style.cssText = [
                    "position:absolute",
                    `left:${Math.max(0, Math.min(100, Number(options.leftPct || 0)))}%`,
                    `width:${Math.max(.8, Math.min(100, Number(options.widthPct || 100)))}%`,
                    "top:0",
                    "bottom:0",
                    "min-width:0",
                    "min-height:0",
                    "box-sizing:border-box",
                    "padding-right:4px",
                ].join(";");
                if (options.slotRelay) {
                    const startLine = document.createElement("div");
                    startLine.title = "Slot Relay prompt division";
                    startLine.style.cssText = [
                        "position:absolute",
                        "left:0",
                        "top:0",
                        "bottom:0",
                        "width:3px",
                        "background:rgba(17,17,17,.84)",
                        "box-shadow:1px 0 0 rgba(111,182,124,.72)",
                        "z-index:3",
                        "pointer-events:none",
                    ].join(";");
                    wrap.appendChild(startLine);
                }
                const caption = document.createElement("textarea");
                caption.value = String(value || "");
                caption.placeholder = placeholder;
                caption.spellcheck = false;
                caption.dataset.iamccsV3SegmentId = String(seg.id);
                caption.dataset.iamccsV3Key = options.slotRelay ? "slot_relay_prompt" : "prompt";
                caption.style.cssText = [
                    "width:100%",
                    "height:100%",
                    "min-height:0",
                    "box-sizing:border-box",
                    `padding:${options.slotRelay ? "6px 8px 6px 10px" : "6px 8px"}`,
                    `background:${purple.valueBg}`,
                    `border:1px solid ${options.slotRelay ? "rgba(111,182,124,.72)" : purple.border}`,
                    "border-radius:5px",
                    `color:${purple.valueText}`,
                    `font:${promptFontSize(nestedSlotRelays.length ? 9 : 10)}/1.22 monospace`,
                    "font-weight:700",
                    "outline:none",
                    "resize:none",
                    "overflow-y:auto",
                    "overflow-x:hidden",
                    "box-shadow:inset 0 1px 0 rgba(255,255,255,.66)",
                ].join(";");
                caption.onpointerdown = (event) => event.stopPropagation();
                caption.onclick = (event) => event.stopPropagation();
                caption.ondblclick = (event) => event.stopPropagation();
                const apply = (source) => {
                    markPromptFieldEdited(source);
                    onValue(source.value, source);
                    writeTimeline({ force: true });
                };
                caption.oninput = () => apply(caption);
                caption.onchange = flushTimelineWrite;
                caption.onblur = flushTimelineWrite;
                caption.onkeyup = () => {
                    markPromptFieldEdited(caption);
                    writeTimeline({ force: true });
                    logPromptPersistence(seg, options.slotRelay ? "slot_relay_prompt_keyup" : "timeline_caption_keyup");
                };
                caption.onpaste = () => setTimeout(() => apply(caption), 0);
                caption.oncompositionend = () => apply(caption);
                protectControlDrag(caption);
                wrap.appendChild(caption);
                return wrap;
            };
            const boundaries = nestedSlotRelays.map((relay) => Math.max(segStart, Math.min(segEnd - 1, Math.round(Number(relay.start || segStart)))));
            const segmentRanges = [];
            let cursorFrame = segStart;
            if (nestedSlotRelays.length) {
                const first = Math.max(segStart + 1, boundaries[0]);
                segmentRanges.push({ kind: "base", start: segStart, end: first });
                nestedSlotRelays.forEach((relay, relayIndex) => {
                    const start = Math.max(segStart, Math.min(segEnd - 1, boundaries[relayIndex]));
                    const end = relayIndex + 1 < boundaries.length
                        ? Math.max(start + 1, Math.min(segEnd, boundaries[relayIndex + 1]))
                        : segEnd;
                    segmentRanges.push({ kind: "relay", relay, relayIndex, start, end });
                });
            } else {
                segmentRanges.push({ kind: "base", start: segStart, end: segEnd });
            }
            segmentRanges.forEach((range) => {
                range.leftPct = ((range.start - segStart) / segLength) * 100;
                range.widthPct = ((Math.max(1, range.end - range.start)) / segLength) * 100;
            });
            const baseRange = segmentRanges.find((range) => range.kind === "base") || { leftPct: 0, widthPct: 100 };
            promptShell.appendChild(makePromptBox(String(seg.prompt || ""), nestedSlotRelays.length ? "Base local prompt..." : "Action in this segment...", (value, source) => {
                seg.prompt = value;
                seg.use_prompt = Boolean(String(value || "").trim());
                if (String(value || "").trim()) {
                    seg.relay_manual_off = false;
                    seg.promptrelay_manual_off = false;
                }
                if (isActionBridgeRelaySegment(seg)) syncActionBridgeSourceFromRelay(seg);
                syncSegmentTextPeers(seg.id, "prompt", value, source);
                syncSegmentRelayPeers(seg.id, Boolean(seg.use_prompt), null);
                logPromptPersistence(seg, "timeline_caption_input");
            }, { leftPct: baseRange.leftPct, widthPct: baseRange.widthPct }));
            segmentRanges.filter((range) => range.kind === "relay").forEach((range) => {
                const relay = range.relay;
                const relayIndex = range.relayIndex;
                relay.start = range.start;
                relay.length = Math.max(1, range.end - range.start);
                promptShell.appendChild(makePromptBox(String(relay.prompt || ""), `Slot Relay ${relayIndex + 1} prompt...`, (value) => {
                    relay.prompt = value;
                    relay.note = value;
                    relay.use_prompt = Boolean(String(value || "").trim());
                    relay.relay_kind = "slot";
                    relay.relayKind = "slot";
                    relay.slotRelay = true;
                    relay.parentSegmentId = String(seg.id || "");
                    logPromptPersistence(seg, "slot_relay_prompt_input");
                }, { slotRelay: true, leftPct: range.leftPct, widthPct: range.widthPct }));
            });
            block.appendChild(promptShell);
        }
        const label = document.createElement("div");
        label.style.cssText = `position:absolute;left:${innerLeft}px;top:${truthRailHeight + 4}px;right:${topRightSafe}px;color:#fff;font-size:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-shadow:0 1px 2px #000;`;
        if (isTransitionRelaySegment(seg)) {
            const owner = generationOwnerForFrame(seg.start);
            const startLocal = owner ? generationFrameFor(owner, seg.start) : Math.max(0, Math.round(Number(seg.start || 0)));
            const endLocal = owner ? generationFrameFor(owner, Number(seg.start || 0) + Number(seg.length || 0)) : Math.max(0, Math.round(Number(seg.start || 0) + Number(seg.length || 0)));
            label.textContent = `Gen F${startLocal} - F${endLocal}`;
        } else {
            label.textContent = `${frameLabel(seg.start)} - ${frameLabel(Number(seg.start || 0) + Number(seg.length || 0))}`;
        }
        block.appendChild(label);
        if (!isAudio && isTimelineImageSegment(seg) && String(seg.type || "image") !== "text") {
            const truthPath = String(seg.imageTruthPath || seg.image_truth_path || seg.imageFile || seg.image_file || seg.path || "").trim();
            if (truthPath) {
                const truthBadge = document.createElement("div");
                const truthName = String(seg.imageTruthName || seg.imageName || "").trim() || truthPath.split(/[\\/]/).pop() || truthPath;
                truthBadge.textContent = truthName;
                truthBadge.title = `Backend guide truth: ${truthPath}`;
                truthBadge.style.cssText = `position:absolute;left:${innerLeft}px;right:${topRightSafe}px;top:3px;z-index:30;padding:0 4px;border-radius:3px;background:rgba(5,12,13,.28);border:0;color:#CDBB92;font:8px/1.15 monospace;font-weight:800;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;pointer-events:none;text-shadow:0 1px 2px #000;opacity:.78;`;
                block.appendChild(truthBadge);
            }
        }
        let resize = null;
        let resizeLeft = null;
        if (isAudio) {
            resize = document.createElement("div");
            resize.title = "Drag right edge to trim audio end";
            resize.style.cssText = [
                "position:absolute",
                "right:0",
                "top:0",
                "bottom:0",
                "width:14px",
                "cursor:ew-resize",
                "z-index:48",
                "background:linear-gradient(90deg,rgba(244,213,158,.18),rgba(244,213,158,.72))",
                "border-left:1px solid rgba(255,240,205,.82)",
                "border-radius:0 4px 4px 0",
                "box-shadow:-2px 0 8px rgba(0,0,0,.44), inset 1px 0 0 rgba(255,255,255,.34)",
            ].join(";");
            const rightGrip = document.createElement("div");
            rightGrip.style.cssText = "position:absolute;right:5px;top:50%;width:3px;height:44px;transform:translateY(-50%);border-radius:999px;background:#FFF0C9;box-shadow:0 0 7px rgba(0,0,0,.55);pointer-events:none;";
            resize.appendChild(rightGrip);
            block.appendChild(resize);
            resizeLeft = document.createElement("div");
            resizeLeft.title = "Drag left edge to trim audio start";
            resizeLeft.style.cssText = [
                "position:absolute",
                "left:0",
                "top:0",
                "bottom:0",
                "width:14px",
                "cursor:ew-resize",
                "z-index:48",
                "background:linear-gradient(90deg,rgba(143,208,204,.72),rgba(143,208,204,.18))",
                "border-right:1px solid rgba(196,255,248,.84)",
                "border-radius:4px 0 0 4px",
                "box-shadow:2px 0 8px rgba(0,0,0,.44), inset -1px 0 0 rgba(255,255,255,.30)",
            ].join(";");
            const leftGrip = document.createElement("div");
            leftGrip.style.cssText = "position:absolute;left:5px;top:50%;width:3px;height:44px;transform:translateY(-50%);border-radius:999px;background:#C9FFF8;box-shadow:0 0 7px rgba(0,0,0,.55);pointer-events:none;";
            resizeLeft.appendChild(leftGrip);
            block.appendChild(resizeLeft);
        }

        block.onpointerdown = (event) => {
            if (resize && (event.target === resize || resize.contains(event.target))) return;
            if (resizeLeft && (event.target === resizeLeft || resizeLeft.contains(event.target))) return;
            if (isAudio && !event.target?.closest?.("textarea,input,select,button")) {
                const rect = block.getBoundingClientRect();
                const edgePx = 12;
                if (event.clientX <= rect.left + edgePx) {
                    startTimelineDrag(event, seg, isAudio, "left");
                    return;
                }
                if (event.clientX >= rect.right - edgePx) {
                    startTimelineDrag(event, seg, isAudio, "right");
                    return;
                }
            }
            startTimelineDrag(event, seg, isAudio, "center");
        };
        if (resize) {
            resize.onpointerdown = (event) => {
                event.preventDefault();
                event.stopPropagation();
                startTimelineDrag(event, seg, isAudio, "right");
            };
        }
        if (resizeLeft) {
            resizeLeft.onpointerdown = (event) => {
                event.preventDefault();
                event.stopPropagation();
                startTimelineDrag(event, seg, isAudio, "left");
            };
        }
        if (!isAudio) {
            block.ondblclick = (event) => {
                if (event.target?.tagName === "TEXTAREA") return;
                event.preventDefault();
                event.stopPropagation();
                if (String(seg.type || "image") === "text") openTimelineAddMenu(event, seg, null);
                else if (seg.placeholder || Number(seg.ref || 0) < 1) openAppendImagePicker(seg.id);
                else splitImageSlotAfterSegment(seg);
            };
        }
        block.onclick = () => { selectedId = seg.id; draw(); };
        return block;
    }

    function appendVisualEdgeHandle(seg, edge, tone = "right") {
        if (!seg || String(seg.type || "image") === "audio") return;
        const total = Math.max(1, getTotalFrames());
        const frame = edge === "left"
            ? Math.max(0, Math.round(Number(seg.start || 0)))
            : Math.max(0, Math.round(Number(seg.start || 0) + Number(seg.length || 1)));
        const pct = Math.max(0, Math.min(100, (frame / total) * 100));
        const handle = document.createElement("div");
        handle.className = `iamccs-v3-solid-edge-handle iamccs-v3-solid-edge-${edge}`;
        handle.title = edge === "left"
            ? "Solid edge handle: drag to resize the start of this slot"
            : "Solid edge handle: drag to resize this slot boundary";
        const slot = tone === "slot";
        const warm = tone === "left";
        handle.style.cssText = [
            "position:absolute",
            `left:calc(${pct}% - 9px)`,
            "top:8px",
            "width:9px",
            "height:222px",
            "box-sizing:border-box",
            "cursor:ew-resize",
            "z-index:58",
            "border-radius:4px",
            `background:${slot ? "linear-gradient(180deg,#A8F2A5,#2E7B48)" : (warm ? "linear-gradient(180deg,#6ABDB9,#2C5E63)" : "linear-gradient(180deg,#F0CE78,#A66E32)")}`,
            `border:1px solid ${slot ? "rgba(209,255,211,.94)" : (warm ? "rgba(158,237,232,.88)" : "rgba(255,229,159,.92)")}`,
            "box-shadow:0 0 0 1px rgba(0,0,0,.72),0 6px 14px rgba(0,0,0,.45),inset 0 1px 0 rgba(255,255,255,.36)",
            "opacity:.94",
        ].join(";");
        const notch = document.createElement("div");
        notch.style.cssText = [
            "position:absolute",
            "left:2px",
            "right:2px",
            "top:50%",
            "height:48px",
            "transform:translateY(-50%)",
            "border-radius:999px",
            "background:rgba(6,10,11,.42)",
            "box-shadow:inset 0 1px 2px rgba(0,0,0,.55)",
            "pointer-events:none",
        ].join(";");
        handle.appendChild(notch);
        handle.onpointerdown = (event) => {
            event.preventDefault();
            event.stopPropagation();
            startTimelineDrag(event, seg, false, edge);
        };
        imageTrack.appendChild(handle);
    }

    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    // Bug #2/#3 fix: include placeholder image segments so new "+" slots get resize handles.
    // Bug #4 fix: skip left handle at timeline start (frame 0) and right handle at timeline end (totalFrames).
    function drawVisualEdgeHandles(segments) {
        const total = getTotalFrames();
        const sorted = (segments || [])
            .filter((seg) => seg && String(seg.type || "image") !== "audio")
            .slice()
            .sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
        sorted.forEach((seg, index) => {
            const start = Math.round(Number(seg.start || 0));
            const end = Math.round(Number(seg.start || 0) + Number(seg.length || 1));
            const prev = sorted[index - 1];
            const prevEnd = prev ? Math.round(Number(prev.start || 0) + Number(prev.length || 1)) : -1;
            // Skip left handle at the very start of the timeline — By IAMCCS
            if (start > 0 && (!prev || Math.abs(prevEnd - start) > 1)) appendVisualEdgeHandle(seg, "left", isSlotRelaySegment(seg) ? "slot" : "left");
            // Skip right handle at the very end of the timeline — By IAMCCS
            if (end < total) appendVisualEdgeHandle(seg, "right", isSlotRelaySegment(seg) ? "slot" : "right");
        });
    }

    function drawActionLaneSegments(segments) {
        const total = Math.max(1, getTotalFrames());
        const fps = getFps();
        actionTrack.innerHTML = "";
        const visualSegments = (segments || [])
            .filter(isTimelineImageSegment)
            .slice()
            .sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
        if (!visualSegments.length) {
            const empty = document.createElement("div");
            empty.textContent = "Action Lane: add image frames, then write what happens between them.";
            empty.style.cssText = `position:absolute;left:10px;right:10px;top:10px;height:26px;display:flex;align-items:center;justify-content:center;border:1px dashed ${purple.borderSoft};border-radius:6px;color:${purple.muted};font:10px/1 monospace;font-weight:800;`;
            actionTrack.appendChild(empty);
            return;
        }
        visualSegments.forEach((seg, index) => {
            if (index >= visualSegments.length - 1) return;
            const next = visualSegments[index + 1];
            const start = Math.max(0, Number(seg.start || 0));
            const nextStart = Math.max(start + 1, Number(next.start || (start + Number(seg.length || 1))));
            const widthFrames = Math.max(1, nextStart - start);
            const active = Boolean(seg.step_transition_enabled || String(seg.step_transition_prompt || "").trim());
            const block = document.createElement("div");
            block.style.cssText = [
                "position:absolute",
                `left:${(start / total) * 100}%`,
                `width:${Math.max(0.8, (widthFrames / total) * 100)}%`,
                "top:5px",
                "height:60px",
                "box-sizing:border-box",
                `border:1px solid ${active ? "rgba(223,164,81,.74)" : "rgba(118,103,83,.44)"}`,
                "border-radius:6px",
                `background:${active ? "linear-gradient(135deg,rgba(58,43,25,.95),rgba(25,54,55,.92))" : "rgba(255,255,255,.035)"}`,
                "display:grid",
                "grid-template-columns:minmax(78px,94px) minmax(156px,2fr)",
                "gap:5px",
                "padding:4px",
                "overflow:hidden",
                "z-index:7",
                "pointer-events:none",
            ].join(";");
            block.title = `Action between ${String(seg.label || `frame ${index + 1}`)} and ${String(next.label || `frame ${index + 2}`)} | ${(widthFrames / fps).toFixed(2)}s`;
            const type = document.createElement("select");
            STEP_TRANSITION_OPTIONS.forEach(({ value, label }) => {
                const opt = document.createElement("option");
                opt.value = value;
                opt.textContent = label;
                type.appendChild(opt);
            });
            type.value = STEP_TRANSITION_OPTIONS.some((item) => item.value === String(seg.step_transition_type || "off")) ? String(seg.step_transition_type || "off") : "action_beat";
            type.style.cssText = inputBase() + "height:42px;font-size:9px;font-weight:900;text-align:center;padding:0 4px;background:#F4EFE6;color:#111;";
            type.style.pointerEvents = "auto";
            type.onchange = () => {
                seg.step_transition_type = type.value;
                seg.step_transition_enabled = type.value !== "off" || Boolean(String(seg.step_transition_prompt || "").trim());
                seg.use_prompt = Boolean(seg.step_transition_enabled || String(seg.prompt || "").trim());
                if (seg.step_transition_enabled && Number(seg.step_transition_duration || 0) <= 0) {
                    seg.step_transition_duration = defaultStepTransitionSeconds(type.value, widthFrames / fps);
                    seg.step_transition_arrival = defaultStepTransitionArrival(type.value);
                }
                syncActionBridgeRelaySegment(seg, { forceTiming: true });
                writeTimeline();
                draw();
            };
            const prompt = document.createElement("textarea");
            prompt.value = String(seg.step_transition_prompt || "");
            prompt.placeholder = "Action to next frame...";
            prompt.rows = 1;
            prompt.spellcheck = false;
            prompt.dataset.iamccsV3SegmentId = String(seg.id);
            prompt.dataset.iamccsV3Key = "step_transition_prompt";
            prompt.style.cssText = inputBase() + `height:42px;min-height:0;resize:none;font-size:${promptFontSize(9)};line-height:1.18;padding:6px 8px;background:#F4EFE6;color:#111;`;
            prompt.style.pointerEvents = "auto";
            prompt.onpointerdown = (event) => event.stopPropagation();
            prompt.onclick = (event) => event.stopPropagation();
            prompt.ondblclick = (event) => event.stopPropagation();
            prompt.oninput = () => {
                seg.step_transition_prompt = prompt.value;
                const hasText = Boolean(String(prompt.value || "").trim());
                seg.step_transition_enabled = hasText || seg.step_transition_enabled;
                if (hasText && String(seg.step_transition_type || "off") === "off") {
                    seg.step_transition_type = "action_beat";
                    type.value = "action_beat";
                }
                seg.use_prompt = Boolean(hasText || seg.step_transition_enabled || String(seg.prompt || "").trim());
                syncSegmentTextPeers(seg.id, "step_transition_prompt", prompt.value, prompt);
                syncActionBridgeRelaySegment(seg, { forceTiming: true });
                writeTimeline();
            };
            prompt.onchange = flushTimelineWrite;
            prompt.onblur = flushTimelineWrite;
            protectControlDrag(type);
            protectControlDrag(prompt);
            block.append(type, prompt);
            actionTrack.appendChild(block);
        });
    }

    function drawStepTransitionBridges(segments) {
        const total = Math.max(1, getTotalFrames());
        const visualSegments = (segments || [])
            .filter(isTimelineImageSegment)
            .slice()
            .sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
        visualSegments.forEach((seg, index) => {
            if (!seg.step_transition_enabled || index >= visualSegments.length - 1) return;
            const next = visualSegments[index + 1];
            const end = Math.max(0, Number(seg.start || 0) + Number(seg.length || 0));
            const nextStart = Math.max(0, Number(next?.start || end));
            const bridgeFrame = Math.max(0, Math.min(total, (end + nextStart) / 2));
            const marker = document.createElement("div");
            marker.title = `${stepTransitionLabel(seg.step_transition_type)}: ${String(seg.step_transition_prompt || "Step transition to next frame")}`;
            marker.innerHTML = `
                <svg viewBox="0 0 176 70" width="176" height="70" aria-hidden="true">
                    <path d="M12 44 C48 19 112 15 158 42" fill="none" stroke="rgba(0,0,0,.42)" stroke-width="8" stroke-linecap="round"/>
                    <path d="M12 44 C48 19 112 15 158 42" fill="none" stroke="#F0C247" stroke-width="4" stroke-linecap="round"/>
                    <path d="M153 28 L171 47 L145 55" fill="none" stroke="rgba(0,0,0,.42)" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M153 28 L171 47 L145 55" fill="none" stroke="#F0C247" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
                    <circle cx="12" cy="44" r="4.2" fill="#F0C247" stroke="rgba(0,0,0,.55)" stroke-width="1.4"/>
                </svg>`;
            marker.style.position = "absolute";
            marker.style.left = `${(bridgeFrame / total) * 100}%`;
            marker.style.top = "23px";
            marker.style.width = "176px";
            marker.style.height = "70px";
            marker.style.transform = "translateX(-50%)";
            marker.style.transformOrigin = "50% 0";
            marker.style.zIndex = "42";
            marker.style.pointerEvents = "none";
            marker.style.filter = "drop-shadow(0 2px 4px rgba(0,0,0,.48))";
            imageTrack.appendChild(marker);
            const fps = getFps();
            const requested = Math.max(0, Number(seg.step_transition_duration || 0) || 0);
            const windowFrames = Math.max(1, Math.round((requested > 0 ? requested : Math.max(0.1, (nextStart - Number(seg.start || 0)) / fps)) * fps));
            const bandStart = Math.max(0, Number(seg.start || 0));
            const bandEnd = Math.max(bandStart + 1, Math.min(total, bandStart + windowFrames));
            const band = document.createElement("div");
            band.style.cssText = [
                "position:absolute",
                `left:${(bandStart / total) * 100}%`,
                `width:${Math.max(0.4, ((bandEnd - bandStart) / total) * 100)}%`,
                "top:4px",
                "height:108px",
                "box-sizing:border-box",
                "border:1px solid rgba(223,164,81,.70)",
                "border-radius:7px",
                "background:linear-gradient(90deg, rgba(223,164,81,.12), rgba(41,132,142,.10))",
                "box-shadow:inset 0 0 0 1px rgba(255,255,255,.05)",
                "pointer-events:none",
                "z-index:18",
            ].join(";");
            const label = document.createElement("div");
            const seconds = (bandEnd - bandStart) / fps;
            label.textContent = `${stepTransitionLabel(seg.step_transition_type)} ${seconds.toFixed(1)}s`;
            label.style.cssText = "position:absolute;left:7px;top:6px;max-width:calc(100% - 14px);padding:2px 6px;border-radius:999px;background:rgba(30,25,20,.86);border:1px solid rgba(223,164,81,.55);color:#F4E5C4;font:9px/1.1 monospace;font-weight:900;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
            band.appendChild(label);
            band.title = `${stepTransitionLabel(seg.step_transition_type)} ${(bandStart / fps).toFixed(2)}s -> ${(bandEnd / fps).toFixed(2)}s`;
            imageTrack.appendChild(band);
        });
    }

    function rippleDeleteVisualSegment(target) {
        const removedLength = Math.max(1, Math.round(Number(target?.length || 1)));
        const sorted = (timeline.segments || []).slice().sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
        const index = sorted.findIndex((item) => item.id === target?.id);
        if (index < 0) return;
        const source = sorted[index];
        if (isTimelineImageSegment(source)) {
            const placeholder = {
                ...source,
                type: "image",
                placeholder: true,
                ref: 0,
                imageTruthRef: 0,
                image_truth_ref: 0,
                imageTruthPinned: false,
                image_truth_pinned: false,
                imageTruthSource: "delete_to_placeholder",
                image_truth_source: "delete_to_placeholder",
                imageTruthPath: "",
                image_truth_path: "",
                imageTruthName: "",
                image_truth_name: "",
                imageFile: "",
                image_file: "",
                path: "",
                imageName: "",
                image_name: "",
                label: "empty_slot",
            };
            delete placeholder.file;
            delete placeholder.resolved;
            delete placeholder.basename;
            delete placeholder.slot_relays;
            delete placeholder.slotRelays;
            placeholder.relay_manual_off = true;
            placeholder.promptrelay_manual_off = true;
            sorted[index] = placeholder;
            timeline.segments = sorted;
            selectedId = placeholder.id;
            previewSegments = null;
            console.info("[IAMCCS WAN PURE][Timeline] image deleted to placeholder", {
                segment_id: placeholder.id,
                start: Math.max(0, Math.round(Number(placeholder.start || 0))),
                length: Math.max(1, Math.round(Number(placeholder.length || 1))),
            });
            writeTimeline({ force: true });
            draw();
            return;
        }
        const prev = index > 0 ? sorted[index - 1] : null;
        sorted.splice(index, 1);
        if (prev && String(prev.type || "image") !== "audio") {
            prev.length = Math.max(1, Math.round(Number(prev.length || 1) + removedLength));
        } else if (sorted.length) {
            const first = sorted[0];
            const oldStart = Math.max(0, Math.round(Number(first.start || 0)));
            first.start = 0;
            first.length = Math.max(1, Math.round(Number(first.length || 1) + oldStart + removedLength));
        }
        timeline.segments = sorted;
        selectedId = prev?.id || timeline.segments[0]?.id || null;
        writeTimeline();
        draw();
    }

    function drawBoxes() {
        boxList.innerHTML = "";
        editBox.innerHTML = "";
        const setGuideLockLinked = (target, linked) => {
            const nextLinked = Boolean(linked);
            target.linkGuideLock = nextLinked;
            target.link_guide_lock = nextLinked;
            if (nextLinked) {
                const nextStrength = clampGuideStrength(target.motion ?? target.guideStrength ?? target.guide_strength ?? target.force ?? target.strength ?? defaultForceWidget?.value ?? 1);
                target.motion = nextStrength;
                target.guideStrength = nextStrength;
                target.guide_strength = nextStrength;
                target.force = nextStrength;
                target.strength = nextStrength;
                target.imageLockStrength = nextStrength;
                target.image_lock_strength = nextStrength;
            }
        };
        const syncGuideLockPeerInputs = (segmentId, sourceKey, value) => {
            const peerKey = sourceKey === "guideStrength" ? "imageLockStrength" : "guideStrength";
            const formatted = formatStepperValue(value, 2);
            boxList.querySelectorAll(
                `input[data-iamccs-v3-segment-id="${CSS.escape(String(segmentId || ""))}"][data-iamccs-v3-number-key="${peerKey}"]`
            ).forEach((input) => {
                if (document.activeElement !== input) input.value = formatted;
            });
        };
        const makeField = (seg, labelText, key, type = "input", style = "") => {
            const segmentId = String(seg?.id || "");
            const currentSegment = () => (timeline.segments || []).find((item) => String(item?.id || "") === segmentId) || seg;
            const wrap = document.createElement("label");
            wrap.style.cssText = `display:flex;flex-direction:column;gap:4px;color:${purple.muted};font-size:10px;font-weight:800;text-align:center;min-width:0;${style}`;
            const span = document.createElement("span");
            span.textContent = labelText;
            span.style.cssText = "display:block;text-align:center;line-height:1.2;min-height:13px;overflow:visible;padding-bottom:1px;";
            if (type === "number") {
                const isStrengthKey = key === "guideStrength" || key === "imageLockStrength";
                const step = isStrengthKey ? "0.01" : "1";
                const min = isStrengthKey ? "1" : "0";
                const max = isStrengthKey ? "2" : null;
                const displayNumberValue = key === "length" && isTimelineImageSegment(seg)
                    ? generationWindowForImage(seg).length
                    : (seg[key] ?? "");
                const ctrl = numberStepperControl(displayNumberValue, step, min, max, (value) => {
                    const target = currentSegment();
                    let shouldRedraw = true;
                    if (key === "start") {
                        const nextStart = Math.max(0, Math.round(Number(value || 0)));
                        timeline.segments = edgeDragPreview(timeline.segments, target.id, nextStart - Number(target.start || 0), "left", getTotalFrames());
                    }
                    else if (key === "length") {
                        const nextLength = Math.max(1, Math.round(Number(value || 1)));
                        if (isTimelineImageSegment(target)) {
                            setGenerationWindowLength(target, nextLength);
                        } else {
                            timeline.segments = edgeDragPreview(timeline.segments, target.id, nextLength - Number(target.length || 1), "right", getTotalFrames());
                        }
                    }
                    else if (key === "ref") setSegmentReference(target, value);
                    else if (key === "guideStrength") {
                        const nextStrength = clampGuideStrength(value);
                        target.motion = nextStrength;
                        target.guideStrength = nextStrength;
                        target.guide_strength = nextStrength;
                        target.force = nextStrength;
                        target.strength = nextStrength;
                        target.imageLockStrength = nextStrength;
                        target.image_lock_strength = nextStrength;
                        target.linkGuideLock = true;
                        target.link_guide_lock = true;
                        syncGuideLockPeerInputs(segmentId, key, nextStrength);
                        target.forceCustom = true;
                        delete target.defaultForceSource;
                        shouldRedraw = false;
                    }
                    else if (key === "imageLockStrength") {
                        const nextLock = clampGuideStrength(value);
                        target.motion = nextLock;
                        target.guideStrength = nextLock;
                        target.guide_strength = nextLock;
                        target.force = nextLock;
                        target.strength = nextLock;
                        target.imageLockStrength = nextLock;
                        target.image_lock_strength = nextLock;
                        target.linkGuideLock = true;
                        target.link_guide_lock = true;
                        syncGuideLockPeerInputs(segmentId, key, nextLock);
                        target.forceCustom = true;
                        delete target.defaultForceSource;
                        shouldRedraw = false;
                    }
                    writeTimeline({ force: true });
                    if (shouldRedraw) draw();
                }, { liveInput: isStrengthKey });
                ctrl.style.minWidth = "0";
                ctrl.style.width = "100%";
                ctrl.style.gridTemplateColumns = "24px minmax(46px,1fr) 24px";
                ctrl.style.gap = "7px";
                ctrl.style.padding = "0 3px";
                const ctrlInput = ctrl.querySelector("input");
                if (ctrlInput) {
                    ctrlInput.dataset.iamccsV3SegmentId = String(segmentId);
                    ctrlInput.dataset.iamccsV3NumberKey = String(key);
                }
                ctrl.querySelectorAll("button").forEach((button) => {
                    button.style.minWidth = "24px";
                    button.style.width = "24px";
                    button.style.margin = "0";
                });
                styleValueControls(ctrl);
                wrap.append(span, ctrl);
                return wrap;
            }
            const input = type === "textarea" ? document.createElement("textarea") : document.createElement("input");
            input.value = seg[key] ?? "";
            input.dataset.iamccsV3SegmentId = String(seg.id);
            input.dataset.iamccsV3Key = String(key);
            input.style.cssText = inputBase() + `background:${purple.valueBg};border-color:${purple.border};color:${purple.valueText};font-weight:${type === "textarea" ? "700" : "800"};text-align:${type === "textarea" ? "left" : "center"};${type === "textarea" ? `min-height:54px;resize:vertical;font-size:${promptFontSize(11)};` : "height:28px;"}`;
            input.oninput = () => {
                const target = currentSegment();
                if (key === "prompt") markPromptFieldEdited(input);
                let shouldRedraw = false;
                if (key === "start") {
                    const nextStart = Math.max(0, Math.round(Number(input.value || 0)));
                    timeline.segments = edgeDragPreview(timeline.segments, target.id, nextStart - Number(target.start || 0), "left", getTotalFrames());
                    shouldRedraw = true;
                }
                else if (key === "length") {
                    const nextLength = Math.max(1, Math.round(Number(input.value || 1)));
                    timeline.segments = edgeDragPreview(timeline.segments, target.id, nextLength - Number(target.length || 1), "right", getTotalFrames());
                    shouldRedraw = true;
                }
                else if (key === "ref") {
                    setSegmentReference(target, input.value);
                    shouldRedraw = true;
                }
                else if (key === "guideStrength") {
                    const nextStrength = clampGuideStrength(input.value);
                    target.motion = nextStrength;
                    target.guideStrength = nextStrength;
                    target.guide_strength = nextStrength;
                    target.force = nextStrength;
                    target.strength = nextStrength;
                    target.imageLockStrength = nextStrength;
                    target.image_lock_strength = nextStrength;
                    target.linkGuideLock = true;
                    target.link_guide_lock = true;
                    syncGuideLockPeerInputs(segmentId, key, nextStrength);
                    target.forceCustom = true;
                    delete target.defaultForceSource;
                    shouldRedraw = true;
                }
                else if (key === "imageLockStrength") {
                    const nextLock = clampGuideStrength(input.value);
                    target.motion = nextLock;
                    target.guideStrength = nextLock;
                    target.guide_strength = nextLock;
                    target.force = nextLock;
                    target.strength = nextLock;
                    target.imageLockStrength = nextLock;
                    target.image_lock_strength = nextLock;
                    target.linkGuideLock = true;
                    target.link_guide_lock = true;
                    syncGuideLockPeerInputs(segmentId, key, nextLock);
                    target.forceCustom = true;
                    delete target.defaultForceSource;
                    shouldRedraw = true;
                }
                else {
                    target[key] = input.value;
                    if (key === "prompt") {
                        target.use_prompt = Boolean(String(input.value || "").trim());
                        if (String(input.value || "").trim()) {
                            target.relay_manual_off = false;
                            target.promptrelay_manual_off = false;
                        }
                        syncSegmentRelayPeers(target.id, Boolean(target.use_prompt), null);
                    }
                    if (type === "textarea") syncSegmentTextPeers(target.id, key, input.value, input);
                }
                writeTimeline(key === "prompt" || key === "start" || key === "length" || key === "ref" || key === "guideStrength" || key === "imageLockStrength" ? { force: true } : {});
                if (key === "prompt") logPromptPersistence(target, "inspector_prompt_input");
                if (shouldRedraw) draw();
            };
            input.onchange = flushTimelineWrite;
            input.onblur = flushTimelineWrite;
            if (key === "prompt") {
                input.onkeyup = () => {
                    markPromptFieldEdited(input);
                    writeTimeline({ force: true });
                };
                input.onpaste = () => setTimeout(() => {
                    markPromptFieldEdited(input);
                    seg[key] = input.value;
                    seg.use_prompt = Boolean(String(input.value || "").trim());
                    if (String(input.value || "").trim()) {
                        seg.relay_manual_off = false;
                        seg.promptrelay_manual_off = false;
                    }
                    syncSegmentTextPeers(seg.id, key, input.value, input);
                    syncSegmentRelayPeers(seg.id, Boolean(seg.use_prompt), null);
                    writeTimeline({ force: true });
                    logPromptPersistence(seg, "inspector_prompt_paste");
                }, 0);
                input.oncompositionend = () => {
                    markPromptFieldEdited(input);
                    seg[key] = input.value;
                    seg.use_prompt = Boolean(String(input.value || "").trim());
                    if (String(input.value || "").trim()) {
                        seg.relay_manual_off = false;
                        seg.promptrelay_manual_off = false;
                    }
                    syncSegmentTextPeers(seg.id, key, input.value, input);
                    syncSegmentRelayPeers(seg.id, Boolean(seg.use_prompt), null);
                    writeTimeline({ force: true });
                    logPromptPersistence(seg, "inspector_prompt_compositionend");
                };
            }
            protectControlDrag(input);
            wrap.append(span, input);
            return wrap;
        };
        const makeStepTransitionControl = (seg, index, total) => {
            const wrap = document.createElement("label");
            wrap.style.cssText = `display:grid;grid-template-columns:minmax(170px,210px) minmax(150px,1fr) minmax(170px,230px);gap:8px;align-items:stretch;color:${purple.muted};font-size:10px;font-weight:800;text-align:center;min-width:0;padding:7px;border:1px solid ${seg.step_transition_enabled ? "rgba(223,164,81,.70)" : purple.borderSoft};border-radius:7px;background:${seg.step_transition_enabled ? "rgba(73,50,28,.42)" : "rgba(255,255,255,.025)"};`;
            const select = document.createElement("select");
            STEP_TRANSITION_OPTIONS.forEach(({ value, label }) => {
                const option = document.createElement("option");
                option.value = value;
                option.textContent = label;
                select.appendChild(option);
            });
            const currentType = String(seg.step_transition_type || (seg.step_transition_enabled ? "slow_dolly_in" : "off"));
            select.value = STEP_TRANSITION_OPTIONS.some((item) => item.value === currentType) ? currentType : "slow_dolly_in";
            select.disabled = index >= total - 1;
            select.title = index >= total - 1 ? "Last box has no next frame" : "Transition instruction from this box to the next box";
            select.style.cssText = inputBase() + `height:27px;line-height:27px;padding-top:0;padding-bottom:0;background:${purple.valueBg};border-color:${seg.step_transition_enabled ? "rgba(223,164,81,.80)" : purple.border};color:${purple.valueText};font-weight:800;text-align:center;`;
            select.onpointerdown = (event) => event.stopPropagation();
            select.onchange = () => {
                seg.step_transition_type = select.value;
                seg.step_transition_enabled = select.value !== "off";
                if (seg.step_transition_enabled) {
                    seg.use_prompt = true;
                    if (Number(seg.step_transition_duration || 0) <= 0) {
                        seg.step_transition_duration = defaultStepTransitionSeconds(select.value, stepTransitionAvailableSeconds(seg));
                        seg.step_transition_auto_fit = true;
                        seg.step_transition_arrival = defaultStepTransitionArrival(select.value);
                    }
                    else if (!seg.step_transition_arrival || seg.step_transition_arrival === "auto") {
                        seg.step_transition_arrival = defaultStepTransitionArrival(select.value);
                    }
                }
                syncActionBridgeRelaySegment(seg, { forceTiming: true });
                writeTimeline();
                draw();
            };
            const prompt = document.createElement("input");
            prompt.type = "text";
            prompt.placeholder = "optional positive motion note";
            prompt.value = String(seg.step_transition_prompt || "");
            prompt.disabled = index >= total - 1;
            prompt.style.cssText = inputBase() + `height:24px;background:${purple.valueBg};border-color:${purple.border};color:${purple.valueText};font-weight:700;text-align:left;font-size:${promptFontSize(9)};`;
            prompt.onpointerdown = (event) => event.stopPropagation();
            prompt.oninput = () => {
                seg.step_transition_prompt = prompt.value;
                syncSegmentTextPeers(seg.id, "step_transition_prompt", prompt.value, prompt);
                if (String(prompt.value || "").trim() && String(seg.step_transition_type || "off") === "off") {
                    seg.step_transition_type = "slow_dolly_in";
                    seg.step_transition_enabled = true;
                    seg.use_prompt = true;
                    select.value = "slow_dolly_in";
                }
                syncActionBridgeRelaySegment(seg, { forceTiming: true });
                writeTimeline();
            };
            prompt.onchange = flushTimelineWrite;
            prompt.onblur = flushTimelineWrite;
            protectControlDrag(select);
            protectControlDrag(prompt);
            const durationSeconds = Math.max(0, Number(seg.step_transition_duration || 0) || 0);
            const availableSeconds = stepTransitionAvailableSeconds(seg);
            const durationWrap = document.createElement("div");
            durationWrap.style.cssText = "display:grid;grid-template-columns:minmax(82px,.72fr) minmax(96px,1fr);gap:6px;align-items:end;min-width:0;";
            const durationCtrl = numberStepperControl(durationSeconds, "0.1", "0", null, (value) => {
                const nextSeconds = Math.max(0, Number(value || 0));
                seg.step_transition_duration = nextSeconds;
                seg.step_transition_enabled = nextSeconds > 0 ? true : seg.step_transition_enabled;
                if (nextSeconds > 0 && String(seg.step_transition_type || "off") === "off") {
                    seg.step_transition_type = "slow_dolly_in";
                    seg.use_prompt = true;
                }
                syncActionBridgeRelaySegment(seg, { forceTiming: true });
                writeTimeline();
                draw();
            }, { liveInput: false });
            durationCtrl.style.gridTemplateColumns = "24px minmax(42px,1fr) 24px";
            durationCtrl.style.gap = "5px";
            durationCtrl.style.minWidth = "0";
            durationCtrl.querySelectorAll("button").forEach((button) => {
                button.style.width = "24px";
                button.style.minWidth = "24px";
            });
            styleValueControls(durationCtrl);
            const durationLabel = document.createElement("label");
            durationLabel.style.cssText = "display:flex;flex-direction:column;gap:3px;min-width:0;";
            const durationTitle = document.createElement("span");
            durationTitle.textContent = "Seconds";
            durationTitle.style.cssText = "display:flex;align-items:center;justify-content:center;height:11px;line-height:10px;font-size:8px;text-align:center;color:inherit;overflow:visible;";
            durationLabel.append(durationTitle, durationCtrl);
            const arrivalSelect = document.createElement("select");
            STEP_TRANSITION_ARRIVAL_OPTIONS.forEach(({ value, label }) => {
                const option = document.createElement("option");
                option.value = value;
                option.textContent = label;
                arrivalSelect.appendChild(option);
            });
            arrivalSelect.value = STEP_TRANSITION_ARRIVAL_OPTIONS.some((item) => item.value === String(seg.step_transition_arrival || "auto")) ? String(seg.step_transition_arrival || "auto") : "auto";
            arrivalSelect.disabled = index >= total - 1;
            arrivalSelect.title = "Where the arrival toward the next frame should happen inside the transition";
            arrivalSelect.style.cssText = inputBase() + `height:27px;line-height:27px;padding-top:0;padding-bottom:0;background:${purple.valueBg};border-color:${purple.border};color:${purple.valueText};font-weight:800;text-align:center;`;
            arrivalSelect.onpointerdown = (event) => event.stopPropagation();
            arrivalSelect.onchange = () => {
                seg.step_transition_arrival = arrivalSelect.value;
                writeTimeline();
                draw();
            };
            protectControlDrag(arrivalSelect);
            const arrivalLabel = document.createElement("label");
            arrivalLabel.style.cssText = "display:flex;flex-direction:column;gap:3px;min-width:0;";
            const arrivalTitle = document.createElement("span");
            arrivalTitle.textContent = "Arrival";
            arrivalTitle.style.cssText = "display:flex;align-items:center;justify-content:center;height:11px;line-height:10px;font-size:8px;text-align:center;color:inherit;overflow:visible;";
            arrivalLabel.append(arrivalTitle, arrivalSelect);
            durationWrap.append(durationLabel, arrivalLabel);
            const autoFit = document.createElement("label");
            autoFit.style.cssText = "display:flex;align-items:center;justify-content:center;gap:6px;color:#DFA451;font-size:8px;font-weight:900;line-height:1;";
            const autoCheck = document.createElement("input");
            autoCheck.type = "checkbox";
            autoCheck.checked = seg.step_transition_auto_fit !== false;
            autoCheck.disabled = index >= total - 1;
            autoCheck.style.cssText = `width:14px;height:14px;accent-color:${purple.accent};`;
            autoCheck.onpointerdown = (event) => event.stopPropagation();
            autoCheck.onchange = () => {
                seg.step_transition_auto_fit = Boolean(autoCheck.checked);
                if (Number(seg.step_transition_duration || 0) > 0) syncActionBridgeRelaySegment(seg, { forceTiming: true });
                writeTimeline();
                draw();
            };
            autoFit.append(autoCheck, document.createTextNode("Auto fit next frame"));
            autoFit.style.justifyContent = "flex-start";
            const applyTiming = document.createElement("button");
            applyTiming.type = "button";
            applyTiming.textContent = "Apply Timing";
            applyTiming.disabled = index >= total - 1 || !seg.step_transition_enabled;
            applyTiming.title = "Apply the transition seconds to the real timeline by rippling the next frames.";
            const justApplied = transitionAppliedStamp && Date.now() - transitionAppliedStamp < 1600;
            applyTiming.style.cssText = `height:25px;border:1px solid ${justApplied ? "rgba(123,210,155,.84)" : "rgba(223,164,81,.72)"};border-radius:5px;background:${justApplied ? "rgba(45,103,67,.90)" : seg.step_transition_enabled ? "rgba(96,64,34,.88)" : "rgba(255,255,255,.035)"};color:${seg.step_transition_enabled ? "#F4E5C4" : purple.muted};font-size:9px;font-weight:900;cursor:pointer;transform:${justApplied ? "translateY(1px)" : "none"};box-shadow:${justApplied ? "inset 0 2px 4px rgba(0,0,0,.35)" : "inset 0 1px 0 rgba(255,255,255,.10)"};`;
            applyTiming.onpointerdown = (event) => {
                event.preventDefault();
                event.stopPropagation();
            };
            applyTiming.onclick = (event) => {
                event.preventDefault();
                event.stopPropagation();
                seg.step_transition_auto_fit = true;
                const nextSeconds = Number(seg.step_transition_duration || 0) || defaultStepTransitionSeconds(seg.step_transition_type, stepTransitionAvailableSeconds(seg));
                seg.step_transition_duration = nextSeconds;
                seg.step_transition_enabled = true;
                if (String(seg.step_transition_type || "off") === "off") seg.step_transition_type = "slow_dolly_in";
                fitStepTransitionDuration(seg, nextSeconds);
                syncActionBridgeRelaySegment(seg, { forceTiming: true });
                transitionAppliedStamp = Date.now();
                writeTimeline();
                draw();
                setTimeout(() => {
                    if (Date.now() - transitionAppliedStamp >= 1200) draw();
                }, 1300);
            };
            protectControlDrag(applyTiming);
            const hint = document.createElement("div");
            const desired = Math.max(0, Number(seg.step_transition_duration || 0) || 0);
            const overflow = desired > 0 && availableSeconds > 0 ? Math.max(0, desired - availableSeconds) : 0;
            const startFrame = Math.round(Number(seg.start || 0));
            const endFrame = startFrame + Math.max(1, Math.round((desired || availableSeconds || 0.1) * getFps()));
            hint.textContent = overflow > 0
                ? `Needs +${overflow.toFixed(2)}s before next frame`
                : (desired > 0 ? `${desired.toFixed(1)}s | ${(startFrame / getFps()).toFixed(2)}s -> ${(endFrame / getFps()).toFixed(2)}s | ${stepTransitionArrivalLabel(seg.step_transition_arrival)}` : "0 = use full segment timing");
            const left = document.createElement("div");
            left.style.cssText = "display:flex;flex-direction:column;gap:5px;min-width:0;";
            left.append(
                makeStepTransitionHeader(Boolean(seg.step_transition_enabled), stepTransitionLabel(seg.step_transition_type), "box to next"),
                select,
                prompt
            );
            const center = document.createElement("div");
            center.style.cssText = "display:flex;flex-direction:column;gap:6px;justify-content:center;min-width:0;";
            center.append(durationWrap, autoFit, applyTiming);
            const monitor = document.createElement("div");
            const available = availableSeconds > 0 ? availableSeconds : desired;
            const arrivalMode = String(seg.step_transition_arrival || defaultStepTransitionArrival(seg.step_transition_type));
            const holdPct = arrivalMode === "very_late" ? 42 : arrivalMode === "late" ? 28 : arrivalMode === "middle" ? 12 : 0;
            const arrivePct = arrivalMode === "early" ? 38 : arrivalMode === "middle" ? 24 : 16;
            const movePct = Math.max(18, 100 - holdPct - arrivePct);
            monitor.style.cssText = [
                "position:relative",
                "display:grid",
                "grid-template-columns:1fr 1fr",
                "gap:6px 9px",
                "align-content:center",
                "min-width:0",
                "padding:8px 9px",
                "border:1px solid rgba(120,168,166,.42)",
                "border-radius:7px",
                "background:linear-gradient(180deg, rgba(7,18,20,.88), rgba(21,25,25,.84))",
                "box-shadow:inset 0 1px 0 rgba(255,255,255,.07), inset 0 -10px 18px rgba(0,0,0,.18)",
                `color:${purple.text}`,
                "font:9px/1.15 monospace",
                "font-weight:900",
            ].join(";");
            monitor.innerHTML = `
                <span style="grid-column:1 / -1;color:#8FD0CC;text-transform:uppercase;font-size:8px;">Timing monitor</span>
                <span style="grid-column:1 / -1;display:grid;grid-template-columns:${holdPct}fr ${movePct}fr ${arrivePct}fr;height:9px;border-radius:999px;overflow:hidden;border:1px solid rgba(143,208,204,.34);background:rgba(0,0,0,.26);">
                    <i title="hold" style="display:block;background:rgba(126,112,92,.58);"></i>
                    <i title="camera move" style="display:block;background:linear-gradient(90deg, rgba(223,164,81,.78), rgba(41,132,142,.82));"></i>
                    <i title="arrival" style="display:block;background:rgba(143,208,204,.82);"></i>
                </span>
                ${justApplied ? `<span style="grid-column:1 / -1;justify-self:start;padding:2px 6px;border-radius:999px;background:rgba(65,122,76,.86);border:1px solid rgba(123,210,155,.72);color:#D7F5D6;font-size:8px;">APPLIED</span>` : ""}
                <span style="color:#F4E5C4;">${desired > 0 ? desired.toFixed(1) : "auto"}s</span>
                <span style="text-align:right;color:${overflow > 0 ? "#F1C77A" : "#AEBBB6"};">${overflow > 0 ? `+${overflow.toFixed(2)}s` : "fit"}</span>
                <span style="grid-column:1 / -1;color:#EDE3D0;">${(startFrame / getFps()).toFixed(2)}s -> ${(endFrame / getFps()).toFixed(2)}s</span>
                <span>${Math.max(1, Math.round((desired || available || 0.1) * getFps()))}f</span>
                <span style="text-align:right;">${stepTransitionArrivalLabel(seg.step_transition_arrival)}</span>
            `;
            monitor.title = hint.textContent;
            wrap.append(left, center, monitor);
            return wrap;
        };
        const makeSegmentSummary = (seg, index, total) => {
            const fps = getFps();
            const startFrame = Math.max(0, Math.round(Number(seg.start || 0)));
            const lenFrame = Math.max(1, Math.round(Number(seg.length || 1)));
            const endFrame = startFrame + lenFrame;
            const summary = document.createElement("div");
            summary.style.cssText = [
                "align-self:start",
                "display:flex",
                "flex-direction:column",
                "align-items:center",
                "justify-content:center",
                "gap:3px",
                "min-width:0",
                "padding:5px 8px",
                `border:1px solid ${purple.borderSoft}`,
                "border-radius:6px",
                "background:linear-gradient(180deg, rgba(39,68,57,.70), rgba(23,42,34,.56))",
                "box-shadow:inset 0 1px 0 rgba(255,255,255,.10), 0 4px 12px rgba(0,0,0,.14)",
                "box-sizing:border-box",
                "overflow:hidden",
                "text-align:center",
            ].join(";");
            const title = document.createElement("div");
            title.textContent = "Segment";
            title.style.cssText = `max-width:100%;padding:2px 7px;border-radius:999px;border:1px solid rgba(111,182,210,.32);background:rgba(111,182,210,.14);color:${purple.muted};font-size:7px;font-weight:900;text-transform:uppercase;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:1;`;
            const range = document.createElement("div");
            range.textContent = `${(startFrame / fps).toFixed(2)}s -> ${(endFrame / fps).toFixed(2)}s`;
            range.style.cssText = `color:${purple.text};font-size:10px;font-weight:900;white-space:nowrap;line-height:1;`;
            const meta = document.createElement("div");
            meta.textContent = `${lenFrame}f / ${(lenFrame / fps).toFixed(2)}s`;
            meta.style.cssText = `max-width:100%;color:${purple.muted};font-size:8px;font-weight:900;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:1;`;
            summary.append(title, range, meta);
            return summary;
        };
        const relayMonitorSegments = (seg) => {
            const win = generationWindowForImage(seg);
            const clampFrame = (value) => Math.max(win.start, Math.min(win.end - 1, Math.round(Number(value || win.start))));
            const active = [];
            const basePrompt = String(seg.prompt || "").trim();
            const baseOn = basePrompt && seg.relay_manual_off !== true && seg.promptrelay_manual_off !== true && seg.use_prompt !== false && String(seg.use_prompt).toLowerCase() !== "false";
            if (baseOn) {
                active.push({
                    kind: "BASE",
                    id: String(seg.id || "base"),
                    label: "Local prompt",
                    start: win.start,
                    prompt: basePrompt,
                });
            }
            if (isTimelineImageSegment(seg) && Array.isArray(seg.slot_relays)) {
                seg.slot_relays
                    .filter((relay) => relay && relay.slotRelay !== false)
                    .forEach((relay, relayIndex) => {
                        const prompt = String(relay.prompt || "").trim();
                        const relayOn = prompt && relay.use_prompt !== false && relay.relay_manual_off !== true && relay.promptrelay_manual_off !== true;
                        if (!relayOn) return;
                        active.push({
                            kind: "SLOT",
                            id: String(relay.id || `slot_${relayIndex + 1}`),
                            label: `Slot Relay ${relayIndex + 1}`,
                            start: clampFrame(relay.start),
                            prompt,
                        });
                    });
            }
            active.sort((a, b) => a.start - b.start || (a.kind === "BASE" ? -1 : 1));
            return active.map((item, index) => {
                const nextStart = index + 1 < active.length ? active[index + 1].start : win.end;
                const start = Math.max(win.start, Math.min(win.end - 1, item.start));
                const end = Math.max(start + 1, Math.min(win.end, nextStart));
                return {
                    ...item,
                    localStart: Math.max(0, start - win.start),
                    length: Math.max(1, end - start),
                    end: Math.max(1, end - win.start),
                };
            });
        };
        const makeRelayGenerationMonitor = (seg) => {
            const fps = getFps();
            const win = generationWindowForImage(seg);
            const relays = relayMonitorSegments(seg);
            const totalFrames = Math.max(1, Math.round(Number(win.length || seg.length || 1)));
            const activeFrames = relays.reduce((sum, relay) => sum + Math.max(1, Math.round(Number(relay.length || 1))), 0);
            const monitor = document.createElement("div");
            monitor.title = "PromptRelay monitor: local prompt is Relay 01; Slot Relays split the same WAN generation frame budget.";
            monitor.style.cssText = [
                "position:relative",
                "display:grid",
                "grid-template-columns:138px minmax(0,1fr)",
                "gap:7px 10px",
                "align-items:stretch",
                "width:100%",
                "min-height:58px",
                "padding:7px 9px",
                "box-sizing:border-box",
                "border:1px solid rgba(118,229,154,.58)",
                "border-radius:6px",
                "background:linear-gradient(180deg,rgba(3,16,11,.96),rgba(1,8,7,.98))",
                "box-shadow:inset 0 0 0 1px rgba(185,255,199,.08), inset 0 0 18px rgba(63,220,123,.10), 0 5px 14px rgba(0,0,0,.22)",
                "overflow:hidden",
                "font:9px/1.18 monospace",
                "font-weight:900",
                "color:#BDF8B4",
            ].join(";");
            const scan = document.createElement("div");
            scan.style.cssText = [
                "position:absolute",
                "inset:0",
                "pointer-events:none",
                "background:repeating-linear-gradient(0deg,rgba(255,255,255,.035) 0,rgba(255,255,255,.035) 1px,transparent 1px,transparent 4px)",
                "mix-blend-mode:screen",
                "opacity:.42",
            ].join(";");
            const summary = document.createElement("div");
            summary.style.cssText = [
                "position:relative",
                "z-index:1",
                "display:grid",
                "grid-template-columns:1fr 1fr",
                "gap:4px",
                "align-content:center",
                "min-width:0",
            ].join(";");
            const summaryLine = (label, value, color = "#BDF8B4") => {
                const box = document.createElement("div");
                box.style.cssText = [
                    "display:flex",
                    "flex-direction:column",
                    "gap:2px",
                    "min-width:0",
                    "padding:3px 5px",
                    "border:1px solid rgba(112,236,149,.22)",
                    "border-radius:3px",
                    "background:rgba(0,0,0,.34)",
                ].join(";");
                const l = document.createElement("span");
                l.textContent = label;
                l.style.cssText = "color:#66BA74;font-size:7px;text-transform:uppercase;letter-spacing:0;";
                const v = document.createElement("span");
                v.textContent = value;
                v.style.cssText = `color:${color};font-size:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;`;
                box.append(l, v);
                return box;
            };
            summary.append(
                summaryLine("Relays", String(relays.length), relays.length ? "#D8FFD0" : "#F1C77A"),
                summaryLine("Frames", String(totalFrames)),
                summaryLine("Seconds", `${(totalFrames / fps).toFixed(2)}s`),
                summaryLine("Covered", `${Math.min(activeFrames, totalFrames)}f`)
            );
            const lanes = document.createElement("div");
            lanes.style.cssText = [
                "position:relative",
                "z-index:1",
                "display:flex",
                "flex-direction:column",
                "gap:4px",
                "min-width:0",
                "max-height:74px",
                "overflow:auto",
                "padding-right:2px",
            ].join(";");
            if (!relays.length) {
                const empty = document.createElement("div");
                empty.textContent = "NO LOCAL PROMPT RELAY ACTIVE";
                empty.style.cssText = "height:100%;display:flex;align-items:center;color:#F1C77A;font-size:10px;";
                lanes.appendChild(empty);
            } else {
                relays.forEach((relay, index) => {
                    const row = document.createElement("div");
                    const pct = Math.max(1, Math.min(100, Math.round((relay.length / totalFrames) * 100)));
                    row.title = `${relay.label}: generation frames F${relay.localStart} -> F${relay.end} (${relay.length}f)`;
                    row.style.cssText = [
                        "display:grid",
                        "grid-template-columns:76px minmax(70px,.55fr) minmax(0,1fr)",
                        "gap:6px",
                        "align-items:center",
                        "min-height:18px",
                        "min-width:0",
                    ].join(";");
                    const tag = document.createElement("span");
                    tag.textContent = `${String(index + 1).padStart(2, "0")} ${relay.kind}`;
                    tag.style.cssText = `display:flex;align-items:center;justify-content:center;height:16px;border-radius:3px;border:1px solid ${relay.kind === "BASE" ? "rgba(124,214,255,.42)" : "rgba(155,255,177,.48)"};background:${relay.kind === "BASE" ? "rgba(56,144,178,.20)" : "rgba(64,186,99,.20)"};color:${relay.kind === "BASE" ? "#9DEBFF" : "#C7FFC7"};font-size:8px;`;
                    const frames = document.createElement("span");
                    frames.textContent = `F${relay.localStart}-${relay.end} / ${relay.length}f`;
                    frames.style.cssText = "color:#E7FFD9;font-size:9px;white-space:nowrap;";
                    const barWrap = document.createElement("span");
                    barWrap.style.cssText = "display:block;height:10px;border-radius:999px;border:1px solid rgba(117,255,154,.25);background:rgba(0,0,0,.38);overflow:hidden;min-width:0;";
                    const bar = document.createElement("i");
                    bar.style.cssText = `display:block;width:${pct}%;height:100%;background:linear-gradient(90deg,#73FF8A,#CFFFAF);box-shadow:0 0 10px rgba(115,255,138,.35);`;
                    barWrap.appendChild(bar);
                    row.append(tag, frames, barWrap);
                    lanes.appendChild(row);
                });
            }
            monitor.append(scan, summary, lanes);
            return monitor;
        };
        const resizeRelayBridgeSeconds = (seg, seconds) => {
            const fps = getFps();
            const nextLength = Math.max(1, Math.round(Math.max(0.05, Number(seconds || 0)) * fps));
            const oldStart = Math.round(Number(seg.start || 0));
            const oldLength = Math.max(1, Math.round(Number(seg.length || 1)));
            if (isSlotRelaySegment(seg)) {
                const parent = (timeline.segments || []).find((item) => item.id === seg.parentSegmentId);
                const parentEnd = parent ? Math.round(Number(parent.start || 0) + Number(parent.length || 1)) : oldStart + nextLength;
                seg.length = Math.max(1, Math.min(nextLength, Math.max(1, parentEnd - oldStart)));
                return;
            }
            const oldEnd = oldStart + oldLength;
            const delta = nextLength - oldLength;
            seg.length = nextLength;
            if (delta !== 0) {
                (timeline.segments || []).forEach((item) => {
                    if (item.id !== seg.id && Number(item.start || 0) >= oldEnd) {
                        item.start = Math.max(0, Math.round(Number(item.start || 0) + delta));
                    }
                });
                (timeline.audioSegments || []).forEach((item) => {
                    if (Number(item.start || 0) >= oldEnd) {
                        item.start = Math.max(0, Math.round(Number(item.start || 0) + delta));
                    }
                });
                ensureDurationForFrames(endOfSegments(timeline.segments));
            }
        };
        const makeRelayBridgeCard = (seg, index) => {
            const fps = getFps();
            const isSlotRelay = isSlotRelaySegment(seg);
            const startFrame = Math.max(0, Math.round(Number(seg.start || 0)));
            const lenFrame = Math.max(1, Math.round(Number(seg.length || 1)));
            const endFrame = startFrame + lenFrame;
            const titleText = isSlotRelay ? "Slot Relay" : "Transition Relay";
            const relayHint = isSlotRelay
                ? "Slot Relay belongs inside one image slot and divides that slot's PromptRelay frame budget."
                : "Transition Relay belongs to the previous WAN generation chunk and shares that chunk's frame budget.";
            const card = document.createElement("div");
            card.dataset.iamccsV3BoxSegmentId = String(seg.id || "");
            card.title = relayHint;
            card.style.cssText = [
                "display:grid",
                "grid-template-columns:38px minmax(190px,240px) minmax(0,1fr) minmax(150px,190px) 32px",
                "gap:10px",
                "align-items:stretch",
                "margin:0 0 8px 46px",
                "padding:8px 10px",
                (selectedId === seg.id ? `border:2px solid ${purple.accent}` : "border:1px solid rgba(111,182,210,.42)"),
                "border-radius:6px",
                (isSlotRelay ? "background:linear-gradient(180deg,rgba(31,43,35,.96),rgba(0,0,0,.92))" : "background:linear-gradient(180deg,rgba(8,28,40,.96),rgba(0,0,0,.92))"),
                (selectedId === seg.id ? "box-shadow:0 0 0 2px rgba(141,231,255,.20),inset 0 1px 0 rgba(255,255,255,.08),0 4px 10px rgba(0,0,0,.18)" : "box-shadow:inset 0 1px 0 rgba(255,255,255,.08),0 4px 10px rgba(0,0,0,.18)"),
                "box-sizing:border-box",
            ].join(";");
            const badge = document.createElement("button");
            badge.type = "button";
            badge.textContent = isSlotRelay ? "S" : "T";
            badge.title = isSlotRelay ? "Select this slot relay" : "Select this transition relay";
            badge.style.cssText = `align-self:center;justify-self:center;display:flex;align-items:center;justify-content:center;width:28px;height:28px;border:1px solid ${isSlotRelay ? "rgba(144,219,160,.68)" : "rgba(111,182,210,.58)"};border-radius:999px;background:#000;color:${isSlotRelay ? "#B7F0B8" : purple.accent};font-size:10px;font-weight:900;cursor:pointer;box-shadow:inset 0 1px 0 rgba(255,255,255,.10);`;
            badge.onclick = () => { selectedId = seg.id; draw(); };
            addPressPreview(badge, {
                pressedBg: purple.buttonPress,
                pressedColor: purple.valueText,
                pressedBorder: purple.accent,
            });
            const meta = document.createElement("div");
            meta.style.cssText = "display:flex;flex-direction:column;gap:5px;justify-content:center;min-width:0;";
            const title = document.createElement("div");
            title.textContent = titleText;
            title.title = relayHint;
            title.style.cssText = `display:flex;align-items:center;justify-content:center;height:18px;border-radius:999px;border:1px solid ${isSlotRelay ? "rgba(144,219,160,.48)" : "rgba(111,182,210,.44)"};background:${isSlotRelay ? "linear-gradient(180deg,rgba(144,219,160,.20),rgba(0,0,0,.32))" : "linear-gradient(180deg,rgba(111,182,210,.20),rgba(0,0,0,.32))"};color:${isSlotRelay ? "#B7F0B8" : purple.accent};font-size:8px;font-weight:900;text-transform:uppercase;`;
            const range = document.createElement("div");
            range.textContent = `${(startFrame / fps).toFixed(2)}s -> ${(endFrame / fps).toFixed(2)}s`;
            range.style.cssText = `color:${purple.text};font:10px/1 monospace;font-weight:900;text-align:center;white-space:nowrap;`;
            const name = document.createElement("input");
            name.type = "text";
            name.value = String(seg.label || (isSlotRelay ? "slot_relay" : "transition_relay"));
            name.title = isSlotRelay ? "Slot relay label" : "Transition relay label";
            name.style.cssText = inputBase() + `height:22px;background:${purple.valueBg};border-color:rgba(95,169,130,.44);color:${purple.valueText};font:${promptFontSize(9)}/1 monospace;font-weight:800;text-align:center;`;
            name.onpointerdown = (event) => event.stopPropagation();
            name.oninput = () => {
                seg.label = name.value || (isSlotRelay ? "slot_relay" : "transition_relay");
                writeTimeline();
            };
            protectControlDrag(name);
            meta.append(title, range, name);
            const prompt = document.createElement("textarea");
            prompt.value = String(seg.prompt || "");
            prompt.placeholder = isSlotRelay
                ? "PromptRelay text for this phase inside the image slot..."
                : "PromptRelay text for the transition toward the next image...";
            prompt.title = relayHint;
            prompt.dataset.iamccsV3SegmentId = String(seg.id);
            prompt.dataset.iamccsV3Key = "prompt";
            prompt.style.cssText = `width:100%;height:68px;min-height:68px;box-sizing:border-box;padding:7px 9px;background:${purple.valueBg};border:1px solid rgba(95,169,130,.42);border-radius:5px;color:${purple.valueText};font:${promptFontSize(10)}/1.26 monospace;font-weight:700;outline:none;resize:none;overflow-y:auto;box-shadow:inset 0 1px 0 rgba(255,255,255,.56);`;
            prompt.onpointerdown = (event) => event.stopPropagation();
            prompt.oninput = () => {
                markPromptFieldEdited(prompt);
                seg.prompt = prompt.value;
                seg.note = prompt.value;
                seg.use_prompt = Boolean(String(prompt.value || "").trim());
                if (String(prompt.value || "").trim()) {
                    seg.relay_manual_off = false;
                    seg.promptrelay_manual_off = false;
                }
                if (isActionBridgeRelaySegment(seg)) syncActionBridgeSourceFromRelay(seg);
                syncSegmentTextPeers(seg.id, "prompt", prompt.value, prompt);
                syncSegmentRelayPeers(seg.id, Boolean(seg.use_prompt), null);
                writeTimeline();
            };
            prompt.onchange = flushTimelineWrite;
            prompt.onblur = flushTimelineWrite;
            protectControlDrag(prompt);
            const timing = document.createElement("div");
            timing.style.cssText = "display:grid;grid-template-rows:auto auto;gap:7px;align-content:center;min-width:0;";
            const secondsTitle = document.createElement("div");
            secondsTitle.textContent = isSlotRelay ? "Slot relay seconds" : "Relay seconds";
            secondsTitle.style.cssText = `color:${purple.muted};font-size:8px;font-weight:900;text-align:center;text-transform:uppercase;`;
            const seconds = numberStepperControl(lenFrame / fps, "0.1", "0.1", null, (value) => {
                resizeRelayBridgeSeconds(seg, value);
                if (String(seg.prompt || "").trim()) seg.use_prompt = true;
                writeTimeline();
                draw();
            }, { liveInput: false });
            seconds.style.gridTemplateColumns = "24px minmax(54px,1fr) 24px";
            seconds.style.gap = "5px";
            seconds.querySelectorAll("button").forEach((button) => {
                button.style.width = "24px";
                button.style.minWidth = "24px";
            });
            styleValueControls(seconds);
            const status = document.createElement("div");
            const active = Boolean(seg.use_prompt !== false && String(seg.prompt || "").trim());
            status.textContent = active ? "PromptRelay ON" : "PromptRelay off";
            status.style.cssText = `display:flex;align-items:center;justify-content:center;height:22px;border-radius:5px;border:1px solid ${active ? "rgba(141,231,255,.58)" : purple.borderSoft};background:${active ? "linear-gradient(180deg,rgba(111,182,210,.22),rgba(0,0,0,.34))" : "rgba(0,0,0,.22)"};color:${active ? purple.accent : purple.muted};font:8px/1 monospace;font-weight:900;text-transform:uppercase;`;
            timing.append(secondsTitle, seconds, status);
            const actions = document.createElement("div");
            actions.style.cssText = "display:grid;gap:5px;align-content:start;";
            const remove = document.createElement("button");
            remove.type = "button";
            remove.textContent = "X";
            remove.title = isSlotRelay ? "Delete this slot relay" : "Delete this transition relay";
            remove.style.cssText = `height:24px;border:1px solid ${purple.danger};border-radius:4px;background:#6B302A;color:#FFF2E4;font-size:10px;font-weight:900;cursor:pointer;`;
            remove.onpointerdown = (event) => { event.preventDefault(); event.stopPropagation(); };
            remove.onclick = (event) => {
                event.preventDefault();
                const parentSegmentId = isSlotRelaySegment(seg) ? String(seg.parentSegmentId || "") : "";
                timeline.segments = (timeline.segments || []).filter((item) => item.id !== seg.id);
                if (parentSegmentId) relayoutSlotRelaysForSource(parentSegmentId);
                selectedId = timeline.segments.find((item) => String(item.type || "image") !== "text")?.id || null;
                writeTimeline({ force: true });
                draw();
            };
            actions.append(protectControlDrag(addPressPreview(remove, {
                pressedBg: "linear-gradient(180deg,#E9A39F,#8A3A32)",
                pressedColor: "#FFF8F0",
                pressedBorder: "#FFC2BA",
            })));
            card.append(badge, meta, prompt, timing, actions);
            return card;
        };
        const orderedBoxSegments = (timeline.segments || [])
            .filter((seg) => String(seg.type || "image") !== "audio")
            .slice()
            .sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
        if (!orderedBoxSegments.length) {
            const emptyState = document.createElement("div");
            emptyState.style.cssText = `display:grid;gap:10px;justify-items:center;padding:18px 14px;border:1px dashed ${purple.border};border-radius:8px;background:linear-gradient(180deg,rgba(255,248,236,.08),rgba(0,0,0,.12));color:${purple.text};text-align:center;`;
            const title = document.createElement("div");
            title.textContent = "No image boxes yet";
            title.style.cssText = "font-size:13px;font-weight:900;letter-spacing:.01em;";
            const hint = document.createElement("div");
            hint.textContent = "Use Add Image to create the first shot, or add text/audio and build the sequence from there.";
            hint.style.cssText = `max-width:460px;color:${purple.muted};font-size:11px;font-weight:800;line-height:1.35;`;
            const controls = document.createElement("div");
            controls.style.cssText = "display:flex;gap:8px;flex-wrap:wrap;justify-content:center;";
            const emptyButton = (label, titleText, handler) => {
                const b = document.createElement("button");
                b.type = "button";
                b.textContent = label;
                b.title = titleText;
                b.style.cssText = `min-width:120px;height:30px;border:1px solid ${purple.borderSoft};border-radius:6px;background:${purple.button};color:${purple.text};font-size:11px;font-weight:900;cursor:pointer;`;
                b.onpointerdown = (event) => { event.preventDefault(); event.stopPropagation(); };
                b.onclick = (event) => { event.preventDefault(); handler(); };
                return b;
            };
            controls.append(
                emptyButton("Add Image", "Create the first image box", () => addImageBtn?.click()),
                emptyButton("Add Text", "Create a text bridge box", () => addTextBtn?.click()),
                emptyButton("Add Audio", "Import audio into the board", () => addAudioBtn?.click())
            );
            emptyState.append(title, hint, controls);
            boxList.appendChild(emptyState);
        }
        orderedBoxSegments.forEach((seg, index) => {
            if (String(seg.type || "image") === "text") {
                boxList.appendChild(makeRelayBridgeCard(seg, index));
                return;
            }
            const card = document.createElement("div");
            card.dataset.iamccsV3BoxSegmentId = String(seg.id || "");
            card.style.cssText = [
                "position:relative",
                "display:grid",
                "grid-template-columns:32px minmax(0,1.45fr) minmax(320px,.95fr) minmax(104px,124px) 38px",
                "gap:10px",
                "align-items:center",
                "margin-bottom:6px",
                "padding:8px 9px",
                "min-height:96px",
                (selectedId === seg.id ? `border:2px solid ${purple.accent}` : `border:1px solid ${purple.borderSoft}`),
                "border-radius:6px",
                `background:${selectedId === seg.id ? "linear-gradient(180deg,#12384B,#061923)" : "linear-gradient(180deg,rgba(13,48,64,.98),rgba(6,25,35,.98))"}`,
                (selectedId === seg.id ? "box-shadow:0 0 0 2px rgba(141,231,255,.22),inset 0 1px 0 rgba(255,255,255,.06),0 4px 10px rgba(0,0,0,.16)" : "box-shadow:inset 0 1px 0 rgba(255,255,255,.06),0 4px 10px rgba(0,0,0,.16)"),
                "box-sizing:border-box",
                "width:100%",
                "min-width:0",
            ].join(";");
            const badge = document.createElement("button");
            badge.type = "button";
            badge.textContent = String(index + 1);
            badge.title = "Select this box";
            badge.style.cssText = `align-self:center;justify-self:center;display:flex;align-items:center;justify-content:center;width:26px;height:26px;border:1px solid ${purple.border};border-radius:999px;background:${purple.button};color:${purple.text};font-size:10px;font-weight:900;cursor:pointer;box-shadow:inset 0 1px 0 rgba(255,255,255,.10);`;
            badge.onclick = () => { selectedId = seg.id; draw(); };
            addPressPreview(badge, {
                pressedBg: purple.buttonPress,
                pressedColor: purple.valueText,
                pressedBorder: purple.accent,
            });
            const relayWrap = document.createElement("label");
            relayWrap.style.cssText = `display:grid;grid-template-columns:34px minmax(0,1fr);grid-template-rows:auto auto;column-gap:6px;row-gap:4px;align-items:center;color:${purple.muted};font-size:9px;font-weight:800;`;
            const relayLabel = document.createElement("span");
            relayLabel.textContent = "Relay";
            relayLabel.style.cssText = "grid-column:1;grid-row:1;text-align:center;align-self:end;";
            const relayToggle = document.createElement("input");
            relayToggle.type = "checkbox";
            relayToggle.checked = Boolean((String(seg.prompt || "").trim() && seg.relay_manual_off !== true && seg.promptrelay_manual_off !== true) || seg.dialogue_pin || seg.image_lock || seg.motion_boost);
            relayToggle.dataset.iamccsV3SegmentId = String(seg.id);
            relayToggle.dataset.iamccsV3Key = "use_prompt";
            relayToggle.title = "Use this box prompt as a local PromptRelay beat";
            relayToggle.style.cssText = `width:18px;height:18px;accent-color:${purple.accent};cursor:pointer;`;
            relayToggle.style.gridColumn = "1";
            relayToggle.style.gridRow = "2";
            relayToggle.style.justifySelf = "center";
            relayToggle.style.alignSelf = "start";
            relayToggle.onpointerdown = (event) => event.stopPropagation();
            relayToggle.onchange = () => {
                seg.use_prompt = relayToggle.checked;
                seg.relay_manual_off = !relayToggle.checked;
                seg.promptrelay_manual_off = !relayToggle.checked;
                syncSegmentRelayPeers(seg.id, Boolean(seg.use_prompt), relayToggle);
                writeTimeline({ force: true });
                draw();
            };
            const localRelayActive = Boolean((String(seg.prompt || "").trim() && seg.relay_manual_off !== true && seg.promptrelay_manual_off !== true) || seg.dialogue_pin || seg.image_lock || seg.motion_boost);
            const relayStatus = document.createElement("span");
            relayStatus.textContent = localRelayActive ? "LOCAL" : "OFF";
            relayStatus.title = "PromptRelay status for this box";
            relayStatus.style.cssText = `display:flex;align-items:center;justify-content:center;min-width:44px;padding:2px 5px;border-radius:999px;border:1px solid ${localRelayActive ? "rgba(141,231,255,.62)" : purple.borderSoft};background:${localRelayActive ? "rgba(0,0,0,.72)" : "rgba(0,0,0,.18)"};color:${localRelayActive ? purple.accent : purple.muted};font-size:7px;font-weight:900;line-height:1;white-space:nowrap;`;
            relayStatus.style.gridColumn = "2";
            relayStatus.style.gridRow = "1";
            relayStatus.style.alignSelf = "end";
            const dialogueHint = document.createElement("span");
            const looksDialogue = /["â€œâ€]|\bsays?\b|\bspeak\b|\bdialog/i.test(String(seg.prompt || ""));
            dialogueHint.textContent = seg.dialogue_pin ? "PIN" : looksDialogue ? "DIALOG" : "PROMPT";
            dialogueHint.title = seg.dialogue_pin ? "Dialogue pin is active for this box" : looksDialogue ? "This prompt appears to contain dialogue timing or spoken text" : "No obvious dialogue marker detected";
            dialogueHint.style.cssText = `display:flex;align-items:center;justify-content:center;min-width:44px;padding:2px 5px;border-radius:999px;border:1px solid ${looksDialogue || seg.dialogue_pin ? "rgba(233,196,106,.58)" : purple.borderSoft};background:${looksDialogue || seg.dialogue_pin ? "rgba(233,196,106,.16)" : "rgba(0,0,0,.10)"};color:${looksDialogue || seg.dialogue_pin ? purple.accent : purple.muted};font-size:7px;font-weight:900;line-height:1;white-space:nowrap;`;
            dialogueHint.style.gridColumn = "2";
            dialogueHint.style.gridRow = "2";
            dialogueHint.style.alignSelf = "start";
            relayWrap.append(relayLabel, relayToggle, relayStatus, dialogueHint);
            const actions = document.createElement("div");
            actions.style.cssText = "display:grid;gap:8px;align-self:start;min-width:0;width:38px;";
            const mini = (label, titleText, action) => {
                const b = document.createElement("button");
                b.type = "button";
                b.textContent = label;
                b.title = titleText;
                b.style.cssText = `width:34px;height:26px;border:1px solid ${purple.borderSoft};border-radius:4px;background:${purple.button};color:${purple.text};font-size:10px;font-weight:900;cursor:pointer;display:flex;align-items:center;justify-content:center;`;
                b.onpointerdown = (event) => { event.preventDefault(); event.stopPropagation(); };
                b.onmousedown = (event) => { event.preventDefault(); event.stopPropagation(); };
                b.onclick = (event) => { event.preventDefault(); action(); };
                return protectControlDrag(addPressPreview(b, {
                    pressedBg: purple.buttonPress,
                    pressedColor: purple.valueText,
                    pressedBorder: purple.accent,
                }));
            };
            actions.append(
                mini("D", "Duplicate", () => {
                    duplicateVisualSegment(seg);
                }),
                mini("X", "Delete", () => {
                    rippleDeleteVisualSegment(seg);
                }),
                mini("+", "Add Slot Relay inside this image slot", () => {
                    createSlotRelayInSegment(seg);
                })
            );
            actions.style.alignSelf = "center";
            actions.style.alignContent = "center";
            actions.style.gridTemplateRows = "26px 26px 26px";
            actions.querySelectorAll("button").forEach((button) => {
                button.style.width = "34px";
                button.style.height = "26px";
            });
            const groupSpine = document.createElement("div");
            groupSpine.style.cssText = [
                "position:absolute",
                "left:-28px",
                "top:9px",
                "bottom:-11px",
                "width:16px",
                "pointer-events:none",
                "border-left:4px solid rgba(112,236,149,.72)",
                "border-top:2px solid rgba(112,236,149,.48)",
                "border-bottom:2px solid rgba(112,236,149,.32)",
                "box-shadow:-2px 0 10px rgba(112,236,149,.34), inset 1px 0 0 rgba(255,255,255,.12)",
                "border-radius:7px 0 0 7px",
                "opacity:.92",
            ].join(";");
            const groupSpineKnee = document.createElement("div");
            groupSpineKnee.style.cssText = [
                "position:absolute",
                "left:-28px",
                "top:50%",
                "width:26px",
                "height:2px",
                "pointer-events:none",
                "background:linear-gradient(90deg,rgba(112,236,149,.74),rgba(112,236,149,.16))",
                "box-shadow:0 0 8px rgba(112,236,149,.30)",
            ].join(";");

            const leftPane = document.createElement("div");
            leftPane.style.cssText = "display:flex;align-items:center;min-width:0;align-self:center;width:100%;";
            const numericRow = document.createElement("div");
            numericRow.style.cssText = "display:grid;grid-template-columns:minmax(64px,.7fr) minmax(64px,.7fr) minmax(58px,.55fr) minmax(126px,1fr) minmax(128px,1fr);gap:7px;align-items:center;min-width:0;width:100%;";
            numericRow.append(
                makeField(seg, "Frame", "start", "number"),
                makeField(seg, "Len", "length", "number"),
                makeField(seg, "Ref", "ref", "number"),
                makeField(seg, "Motion", "guideStrength", "number"),
                makeSegmentSummary(seg, index, timeline.segments.length)
            );
            leftPane.append(numericRow);

            const rightPane = document.createElement("div");
            rightPane.style.cssText = "display:flex;align-items:center;min-width:0;align-self:center;padding-bottom:0;width:100%;";
            const relayMonitor = makeRelayGenerationMonitor(seg);
            relayWrap.style.alignSelf = "center";
            relayWrap.style.minHeight = "58px";
            relayWrap.style.padding = "5px 6px";
            relayWrap.style.border = `1px solid ${localRelayActive ? "rgba(141,231,255,.58)" : purple.borderSoft}`;
            relayWrap.style.borderRadius = "6px";
            relayWrap.style.background = localRelayActive ? "linear-gradient(180deg,rgba(0,0,0,.72),rgba(12,55,75,.42))" : "linear-gradient(180deg,rgba(234,248,255,.045),rgba(0,0,0,.14))";
            rightPane.append(relayMonitor);

            card.append(groupSpine, groupSpineKnee, badge, leftPane, rightPane, relayWrap, actions);
            boxList.appendChild(card);
            if (isTimelineImageSegment(seg)) {
                const win = generationWindowForImage(seg);
                const slotRelayList = Array.isArray(seg.slot_relays)
                    ? seg.slot_relays
                    .filter((relay) => relay && relay.slotRelay !== false)
                    .sort((a, b) => Number(a.start || 0) - Number(b.start || 0))
                    : [];
                const baseActive = Boolean(String(seg.prompt || "").trim() && seg.use_prompt !== false && String(seg.use_prompt).toLowerCase() !== "false" && seg.relay_manual_off !== true && seg.promptrelay_manual_off !== true);
                const baseLen = slotRelayList.length
                    ? Math.max(1, Math.max(win.start + 1, Math.round(Number(slotRelayList[0].start || win.start + 1))) - win.start)
                    : Math.max(1, win.length);
                const relayRows = [];
                if (baseActive) {
                    relayRows.push({
                        kind: "base",
                        label: "Base Relay",
                        relay: null,
                        relayIndex: 0,
                        localStart: 0,
                        length: baseLen,
                    });
                }
                slotRelayList.forEach((relay, relayIndex) => {
                    const startAbs = Math.max(win.start, Math.min(win.end - 1, Math.round(Number(relay.start || win.start))));
                    const nextStart = relayIndex + 1 < slotRelayList.length
                        ? Math.max(startAbs + 1, Math.min(win.end, Math.round(Number(slotRelayList[relayIndex + 1].start || win.end))))
                        : win.end;
                    relayRows.push({
                        kind: "slot",
                        label: `Slot Relay ${relayIndex + 1}`,
                        relay,
                        relayIndex: relayRows.length,
                        localStart: Math.max(0, startAbs - win.start),
                        length: Math.max(1, nextStart - startAbs),
                    });
                });
                relayRows.forEach((relayRow, relayRowIndex) => {
                        const relay = relayRow.relay;
                        const isBaseRelay = relayRow.kind === "base";
                        const localStart = relayRow.localStart;
                        const relayCard = document.createElement("div");
                        relayCard.dataset.iamccsV3BoxSegmentId = String(`${seg.id || "image"}::${isBaseRelay ? "base" : relay.id || relayRowIndex}`);
                        relayCard.style.cssText = [
                            "position:relative",
                            "display:grid",
                            "grid-template-columns:32px 92px 92px minmax(320px,1fr) 34px",
                            "gap:8px",
                            "align-items:center",
                            "margin:-2px 0 7px 46px",
                            "padding:7px 9px",
                            "min-height:58px",
                            (isBaseRelay ? "border:1px solid rgba(124,214,255,.48)" : "border:1px solid rgba(137,238,164,.48)"),
                            "border-radius:6px",
                            (isBaseRelay ? "background:linear-gradient(180deg,rgba(12,38,48,.94),rgba(4,18,24,.94))" : "background:linear-gradient(180deg,rgba(18,47,36,.94),rgba(6,24,19,.94))"),
                            "box-shadow:inset 0 1px 0 rgba(255,255,255,.05),0 3px 8px rgba(0,0,0,.16)",
                            "box-sizing:border-box",
                        ].join(";");
                        const relaySpineKnee = document.createElement("div");
                        const relaySpineVertical = document.createElement("div");
                        relaySpineVertical.style.cssText = [
                            "position:absolute",
                            "left:-28px",
                            "top:-10px",
                            "bottom:-9px",
                            "width:16px",
                            "pointer-events:none",
                            `border-left:4px solid ${isBaseRelay ? "rgba(142,232,255,.66)" : "rgba(112,236,149,.70)"}`,
                            "box-shadow:-2px 0 9px rgba(112,236,149,.24)",
                        ].join(";");
                        relaySpineKnee.style.cssText = [
                            "position:absolute",
                            "left:-28px",
                            "top:50%",
                            "width:26px",
                            "height:2px",
                            "pointer-events:none",
                            `background:linear-gradient(90deg,${isBaseRelay ? "rgba(142,232,255,.72)" : "rgba(112,236,149,.74)"},rgba(112,236,149,.10))`,
                            "box-shadow:0 0 8px rgba(112,236,149,.28)",
                        ].join(";");
                        const badgeRelay = document.createElement("div");
                        badgeRelay.textContent = isBaseRelay ? "B" : "S";
                        badgeRelay.title = isBaseRelay ? "Base local prompt relay for this generation" : "Slot Relay prompt segment inside this image slot";
                        badgeRelay.style.cssText = `display:flex;align-items:center;justify-content:center;width:24px;height:24px;border-radius:999px;border:1px solid ${isBaseRelay ? "rgba(142,232,255,.78)" : "rgba(190,255,201,.78)"};background:${isBaseRelay ? "#DDF7FF" : "#DDF8D8"};color:#10130F;font:10px/1 monospace;font-weight:900;`;
                        const startBox = document.createElement("label");
                        startBox.style.cssText = `display:flex;flex-direction:column;gap:4px;color:${purple.muted};font-size:9px;font-weight:900;text-align:center;`;
                        const startLabel = document.createElement("span");
                        startLabel.textContent = "Gen frame";
                        const startInput = document.createElement("input");
                        startInput.type = "number";
                        startInput.value = String(localStart);
                        startInput.min = "0";
                        startInput.max = String(Math.max(0, win.length - 1));
                        startInput.style.cssText = inputBase() + `height:26px;background:${purple.valueBg};color:${purple.valueText};text-align:center;font-weight:900;`;
                        startInput.disabled = isBaseRelay;
                        if (isBaseRelay) startInput.title = "Base Relay always starts at generation frame 0.";
                        startInput.oninput = () => {
                            if (isBaseRelay) return;
                            const nextLocal = Math.max(0, Math.min(win.length - 1, Math.round(Number(startInput.value || 0))));
                            const absStart = win.start + nextLocal;
                            const oldEnd = Math.min(win.end, Math.round(Number(relay.start || absStart) + Number(relay.length || 1)));
                            relay.start = absStart;
                            relay.length = Math.max(1, oldEnd - absStart);
                            relay.parentSegmentId = String(seg.id || "");
                            relay.relay_kind = "slot";
                            relay.relayKind = "slot";
                            relay.slotRelay = true;
                            writeTimeline({ force: true });
                            draw();
                        };
                        protectControlDrag(startInput);
                        startBox.append(startLabel, startInput);
                        const lenBox = document.createElement("label");
                        lenBox.style.cssText = startBox.style.cssText;
                        const lenLabel = document.createElement("span");
                        lenLabel.textContent = "Len";
                        const lenInput = document.createElement("input");
                        lenInput.type = "number";
                        lenInput.value = String(Math.max(1, Math.round(Number(relayRow.length || relay?.length || 1))));
                        lenInput.min = "1";
                        lenInput.max = String(isBaseRelay ? Math.max(1, win.length) : Math.max(1, win.end - Math.round(Number(relay.start || win.start))));
                        lenInput.style.cssText = inputBase() + `height:26px;background:${purple.valueBg};color:${purple.valueText};text-align:center;font-weight:900;`;
                        lenInput.disabled = isBaseRelay;
                        if (isBaseRelay) lenInput.title = "Base Relay length is derived from the next relay start, or the full generation when it is the only relay.";
                        lenInput.oninput = () => {
                            if (isBaseRelay) return;
                            const startAbs = Math.max(win.start, Math.min(win.end - 1, Math.round(Number(relay.start || win.start))));
                            relay.length = Math.max(1, Math.min(win.end - startAbs, Math.round(Number(lenInput.value || 1))));
                            relay.parentSegmentId = String(seg.id || "");
                            relay.relay_kind = "slot";
                            relay.relayKind = "slot";
                            relay.slotRelay = true;
                            writeTimeline({ force: true });
                            draw();
                        };
                        protectControlDrag(lenInput);
                        lenBox.append(lenLabel, lenInput);
                        const promptWrap = document.createElement("label");
                        promptWrap.style.cssText = `display:flex;flex-direction:column;gap:4px;color:${purple.muted};font-size:9px;font-weight:900;text-align:left;min-width:0;`;
                        const promptLabel = document.createElement("span");
                        promptLabel.textContent = `${String(relayRowIndex + 1).padStart(2, "0")} ${isBaseRelay ? "Base Relay" : `Slot Relay ${slotRelayList.indexOf(relay) + 1}`} prompt`;
                        const promptInput = document.createElement("textarea");
                        promptInput.value = String(isBaseRelay ? seg.prompt || "" : relay.prompt || "");
                        promptInput.placeholder = isBaseRelay ? "Base local PromptRelay segment..." : "Prompt for this Slot Relay segment...";
                        promptInput.style.cssText = inputBase() + `height:38px;min-height:38px;background:${purple.valueBg};color:${purple.valueText};font:${promptFontSize(10)}/1.22 monospace;font-weight:700;text-align:left;resize:vertical;`;
                        promptInput.oninput = () => {
                            if (isBaseRelay) {
                                seg.prompt = promptInput.value;
                                seg.note = promptInput.value;
                                seg.use_prompt = Boolean(String(promptInput.value || "").trim());
                                seg.relay_manual_off = !Boolean(String(promptInput.value || "").trim());
                                seg.promptrelay_manual_off = !Boolean(String(promptInput.value || "").trim());
                            } else {
                                relay.prompt = promptInput.value;
                                relay.note = promptInput.value;
                                relay.use_prompt = Boolean(String(promptInput.value || "").trim());
                                relay.parentSegmentId = String(seg.id || "");
                                relay.relay_kind = "slot";
                                relay.relayKind = "slot";
                                relay.slotRelay = true;
                            }
                            writeTimeline({ force: true });
                        };
                        protectControlDrag(promptInput);
                        promptWrap.append(promptLabel, promptInput);
                        const removeRelay = document.createElement("button");
                        removeRelay.type = "button";
                        removeRelay.textContent = isBaseRelay ? "•" : "X";
                        removeRelay.title = isBaseRelay ? "Base Relay cannot be deleted; clear the prompt or turn Relay off on the main card." : "Delete this Slot Relay";
                        removeRelay.style.cssText = `width:30px;height:26px;border:1px solid ${isBaseRelay ? "rgba(124,214,255,.42)" : purple.danger};border-radius:4px;background:${isBaseRelay ? "rgba(0,0,0,.28)" : "#6B302A"};color:${isBaseRelay ? "#9DEBFF" : "#FFF2E4"};font-size:10px;font-weight:900;cursor:${isBaseRelay ? "default" : "pointer"};`;
                        removeRelay.onclick = (event) => {
                            event.preventDefault();
                            event.stopPropagation();
                            if (isBaseRelay) return;
                            const parentId = String(seg.id || "");
                            const relayId = String(relay?.id || "");
                            if (deleteNestedSlotRelay(parentId, relayId)) {
                                writeTimeline({ force: true });
                                draw();
                            }
                        };
                        relayCard.append(relaySpineVertical, relaySpineKnee, badgeRelay, startBox, lenBox, promptWrap, removeRelay);
                        boxList.appendChild(relayCard);
                    });
            }
        });

        const audioAnchorSegmentId = (audio) => {
            if (!audio) return "";
            const linked = String(audio.linkedVisualId || "");
            if (linked && orderedBoxSegments.some((seg) => String(seg.id || "") === linked)) return linked;
            const start = Math.max(0, Math.round(Number(audio.start || 0)));
            const end = start + Math.max(1, Math.round(Number(audio.length || 1)));
            const overlapped = orderedBoxSegments.find((seg) => {
                const segStart = Math.max(0, Math.round(Number(seg.start || 0)));
                const segEnd = segStart + Math.max(1, Math.round(Number(seg.length || 1)));
                return start < segEnd && segStart < end;
            });
            if (overlapped) return String(overlapped.id || "");
            const previous = orderedBoxSegments.slice().reverse().find((seg) => Number(seg.start || 0) <= start);
            return String((previous || orderedBoxSegments[0] || {})?.id || "");
        };

        const audioCardsToShow = (timeline.audioSegments || [])
            .filter((seg) => audioSegmentHasMedia(seg) && !seg.placeholder)
            .slice()
            .sort((a, b) => (Number(a.start || 0) - Number(b.start || 0)) || (Number(a.track || 0) - Number(b.track || 0)));
        audioCardsToShow.forEach((selectedAudio) => {
            const audioCard = document.createElement("div");
            audioCard.style.cssText = [
                "display:grid",
                "grid-template-columns:150px 82px 92px 92px 84px 84px 84px 82px minmax(280px,1.15fr) minmax(190px,.85fr) 32px",
                "gap:10px",
                "align-items:end",
                "margin:12px 0 10px 42px",
                "padding:10px",
                `border:1px solid ${purple.audio}`,
                "border-radius:6px",
                "background:linear-gradient(180deg,rgba(57,44,30,.92),rgba(35,34,32,.92))",
                "box-sizing:border-box",
            ].join(";");
            const audioField = (labelText, key, type = "input", options = []) => {
                const wrap = document.createElement("label");
                wrap.style.cssText = `display:flex;flex-direction:column;gap:4px;color:${purple.muted};font-size:9px;font-weight:900;text-align:center;min-width:0;`;
                const span = document.createElement("span");
                span.textContent = labelText;
                span.style.cssText = "line-height:1.1;";
                if (type === "number") {
                    const step = key === "gain" ? "0.05" : "1";
                    const max = key === "gain" ? "2" : null;
                    const ctrl = numberStepperControl(selectedAudio[key] ?? (key === "gain" ? 1 : 0), step, "0", max, (value) => {
                        const raw = Number(value || 0);
                        selectedAudio[key] = key === "gain" ? Math.max(0, Math.min(2, raw)) : Math.max(0, Math.round(raw));
                        if (key === "start" || key === "length" || key === "track") {
                            selectedAudio.length = Math.max(1, Math.round(Number(selectedAudio.length || 1)));
                            selectedAudio.start = Math.max(0, Math.round(Number(selectedAudio.start || 0)));
                            selectedAudio.track = Math.max(0, Math.round(Number(selectedAudio.track || 0)));
                        }
                        writeTimeline({ force: true });
                        if (key === "start" || key === "length" || key === "track") draw();
                    }, { liveInput: false });
                    ctrl.style.minWidth = "0";
                    ctrl.style.width = "100%";
                    ctrl.style.gridTemplateColumns = "22px minmax(36px,1fr) 22px";
                    ctrl.style.gap = "4px";
                    ctrl.querySelectorAll("button").forEach((button) => {
                        button.style.width = "22px";
                        button.style.minWidth = "22px";
                        button.style.height = "24px";
                        button.style.margin = "0";
                    });
                    const input = ctrl.querySelector("input");
                    if (input) input.style.height = "24px";
                    styleValueControls(ctrl);
                    wrap.append(span, ctrl);
                    return wrap;
                }
                let input;
                if (type === "select") {
                    input = document.createElement("select");
                    options.forEach(([value, text]) => {
                        const opt = document.createElement("option");
                        opt.value = value;
                        opt.textContent = text;
                        input.appendChild(opt);
                    });
                    input.value = String(selectedAudio[key] || options[0]?.[0] || "");
                } else {
                    input = document.createElement("input");
                    input.type = type === "number" ? "number" : "text";
                    if (type === "number") input.step = key === "gain" ? "0.05" : "1";
                    input.value = selectedAudio[key] ?? "";
                }
                input.style.cssText = inputBase() + `height:28px;background:${purple.valueBg};border-color:${purple.border};color:${purple.valueText};font-weight:800;text-align:center;`;
                if (key === "linkedVisualId") {
                    wrap.style.width = "100%";
                    input.style.width = "100%";
                }
                input.onpointerdown = (event) => event.stopPropagation();
                input.onchange = input.oninput = () => {
                    if (type === "number") {
                        const raw = Number(input.value || 0);
                        selectedAudio[key] = key === "gain" ? Math.max(0, Math.min(2, raw)) : Math.max(0, Math.round(raw));
                    } else {
                        selectedAudio[key] = input.value;
                    }
                    if (key === "linkedVisualId") selectedId = selectedAudio.id;
                    if (key === "start" || key === "length" || key === "track") {
                        selectedAudio.length = Math.max(1, Math.round(Number(selectedAudio.length || 1)));
                        selectedAudio.start = Math.max(0, Math.round(Number(selectedAudio.start || 0)));
                        selectedAudio.track = Math.max(0, Math.round(Number(selectedAudio.track || 0)));
                    }
                    writeTimeline({ force: key === "linkedVisualId" });
                    if (key === "start" || key === "length" || key === "track" || key === "linkedVisualId") draw();
                };
                protectControlDrag(input);
                wrap.append(span, input);
                return wrap;
            };
            const audioToggle = (labelText, key) => {
                const wrap = document.createElement("label");
                wrap.style.cssText = `display:flex;flex-direction:column;gap:5px;color:${purple.muted};font-size:9px;font-weight:900;text-align:center;min-width:0;align-items:center;`;
                const span = document.createElement("span");
                span.textContent = labelText;
                span.style.cssText = "line-height:1.1;";
                const input = document.createElement("input");
                input.type = "checkbox";
                input.checked = Boolean(selectedAudio[key]);
                input.style.cssText = `width:18px;height:18px;accent-color:${purple.accent};cursor:pointer;`;
                input.onpointerdown = (event) => event.stopPropagation();
                input.onchange = () => {
                    selectedAudio[key] = Boolean(input.checked);
                    writeTimeline({ force: true });
                    draw();
                };
                protectControlDrag(input);
                wrap.append(span, input);
                return wrap;
            };
            const linkOptions = [["", "Auto / none"]];
            (timeline.segments || [])
                .filter((seg) => String(seg.type || "image") !== "audio" && !seg.placeholder)
                .forEach((seg, idx) => linkOptions.push([String(seg.id), `${idx + 1}. ${String(seg.label || `shot_${idx + 1}`)}`]));
            const remove = document.createElement("button");
            remove.type = "button";
            remove.textContent = "X";
            remove.title = "Remove selected audio clip";
            remove.style.cssText = `height:30px;border:1px solid ${purple.danger};border-radius:5px;background:#6B302A;color:${purple.text};font-size:10px;font-weight:900;cursor:pointer;`;
            remove.onpointerdown = (event) => { event.preventDefault(); event.stopPropagation(); };
            remove.onclick = () => {
                timeline.audioSegments = (timeline.audioSegments || []).filter((seg) => seg.id !== selectedAudio.id);
                selectedId = timeline.segments[0]?.id || null;
                writeTimeline({ force: true });
                draw();
            };
            const makeAudioHelper = () => {
                const fps = getFps();
                const peaks = Array.isArray(selectedAudio.waveformPeaks) ? selectedAudio.waveformPeaks.map((item) => Math.abs(Number(item) || 0)) : [];
                const rawPeak = peaks.length ? Math.max(...peaks) : 0;
                const gain = Math.max(0, Math.min(2, Number(selectedAudio.gain ?? selectedAudio.volume ?? 1) || 1));
                const levelGain = selectedAudio.normalizeAudio && rawPeak > 0.0001 ? Math.min(4, 0.92 / rawPeak) : 1;
                const masterGain = Math.max(0, Math.min(2, Number(timeline.masterAudioGain ?? 1) || 1));
                const masterLevelGain = timeline.masterAudioNormalize && rawPeak > 0.0001 ? Math.min(4, 0.92 / rawPeak) : 1;
                const localFrame = Math.round(Number(playFrame || 0) - Number(selectedAudio.start || 0) + Number(selectedAudio.trimStart || 0));
                const peakIndex = peaks.length && Number(selectedAudio.audioDurationFrames || 0) > 0
                    ? Math.max(0, Math.min(peaks.length - 1, Math.floor((localFrame / Math.max(1, Number(selectedAudio.audioDurationFrames || selectedAudio.length || 1))) * peaks.length)))
                    : -1;
                const livePeak = peakIndex >= 0 && playFrame >= Number(selectedAudio.start || 0) && playFrame <= Number(selectedAudio.start || 0) + Number(selectedAudio.length || 1)
                    ? peaks[peakIndex]
                    : 0;
                const finalPeak = Math.max(0, Math.min(1, livePeak * gain * levelGain * masterGain * masterLevelGain));
                const seconds = Math.max(0, Number(selectedAudio.length || 0) / fps);
                const trimSeconds = Math.max(0, Number(selectedAudio.trimStart || 0) / fps);
                const panel = document.createElement("div");
                panel.style.cssText = [
                    "display:grid",
                    "grid-template-columns:42px 1fr",
                    "gap:3px 6px",
                    "align-items:center",
                    "padding:5px 6px",
                    `border:1px solid ${purple.borderSoft}`,
                    "border-radius:6px",
                    "background:linear-gradient(180deg,rgba(0,0,0,.18),rgba(255,255,255,.025))",
                    "color:#EDE3D0",
                    "font:8px/1.15 monospace",
                    "font-weight:900",
                    "box-sizing:border-box",
                    "width:100%",
                    "min-width:190px",
                    "min-height:44px",
                ].join(";");
                const line = (label, value) => {
                    const l = document.createElement("span");
                    l.textContent = label;
                    l.style.cssText = `color:${purple.muted};text-transform:uppercase;white-space:nowrap;`;
                    const v = document.createElement("span");
                    v.textContent = value;
                    v.style.cssText = "text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
                    panel.append(l, v);
                };
                const meter = (label, value, color) => {
                    const l = document.createElement("span");
                    l.textContent = label;
                    l.style.cssText = `color:${purple.muted};text-transform:uppercase;white-space:nowrap;`;
                    const shell = document.createElement("div");
                    shell.style.cssText = "height:8px;border:1px solid rgba(255,255,255,.18);border-radius:999px;background:rgba(0,0,0,.35);overflow:hidden;";
                    const fill = document.createElement("div");
                    fill.style.cssText = `width:${Math.round(Math.max(0, Math.min(1, value)) * 100)}%;height:100%;background:${color};box-shadow:0 0 8px rgba(244,212,158,.28);`;
                    shell.appendChild(fill);
                    panel.append(l, shell);
                };
                line("Len", `${seconds.toFixed(2)}s`);
                line("Trim", `${trimSeconds.toFixed(2)}s`);
                meter("Live", livePeak, "linear-gradient(90deg,#5EB9B4,#F4D49E)");
                meter("Out", finalPeak, "linear-gradient(90deg,#6FCF8D,#F4D49E,#D8792B)");
                return panel;
            };
            audioCard.append(
                audioField("Audio purpose", "purpose", "select", [["music", "Music / score"], ["dialogue", "Dialogue / lipsync"], ["sfx", "SFX"], ["lipsync_or_music", "Auto"]]),
                audioField("Track", "track", "number"),
                audioField("Start frame", "start", "number"),
                audioField("Len frames", "length", "number"),
                audioField("Volume", "gain", "number"),
                audioField("Fade in f", "fadeInFrames", "number"),
                audioField("Fade out f", "fadeOutFrames", "number"),
                audioToggle("Level", "normalizeAudio"),
                audioField("Linked shot", "linkedVisualId", "select", linkOptions),
                makeAudioHelper(),
                remove
            );
            const title = document.createElement("div");
            title.textContent = `Selected audio: ${String(selectedAudio.name || selectedAudio.fileName || selectedAudio.audioFile || "clip")}`;
            title.style.cssText = `grid-column:1 / -1;color:#F4D49E;font-size:10px;font-weight:900;text-transform:uppercase;letter-spacing:.02em;`;
            audioCard.prepend(title);
            const anchorId = audioAnchorSegmentId(selectedAudio);
            const anchorCard = Array.from(boxList.children).find((item) => String(item?.dataset?.iamccsV3BoxSegmentId || "") === anchorId);
            if (anchorCard?.nextSibling) boxList.insertBefore(audioCard, anchorCard.nextSibling);
            else boxList.appendChild(audioCard);
        });
    }

    function draw() {
        if (!dragState && migrateSlotRelaySegmentsToNested()) writeTimeline({ force: true });
        if (!dragState) writeTimeline();
        drawFrameRuler();
        drawRuler();
        updatePlayUI();
        drawAudioPlaybarControls();
        timelineMeterSeconds = clampTimelineMeterSeconds(timelineMeterSeconds);
        promptSizeReadout.textContent = `${Math.round(promptTextScale * 100)}%`;
        frameMeterReadout.textContent = `${getTotalFrames()}f @ ${getFps()}fps`;
        secondsMeterReadout.textContent = `${getDuration().toFixed(3)}s`;
        const timelineZoomSteps = Math.round((timelineMeterSeconds - Math.max(0.5, getDuration())) * 2);
        timelineMeterReadout.textContent = `${Math.round(Math.max(0.2, Math.min(8, Math.pow(1.18, timelineZoomSteps))) * 100)}%`;
        promptArea.style.fontSize = promptFontSize(12);
        timelineViewport.style.display = "block";
        playbar.style.display = "flex";
        imageTrack.innerHTML = "";
        actionTrack.innerHTML = "";
        audioTracks.innerHTML = "";
        const visualSegments = activeVisualSegments();
        const audioSegmentsForDraw = visibleAudioSegments();
        const trackHasRealAudio = (trackIndex) => audioSegmentsForDraw.some((seg) =>
            Number(seg?.track || 0) === Number(trackIndex || 0) && audioSegmentHasMedia(seg) && !seg.placeholder
        );
        const canvasWidth = computeTimelineCanvasWidth(visualSegments);
        const previousScrollLeft = Number(timelineViewport.scrollLeft || 0);
        timelineCanvas.style.width = `${canvasWidth}px`;
        frameRuler.style.width = "100%";
        ruler.style.width = "100%";
        timelineBox.style.width = "100%";
        timelineViewport.scrollLeft = Math.min(previousScrollLeft, Math.max(0, canvasWidth - Number(timelineViewport.clientWidth || 0)));
        // Read user timeline-height resize value, update all related heights
        // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        timelineExtraH = Math.max(0, Math.min(600, Math.round(Number(node.properties?.iamccs_v3_timeline_extra_height || 0))));
        const _atCount = Math.max(1, Number(timeline.audioTrackCount || 1));
        timelineBox.style.height = `${264 + _atCount * 90 + timelineExtraH}px`;
        audioTracks.style.height = `${_atCount * 90}px`;
        imageTrack.style.height = `${254 + timelineExtraH}px`;
        audioTracks.style.top = `${254 + timelineExtraH}px`;
        for (let i = 0; i < Math.max(1, Number(timeline.audioTrackCount || 1)); i += 1) {
            const lane = document.createElement("div");
            lane.style.cssText = `position:absolute;left:0;right:0;top:${i * 90}px;height:90px;border-bottom:1px solid ${purple.borderSoft};background:${i % 2 ? "rgba(255,255,255,.025)" : "transparent"};`;
            const plus = document.createElement("button");
            plus.type = "button";
            plus.innerHTML = `<span>+</span><b>Audio</b>`;
            plus.title = "Import audio into this track";
            plus.style.cssText = [
                "position:sticky",
                "left:8px",
                "top:27px",
                "width:118px",
                "height:35px",
                "box-sizing:border-box",
                `border:1px dashed ${purple.border}`,
                "border-radius:6px",
                "background:linear-gradient(180deg,rgba(244,239,231,.92),rgba(213,204,190,.92))",
                `color:${purple.valueText}`,
                "display:grid",
                "grid-template-columns:28px 1fr",
                "gap:5px",
                "align-items:center",
                "justify-items:center",
                "padding:0 9px",
                "font-size:11px",
                "font-weight:900",
                "line-height:1",
                "cursor:pointer",
                "opacity:.94",
                "z-index:12",
                "box-shadow:0 3px 9px rgba(0,0,0,.34), inset 0 1px 0 rgba(255,255,255,.72)",
            ].join(";");
            const plusIcon = plus.querySelector("span");
            if (plusIcon) plusIcon.style.cssText = "display:flex;align-items:center;justify-content:center;width:22px;height:22px;border-radius:999px;background:#1B1713;color:#F4EFE7;font-size:18px;font-weight:900;";
            const plusText = plus.querySelector("b");
            if (plusText) plusText.style.cssText = "display:block;justify-self:start;text-transform:uppercase;font-size:10px;letter-spacing:0;color:#181512;";
            bindReliableTimelineButton(plus, () => {
                pendingAudioTrack = i;
                pendingAudioInsertFrame = 0;
                audioInput.click();
            });
            if (!trackHasRealAudio(i)) lane.appendChild(plus);
            audioTracks.appendChild(lane);
        }
        visualSegments.forEach((seg) => imageTrack.appendChild(makeBlock(seg, false)));
        if (!visualSegments.length) imageTrack.appendChild(makeImagePlaceholderBlock());
        drawVisualEdgeHandles(visualSegments);
        audioSegmentsForDraw.forEach((seg) => audioTracks.appendChild(makeBlock(seg, true)));
        timelineBox.querySelectorAll(".iamccs-v3-playhead").forEach((item) => item.remove());
        const playhead = document.createElement("div");
        playhead.className = "iamccs-v3-playhead";
        playhead.style.cssText = `position:absolute;top:0;bottom:0;left:${(playFrame / Math.max(1, getTotalFrames())) * 100}%;width:2px;background:${purple.play};box-shadow:0 0 0 1px rgba(0,0,0,.55),0 0 12px rgba(214,182,93,.7);pointer-events:none;z-index:30;`;
        timelineBox.appendChild(playhead);
        promptWrap.style.display = "block"; // always visible — global prompt shown in collapsed mode too — By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        settings.style.display = collapsed ? "none" : "grid";
        timelineNotice.style.display = collapsed ? "none" : timelineNotice.style.display;
        inspector.style.display = collapsed ? "none" : "block";
        refsPanel.style.display = "none";
        collapseBtn.textContent = collapsed ? "Show Boxes" : "Collapse Boxes";
        // In fullscreen editor mode, let root grow freely so panel can scroll vertically
        // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        if (root._iamccsFullscreenState) {
            root.style.maxHeight = "none";
            root.style.height = "auto";
            root.style.overflow = "visible";
        } else {
            root.style.maxHeight = `${currentNodeHeight() + timelineExtraH}px`;
        }
        boxList.style.maxHeight = collapsed ? "0" : "none";
        boxList.style.overflow = collapsed ? "hidden" : "visible";
        if (!collapsed && !dragState) drawBoxes();
        const desiredHeight = currentNodeHeight() + timelineExtraH;
        node._iamccsCineMinSize = [SHOTBOARD_V3_RIGID_WIDTH, desiredHeight];
        if (!root._iamccsFullscreenState) {
            if (Math.abs(Number(node.size?.[1] || 0) - desiredHeight) > 18) {
                if (typeof node.setSize === "function") node.setSize([SHOTBOARD_V3_RIGID_WIDTH, desiredHeight]);
                else node.size = [SHOTBOARD_V3_RIGID_WIDTH, desiredHeight];
            }
        }
        try { node.setDirtyCanvas?.(true, true); app.graph?.setDirtyCanvas?.(true, true); } catch {}
    }

    function spreadVisualSegmentsAcrossDuration() {
        const total = getTotalFrames();
        const sorted = (timeline.segments || []).slice().sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
        if (!sorted.length) return;
        const base = Math.max(1, Math.floor(total / sorted.length));
        let cursor = 0;
        sorted.forEach((seg, index) => {
            seg.start = cursor;
            seg.length = index === sorted.length - 1 ? Math.max(1, total - cursor) : base;
            cursor += seg.length;
        });
        timeline.segments = sorted;
    }

    function isSingleEmptyDefaultVisualSegment() {
        if (refPaths().length || timeline.segments.length !== 1) return false;
        const seg = timeline.segments[0] || {};
        return String(seg.type || "image") === "image"
            && Number(seg.ref || 1) === 1
            && String(seg.label || "") === "ref_1"
            && !String(seg.prompt || "").trim()
            && !String(seg.note || "").trim();
    }

    function addUploadedImagesToTimeline(uploaded) {
        if (!uploaded.length) return;
        const current = getOwnReferencePaths(node);
        const nextPaths = current.concat(uploaded);
        setOwnReferencePaths(node, nextPaths);
        if (isSingleEmptyDefaultVisualSegment()) timeline.segments = [];
        const targetId = pendingImageTargetId;
        pendingImageTargetId = null;
        pendingImageInsertFrame = null;
        let cursor = endOfSegments(timeline.segments);
        let refOffset = 0;
        if (targetId) {
        const target = (timeline.segments || []).find((seg) => seg.id === targetId);
        if (target) {
            const oldTargetRef = Math.max(0, Math.round(Number(target.ref || 0)));
            const oldTargetPath = segmentReferencePath(target);
            target.type = "image";
            target.placeholder = false;
            target.ref = current.length + 1;
            // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
            // Bug #5 fix: clear all old truth/path fields before assigning the new image,
            // so imported-board pins or stale values never shadow the replacement.
            delete target.imageTruthPath;
            delete target.image_truth_path;
            delete target.imageTruthRef;
            delete target.imageTruthPinned;
            target.imageFile = uploaded[0] || target.imageFile || "";
            target.path = target.imageFile;
            target.imageTruthPath = target.imageFile;
            target.imageTruthName = target.imageFile.split(/[\\/]/).pop() || target.imageFile;
            target.imageName = target.imageTruthName;
            target.imageTruthRef = target.ref;
            target.imageTruthPinned = true;
            target.imageTruthSource = "upload_replace_slot";
            delete target.image_file;
            delete target.image_truth_path;
            console.info("[IAMCCS WAN PURE REF SYNC] uploaded replacement applied to selected timeline slot only", {
                segmentId: target.id,
                oldRef: oldTargetRef,
                oldPath: oldTargetPath,
                newRef: target.ref,
                newPath: target.imageFile,
            });
            refPreviewBusters.set(String(target.imageFile), String(Date.now()));
            target.label = `ref_${current.length + 1}`;
                target.use_guide = true;
                target.motion = clampGuideStrength(target.motion ?? target.guideStrength ?? defaultForceWidget?.value ?? 1);
                target.guideStrength = target.motion;
                target.imageLockStrength = target.motion;
                target.defaultForceSource = clampGuideStrength(target.defaultForceSource ?? target.motion);
                uploaded = uploaded.slice(1);
                refOffset = 1;
                cursor = endOfSegments(timeline.segments);
            }
        }
        const requiredEnd = cursor + (uploaded.length * defaultLen());
        ensureDurationForFrames(requiredEnd);
        uploaded.forEach((_, i) => {
            const imageFile = uploaded[i];
            const room = Math.max(1, getTotalFrames() - Math.min(cursor, getTotalFrames() - 1));
            const length = Math.min(defaultLen(), room);
            timeline.segments.push({
                id: newId("seg"),
                type: "image",
                start: Math.min(cursor, Math.max(0, getTotalFrames() - 1)),
                length,
                ref: current.length + refOffset + i + 1,
                imageFile,
                path: imageFile,
                imageTruthPath: imageFile,
                imageTruthName: imageFile.split(/[\\/]/).pop() || imageFile,
                imageName: imageFile.split(/[\\/]/).pop() || imageFile,
                imageTruthRef: current.length + refOffset + i + 1,
                imageTruthPinned: true,
                imageTruthSource: "upload_new_slot",
                label: `ref_${current.length + refOffset + i + 1}`,
                prompt: "",
                note: "",
                camera: "continuous dolly-in",
                transition: "continuous_motion",
                motion: clampGuideStrength(defaultForceWidget?.value || 1),
                guideStrength: clampGuideStrength(defaultForceWidget?.value || 1),
                imageLockStrength: clampGuideStrength(defaultForceWidget?.value || 1),
                defaultForceSource: clampGuideStrength(defaultForceWidget?.value || 1),
                forceCustom: false,
                use_guide: true,
            });
            cursor += length;
        });
        uploaded.forEach((path) => refPreviewBusters.set(String(path), String(Date.now())));
        if (endOfSegments(timeline.segments) > getTotalFrames() || timeline.segments.some((seg) => Number(seg.length || 0) <= 1)) {
            spreadVisualSegmentsAcrossDuration();
        }
        selectedId = timeline.segments[timeline.segments.length - 1]?.id || selectedId;
        writeTimeline();
        draw();
    }

    fileInput.onchange = async (event) => {
        const uploaded = await uploadShotboardImages(event.target.files);
        addUploadedImagesToTimeline(uploaded);
        fileInput.value = "";
    };
    addImageBtn.onclick = () => {
        openAppendImagePicker();
    };
    openEditorBtn.onclick = () => {
        toggleFullscreenEditor(root, node);
    };
    addTextBtn.onclick = () => {
        if (!refPaths().length && timeline.segments.length === 1 && String(timeline.segments[0]?.label || "") === "ref_1") {
            const seg = timeline.segments[0];
            seg.type = "text";
            seg.start = 0;
            seg.length = getTotalFrames();
            seg.label = "t2v_shot";
            seg.prompt = seg.prompt || "cinematic motion, coherent subject, stable camera";
            seg.camera = "cinematic motion";
            seg.transition = "continuous_motion";
            seg.guideStrength = 0;
            seg.imageLockStrength = 0;
            seg.use_guide = false;
            selectedId = seg.id;
            draw();
            return;
        }
        const start = Math.min(endOfSegments(timeline.segments), Math.max(0, getTotalFrames() - 1));
        const seg = { id: newId("text"), type: "text", start, length: Math.min(defaultLen(), Math.max(1, getTotalFrames() - start)), label: "text_beat", prompt: "camera continues the movement", camera: "cinematic motion", transition: "continuous_motion", guideStrength: 0, imageLockStrength: 0, use_guide: false };
        timeline.segments.push(seg);
        selectedId = seg.id;
        draw();
    };
    async function decodeAudioInfo(file) {
        const fallbackFrames = Math.min(defaultLen() * 2, getTotalFrames());
        try {
            const buffer = await file.arrayBuffer();
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            if (!AudioContextClass) return { durationFrames: fallbackFrames, peaks: [] };
            const audioCtx = new AudioContextClass();
            const audioBuffer = await audioCtx.decodeAudioData(buffer.slice(0));
            if (typeof audioCtx.close === "function") audioCtx.close().catch(() => {});
            const durationFrames = Math.max(1, Math.ceil(Number(audioBuffer.duration || 0) * getFps()));
            const channelData = audioBuffer.getChannelData(0);
            const peakCount = 200;
            const step = Math.max(1, Math.floor(channelData.length / peakCount));
            const peaks = [];
            for (let i = 0; i < peakCount; i++) {
                let peak = 0;
                for (let j = 0; j < step; j++) {
                    peak = Math.max(peak, Math.abs(channelData[(i * step) + j] || 0));
                }
                peaks.push(Number(peak.toFixed(4)));
            }
            return { durationFrames, peaks };
        } catch (err) {
            console.warn("[IAMCCS Cine Shotboard V3] audio decode failed", err);
            return { durationFrames: fallbackFrames, peaks: [] };
        }
    }

    function nearestVisualSegmentId(frame) {
        const visual = (timeline.segments || [])
            .filter((seg) => String(seg.type || "image") !== "audio" && !seg.placeholder)
            .slice()
            .sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
        if (!visual.length) return "";
        const f = Math.max(0, Math.round(Number(frame || 0)));
        let best = visual[0];
        let bestDistance = Infinity;
        for (const seg of visual) {
            const start = Number(seg.start || 0);
            const end = start + Number(seg.length || 1);
            const distance = f >= start && f <= end ? 0 : Math.min(Math.abs(f - start), Math.abs(f - end));
            if (distance < bestDistance) {
                best = seg;
                bestDistance = distance;
            }
        }
        return String(best?.id || "");
    }

    function visualSegmentIdByAudioIndex(frame, indexOffset = 0) {
        const visual = (timeline.segments || [])
            .filter((seg) => String(seg.type || "image") !== "audio" && !seg.placeholder)
            .slice()
            .sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
        if (!visual.length) return "";
        const nearestId = nearestVisualSegmentId(frame);
        const nearestIndex = Math.max(0, visual.findIndex((seg) => String(seg.id || "") === nearestId));
        const targetIndex = Math.max(0, Math.min(visual.length - 1, nearestIndex + Math.max(0, Math.round(Number(indexOffset || 0)))));
        return String(visual[targetIndex]?.id || nearestId || "");
    }

    async function uploadAudioFiles(files, targetFrameStart = null, targetTrack = 0) {
        const audioFiles = Array.from(files || []).filter((file) => String(file.type || "").startsWith("audio/"));
        const track = Math.max(0, Math.round(Number(targetTrack || 0)));
        let cursor = targetFrameStart == null
            ? endOfSegments((timeline.audioSegments || []).filter((item) => Number(item.track || 0) === track && audioSegmentHasMedia(item) && !item.placeholder))
            : Math.max(0, Math.round(Number(targetFrameStart || 0)));
        const existingAudioCount = (timeline.audioSegments || []).filter((item) => audioSegmentHasMedia(item) && !item.placeholder).length;
        for (const [fileIndex, file] of audioFiles.entries()) {
            try {
                const body = new FormData();
                body.append("image", file);
                const resp = await api.fetchApi("/upload/image", { method: "POST", body });
                if (!resp || resp.status !== 200) throw new Error(`upload failed: ${resp?.status || "no response"}`);
                const data = await resp.json();
                const filename = data?.name || file.name;
                const subfolder = data?.subfolder || "";
                const audioFile = subfolder ? `${subfolder}/${filename}` : filename;
                const info = await decodeAudioInfo(file);
                const room = Math.max(1, getTotalFrames() - cursor);
                const length = Math.min(Math.max(1, info.durationFrames), room);
                removeEmptyAudioPlaceholdersInRange(track, cursor, length);
                const seg = {
                    id: newId("aud"),
                    type: "audio",
                    start: cursor,
                    length,
                    track,
                    trimStart: 0,
                    audioDurationFrames: Math.max(1, info.durationFrames),
                    audioFile,
                    fileName: file.name,
                    name: file.name,
                    mime: file.type,
                    size: file.size,
                    waveformPeaks: info.peaks,
                    purpose: "lipsync_or_music",
                    gain: 1,
                    fadeInFrames: 0,
                    fadeOutFrames: 0,
                    normalizeAudio: false,
                    linkedVisualId: visualSegmentIdByAudioIndex(cursor, existingAudioCount + fileIndex),
                };
                timeline.audioSegments.push(seg);
                cursor += length;
            } catch (err) {
                console.error("[IAMCCS Cine Shotboard V3] audio upload failed", err);
            }
        }
        timeline.audioSegments.sort((a, b) => (Number(a.track || 0) - Number(b.track || 0)) || (Number(a.start || 0) - Number(b.start || 0)));
        writeTimeline({ force: true });
        draw();
    }

    audioInput.onchange = async (event) => {
        await uploadAudioFiles(event.target.files || [], pendingAudioInsertFrame, pendingAudioTrack);
        pendingAudioInsertFrame = null;
        pendingAudioTrack = 0;
        audioInput.value = "";
    };
    addAudioBtn.onclick = () => {
        pendingAudioInsertFrame = null;
        pendingAudioTrack = 0;
        audioInput.click();
    };
    addTrackBtn.onclick = () => { timeline.audioTrackCount = Math.max(1, Number(timeline.audioTrackCount || 1)) + 1; draw(); };
    timelineBox.ondblclick = (event) => {
        const rect = timelineBox.getBoundingClientRect();
        const frame = Math.round(((event.clientX - rect.left) / Math.max(1, rect.width)) * getTotalFrames());
        const y = event.clientY - rect.top;
        if (y >= 254) {
            pendingAudioInsertFrame = frame;
            pendingAudioTrack = Math.max(0, Math.min(Math.max(1, Number(timeline.audioTrackCount || 1)) - 1, Math.floor((y - 254) / 90)));
            audioInput.click();
        } else {
            openAppendImagePicker();
        }
    };
    scrub.oninput = () => {
        playFrame = Math.round(Number(scrub.value || 0));
        if (isPlaying) {
            playbackStartFrame = playFrame;
            playbackStartTimestamp = performance.now();
            scheduleAudioFromFrame(playFrame);
        }
        draw();
    };
    playBtn.onclick = () => {
        if (isPlaying) stopPlayback();
        else startPlayback();
    };
    loopBtn.onclick = () => {
        isLooping = !isLooping;
        updatePlayUI();
    };
    function rowsToSegments(importedRows) {
        const fps = getFps();
        const sorted = (Array.isArray(importedRows) ? importedRows : [])
            .filter((row) => row && typeof row === "object")
            .map((row, index) => ({
                row,
                index,
                frame: Number.isFinite(Number(row.frame))
                    ? Math.max(0, Math.round(Number(row.frame)))
                    : Math.max(0, Math.round(Number(row.second ?? row.seconds ?? row.time ?? 0) * fps)),
            }))
            .sort((a, b) => a.frame - b.frame || a.index - b.index);
        return sorted.map((item, index) => {
            const next = sorted[index + 1];
            const start = item.frame;
            const end = next ? next.frame : getTotalFrames();
            const row = item.row;
            const prompt = String(row.relay_prompt ?? row.local_prompt ?? row.prompt ?? "").trim();
            const isText = String(row.type || "").toLowerCase() === "text" || Number(row.ref ?? row.image_ref ?? row.reference_index ?? 1) <= 0;
            const rowTruthPath = isText ? "" : String(row.imageTruthPath || row.image_truth_path || row.imageFile || row.image_file || row.path || "").trim();
            const rowTruthName = isText ? "" : (String(row.imageTruthName || row.imageName || row.name || row.filename || "").trim() || rowTruthPath.split(/[\\/]/).pop() || "");
            const defaultForceValue = clampGuideStrength(defaultForceWidget?.value ?? 1);
            const rowForceValue = clampGuideStrength(row.motion ?? row.motion_force ?? row.force ?? row.strength ?? defaultForceValue);
            const rowGuideValue = clampGuideStrength(row.motion ?? row.guide_strength ?? row.guideStrength ?? row.strength ?? row.force ?? row.motion_force ?? row.image_lock_strength ?? row.imageLockStrength ?? rowForceValue);
            const explicitForceCustom = typeof row.forceCustom === "boolean"
                ? row.forceCustom
                : typeof row.force_custom === "boolean"
                    ? row.force_custom
                    : Number.isFinite(Number(row.force ?? row.strength)) && Math.abs(rowForceValue - defaultForceValue) > 0.0005;
            return {
                id: newId(isText ? "text" : "seg"),
                type: isText ? "text" : "image",
                start,
                length: Math.max(1, end - start),
                ref: Math.max(1, Math.round(Number(row.ref ?? row.image_ref ?? row.reference_index ?? index + 1) || index + 1)),
                imageFile: rowTruthPath,
                path: rowTruthPath,
                imageTruthPath: rowTruthPath,
                imageTruthName: rowTruthName,
                imageName: rowTruthName,
                imageTruthPinned: Boolean(rowTruthPath),
                imageTruthSource: rowTruthPath ? "row_import_truth" : "",
                imageTruthRef: Math.max(1, Math.round(Number(row.ref ?? row.image_ref ?? row.reference_index ?? index + 1) || index + 1)),
                label: String(row.label ?? row.shot_label ?? (rowTruthName ? rowTruthName.replace(/\.[^.]+$/, "") : `${isText ? "text" : "shot"}_${index + 1}`)),
                prompt,
                note: String(row.note ?? row.camera_note ?? ""),
                camera: String(row.camera ?? row.camera_move ?? "cinematic motion"),
                transition: String(row.transition ?? row.transition_intent ?? "continuous_motion"),
                motion: isText ? 0 : rowGuideValue,
                guideStrength: isText ? 0 : rowGuideValue,
                guide_strength: isText ? 0 : rowGuideValue,
                force: isText ? 0 : rowGuideValue,
                motion_force: isText ? 0 : rowForceValue,
                strength: isText ? 0 : rowGuideValue,
                imageLockStrength: isText ? 0 : rowGuideValue,
                image_lock_strength: isText ? 0 : rowGuideValue,
                linkGuideLock: Boolean(!isText),
                link_guide_lock: Boolean(!isText),
                defaultForceSource: isText ? 0 : (explicitForceCustom ? rowForceValue : defaultForceValue),
                forceCustom: Boolean(!isText && explicitForceCustom),
                use_guide: !isText && row.use_guide !== false,
                use_prompt: Boolean(prompt || row.dialogue_pin || row.dialoguePin || row.image_lock || row.imageLock || row.motion_boost || row.motionBoost),
                dialogue_pin: Boolean(row.dialogue_pin || row.dialoguePin),
                image_lock: Boolean(row.image_lock || row.imageLock),
                motion_boost: Boolean(row.motion_boost || row.motionBoost),
                clean_relay: Boolean(row.clean_relay || row.cleanRelay),
                step_transition_enabled: false,
                step_transition_type: "off",
                step_transition_prompt: "",
                step_transition_easing: String(row.step_transition_easing || row.stepTransitionEasing || "ease_in_out"),
                step_transition_force_curve: String(row.step_transition_force_curve || row.stepTransitionForceCurve || "late_target"),
                step_transition_duration: 0,
                step_transition_arrival: "auto",
                step_transition_auto_fit: true,
            };
        });
    }
    async function applyImportedBoard(data) {
        const workflowBoard = boardFromWorkflowJson(data);
        const nestedBoard = data?.board && typeof data.board === "object" ? data.board : null;
        const rootLooksLikeBoard = Boolean(
            data?.timeline ||
            data?.timeline_data ||
            data?.image_paths ||
            data?.package ||
            Array.isArray(data?.images) ||
            String(data?.metadata?.schema || data?.schema || "").includes("shotboard")
        );
        const board = (rootLooksLikeBoard ? (nestedBoard || data) : null) || workflowBoard || nestedBoard || data || {};
        if ((!Array.isArray(board.images) || !board.images.length) && Array.isArray(data?.images)) board.images = data.images;
        console.log("[IAMCCS V3 BOARD IMPORT] board source selected", {
            nodeId: node?.id,
            source: rootLooksLikeBoard ? (nestedBoard ? "nested_package_board" : "root_package_board") : (workflowBoard ? "workflow_node" : "root_fallback"),
            workflowNode: Boolean(workflowBoard),
            hasPackage: Boolean(board?.package || data?.package),
            images: Array.isArray(board?.images) ? board.images.length : 0,
            imagePaths: splitReferencePaths(board?.image_paths).length,
        });
        let importedDefaultForceExplicit = false;
        const validImportedValue = (value) => value !== undefined && value !== null && value !== "";
        const applyImportedSettings = (source, sourceLabel = "settings") => {
            if (!source || typeof source !== "object") return;
            for (const name of v3SettingNames) {
                if (!Object.prototype.hasOwnProperty.call(source, name) || !getWidget(node, name)) continue;
                const value = name === "image_resize_method" ? cineResizeMethodValue(source[name]) : source[name];
                if (!validImportedValue(value)) continue;
                setWidgetValue(node, name, value);
                if (name === "default_force") importedDefaultForceExplicit = true;
            }
            console.log("[IAMCCS V3 BOARD IMPORT] settings absorbed", {
                nodeId: node?.id,
                source: sourceLabel,
                explicitDefaultForce: importedDefaultForceExplicit,
                settings: v3SettingsSnapshot(),
            });
        };
        const inferDefaultForceFromTimeline = (source) => {
            const values = [];
            const defaultSourceValues = [];
            const add = (value, target = values) => {
                const n = Number(value);
                if (Number.isFinite(n) && n >= 0 && n <= 1) target.push(Math.round(n * 1000) / 1000);
            };
            const addSegment = (seg) => {
                if (!seg || typeof seg !== "object" || seg.placeholder) return;
                const type = String(seg.type || "image").toLowerCase();
                if (type === "audio" || type === "text") return;
                add(seg.defaultForceSource ?? seg.default_force_source, defaultSourceValues);
                add(seg.guideStrength ?? seg.guide_strength ?? seg.force ?? seg.strength);
            };
            const addRow = (row) => {
                if (!row || typeof row !== "object") return;
                add(row.default_force ?? row.defaultForce ?? row.defaultForceSource, defaultSourceValues);
                add(row.force ?? row.strength ?? row.guideStrength ?? row.guide_strength);
            };
            const visit = (item) => {
                if (!item || typeof item !== "object") return;
                if (Array.isArray(item.segments)) item.segments.forEach(addSegment);
                if (Array.isArray(item.rows)) item.rows.forEach(addRow);
                if (item.timeline && typeof item.timeline === "object") visit(item.timeline);
                if (typeof item.timeline_data === "string" && item.timeline_data.trim()) {
                    try { visit(JSON.parse(item.timeline_data)); } catch {}
                }
            };
            visit(source);
            const pickMode = (list) => {
                if (!list.length) return null;
                const counts = new Map();
                list.forEach((value) => {
                    const key = value.toFixed(3);
                    counts.set(key, (counts.get(key) || 0) + 1);
                });
                return Number([...counts.entries()].sort((a, b) => b[1] - a[1] || Number(b[0]) - Number(a[0]))[0][0]);
            };
            return pickMode(defaultSourceValues) ?? pickMode(values);
        };
        const settingsData = board.settings && typeof board.settings === "object" ? board.settings : {};
        const importedPrompt = board.global_prompt ?? board.prompt ?? board.globalPrompt;
        if (typeof importedPrompt === "string") {
            promptArea.value = importedPrompt;
            setWidgetValue(node, "global_prompt", importedPrompt);
        }
        applyImportedSettings(settingsData, "board.settings");
        applyImportedSettings(board, workflowBoard ? "workflow.shotboard.widgets" : "board.root");
        const paths = await packagedReferencePathsForImport(board, (message) => {
            showTimelineNotice(message, /failed/i.test(String(message || "")) ? "error" : "warn");
        });
        if (paths.length) setOwnReferencePaths(node, paths);
        else clearOwnReferencePaths(node);

        let loadedTimeline = null;
        let loadedTimelineSource = "";
        const timelineText = String(board.timeline_data || data?.timeline_data || "").trim();
        const timelineCandidates = [];
        if (board.timeline && typeof board.timeline === "object") {
            timelineCandidates.push({ source: "board.timeline", value: board.timeline });
        }
        if (timelineText) {
            try { timelineCandidates.push({ source: "board.timeline_data", value: JSON.parse(timelineText) }); } catch {}
        }
        if (data?.timeline && typeof data.timeline === "object" && data.timeline !== board.timeline) {
            timelineCandidates.push({ source: "root.timeline", value: data.timeline });
        }
        const scoreTimelineCandidate = (candidate) => {
            const value = candidate?.value;
            if (!value || typeof value !== "object") return -1;
            const segments = Array.isArray(value.segments) ? value.segments : [];
            const rows = Array.isArray(value.rows) ? value.rows : [];
            const prompts = segments.length
                ? segments.filter((seg) => String(seg?.prompt || "").trim()).length
                : rows.filter((row) => String(row?.relay_prompt ?? row?.local_prompt ?? row?.prompt ?? "").trim()).length;
            const sourceBonus = candidate.source === "board.timeline" || candidate.source === "root.timeline" ? 1000 : 0;
            return sourceBonus + segments.length * 20 + rows.length * 10 + prompts;
        };
        timelineCandidates.sort((a, b) => scoreTimelineCandidate(b) - scoreTimelineCandidate(a));
        if (timelineCandidates.length) {
            loadedTimeline = timelineCandidates[0].value;
            loadedTimelineSource = timelineCandidates[0].source;
            console.log("[IAMCCS V3 BOARD IMPORT]", {
                selected: loadedTimelineSource,
                candidates: timelineCandidates.map((candidate) => ({
                    source: candidate.source,
                    score: scoreTimelineCandidate(candidate),
                    segments: Array.isArray(candidate.value?.segments) ? candidate.value.segments.length : 0,
                    rows: Array.isArray(candidate.value?.rows) ? candidate.value.rows.length : 0,
                    firstPrompt: Array.isArray(candidate.value?.segments)
                        ? String(candidate.value.segments.find((seg) => String(seg?.prompt || "").trim())?.prompt || "").slice(0, 220)
                        : String((candidate.value?.rows || []).find((row) => String(row?.relay_prompt ?? row?.local_prompt ?? row?.prompt ?? "").trim())?.relay_prompt || "").slice(0, 220),
                })),
            });
            applyImportedSettings(loadedTimeline.settings, `${loadedTimelineSource}.settings`);
            applyImportedSettings(loadedTimeline, loadedTimelineSource);
        }
        if (!importedDefaultForceExplicit) {
            const inferredDefaultForce = inferDefaultForceFromTimeline(loadedTimeline || board);
            if (Number.isFinite(Number(inferredDefaultForce))) {
                setWidgetValue(node, "default_force", clampGuideStrength(inferredDefaultForce));
                showTimelineNotice(`Default motion inferred from imported board: ${Number(inferredDefaultForce).toFixed(3)}`, "warn");
                console.log("[IAMCCS V3 BOARD IMPORT] default_force inferred", {
                    nodeId: node?.id,
                    value: Number(inferredDefaultForce),
                    source: loadedTimelineSource || "board",
                });
            }
        }
        if (loadedTimeline && typeof loadedTimeline === "object" && Array.isArray(loadedTimeline.segments)) {
            const baseTimeline = cloneJsonData(loadedTimeline);
            let audioData = {};
            if (typeof baseTimeline.audio_data === "string" && baseTimeline.audio_data.trim()) {
                try { audioData = JSON.parse(baseTimeline.audio_data); } catch {}
            } else if (baseTimeline.audio_data && typeof baseTimeline.audio_data === "object") {
                audioData = baseTimeline.audio_data;
            }
            const importedAudioSegments = Array.isArray(baseTimeline.audioSegments)
                ? baseTimeline.audioSegments
                : Array.isArray(audioData.audioSegments)
                    ? audioData.audioSegments
                    : [];
            timeline = {
                ...baseTimeline,
                schema: "iamccs.cine.filmmaker_timeline",
                schema_version: Number(baseTimeline.schema_version || 1),
                segments: baseTimeline.segments.map((seg) => normalizeV3RelayOnlySegment({ ...seg, id: seg.id || newId(seg.type === "text" ? "text" : "seg") })),
                rows: Array.isArray(baseTimeline.rows) ? cloneJsonData(baseTimeline.rows) : [],
                audioSegments: importedAudioSegments.map((seg) => ({ ...seg, id: seg.id || newId("aud") })),
                audioTrackCount: Math.max(1, Number(baseTimeline.audioTrackCount || audioData.audioTrackCount || importedAudioSegments.length || 2)),
                audioSyncMode: String(baseTimeline.audioSyncMode || audioData.audioSyncMode || "timeline_audio"),
                generationStrategy: String(baseTimeline.generationStrategy || "single_timeline"),
                flfrealMode: String(baseTimeline.flfrealMode || baseTimeline.flfreal_mode || "iamccs_enhanced"),
                globalPromptOnly: Boolean(baseTimeline.globalPromptOnly ?? baseTimeline.global_prompt_only ?? baseTimeline.use_global_prompt_only ?? false),
                verboseLog: baseTimeline.verboseLog ?? baseTimeline.verbose_log ?? true,
                masterAudioGain: Number(baseTimeline.masterAudioGain ?? audioData.masterAudioGain ?? 1),
                masterAudioNormalize: Boolean(baseTimeline.masterAudioNormalize ?? audioData.masterAudioNormalize ?? false),
                multiGeneration: baseTimeline.multiGeneration && typeof baseTimeline.multiGeneration === "object" ? cloneJsonData(baseTimeline.multiGeneration) : {},
            };
        } else {
            const rows = Array.isArray(board.rows)
                ? board.rows
                : Array.isArray(loadedTimeline?.rows)
                    ? loadedTimeline.rows
                    : parseShotboardTimelineString(timelineText);
            const converted = rowsToSegments(rows);
            if (converted.length) {
                timeline = {
                    schema: "iamccs.cine.filmmaker_timeline",
                    schema_version: 1,
                    segments: converted,
                    audioSegments: [],
                    audioTrackCount: 1,
                    audioSyncMode: "timeline_audio",
                    generationStrategy: "single_timeline",
                };
            }
        }
        const importedDurationTruth = objectDurationTruth(timeline) ?? objectDurationTruth(loadedTimeline) ?? objectDurationTruth(board) ?? objectDurationTruth(settingsData) ?? objectDurationTruth(data);
        const importedFpsTruth = objectFpsTruth(timeline) ?? objectFpsTruth(loadedTimeline) ?? objectFpsTruth(board) ?? objectFpsTruth(settingsData) ?? objectFpsTruth(data);
        if (importedDurationTruth !== null) setDurationSeconds(importedDurationTruth, `import:${loadedTimelineSource || "board"}`);
        if (importedFpsTruth !== null) setFrameRateValue(importedFpsTruth, `import:${loadedTimelineSource || "board"}`);
        console.log("[IAMCCS V3 BOARD IMPORT] timing truth applied", {
            nodeId: node?.id,
            source: loadedTimelineSource || "board",
            duration_seconds: getDuration(),
            frame_rate: getFps(),
            timelineDuration: objectDurationTruth(timeline),
            timelineFps: objectFpsTruth(timeline),
        });
        showTimelineNotice(`Imported board timeline source: ${loadedTimelineSource || "rows/timeline_data"}`, "warn");
        selectedId = timeline.segments?.[0]?.id || null;
        clearV3BoardTransientState();
        writeTimeline({ force: true });
        draw();
    }
    importBoardBtn.onclick = () => boardInput.click();
    boardInput.onchange = async (event) => {
        const file = event.target.files?.[0];
        if (!file) return;
        try {
            await applyImportedBoard(await readJsonFile(file));
        } catch (err) {
            console.error("[IAMCCS Cine Shotboard V3] board import failed", err);
        } finally {
            boardInput.value = "";
        }
    };
    root.addEventListener("dragover", (event) => {
        if (!hasFileDrag(event)) return;
        event.preventDefault();
        event.stopPropagation();
        if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
    }, { capture: true });
    root.addEventListener("drop", async (event) => {
        const files = Array.from(event.dataTransfer?.files || []);
        const file = files.find((item) => /\.json$/i.test(item.name || ""));
        const audioFiles = files.filter((item) => String(item.type || "").startsWith("audio/"));
        const imageFiles = files.filter((item) => String(item.type || "").startsWith("image/"));
        if (!file && !audioFiles.length && !imageFiles.length) return;
        event.preventDefault();
        event.stopPropagation();
        if (file) {
            try {
                await applyImportedBoard(await readJsonFile(file));
            } catch (err) {
                console.error("[IAMCCS Cine Shotboard V3] board drop import failed", err);
            }
            return;
        }
        if (imageFiles.length) {
            addUploadedImagesToTimeline(await uploadShotboardImages(imageFiles));
            return;
        }
        if (audioFiles.length) {
            await uploadAudioFiles(audioFiles);
            return;
        }
    }, { capture: true });
    collapseBtn.onclick = () => {
        collapsed = !collapsed;
        node.properties = node.properties || {};
        node.properties.iamccs_v3_collapsed = collapsed;
        const nextHeight = currentNodeHeight();
        node._iamccsCineMinSize = [SHOTBOARD_V3_RIGID_WIDTH, nextHeight];
        if (typeof node.setSize === "function") node.setSize([SHOTBOARD_V3_RIGID_WIDTH, nextHeight]);
        draw();
    };
    saveBtn.onclick = async () => {
        try {
            saveBtn.disabled = true;
            writeTimeline({ force: true });
            const timelineText = getWidget(node, "timeline_data")?.value || "";
            const exportedTimeline = boardTimelineForExport(timelineText);
            const timelinePayload = exportedTimeline.payload;
            const promptText = exportedTimeline.text || timelineText || JSON.stringify(timeline || {});
            console.log("[IAMCCS V3 BOARD SAVE]", {
                kind: "board",
                nodeId: node?.id,
                containsCoastline: /\bcoastline\b/i.test(promptText),
                firstPrompt: String(timelinePayload?.segments?.[0]?.prompt || timelinePayload?.rows?.[0]?.relay_prompt || "").replace(/\s+/g, " ").slice(0, 220),
                settings: v3SettingsSnapshot(),
            });
            const board = {
                metadata: { schema: "iamccs.cine.filmmaker_board", schema_version: 1, saved_at: new Date().toISOString(), node_type: nodeClassName(node) },
                global_prompt: promptArea.value,
                timeline_data: exportedTimeline.text,
                timeline: timelinePayload,
                image_paths: refPaths(),
                settings: v3SettingsSnapshot(),
            };
            Object.assign(board, board.settings);
            attachPackageHintToBoard(board);
            await saveBoardJsonAs(board, safeBoardFilename("cine_filmmaker_v3"), (message) => {
                showTimelineNotice(message, message && /failed/i.test(message) ? "error" : "warn");
            });
        } catch (err) {
            console.error("[IAMCCS Cine Shotboard V3] board save failed", err);
            showTimelineNotice(`Save Board failed: ${err?.message || err}`, "error");
        } finally {
            saveBtn.disabled = false;
        }
    };
    savePackageBtn.onclick = async () => {
        try {
            savePackageBtn.disabled = true;
            writeTimeline({ force: true });
            const timelineText = getWidget(node, "timeline_data")?.value || "";
            const exportedTimeline = boardTimelineForExport(timelineText);
            const timelinePayload = exportedTimeline.payload;
            const promptText = exportedTimeline.text || timelineText || JSON.stringify(timeline || {});
            console.log("[IAMCCS V3 BOARD SAVE]", {
                kind: "package",
                nodeId: node?.id,
                containsCoastline: /\bcoastline\b/i.test(promptText),
                firstPrompt: String(timelinePayload?.segments?.[0]?.prompt || timelinePayload?.rows?.[0]?.relay_prompt || "").replace(/\s+/g, " ").slice(0, 220),
                settings: v3SettingsSnapshot(),
            });
            const board = {
                metadata: { schema: "iamccs.cine.filmmaker_board", schema_version: 1, saved_at: new Date().toISOString(), node_type: nodeClassName(node) },
                global_prompt: promptArea.value,
                timeline_data: exportedTimeline.text,
                timeline: timelinePayload,
                image_paths: refPaths(),
                settings: v3SettingsSnapshot(),
            };
            Object.assign(board, board.settings);
            const packageBoard = compactV3BoardForPackageExport(board);
            const packageName = await askShotboardPackageName(`cine_filmmaker_v3_${timestampForPackageName()}`);
            if (!packageName) {
                showTimelineNotice("Save Package cancelled.", "warn");
                return;
            }
            await saveShotboardPackageFolder(packageBoard, "cine_filmmaker_v3", (message) => {
                showTimelineNotice(message, message && /failed/i.test(message) ? "error" : "warn");
            }, packageName);
        } catch (err) {
            console.error("[IAMCCS Cine Shotboard V3] package save failed", err);
            showTimelineNotice(`Save Package failed: ${err?.message || err}`, "error");
        } finally {
            savePackageBtn.disabled = false;
        }
    };
    clearBtn.onclick = () => {
        stopPlayback();
        promptArea.value = "";
        setWidgetValue(node, "global_prompt", "");
        timeline = { schema: "iamccs.cine.filmmaker_timeline", schema_version: 2, segments: [], audioSegments: [], audioTrackCount: 1, audioSyncMode: "timeline_audio", generationStrategy: "single_timeline", flfrealMode: "iamccs_enhanced", globalPromptOnly: false, verboseLog: true };
        selectedId = null;
        clearOwnReferencePaths(node);
        writeTimeline();
        draw();
    };

    const widget = node.addDOMWidget("Cine Shotboard V3", "iamccs_cine_shotboard_v3", root, { serialize: false });
    node._iamccsCineShotboardV3Widget = widget;
    const v3RigidWidth = SHOTBOARD_V3_RIGID_WIDTH;
    widget.computeSize = (width) => {
        return [
            Math.max(v3RigidWidth, Number(width || v3RigidWidth)),
            currentNodeHeight(),
        ];
    };
    lockNodeMinimumSize(node, [v3RigidWidth, currentNodeHeight()], {
        lockResize: false,
        lockWidth: true,
        preferredSize: [v3RigidWidth, currentNodeHeight()],
    });
    if (!node._iamccsCineShotboardV3SerializeWrapped) {
        node._iamccsCineShotboardV3SerializeWrapped = true;
        const originalOnSerialize = node.onSerialize;
        node.onSerialize = function (serialized) {
            const result = originalOnSerialize?.call?.(this, serialized);
            try {
                if (node._iamccsCineShotboardV3WriteTimeline) node._iamccsCineShotboardV3WriteTimeline({ force: true });
                const timelineValue = String(getWidget(node, "timeline_data")?.value || "");
                const globalValue = String(getWidget(node, "global_prompt")?.value || "");
                syncSerializedWidgetValue(node, serialized, "timeline_data", timelineValue);
                syncSerializedWidgetValue(node, serialized, "global_prompt", globalValue);
                serialized.properties = serialized.properties || {};
                serialized.properties.iamccs_v3_timeline_data_backup = timelineValue;
                serialized.properties.iamccs_v3_global_prompt_backup = globalValue;
            } catch (err) {
                console.warn("[IAMCCS Cine Shotboard V3] timeline serialize flush failed", err);
            }
            return result;
        };
    }
    if (!node._iamccsCineShotboardV3WidgetCallbackWrapped) {
        node._iamccsCineShotboardV3WidgetCallbackWrapped = true;
        const originalPromptCallback = promptWidget?.callback;
        const originalTimelineCallback = timelineWidget?.callback;
        if (promptWidget) {
            promptWidget.callback = function (...args) {
                const result = originalPromptCallback?.apply?.(this, args);
                try {
                    const nextPrompt = String(promptWidget.value || "");
                    if (nextPrompt !== String(node._iamccsCineShotboardV3LastPromptText || "")) {
                        node._iamccsCineShotboardV3LastPromptText = nextPrompt;
                        promptArea.value = nextPrompt;
                    }
                } catch {}
                return result;
            };
        }
        if (timelineWidget) {
            timelineWidget.callback = function (...args) {
                const result = originalTimelineCallback?.apply?.(this, args);
                try {
                    const nextTimelineText = String(timelineWidget.value || "");
                    if (nextTimelineText !== String(node._iamccsCineShotboardV3LastTimelineText || "")) {
                        node._iamccsCineShotboardV3LastTimelineText = nextTimelineText;
                        timeline = readTimeline();
                        syncTimingWidgetsFromTimelineTruth("external_timeline_widget_update");
                        draw();
                    }
                } catch (err) {
                    console.warn("[IAMCCS Cine Shotboard V3] external timeline refresh failed", err);
                }
                return result;
            };
        }
    }
    node._iamccsCineShotboardV3WriteTimeline = writeTimeline;
    node._iamccsCineShotboardV3ApplyExternalTimeline = applyExternalTimelineData;
    setTimeout(draw, 0);
}

function renderCinePromptArchitect(node) {
    if (node._iamccsCinePromptArchitectReady === CINE_VERSION) return;
    node._iamccsCinePromptArchitectReady = CINE_VERSION;
    const cp = {
        node: "#2B2118",
        nodeBg: "#17120D",
        rootA: "#101914",
        rootB: "#2A1A10",
        rootC: "#3B2314",
        panel: "rgba(18,27,21,.86)",
        panel2: "rgba(44,30,18,.74)",
        paper: "#FFFFFF",
        ink: "#141414",
        text: "#F4EFE6",
        muted: "#C7B49B",
        border: "rgba(199,151,94,.42)",
        borderStrong: "rgba(226,172,95,.68)",
        accent: "#5B7A55",
        accent2: "#B06B35",
        dark: "#182219",
        danger: "#8A3434",
    };
    node.color = cp.node;
    node.bgcolor = cp.nodeBg;
    node.boxcolor = cp.accent;

    const widgetNames = [
        "template",
        "subject_identity",
        "environment",
        "lighting_weather",
        "visual_style",
        "camera_language",
        "continuity_rules",
        "shot_goal",
        "movement_path",
        "target_reveal",
        "performance_or_emotion",
        "audio_or_dialogue",
        "avoid",
        "duration_seconds",
        "frame_rate",
        "beat_count",
        "beat_data",
    ];
    widgetNames.forEach((name) => hideWidget(getWidget(node, name)));

    const root = document.createElement("div");
    root.style.cssText = [
        "box-sizing:border-box",
        "width:100%",
        "min-height:760px",
        "padding:14px",
        `background:linear-gradient(145deg,${cp.rootA} 0%,${cp.rootB} 54%,${cp.rootC} 100%)`,
        `border:1px solid ${cp.borderStrong}`,
        "border-radius:8px",
        `color:${cp.text}`,
        "font-family:Arial,sans-serif",
        "font-size:12px",
        "overflow:hidden",
        "box-shadow:inset 0 1px 0 rgba(255,255,255,.06), inset 0 -28px 70px rgba(0,0,0,.22)",
    ].join(";");

    const widgetValue = (name, fallback = "") => {
        const widget = getWidget(node, name);
        const value = widget?.value;
        if (value === undefined || value === null || String(value) === "") return fallback;
        return value;
    };
    const setValue = (name, value) => setWidgetValue(node, name, value);
    const templates = [
        "continuous_dolly",
        "future_keyframe",
        "image_text_image",
        "dialogue_lipsync",
        "reveal",
        "environmental_transition",
    ];
    const defaultBeatRows = () => ([
        {
            label: "opening_contract",
            duration: 3,
            image_role: "opening anchor",
            action: "the camera establishes the subject and begins one continuous physical movement",
            bridge: "carry the same camera movement into the next beat without a hard cut",
        },
        {
            label: "development",
            duration: 3,
            image_role: "motion checkpoint",
            action: "the movement deepens; environment details react physically as the camera advances",
            bridge: "arrive toward the next visual target only near the end of the beat",
        },
        {
            label: "arrival",
            duration: 3,
            image_role: "target keyframe",
            action: "the shot arrives at the strongest visual target with coherent identity and space",
            bridge: "",
        },
    ]);
    const normalizeBeat = (beat, index) => ({
        label: String(beat?.label || beat?.name || `beat_${index + 1}`),
        duration: Math.max(0.1, Number(beat?.duration ?? beat?.seconds ?? 3) || 3),
        image_role: String(beat?.image_role ?? beat?.image_hint ?? ""),
        action: String(beat?.action ?? beat?.local_prompt ?? beat?.prompt ?? "the camera continues one clear cinematic action through this timed beat"),
        bridge: String(beat?.bridge ?? beat?.step_transition_prompt ?? beat?.action_to_next ?? ""),
    });
    const parseBeats = () => {
        try {
            const data = JSON.parse(String(widgetValue("beat_data", "") || "").trim() || "{}");
            const raw = Array.isArray(data) ? data : Array.isArray(data?.beats) ? data.beats : [];
            const rows = raw.map(normalizeBeat).filter(Boolean);
            if (rows.length) return rows;
        } catch {
            // Fall through to a clean starter structure.
        }
        return defaultBeatRows();
    };
    let beats = parseBeats();
    let architectTextScale = Math.max(0.85, Math.min(1.45, Number(node.properties?.iamccs_cineprompt_text_scale || 1) || 1));
    const architectFontSize = (base = 12) => `${Math.max(9, Math.round(Number(base || 12) * architectTextScale * 10) / 10)}px`;

    const paperTextStyle = (opts = {}) => [
        inputBase(),
        "width:100%",
        "box-sizing:border-box",
        `background:${cp.paper}!important`,
        `color:${cp.ink}!important`,
        `border:1px solid ${cp.border}`,
        "border-radius:6px",
        "font-family:'Courier New',Courier,monospace",
        "font-weight:700",
        `font-size:${architectFontSize(opts.preview ? 11 : 12)}`,
        "letter-spacing:0",
        "line-height:1.42",
        "box-shadow:inset 0 1px 3px rgba(0,0,0,.08)",
        opts.multiline ? `min-height:${opts.height || 74}px;resize:vertical;` : "height:32px;",
    ].join(";");
    const fieldLabel = (label, control, hint = "") => {
        const wrap = document.createElement("label");
        wrap.style.cssText = "display:flex;flex-direction:column;gap:6px;min-width:0;";
        const top = document.createElement("div");
        top.style.cssText = `display:grid;gap:3px;color:${cp.muted};font-size:10px;font-weight:900;text-transform:uppercase;letter-spacing:0;`;
        const text = document.createElement("span");
        text.textContent = label;
        top.appendChild(text);
        if (hint) {
            const h = document.createElement("span");
            h.textContent = hint;
            h.style.cssText = "font-size:10px;font-weight:800;opacity:.92;text-transform:none;line-height:1.35;color:#E5D0AA;";
            top.appendChild(h);
        }
        wrap.append(top, control);
        return wrap;
    };
    const makeTextarea = (name, label, height, hint = "") => {
        const el = document.createElement("textarea");
        el.value = String(widgetValue(name, ""));
        el.rows = Math.max(2, Math.round((height || 74) / 24));
        el.style.cssText = paperTextStyle({ multiline: true, height });
        el.dataset.iamccsCpaPaper = "1";
        el.dataset.iamccsCpaHeight = String(height || 74);
        el.oninput = () => {
            setValue(name, el.value);
            refreshPreview();
        };
        protectControlDrag(el);
        return fieldLabel(label, el, hint);
    };
    const makeBeatTextarea = (beat, key, placeholder, height = 96) => {
        const el = document.createElement("textarea");
        el.value = String(beat[key] || "");
        el.placeholder = placeholder || "";
        el.rows = 4;
        el.style.cssText = paperTextStyle({ multiline: true, height });
        el.dataset.iamccsCpaPaper = "1";
        el.dataset.iamccsCpaHeight = String(height || 96);
        el.oninput = () => {
            beat[key] = el.value;
            syncBeats();
        };
        protectControlDrag(el);
        return el;
    };
    const makePaperInput = (value, onInput, placeholder = "") => {
        const el = document.createElement("input");
        el.type = "text";
        el.value = String(value || "");
        el.placeholder = placeholder;
        el.style.cssText = paperTextStyle();
        el.dataset.iamccsCpaPaper = "1";
        el.oninput = () => onInput(el.value);
        protectControlDrag(el);
        return el;
    };
    const refreshArchitectPaperStyles = () => {
        root.querySelectorAll?.("[data-iamccs-cpa-paper='1']").forEach((el) => {
            const multiline = el.tagName === "TEXTAREA";
            const height = Number(el.dataset.iamccsCpaHeight || 74);
            const preview = el.dataset.iamccsCpaPreview === "1";
            el.style.cssText = paperTextStyle({ multiline, height, preview }) + (el.readOnly ? "opacity:.82;" : "");
        });
    };
    const makeSelect = (value, options, onChange) => {
        const select = document.createElement("select");
        options.forEach((item) => {
            const opt = document.createElement("option");
            opt.value = item;
            opt.textContent = String(item).replace(/_/g, " ");
            select.appendChild(opt);
        });
        select.value = options.includes(String(value)) ? String(value) : options[0];
        select.style.cssText = [
            inputBase(),
            "height:32px",
            `background:${cp.dark}`,
            "color:#FFFFFF",
            `border:1px solid ${cp.accent}`,
            "border-radius:6px",
            "font-weight:900",
            "text-transform:capitalize",
        ].join(";");
        select.onchange = () => onChange(select.value);
        return protectControlDrag(select);
    };
    const makeDarkNumberInput = (value, min, max, step, onChange) => {
        const input = document.createElement("input");
        input.type = "number";
        input.value = String(value);
        input.min = String(min);
        if (max != null) input.max = String(max);
        input.step = String(step);
        input.style.cssText = [
            inputBase(),
            "height:32px",
            `background:${cp.dark}`,
            "color:#FFFFFF",
            `border:1px solid ${cp.accent}`,
            "border-radius:6px",
            "font-weight:900",
            "text-align:center",
            "font-variant-numeric:tabular-nums",
        ].join(";");
        input.onchange = input.oninput = () => onChange(Number(input.value || value));
        input._iamccsSetValue = (next) => { input.value = String(next); };
        return protectControlDrag(input);
    };
    const makeBtn = (label, tone = "neutral") => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = label;
        const bg = tone === "primary" ? cp.accent : tone === "danger" ? cp.danger : "#E8E5D9";
        const fg = tone === "neutral" ? cp.ink : "#FFFFFF";
        const border = tone === "neutral" ? cp.borderStrong : bg;
        btn.style.cssText = [
            "height:32px",
            "padding:0 12px",
            `border:1px solid ${border}`,
            "border-radius:6px",
            `background:${bg}`,
            `color:${fg}`,
            "font-weight:900",
            "cursor:pointer",
            "white-space:nowrap",
        ].join(";");
        return protectControlDrag(btn);
    };
    const totalBeatSeconds = () => beats.reduce((sum, beat) => sum + Math.max(0.1, Number(beat.duration || 0)), 0);
    const buildGlobalPreview = () => {
        const parts = [
            widgetValue("shot_goal", ""),
            widgetValue("subject_identity", ""),
            widgetValue("environment", ""),
            widgetValue("lighting_weather", ""),
            widgetValue("visual_style", ""),
            widgetValue("camera_language", ""),
            widgetValue("movement_path", ""),
            widgetValue("target_reveal", ""),
            widgetValue("performance_or_emotion", ""),
            widgetValue("audio_or_dialogue", ""),
            widgetValue("continuity_rules", ""),
        ].map((part) => String(part || "").trim()).filter(Boolean);
        return parts.join(", ").replace(/\s+/g, " ");
    };
    const buildLocalPreview = (beat, index) => {
        const parts = [
            beat.action,
            index < beats.length - 1 ? beat.bridge : "",
            String(widgetValue("template", "continuous_dolly")) === "dialogue_lipsync" && String(widgetValue("audio_or_dialogue", "")).trim()
                ? "lips sync naturally to the dialogue, diction is clear, facial expression remains alive and precise"
                : "",
            index === beats.length - 1 ? widgetValue("target_reveal", "") : "",
        ].map((part) => String(part || "").trim()).filter(Boolean);
        return parts.join(", ").replace(/\s+/g, " ");
    };

    let previewBox = null;
    let status = null;
    let list = null;
    let durationCtrl = null;
    let beatCountCtrl = null;
    const refreshPreview = () => {
        if (!previewBox) return;
        const fps = Math.max(1, Number(widgetValue("frame_rate", 24)) || 24);
        const duration = Math.max(0.1, Number(widgetValue("duration_seconds", totalBeatSeconds())) || totalBeatSeconds());
        const scale = duration / Math.max(0.1, totalBeatSeconds());
        const frames = beats.map((beat) => Math.max(1, Math.round(Math.max(0.1, Number(beat.duration || 0)) * scale * fps)));
        if (frames.length) frames[frames.length - 1] = Math.max(1, frames[frames.length - 1] + Math.round(duration * fps) - frames.reduce((a, b) => a + b, 0));
        const locals = beats.map(buildLocalPreview);
        previewBox.value = [
            "GLOBAL PROMPT",
            buildGlobalPreview(),
            "",
            "LOCAL PROMPTS",
            locals.map((text, index) => `${index + 1}. ${text}`).join("\n"),
            "",
            "SEGMENT LENGTHS",
            frames.join(","),
        ].join("\n");
        if (status) {
            status.textContent = `Template: ${String(widgetValue("template", "continuous_dolly")).replace(/_/g, " ")} | Beats: ${beats.length} | Beat script: ${totalBeatSeconds().toFixed(1)}s | Output target: ${duration.toFixed(1)}s @ ${fps}fps`;
        }
    };
    const syncBeats = () => {
        beats = beats.map(normalizeBeat);
        setValue("beat_data", JSON.stringify({ beats }, null, 2));
        setValue("beat_count", beats.length);
        beatCountCtrl?._iamccsSetValue?.(beats.length);
        refreshPreview();
    };
    const fitDuration = () => {
        const total = Number(totalBeatSeconds().toFixed(3));
        setValue("duration_seconds", total);
        durationCtrl?._iamccsSetValue?.(total);
        refreshPreview();
    };

    const head = document.createElement("div");
    head.style.cssText = [
        "display:flex",
        "align-items:center",
        "gap:12px",
        "padding:11px 12px",
        `background:${cp.dark}`,
        "color:#FFFFFF",
        "border-radius:7px",
        "border:1px solid rgba(0,0,0,.25)",
    ].join(";");
    const titleBlock = document.createElement("div");
    titleBlock.style.cssText = "display:grid;gap:2px;flex:1;min-width:0;";
    const title = document.createElement("div");
    title.textContent = "IAMCCS CinePrompt Architect";
    title.style.cssText = "font-size:18px;font-weight:900;letter-spacing:0;";
    const subtitle = document.createElement("div");
    subtitle.textContent = "Build WAN shotboard prompts, then send timing and text into Shotboard V3 or V2.";
    subtitle.style.cssText = "font-size:11px;font-weight:800;color:#DDE5DE;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
    titleBlock.append(title, subtitle);
    const headActions = document.createElement("div");
    headActions.style.cssText = "display:flex;align-items:center;gap:8px;flex-wrap:wrap;justify-content:flex-end;";
    head.append(titleBlock, headActions);

    const topGrid = document.createElement("div");
    topGrid.style.cssText = [
        "display:grid",
        "grid-template-columns:minmax(250px,1.2fr) 150px 120px 120px 150px",
        "gap:10px",
        "margin-top:10px",
        `padding:10px`,
        `background:${cp.panel}`,
        `border:1px solid ${cp.border}`,
        "border-radius:7px",
    ].join(";");
    const templateSelect = makeSelect(widgetValue("template", "continuous_dolly"), templates, (value) => {
        setValue("template", value);
        refreshPreview();
    });
    durationCtrl = makeDarkNumberInput(Number(widgetValue("duration_seconds", totalBeatSeconds() || 9)), 0.1, 600, 0.1, (value) => {
        setValue("duration_seconds", Number(value));
        refreshPreview();
    });
    const fpsCtrl = makeDarkNumberInput(Number(widgetValue("frame_rate", 24)), 1, 120, 1, (value) => {
        setValue("frame_rate", Number(value));
        refreshPreview();
    });
    beatCountCtrl = makeDarkNumberInput(beats.length, 1, 8, 1, (value) => {
        const target = Math.max(1, Math.min(8, Number(value) || 1));
        while (beats.length < target) beats.push(normalizeBeat({}, beats.length));
        while (beats.length > target) beats.pop();
        syncBeats();
        drawBeats();
    });
    const fontSizeSelect = makeSelect(
        architectTextScale <= 0.92 ? "small" : architectTextScale >= 1.18 ? "large" : "medium",
        ["small", "medium", "large"],
        (value) => {
            architectTextScale = value === "small" ? 0.9 : value === "large" ? 1.22 : 1;
            node.properties = node.properties || {};
            node.properties.iamccs_cineprompt_text_scale = architectTextScale;
            refreshArchitectPaperStyles();
            drawBeats();
        }
    );
    topGrid.append(
        fieldLabel("Template", templateSelect),
        fieldLabel("Duration", durationCtrl, "seconds"),
        fieldLabel("Frame rate", fpsCtrl, "fps"),
        fieldLabel("Beat count", beatCountCtrl),
        fieldLabel("Text size", fontSizeSelect)
    );

    const structureGrid = document.createElement("div");
    structureGrid.style.cssText = [
        "display:grid",
        "grid-template-columns:minmax(640px,1fr)",
        "gap:12px",
        "margin-top:10px",
        `padding:12px`,
        `background:linear-gradient(180deg,${cp.panel} 0%,${cp.panel2} 100%)`,
        `border:1px solid ${cp.border}`,
        "border-radius:7px",
    ].join(";");
    const globalTitle = document.createElement("div");
    globalTitle.textContent = "Global prompt hierarchy";
    globalTitle.style.cssText = `color:${cp.text};font-size:13px;font-weight:900;text-transform:uppercase;letter-spacing:0;border-bottom:1px solid ${cp.border};padding-bottom:8px;`;
    structureGrid.appendChild(globalTitle);
    [
        ["shot_goal", "01. Shot goal", 72, "Define the promise of the shot. Example: \"one continuous dolly from a living face into the ocean inside the pupil\"."],
        ["subject_identity", "02. Subject identity", 72, "Permanent identity anchors. Example: \"same adult man, same blue eye, calm attentive expression, natural skin texture\"."],
        ["environment", "03. Environment", 72, "Where the shot physically exists. Example: \"dim studio face close-up that opens into a coastal ocean world\"."],
        ["lighting_weather", "04. Lighting / weather", 72, "Stable light and atmosphere. Example: \"soft catchlights on the eye, overcast coastal light, gentle cloud movement\"."],
        ["visual_style", "05. Visual style", 72, "Image language and realism level. Example: \"cinematic realism, moist macro detail, natural parallax, grounded camera physics\"."],
        ["camera_language", "06. Camera language", 72, "The camera behavior. Example: \"slow forward dolly, macro push-in, optical tunnel movement through the pupil\"."],
        ["movement_path", "07. Movement path", 72, "The exact travel path over time. Example: \"face to eye, eye to pupil, pupil to coastline, coastline to waves\"."],
        ["target_reveal", "08. Target / reveal", 72, "What the shot is moving toward. Example: \"the pupil becomes a coastal sea view that fills the frame\"."],
        ["performance_or_emotion", "09. Performance / emotion", 72, "Human or action performance. Example: \"tiny eye-focus shifts, faint breathing, calm concentration\"."],
        ["audio_or_dialogue", "10. Audio / dialogue", 72, "Speech, music or sync intent. Example: \"low ocean rumble grows as the camera reaches the surf\"."],
        ["continuity_rules", "11. Continuity rules", 72, "Positive continuity instructions. Example: \"every waypoint belongs to the same forward camera path with stable exposure\"."],
        ["avoid", "12. Avoid / risk notes", 72, "Use sparingly for internal risk notes. Example: \"identity drift, frozen water, early arrival at the ocean\"."],
    ].forEach(([name, label, height, hint]) => {
        structureGrid.appendChild(makeTextarea(name, label, height, hint || ""));
    });

    status = document.createElement("div");
    status.style.cssText = [
        "margin-top:10px",
        "height:34px",
        "display:flex",
        "align-items:center",
        "padding:0 10px",
        `background:${cp.dark}`,
        "color:#FFFFFF",
        "border-radius:6px",
        "font-size:12px",
        "font-weight:900",
        "white-space:nowrap",
        "overflow:hidden",
        "text-overflow:ellipsis",
    ].join(";");

    const beatTitle = document.createElement("div");
    beatTitle.textContent = "PromptRelay beat builder";
    beatTitle.style.cssText = `margin-top:12px;margin-bottom:6px;font-size:13px;font-weight:900;color:${cp.text};text-transform:uppercase;letter-spacing:0;border-bottom:1px solid ${cp.border};padding-bottom:7px;`;
    list = document.createElement("div");
    list.style.cssText = "display:flex;flex-direction:column;gap:9px;max-height:660px;overflow-y:auto;overflow-x:hidden;padding-right:4px;scrollbar-gutter:stable;";

    const drawBeats = () => {
        if (!list) return;
        list.innerHTML = "";
        beats.forEach((beat, index) => {
            const row = document.createElement("div");
            row.style.cssText = [
                "display:grid",
                "grid-template-columns:82px minmax(190px,.6fr) minmax(360px,1fr) minmax(360px,1fr) minmax(360px,1fr)",
                "gap:12px",
                "align-items:stretch",
                "padding:12px",
                `background:${cp.panel2}`,
                `border:1px solid ${cp.border}`,
                "border-radius:7px",
                "box-shadow:0 7px 22px rgba(0,0,0,.22), inset 0 1px 0 rgba(255,255,255,.04)",
            ].join(";");
            const badge = document.createElement("div");
            badge.textContent = `BEAT\n${String(index + 1).padStart(2, "0")}`;
            badge.style.cssText = [
                "white-space:pre-line",
                "display:flex",
                "align-items:center",
                "justify-content:center",
                "text-align:center",
                `background:${cp.dark}`,
                "color:#FFFFFF",
                "border-radius:6px",
                "font-weight:900",
                "line-height:1.2",
            ].join(";");
            const meta = document.createElement("div");
            meta.style.cssText = "display:grid;grid-template-rows:auto auto auto;gap:8px;min-width:0;";
            const labelInput = makePaperInput(beat.label, (value) => {
                beat.label = value;
                syncBeats();
            }, "beat label");
            const roleInput = makePaperInput(beat.image_role, (value) => {
                beat.image_role = value;
                syncBeats();
            }, "image role / anchor");
            const dur = numberStepperControl(Number(beat.duration || 3), "0.1", "0.1", "120", (value) => {
                beat.duration = Math.max(0.1, Number(value) || 0.1);
                syncBeats();
            }, { liveInput: false });
            dur.style.gridTemplateColumns = "30px minmax(70px,1fr) 30px";
            styleValueControls(dur);
            meta.append(
                fieldLabel("Label", labelInput, "Short timeline name. Example: \"pupil_to_ocean\"."),
                fieldLabel("Image role", roleInput, "What this waypoint represents. Example: \"macro eye anchor\" or \"wave endpoint\"."),
                fieldLabel("Beat length", dur, "Seconds for this beat. Use this as the main timing control.")
            );
            const action = makeBeatTextarea(beat, "action", "Describe exactly what happens in this timed beat.", 128);
            const bridge = makeBeatTextarea(beat, "bridge", "Optional: how this beat moves into the next beat.", 128);
            const localPreview = document.createElement("textarea");
            localPreview.readOnly = true;
            localPreview.value = buildLocalPreview(beat, index);
            localPreview.style.cssText = paperTextStyle({ multiline: true, height: 128, preview: true }) + "opacity:.82;";
            localPreview.dataset.iamccsCpaPaper = "1";
            localPreview.dataset.iamccsCpaPreview = "1";
            localPreview.dataset.iamccsCpaHeight = "128";
            protectControlDrag(localPreview);
            row.append(
                badge,
                meta,
                fieldLabel("Action text", action, "Visible movement for this beat. Example: \"the camera moves closer to the pupil while iris fibers expand with parallax\"."),
                fieldLabel("Bridge / relay", bridge, "How this beat arrives into the next one. Example: \"the pupil grows into a coastal horizon at the end of the beat\"."),
                fieldLabel("Local prompt preview", localPreview, "Read-only preview of the text that will be sent to PromptRelay for this beat.")
            );
            list.appendChild(row);
        });
        refreshPreview();
    };

    const previewTitle = document.createElement("div");
    previewTitle.textContent = "Generated prompt preview";
    previewTitle.style.cssText = `margin-top:12px;margin-bottom:6px;font-size:13px;font-weight:900;color:${cp.text};text-transform:uppercase;letter-spacing:0;border-bottom:1px solid ${cp.border};padding-bottom:7px;`;
    previewBox = document.createElement("textarea");
    previewBox.readOnly = true;
    previewBox.style.cssText = paperTextStyle({ multiline: true, height: 220 }) + "min-height:220px;";
    previewBox.dataset.iamccsCpaPaper = "1";
    previewBox.dataset.iamccsCpaHeight = "220";
    protectControlDrag(previewBox);

    root.append(head, topGrid, structureGrid, status, beatTitle, list, previewTitle, previewBox);
    const widget = node.addDOMWidget("IAMCCS CinePrompt Architect", "iamccs_cineprompt_architect", root, { serialize: false });
    widget.computeSize = (width) => [Math.max(1720, Number(width || 1720)), 1180];
    lockNodeMinimumSize(node, [1720, 1220], { lockResize: false, preferredSize: [1720, 1240] });
    syncBeats();
    drawBeats();
}

function renderBoardMaker(node) {
    if (node._iamccsBoardMakerReady === CINE_VERSION) return;
    node._iamccsBoardMakerReady = CINE_VERSION;
    const bm = {
        node: "#3A1014",
        nodeBg: "#211012",
        rootA: "#19080A",
        rootB: "#0E0607",
        panel: "#240D10",
        panelDark: "#170709",
        card: "#251011",
        paperCard: "#F3E7E0",
        border: "#7D2B35",
        borderSoft: "#51202A",
        accent: "#A93645",
        accentBorder: "#F08A95",
        danger: "#762A30",
        dangerBorder: "#D85A65",
        muted: "#E4B7BB",
        text: "#FFF1F2",
        valueBg: "#120608",
    };
    node.color = bm.node;
    node.bgcolor = bm.nodeBg;
    node.boxcolor = bm.accent;
    const root = document.createElement("div");
    root.style.cssText = [
        "box-sizing:border-box",
        "width:100%",
        "min-height:700px",
        "padding:12px",
        `background:linear-gradient(180deg,${bm.rootA} 0%,${bm.rootB} 100%)`,
        `border:1px solid ${bm.border}`,
        "border-radius:8px",
        "box-shadow:inset 0 1px 0 rgba(255,255,255,.06)",
        "overflow:hidden",
        `color:${bm.text}`,
        "font-family:Arial,sans-serif",
        "font-size:12px",
    ].join(";");

    const globalWidget = getWidget(node, "global_prompt");
    const durationWidget = getWidget(node, "duration_seconds");
    const fpsWidget = getWidget(node, "frame_rate");
    const imageWidthWidget = getWidget(node, "image_width");
    const imageHeightWidget = getWidget(node, "image_height");
    const forceWidget = getWidget(node, "default_force");
    const guidePolicyWidget = getWidget(node, "guide_policy");
    const boardNameWidget = getWidget(node, "board_name");
    const dataWidget = getWidget(node, "board_data");
    ["global_prompt", "duration_seconds", "frame_rate", "image_width", "image_height", "default_force", "guide_policy", "board_name", "board_data"].forEach((name) => hideWidget(getWidget(node, name)));

    const defaultRows = () => ([
        { label: "box_1", duration: 3, local_prompt: "opening beat, establish subject and camera direction", bridge: "camera continues into the next beat through one connected movement", camera: "slow push-in", transition: "continuous_motion", force: Number(forceWidget?.value || 0.28), use_guide: true, use_prompt: true },
        { label: "box_2", duration: 3, local_prompt: "second beat, deepen action and maintain visual continuity", bridge: "movement carries the viewer toward the final beat", camera: "continuous dolly-in", transition: "continuous_motion", force: Number(forceWidget?.value || 0.28), use_guide: true, use_prompt: true },
        { label: "box_3", duration: 3, local_prompt: "final beat, arrive at the strongest visual target", bridge: "", camera: "macro push-in", transition: "continuous_motion", force: Number(forceWidget?.value || 0.28), use_guide: true, use_prompt: true },
    ]);
    const normalizeMakerRow = (row, index) => ({
        label: String(row?.label || row?.name || `box_${index + 1}`),
        duration: Math.max(0.1, Number(row?.duration ?? row?.seconds ?? row?.len ?? 3) || 3),
        local_prompt: String(row?.local_prompt ?? row?.relay_prompt ?? row?.prompt ?? ""),
        bridge: String(row?.bridge ?? row?.step_transition_prompt ?? row?.action_to_next ?? row?.note ?? ""),
        camera: String(row?.camera || "continuous dolly-in"),
        transition: String(row?.transition || "continuous_motion"),
        force: Math.max(0, Math.min(1, Number(row?.force ?? row?.guide_strength ?? forceWidget?.value ?? 0.28) || 0)),
        image_hint: String(row?.image_hint ?? row?.image_note ?? row?.image_name ?? ""),
        use_guide: row?.use_guide !== false,
        use_prompt: row?.use_prompt !== false,
    });
    const parseRows = () => {
        try {
            const data = JSON.parse(String(dataWidget?.value || "").trim() || "{}");
            const raw = Array.isArray(data) ? data : Array.isArray(data?.rows) ? data.rows : [];
            const parsed = raw.map(normalizeMakerRow).filter(Boolean);
            return parsed.length ? parsed : defaultRows();
        } catch {
            return defaultRows();
        }
    };
    let rows = parseRows();
    let boxStyleMode = String(node.properties?.iamccs_boardmaker_box_style || "dark") === "paper" ? "paper" : "dark";
    let boxTextScale = Math.max(0.85, Math.min(1.7, Number(node.properties?.iamccs_boardmaker_text_scale || 1) || 1));
    const boardMakerBoxTextCss = () => boxStyleMode === "paper"
        ? "background:#F8F1EC!important;border-color:#B9949A!important;color:#050505!important;font-family:'Courier New',Courier,monospace!important;font-weight:700!important;"
        : `background:${bm.valueBg};border-color:${bm.border};color:${bm.text};`;
    const boardMakerBoxTextFontCss = (opts = {}) => opts.boxText
        ? `font-size:${Math.round(12 * boxTextScale)}px;line-height:1.38;`
        : "";
    const boardMakerTextBoxStyle = (opts = {}) => inputBase() + `width:100%;box-sizing:border-box;${boardMakerBoxTextCss()}${boardMakerBoxTextFontCss({ boxText: true })}border-radius:6px;${opts.multiline ? `min-height:${opts.height || 58}px;resize:vertical;` : "height:32px;"}`;
    let refreshBoardMakerTextStyles = () => {
        root.querySelectorAll?.("[data-iamccs-boardmaker-textbox='1']").forEach((el) => {
            const multiline = el.tagName === "TEXTAREA";
            const height = Number(el.dataset.iamccsHeight || 58);
            el.style.cssText = boardMakerTextBoxStyle({ multiline, height });
        });
    };

    const makeBtn = (label, tone = "neutral") => {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = label;
        const bg = tone === "primary" ? bm.accent : tone === "danger" ? bm.danger : "#321216";
        const border = tone === "primary" ? bm.accentBorder : tone === "danger" ? bm.dangerBorder : bm.border;
        b.style.cssText = `height:32px;padding:0 12px;border:1px solid ${border};border-radius:6px;background:${bg};color:${bm.text};font-weight:900;cursor:pointer;white-space:nowrap;box-shadow:inset 0 1px 0 rgba(255,255,255,.08);`;
        return protectControlDrag(b);
    };
    const makeText = (value, onInput, opts = {}) => {
        const el = document.createElement(opts.multiline ? "textarea" : "input");
        if (!opts.multiline) el.type = "text";
        el.value = value || "";
        el.placeholder = opts.placeholder || "";
        el.rows = opts.rows || 2;
        if (opts.boxText) {
            el.dataset.iamccsBoardmakerTextbox = "1";
            el.dataset.iamccsHeight = String(opts.height || 58);
            el.style.cssText = boardMakerTextBoxStyle(opts);
        } else {
            const colorCss = `background:${bm.valueBg};border-color:${bm.border};color:${bm.text};`;
            el.style.cssText = inputBase() + `width:100%;box-sizing:border-box;${colorCss}border-radius:6px;${opts.multiline ? `min-height:${opts.height || 58}px;resize:vertical;line-height:1.35;` : "height:32px;"}`;
        }
        el.oninput = () => onInput(el.value);
        protectControlDrag(el);
        return el;
    };
    const makeSelect = (value, options, onChange) => {
        const el = document.createElement("select");
        options.forEach((item) => {
            const opt = document.createElement("option");
            opt.value = item;
            opt.textContent = item.replace(/_/g, " ");
            el.appendChild(opt);
        });
        el.value = options.includes(String(value)) ? String(value) : options[0];
        el.style.cssText = inputBase() + `height:32px;background:${bm.valueBg};border-color:${bm.border};color:${bm.text};border-radius:6px;`;
        el.onchange = () => onChange(el.value);
        return protectControlDrag(el);
    };
    const sync = () => {
        const cleanRows = rows.map(normalizeMakerRow);
        rows = cleanRows;
        setWidgetValue(node, "board_data", JSON.stringify({ rows: cleanRows }, null, 2));
        setWidgetValue(node, "global_prompt", globalPrompt.value);
        setWidgetValue(node, "board_name", boardName.value);
        setWidgetValue(node, "duration_seconds", Number(durationControlValue()));
        setWidgetValue(node, "frame_rate", Number(fpsControlValue()));
        setWidgetValue(node, "image_width", Number(imageWidthControlValue()));
        setWidgetValue(node, "image_height", Number(imageHeightControlValue()));
        setWidgetValue(node, "default_force", Number(defaultForceValue()));
        setWidgetValue(node, "guide_policy", guidePolicy.value);
    };
    const totalRowsDuration = () => rows.reduce((sum, row) => sum + Math.max(0.1, Number(row.duration || 0)), 0);
    let durationControlValue = () => Number(durationWidget?.value || 9);
    let fpsControlValue = () => Number(fpsWidget?.value || 24);
    let imageWidthControlValue = () => Number(imageWidthWidget?.value || 768);
    let imageHeightControlValue = () => Number(imageHeightWidget?.value || 432);
    let defaultForceValue = () => Number(forceWidget?.value || 0.28);
    const makerRowsFromBoard = (data) => {
        const board = boardFromWorkflowJson(data) || data || {};
        const rawRows = Array.isArray(board.rows) ? board.rows : parseShotboardTimelineString(String(board.timeline_data || ""));
        const images = Array.isArray(board.images) ? board.images : [];
        return rawRows.map((row, index) => {
            const next = rawRows[index + 1] || null;
            const currentSecond = Number(row?.second ?? 0);
            const nextSecond = next ? Number(next?.second ?? currentSecond + 3) : null;
            const inferredDuration = Number(row?.duration ?? row?.len ?? row?.seconds ?? (nextSecond != null ? nextSecond - currentSecond : 3));
            const imageInfo = images.find((img) => Number(img?.ref) === Number(row?.ref ?? index + 1)) || null;
            return normalizeMakerRow({
                ...row,
                duration: Math.max(0.1, Number.isFinite(inferredDuration) ? inferredDuration : 3),
                local_prompt: row?.relay_prompt ?? row?.local_prompt ?? row?.prompt ?? "",
                bridge: row?.step_transition_prompt ?? row?.bridge ?? row?.note ?? "",
                image_hint: imageInfo?.name || imageInfo?.path || row?.image_hint || "",
            }, index);
        }).filter(Boolean);
    };

    const collectBoard = () => {
        sync();
        const fps = Math.max(1, Number(fpsControlValue()) || 24);
        const imageWidth = Math.max(64, Math.round(Number(imageWidthControlValue()) || 768));
        const imageHeight = Math.max(64, Math.round(Number(imageHeightControlValue()) || 432));
        const totalDuration = Math.max(0.1, Number(durationControlValue()) || 0, totalRowsDuration());
        let cursor = 0;
        const shotRows = rows.map((row, index) => {
            const duration = Math.max(0.1, Number(row.duration || 0) || 0.1);
            const bridge = String(row.bridge || "").trim();
            const local = String(row.local_prompt || "").trim();
            const shot = {
                _ui_id: `boardmaker_${index + 1}_${String(row.label || `box_${index + 1}`).replace(/[^\w.-]+/g, "_")}`,
                second: Number(cursor.toFixed(3)),
                frame: Math.round(cursor * fps),
                ref: index + 1,
                force: Math.max(0, Math.min(1, Number(row.force || defaultForceValue()) || 0)),
                image_lock_strength: Math.max(0, Math.min(1, Number(row.force || defaultForceValue()) || 0)),
                use_guide: row.use_guide !== false,
                use_prompt: row.use_prompt !== false && Boolean(local || bridge),
                label: String(row.label || `box_${index + 1}`),
                camera: String(row.camera || "continuous dolly-in"),
                transition: String(row.transition || "continuous_motion"),
                image_hint: String(row.image_hint || ""),
                note: bridge,
                relay_prompt: local,
                use_relay_modifiers: false,
                camera_relay_mode: "off",
                transition_relay_mode: "off",
                relay_addon_position: "after",
                relay_modifier_text: "",
                step_transition_enabled: Boolean(bridge && index < rows.length - 1),
                step_transition_type: bridge && index < rows.length - 1 ? "action_beat" : "off",
                step_transition_prompt: index < rows.length - 1 ? bridge : "",
                step_transition_easing: "ease_in_out",
                step_transition_force_curve: "balanced",
                step_transition_duration: index < rows.length - 1 ? duration : 0,
                step_transition_arrival: "auto",
                step_transition_auto_fit: true,
                duration,
            };
            cursor += duration;
            return shot;
        });
        const timelineData = JSON.stringify({ rows: shotRows }, null, 2);
        const boardNameClean = String(boardName.value || "iamccs_boardmaker_board").trim() || "iamccs_boardmaker_board";
        return {
            metadata: {
                schema: "iamccs.cine.shotboard.board",
                schema_version: 1,
                cine_ui_version: "2026-05-19-boardmaker-1",
                saved_at: new Date().toISOString(),
                node_type: "IAMCCS_BoardMaker",
                board_name: boardNameClean,
                image_storage: "manual_after_import",
                notes: "BoardMaker exports prompts and timing only. Add reference images manually in Shotboard after import.",
            },
            global_prompt: globalPrompt.value,
            prompt: globalPrompt.value,
            timeline_data: timelineData,
            rows: shotRows,
            settings: {
                duration_seconds: totalDuration,
                frame_rate: fps,
                guide_policy: guidePolicy.value,
                min_guide_gap_seconds: 0,
                max_guides: Math.max(1, shotRows.length),
                default_force: Number(defaultForceValue()) || 0.28,
                promptrelay_epsilon: 0.6,
                wan_frame_round_mode: "up_8n_plus_1",
                image_width: imageWidth,
                image_height: imageHeight,
                image_resize_method: "crop",
                image_multiple_of: 32,
                img_compression: 0,
            },
            duration_seconds: totalDuration,
            frame_rate: fps,
            guide_policy: guidePolicy.value,
            default_force: Number(defaultForceValue()) || 0.28,
            image_width: imageWidth,
            image_height: imageHeight,
            image_paths: "",
            images: [],
        };
    };

    const head = document.createElement("div");
    head.style.cssText = `display:flex;align-items:center;gap:10px;margin-bottom:10px;padding:10px;border:1px solid ${bm.border};border-radius:7px;background:${bm.panel};`;
    const titleBlock = document.createElement("div");
    titleBlock.style.cssText = "display:grid;gap:2px;flex:1;min-width:0;";
    const title = document.createElement("div");
    title.textContent = "IAMCCS_BoardMaker";
    title.style.cssText = `font-size:17px;font-weight:900;color:${bm.text};letter-spacing:0;`;
    const subtitle = document.createElement("div");
    subtitle.textContent = "Script assistant for Shotboard V2/V3 boards: timing, prompts, relay beats first; images are added manually after import.";
    subtitle.style.cssText = `font-size:11px;font-weight:800;color:${bm.muted};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;`;
    titleBlock.append(title, subtitle);
    const actions = document.createElement("div");
    actions.style.cssText = "display:flex;align-items:center;gap:7px;flex-wrap:wrap;justify-content:flex-end;";
    const addBtn = makeBtn("Add Beat", "primary");
    const fitBtn = makeBtn("Fit Duration");
    const styleBtn = makeBtn(boxStyleMode === "paper" ? "Dark Boxes" : "Paper Boxes");
    styleBtn.title = "Switch prompt boxes between the dark assistant style and white Courier New paper boxes.";
    const textSizeWrap = document.createElement("div");
    textSizeWrap.style.cssText = `height:32px;display:flex;align-items:center;gap:5px;padding:0 6px;border:1px solid ${bm.border};border-radius:6px;background:${bm.panelDark};`;
    const textSizeLabel = document.createElement("span");
    textSizeLabel.style.cssText = `min-width:42px;text-align:center;color:${bm.muted};font-size:10px;font-weight:900;`;
    const refreshTextSizeLabel = () => { textSizeLabel.textContent = `${Math.round(boxTextScale * 100)}%`; };
    const textSizeBtn = (label, delta) => {
        const btn = makeBtn(label);
        btn.style.width = "28px";
        btn.style.height = "24px";
        btn.style.padding = "0";
        btn.title = label === "+" ? "Increase prompt text size" : "Decrease prompt text size";
        btn.onclick = () => {
            boxTextScale = Math.max(0.85, Math.min(1.7, Math.round((boxTextScale + delta) * 100) / 100));
            node.properties = node.properties || {};
            node.properties.iamccs_boardmaker_text_scale = boxTextScale;
            refreshTextSizeLabel();
            refreshBoardMakerTextStyles();
            draw();
        };
        return btn;
    };
    refreshTextSizeLabel();
    textSizeWrap.append(textSizeBtn("-", -0.1), textSizeLabel, textSizeBtn("+", 0.1));
    const openEditorBtn = makeBtn("Open Editor");
    openEditorBtn.title = "Open this BoardMaker in a full-frame editor view.";
    const importBtn = makeBtn("Import Board");
    importBtn.title = "Import an existing Shotboard board JSON into BoardMaker for editing.";
    const exportBtn = makeBtn("Export Board", "primary");
    const clearBtn = makeBtn("Reset", "danger");
    const importInput = document.createElement("input");
    importInput.type = "file";
    importInput.accept = ".json,application/json";
    importInput.style.display = "none";
    root.appendChild(importInput);
    actions.append(styleBtn, textSizeWrap, openEditorBtn, importBtn, addBtn, fitBtn, exportBtn, clearBtn);
    head.append(titleBlock, actions);
    root.addEventListener("iamccs:cine-fullscreen", (event) => {
        openEditorBtn.textContent = event.detail?.open ? "Close Editor" : "Open Editor";
    });

    const globalPrompt = makeText(String(globalWidget?.value || ""), (value) => { setWidgetValue(node, "global_prompt", value); }, { multiline: true, rows: 4, height: 88, placeholder: "Global prompt for the whole board...", boxText: true });
    const topGrid = document.createElement("div");
    topGrid.style.cssText = `display:grid;grid-template-columns:minmax(320px,1fr) 190px 130px 150px 150px 170px 230px;gap:12px;margin:9px 0;padding:10px;border:1px solid ${bm.borderSoft};border-radius:7px;background:${bm.panelDark};`;
    const labelWrap = (label, control) => {
        const wrap = document.createElement("label");
        wrap.style.cssText = `display:flex;flex-direction:column;gap:5px;color:${bm.muted};font-size:10px;font-weight:900;min-width:0;`;
        const span = document.createElement("span");
        span.textContent = label;
        wrap.append(span, control);
        return wrap;
    };
    const boardName = makeText(String(boardNameWidget?.value || "iamccs_boardmaker_board"), (value) => { setWidgetValue(node, "board_name", value); }, { placeholder: "board name" });
    const durationCtrl = numberStepperControl(Number(durationWidget?.value || totalRowsDuration() || 9), "0.1", "0.1", null, (value) => {
        durationControlValue = () => Number(value);
        setWidgetValue(node, "duration_seconds", Number(value));
    }, { liveInput: false });
    styleValueControls(durationCtrl);
    const fpsCtrl = numberStepperControl(Number(fpsWidget?.value || 24), "1", "1", null, (value) => {
        fpsControlValue = () => Number(value);
        setWidgetValue(node, "frame_rate", Number(value));
    }, { liveInput: false });
    styleValueControls(fpsCtrl);
    const widthCtrl = numberStepperControl(Number(imageWidthWidget?.value || 768), "32", "64", null, (value) => {
        imageWidthControlValue = () => Number(value);
        setWidgetValue(node, "image_width", Number(value));
    }, { liveInput: false });
    styleValueControls(widthCtrl);
    const heightCtrl = numberStepperControl(Number(imageHeightWidget?.value || 432), "32", "64", null, (value) => {
        imageHeightControlValue = () => Number(value);
        setWidgetValue(node, "image_height", Number(value));
    }, { liveInput: false });
    styleValueControls(heightCtrl);
    const forceCtrl = numberStepperControl(Number(forceWidget?.value || 0.28), "0.01", "0", "1", (value) => {
        defaultForceValue = () => Number(value);
        setWidgetValue(node, "default_force", Number(value));
    }, { liveInput: false });
    styleValueControls(forceCtrl);
    const guidePolicy = makeSelect(String(guidePolicyWidget?.value || "every_checked_row"), ["every_checked_row", "first_last", "all", "none"], (value) => setWidgetValue(node, "guide_policy", value));
    topGrid.append(
        labelWrap("Board name", boardName),
        labelWrap("Total duration", durationCtrl),
        labelWrap("FPS", fpsCtrl),
        labelWrap("Project width", widthCtrl),
        labelWrap("Project height", heightCtrl),
        labelWrap("Default motion", forceCtrl),
        labelWrap("Guide policy", guidePolicy)
    );

    const status = document.createElement("div");
    status.style.cssText = `min-height:34px;display:flex;align-items:center;gap:8px;padding:5px 8px;margin:9px 0;border:1px solid ${bm.border};border-radius:7px;background:${bm.valueBg};color:${bm.muted};font-weight:800;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;`;
    const list = document.createElement("div");
    list.style.cssText = "display:flex;flex-direction:column;gap:8px;max-height:780px;overflow-y:auto;overflow-x:hidden;padding-right:4px;scrollbar-gutter:stable;";
    const promptPanel = document.createElement("div");
    promptPanel.style.cssText = `padding:10px;border:1px solid ${bm.borderSoft};border-radius:7px;background:${bm.panelDark};`;
    promptPanel.appendChild(labelWrap("Global prompt", globalPrompt));

    const draw = () => {
        list.innerHTML = "";
        rows.forEach((row, index) => {
            const card = document.createElement("div");
            card.style.cssText = `display:grid;grid-template-columns:76px 170px minmax(300px,340px) minmax(390px,1fr) minmax(390px,1fr) 92px;gap:12px;align-items:stretch;padding:12px;border:1px solid ${bm.border};border-radius:7px;background:${bm.card}!important;box-shadow:inset 0 1px 0 rgba(255,255,255,.04);`;
            const badge = document.createElement("div");
            badge.textContent = `BEAT\n${String(index + 1).padStart(2, "0")}\nref ${index + 1}`;
            badge.style.cssText = `white-space:pre-line;display:flex;align-items:center;justify-content:center;text-align:center;border:1px solid ${bm.accentBorder};border-radius:6px;background:linear-gradient(180deg,${bm.accent} 0%,${bm.valueBg} 100%);color:${bm.text};font-weight:900;line-height:1.25;`;
            const dur = numberStepperControl(row.duration, "0.1", "0.1", null, (value) => {
                row.duration = Math.max(0.1, Number(value) || 0.1);
                sync();
                status.textContent = `Beats: ${rows.length} | Script duration: ${totalRowsDuration().toFixed(1)}s`;
            }, { liveInput: false });
            dur.style.gridTemplateColumns = "30px minmax(70px,1fr) 30px";
            dur.style.gap = "8px";
            styleValueControls(dur);
            const beatColumn = document.createElement("div");
            beatColumn.style.cssText = "display:grid;grid-template-rows:auto auto;gap:9px;align-content:start;min-width:0;";
            const imageHint = makeText(row.image_hint, (value) => {
                row.image_hint = value;
                sync();
            }, { placeholder: "optional image / ref note" });
            imageHint.style.borderStyle = "dashed";
            imageHint.title = "Optional planning note only. Images can still be added manually after importing this board into Shotboard V2/V3.";
            const imageWrap = labelWrap("Image placeholder", imageHint);
            beatColumn.append(labelWrap("Beat duration", dur), imageWrap);
            const meta = document.createElement("div");
            meta.style.cssText = "display:grid;gap:6px;";
            const labelInput = makeText(row.label, (value) => { row.label = value; sync(); }, { placeholder: "label" });
            const cameraSelect = makeSelect(row.camera, ["slow push-in", "continuous dolly-in", "macro push-in", "tracking shot", "crane descent", "orbit move", "locked-off camera"], (value) => { row.camera = value; sync(); });
            const transitionSelect = makeSelect(row.transition, ["continuous_motion", "match_cut", "soft_morph", "hard_cut"], (value) => { row.transition = value; sync(); });
            const rowForce = numberStepperControl(row.force, "0.01", "0", "1", (value) => {
                row.force = Math.max(0, Math.min(1, Number(value) || 0));
                sync();
            }, { liveInput: false });
            rowForce.style.gridTemplateColumns = "30px minmax(90px,1fr) 30px";
            rowForce.style.gap = "8px";
            styleValueControls(rowForce);
            meta.append(labelWrap("Label", labelInput), labelWrap("Camera", cameraSelect), labelWrap("Transition", transitionSelect), labelWrap("Guide force", rowForce));
            const local = makeText(row.local_prompt, (value) => { row.local_prompt = value; sync(); }, { multiline: true, rows: 6, height: 148, placeholder: "Local prompt for this beat...", boxText: true });
            const bridge = makeText(row.bridge, (value) => { row.bridge = value; sync(); }, { multiline: true, rows: 6, height: 148, placeholder: "Action/bridge to next beat...", boxText: true });
            const rowActions = document.createElement("div");
            rowActions.style.cssText = "display:grid;grid-template-rows:32px 32px 1fr;gap:7px;align-content:start;";
            const duplicate = makeBtn("Copy");
            duplicate.style.padding = "0 8px";
            duplicate.onclick = () => {
                const copy = normalizeMakerRow({ ...row, label: `${row.label || `box_${index + 1}`}_copy` }, index + 1);
                rows.splice(index + 1, 0, copy);
                sync();
                draw();
            };
            const del = makeBtn("x", "danger");
            del.style.height = "32px";
            del.onclick = () => {
                rows.splice(index, 1);
                if (!rows.length) rows = defaultRows();
                sync();
                draw();
            };
            rowActions.append(duplicate, del);
            card.append(badge, beatColumn, meta, labelWrap("Local prompt", local), labelWrap("Relay / bridge to next", bridge), rowActions);
            list.appendChild(card);
        });
        refreshBoardMakerTextStyles();
        status.textContent = `Beats: ${rows.length} | Script duration: ${totalRowsDuration().toFixed(1)}s | Target: ${Number(durationControlValue() || 0).toFixed(1)}s | Export creates a Shotboard V2/V3 board, then images are added manually.`;
    };

    addBtn.onclick = () => {
        rows.push({ label: `beat_${rows.length + 1}`, duration: 3, image_hint: "", local_prompt: "describe this beat with concrete cinematic action", bridge: "carry the same movement into the next visual beat", camera: "continuous dolly-in", transition: "continuous_motion", force: Number(defaultForceValue()) || 0.28, use_guide: true, use_prompt: true });
        sync();
        draw();
    };
    fitBtn.onclick = () => {
        const sum = Number(totalRowsDuration().toFixed(3));
        durationControlValue = () => sum;
        durationCtrl._iamccsSetValue?.(sum);
        setWidgetValue(node, "duration_seconds", sum);
        sync();
        status.textContent = `Total duration fitted to ${sum}s.`;
    };
    clearBtn.onclick = () => {
        rows = defaultRows();
        sync();
        draw();
    };
    styleBtn.onclick = () => {
        boxStyleMode = boxStyleMode === "paper" ? "dark" : "paper";
        node.properties = node.properties || {};
        node.properties.iamccs_boardmaker_box_style = boxStyleMode;
        styleBtn.textContent = boxStyleMode === "paper" ? "Dark Boxes" : "Paper Boxes";
        refreshBoardMakerTextStyles();
        draw();
    };
    openEditorBtn.onclick = () => { toggleFullscreenEditor(root, node); };
    importBtn.onclick = () => importInput.click();
    importInput.onchange = async (event) => {
        const file = event.target.files?.[0];
        importInput.value = "";
        if (!file) return;
        try {
            const raw = await readJsonFile(file);
            const board = boardFromWorkflowJson(raw) || raw || {};
            const settings = board.settings && typeof board.settings === "object" ? board.settings : {};
            const boardValue = (name) => Object.prototype.hasOwnProperty.call(board, name) ? board[name] : settings[name];
            const importedRows = makerRowsFromBoard(board);
            rows = importedRows.length ? importedRows : defaultRows();
            const importedPrompt = board.global_prompt ?? board.prompt ?? board.globalPrompt ?? "";
            globalPrompt.value = String(importedPrompt || "");
            setWidgetValue(node, "global_prompt", globalPrompt.value);
            const importedName = board?.metadata?.board_name || board?.metadata?.name || file.name.replace(/\.json$/i, "") || "iamccs_boardmaker_board";
            boardName.value = String(importedName || "iamccs_boardmaker_board");
            setWidgetValue(node, "board_name", boardName.value);
            const duration = Number(boardValue("duration_seconds") ?? totalRowsDuration());
            if (Number.isFinite(duration) && duration > 0) {
                durationControlValue = () => duration;
                durationCtrl._iamccsSetValue?.(duration);
                setWidgetValue(node, "duration_seconds", duration);
            }
            const fps = Number(boardValue("frame_rate") ?? fpsControlValue());
            if (Number.isFinite(fps) && fps > 0) {
                fpsControlValue = () => fps;
                fpsCtrl._iamccsSetValue?.(fps);
                setWidgetValue(node, "frame_rate", fps);
            }
            const width = Number(boardValue("image_width") ?? imageWidthControlValue());
            if (Number.isFinite(width) && width >= 64) {
                imageWidthControlValue = () => width;
                widthCtrl._iamccsSetValue?.(width);
                setWidgetValue(node, "image_width", width);
            }
            const height = Number(boardValue("image_height") ?? imageHeightControlValue());
            if (Number.isFinite(height) && height >= 64) {
                imageHeightControlValue = () => height;
                heightCtrl._iamccsSetValue?.(height);
                setWidgetValue(node, "image_height", height);
            }
            const force = Number(boardValue("default_force") ?? defaultForceValue());
            if (Number.isFinite(force)) {
                defaultForceValue = () => force;
                forceCtrl._iamccsSetValue?.(force);
                setWidgetValue(node, "default_force", force);
            }
            if (boardValue("guide_policy") != null) {
                guidePolicy.value = String(boardValue("guide_policy"));
                setWidgetValue(node, "guide_policy", guidePolicy.value);
            }
            sync();
            draw();
            status.textContent = `Imported ${rows.length} beats from ${file.name} at ${Number(imageWidthControlValue())}x${Number(imageHeightControlValue())}.`;
        } catch (err) {
            console.error("[IAMCCS_BoardMaker] import failed", err);
            status.textContent = `Import failed: ${err?.message || err}`;
        }
    };
    exportBtn.onclick = async () => {
        try {
            exportBtn.disabled = true;
            const board = collectBoard();
            const filename = safeBoardFilename(board.metadata.board_name || "iamccs_boardmaker_board");
            await saveBoardJsonToChosenFolder(board, filename, (message) => { status.textContent = message; });
        } catch (err) {
            console.error("[IAMCCS_BoardMaker] export failed", err);
            status.textContent = `Export failed: ${err?.message || err}`;
        } finally {
            exportBtn.disabled = false;
        }
    };

    root.append(head, promptPanel, topGrid, status, list);
    const widget = node.addDOMWidget("IAMCCS_BoardMaker", "iamccs_boardmaker", root, { serialize: false });
    widget.computeSize = (width) => [Math.max(1560, Number(width || 1560)), 1160];
    lockNodeMinimumSize(node, [1560, 1200], { lockResize: false, preferredSize: [1560, 1220] });
    sync();
    draw();
}

function renderForNode(node) {
    const klass = nodeClassName(node);
    if (klass !== WAN_NODE_CLASS) return;
    try {
        const title = String(node?.title || "");
        if (klass === "IAMCCS_CineShotboardLite") {
            lockNodeMinimumSize(node, SHOTBOARD_LITE_NODE_MIN_SIZE, { lockResize: true });
        }
        if (isShotboardV3Class(klass)) {
            const collapsed = Boolean(node.properties?.iamccs_v3_collapsed);
            lockNodeMinimumSize(node, [SHOTBOARD_V3_RIGID_WIDTH, collapsed ? SHOTBOARD_V3_COLLAPSED_HEIGHT : SHOTBOARD_V3_OPEN_HEIGHT], { lockResize: false, lockWidth: true });
        }
        if (klass === "IAMCCS_CineShotboardTimelinePro" || klass === "IAMCCS_CineShotboardPlannerPro" || klass === "IAMCCS_CineShotboardPlannerProV2" || klass === "IAMCCS_CineShotboardPlannerProLegacy") {
            lockNodeMinimumSize(node, SHOTBOARD_NODE_MIN_SIZE, { lockResize: true });
        }
        if (klass === "IAMCCS_WanLegacySequencerDisabled") applyCineChrome(node, "flfEngine");
        if (klass === "IAMCCS_CinePromptRelayLatentShapeSync") renderPromptRelayShapeSync(node);
        if (klass === "IAMCCS_WanLegacySequencerDisabled") renderKeyframeEditor(node);
        if (klass === "IAMCCS_CinePromptRelayTimeline") renderPromptRelayEditor(node);
        if (isShotboardV3Class(klass)) {
            if (isWanShotboardV3Class(klass)) applyCineChrome(node, "wanShotboardV3");
            renderShotboardV3(node);
        }
        if (klass === "IAMCCS_CineShotboardLite") renderShotboardLite(node);
        if (klass === "IAMCCS_CineShotboardTimelinePro" || klass === "IAMCCS_CineShotboardPlannerPro" || klass === "IAMCCS_CineShotboardPlannerProV2" || klass === "IAMCCS_CineShotboardPlannerProLegacy") renderShotboardPro(node);
        if (klass === "IAMCCS_CineFLFEngineSimple") renderCineFLFEngineSimple(node);
        if (klass === "IAMCCS_CineInfo") renderCineInfo(node);
        if (klass === "IAMCCS_CinePromptArchitect") renderCinePromptArchitect(node);
        if (klass === "IAMCCS_BoardMaker") renderBoardMaker(node);
        if (klass === "IAMCCS_CineMusicVideoPlanner") renderCineMusicVideoPlanner(node);
    } catch (err) {
        showRenderError(node, err);
    }
}

function scheduleRender(node, options = {}) {
    const klass = nodeClassName(node);
    if (klass !== WAN_NODE_CLASS) return;
    if (Array.isArray(node._iamccsCineRenderTimers)) {
        node._iamccsCineRenderTimers.forEach((timer) => window.clearTimeout(timer));
    }
    const delay = Math.max(0, Number(options.delay ?? 80));
    const secondPass = options.secondPass !== false;
    node._iamccsCineRenderTimers = [
        window.setTimeout(() => renderForNode(node), delay),
    ];
    if (secondPass) {
        node._iamccsCineRenderTimers.push(window.setTimeout(() => renderForNode(node), Math.max(delay + 220, Number(options.secondDelay ?? 450))));
    }
}

function flushAllShotboardV3Timelines(reason = "flush") {
    const nodes = Array.isArray(app.graph?._nodes) ? app.graph._nodes : [];
    let flushed = 0;
    for (const node of nodes) {
        if (nodeClassName(node) !== WAN_NODE_CLASS) continue;
        if (typeof node._iamccsCineShotboardV3WriteTimeline !== "function") continue;
        try {
            node._iamccsCineShotboardV3WriteTimeline({ force: true });
            flushed += 1;
            const text = String(getWidget(node, "timeline_data")?.value || "");
            let firstPrompt = "";
            let containsCoastline = false;
            try {
                const data = JSON.parse(text || "{}");
                firstPrompt = String(data?.segments?.[0]?.prompt || data?.rows?.[0]?.relay_prompt || "");
                containsCoastline = /\bcoastline\b/i.test(text);
            } catch {}
            console.log("[IAMCCS WAN V3 QUEUE FLUSH]", {
                reason,
                nodeId: node?.id,
                timelineLength: text.length,
                containsCoastline,
                firstPrompt: firstPrompt.replace(/\s+/g, " ").slice(0, 220),
            });
        } catch (err) {
            console.warn("[IAMCCS WAN V3 QUEUE FLUSH] failed", { reason, nodeId: node?.id, err });
        }
    }
    return flushed;
}

function wrapQueueFlush(target, methodName, label) {
    if (!target || typeof target[methodName] !== "function") return;
    const guard = `_iamccsWanV3QueueFlushWrapped_${methodName}`;
    if (target[guard]) return;
    target[guard] = true;
    const original = target[methodName];
    target[methodName] = function (...args) {
        flushAllShotboardV3Timelines(label);
        return original.apply(this, args);
    };
}

app.registerExtension({
    name: "iamccs.wan.v3.timeline.ui",
    async setup() {},
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const name = String(nodeData?.name || nodeData?.class_type || "");
        if (name !== WAN_NODE_CLASS) return;
        if (nodeType.prototype._iamccsWanV3TimelineWrapped) return;
        nodeType.prototype._iamccsWanV3TimelineWrapped = true;
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

// Ã¢â€â‚¬Ã¢â€â‚¬ NarrativePlanner push listener Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
// When the NarrativePlanner "Ã¢â€ â€™ Push to PlannerPro" button fires, re-render
// the shotboard so the new rows appear immediately without a page reload.
// By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
document.addEventListener("iamccs:planner_rows_updated", (ev) => {
    const nodeId = ev?.detail?.node_id;
    if (!nodeId) return;
    if (ev?.detail?.render === false) return;
    const plannerNode = app.graph?.getNodeById(nodeId);
    if (!plannerNode) return;
    const klass = String(plannerNode?.comfyClass || plannerNode?.type || "");
    if (klass !== WAN_NODE_CLASS) return;
    // Clear render guard so renderShotboardPro rebuilds the table
    plannerNode._iamccsCineShotboardReady = false;
    plannerNode._iamccsCineShotboardVersion = "";
    plannerNode._iamccsCineShotboardLiteReady = false;
    plannerNode._iamccsCineShotboardV3Ready = false;
    plannerNode._iamccsCineShotboardV3Version = "";
    const fromAudioBoard = ev?.detail?.source === "IAMCCS_AudioBoardArranger";
    scheduleRender(plannerNode, fromAudioBoard ? { delay: 180, secondPass: false } : {});
});




