// IAMCCS AutoLink - Based on KJ nodes SetNode/GetNode
// Nodi puramente frontend - non eseguono niente lato Python

import { app } from "../../scripts/app.js";

const SET_TYPE = "IAMCCS_SetAutoLink";
const GET_TYPE = "IAMCCS_GetAutoLink";
const CONVERTER_TYPE = "IAMCCS_AutoLinkConverter";
const ARGUMENTS_TYPE = "IAMCCS_AutoLinkArguments";
const KJ_SET_TYPE = "SetNode";
const KJ_GET_TYPE = "GetNode";

console.log("[IAMCCS AutoLink] Loading extension...");

// === Graph compatibility helpers ===
// ComfyUI/LiteGraph internals vary across versions. These helpers keep AutoLink
// working when nodes/links are stored differently.
function _iamccsGraphNodes(graph) {
    if (!graph) return [];
    if (Array.isArray(graph._nodes)) return graph._nodes;
    if (Array.isArray(graph.nodes)) return graph.nodes;
    const byId = graph._nodes_by_id || graph._nodesById;
    if (byId && typeof byId === "object") {
        try { return Object.values(byId).filter(Boolean); } catch { return []; }
    }
    return [];
}

function _iamccsGraphLinksEntries(graph) {
    const links = graph?.links;
    if (!links) return [];

    // Map-like
    if (typeof links.entries === "function") {
        try {
            const out = [];
            for (const [k, v] of links.entries()) {
                const nk = Number(k);
                const id = Number.isFinite(nk) ? nk : (Number.isFinite(Number(v?.id)) ? Number(v.id) : nk);
                out.push([id, v]);
            }
            return out;
        } catch { /* fallthrough */ }
    }

    // Array-like
    if (Array.isArray(links)) {
        return links.map(l => [Number(l?.id), l]);
    }

    // Plain object
    if (typeof links === "object") {
        try { return Object.entries(links).map(([k, v]) => [Number(k), v]); } catch { return []; }
    }

    return [];
}

function _iamccsGetLink(graph, linkId) {
    const links = graph?.links;
    if (!links || linkId == null) return null;
    const id = Number(linkId);
    if (!Number.isFinite(id)) return null;

    try {
        if (typeof links.get === "function") {
            return links.get(id) || links.get(String(id)) || null;
        }
    } catch {}

    try {
        return links[id] || links[String(id)] || null;
    } catch {
        return null;
    }
}

function _iamccsDeleteLink(graph, linkId) {
    const links = graph?.links;
    if (!links || linkId == null) return;
    const id = Number(linkId);
    if (!Number.isFinite(id)) return;

    try {
        if (typeof links.delete === "function") {
            links.delete(id);
            links.delete(String(id));
            return;
        }
    } catch {}

    try { delete links[id]; } catch {}
    try { delete links[String(id)]; } catch {}
}

function _iamccsIdEq(a, b) {
    // Robust id comparison across number/string ids.
    if (a == null || b == null) return false;
    const an = Number(a);
    const bn = Number(b);
    if (Number.isFinite(an) && Number.isFinite(bn)) return an === bn;
    return String(a) === String(b);
}

function alphaSort(values) {
    return [...values].sort((a, b) => String(a).localeCompare(String(b), undefined, { sensitivity: 'base' }));
}

function getWidget(node, widgetName) {
    if (!node?.widgets?.length) return null;
    return node.widgets.find(w => w?.name === widgetName || w?.label === widgetName) || null;
}

function getWidgetValue(node, widgetName) {
    return getWidget(node, widgetName)?.value;
}

function isValidAutolinkKey(value) {
    const s = String(value ?? "").trim();
    return !!s && s !== "*";
}

function getAutolinkKey(node) {
    const v = getWidgetValue(node, "name");
    if (isValidAutolinkKey(v)) return String(v).trim();
    const outName = node?.outputs?.[0]?.name;
    if (isValidAutolinkKey(outName)) return String(outName).trim();
    const inName = node?.inputs?.[0]?.name;
    if (isValidAutolinkKey(inName)) return String(inName).trim();
    return "";
}

function setAutolinkKeyAndTitle(node, key) {
    const safeKey = String(key ?? "").trim();
    if (!node || !isValidAutolinkKey(safeKey)) return;

    setWidgetValue(node, "name", safeKey);
    node.title = safeKey;

    if (node.type === SET_TYPE) {
        if (node.inputs?.[0]) node.inputs[0].name = safeKey;
        if (node.outputs?.[0]) node.outputs[0].name = safeKey;
    }

    const nameWidget = getWidget(node, "name");
    if (nameWidget) nameWidget.lastValue = safeKey;
}

function normalizeAutolinkIOSlots(graph, node, { wantInputs = 0, wantOutputs = 0 } = {}) {
    if (!node) return;

    // Ensure arrays exist
    if (!node.inputs) node.inputs = [];
    if (!node.outputs) node.outputs = [];

    // Ensure at least one slot exists when requested
    if (wantInputs > 0 && node.inputs.length === 0 && typeof node.addInput === "function") {
        node.addInput("*", "*");
    }
    if (wantOutputs > 0 && node.outputs.length === 0 && typeof node.addOutput === "function") {
        node.addOutput("*", "*");
    }

    // If multiple inputs got serialized (buggy old workflows), try to move any link onto slot 0
    if (wantInputs > 0 && node.inputs.length > 1) {
        try {
            const in0HasLink = !!node.inputs?.[0]?.link;
            if (!in0HasLink) {
                const idx = node.inputs.findIndex((inp, i) => i > 0 && inp?.link != null);
                if (idx > 0) {
                    const linkId = node.inputs[idx].link;
                    const link = _iamccsGetLink(graph, linkId);
                    if (link) {
                        try { node.disconnectInput?.(idx); } catch {}
                        const src = getNodeById(graph, link.origin_id);
                        if (src) {
                            try {
                                // Reconnect to slot 0
                                src.connect(link.origin_slot, node, 0);
                            } catch {}
                        }
                    }
                }
            }
        } catch {}

        // Remove extra input slots (keep only slot 0)
        try {
            while (node.inputs.length > 1) {
                if (typeof node.removeInput === "function") node.removeInput(1);
                else node.inputs.splice(1, 1);
            }
        } catch {}
    }

    // Same for outputs: if multiple outputs exist, try to migrate links to slot 0
    if (wantOutputs > 0 && node.outputs.length > 1) {
        try {
            const out0Links = node.outputs?.[0]?.links;
            const out0HasLinks = Array.isArray(out0Links) && out0Links.length > 0;
            if (!out0HasLinks) {
                for (let i = 1; i < node.outputs.length; i++) {
                    const links = node.outputs?.[i]?.links;
                    if (!Array.isArray(links) || links.length === 0) continue;
                    for (const linkId of [...links]) {
                        const link = _iamccsGetLink(graph, linkId);
                        if (!link) continue;
                        const dst = getNodeById(graph, link.target_id);
                        if (!dst) continue;
                        const ts = link.target_slot;
                        try { dst.disconnectInput?.(ts); } catch {}
                        try { node.connect(0, dst, ts); } catch {}
                    }
                }
            }
        } catch {}

        // Remove extra output slots (keep only slot 0)
        try {
            while (node.outputs.length > 1) {
                if (typeof node.removeOutput === "function") node.removeOutput(1);
                else node.outputs.splice(1, 1);
            }
        } catch {}
    }
}

function makeUniqueAutolinkSetName(graph, desired) {
    const raw = String(desired ?? "").trim();
    let base = raw;

    // sanitize
    base = base.replace(/\s+/g, "_");
    base = base.replace(/^\*+|\*+$/g, "");
    base = base.trim();
    if (!isValidAutolinkKey(base)) base = "output";

    // prefer lower-case keys for readability
    base = String(base).trim();
    const baseLower = base.toLowerCase();

    const used = new Set(
        _iamccsGraphNodes(graph)
            .filter(n => n?.type === SET_TYPE)
            .map(n => getAutolinkKey(n))
            .filter(v => isValidAutolinkKey(v))
    );

    // If already has _N suffix, keep it unless it collides.
    const m = baseLower.match(/^(.+?)_(\d+)$/);
    if (m) {
        const stem = m[1];
        let n = parseInt(m[2], 10);
        let candidate = `${stem}_${n}`;
        while (used.has(candidate)) {
            n++;
            candidate = `${stem}_${n}`;
        }
        return candidate;
    }

    // Always add _0, _1, ... (matches convertAllLinks behavior)
    let n = 0;
    let candidate = `${baseLower}_${n}`;
    while (used.has(candidate)) {
        n++;
        candidate = `${baseLower}_${n}`;
    }
    return candidate;
}

function parseHexColor(hex) {
    if (!hex || typeof hex !== 'string') return null;
    const s = hex.trim();
    if (!s.startsWith('#')) return null;
    const h = s.slice(1);
    if (h.length === 3) {
        const r = parseInt(h[0] + h[0], 16);
        const g = parseInt(h[1] + h[1], 16);
        const b = parseInt(h[2] + h[2], 16);
        if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) return null;
        return { r, g, b };
    }
    if (h.length === 6) {
        const r = parseInt(h.slice(0, 2), 16);
        const g = parseInt(h.slice(2, 4), 16);
        const b = parseInt(h.slice(4, 6), 16);
        if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) return null;
        return { r, g, b };
    }
    return null;
}

function getTitleTextColor(mode, bgcolorHex) {
    const m = String(mode || 'White');
    if (m === 'Black') return '#000000';
    if (m === 'White') return '#ffffff';
    // Auto
    const rgb = parseHexColor(bgcolorHex);
    if (!rgb) return '#ffffff';
    // relative luminance-ish (sufficient for UI contrast)
    const lum = (0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b) / 255;
    return lum > 0.55 ? '#000000' : '#ffffff';
}

function applyNodeTitleTextColor(node, mode) {
    if (!node) return;
    const color = getTitleTextColor(mode, node.bgcolor);
    // LiteGraph variants (best-effort)
    node.title_text_color = color;
    node.titleTextColor = color;
    // Some variants read textcolor for title text too
    node.textcolor = color;
    node.textColor = color;
    node.properties = node.properties || {};
    node.properties.autolink_title_color = color;
}

let AUTOLINK_TITLE_COLOR_HOOK_INSTALLED = false;
function installAutolinkTitleColorCanvasHook() {
    if (AUTOLINK_TITLE_COLOR_HOOK_INSTALLED) return;
    try {
        const LGraphCanvas = window?.LGraphCanvas;
        const LiteGraph = window?.LiteGraph;
        if (!LGraphCanvas?.prototype?.drawNode || !LiteGraph) return;

        const originalDrawNode = LGraphCanvas.prototype.drawNode;
        if (originalDrawNode?.__iamccs_autolink_title_hook) {
            AUTOLINK_TITLE_COLOR_HOOK_INSTALLED = true;
            return;
        }

        function wrappedDrawNode(node, ctx) {
            const isAutolink = node && (node.type === SET_TYPE || node.type === GET_TYPE);
            if (!isAutolink) return originalDrawNode.apply(this, arguments);

            const desired = node?.properties?.autolink_title_color || node?.title_text_color || node?.textcolor;
            if (!desired) return originalDrawNode.apply(this, arguments);

            // Some ComfyUI/LiteGraph builds force title text to white on "dark" nodes.
            // To respect ColorTitles (White/Black/Auto) without drawing a second title,
            // we temporarily wrap ctx.fillText and override fillStyle only for the title draw call.
            const prevFillText = ctx?.fillText;
            let titleA = "";
            let titleB = "";
            try {
                const rawTitle = (typeof node?.getTitle === "function") ? node.getTitle() : node?.title;
                titleA = String(rawTitle ?? "");
                titleB = String(node?.title ?? "");
                if (node?.pinned) {
                    if (titleA) titleA += "📌";
                    if (titleB) titleB += "📌";
                }
            } catch {
                // ignore
            }

            if (typeof prevFillText === "function") {
                ctx.fillText = function(text, x, y, maxWidth) {
                    try {
                        const t = String(text ?? "");
                        const isTitle = (t && (t === titleA || t === titleB || (titleA && titleA.startsWith(t)) || (titleB && titleB.startsWith(t))));
                        // title is drawn in the title bar area (usually negative y)
                        if (isTitle && typeof y === "number" && y < 0) {
                            const prevStyle = ctx.fillStyle;
                            ctx.fillStyle = desired;
                            try {
                                return prevFillText.apply(this, arguments);
                            } finally {
                                ctx.fillStyle = prevStyle;
                            }
                        }
                    } catch {
                        // ignore
                    }
                    return prevFillText.apply(this, arguments);
                };
            }

            const prevNodeTitleColor = LiteGraph.NODE_TITLE_COLOR;
            const prevSelectedTitleColor = LiteGraph.NODE_SELECTED_TITLE_COLOR;
            const prevNodeTextColor = LiteGraph.NODE_TEXT_COLOR;
            const prevSelectedTextColor = LiteGraph.NODE_SELECTED_TEXT_COLOR;
            if (typeof LiteGraph.NODE_TITLE_COLOR !== "undefined") LiteGraph.NODE_TITLE_COLOR = desired;
            if (typeof LiteGraph.NODE_SELECTED_TITLE_COLOR !== "undefined") LiteGraph.NODE_SELECTED_TITLE_COLOR = desired;
            if (typeof LiteGraph.NODE_TEXT_COLOR !== "undefined") LiteGraph.NODE_TEXT_COLOR = desired;
            if (typeof LiteGraph.NODE_SELECTED_TEXT_COLOR !== "undefined") LiteGraph.NODE_SELECTED_TEXT_COLOR = desired;

            try {
                return originalDrawNode.apply(this, arguments);
            } finally {
                if (typeof prevFillText === "function") ctx.fillText = prevFillText;
                if (typeof prevNodeTitleColor !== "undefined") LiteGraph.NODE_TITLE_COLOR = prevNodeTitleColor;
                if (typeof prevSelectedTitleColor !== "undefined") LiteGraph.NODE_SELECTED_TITLE_COLOR = prevSelectedTitleColor;
                if (typeof prevNodeTextColor !== "undefined") LiteGraph.NODE_TEXT_COLOR = prevNodeTextColor;
                if (typeof prevSelectedTextColor !== "undefined") LiteGraph.NODE_SELECTED_TEXT_COLOR = prevSelectedTextColor;
            }
        }
        wrappedDrawNode.__iamccs_autolink_title_hook = true;
        LGraphCanvas.prototype.drawNode = wrappedDrawNode;
        AUTOLINK_TITLE_COLOR_HOOK_INSTALLED = true;
    } catch (e) {
        console.warn("[IAMCCS AutoLink] Failed to install title color hook", e);
    }
}

function getCurrentArgumentsNode() {
    try {
        return app?.graph?._nodes?.find(n => n?.type === ARGUMENTS_TYPE) || null;
    } catch {
        return null;
    }
}

function getCurrentColorTitlesMode() {
    const argNode = getCurrentArgumentsNode();
    return argNode?.properties?.color_titles || "White";
}

const AUTOLINK_COLORS = {
    Gray:    { set: { color: "#1f1f1f", bgcolor: "#3a3a3a" }, get: { color: "#2a2a2a", bgcolor: "#555555" } },
    Blue:    { set: { color: "#1b4669", bgcolor: "#29699c" }, get: { color: "#234f73", bgcolor: "#347cb8" } },
    Green:   { set: { color: "#1f5a3a", bgcolor: "#2d7d52" }, get: { color: "#2a6b46", bgcolor: "#3aa66c" } },
    Red:     { set: { color: "#6a1b1b", bgcolor: "#9c2929" }, get: { color: "#7a2323", bgcolor: "#b83434" } },
    Orange:  { set: { color: "#6b3e1a", bgcolor: "#9c5b29" }, get: { color: "#7a4a22", bgcolor: "#b86c34" } },
    Purple:  { set: { color: "#3f1b69", bgcolor: "#5e299c" }, get: { color: "#4a237a", bgcolor: "#7034b8" } },
    Yellow:  { set: { color: "#6b651a", bgcolor: "#9c9229" }, get: { color: "#7a7422", bgcolor: "#b8aa34" } },
    Teal:    { set: { color: "#1b6961", bgcolor: "#299c90" }, get: { color: "#237a71", bgcolor: "#34b8aa" } },
    Pink:    { set: { color: "#691b46", bgcolor: "#9c2969" }, get: { color: "#7a2351", bgcolor: "#b8347c" } },
};

function getAutolinkColorPreset(colorName, role, separateCol, colorGetName) {
    const safeRole = role === 'get' ? 'get' : 'set';
    const base = AUTOLINK_COLORS[colorName] || AUTOLINK_COLORS.Gray;
    if (!separateCol) return base.set; // stessa identità colore per set/get

    if (safeRole === 'get' && colorGetName && AUTOLINK_COLORS[colorGetName]) {
        return AUTOLINK_COLORS[colorGetName].get;
    }

    return base[safeRole];
}

function applyNodeColors(node, preset) {
    if (!node || !preset) return;
    node.color = preset.color;
    node.bgcolor = preset.bgcolor;
}

function recolorExistingAutoLinks(graph, colorSetName = "Gray", separateCol = false, colorGetName = "Gray", colorTitles = "White") {
    if (!graph) return;
    const nodes = _iamccsGraphNodes(graph);
    const sets = nodes.filter(n => n?.type === SET_TYPE);
    const gets = nodes.filter(n => n?.type === GET_TYPE);

    for (const setNode of sets) {
        // Migrazione: vecchi workflow (prima del widget colore) possono aver scritto il nome nel widget colore.
        const maybeName = getWidgetValue(setNode, "name");
        const maybeColor = getWidgetValue(setNode, "AutoLinkColor");
        if ((!maybeName || !String(maybeName).trim()) && maybeColor && !AUTOLINK_COLORS[maybeColor]) {
            const migrated = String(maybeColor).trim();
            setAutolinkKeyAndTitle(setNode, migrated);
            setWidgetValue(setNode, "AutoLinkColor", "Gray");
            setNode.properties = setNode.properties || {};
            setNode.properties.autolink_color_name = "Gray";
        }

        const key = getAutolinkKey(setNode);

        setNode.properties = setNode.properties || {};
        if (setNode.properties.autolink_color_locked === undefined) {
            setNode.properties.autolink_color_locked = false;
        }

        const locked = !!setNode.properties.autolink_color_locked;
        const chosen = locked
            ? (setNode.properties.autolink_color_name || colorSetName)
            : colorSetName;

        // mantieni la sorgente di verità sempre in autolink_color_name
        setNode.properties.autolink_color_name = chosen;
        applyNodeColors(setNode, getAutolinkColorPreset(chosen, 'set', separateCol, colorGetName));
        applyNodeTitleTextColor(setNode, colorTitles);
    }

    for (const getNode of gets) {
        const key = getAutolinkKey(getNode);
        const chosen = key
            ? (nodes.find(n => n?.type === SET_TYPE && getAutolinkKey(n) === key)?.properties?.autolink_color_name || colorSetName)
            : colorSetName;
        applyNodeColors(getNode, getAutolinkColorPreset(chosen, 'get', separateCol, colorGetName));
        getNode.properties = getNode.properties || {};
        getNode.properties.autolink_color_name = chosen;
        applyNodeTitleTextColor(getNode, colorTitles);
    }

    graph.setDirtyCanvas(true, true);
}

function applyColorToAutolinkKey(graph, key, colorName, separateCol = false, colorGetName = "Gray", colorTitles = "White") {
    if (!graph || !key) return;
    const safeKey = String(key).trim();
    const nodes = _iamccsGraphNodes(graph);
    const sets = nodes.filter(n => n?.type === SET_TYPE && getAutolinkKey(n) === safeKey);
    const gets = nodes.filter(n => n?.type === GET_TYPE && getAutolinkKey(n) === safeKey);

    for (const setNode of sets) {
        setNode.properties = setNode.properties || {};
        setNode.properties.autolink_color_name = colorName;
        setNode.properties.autolink_color_locked = true;
        applyNodeColors(setNode, getAutolinkColorPreset(colorName, 'set', separateCol, colorGetName));
        applyNodeTitleTextColor(setNode, colorTitles);
    }
    for (const getNode of gets) {
        getNode.properties = getNode.properties || {};
        getNode.properties.autolink_color_name = colorName;
        applyNodeColors(getNode, getAutolinkColorPreset(colorName, 'get', separateCol, colorGetName));
        applyNodeTitleTextColor(getNode, colorTitles);
    }

    graph.setDirtyCanvas(true, true);
}

// === VISUAL FLOW TRACER ===
let flowTracerEnabled = false;
let flowCanvas = null;
let animationFrameId = null;
let activeFlows = [];

function toggleFlowTracer(enabled) {
    flowTracerEnabled = enabled;
    
    if (enabled && !flowCanvas) {
        flowCanvas = document.createElement('canvas');
        flowCanvas.id = 'autolink-flow-overlay';
        flowCanvas.style.position = 'absolute';
        flowCanvas.style.top = '0';
        flowCanvas.style.left = '0';
        flowCanvas.style.pointerEvents = 'none';
        flowCanvas.style.zIndex = '999';
        document.body.appendChild(flowCanvas);
        
        const resizeCanvas = () => {
            flowCanvas.width = window.innerWidth;
            flowCanvas.height = window.innerHeight;
        };
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);
        
        startFlowAnimation();
        console.log("[IAMCCS AutoLink] ✨ Flow Tracer enabled");
    } else if (!enabled && flowCanvas) {
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
        flowCanvas.remove();
        flowCanvas = null;
        activeFlows = [];
        console.log("[IAMCCS AutoLink] Flow Tracer disabled");
    }
}

function startFlowAnimation() {
    if (!flowTracerEnabled || !flowCanvas) return;
    
    const ctx = flowCanvas.getContext('2d');
    ctx.clearRect(0, 0, flowCanvas.width, flowCanvas.height);
    
    // Trova tutte le coppie Set-Get attive
    activeFlows = [];
    const nodes = _iamccsGraphNodes(app.graph);
    const setNodes = nodes.filter(n => n.type === SET_TYPE);
    const getNodes = nodes.filter(n => n.type === GET_TYPE);
    
    setNodes.forEach(setNode => {
        const setName = getAutolinkKey(setNode);
        if (!setName) return;
        
        const matchingGets = getNodes.filter(g => getAutolinkKey(g) === setName);
        matchingGets.forEach(getNode => {
            const dataType = setNode.outputs[0]?.type || '*';
            activeFlows.push({
                setNode: setNode,
                getNode: getNode,
                dataType: dataType,
                particles: initParticles(5)
            });
        });
    });
    
    animateFlows(ctx);
}

function initParticles(count) {
    const particles = [];
    for (let i = 0; i < count; i++) {
        particles.push({
            progress: Math.random(),
            speed: 0.005 + Math.random() * 0.01,
            size: 3 + Math.random() * 3
        });
    }
    return particles;
}

function animateFlows(ctx) {
    if (!flowTracerEnabled || !flowCanvas) return;
    
    ctx.clearRect(0, 0, flowCanvas.width, flowCanvas.height);
    
    activeFlows.forEach(flow => {
        const setPos = getNodeScreenPosition(flow.setNode);
        const getPos = getNodeScreenPosition(flow.getNode);
        const color = getTypeColor(flow.dataType);
        
        // Disegna linea di connessione
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.3;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(setPos.x, setPos.y);
        ctx.lineTo(getPos.x, getPos.y);
        ctx.stroke();
        ctx.setLineDash([]);
        
        // Anima particelle
        flow.particles.forEach(particle => {
            particle.progress += particle.speed;
            if (particle.progress > 1) particle.progress = 0;
            
            const x = setPos.x + (getPos.x - setPos.x) * particle.progress;
            const y = setPos.y + (getPos.y - setPos.y) * particle.progress;
            
            ctx.globalAlpha = 0.8;
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(x, y, particle.size, 0, Math.PI * 2);
            ctx.fill();
        });
    });
    
    ctx.globalAlpha = 1.0;
    animationFrameId = requestAnimationFrame(() => animateFlows(ctx));
}

function getNodeScreenPosition(node) {
    const canvasElement = document.querySelector('.litegraph');
    if (!canvasElement || !app.canvas) {
        return { x: 0, y: 0 };
    }
    
    const rect = canvasElement.getBoundingClientRect();
    const scale = app.canvas.ds.scale;
    const offset = app.canvas.ds.offset;
    
    const x = rect.left + (node.pos[0] + node.size[0] / 2) * scale + offset[0] * scale;
    const y = rect.top + (node.pos[1] + node.size[1] / 2) * scale + offset[1] * scale;
    
    return { x, y };
}

function getTypeColor(dataType) {
    const colors = {
        'MODEL': '#4a9eff',
        'VAE': '#ff4a9e',
        'CLIP': '#4aff9e',
        'IMAGE': '#ffff4a',
        'LATENT': '#ff9e4a',
        'CONDITIONING': '#9e4aff',
        'MASK': '#ff6b6b',
        'FLOAT': '#6bffa3',
        'INT': '#a3c6ff'
    };
    return colors[dataType] || '#ffffff';
}

function _iamccsFixLinkIntegrity(graph) {
    try {
        if (!graph || !graph.links) return;

        // 1) Ensure each graph.links entry is reflected in origin.outputs[*].links and target.inputs[*].link
        for (const [linkId, link] of _iamccsGraphLinksEntries(graph)) {
            if (!link || !Number.isFinite(linkId)) continue;

            const origin = getNodeById(graph, link.origin_id);
            const target = getNodeById(graph, link.target_id);
            const os = Number(link.origin_slot);
            const ts = Number(link.target_slot);
            if (!origin || !target || !Number.isFinite(os) || !Number.isFinite(ts)) continue;

            // Origin
            try {
                origin.outputs = origin.outputs || [];
                const out = origin.outputs[os];
                if (out) {
                    if (!Array.isArray(out.links)) out.links = [];
                    if (!out.links.includes(linkId)) out.links.push(linkId);
                }
            } catch {}

            // Target
            try {
                target.inputs = target.inputs || [];
                const inp = target.inputs[ts];
                if (inp) {
                    // Never overwrite an existing target link: if this link isn't the one
                    // referenced by the input, it's an orphan/duplicate and should be removed.
                    if (inp.link == null) {
                        inp.link = linkId;
                    } else if (Number(inp.link) !== linkId) {
                        try {
                            // Remove orphan link from origin output list
                            const o2 = getNodeById(graph, link.origin_id);
                            const os2 = Number(link.origin_slot);
                            if (o2?.outputs?.[os2]?.links && Array.isArray(o2.outputs[os2].links)) {
                                o2.outputs[os2].links = o2.outputs[os2].links.filter(x => Number(x) !== linkId);
                            }
                        } catch {}
                        _iamccsDeleteLink(graph, linkId);
                        continue;
                    }
                }
            } catch {}
        }

        // 2) Remove stale link IDs from outputs[*].links that don't exist in graph.links
        const existing = new Set(_iamccsGraphLinksEntries(graph).map(([id]) => id).filter(Number.isFinite));
        for (const node of _iamccsGraphNodes(graph)) {
            if (!node?.outputs?.length) continue;
            for (const out of node.outputs) {
                if (!out || !Array.isArray(out.links) || out.links.length === 0) continue;
                out.links = out.links.filter(id => existing.has(Number(id)));
            }
        }
    } catch (e) {
        console.warn("[IAMCCS AutoLink] Link integrity fix failed", e);
    }
}

function _iamccsForceRemoveLink(graph, linkId) {
    if (!graph || !graph.links || linkId == null) return;
    const id = Number(linkId);
    if (!Number.isFinite(id)) return;
    const link = _iamccsGetLink(graph, id);
    if (!link) return;

    // Remove from origin output links array if possible
    try {
        const origin = getNodeById(graph, link.origin_id);
        const os = Number(link.origin_slot);
        if (origin?.outputs?.[os]?.links && Array.isArray(origin.outputs[os].links)) {
            origin.outputs[os].links = origin.outputs[os].links.filter(x => Number(x) !== id);
        }
    } catch {}

    // Remove from target input pointer if it points to this link
    try {
        const target = getNodeById(graph, link.target_id);
        const ts = Number(link.target_slot);
        if (target?.inputs?.[ts] && Number(target.inputs[ts].link) === id) {
            target.inputs[ts].link = null;
        }
    } catch {}

    try {
        // Prefer native removal if present
        if (typeof graph.removeLink === "function") {
            graph.removeLink(id);
            return;
        }
    } catch {}

    _iamccsDeleteLink(graph, id);
}

function _iamccsDisconnectTargetInput(graph, dstNode, ts) {
    try {
        const slot = Number(ts);
        if (!Number.isFinite(slot) || !dstNode) return;
        const prevId = dstNode?.inputs?.[slot]?.link;

        try { dstNode.disconnectInput?.(slot); } catch {}

        const still = dstNode?.inputs?.[slot]?.link;
        if (still != null) {
            // If disconnect failed, force-remove whatever link the input points to
            _iamccsForceRemoveLink(graph, still);
            try {
                if (dstNode?.inputs?.[slot]) dstNode.inputs[slot].link = null;
            } catch {}
        } else if (prevId != null) {
            // Some implementations clear input.link but keep graph.links/origin.outputs stale
            _iamccsForceRemoveLink(graph, prevId);
        }
    } catch {}
}

function _iamccsRemoveOtherLinksToTarget(graph, targetId, targetSlot, keepLinkId) {
    try {
        const ts = Number(targetSlot);
        const keep = keepLinkId != null ? Number(keepLinkId) : null;
        if (!graph?.links || !Number.isFinite(ts)) return;

        for (const [id, link] of _iamccsGraphLinksEntries(graph)) {
            if (!Number.isFinite(id) || !link) continue;
            if (keep != null && id === keep) continue;
            if (_iamccsIdEq(link.target_id, targetId) && Number(link.target_slot) === ts) {
                _iamccsForceRemoveLink(graph, id);
            }
        }
    } catch {}
}

// === CSS FIXES ===
// Fix per titoli che escono dai nodi contratti
const style = document.createElement('style');
style.textContent = `
    .litegraph .node.collapsed .title {
        max-width: calc(100% - 40px);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        display: inline-block;
    }
`;
document.head.appendChild(style);

// === CONVERTER NODE ===
app.registerExtension({
    name: "iamccs.autolink.converter",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Ensure native title coloring works without overlay text
        installAutolinkTitleColorCanvasHook();

        if (nodeData?.name === CONVERTER_TYPE) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                this.addWidget("button", "🔗 Convert All Links", null, () => {
                    const config = getBlacklistFromInput(this);
                    convertAllLinks(
                        app.graph,
                        config.blacklist,
                        config.blacklistTypes,
                        config.blacklistNodeModes,
                        config.includeKijNodes,
                        config.groupExclude,
                        config.groupInOutExclude,
                        config.alignMode,
                        config.packingMode,
                        config.colorSet,
                        config.colorGet,
                        config.separateCol,
                        config.colorTitles
                    );
                });
                
                this.addWidget("button", "↩️ Restore Direct Links", null, () => {
                    restoreDirectLinks(app.graph);
                });
                
                return result;
            };
        }
        
        if (nodeData?.name === ARGUMENTS_TYPE) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                if (!this.properties) this.properties = {};
                if (!this.properties.blacklist) this.properties.blacklist = [];
                if (!this.properties.blacklist_types) this.properties.blacklist_types = [];
                if (!this.properties.blacklist_node_modes) this.properties.blacklist_node_modes = {};
                if (this.properties.blacklist_add_mode === undefined) this.properties.blacklist_add_mode = "both";
                if (this.properties.blacklist_pending_value === undefined) this.properties.blacklist_pending_value = "";
                if (this.properties.blacklist_view_selected === undefined) this.properties.blacklist_view_selected = "";
                if (this.properties.include_kijnodes === undefined) this.properties.include_kijnodes = false;
                if (this.properties.flow_tracer === undefined) this.properties.flow_tracer = false;
                if (this.properties.all_nodes_sel === undefined) this.properties.all_nodes_sel = false;
                if (this.properties.align_mode === undefined) this.properties.align_mode = "TopToDown";
                if (this.properties.packing_mode === undefined) this.properties.packing_mode = "AvoidAll";
                if (this.properties.autolink_color_set === undefined) this.properties.autolink_color_set = "Gray";
                if (this.properties.autolink_color_get === undefined) this.properties.autolink_color_get = "Gray";
                if (this.properties.separate_col === undefined) this.properties.separate_col = false;
                if (this.properties.color_titles === undefined) this.properties.color_titles = "White";
                if (this.properties.group_inout_exclude === undefined) this.properties.group_inout_exclude = "None";
                // Backward compat: old typo key was group_exlude
                if (this.properties.group_exclude === undefined) {
                    if (this.properties.group_exlude !== undefined) {
                        this.properties.group_exclude = this.properties.group_exlude;
                    } else {
                        this.properties.group_exclude = false;
                    }
                }

                const addDivider = () => {
                    // Inert divider (non-interactive, non-serializzato). Serve solo come separazione visiva.
                    const divider = {
                        type: "iamccs_divider",
                        name: "",
                        value: "",
                        options: { serialize: false },
                        computeSize: function (width) {
                            const w = Number.isFinite(width) ? width : 200;
                            return [Math.max(40, w), 10];
                        },
                        draw: function (ctx, node, widget_width, y, H) {
                            try {
                                ctx.save();
                                const margin = 10;
                                const w = Number.isFinite(widget_width) ? widget_width : (node?.size?.[0] || 200);
                                const x1 = margin;
                                const x2 = Math.max(margin + 10, w - margin);
                                const yy = y + Math.floor((H || 10) / 2);

                                ctx.globalAlpha = 0.35;
                                ctx.strokeStyle = "rgba(128,128,128,0.85)";
                                ctx.lineWidth = 1;
                                ctx.beginPath();
                                ctx.moveTo(x1, yy);
                                ctx.lineTo(x2, yy);
                                ctx.stroke();
                                ctx.restore();
                            } catch (e) {
                                // mute
                            }
                        },
                        mouse: function () {
                            // Non cattura i click: resta "muto" e permette drag del nodo.
                            return false;
                        },
                    };

                    if (typeof this.addCustomWidget === "function") {
                        this.addCustomWidget(divider);
                    }
                };

                // Normalize per-node modes: every blacklisted node must have a mode, default = both
                try {
                    this.properties.blacklist_node_modes = this.properties.blacklist_node_modes || {};
                    const ids = new Set((this.properties.blacklist || []).map(v => Number(v)).filter(v => Number.isFinite(v)));
                    for (const id of ids) {
                        const k = String(id);
                        if (!this.properties.blacklist_node_modes[k]) this.properties.blacklist_node_modes[k] = "both";
                    }
                    for (const k of Object.keys(this.properties.blacklist_node_modes)) {
                        const id = Number(k);
                        if (!ids.has(id)) delete this.properties.blacklist_node_modes[k];
                    }
                } catch (e) {
                    console.warn("[IAMCCS AutoLink] blacklist_node_modes normalize failed", e);
                }
                
                // Toggle per Visual Flow Tracer
                this.addWidget(
                    "toggle",
                    "visual_flow_tracer",
                    this.properties.flow_tracer,
                    (value) => {
                        this.properties.flow_tracer = value;
                        toggleFlowTracer(value);
                        console.log("[IAMCCS AutoLink] Flow Tracer:", value);
                    }
                );
                
                // Toggle per mostrare tutti i nodi singolarmente nella blacklist
                this.addWidget(
                    "toggle",
                    "all_nodes_sel",
                    this.properties.all_nodes_sel,
                    (value) => {
                        this.properties.all_nodes_sel = value;
                        console.log("[IAMCCS AutoLink] Show all nodes individually:", value);
                    }
                );
                
                // Toggle per includere KijNodes nella conversione
                this.addWidget(
                    "toggle",
                    "include_kijnodes",
                    this.properties.include_kijnodes,
                    (value) => {
                        this.properties.include_kijnodes = value;
                        console.log("[IAMCCS AutoLink] Include KijNodes:", value);
                    }
                );

                // Toggle: esclude i link interni allo stesso Group (ma non quelli che attraversano il confine)
                this.addWidget(
                    "toggle",
                    "GroupExclude",
                    this.properties.group_exclude,
                    (value) => {
                        this.properties.group_exclude = value;
                        // keep legacy key in sync
                        this.properties.group_exlude = value;
                        console.log("[IAMCCS AutoLink] GroupExclude:", value);
                    }
                );

                // Dropdown: filtra i link che entrano/escono dai group
                this.addWidget(
                    "combo",
                    "GroupInOutExclude",
                    this.properties.group_inout_exclude,
                    (value) => {
                        this.properties.group_inout_exclude = value;
                        console.log("[IAMCCS AutoLink] GroupInOutExclude:", value);
                    },
                    {
                        values: () => ["None", "ExcludeEnter", "ExcludeExit", "ExcludeBoth"]
                    }
                );

                // Divider: GroupInOutExclude -> Align
                addDivider();

                // Dropdown: modalità di allineamento/packing per Set/Get creati automaticamente
                this.addWidget(
                    "combo",
                    "align_mode",
                    this.properties.align_mode,
                    (value) => {
                        this.properties.align_mode = value;
                        console.log("[IAMCCS AutoLink] Align mode:", value);

                        // Re-layout esistente: permette di riallineare dopo che gli AutoLink sono stati creati
                        try {
                            relayoutExistingAutoLinks(app.graph, value, this.properties.packing_mode);
                        } catch (e) {
                            console.error("[IAMCCS AutoLink] Relayout error:", e);
                        }
                    },
                    {
                        values: () => [
                            "TopToDown",
                            "BottomToTop",
                            "CenterUpDown",
                            "CenterDownUp",
                            "Proportional",
                            "AlignX_Right",
                            "AlignX_Left",
                            "Columns_Down",
                            "Columns_Up",
                            "Rake_Down",
                            "Rake_Up",
                        ]
                    }
                );

                // Dropdown: regole overlap/packing (operativo anche post conversione)
                this.addWidget(
                    "combo",
                    "packing_mode",
                    this.properties.packing_mode,
                    (value) => {
                        this.properties.packing_mode = value;
                        console.log("[IAMCCS AutoLink] Packing mode:", value);

                        try {
                            relayoutExistingAutoLinks(app.graph, this.properties.align_mode, value);
                        } catch (e) {
                            console.error("[IAMCCS AutoLink] Relayout error:", e);
                        }
                    },
                    {
                        values: () => [
                            "AvoidAll",
                            "AvoidNonAutoLink",
                        ]
                    }
                );

                // Divider: Packing -> SeparateCol
                addDivider();

                // Toggle + dropdown colori AutoLink (operativo anche post conversione)
                this.addWidget(
                    "toggle",
                    "SeparateCol",
                    this.properties.separate_col,
                    (value) => {
                        this.properties.separate_col = value;
                        console.log("[IAMCCS AutoLink] SeparateCol:", value);
                        recolorExistingAutoLinks(app.graph, this.properties.autolink_color_set, value, this.properties.autolink_color_get, this.properties.color_titles);
                    }
                );

                this.addWidget(
                    "combo",
                    "AutoLinkColor",
                    this.properties.autolink_color_set,
                    (value) => {
                        this.properties.autolink_color_set = value;
                        console.log("[IAMCCS AutoLink] AutoLinkColor (set/base):", value);
                        recolorExistingAutoLinks(app.graph, value, this.properties.separate_col, this.properties.autolink_color_get, this.properties.color_titles);
                    },
                    {
                        values: () => alphaSort(Object.keys(AUTOLINK_COLORS))
                    }
                );

                this.addWidget(
                    "combo",
                    "AutoLinkColorGet",
                    this.properties.autolink_color_get,
                    (value) => {
                        this.properties.autolink_color_get = value;
                        console.log("[IAMCCS AutoLink] AutoLinkColorGet:", value);
                        recolorExistingAutoLinks(app.graph, this.properties.autolink_color_set, this.properties.separate_col, value, this.properties.color_titles);
                    },
                    {
                        values: () => alphaSort(Object.keys(AUTOLINK_COLORS))
                    }
                );

                // Dropdown: colore testo titoli (AutoLink)
                this.addWidget(
                    "combo",
                    "ColorTitles",
                    this.properties.color_titles,
                    (value) => {
                        this.properties.color_titles = value;
                        console.log("[IAMCCS AutoLink] ColorTitles:", value);
                        recolorExistingAutoLinks(app.graph, this.properties.autolink_color_set, this.properties.separate_col, this.properties.autolink_color_get, value);
                    },
                    {
                        values: () => ["White", "Black", "Auto"]
                    }
                );

                // Divider: ColorTitles -> Blacklist
                addDivider();

                // Dropdown per aggiungere nodi/tipi alla blacklist
                this.addWidget(
                    "combo",
                    "add_to_blacklist",
                    this.properties.blacklist_pending_value,
                    (value) => {
                        // Keep selection "loaded"; actual add happens via EXECUTE
                        this.properties.blacklist_pending_value = value || "";
                        console.log("[IAMCCS AutoLink] Pending blacklist selection:", this.properties.blacklist_pending_value);
                    },
                    {
                        values: () => {
                            if (this.properties.all_nodes_sel) {
                                // Mostra tutti i nodi singolarmente
                                const nodes = _iamccsGraphNodes(app.graph).filter(n => 
                                    n.type !== SET_TYPE && 
                                    n.type !== GET_TYPE && 
                                    n.type !== CONVERTER_TYPE &&
                                    n.type !== ARGUMENTS_TYPE &&
                                    true
                                );
                                // Solo ID (no title/type)
                                const sorted = alphaSort(nodes.map(n => String(n.id)));
                                return ["", ...sorted];
                            } else {
                                // Mostra solo tipi di nodi
                                const nodeTypes = new Set();
                                _iamccsGraphNodes(app.graph).forEach(n => {
                                    if (n.type !== SET_TYPE && 
                                        n.type !== GET_TYPE && 
                                        n.type !== CONVERTER_TYPE &&
                                        n.type !== ARGUMENTS_TYPE &&
                                        !this.properties.blacklist_types.includes(n.type)) {
                                        nodeTypes.add(n.type);
                                    }
                                });
                                const sorted = alphaSort(Array.from(nodeTypes)).map(t => `[TYPE] ${t}`);
                                return ["", ...sorted];
                            }
                        }
                    }
                );

                // Sposta blacklist mode sotto add_to_blacklist
                this.addWidget(
                    "combo",
                    "blacklist_mode",
                    this.properties.blacklist_add_mode,
                    (value) => {
                        this.properties.blacklist_add_mode = value;
                        console.log("[IAMCCS AutoLink] Blacklist mode:", value);
                    },
                    {
                        values: () => ["", "both", "only_output", "only_input"]
                    }
                );

                // EXECUTE: applica la modalità al nodo selezionato (o aggiunge tipo)
                this.addWidget(
                    "button",
                    "EXECUTE",
                    null,
                    () => {
                        const value = (this.properties.blacklist_pending_value || "").trim();
                        if (!value) return;

                        const pickedMode = (this.properties.blacklist_add_mode || "").trim() || "both";

                        if (value.startsWith("[TYPE] ")) {
                            const nodeType = value.replace(/^\[TYPE\] /, "");
                            this.properties.blacklist_types = this.properties.blacklist_types || [];
                            if (nodeType && !this.properties.blacklist_types.includes(nodeType)) {
                                this.properties.blacklist_types.push(nodeType);
                                console.log("[IAMCCS AutoLink] Added node type to blacklist:", nodeType);
                            }
                        } else {
                            const nodeId = parseInt(value);
                            if (!Number.isFinite(nodeId)) return;
                            this.properties.blacklist = this.properties.blacklist || [];
                            this.properties.blacklist_node_modes = this.properties.blacklist_node_modes || {};

                            if (!this.properties.blacklist.includes(nodeId)) this.properties.blacklist.push(nodeId);
                            this.properties.blacklist_node_modes[String(nodeId)] = pickedMode;
                            console.log("[IAMCCS AutoLink] Added/Updated node blacklist mode:", nodeId, this.properties.blacklist_node_modes[String(nodeId)]);
                        }

                        // Clear pending selection + mode after executing (so they "disappear" from UI)
                        this.properties.blacklist_pending_value = "";
                        this.properties.blacklist_add_mode = "";
                        const widget = this.widgets?.find(w => w?.name === "add_to_blacklist" || w?.label === "add_to_blacklist");
                        if (widget) widget.value = "";

                        const modeWidget = this.widgets?.find(w => w?.name === "blacklist_mode" || w?.label === "blacklist_mode");
                        if (modeWidget) modeWidget.value = "";
                    }
                );
                
                // Dropdown che mostra la blacklist (selezione non distruttiva)
                this.addWidget(
                    "combo",
                    "blacklist_view",
                    this.properties.blacklist_view_selected,
                    (value) => {
                        this.properties.blacklist_view_selected = value || "";
                        console.log("[IAMCCS AutoLink] Blacklist selection:", this.properties.blacklist_view_selected);
                    },
                    {
                        values: () => {
                            const items = [];
                            
                            // Aggiungi tipi
                            (this.properties.blacklist_types || []).forEach(type => {
                                items.push(`[TYPE] ${type}`);
                            });
                            
                            // Aggiungi singoli nodi
                            (this.properties.blacklist || []).forEach(id => {
                                const node = app.graph.getNodeById(id);
                                const mode = this.properties.blacklist_node_modes?.[String(id)] || "both";
                                const name = node ? (node.title || node.type) : "(deleted)";
                                items.push(`${id} - ${name} - (${mode})`);
                            });
                            
                            return ["", ...alphaSort(items)];
                        }
                    }
                );

                // Remove button: rimuove l'elemento selezionato nella blacklist_view
                this.addWidget(
                    "button",
                    "remove_blacklist",
                    null,
                    () => {
                        const value = (this.properties.blacklist_view_selected || "").trim();
                        if (!value) return;

                        if (value.startsWith("[TYPE] ")) {
                            const nodeType = value.replace(/^\[TYPE\] /, "");
                            const index = (this.properties.blacklist_types || []).indexOf(nodeType);
                            if (index !== -1) {
                                this.properties.blacklist_types.splice(index, 1);
                                console.log("[IAMCCS AutoLink] Removed node type from blacklist:", nodeType);
                            }
                        } else {
                            const [nodeIdStr] = value.split(" - ");
                            const nodeId = parseInt(nodeIdStr);
                            const index = (this.properties.blacklist || []).indexOf(nodeId);
                            if (index !== -1) {
                                this.properties.blacklist.splice(index, 1);
                                if (this.properties.blacklist_node_modes) delete this.properties.blacklist_node_modes[String(nodeId)];
                                console.log("[IAMCCS AutoLink] Removed node from blacklist:", nodeId);
                            }
                        }

                        // Clear selection after removal
                        this.properties.blacklist_view_selected = "";
                        const widget = this.widgets?.find(w => w?.name === "blacklist_view" || w?.label === "blacklist_view");
                        if (widget) widget.value = "";
                    }
                );
                
                // Dropdown per navigare tra Set/Get
                this.addWidget(
                    "combo",
                    "jump_to_autolink",
                    "",
                    (value) => {
                        if (!value) return;
                        
                        const [nodeIdStr] = value.split(" - ");
                        const nodeId = parseInt(nodeIdStr);
                        const node = app.graph.getNodeById(nodeId);
                        
                        if (node) {
                            app.canvas.centerOnNode(node);
                            app.canvas.selectNode(node);
                        }
                    },
                    {
                        values: () => {
                            const autoLinkNodes = _iamccsGraphNodes(app.graph).filter(n => 
                                n.type === SET_TYPE || n.type === GET_TYPE
                            );
                            return ["", ...alphaSort(autoLinkNodes.map(n => `${n.id} - ${n.title || n.type}`))];
                        }
                    }
                );

                // Auto-resize so newly added widgets are not hidden
                try {
                    const s = this.computeSize?.();
                    if (Array.isArray(s) && s.length >= 2) {
                        const w = Math.max(this.size?.[0] || 0, s[0] || 0);
                        const h = Math.max(this.size?.[1] || 0, s[1] || 0);
                        this.size = [w, h];
                    }
                } catch (e) {
                    console.warn("[IAMCCS AutoLink] Arguments auto-resize failed", e);
                }
                try { app?.graph?.setDirtyCanvas?.(true, true); } catch {}
                
                this.isVirtualNode = true;
                
                return result;
            };
        }
        
        // Set node - come KJ SetNode
        if (nodeData?.name === SET_TYPE) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                const node = this;

                // Dropdown colore (se cambi un Set, cambia anche i Get corrispondenti)
                node.properties = node.properties || {};
                if (!node.properties.autolink_color_name) node.properties.autolink_color_name = "Gray";
                this.addWidget(
                    "combo",
                    "AutoLinkColor",
                    node.properties.autolink_color_name,
                    (value) => {
                        const key = getAutolinkKey(node);

                        // prendi lo stato SeparateCol + colore get dal primo Arguments disponibile (se presente)
                        const argNode = _iamccsGraphNodes(app.graph).find(n => n?.type === ARGUMENTS_TYPE);
                        const separateCol = !!argNode?.properties?.separate_col;
                        const colorGetName = argNode?.properties?.autolink_color_get || "Gray";
                        const colorTitles = argNode?.properties?.color_titles || "White";

                        node.properties.autolink_color_name = value;
                        node.properties.autolink_color_locked = true;
                        applyColorToAutolinkKey(app.graph, key, value, separateCol, colorGetName, colorTitles);
                    },
                    {
                        values: () => alphaSort(Object.keys(AUTOLINK_COLORS))
                    }
                );
                
                // Aggiungi widget per il nome
                const nameWidget = this.addWidget("text", "name", "", (value) => {
                    if (isValidAutolinkKey(value)) {
                        const safe = String(value).trim();
                        node.title = safe;

                        // Mantieni i port name coerenti con la chiave
                        if (node.inputs && node.inputs[0]) node.inputs[0].name = safe;
                        if (node.outputs && node.outputs[0]) node.outputs[0].name = safe;

                        // Salva il valore corrente per il prossimo cambiamento (tracking locale)
                        nameWidget.lastValue = safe;

                        // Applica colore testo titolo (se configurato)
                        applyNodeTitleTextColor(node, getCurrentColorTitlesMode());
                    } else {
                        // evita titoli '*' o vuoti
                        if (String(value ?? "").trim() === "*") {
                            try { setWidgetValue(node, "name", ""); } catch {}
                        }
                        if (String(node.title ?? "").trim() === "*" || !String(node.title ?? "").trim()) {
                            node.title = "Set AutoLink";
                        }
                    }
                });
                
                // Inizializza lastValue quando il nodo viene caricato
                if (!nameWidget.lastValue && nameWidget.value) {
                    nameWidget.lastValue = nameWidget.value;
                }
                
                // Input/output: normalizza workflow vecchi (che possono avere input duplicati)
                normalizeAutolinkIOSlots(app.graph, node, { wantInputs: 1, wantOutputs: 1 });
                
                // Callback quando si collega
                this.onConnectionsChange = function(slotType, slot, isConnect, link_info) {
                    if (slotType === 1 && isConnect && link_info) {
                        const fromNode = app.graph.getNodeById(link_info.origin_id);
                        if (fromNode && fromNode.outputs && fromNode.outputs[link_info.origin_slot]) {
                            const outputType = fromNode.outputs[link_info.origin_slot].type;

                            // Usa sempre un nome leggibile e stabile (slot-name), non il tipo puro.
                            // Questo produce base tipo "model"/"image" e poi lo rendiamo unico: model_0, image_2, ...
                            let suggestedBase = getSlotName(fromNode, link_info.origin_slot, true);
                            if (!isValidAutolinkKey(suggestedBase)) {
                                if (outputType && outputType !== "*") suggestedBase = String(outputType).trim().toLowerCase();
                                else suggestedBase = `output_${link_info.origin_slot}`;
                            }
                            
                            // Imposta tipo
                            node.inputs[0].type = outputType;
                            node.outputs[0].type = outputType;

                            // Se il converter ha già impostato un nome unico, NON sovrascriverlo qui.
                            // Auto-fill solo quando il widget è vuoto.
                            const currentKey = getAutolinkKey(node);
                            const desiredKey = isValidAutolinkKey(currentKey)
                                ? currentKey
                                : makeUniqueAutolinkSetName(app.graph, suggestedBase);

                            // Imposta chiave + UI + porta coerenti
                            setAutolinkKeyAndTitle(node, desiredKey);

                            // Applica colore testo titolo (se configurato)
                            applyNodeTitleTextColor(node, getCurrentColorTitlesMode());
                        }
                    }
                };

                // Overlay draw: rende visibile ColorTitles anche se LiteGraph ignora title_text_color
                // (removed) per-node overlay title drawing; native title is colored via canvas hook

                // Migrazione: vecchi workflow (prima del widget colore) possono aver scritto il nome nel widget colore.
                const colorWidget = getWidget(node, "AutoLinkColor");
                const colorVal = colorWidget?.value;
                const nameVal = nameWidget?.value;
                if (colorVal && !AUTOLINK_COLORS[colorVal] && (!nameVal || !String(nameVal).trim())) {
                    const migrated = String(colorVal).trim();
                    setAutolinkKeyAndTitle(node, migrated);
                    if (colorWidget) colorWidget.value = "Gray";
                    node.properties = node.properties || {};
                    node.properties.autolink_color_name = "Gray";
                    node.properties.autolink_color_locked = false;
                }

                // Fix workflow vecchi: titolo o name a '*'
                try {
                    const rawName = getWidgetValue(node, "name");
                    if (String(rawName ?? "").trim() === "*") setWidgetValue(node, "name", "");
                    if (String(node.title ?? "").trim() === "*") node.title = "Set AutoLink";
                } catch {}
                
                // Nodo virtuale - non serializza per il prompt
                this.isVirtualNode = true;
                
                return result;
            };
        }
        
        // Get node - come KJ GetNode  
        if (nodeData?.name === GET_TYPE) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                const node = this;
                
                // Combo dinamico con lista Set disponibili
                this.addWidget("combo", "name", "", (value) => {
                    node.onRename();
                }, {
                    values: () => {
                        const setNodes = _iamccsGraphNodes(app.graph).filter(n => n.type === SET_TYPE);
                        return alphaSort(setNodes.map(n => getAutolinkKey(n)).filter(v => v));
                    }
                });

                // Normalizza output duplicati su workflow vecchi
                normalizeAutolinkIOSlots(app.graph, node, { wantInputs: 0, wantOutputs: 1 });
                
                this.onRename = function() {
                    const setterName = getAutolinkKey(node);
                    const setter = _iamccsGraphNodes(app.graph).find(n => 
                        n.type === SET_TYPE && getAutolinkKey(n) === setterName
                    );
                    
                    if (setter) {
                        const linkType = setter.outputs[0].type;
                        node.outputs[0].type = linkType;
                        node.outputs[0].name = linkType;
                        node.title = setterName;

                        applyNodeTitleTextColor(node, getCurrentColorTitlesMode());
                    }
                };

                // Fix workflow vecchi: titolo a '*'
                try {
                    if (String(node.title ?? "").trim() === "*") node.title = "Get AutoLink";
                } catch {}
                // (removed) per-node overlay title drawing; native title is colored via canvas hook
                
                // Override getInputLink per prendere da Set
                this.getInputLink = function(slot) {
                    const setterName = getAutolinkKey(node);
                    const setter = _iamccsGraphNodes(app.graph).find(n => 
                        n.type === SET_TYPE && getAutolinkKey(n) === setterName
                    );
                    
                    if (setter) {
                        const slotInfo = setter.inputs[slot];
                        if (slotInfo) {
                            const linkId = slotInfo.link != null
                                ? slotInfo.link
                                : (Array.isArray(slotInfo.links) ? slotInfo.links[0] : null);
                            const link = _iamccsGetLink(app.graph, linkId);
                            return link || null;
                        }
                    }
                    return null;
                };
                
                // Nodo virtuale - non serializza per il prompt
                this.isVirtualNode = true;
                
                return result;
            };
        }
    }
});

// === FUNZIONI CONVERSIONE ===

function getNodeById(graph, id) {
    if (!graph || id == null) return null;

    try {
        if (typeof graph.getNodeById === "function") {
            const direct = graph.getNodeById(id);
            if (direct) return direct;
        }
    } catch {}

    const byId = graph._nodes_by_id || graph._nodesById;
    if (byId && typeof byId === "object") {
        // Try raw key first (string/uuid)
        const raw = byId[id] || byId[String(id)];
        if (raw) return raw;
        // Also try numeric key if applicable
        const nid = Number(id);
        if (Number.isFinite(nid)) return byId[nid] || byId[String(nid)] || null;
    }

    const nodes = _iamccsGraphNodes(graph);
    // Match by string id first (covers UUID)
    const sid = String(id);
    const byString = nodes.find(n => n?.id != null && String(n.id) === sid);
    if (byString) return byString;
    // Then numeric match (covers classic LiteGraph ids)
    const nid = Number(id);
    if (Number.isFinite(nid)) {
        return nodes.find(n => Number(n?.id) === nid) || null;
    }
    return null;
}

function createNode(graph, type, x, y) {
    const node = LiteGraph.createNode(type);
    if (!node) return null;
    node.pos = [x, y];
    graph.add(node);
    return node;
}

function setWidgetValue(node, name, value) {
    if (!node || !node.widgets || node.widgets.length === 0) return false;

    // 1) Exact match (ideal case)
    const exact = node.widgets.find(w => w?.name === name);
    if (exact) {
        exact.value = value;
        return true;
    }

    // 2) Case-insensitive / substring match (some custom nodes don't use "name" literally)
    const lowered = String(name).toLowerCase();
    const fuzzy = node.widgets.find(w => typeof w?.name === 'string' && w.name.toLowerCase().includes(lowered));
    if (fuzzy) {
        fuzzy.value = value;
        return true;
    }

    // 3) Fallback: if there's a single widget, assume it's the name
    if (node.widgets.length === 1) {
        node.widgets[0].value = value;
        return true;
    }

    return false;
}

function getSlotName(node, slot, isOutput) {
    if (!node) return `slot_${slot}`;
    const slots = isOutput ? node.outputs : node.inputs;

    const info = slots?.[slot];
    const name = info?.name;
    if (name === "*") {
        const t = info?.type;
        if (t && t !== "*") return String(t).trim().toLowerCase();
        return isOutput ? `output_${slot}` : `input_${slot}`;
    }

    return name || (isOutput ? `output_${slot}` : `input_${slot}`);
}

function getBlacklistFromInput(converterNode) {
    // Cerca il nodo Arguments collegato all'input arg
    if (!converterNode.inputs || converterNode.inputs.length === 0) {
        return {
            blacklist: [],
            blacklistTypes: [],
            blacklistNodeModes: {},
            includeKijNodes: false,
            groupExclude: false,
            groupInOutExclude: "None",
            alignMode: "TopToDown",
            packingMode: "AvoidAll",
            colorSet: "Gray",
            colorGet: "Gray",
            separateCol: false,
            colorTitles: "White",
        };
    }
    
    const argInput = converterNode.inputs.find(i => i.name === "arg");
    if (!argInput || !argInput.link) {
        return {
            blacklist: [],
            blacklistTypes: [],
            blacklistNodeModes: {},
            includeKijNodes: false,
            groupExclude: false,
            groupInOutExclude: "None",
            alignMode: "TopToDown",
            packingMode: "AvoidAll",
            colorSet: "Gray",
            colorGet: "Gray",
            separateCol: false,
            colorTitles: "White",
        };
    }
    
    const link = _iamccsGetLink(app.graph, argInput.link);
    if (!link) {
        return {
            blacklist: [],
            blacklistTypes: [],
            blacklistNodeModes: {},
            includeKijNodes: false,
            groupExclude: false,
            groupInOutExclude: "None",
            alignMode: "TopToDown",
            packingMode: "AvoidAll",
            colorSet: "Gray",
            colorGet: "Gray",
            separateCol: false,
            colorTitles: "White",
        };
    }
    
    const argNode = getNodeById(app.graph, link.origin_id);
    if (!argNode || argNode.type !== ARGUMENTS_TYPE) {
        return {
            blacklist: [],
            blacklistTypes: [],
            blacklistNodeModes: {},
            includeKijNodes: false,
            groupExclude: false,
            groupInOutExclude: "None",
            alignMode: "TopToDown",
            packingMode: "AvoidAll",
            colorSet: "Gray",
            colorGet: "Gray",
            separateCol: false,
            colorTitles: "White",
        };
    }
    
    return {
        blacklist: argNode.properties?.blacklist || [],
        blacklistTypes: argNode.properties?.blacklist_types || [],
        blacklistNodeModes: argNode.properties?.blacklist_node_modes || {},
        includeKijNodes: argNode.properties?.include_kijnodes || false,
        groupExclude: (argNode.properties?.group_exclude ?? argNode.properties?.group_exlude) || false,
        groupInOutExclude: argNode.properties?.group_inout_exclude || "None",
        alignMode: argNode.properties?.align_mode || "TopToDown",
        packingMode: argNode.properties?.packing_mode || "AvoidAll",
        colorSet: argNode.properties?.autolink_color_set || "Gray",
        colorGet: argNode.properties?.autolink_color_get || "Gray",
        separateCol: !!argNode.properties?.separate_col,
        colorTitles: argNode.properties?.color_titles || "White",
    };
}

function relayoutExistingAutoLinks(graph, alignMode = "TopToDown", packingMode = "AvoidAll") {
    if (!graph) return;

    const GRID_SIZE = 80;
    const GET_OFFSET_X = -220;

    const isArrayLike = (v, minLen) => v != null && typeof v.length === 'number' && v.length >= minLen;
    const getNodePos = (n) => {
        const x = (isArrayLike(n?.pos, 2) ? n.pos[0] : n?.pos?.[0]) ?? 0;
        const y = (isArrayLike(n?.pos, 2) ? n.pos[1] : n?.pos?.[1]) ?? 0;
        return [x, y];
    };
    const getNodeSize = (n) => {
        const w = (isArrayLike(n?.size, 2) ? n.size[0] : n?.size?.[0]) ?? 200;
        const h = (isArrayLike(n?.size, 2) ? n.size[1] : n?.size?.[1]) ?? 100;
        return [w, h];
    };

    const isAutoLinkNode = (node) => node?.type === SET_TYPE || node?.type === GET_TYPE;
    const shouldTreatAsObstacle = (node) => {
        if (!node) return false;
        if (packingMode === "AvoidAll") return true;
        if (packingMode === "AvoidNonAutoLink") return !isAutoLinkNode(node);
        return true;
    };

    const rectsOverlap = (ax, ay, aw, ah, bx, by, bw, bh) =>
        ax < bx + bw && ax + aw > bx && ay < by + bh && ay + ah > by;

    function seriesCentered(firstUp = true) {
        return (i) => {
            if (i === 0) return 0;
            const k = Math.ceil(i / 2);
            const sign = (i % 2 === 1) ? (firstUp ? -1 : 1) : (firstUp ? 1 : -1);
            return sign * k;
        };
    }

    function layoutDelta(i, mode) {
        switch (mode) {
            case "TopToDown":
            case "stack_down":
                return { dx: 0, dy: i };
            case "BottomToTop":
            case "stack_up":
                return { dx: 0, dy: -i };
            case "CenterUpDown":
            case "stack_center_up":
                return { dx: 0, dy: seriesCentered(true)(i) };
            case "CenterDownUp":
            case "stack_center_down":
                return { dx: 0, dy: seriesCentered(false)(i) };
            case "AlignX_Right":
            case "row_right":
                return { dx: i, dy: 0 };
            case "AlignX_Left":
            case "row_left":
                return { dx: -i, dy: 0 };
            case "Columns_Down":
            case "columns_down": {
                const rows = 10;
                const col = Math.floor(i / rows);
                const row = i % rows;
                return { dx: col, dy: row };
            }
            case "Columns_Up":
            case "columns_up": {
                const rows = 10;
                const col = Math.floor(i / rows);
                const row = i % rows;
                return { dx: col, dy: -row };
            }
            case "Rake_Down":
            case "rake_down":
                return { dx: i, dy: i };
            case "Rake_Up":
            case "rake_up":
                return { dx: i, dy: -i };
            case "Proportional":
                // Keep Y anchored; move horizontally if collisions
                return { dx: seriesCentered(true)(i), dy: 0 };
            default:
                return { dx: 0, dy: seriesCentered(true)(i) };
        }
    }

    function findFreePosition(baseX, baseY, offsetX, occupied, mode, ignoreNode, extraObstacles = []) {
        const maxAttempts = 250;
        for (let attempts = 0; attempts < maxAttempts; attempts++) {
            const { dx, dy } = layoutDelta(attempts, mode);
            const testX = baseX + offsetX + dx * GRID_SIZE;
            const testY = baseY + dy * GRID_SIZE;
            const posKey = `${Math.round(testX / GRID_SIZE)}_${Math.round(testY / GRID_SIZE)}`;
            if (occupied.has(posKey)) continue;

            let overlaps = false;
            for (const node of _iamccsGraphNodes(graph)) {
                if (!shouldTreatAsObstacle(node)) continue;
                if (ignoreNode && node === ignoreNode) continue;
                const [nx, ny] = getNodePos(node);
                const [nw, nh] = getNodeSize(node);
                if (rectsOverlap(testX, testY, 150, 26, nx, ny, nw, nh)) {
                    overlaps = true;
                    break;
                }
            }

            if (!overlaps && extraObstacles && extraObstacles.length) {
                for (const r of extraObstacles) {
                    if (rectsOverlap(testX, testY, 150, 26, r.x, r.y, r.w, r.h)) {
                        overlaps = true;
                        break;
                    }
                }
            }

            if (!overlaps) {
                occupied.add(posKey);
                return [testX, testY];
            }
        }
        return [baseX + offsetX, baseY];
    }

    const nodes = _iamccsGraphNodes(graph);
    const sets = nodes.filter(n => n?.type === SET_TYPE);
    const gets = nodes.filter(n => n?.type === GET_TYPE);

    // 1) Re-layout Sets: ancorati al nodo origine che li alimenta
    const occupiedSet = new Set();
    const setRecords = [];
    for (const setNode of sets) {
        const inLinkId = setNode?.inputs?.[0]?.link;
        const inLink = inLinkId != null ? _iamccsGetLink(graph, inLinkId) : null;
        if (!inLink) continue;

        const originNode = getNodeById(graph, inLink.origin_id);
        if (!originNode) continue;

        const originSlot = inLink.origin_slot ?? 0;
        const name = getAutolinkKey(setNode);
        setRecords.push({ setNode, originNode, originSlot, name });
    }

    setRecords.sort((a, b) => {
        if (a.originNode.id !== b.originNode.id) return a.originNode.id - b.originNode.id;
        if (a.originSlot !== b.originSlot) return a.originSlot - b.originSlot;
        return String(a.name).localeCompare(String(b.name));
    });

    const placedAutoLinkRects = [];

    for (const rec of setRecords) {
        const [ox, oy] = getNodePos(rec.originNode);
        const [ow] = getNodeSize(rec.originNode);
        const baseX = ox + ow;
        let baseY = oy;
        if (alignMode === "Proportional" && typeof rec.originNode?.getConnectionPos === 'function') {
            const p = rec.originNode.getConnectionPos(false, rec.originSlot);
            if (Array.isArray(p) && p.length >= 2) baseY = p[1] ?? baseY;
        }
        const [nx, ny] = findFreePosition(baseX, baseY, 20, occupiedSet, alignMode, rec.setNode, placedAutoLinkRects);
        rec.setNode.pos = [nx, ny];
        placedAutoLinkRects.push({ x: nx, y: ny, w: 150, h: 26 });
    }

    // 2) Re-layout Gets: ancorati al nodo target che alimentano
    const occupiedGet = new Set();
    const getRecords = [];
    for (const getNode of gets) {
        // preferisci link reale in uscita
        const outLinks = getNode?.outputs?.[0]?.links;
        const linkId = Array.isArray(outLinks) && outLinks.length > 0 ? outLinks[0] : null;
        const outLink = linkId != null ? _iamccsGetLink(graph, linkId) : null;
        const targetId = outLink?.target_id ?? getNode?.properties?.metadata?.target?.id;
        if (targetId == null) continue;
        const targetNode = getNodeById(graph, targetId);
        if (!targetNode) continue;
        const name = getAutolinkKey(getNode);
        const targetSlot = outLink?.target_slot;
        getRecords.push({ getNode, targetNode, targetSlot, name });
    }

    getRecords.sort((a, b) => {
        if (a.targetNode.id !== b.targetNode.id) return a.targetNode.id - b.targetNode.id;
        if (alignMode === "Proportional") {
            const as = Number.isFinite(a.targetSlot) ? a.targetSlot : 0;
            const bs = Number.isFinite(b.targetSlot) ? b.targetSlot : 0;
            if (as !== bs) return as - bs;
        }
        return String(a.name).localeCompare(String(b.name));
    });

    for (const rec of getRecords) {
        const [tx, ty] = getNodePos(rec.targetNode);
        const baseX = tx;
        let baseY = ty;
        if (alignMode === "Proportional" && Number.isFinite(rec.targetSlot) && typeof rec.targetNode?.getConnectionPos === 'function') {
            const p = rec.targetNode.getConnectionPos(true, rec.targetSlot);
            if (Array.isArray(p) && p.length >= 2) baseY = p[1] ?? baseY;
        }
        const [nx, ny] = findFreePosition(baseX, baseY, GET_OFFSET_X, occupiedGet, alignMode, rec.getNode, placedAutoLinkRects);
        rec.getNode.pos = [nx, ny];
        placedAutoLinkRects.push({ x: nx, y: ny, w: 150, h: 26 });
    }

    graph.setDirtyCanvas(true, true);
}

function getGroupForNode(graph, node) {
    const rawGroups = graph?._groups ?? graph?.groups ?? [];
    const groups = Array.isArray(rawGroups)
        ? rawGroups
        : (rawGroups && typeof rawGroups === 'object' ? Object.values(rawGroups) : []);

    if (!node || !groups || groups.length === 0) return null;

    const isArrayLike = (v, minLen) => v != null && typeof v.length === 'number' && v.length >= minLen;

    const nodeW = (isArrayLike(node.size, 2) ? node.size[0] : node.size?.[0]) ?? 0;
    const nodeH = (isArrayLike(node.size, 2) ? node.size[1] : node.size?.[1]) ?? 0;
    const nodeX = (isArrayLike(node.pos, 2) ? node.pos[0] : node.pos?.[0]) ?? 0;
    const nodeY = (isArrayLike(node.pos, 2) ? node.pos[1] : node.pos?.[1]) ?? 0;
    const cx = nodeX + nodeW / 2;
    const cy = nodeY + nodeH / 2;

    let bestGroup = null;
    let bestArea = Infinity;

    for (let i = 0; i < groups.length; i++) {
        const g = groups[i];
        if (!g) continue;

        let bounding = null;
        if (isArrayLike(g.bounding, 4)) {
            bounding = [g.bounding[0], g.bounding[1], g.bounding[2], g.bounding[3]];
        } else if (typeof g.getBounding === 'function') {
            const b = g.getBounding();
            if (isArrayLike(b, 4)) bounding = [b[0], b[1], b[2], b[3]];
        }

        if (!bounding && isArrayLike(g.pos, 2) && isArrayLike(g.size, 2)) {
            bounding = [g.pos[0], g.pos[1], g.size[0], g.size[1]];
        }

        if (!bounding) continue;

        const [gx, gy, gw, gh] = bounding;
        if (cx >= gx && cx <= gx + gw && cy >= gy && cy <= gy + gh) {
            const area = Math.abs(gw * gh);
            if (area < bestArea) {
                bestArea = area;
                bestGroup = g;
            }
        }
    }

    return bestGroup;
}

function getGroupForPoint(graph, x, y) {
    const rawGroups = graph?._groups ?? graph?.groups ?? [];
    const groups = Array.isArray(rawGroups)
        ? rawGroups
        : (rawGroups && typeof rawGroups === 'object' ? Object.values(rawGroups) : []);
    if (!groups || groups.length === 0) return null;

    const px = Number(x);
    const py = Number(y);
    if (!Number.isFinite(px) || !Number.isFinite(py)) return null;

    const isArrayLike = (v, minLen) => v != null && typeof v.length === 'number' && v.length >= minLen;
    let bestGroup = null;
    let bestArea = Infinity;

    for (let i = 0; i < groups.length; i++) {
        const g = groups[i];
        if (!g) continue;

        let bounding = null;
        if (isArrayLike(g.bounding, 4)) {
            bounding = [g.bounding[0], g.bounding[1], g.bounding[2], g.bounding[3]];
        } else if (typeof g.getBounding === 'function') {
            const b = g.getBounding();
            if (isArrayLike(b, 4)) bounding = [b[0], b[1], b[2], b[3]];
        }
        if (!bounding && isArrayLike(g.pos, 2) && isArrayLike(g.size, 2)) {
            bounding = [g.pos[0], g.pos[1], g.size[0], g.size[1]];
        }
        if (!bounding) continue;

        const [gx, gy, gw, gh] = bounding;
        if (px >= gx && px <= gx + gw && py >= gy && py <= gy + gh) {
            const area = Math.abs(gw * gh);
            if (area < bestArea) {
                bestArea = area;
                bestGroup = g;
            }
        }
    }

    return bestGroup;
}

function _iamccsGetGroupState(group) {
    const g = group || null;
    if (!g) return { hidden: false, disabled: false };

    const props = (g.properties && typeof g.properties === 'object') ? g.properties : {};
    const flags = (g.flags && typeof g.flags === 'object') ? g.flags : {};

    const hidden = !!(
        g.hidden || g.is_hidden || g.isHidden ||
        flags.hidden ||
        props.hidden || props.is_hidden || props.isHidden ||
        g.visible === false || props.visible === false
    );

    // "collapsed" is used by some LiteGraph/ComfyUI variants to mean the group is folded/hidden.
    const collapsed = !!(
        g.collapsed || g.is_collapsed || g.isCollapsed ||
        flags.collapsed ||
        props.collapsed || props.is_collapsed || props.isCollapsed
    );

    const disabled = !!(
        g.enabled === false || g.disabled === true ||
        props.enabled === false || props.disabled === true
    );

    return { hidden: hidden || collapsed, disabled };
}

function _iamccsApplyGroupStateToNode(graph, node, referenceNode, pos) {
    if (!node) return;

    const NEVER = (window?.LiteGraph?.NEVER != null) ? window.LiteGraph.NEVER : 2;

    // Prefer group detected at the node position; fallback to reference node's group.
    const cx = pos && Number.isFinite(pos.x) ? pos.x : null;
    const cy = pos && Number.isFinite(pos.y) ? pos.y : null;
    const groupAtPos = (cx != null && cy != null) ? getGroupForPoint(graph, cx, cy) : null;
    const groupAtRef = referenceNode ? getGroupForNode(graph, referenceNode) : null;
    const group = groupAtPos || groupAtRef;

    const { hidden, disabled } = _iamccsGetGroupState(group);

    node.properties = node.properties || {};
    node.properties.autolink_inherit_group_state = true;

    // If the group is hidden, AutoLink nodes must be hidden AND non-active.
    if (hidden) {
        node.flags = node.flags || {};
        node.flags.hidden = true;
        node.hidden = true;
        node.flags.collapsed = true;
        node.collapsed = true;
        node.mode = NEVER;
        try {
            if (typeof node.collapse === 'function') node.collapse();
        } catch {}
        return;
    }

    // If the group is disabled (but still visible), keep the node visible but non-active.
    if (disabled) {
        node.mode = NEVER;
    } else if (referenceNode && referenceNode.mode != null) {
        // Otherwise, inherit the reference node mode (covers workflows where the group implementation
        // propagates "disabled" via node.mode instead of group flags).
        node.mode = referenceNode.mode;
    }

    // Best-effort: inherit hidden flag if reference node is currently hidden by some custom UI logic.
    if (referenceNode?.flags?.hidden || referenceNode?.hidden || referenceNode?.is_hidden) {
        node.flags = node.flags || {};
        node.flags.hidden = true;
        node.hidden = true;
        node.mode = NEVER;
    }
}

function convertAllLinks(
    graph,
    blacklist = [],
    blacklistTypes = [],
    blacklistNodeModes = {},
    includeKijNodes = false,
    groupExclude = false,
    groupInOutExclude = "None",
    alignMode = "TopToDown",
    packingMode = "AvoidAll",
    colorSet = "Gray",
    colorGet = "Gray",
    separateCol = false,
    colorTitles = "White"
) {
    // Backward compat for older call signature that didn't include blacklistNodeModes / colorTitles
    if (typeof blacklistNodeModes === 'boolean') {
        const oldIncludeKijNodes = blacklistNodeModes;
        const oldGroupExclude = includeKijNodes;
        const oldAlignMode = groupExclude;
        const oldPackingMode = alignMode;
        const oldColorSet = packingMode;
        const oldColorGet = colorSet;
        const oldSeparateCol = colorGet;

        blacklistNodeModes = {};
        includeKijNodes = !!oldIncludeKijNodes;
        groupExclude = !!oldGroupExclude;
        groupInOutExclude = "None";
        alignMode = oldAlignMode || "TopToDown";
        packingMode = oldPackingMode || "AvoidAll";
        colorSet = oldColorSet || "Gray";
        colorGet = oldColorGet || "Gray";
        separateCol = !!oldSeparateCol;
        colorTitles = "White";
    }

    console.log("[IAMCCS AutoLink] Converting links...");
    console.log("[IAMCCS AutoLink] Blacklist IDs:", blacklist);
    console.log("[IAMCCS AutoLink] Blacklist Types:", blacklistTypes);
    console.log("[IAMCCS AutoLink] Blacklist Node Modes:", blacklistNodeModes);
    console.log("[IAMCCS AutoLink] Include KijNodes:", includeKijNodes);
    console.log("[IAMCCS AutoLink] GroupExclude:", groupExclude);
    console.log("[IAMCCS AutoLink] GroupInOutExclude:", groupInOutExclude);
    console.log("[IAMCCS AutoLink] Align mode:", alignMode);
    console.log("[IAMCCS AutoLink] Packing mode:", packingMode);
    console.log("[IAMCCS AutoLink] ColorSet:", colorSet);
    console.log("[IAMCCS AutoLink] ColorGet:", colorGet);
    console.log("[IAMCCS AutoLink] SeparateCol:", separateCol);
    console.log("[IAMCCS AutoLink] ColorTitles:", colorTitles);

    // Safety cleanup: if previous Restore left some AutoLink nodes behind (due to older bugs
    // or partial restores), they can cause repeated Convert/Restore cycles to spawn new
    // Set/Get nodes with suffixed names (model_0 -> model_1 -> model_2 ...).
    // We only prune nodes that are clearly dangling (no links / Set has no Gets).
    try {
        const NEVER = (window?.LiteGraph?.NEVER != null) ? window.LiteGraph.NEVER : 2;
        const isInactive = (n) => !!(n?.flags?.hidden || n?.hidden || n?.is_hidden || Number(n?.mode) === Number(NEVER));

        const nodes = _iamccsGraphNodes(graph);
        const sets = nodes.filter(n => n?.type === SET_TYPE);
        const gets = nodes.filter(n => n?.type === GET_TYPE);

        const links = _iamccsGraphLinksEntries(graph).map(([, l]) => l).filter(Boolean);
        const linkCountById = new Map();
        const incomingCountById = new Map();
        const outgoingCountById = new Map();

        for (const l of links) {
            const oid = l.origin_id;
            const tid = l.target_id;
            const ok = (id) => id != null;
            if (ok(oid)) {
                const k = String(oid);
                outgoingCountById.set(k, (outgoingCountById.get(k) || 0) + 1);
                linkCountById.set(k, (linkCountById.get(k) || 0) + 1);
            }
            if (ok(tid)) {
                const k = String(tid);
                incomingCountById.set(k, (incomingCountById.get(k) || 0) + 1);
                linkCountById.set(k, (linkCountById.get(k) || 0) + 1);
            }
        }

        const keyHasGet = new Set(gets.map(g => getAutolinkKey(g)).filter(Boolean));

        let pruned = 0;
        // Remove Get nodes with no outgoing links in the graph.
        for (const g of gets) {
            const id = g?.id;
            if (id == null) continue;
            const out = outgoingCountById.get(String(id)) || 0;
            const any = linkCountById.get(String(id)) || 0;
            if (any === 0 || out === 0) {
                try { graph.remove(g); pruned++; } catch {}
            }
        }

        // Remove Set nodes that have no Gets using their key and are not inactive-hidden placeholders.
        for (const s of sets) {
            const id = s?.id;
            if (id == null) continue;
            const key = getAutolinkKey(s);
            if (key && keyHasGet.has(key)) continue;
            if (isInactive(s)) continue;

            const out = outgoingCountById.get(String(id)) || 0;
            const any = linkCountById.get(String(id)) || 0;
            // Typical Set has exactly one incoming link (source -> set) and no outgoing.
            if (any <= 1 && out === 0) {
                try { graph.remove(s); pruned++; } catch {}
            }
        }

        if (pruned > 0) {
            console.log(`[IAMCCS AutoLink] Pruned ${pruned} dangling AutoLink nodes before converting`);
            try { _iamccsFixLinkIntegrity(graph); } catch {}
            try { graph.setDirtyCanvas(true, true); } catch {}
        }
    } catch {}
    
    const linksToConvert = [];
    const groupExcludedCandidates = [];
    const kijNodesToConvert = [];

    const linkEntries = _iamccsGraphLinksEntries(graph);
    console.log("[IAMCCS AutoLink] Total links visible:", linkEntries.length);

    const _didConnect = (srcNode, originSlot, dstNode, targetSlot) => {
        try {
            const ts = Number(targetSlot);
            const os = Number(originSlot);
            const linkId = dstNode?.inputs?.[ts]?.link;
            if (linkId == null) return null;
            const link = _iamccsGetLink(graph, linkId);
            if (!link) return null;
            if (!_iamccsIdEq(link.origin_id, srcNode.id)) return null;
            if (!_iamccsIdEq(link.target_id, dstNode.id)) return null;
            if (Number(link.origin_slot) !== os) return null;
            if (Number(link.target_slot) !== ts) return null;
            return linkId;
        } catch {
            return null;
        }
    };

    const _safeConnect = (srcNode, originSlot, dstNode, targetSlot) => {
        const os = Number(originSlot);
        const ts = Number(targetSlot);
        if (!Number.isFinite(os) || !Number.isFinite(ts)) return null;

        try {
            srcNode.connect(os, dstNode, ts);
            const ok = _didConnect(srcNode, os, dstNode, ts);
            if (ok != null) return ok;
        } catch {}

        try {
            if (typeof graph.connect === "function") {
                graph.connect(srcNode.id, os, dstNode.id, ts);
                const ok2 = _didConnect(srcNode, os, dstNode, ts);
                if (ok2 != null) return ok2;
            }
        } catch {}

        return null;
    };

    const skip = {
        missing_nodes: 0,
        autolink: 0,
        system: 0,
        kij: 0,
        blacklisted_mode: 0,
        blacklisted_type: 0,
        group_exclude: 0,
        group_inout_exclude: 0,
    };

    for (const [linkId, linkRaw] of linkEntries) {
        const link = linkRaw;
        if (!link) continue;
        
        const srcNode = getNodeById(graph, link.origin_id);
        const dstNode = getNodeById(graph, link.target_id);
        
        if (!srcNode || !dstNode) {
            skip.missing_nodes++;
            continue;
        }
        
        // Skip già AutoLink
        if (srcNode.type === SET_TYPE || srcNode.type === GET_TYPE) { skip.autolink++; continue; }
        if (dstNode.type === SET_TYPE || dstNode.type === GET_TYPE) { skip.autolink++; continue; }
        
        // Blacklist permanente: nodi Arguments e Converter non vengono mai convertiti
        if (srcNode.type === ARGUMENTS_TYPE || srcNode.type === CONVERTER_TYPE ||
            dstNode.type === ARGUMENTS_TYPE || dstNode.type === CONVERTER_TYPE) {
            console.log(`[IAMCCS AutoLink] Skipping AutoLink system node: ${srcNode.type} -> ${dstNode.type}`);
            skip.system++;
            continue;
        }
        
        // Gestione KijNodes
        const srcIsKij = (srcNode.type === KJ_SET_TYPE || srcNode.type === KJ_GET_TYPE);
        const dstIsKij = (dstNode.type === KJ_SET_TYPE || dstNode.type === KJ_GET_TYPE);
        
        if (srcIsKij || dstIsKij) {
            if (includeKijNodes) {
                // Converte KijNodes in AutoLink
                kijNodesToConvert.push({ link, srcNode, dstNode, srcIsKij, dstIsKij });
            } else {
                // Esclude permanentemente KijNodes e i loro collegamenti
                console.log(`[IAMCCS AutoLink] Skipping KijNode: ${srcNode.type} -> ${dstNode.type}`);
                skip.kij++;
            }
            continue;
        }
        
        // Salta i nodi in blacklist utente (per ID o per tipo)
        {
            const srcMode = blacklistNodeModes?.[String(srcNode.id)] || (blacklist.includes(srcNode.id) ? "both" : null);
            const dstMode = blacklistNodeModes?.[String(dstNode.id)] || (blacklist.includes(dstNode.id) ? "both" : null);

            const skipBecauseSrc = !!srcMode && (srcMode === "both" || srcMode === "only_output");
            const skipBecauseDst = !!dstMode && (dstMode === "both" || dstMode === "only_input");

            if (skipBecauseSrc || skipBecauseDst) {
                console.log(
                    `[IAMCCS AutoLink] Skipping blacklisted node by mode: src=${srcNode.id}(${srcMode || "-"}) dst=${dstNode.id}(${dstMode || "-"})`
                );
                skip.blacklisted_mode++;
                continue;
            }
        }
        
        if (blacklistTypes.includes(srcNode.type) || blacklistTypes.includes(dstNode.type)) {
            console.log(`[IAMCCS AutoLink] Skipping blacklisted node type: ${srcNode.type} or ${dstNode.type}`);
            skip.blacklisted_type++;
            continue;
        }

        // GroupExlude: se entrambi i nodi sono dentro lo stesso Group, NON convertire quel collegamento.
        // (I collegamenti in entrata/uscita dal group restano convertibili.)
        if (groupExclude) {
            const srcGroup = getGroupForNode(graph, srcNode);
            const dstGroup = getGroupForNode(graph, dstNode);
            if (srcGroup && srcGroup === dstGroup) {
                skip.group_exclude++;
                groupExcludedCandidates.push({ link, linkId, srcNode, dstNode });
                continue;
            }
        }

        // GroupInOutExclude: se il link attraversa il confine di un group, puoi escludere le entrate/uscite
        if (groupInOutExclude && groupInOutExclude !== "None") {
            const srcGroup = getGroupForNode(graph, srcNode);
            const dstGroup = getGroupForNode(graph, dstNode);
            const crossesBoundary = (srcGroup !== dstGroup) && (srcGroup || dstGroup);

            if (crossesBoundary) {
                const entersDstGroup = !!dstGroup && srcGroup !== dstGroup;
                const exitsSrcGroup = !!srcGroup && srcGroup !== dstGroup;

                if (
                    (groupInOutExclude === "ExcludeEnter" && entersDstGroup) ||
                    (groupInOutExclude === "ExcludeExit" && exitsSrcGroup) ||
                    (groupInOutExclude === "ExcludeBoth" && (entersDstGroup || exitsSrcGroup))
                ) {
                    skip.group_inout_exclude++;
                    continue;
                }
            }
        }
        
        linksToConvert.push({ link, linkId, srcNode, dstNode });
    }

    console.log("[IAMCCS AutoLink] Skip stats:", skip);

    // Fallback: if GroupExclude filtered everything, convert inside groups rather than doing nothing.
    if (linksToConvert.length === 0 && groupExclude && groupExcludedCandidates.length > 0) {
        console.warn(
            `[IAMCCS AutoLink] GroupExclude removed all candidates (${groupExcludedCandidates.length}); proceeding with in-group conversion to avoid 0-links result.`
        );
        linksToConvert.push(...groupExcludedCandidates);
    }
    
    // Raggruppa i link per origine (stesso nodo + stesso slot) per creare un solo Set per output multipli
    const linksByOrigin = new Map();
    for (const linkData of linksToConvert) {
        const key = `${linkData.srcNode.id}_${linkData.link.origin_slot}`;
        if (!linksByOrigin.has(key)) {
            linksByOrigin.set(key, {
                srcNode: linkData.srcNode,
                originSlot: linkData.link.origin_slot,
                outputName: getSlotName(linkData.srcNode, linkData.link.origin_slot, true),
                destinations: []
            });
        }
        linksByOrigin.get(key).destinations.push({
            dstNode: linkData.dstNode,
            targetSlot: linkData.link.target_slot,
            linkId: linkData.linkId
        });
    }
    
    // Sistema di griglia per posizionamento non sovrapposto
    const GRID_SIZE = 80;
    const SET_OFFSET_X = 250; // Distanza dal nodo sorgente
    const GET_OFFSET_X = -220; // Distanza dal nodo destinazione

    const isAutoLinkNode = (node) => node?.type === SET_TYPE || node?.type === GET_TYPE;
    function shouldTreatAsObstacle(node) {
        if (!node) return false;
        if (packingMode === "AvoidAll") return true;
        if (packingMode === "AvoidNonAutoLink") return !isAutoLinkNode(node);
        return true;
    }

    function seriesCentered(firstUp = true) {
        // 0, -1, +1, -2, +2 ... (firstUp=true) oppure 0, +1, -1, +2, -2
        return (i) => {
            if (i === 0) return 0;
            const k = Math.ceil(i / 2);
            const sign = (i % 2 === 1) ? (firstUp ? -1 : 1) : (firstUp ? 1 : -1);
            return sign * k;
        };
    }

    function layoutDelta(i, mode) {
        // Backward compat: accetta anche i vecchi nomi interni usati durante lo sviluppo
        switch (mode) {
            case "TopToDown":
            case "stack_down":
                return { dx: 0, dy: i };
            case "BottomToTop":
            case "stack_up":
                return { dx: 0, dy: -i };
            case "CenterUpDown":
            case "stack_center_up":
                return { dx: 0, dy: seriesCentered(true)(i) };
            case "CenterDownUp":
            case "stack_center_down":
                return { dx: 0, dy: seriesCentered(false)(i) };
            case "AlignX_Right":
            case "row_right":
                return { dx: i, dy: 0 };
            case "AlignX_Left":
            case "row_left":
                return { dx: -i, dy: 0 };
            case "Columns_Down":
            case "columns_down": {
                const rows = 10;
                const col = Math.floor(i / rows);
                const row = i % rows;
                return { dx: col, dy: row };
            }
            case "Columns_Up":
            case "columns_up": {
                const rows = 10;
                const col = Math.floor(i / rows);
                const row = i % rows;
                return { dx: col, dy: -row };
            }
            case "Rake_Down":
            case "rake_down":
                return { dx: i, dy: i };
            case "Rake_Up":
            case "rake_up":
                return { dx: i, dy: -i };
            case "Proportional":
                // Keep Y anchored; move horizontally if collisions
                return { dx: seriesCentered(true)(i), dy: 0 };
            default:
                return { dx: 0, dy: seriesCentered(true)(i) };
        }
    }

    function getAnchorY(node, slot, isOutput) {
        if (!node) return 0;
        // Prefer LiteGraph connector Y (best match with what you see on screen)
        if (typeof node.getConnectionPos === 'function') {
            const isInput = !isOutput;
            const p = node.getConnectionPos(isInput, slot);
            if (Array.isArray(p) && p.length >= 2 && Number.isFinite(p[1])) return p[1];
        }
        // Fallback: proportional inside node bounding box
        const y = node.pos?.[1] ?? 0;
        const h = node.size?.[1] ?? 100;
        const count = isOutput ? (node.outputs?.length || 1) : (node.inputs?.length || 1);
        const idx = Number.isFinite(slot) ? slot : 0;
        const t = (idx + 0.5) / Math.max(1, count);
        return y + t * h;
    }

    function findFreePosition(graph, baseX, baseY, offsetX, occupiedPositions, mode) {
        const maxAttempts = 250;

        for (let attempts = 0; attempts < maxAttempts; attempts++) {
            const { dx, dy } = layoutDelta(attempts, mode);
            const testX = baseX + offsetX + dx * GRID_SIZE;
            const testY = baseY + dy * GRID_SIZE;
            const posKey = `${Math.round(testX / GRID_SIZE)}_${Math.round(testY / GRID_SIZE)}`;

            if (occupiedPositions.has(posKey)) continue;

            // Controlla sovrapposizione con nodi esistenti
            let overlaps = false;
            for (const node of _iamccsGraphNodes(graph)) {
                if (!shouldTreatAsObstacle(node)) continue;
                const nodeRight = node.pos[0] + (node.size?.[0] || 200);
                const nodeBottom = node.pos[1] + (node.size?.[1] || 100);

                if (testX < nodeRight && testX + 150 > node.pos[0] &&
                    testY < nodeBottom && testY + 26 > node.pos[1]) {
                    overlaps = true;
                    break;
                }
            }

            if (!overlaps) {
                occupiedPositions.add(posKey);
                return [testX, testY];
            }
        }

        // Fallback
        return [baseX + offsetX, baseY];
    }
    
    const occupiedSetPositions = new Set();
    const occupiedGetPositions = new Set();
    const createdSets = new Map();
    
    // Traccia i nomi dei Set già esistenti/creati per evitare duplicati
    // - Preserva nomi numerati come "model_0" (non li riduce a "model")
    const usedExactSetNames = new Set();
    const NEVER = (window?.LiteGraph?.NEVER != null) ? window.LiteGraph.NEVER : 2;
    const isInactiveSetForNaming = (n) => !!(n?.flags?.hidden || n?.hidden || n?.is_hidden || Number(n?.mode) === Number(NEVER));
    const existingSets = _iamccsGraphNodes(graph).filter(n => n.type === SET_TYPE && !isInactiveSetForNaming(n));
    for (const existingSet of existingSets) {
        const name = getAutolinkKey(existingSet);
        if (name && String(name).trim()) usedExactSetNames.add(String(name).trim());
    }

    function makeUniqueSetName(desiredName) {
        const desired = String(desiredName ?? "").trim();
        if (!desired) {
            let i = 0;
            while (usedExactSetNames.has(`output_${i}`)) i++;
            const fallback = `output_${i}`;
            usedExactSetNames.add(fallback);
            return fallback;
        }

        if (!usedExactSetNames.has(desired)) {
            usedExactSetNames.add(desired);
            return desired;
        }

        const m = desired.match(/^(.+?)_(\d+)$/);
        if (m) {
            const base = m[1];
            let n = parseInt(m[2], 10) + 1;
            while (usedExactSetNames.has(`${base}_${n}`)) n++;
            const candidate = `${base}_${n}`;
            usedExactSetNames.add(candidate);
            return candidate;
        }

        let n = 0;
        while (usedExactSetNames.has(`${desired}_${n}`)) n++;
        const candidate = `${desired}_${n}`;
        usedExactSetNames.add(candidate);
        return candidate;
    }
    
    console.log(`[IAMCCS AutoLink] Creating ${linksByOrigin.size} Set nodes...`);
    
    // Crea Set nodes (uno per origine)
    for (const [key, originData] of linksByOrigin) {
        const { srcNode, originSlot, outputName, destinations } = originData;
        
        const setPos = findFreePosition(
            graph, 
            srcNode.pos[0] + (srcNode.size?.[0] || 200), 
            (alignMode === "Proportional" ? getAnchorY(srcNode, originSlot, true) : srcNode.pos[1]),
            20,
            occupiedSetPositions,
            alignMode
        );
        
        const setNode = createNode(graph, SET_TYPE, setPos[0], setPos[1]);
        if (!setNode) continue;

        // If the source is inside a hidden/disabled group, the created AutoLink must follow.
        _iamccsApplyGroupStateToNode(
            graph,
            setNode,
            srcNode,
            { x: setPos[0] + 75, y: setPos[1] + 13 }
        );

        setNode.properties = setNode.properties || {};
        setNode.properties.autolink_color_name = colorSet;
        if (setNode.properties.autolink_color_locked === undefined) setNode.properties.autolink_color_locked = false;
        applyNodeColors(setNode, getAutolinkColorPreset(colorSet, 'set', separateCol, colorGet));
        applyNodeTitleTextColor(setNode, colorTitles);
        
        // Ottieni il tipo dall'output del nodo sorgente
        const outputType = srcNode.outputs?.[originSlot]?.type || "*";
        let outputSlotName = getSlotName(srcNode, originSlot, true);

        // Ulteriore fix: non permettere mai "*" come chiave
        if (!outputSlotName || String(outputSlotName).trim() === "*") {
            if (outputType && outputType !== "*") outputSlotName = String(outputType).trim().toLowerCase();
            else outputSlotName = `output_${originSlot}`;
        }
        
        console.log(`[IAMCCS AutoLink] Processing: ${srcNode.title || srcNode.type}[${originSlot}] with name "${outputSlotName}"`);
        
        // Genera nome unico se esiste già un Set con questo nome.
        // Importante: NON ridurre mai "model_0" a "model".
        const uniqueName = makeUniqueSetName(outputSlotName);
        
        // Imposta tipo e nome correttamente
        if (setNode.inputs && setNode.inputs[0]) {
            setNode.inputs[0].type = outputType;
            setNode.inputs[0].name = uniqueName;
        }
        if (setNode.outputs && setNode.outputs[0]) {
            setNode.outputs[0].type = outputType;
            setNode.outputs[0].name = uniqueName;
        }
        
        setWidgetValue(setNode, "name", uniqueName);
        setNode.title = `${uniqueName}`;
        
        console.log(`[IAMCCS AutoLink] ✓ Created Set node: "${uniqueName}" (from ${srcNode.title || srcNode.type})`);
        
        // Inizializza lastValue per il tracking delle modifiche
        const nameWidget = getWidget(setNode, "name");
        if (nameWidget) nameWidget.lastValue = uniqueName;
        
        // Collassa
        setTimeout(() => {
            if (setNode.collapse) setNode.collapse();
            setNode.size = [150, 26];
        }, 0);
        
        // Collega il Set al nodo sorgente
        srcNode.connect(originSlot, setNode, 0);
        
        // Salva il Set creato per creare i Get dopo
        createdSets.set(key, {
            setNode,
            outputName: uniqueName,
            outputType,
            srcNode,
            originSlot,
            destinations
        });
    }
    
    // Crea Get nodes (uno per destinazione)
    for (const [key, setData] of createdSets) {
        const { setNode, outputName, outputType, srcNode, originSlot, destinations } = setData;

        const sortedDest = [...destinations].sort((a, b) => {
            const ay = a.dstNode?.pos?.[1] ?? 0;
            const by = b.dstNode?.pos?.[1] ?? 0;
            const ax = a.dstNode?.pos?.[0] ?? 0;
            const bx = b.dstNode?.pos?.[0] ?? 0;
            if (alignMode === "Proportional") {
                const as = Number.isFinite(a.targetSlot) ? a.targetSlot : 0;
                const bs = Number.isFinite(b.targetSlot) ? b.targetSlot : 0;
                if (as !== bs) return as - bs;
            }
            if (alignMode === "AlignX_Right" || alignMode === "AlignX_Left" || alignMode === "row_right" || alignMode === "row_left") {
                return ax - bx;
            }
            return ay - by;
        });

        for (const dest of sortedDest) {
            const { dstNode, targetSlot, linkId } = dest;
            
            const getPos = findFreePosition(
                graph,
                dstNode.pos[0],
                (alignMode === "Proportional" ? getAnchorY(dstNode, targetSlot, false) : dstNode.pos[1]),
                GET_OFFSET_X,
                occupiedGetPositions,
                alignMode
            );
            
            const getNode = createNode(graph, GET_TYPE, getPos[0], getPos[1]);
            if (!getNode) continue;

            // If the destination is inside a hidden/disabled group, the created AutoLink must follow.
            _iamccsApplyGroupStateToNode(
                graph,
                getNode,
                dstNode,
                { x: getPos[0] + 75, y: getPos[1] + 13 }
            );

            getNode.properties = getNode.properties || {};
            // il get segue sempre il colore della sua chiave (quindi del set)
            getNode.properties.autolink_color_name = setNode.properties?.autolink_color_name || colorSet;
            applyNodeColors(getNode, getAutolinkColorPreset(getNode.properties.autolink_color_name, 'get', separateCol, colorGet));
            applyNodeTitleTextColor(getNode, colorTitles);
            
            // Imposta tipo e nome correttamente
            if (getNode.outputs && getNode.outputs[0]) {
                getNode.outputs[0].type = outputType;
                getNode.outputs[0].name = outputName;
            }
            
            setWidgetValue(getNode, "name", outputName);
            getNode.title = `${outputName}`;
            
            // Collassa
            setTimeout(() => {
                if (getNode.collapse) getNode.collapse();
                getNode.size = [150, 26];
            }, 0);
            
            // Salva metadata per restore
            const metadata = {
                iamccs_autolink: true,
                output_name: outputName,
                origin: { id: srcNode.id, slot: originSlot },
                target: { id: dstNode.id, slot: targetSlot }
            };
            
            setNode.properties = setNode.properties || {};
            getNode.properties = getNode.properties || {};
            if (!setNode.properties.metadata) {
                setNode.properties.metadata = metadata;
            }
            getNode.properties.metadata = metadata;

            // Safe rewire: preserve the previous direct link if any.
            const ts = Number(targetSlot);
            if (!Number.isFinite(ts)) {
                try { graph.remove(getNode); } catch {}
                continue;
            }

            let prev = null;
            try {
                const prevId = dstNode?.inputs?.[ts]?.link;
                const prevLink = prevId != null ? _iamccsGetLink(graph, prevId) : null;
                if (prevLink) prev = { origin_id: prevLink.origin_id, origin_slot: prevLink.origin_slot };
            } catch {}

            // 1) Try connect without disconnect (some graphs auto-replace). If it fails, disconnect and retry.
            let ok = _safeConnect(getNode, 0, dstNode, ts);
            if (ok === null) {
                _iamccsDisconnectTargetInput(graph, dstNode, ts);
                ok = _safeConnect(getNode, 0, dstNode, ts);
            }

            if (ok === null) {
                // Rollback previous direct link
                if (prev) {
                    try {
                        const prevSrc = getNodeById(graph, prev.origin_id);
                        if (prevSrc) _safeConnect(prevSrc, prev.origin_slot, dstNode, ts);
                    } catch {}
                }
                try { graph.remove(getNode); } catch {}
                continue;
            }

            // Remove any other links competing for this target input
            _iamccsRemoveOtherLinksToTarget(graph, dstNode.id, ts, ok);
        }
    }
    
    // Converti KijNodes in AutoLink se richiesto
    if (includeKijNodes && kijNodesToConvert.length > 0) {
        // NOTE: kijNodesToConvert is collected per-link; converting/removing nodes inside that loop
        // can lead to partial conversions (Set converted, Get left behind) and broken workflows.
        // Fix: dedupe by node id and convert BOTH src and dst KJ nodes.
        const kijNodesById = new Map();
        for (const { srcNode, dstNode, srcIsKij, dstIsKij } of kijNodesToConvert) {
            if (srcIsKij && srcNode?.id != null) kijNodesById.set(String(srcNode.id), srcNode);
            if (dstIsKij && dstNode?.id != null) kijNodesById.set(String(dstNode.id), dstNode);
        }

        const kijNodes = [...kijNodesById.values()].filter(n => n && (n.type === KJ_SET_TYPE || n.type === KJ_GET_TYPE));
        console.log(`[IAMCCS AutoLink] Converting ${kijNodes.length} KijNodes (deduped)...`);

        // Snapshot link entries before we start removing nodes.
        const linkEntriesSnapshot = _iamccsGraphLinksEntries(graph)
            .map(([id, l]) => [id, l])
            .filter(([, l]) => !!l);

        const getKijName = (node) => {
            try {
                // Prefer a named widget when present; fallback to first widget value.
                const byName = getWidgetValue(node, "name");
                const v = (byName != null && String(byName).trim()) ? byName : node?.widgets?.[0]?.value;
                const s = String(v ?? "output").trim();
                return s || "output";
            } catch {
                return "output";
            }
        };

        const outgoingFor = (nodeId) => {
            const out = [];
            for (const [, l] of linkEntriesSnapshot) {
                if (_iamccsIdEq(l?.origin_id, nodeId)) out.push(l);
            }
            return out;
        };

        const incomingFor = (nodeId) => {
            const inc = [];
            for (const [, l] of linkEntriesSnapshot) {
                if (_iamccsIdEq(l?.target_id, nodeId)) inc.push(l);
            }
            return inc;
        };

        // Convert nodes in a stable order (left-to-right, top-to-bottom) to minimize visual jitter.
        kijNodes.sort((a, b) => {
            const ax = Array.isArray(a?.pos) ? a.pos[0] : 0;
            const ay = Array.isArray(a?.pos) ? a.pos[1] : 0;
            const bx = Array.isArray(b?.pos) ? b.pos[0] : 0;
            const by = Array.isArray(b?.pos) ? b.pos[1] : 0;
            return ax === bx ? (ay - by) : (ax - bx);
        });

        for (const oldNode of kijNodes) {
            // Node might already be removed by a previous conversion step.
            if (!oldNode || oldNode.id == null) continue;
            const stillThere = getNodeById(graph, oldNode.id);
            if (!stillThere || stillThere.type !== oldNode.type) continue;

            const kijName = getKijName(oldNode);
            const x = Array.isArray(oldNode.pos) ? oldNode.pos[0] : 0;
            const y = Array.isArray(oldNode.pos) ? oldNode.pos[1] : 0;

            const newType = (oldNode.type === KJ_SET_TYPE) ? SET_TYPE : GET_TYPE;
            const newNode = createNode(graph, newType, x, y);
            if (!newNode) continue;

            // Preserve group hidden/disabled state (and node mode) from the original.
            _iamccsApplyGroupStateToNode(
                graph,
                newNode,
                oldNode,
                { x: x + 75, y: y + 13 }
            );

            setWidgetValue(newNode, "name", kijName);
            newNode.title = `${kijName}`;

            // Ensure expected slots exist.
            if (newType === SET_TYPE) {
                normalizeAutolinkIOSlots(graph, newNode, { wantInputs: 1, wantOutputs: 1 });
            } else {
                normalizeAutolinkIOSlots(graph, newNode, { wantInputs: 0, wantOutputs: 1 });
            }

            const oldId = oldNode.id;
            const outLinks = outgoingFor(oldId);
            const inLinks = incomingFor(oldId);

            let rewiredOutgoing = 0;
            let rewiredIncoming = 0;

            // Rewire outgoing links (covers GetNode outputs, and any SetNode passthrough outputs).
            for (const l of outLinks) {
                const dst = getNodeById(graph, l.target_id);
                if (!dst) continue;
                const ts = Number(l.target_slot);
                if (!Number.isFinite(ts)) continue;

                // Prefer slot 0 (AutoLink nodes are 1-output helpers)
                let ok = _safeConnect(newNode, 0, dst, ts);
                if (ok === null) {
                    _iamccsDisconnectTargetInput(graph, dst, ts);
                    ok = _safeConnect(newNode, 0, dst, ts);
                }
                if (ok !== null) {
                    rewiredOutgoing++;
                    _iamccsRemoveOtherLinksToTarget(graph, dst.id, ts, ok);
                }
            }

            // Rewire incoming links ONLY for SetNode (it is an input sink).
            if (newType === SET_TYPE) {
                // SetNode should have a single input; if multiple exist, keep the first.
                const first = inLinks.find(l => Number(l.target_slot) === 0) || inLinks[0];
                if (first) {
                    const src = getNodeById(graph, first.origin_id);
                    const os = Number(first.origin_slot);
                    if (src && Number.isFinite(os)) {
                        let ok = _safeConnect(src, os, newNode, 0);
                        if (ok === null) {
                            _iamccsDisconnectTargetInput(graph, newNode, 0);
                            ok = _safeConnect(src, os, newNode, 0);
                        }
                        if (ok !== null) rewiredIncoming++;
                    }
                }
            }

            // Decide whether to remove the original node.
            // If it had connections and we couldn't rewire them, rollback by removing the new node.
            const hadOut = outLinks.length > 0;
            const hadIn = inLinks.length > 0;

            const outgoingOk = !hadOut || rewiredOutgoing > 0;
            const incomingOk = (newType !== SET_TYPE) || (!hadIn) || rewiredIncoming > 0;

            if (outgoingOk && incomingOk) {
                try { graph.remove(oldNode); } catch {}
            } else {
                console.warn(
                    `[IAMCCS AutoLink] KijNode conversion incomplete for node ${oldNode.id} (${oldNode.type}). Keeping original.`
                );
                try { graph.remove(newNode); } catch {}
            }
        }
    }
    
    console.log(`[IAMCCS AutoLink] ✓ Converted ${linksToConvert.length} links`);
    // Normalizza colori (utile se ci sono già Set/Get in scena)
    recolorExistingAutoLinks(graph, colorSet, separateCol, colorGet, colorTitles);
    _iamccsFixLinkIntegrity(graph);
    graph.setDirtyCanvas(true, true);
}

function restoreDirectLinks(graph, options = {}) {
    const opts = {
        // When true, removes Set/Get nodes after successful restore (UI button behavior)
        removeNodes: true,
        // When true, removal is delayed via setTimeout to let LiteGraph settle
        asyncRemove: true,
        // When true, prune duplicate links to the same target input
        // (dangerous on some workflows; default off)
        pruneTargetDuplicates: false,
        ...options,
    };

    console.log("[IAMCCS AutoLink] Restoring links...");

    // Safety: snapshot the graph so we can rollback if restore fails.
    let __iamccsSnapshot = null;
    let __iamccsLinkCountBefore = 0;
    try {
        __iamccsSnapshot = (typeof graph?.serialize === "function") ? graph.serialize() : null;
        __iamccsLinkCountBefore = _iamccsGraphLinksEntries(graph).length;
    } catch (e) {
        __iamccsSnapshot = null;
        __iamccsLinkCountBefore = 0;
    }

    const nodes = _iamccsGraphNodes(graph);
    const setNodes = [];
    const getNodes = [];

    const NEVER = (window?.LiteGraph?.NEVER != null) ? window.LiteGraph.NEVER : 2;
    const isInactiveAutoLinkNode = (n) => {
        if (!n) return true;
        // LiteGraph/ComfyUI can represent hidden/disabled in several ways.
        if (n?.flags?.hidden || n?.hidden || n?.is_hidden) return true;
        if (Number(n?.mode) === Number(NEVER)) return true;
        return false;
    };

    const hasAnyGraphLink = (nodeId) => {
        if (nodeId == null) return false;
        try {
            for (const [, l] of _iamccsGraphLinksEntries(graph)) {
                if (!l) continue;
                if (_iamccsIdEq(l.origin_id, nodeId) || _iamccsIdEq(l.target_id, nodeId)) return true;
            }
        } catch {}
        return false;
    };
    
    // Raccogli tutti i nodi Set e Get
    for (const node of nodes) {
        if (node.type === SET_TYPE) {
            // For UI "restore all links" we want to restore/remove *everything*.
            // For queuePrompt execution (removeNodes=false) we must ignore hidden/inactive autolinks.
            if (!opts.removeNodes && isInactiveAutoLinkNode(node)) continue;
            setNodes.push(node);
        } else if (node.type === GET_TYPE) {
            if (!opts.removeNodes && isInactiveAutoLinkNode(node)) continue;
            getNodes.push(node);
        }
    }
    
    console.log(`[IAMCCS AutoLink] Found ${setNodes.length} Set nodes, ${getNodes.length} Get nodes`);

    if (setNodes.length === 0 && getNodes.length === 0) {
        console.warn("[IAMCCS AutoLink] No AutoLink nodes found; restore skipped (graph format mismatch or nothing to restore)");
        return;
    }
    
    // Raggruppa per nome
    const byName = new Map();
    
    for (const setNode of setNodes) {
        const name = getAutolinkKey(setNode);
        if (!name) continue;
        
        if (!byName.has(name)) {
            byName.set(name, { set: null, gets: [] });
        }
        byName.get(name).set = setNode;
    }
    
    for (const getNode of getNodes) {
        const name = getAutolinkKey(getNode);
        if (!name) continue;
        
        if (!byName.has(name)) {
            byName.set(name, { set: null, gets: [] });
        }
        byName.get(name).gets.push(getNode);
    }

    const _iamccsDidConnect = (srcNode, originSlot, dstNode, targetSlot) => {
        try {
            const ts = Number(targetSlot);
            const os = Number(originSlot);
            const linkId = dstNode?.inputs?.[ts]?.link;
            if (linkId == null) return null;
            const link = _iamccsGetLink(graph, linkId);
            if (!link) return null;
            if (!_iamccsIdEq(link.origin_id, srcNode.id)) return null;
            if (!_iamccsIdEq(link.target_id, dstNode.id)) return null;
            if (Number(link.origin_slot) !== os) return null;
            if (Number(link.target_slot) !== ts) return null;
            return linkId;
        } catch (e) {
            return null;
        }
    };
    
    const safeConnect = (srcNode, originSlot, dstNode, targetSlot) => {
        const os = Number(originSlot);
        const ts = Number(targetSlot);
        if (!Number.isFinite(os) || !Number.isFinite(ts)) return null;

        try {
            srcNode.connect(os, dstNode, ts);
            const ok = _iamccsDidConnect(srcNode, os, dstNode, ts);
            if (ok != null) return ok;
        } catch (e) {
            // fallthrough
        }

        try {
            if (typeof graph.connect === "function") {
                graph.connect(srcNode.id, os, dstNode.id, ts);
                const ok2 = _iamccsDidConnect(srcNode, os, dstNode, ts);
                if (ok2 != null) return ok2;
            }
        } catch (e) {
            // fallthrough
        }

        try {
            if (typeof graph.addLink === "function") {
                graph.addLink(srcNode, os, dstNode, ts);
                const ok3 = _iamccsDidConnect(srcNode, os, dstNode, ts);
                if (ok3 != null) return ok3;
            }
        } catch (e) {
            // fallthrough
        }

        return null;
    };

    const isTargetCurrentlyFromGetNode = (dstNode, targetSlot, getNodeId) => {
        try {
            const ts = Number(targetSlot);
            if (!Number.isFinite(ts) || !dstNode) return false;
            const linkId = dstNode?.inputs?.[ts]?.link;
            if (linkId == null) return false;
            const link = _iamccsGetLink(graph, linkId);
            if (!link) return false;
            return _iamccsIdEq(link.origin_id, getNodeId);
        } catch {
            return false;
        }
    };

    let restored = 0;
    let failed = 0;

    const keyToSet = new Map();
    for (const setNode of setNodes) {
        const key = getAutolinkKey(setNode);
        if (key) keyToSet.set(key, setNode);
    }
    const keyToGets = new Map();
    for (const getNode of getNodes) {
        const key = getAutolinkKey(getNode);
        if (!key) continue;
        if (!keyToGets.has(key)) keyToGets.set(key, []);
        keyToGets.get(key).push(getNode);
    }

    const restoredGetIds = new Set();
    const keysWithFailures = new Set();

    for (const getNode of getNodes) {
        const key = getAutolinkKey(getNode) || "";
        const md = getNode?.properties?.metadata;
        const origin = md?.origin;
        const target = md?.target;

        if (!origin?.id || origin?.slot === undefined || !target?.id || target?.slot === undefined) {
            continue;
        }

        const srcNode = getNodeById(graph, origin.id);
        const dstNode = getNodeById(graph, target.id);
        const originSlot = origin.slot;
        const targetSlot = target.slot;

        if (!srcNode || !dstNode) {
            failed++;
            if (key) keysWithFailures.add(key);
            continue;
        }

        const ts = Number(targetSlot);
        if (!Number.isFinite(ts)) {
            failed++;
            if (key) keysWithFailures.add(key);
            continue;
        }

        // If the target is already wired to the expected source, treat it as restored.
        // This prevents leftover Get/Set nodes when the user runs restore multiple times
        // or when earlier restores already rewired the connection.
        try {
            const existingId = dstNode?.inputs?.[ts]?.link;
            if (existingId != null) {
                const existing = _iamccsGetLink(graph, existingId);
                if (existing && _iamccsIdEq(existing.origin_id, srcNode.id) && Number(existing.origin_slot) === Number(originSlot)) {
                    restored++;
                    restoredGetIds.add(getNode.id);
                    continue;
                }
            }
        } catch {}

        let hadGetLink = false;
        try {
            const out = getNode.outputs?.[0];
            hadGetLink = !!(out?.links && out.links.length);
        } catch (e) {
            hadGetLink = false;
        }

        // Non-destructive restore:
        // - Only touch the target input if it's currently fed by this Get node.
        // - Never delete other links (unless opts.pruneTargetDuplicates is explicitly enabled).
        let ok = null;
        const shouldReplace = isTargetCurrentlyFromGetNode(dstNode, ts, getNode.id);
        if (shouldReplace) {
            _iamccsDisconnectTargetInput(graph, dstNode, ts);
            ok = safeConnect(srcNode, originSlot, dstNode, ts);
        } else {
            // If user has already changed wiring, do not override it.
            // Still try a safe connect without disconnect (won't usually succeed if input is taken).
            ok = safeConnect(srcNode, originSlot, dstNode, ts);
        }
        if (ok === null) {
            failed++;
            if (key) keysWithFailures.add(key);

            if (hadGetLink) {
                try { getNode.connect(0, dstNode, ts); } catch (e) {}
            }
            continue;
        }

        // Optional pruning (disabled by default; can break complex graphs)
        if (opts.pruneTargetDuplicates) {
            _iamccsRemoveOtherLinksToTarget(graph, dstNode.id, ts, ok);
        }

        restored++;
        restoredGetIds.add(getNode.id);
    }

    if (restored === 0) {
        console.warn("[IAMCCS AutoLink] No metadata restores performed; falling back to legacy restore");

        for (const [name, { set, gets }] of byName) {
            if (!set || !gets.length) continue;

            const setInputLink = set.inputs?.[0]?.link;
            if (!setInputLink) continue;
            const link = _iamccsGetLink(graph, setInputLink);
            if (!link) continue;

            const srcNode = getNodeById(graph, link.origin_id);
            if (!srcNode) continue;
            const originSlot = link.origin_slot;

            for (const getNode of gets) {
                const getOutput = getNode.outputs?.[0];
                if (!getOutput?.links?.length) continue;

                const outLinks = [...getOutput.links];
                for (const linkId of outLinks) {
                    const outLink = _iamccsGetLink(graph, linkId);
                    if (!outLink) continue;

                    const dstNode = getNodeById(graph, outLink.target_id);
                    if (!dstNode) continue;
                    const targetSlot = outLink.target_slot;
                    const ts = Number(targetSlot);
                    if (!Number.isFinite(ts)) continue;

                    // Already correct? Mark restored so we can remove the AutoLink nodes.
                    try {
                        const existingId = dstNode?.inputs?.[ts]?.link;
                        if (existingId != null) {
                            const existing = _iamccsGetLink(graph, existingId);
                            if (existing && _iamccsIdEq(existing.origin_id, srcNode.id) && Number(existing.origin_slot) === Number(originSlot)) {
                                restored++;
                                restoredGetIds.add(getNode.id);
                                continue;
                            }
                        }
                    } catch {}

                    // Same non-destructive semantics as metadata restore.
                    let ok = null;
                    const shouldReplace = isTargetCurrentlyFromGetNode(dstNode, ts, getNode.id);
                    if (shouldReplace) {
                        _iamccsDisconnectTargetInput(graph, dstNode, ts);
                        ok = safeConnect(srcNode, originSlot, dstNode, ts);
                    } else {
                        ok = safeConnect(srcNode, originSlot, dstNode, ts);
                    }
                    if (ok === null) {
                        failed++;
                        keysWithFailures.add(name);
                        continue;
                    }

                    if (opts.pruneTargetDuplicates) {
                        _iamccsRemoveOtherLinksToTarget(graph, dstNode.id, ts, ok);
                    }

                    restored++;
                    restoredGetIds.add(getNode.id);
                }
            }
        }
    }

    // Rollback if we made things worse (e.g. SubgraphNode refuses connections).
    try {
        const after = _iamccsGraphLinksEntries(graph).length;
        if (__iamccsSnapshot && restored === 0 && failed > 0 && after < __iamccsLinkCountBefore) {
            console.warn("[IAMCCS AutoLink] Restore appears to have reduced links; rolling back snapshot");
            if (typeof graph?.configure === "function") {
                graph.configure(__iamccsSnapshot);
                try { _iamccsFixLinkIntegrity(graph); } catch {}
                try { graph.setDirtyCanvas(true, true); } catch {}
            }
            return;
        }
    } catch {}

    const nodesToRemove = new Set();
    if (opts.removeNodes) {
        // Remove restored Gets + any dangling Gets (prevents Convert/Restore duplication loops).
        for (const getNode of getNodes) {
            if (!getNode) continue;
            if (restoredGetIds.has(getNode.id)) {
                nodesToRemove.add(getNode);
                continue;
            }
            // If a Get is no longer connected to anything, it is safe to remove.
            if (!hasAnyGraphLink(getNode.id)) {
                nodesToRemove.add(getNode);
            }
        }

        // Remove Sets when their Gets are fully restored OR when there are no Gets at all for that key.
        // A Set without Gets is useless after restore and causes name-suffix explosions on the next Convert.
        for (const [key, setNode] of keyToSet) {
            if (!setNode) continue;
            const gets = keyToGets.get(key) || [];
            const allGetsRestored = gets.length > 0 && gets.every(g => restoredGetIds.has(g.id));
            const noGetsExist = gets.length === 0;

            // If restore had explicit failures for this key, keep the Set to avoid data loss.
            if (keysWithFailures.has(key)) continue;

            // If the Set isn't linked to anything, remove it unconditionally.
            if (!hasAnyGraphLink(setNode.id)) {
                nodesToRemove.add(setNode);
                continue;
            }

            if (allGetsRestored || noGetsExist) {
                nodesToRemove.add(setNode);
            }
        }
    }

    console.log(`[IAMCCS AutoLink] Removing ${nodesToRemove.size} AutoLink nodes (safe mode)...`);

    // Make sure UI updates immediately after link rewiring
    try { _iamccsFixLinkIntegrity(graph); } catch (e) {}
    try { graph.setDirtyCanvas(true, true); } catch (e) {}
    
    const removeNow = () => {
        for (const node of nodesToRemove) {
            try {
                // Disconnetti tutti i collegamenti prima di rimuovere
                if (node.inputs) {
                    for (let i = node.inputs.length - 1; i >= 0; i--) {
                        try { node.disconnectInput?.(i); } catch {}
                    }
                }
                if (node.outputs) {
                    for (let i = node.outputs.length - 1; i >= 0; i--) {
                        try { node.disconnectOutput?.(i); } catch {}
                    }
                }
                graph.remove(node);
            } catch (e) {
                console.error(`[IAMCCS AutoLink] Error removing node ${node.id}:`, e);
            }
        }
        try { _iamccsFixLinkIntegrity(graph); } catch {}
        try { graph.setDirtyCanvas(true, true); } catch {}
    };

    // Async removal is ONLY for manual UI use; never do it in queuePrompt.
    if (nodesToRemove.size > 0) {
        if (opts.asyncRemove) setTimeout(removeNow, 100);
        else removeNow();
    }

    console.log(`[IAMCCS AutoLink] ✓ Restored ${restored} links`);
    if (failed > 0) console.log(`[IAMCCS AutoLink] ⚠ Restore failures: ${failed}`);
    console.log(`[IAMCCS AutoLink] ✓ Removed ${nodesToRemove.size} AutoLink nodes`);
}

console.log("[IAMCCS AutoLink] Extension loaded");

// ---- Runtime patch: ensure AutoLink graphs can execute ----
// AutoLink Set/Get nodes are frontend helpers; the backend nodes are no-op.
// To run a workflow, we temporarily restore direct links before queueing,
// then reload the original graph so the user keeps AutoLink nodes.
function _iamccsPatchQueuePromptForAutolink() {
    try {
        if (app.__iamccs_autolink_queue_patch_installed) return;
        if (typeof app?.queuePrompt !== "function") return;

        const originalQueuePrompt = app.queuePrompt;
        app.queuePrompt = async function (...args) {
            const graph = app?.graph;
            const hasAutoLink = _iamccsGraphNodes(graph).some(n => n?.type === SET_TYPE || n?.type === GET_TYPE);
            if (!hasAutoLink) {
                return await originalQueuePrompt.apply(this, args);
            }

            let snapshot = null;
            try {
                snapshot = typeof graph.serialize === "function" ? graph.serialize() : null;
            } catch (e) {
                snapshot = null;
            }

            try {
                // Convert AutoLink nodes into direct links (and remove them) for execution.
                // This mutates the live graph, so we restore from snapshot in finally.
                try {
                    restoreDirectLinks(graph, { removeNodes: false, asyncRemove: false, pruneTargetDuplicates: false });
                } catch (e) {
                    console.warn("[IAMCCS AutoLink] restoreDirectLinks failed before queuePrompt", e);
                }

                try { _iamccsFixLinkIntegrity(graph); } catch {}

                return await originalQueuePrompt.apply(this, args);
            } finally {
                // Restore the original AutoLink graph for the UI.
                if (snapshot) {
                    try {
                        if (typeof app.loadGraphData === "function") {
                            await app.loadGraphData(snapshot);
                        } else if (typeof graph?.configure === "function") {
                            graph.configure(snapshot);
                            graph.setDirtyCanvas?.(true, true);
                        }
                    } catch (e) {
                        console.warn("[IAMCCS AutoLink] Failed to restore graph snapshot after queuePrompt", e);
                    }
                }
            }
        };

        app.__iamccs_autolink_queue_patch_installed = true;
        console.log("[IAMCCS AutoLink] Patched app.queuePrompt (temporary restore for execution)");
    } catch (e) {
        console.warn("[IAMCCS AutoLink] Failed to patch queuePrompt", e);
    }
}

_iamccsPatchQueuePromptForAutolink();

