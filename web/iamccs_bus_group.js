// IAMCCS Bus Group - frontend-only group mute/solo utility
// Inspired by rgthree "Fast Groups Muter" (UI pattern only).

import { app } from "../../scripts/app.js";

const NODE_TYPE = "IAMCCS_bus_group";
const ENABLED_MODE = 0;   // LiteGraph.ALWAYS
const MAX_CHANNELS = 32;

function getMuteMode() {
    // ComfyUI uses LiteGraph.NEVER for mute; some builds also have a separate "bypass" mode.
    return window?.LiteGraph?.NEVER ?? 2;
}

function getGraph() {
    return app?.canvas?.getCurrentGraph?.() || app?.graph || null;
}

function getGroups(graph) {
    const groups = graph?._groups || graph?.groups;
    return Array.isArray(groups) ? groups.filter(Boolean) : [];
}

function getGroupKey(group) {
    if (!group) return "";

    // Prefer a real id if the LiteGraph build provides one.
    const gid = group?.id ?? group?._id ?? group?.uid ?? group?._uid;
    if (gid != null) return `id:${String(gid)}`;

    // Use a stable key that does NOT change while dragging.
    // The previous implementation used the *current* position, which changes on drag and caused:
    // - groupKey mismatch -> saved mute/solo state not found -> defaults applied
    // - applyModes() enabling everything
    // We fix this by capturing the initial position once per group instance.
    const props = (group.properties && typeof group.properties === "object") ? group.properties : null;
    const stored = props?._iamccs_stable_key || group._iamccs_stable_key;
    if (stored) return String(stored);

    const title = String(group?.title ?? "");
    let initX = null;
    let initY = null;

    // 1) If group already has an init-pos recorded, keep using it.
    try {
        const p = props?._iamccs_init_pos || group._iamccs_init_pos;
        if (Array.isArray(p) && p.length >= 2) {
            initX = Math.round(Number(p[0] ?? 0));
            initY = Math.round(Number(p[1] ?? 0));
        }
    } catch {}

    // 2) Otherwise capture current pos as the initial seed.
    if (initX == null || initY == null) {
        initX = Math.round(Number(group?.pos?.[0] ?? 0));
        initY = Math.round(Number(group?.pos?.[1] ?? 0));
        try { group._iamccs_init_pos = [initX, initY]; } catch {}
        try {
            if (props) props._iamccs_init_pos = [initX, initY];
        } catch {}
    }

    // Title alone can collide; title + init-pos stays stable across drags.
    const key = `${title}@@${initX},${initY}`;
    try { group._iamccs_stable_key = key; } catch {}
    try { if (props) props._iamccs_stable_key = key; } catch {}
    return key;
}

function getGroupNodes(group, graph) {
    const g = graph || group?.graph || getGraph();
    const out = [];
    const seen = new Set();

    function addNode(n) {
        if (!n) return;
        const id = n?.id;
        if (id != null) {
            if (seen.has(id)) return;
            seen.add(id);
        }
        out.push(n);
    }

    function resolveChild(c) {
        if (!c) return null;
        const LGraphNode = window?.LGraphNode;
        if (LGraphNode && c instanceof LGraphNode) return c;
        if (c?.constructor?.name === "LGraphNode") return c;

        // Sometimes children may be node ids.
        const id = (typeof c === "number") ? c : c?.id;
        if (id != null && g?.getNodeById) {
            return g.getNodeById(id);
        }
        return null;
    }

    try {
        const candidates = [group?._children, group?._nodes, group?.nodes];
        for (const col of candidates) {
            if (!col) continue;
            if (typeof col[Symbol.iterator] !== "function") continue;
            for (const c of col) {
                addNode(resolveChild(c));
            }
        }
    } catch {}

    // Fallback: infer by geometry (nodes whose pos is inside group bounds).
    try {
        if (out.length === 0 && g) {
            const nodes = g?._nodes || g?.nodes;
            const gx = Number(group?.pos?.[0] ?? 0);
            const gy = Number(group?.pos?.[1] ?? 0);
            const gw = Number(group?.size?.[0] ?? 0);
            const gh = Number(group?.size?.[1] ?? 0);
            if (Array.isArray(nodes) && gw > 0 && gh > 0) {
                for (const n of nodes) {
                    const nx = Number(n?.pos?.[0] ?? NaN);
                    const ny = Number(n?.pos?.[1] ?? NaN);
                    if (!Number.isFinite(nx) || !Number.isFinite(ny)) continue;
                    if (nx >= gx && ny >= gy && nx <= gx + gw && ny <= gy + gh) {
                        addNode(n);
                    }
                }
            }
        }
    } catch {}

    return out.filter(Boolean);
}

function _getGroupBounds(group) {
    if (!group) return null;
    const gx = Number(group?.pos?.[0] ?? NaN);
    const gy = Number(group?.pos?.[1] ?? NaN);
    const gw = Number(group?.size?.[0] ?? NaN);
    const gh = Number(group?.size?.[1] ?? NaN);
    if (![gx, gy, gw, gh].every(Number.isFinite)) return null;
    if (gw <= 0 || gh <= 0) return null;
    return { gx, gy, gw, gh };
}

function _nodeCenter(n) {
    const nx = Number(n?.pos?.[0] ?? NaN);
    const ny = Number(n?.pos?.[1] ?? NaN);
    const nw = Number(n?.size?.[0] ?? 0);
    const nh = Number(n?.size?.[1] ?? 0);
    if (!Number.isFinite(nx) || !Number.isFinite(ny)) return null;
    return { cx: nx + Math.max(0, nw) / 2, cy: ny + Math.max(0, nh) / 2 };
}

function _bestGroupForNode(graph, node) {
    const groups = getGroups(graph);
    if (!node || groups.length === 0) return null;
    const c = _nodeCenter(node);
    if (!c) return null;

    let best = null;
    let bestArea = Infinity;
    for (const g of groups) {
        const b = _getGroupBounds(g);
        if (!b) continue;
        const inside = (c.cx >= b.gx && c.cx <= b.gx + b.gw && c.cy >= b.gy && c.cy <= b.gy + b.gh);
        if (!inside) continue;
        const area = Math.abs(b.gw * b.gh);
        if (area < bestArea) {
            bestArea = area;
            best = g;
        }
    }
    return best;
}

function getGroupNodesStable(group, graph) {
    // IMPORTANT:
    // Some LiteGraph/ComfyUI builds update group membership dynamically while dragging groups.
    // When two groups touch/overlap, group._nodes can temporarily include nodes from the other group,
    // causing a muted group to get re-enabled by mistake.
    // This helper builds a stable membership list filtered by "best" group ownership.
    const g = graph || group?.graph || getGraph();
    if (!g || !group) return [];

    const explicit = getGroupNodes(group, g);
    if (explicit.length > 0) {
        // Filter explicit list by best-group ownership to avoid overlap pollution.
        const gk = getGroupKey(group);
        return explicit.filter(n => {
            const bg = _bestGroupForNode(g, n);
            return bg ? (getGroupKey(bg) === gk) : false;
        });
    }

    // If explicit is empty, infer by geometry once, but still apply best-group filtering.
    const bounds = _getGroupBounds(group);
    if (!bounds) return [];

    const nodes = g?._nodes || g?.nodes;
    if (!Array.isArray(nodes)) return [];

    const gk = getGroupKey(group);
    const inferred = [];
    for (const n of nodes) {
        const c = _nodeCenter(n);
        if (!c) continue;
        if (c.cx < bounds.gx || c.cx > bounds.gx + bounds.gw || c.cy < bounds.gy || c.cy > bounds.gy + bounds.gh) continue;
        const bg = _bestGroupForNode(g, n);
        if (bg && getGroupKey(bg) === gk) inferred.push(n);
    }
    return inferred;
}

function _resolveNodesByIds(graph, ids) {
    const g = graph || getGraph();
    const out = [];
    if (!g || !Array.isArray(ids) || ids.length === 0) return out;

    const byId = (id) => {
        try {
            if (typeof g?.getNodeById === "function") return g.getNodeById(id);
        } catch {}
        try {
            const nodes = g?._nodes || g?.nodes;
            if (Array.isArray(nodes)) return nodes.find(n => n?.id === id) || null;
        } catch {}
        return null;
    };

    for (const id of ids) {
        const n = byId(id);
        if (n) out.push(n);
    }
    return out;
}

function _matchesGroupFilters(node, group) {
    const ft = String(node?.properties?.iamccs_bus_group_filter_title || "").trim().toLowerCase();
    if (ft) {
        const t = String(group?.title || "").toLowerCase();
        if (!t.includes(ft)) return false;
    }

    const fcRaw = String(node?.properties?.iamccs_bus_group_filter_color || "").trim();
    if (fcRaw) {
        const cRaw = String(group?.color || group?.bgcolor || group?._color || group?._bgcolor || "").trim();
        const fRgb = cssColorToRgb(fcRaw);
        if (fRgb) {
            const gRgb = cssColorToRgb(cRaw);
            if (!gRgb) return false;
            if (!isColorNear(gRgb, fRgb)) return false;
        } else {
            const fc = fcRaw.toLowerCase();
            const c = cRaw.toLowerCase();
            if (!c.includes(fc)) return false;
        }
    }

    return true;
}

function cssColorToRgb(color) {
    const s = String(color || "").trim();
    if (!s) return null;
    try {
        const canvas = cssColorToRgb._c || (cssColorToRgb._c = document.createElement("canvas"));
        const ctx = canvas.getContext("2d");
        if (!ctx) return null;
        ctx.fillStyle = "#000";
        ctx.fillStyle = s;
        const normalized = String(ctx.fillStyle || "");
        if (!normalized) return null;

        if (normalized.startsWith("#")) {
            const hex = normalized.slice(1);
            const h = hex.length === 3
                ? hex.split("").map(ch => ch + ch).join("")
                : hex.padEnd(6, "0").slice(0, 6);
            const r = parseInt(h.slice(0, 2), 16);
            const g = parseInt(h.slice(2, 4), 16);
            const b = parseInt(h.slice(4, 6), 16);
            if (![r, g, b].every(Number.isFinite)) return null;
            return { r, g, b };
        }

        const m = normalized.match(/rgba?\((\d+(?:\.\d+)?)[,\s]+(\d+(?:\.\d+)?)[,\s]+(\d+(?:\.\d+)?)/i);
        if (!m) return null;
        const r = Math.round(Number(m[1]));
        const g = Math.round(Number(m[2]));
        const b = Math.round(Number(m[3]));
        if (![r, g, b].every(Number.isFinite)) return null;
        return { r, g, b };
    } catch {
        return null;
    }
}

function rgbToHsl(r, g, b) {
    r /= 255; g /= 255; b /= 255;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    let h = 0;
    const l = (max + min) / 2;
    const d = max - min;
    const s = d === 0 ? 0 : d / (1 - Math.abs(2 * l - 1));
    if (d !== 0) {
        switch (max) {
            case r: h = ((g - b) / d) % 6; break;
            case g: h = (b - r) / d + 2; break;
            case b: h = (r - g) / d + 4; break;
        }
        h *= 60;
        if (h < 0) h += 360;
    }
    return { h, s, l };
}

function hueDistance(a, b) {
    const d = Math.abs((a - b) % 360);
    return Math.min(d, 360 - d);
}

function isColorNear(a, b, tolerance) {
    const ar = Number(a?.r ?? NaN);
    const ag = Number(a?.g ?? NaN);
    const ab = Number(a?.b ?? NaN);
    const br = Number(b?.r ?? NaN);
    const bg = Number(b?.g ?? NaN);
    const bb = Number(b?.b ?? NaN);
    if (![ar, ag, ab, br, bg, bb].every(Number.isFinite)) return false;

    const aHsl = rgbToHsl(ar, ag, ab);
    const bHsl = rgbToHsl(br, bg, bb);

    if (aHsl.s > 0.12 && bHsl.s > 0.12) {
        const dh = hueDistance(aHsl.h, bHsl.h);
        const ds = Math.abs(aHsl.s - bHsl.s);
        const dl = Math.abs(aHsl.l - bHsl.l);
        return (dh <= 22 && ds <= 0.60 && dl <= 0.60);
    }

    const t = Math.max(0, Number(tolerance) || 160);
    const dr = Math.abs(ar - br);
    const dg = Math.abs(ag - bg);
    const db = Math.abs(ab - bb);
    return (dr + dg + db) <= t;
}

class BusGroupSpacerWidget {
    constructor(height = 8) {
        this.type = "custom";
        this.name = "iamccs_bus_group_spacer";
        this.label = "";
        this.options = { serialize: false };
        this._height = Math.max(2, Number(height) || 8);
    }

    computeSize(width) {
        const w = Number.isFinite(width) ? width : 260;
        return [Math.max(10, w), this._height];
    }

    draw() {}
    mouse() { return false; }
}

class BusGroupDividerWidget {
    constructor() {
        this.type = "custom";
        this.name = "iamccs_bus_group_divider";
        this.label = "";
        this.options = { serialize: false };
        this._height = 10;
    }

    computeSize(width) {
        const w = Number.isFinite(width) ? width : 260;
        return [Math.max(10, w), this._height];
    }

    draw(ctx, node, width, posY, height) {
        try {
            const h = height || this._height;
            const x0 = 12;
            const y = posY + Math.floor(h / 2);
            ctx.save();
            ctx.strokeStyle = "rgba(255,255,255,0.12)";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(x0, y);
            ctx.lineTo(width - x0, y);
            ctx.stroke();
            ctx.restore();
        } catch {}
    }

    mouse() { return false; }
}

function _safeSlotHide(slot) {
    if (!slot) return;
    slot.hidden = true;
    slot.name = "";
    slot.label = "";
    // Keep type so links remain valid if present.
}

function _safeSlotShow(slot, name) {
    if (!slot) return;
    slot.hidden = false;
    slot._iamccs_logical_name = name;
    slot.name = "";
    slot.label = "";
}

function _syncSlotCount(node, isReceiver, count) {
    // IMPORTANT: LiteGraph does not reliably respect "hidden" slots when drawing.
    // To avoid header overlap, physically remove unused slots.
    const want = Math.max(0, Math.min(Number(count) || 0, MAX_CHANNELS));

    if (isReceiver) {
        node.inputs = node.inputs || [];
        while (node.inputs.length > want) {
            try { node.removeInput(node.inputs.length - 1); }
            catch { node.inputs.pop(); }
        }
        while (node.inputs.length < want) {
            try { node.addInput("", "*"); }
            catch {
                node.inputs.push({ name: "", type: "*", link: null, label: "" });
            }
        }
        for (const inp of node.inputs) {
            if (!inp) continue;
            inp.name = "";
            inp.label = "";
        }
    } else {
        node.outputs = node.outputs || [];
        while (node.outputs.length > want) {
            try { node.removeOutput(node.outputs.length - 1); }
            catch { node.outputs.pop(); }
        }
        while (node.outputs.length < want) {
            try { node.addOutput("", "*"); }
            catch {
                node.outputs.push({ name: "", type: "*", links: null, label: "" });
            }
        }
        for (const out of node.outputs) {
            if (!out) continue;
            out.name = "";
            out.label = "";
        }
    }
}

function _removeAllSlots(node) {
    try {
        while ((node.inputs || []).length) node.removeInput(0);
    } catch {
        node.inputs = [];
    }
    try {
        while ((node.outputs || []).length) node.removeOutput(0);
    } catch {
        node.outputs = [];
    }
}

function _computeRowCentersLocalY(node) {
    // IMPORTANT: connection points are drawn before custom widget draw() runs in some builds.
    // So we cannot rely on draw-time posY; we must compute row centers from the widget layout.
    const LG = window?.LiteGraph;
    const titleH = Number(LG?.NODE_TITLE_HEIGHT ?? 30);
    const defaultWidgetH = Number(LG?.NODE_WIDGET_HEIGHT ?? 20);
    const margin = 4;

    let y = titleH + margin;
    const centers = [];

    for (const w of node?.widgets || []) {
        let h = defaultWidgetH;
        try {
            const sz = w?.computeSize?.(node?.size?.[0] || 260);
            if (Array.isArray(sz) && Number.isFinite(sz[1])) h = sz[1];
        } catch {}

        if (w instanceof BusGroupRowWidget) {
            centers.push(y + h * 0.5);
        }

        y += h + margin;
    }

    node._iamccsRowCentersLocalY = centers;
}

function _applyRowSlotPositions(node) {
    const centers = node?._iamccsRowCentersLocalY;
    if (!Array.isArray(centers) || centers.length === 0) return;

    const isReceiver = !!node?.properties?._iamccs_is_receiver;
    const count = Math.min(centers.length, MAX_CHANNELS);

    // Use a slightly inset X so sockets visually sit on the row.
    const x = isReceiver ? 8 : Math.max(8, (node?.size?.[0] || 0) - 8);

    for (let i = 0; i < MAX_CHANNELS; i++) {
        const slot = isReceiver ? node.inputs?.[i] : node.outputs?.[i];
        if (!slot) continue;

        if (i < count && Number.isFinite(centers[i])) {
            // LiteGraph respects slot.pos in getConnectionPos and also uses it for drawing sockets.
            slot.pos = [x, centers[i]];
        } else {
            // Remove custom positioning for unused/hidden slots.
            try { delete slot.pos; } catch { slot.pos = null; }
        }
    }
}

function fitString(ctx, str, maxWidth) {
    const s = String(str ?? "");
    if (!maxWidth || maxWidth <= 0) return s;
    if (ctx.measureText(s).width <= maxWidth) return s;
    const ell = "…";
    let lo = 0;
    let hi = s.length;
    while (lo < hi) {
        const mid = Math.floor((lo + hi) / 2);
        const t = s.slice(0, mid) + ell;
        if (ctx.measureText(t).width <= maxWidth) lo = mid + 1;
        else hi = mid;
    }
    return s.slice(0, Math.max(0, lo - 1)) + ell;
}

class BusGroupRowWidget {
    constructor(group, node) {
        this.type = "custom";
        this.name = "iamccs_bus_group_row";
        this.label = "";
        this.options = { serialize: true };
        this.value = { mute: false, solo: false };
        this.group = group;
        this.node = node;
        // Keep rows compact: user wants less vertical padding between groups.
        this._height = 22;
        this._tapAnimUntil = 0;
    }

    computeSize(width) {
        if (this._iamccsHiddenByShowMode) return [0, 0];
        const w = Number.isFinite(width) ? width : 260;
        return [Math.max(180, w), this._height];
    }

    serializeValue() {
        return this.value;
    }

    _toggleMute() {
        if (this.node?.properties?.iamccs_bus_group_lock_groups) return;
        this.value.mute = !this.value.mute;
        // If a group is muted, it can't be solo at the same time.
        if (this.value.mute) this.value.solo = false;
        this.node?._iamccsApplyModes?.();
    }

    _toggleSolo() {
        if (this.node?.properties?.iamccs_bus_group_lock_groups) return;
        const newValue = !this.value.solo;
        // Multi-solo allowed. If enabling solo, clear mute on this row.
        if (newValue) this.value.mute = false;
        this.value.solo = newValue;
        this.node?._iamccsApplyModes?.();
    }

    draw(ctx, node, width, posY, height) {
        const h = height || this._height;

        // Cache last drawn bounds so hit-testing can include Y.
        try {
            this._iamccsHitY0 = posY;
            this._iamccsHitY1 = posY + h;
        } catch {}

        const now = (typeof performance !== "undefined" && performance.now) ? performance.now() : Date.now();
        const tapDur = 140;
        const tLeft = Math.max(0, (this._tapAnimUntil || 0) - now);
        const tapProgress = tLeft > 0 ? (1 - (tLeft / tapDur)) : 0;
        const tapOffset = tLeft > 0 ? Math.round(2 * Math.sin(tapProgress * Math.PI)) : 0;
        if (tapOffset) {
            ctx.save();
            ctx.translate(0, tapOffset);
        }

        const x0 = 10;
        // Slightly wider box (2px) without changing inner layout/hitboxes.
        const boxX0 = x0 - 1;
        const boxW = width - x0 * 2 + 2;
        const yMid = posY + h * 0.5;

        const isReceiver = !!node?.properties?._iamccs_is_receiver;
        const macroMode = !!node?.properties?.iamccs_bus_group_macro_mode;
        const key = getGroupKey(this.group);
        const selected = macroMode && !!node?._iamccsMacroSelection?.has?.(key);

        // If a macro is selected via dropdown/click, highlight member groups.
        const selectedMacroNameRaw = String(node?.properties?.iamccs_bus_group_macro_selected || "");
        const selectedMacroName = selectedMacroNameRaw === "none" ? "" : selectedMacroNameRaw;
        const selectedMacro = (Array.isArray(node?.properties?.iamccs_bus_group_macros) ? node.properties.iamccs_bus_group_macros : [])
            .find(m => String(m?.name || "") === selectedMacroName);
        const isMemberOfSelectedMacro = !!selectedMacro && Array.isArray(selectedMacro?.keys) && selectedMacro.keys.includes(key);

        const stripeColorRaw = String(this.group?.color || this.group?.bgcolor || "").trim();
        const stripeColor = stripeColorRaw || "#666";

        // Background strip
        ctx.save();
        const useGroupFill = !!node?.properties?.iamccs_bus_group_widget_colors;
        ctx.globalAlpha = useGroupFill ? 0.14 : 0.10;
        ctx.fillStyle = useGroupFill ? stripeColor : "#000";
        ctx.fillRect(boxX0, posY + 1, boxW, h - 2);
        ctx.restore();

        // Subtle relief so rows are easier to distinguish
        try {
            ctx.save();
            ctx.strokeStyle = "rgba(255,255,255,0.07)";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(boxX0 + 1, posY + 2);
            ctx.lineTo(boxX0 + boxW - 1, posY + 2);
            ctx.stroke();

            ctx.strokeStyle = "rgba(0,0,0,0.25)";
            ctx.beginPath();
            ctx.moveTo(boxX0 + 1, posY + h - 2);
            ctx.lineTo(boxX0 + boxW - 1, posY + h - 2);
            ctx.stroke();
            ctx.restore();
        } catch {}

        if (selected) {
            ctx.save();
            ctx.strokeStyle = "rgba(46, 204, 113, 0.9)";
            ctx.lineWidth = 2;
            ctx.strokeRect(boxX0 + 1, posY + 2, boxW - 2, h - 4);
            ctx.restore();
        }

        if (!selected && isMemberOfSelectedMacro) {
            ctx.save();
            ctx.strokeStyle = "rgba(30, 144, 255, 0.65)";
            ctx.lineWidth = 2;
            ctx.strokeRect(boxX0 + 1, posY + 2, boxW - 2, h - 4);
            ctx.restore();
        }

        ctx.save();
        ctx.globalAlpha = 0.9;
        ctx.fillStyle = stripeColor;
        ctx.fillRect(x0, posY + 2, 4, h - 4);
        ctx.restore();

        // A few extra pixels after the S toggle.
        const rightPad = 14;
        const toggleR = 9;
        const gap = 10;
        const soloCX = width - rightPad - toggleR;
        const muteCX = soloCX - (toggleR * 2 + gap);

        // Group name
        ctx.save();
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillStyle = "#ddd";
        const ledR = 3;
        const ledX = x0 + 10;
        const nameX = x0 + 14 + (this.value.mute || this.value.solo ? 10 : 0);
        const nameMaxW = Math.max(10, muteCX - 16 - nameX);
        const gTitle = String(this.group?.title ?? "(group)");

        // Red LED for mute, green for solo (next to name).
        if (this.value.mute || this.value.solo) {
            ctx.beginPath();
            ctx.fillStyle = this.value.solo ? "#2ecc71" : "#ff4d4d";
            ctx.arc(ledX, yMid, ledR, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = "#ddd";
        }

        ctx.fillText(fitString(ctx, gTitle, nameMaxW), nameX, yMid);
        ctx.restore();

        if (!isReceiver && !macroMode) {
            // Mute toggle
            ctx.save();
            ctx.fillStyle = "#333";
            ctx.beginPath();
            ctx.arc(muteCX, yMid, toggleR, 0, Math.PI * 2);
            ctx.fill();

            // Red ring indicator when muted
            if (this.value.mute) {
                ctx.strokeStyle = "#ff4d4d";
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(muteCX, yMid, toggleR + 1, 0, Math.PI * 2);
                ctx.stroke();
            }

            ctx.fillStyle = "#fff";
            ctx.font = "bold 10px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("M", muteCX, yMid + 0.5);
            ctx.restore();

            // Solo toggle
            ctx.save();
            ctx.fillStyle = "#333";
            ctx.beginPath();
            ctx.arc(soloCX, yMid, toggleR, 0, Math.PI * 2);
            ctx.fill();

            // Green ring indicator when solo
            if (this.value.solo) {
                ctx.strokeStyle = "#2ecc71";
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(soloCX, yMid, toggleR + 1, 0, Math.PI * 2);
                ctx.stroke();
            }

            ctx.fillStyle = "#fff";
            ctx.font = "bold 10px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("S", soloCX, yMid + 0.5);
            ctx.restore();
        }

        // Small hint labels (only if enough width)
        // (intentionally no extra labels under M/S)

        if (tapOffset) {
            ctx.restore();
        }
        if (tLeft > 0) {
            try { node?.graph?.setDirtyCanvas(true, true); } catch {}
        }
    }

    mouse(event, pos, node) {
        const isDown = (event.type === "pointerdown" || event.type === "mousedown");
        const isUp = (event.type === "pointerup" || event.type === "mouseup");
        if (!isDown && !isUp) return false;

        if (node?.properties?.iamccs_bus_group_safety_shift && !event.shiftKey) {
            return false;
        }

        const x = pos?.[0] ?? 0;
        const y = pos?.[1] ?? 0;

        const macroMode = !!node?.properties?.iamccs_bus_group_macro_mode;

        const width = node?.size?.[0] || 260;
        const rightPad = 14;
        const toggleR = 9;
        const gap = 10;
        const soloCX = width - rightPad - toggleR;
        const muteCX = soloCX - (toggleR * 2 + gap);

        const y0 = Number(this._iamccsHitY0 ?? NaN);
        const y1 = Number(this._iamccsHitY1 ?? NaN);
        const yMid = (Number.isFinite(y0) && Number.isFinite(y1)) ? ((y0 + y1) * 0.5) : NaN;

        const hitR = toggleR * 2.15;
        const hitSolo = Number.isFinite(yMid)
            ? (Math.hypot(x - soloCX, y - yMid) <= hitR)
            : (Math.abs(x - soloCX) <= hitR);
        const hitMute = Number.isFinite(yMid)
            ? (Math.hypot(x - muteCX, y - yMid) <= hitR)
            : (Math.abs(x - muteCX) <= hitR);

        if (!node?.properties?._iamccs_is_receiver) {
            if (hitSolo) {
                if (isDown) {
                    // Tap feedback only when actually clicking a control.
                    try {
                        const now = (typeof performance !== "undefined" && performance.now) ? performance.now() : Date.now();
                        this._tapAnimUntil = now + 140;
                        node?.graph?.setDirtyCanvas(true, true);
                    } catch {}
                }
                if (isUp) this._toggleSolo();
                return true;
            }
            if (hitMute) {
                if (isDown) {
                    try {
                        const now = (typeof performance !== "undefined" && performance.now) ? performance.now() : Date.now();
                        this._tapAnimUntil = now + 140;
                        node?.graph?.setDirtyCanvas(true, true);
                    } catch {}
                }
                if (isUp) this._toggleMute();
                return true;
            }
        }

        // For row background/label, do NOT consume mousedown so the node can be dragged from anywhere.
        // We only perform actions on mouseup if it was a click (not a drag).
        const key = getGroupKey(this.group);
        if (isDown) {
            try {
                node._iamccsClickCandidate = {
                    kind: "group",
                    key,
                    macroMode: !!macroMode,
                    goTo: !!node?.properties?.iamccs_bus_group_go_to,
                    x,
                    y,
                    t: Date.now(),
                };
            } catch {}
            return false;
        }

        if (isUp) {
            const c = node?._iamccsClickCandidate;
            try { node._iamccsClickCandidate = null; } catch {}
            if (!c || c.kind !== "group" || c.key !== key) return false;
            const dx = Math.abs((c.x ?? 0) - x);
            const dy = Math.abs((c.y ?? 0) - y);
            if (dx > 3 || dy > 3) return false;

            // Click action (not drag)
            if (c.macroMode) {
                node._iamccsMacroSelection = node._iamccsMacroSelection || new Set();
                if (node._iamccsMacroSelection.has(key)) node._iamccsMacroSelection.delete(key);
                else node._iamccsMacroSelection.add(key);
                try { node.graph?.setDirtyCanvas(true, true); } catch {}
                return true;
            }

            // Optional go-to (click-to-center) while still allowing drag.
            try {
                if (c.goTo && app?.canvas?.centerOnNode && this.group) {
                    app.canvas.centerOnNode(this.group);
                    app.canvas.setDirty(true, true);
                    return true;
                }
            } catch {}
        }

        return false;
    }
}

class BusGroupMacroRowWidget {
    constructor(macro, node) {
        this.type = "custom";
        this.name = "iamccs_bus_group_macro_row";
        this.label = "";
        this.options = { serialize: true };
        this.macro = macro;
        this.node = node;
        // Tighter spacing between macros, but with a larger usable box area.
        this._height = 26;
    }

    computeSize(width) {
        const w = Number.isFinite(width) ? width : 260;
        return [Math.max(180, w), this._height];
    }

    _iterTargetRows() {
        const keys = Array.isArray(this.macro?.keys) ? this.macro.keys : [];
        const rows = (this.node?.widgets || []).filter(w => w instanceof BusGroupRowWidget);
        const byKey = new Map(rows.map(r => [getGroupKey(r.group), r]));
        const out = [];
        for (const k of keys) {
            const r = byKey.get(k);
            if (r) out.push(r);
        }
        return out;
    }

    _toggleMuteAll() {
        if (this.node?.properties?.iamccs_bus_group_lock_groups) return;
        const rows = this._iterTargetRows();
        if (!rows.length) return;
        const allMuted = rows.every(r => !!r.value.mute);
        const next = !allMuted;
        for (const r of rows) {
            r.value.mute = next;
            if (next) r.value.solo = false;
        }
        this.node?._iamccsApplyModes?.();
    }

    _toggleActivateAll() {
        if (this.node?.properties?.iamccs_bus_group_lock_groups) return;
        const rows = this._iterTargetRows();
        if (!rows.length) return;
        const allSolo = rows.every(r => !!r.value.solo);
        const next = !allSolo;
        for (const r of rows) {
            r.value.solo = next;
            if (next) r.value.mute = false;
        }
        this.node?._iamccsApplyModes?.();
    }

    draw(ctx, node, width, posY, height) {
        const h = height || this._height;
        const x0 = 10;
        const yMid = posY + h * 0.5;

        // Cache last drawn bounds so hit-testing can include Y.
        // Some LiteGraph builds dispatch mouse events to multiple widgets;
        // using X-only hit tests can make every macro row react.
        try {
            this._iamccsHitY0 = posY;
            this._iamccsHitY1 = posY + h;
        } catch {}

        // Match group-row spacing: keep a few pixels from the right edge.
        const rightPad = 14;
        const toggleR = 8;
        const gap = 10;
        const actCX = width - rightPad - toggleR;
        const muteCX = actCX - (toggleR * 2 + gap);

        const isSelected = String(node?.properties?.iamccs_bus_group_macro_selected || "") === String(this.macro?.name || "");

        // Macro state (for ring indicators).
        const rows = this._iterTargetRows();
        const allMuted = rows.length ? rows.every(r => !!r.value.mute) : false;
        const allSolo = rows.length ? rows.every(r => !!r.value.solo) : false;

        // Macro background + distinctive frame (with minimal padding)
        const padY = 1;
        ctx.save();
        ctx.globalAlpha = isSelected ? 0.14 : 0.10;
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(x0, posY + padY, width - x0 * 2, h - padY * 2);
        ctx.restore();

        // White stripe on the left
        ctx.save();
        ctx.globalAlpha = 0.95;
        ctx.fillStyle = "#fff";
        ctx.fillRect(x0 + 1, posY + padY + 1, 4, h - (padY + 1) * 2);
        ctx.restore();

        ctx.save();
        // Subtle border; selected is slightly stronger
        ctx.strokeStyle = isSelected ? "rgba(255, 255, 255, 0.45)" : "rgba(255, 255, 255, 0.25)";
        ctx.lineWidth = 2;
        ctx.strokeRect(x0 + 1, posY + padY + 1, width - x0 * 2 - 2, h - (padY + 1) * 2);
        ctx.restore();

        // Title
        ctx.save();
        ctx.font = "12px sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillStyle = "#f4f8ff";
        const name = String(this.macro?.name || "(macro)");
        const nameX = x0 + 10;
        const nameMaxW = Math.max(10, muteCX - 16 - nameX);
        ctx.fillText(fitString(ctx, `MACRO: ${name}`, nameMaxW), nameX, yMid);
        ctx.restore();

        // Mute button
        ctx.save();
        ctx.fillStyle = "#333";
        ctx.beginPath();
        ctx.arc(muteCX, yMid, toggleR, 0, Math.PI * 2);
        ctx.fill();

        // Red ring indicator when macro is fully muted
        if (allMuted) {
            ctx.strokeStyle = "#ff4d4d";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(muteCX, yMid, toggleR + 1, 0, Math.PI * 2);
            ctx.stroke();
        }
        ctx.fillStyle = "#fff";
        ctx.font = "bold 10px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("M", muteCX, yMid + 0.5);
        ctx.restore();

        // Activate button
        ctx.save();
        ctx.fillStyle = "#333";
        ctx.beginPath();
        ctx.arc(actCX, yMid, toggleR, 0, Math.PI * 2);
        ctx.fill();

        // Green ring indicator when macro is fully solo/active
        if (allSolo) {
            ctx.strokeStyle = "#2ecc71";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(actCX, yMid, toggleR + 1, 0, Math.PI * 2);
            ctx.stroke();
        }
        ctx.fillStyle = "#fff";
        ctx.font = "bold 10px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("A", actCX, yMid + 0.5);
        ctx.restore();
    }

    mouse(event, pos, node) {
        const isDown = (event.type === "pointerdown" || event.type === "mousedown");
        const isUp = (event.type === "pointerup" || event.type === "mouseup");
        if (!isDown && !isUp) return false;
        if (node?.properties?.iamccs_bus_group_safety_shift && !event.shiftKey) return false;

        const width = node?.size?.[0] || 260;
        // Keep consistent with draw(): a little space from the right edge.
        const rightPad = 14;
        const toggleR = 7;
        const gap = 10;
        const actCX = width - rightPad - toggleR;
        const muteCX = actCX - (toggleR * 2 + gap);

        const x = pos?.[0] ?? 0;
        const y = pos?.[1] ?? 0;
        const name = String(this.macro?.name || "");

        // Y-aware hit test (prevents all macro rows from reacting).
        const y0 = Number(this._iamccsHitY0 ?? NaN);
        const y1 = Number(this._iamccsHitY1 ?? NaN);
        const inRowY = Number.isFinite(y0) && Number.isFinite(y1) ? (y >= (y0 - 2) && y <= (y1 + 2)) : true;
        const yMid = (Number.isFinite(y0) && Number.isFinite(y1)) ? ((y0 + y1) * 0.5) : NaN;

        // IMPORTANT:
        // Some ComfyUI/LiteGraph builds may dispatch the mouseup event to multiple widgets.
        // If we only key off X, every macro row sees "button hit" and toggles.
        // So we latch the pointerdown target (macro+button) and only act on the matching pointerup.
        const hitR = toggleR * 2.15;
        const hitActivate = Number.isFinite(yMid)
            ? (Math.hypot(x - actCX, y - yMid) <= hitR)
            : (Math.abs(x - actCX) <= hitR);
        const hitMute = Number.isFinite(yMid)
            ? (Math.hypot(x - muteCX, y - yMid) <= hitR)
            : (Math.abs(x - muteCX) <= hitR);

        const pointerId = (typeof event?.pointerId === "number") ? event.pointerId : 0;

        if (isDown && inRowY && (hitActivate || hitMute)) {
            try {
                node._iamccsClickCandidate = {
                    kind: "macro_btn",
                    name,
                    btn: hitActivate ? "activate" : "mute",
                    x,
                    y,
                    pointerId,
                    t: Date.now(),
                };
            } catch {}
            // Consume down so the canvas doesn't treat it as a select/drag.
            return true;
        }

        if (isUp) {
            const c = node?._iamccsClickCandidate;
            // Clear regardless to avoid sticky state.
            try { node._iamccsClickCandidate = null; } catch {}
            if (c && c.kind === "macro_btn" && String(c.name || "") === name && (c.pointerId ?? 0) === pointerId) {
                const dx = Math.abs((c.x ?? 0) - x);
                const dy = Math.abs((c.y ?? 0) - y);
                if (inRowY && dx <= 3 && dy <= 3) {
                    if (c.btn === "activate" && hitActivate) {
                        this._toggleActivateAll();
                        return true;
                    }
                    if (c.btn === "mute" && hitMute) {
                        this._toggleMuteAll();
                        return true;
                    }
                }
            }
        }

        // For row background/label, do NOT consume mousedown so the node can be dragged from anywhere.
        if (isDown && inRowY) {
            try {
                node._iamccsClickCandidate = { kind: "macro", name, x, y, pointerId, t: Date.now() };
            } catch {}
            return false;
        }

        if (isUp) {
            const c = node?._iamccsClickCandidate;
            try { node._iamccsClickCandidate = null; } catch {}
            if (!c || c.kind !== "macro" || c.name !== name || (c.pointerId ?? 0) !== pointerId) return false;
            const dx = Math.abs((c.x ?? 0) - x);
            const dy = Math.abs((c.y ?? 0) - y);
            if (dx > 3 || dy > 3) return false;
            if (!inRowY) return false;

            // Click action (not drag): select macro
            try {
                node.properties = node.properties || {};
                node.properties.iamccs_bus_group_macro_selected = name;
                try {
                    if (node._iamccsMacroComboWidget) node._iamccsMacroComboWidget.value = name || "none";
                } catch {}
                try {
                    const keys = Array.isArray(this.macro?.keys) ? this.macro.keys : [];
                    node._iamccsMacroSelection = new Set(keys);
                } catch {}
                node.graph?.setDirtyCanvas(true, true);
            } catch {}
            return true;
        }

        return false;
    }
}

function rebuildWidgets(node) {
    const graph = getGraph();
    if (!graph) return;

    node.properties = node.properties || {};
    node.properties.bus_group_state = node.properties.bus_group_state || {};

    // Normalize "none" sentinel to "no selection".
    try {
        if (node.properties.iamccs_bus_group_macro_selected === "none") {
            node.properties.iamccs_bus_group_macro_selected = "";
        }
    } catch {}

    // If macro mode is off, always clear selection (prevents lingering green borders).
    try {
        if (!node.properties.iamccs_bus_group_macro_mode) {
            if (node._iamccsMacroSelection?.size) node._iamccsMacroSelection.clear();
        }
    } catch {}

    const prevState = node.properties.bus_group_state;

    // Always remove slots: this is now a socket-less controller UI.
    try { _removeAllSlots(node); } catch {}

    // Keep only non-row widgets and rebuild rows.
    const keep = [];
    for (const w of node.widgets || []) {
        // Divider widgets are dynamically inserted during rebuild; do not keep them or they will duplicate.
        if (
            !(w instanceof BusGroupRowWidget) &&
            !(w instanceof BusGroupMacroRowWidget) &&
            !(w instanceof BusGroupDividerWidget)
        ) {
            keep.push(w);
        }
    }
    node.widgets = keep;

    const groups = getGroups(graph).filter(g => _matchesGroupFilters(node, g));

    // Cache stable membership per group key. This prevents accidental cross-group toggles
    // when groups overlap during dragging.
    try {
        node._iamccsGroupMembersByKey = new Map();
        for (const g of groups) {
            const key = getGroupKey(g);
            const members = getGroupNodesStable(g, graph);
            const ids = members.map(n => n?.id).filter(id => id != null);
            node._iamccsGroupMembersByKey.set(key, ids);
        }
    } catch {
        node._iamccsGroupMembersByKey = new Map();
    }

    // Auto widen node so the row box is wider than the group name.
    // This is intentionally one-way (only grows) to avoid fighting manual resizing.
    try {
        const measureCanvas = rebuildWidgets._iamccsMeasureCanvas || (rebuildWidgets._iamccsMeasureCanvas = document.createElement("canvas"));
        const measureCtx = measureCanvas.getContext("2d");
        if (measureCtx) {
            // Match the typical LiteGraph text size for widgets.
            measureCtx.font = "14px sans-serif";
            let maxTitleW = 0;
            for (const g of groups) {
                const t = String(g?.title ?? "(group)");
                const w = measureCtx.measureText(t).width;
                if (w > maxTitleW) maxTitleW = w;
            }

            // Layout constants from BusGroupRowWidget.draw()
            const x0 = 10;
            const rightPad = 14;
            const toggleR = 8;
            const gap = 10;
            // Worst-case nameX when LED is present (adds 10).
            const nameX = x0 + 14 + 10;
            const extra = 24; // make the box a bit wider than the text

            // Ensure nameMaxW >= maxTitleW + extra.
            // nameMaxW = width - (rightPad + 3*toggleR + gap + 16 + nameX)
            const needed = Math.ceil(maxTitleW + extra + (rightPad + 3 * toggleR + gap + 16 + nameX));
            if (Number.isFinite(needed) && needed > 0) {
                node.size = node.size || [260, 60];
                node.size[0] = Math.max(node.size[0] || 0, Math.min(1600, needed));
            }
        }
    } catch {}

    // Keep macro dropdown up to date.
    try {
        const macrosNow = Array.isArray(node?.properties?.iamccs_bus_group_macros) ? node.properties.iamccs_bus_group_macros : [];
        const names = macrosNow.map(m => String(m?.name || "")).filter(Boolean);
        if (node._iamccsMacroComboWidget?.options) {
            node._iamccsMacroComboWidget.options.values = ["none", ...names];
        }
        const selected = String(node?.properties?.iamccs_bus_group_macro_selected || "");
        if (selected && !names.includes(selected)) {
            node.properties.iamccs_bus_group_macro_selected = "";
            if (node._iamccsMacroComboWidget) node._iamccsMacroComboWidget.value = "none";
        }
        if (!selected) {
            if (node._iamccsMacroComboWidget) node._iamccsMacroComboWidget.value = "none";
        }
    } catch {}

    const showMode = String(node?.properties?.iamccs_bus_group_show_mode || "both");

    // Macros first (unless filtered out)
    const macros = Array.isArray(node?.properties?.iamccs_bus_group_macros) ? node.properties.iamccs_bus_group_macros : [];
    if (showMode !== "groups") {
        for (const m of macros) {
            if (!m || typeof m !== "object") continue;
            node.addCustomWidget(new BusGroupMacroRowWidget(m, node));
        }

        // Divider between macros and groups
        if (macros.length && showMode === "both") {
            node.addCustomWidget(new BusGroupDividerWidget());
        }
    }

    // Build group rows.
    // In Show: macro we keep them (hidden) so macro buttons still work and applyModes has state.
    node._iamccsGroupKeysByChannel = [];
    for (const group of groups) {
        const w = new BusGroupRowWidget(group, node);
        const key = getGroupKey(group);
        const saved = prevState[key];
        if (saved && typeof saved === "object") {
            w.value.mute = !!saved.mute;
            w.value.solo = !!saved.solo;
        }

        const hideByShow = (showMode === "macro");
        w._iamccsHiddenByShowMode = hideByShow;
        w.hidden = !!hideByShow;

        node.addCustomWidget(w);
        node._iamccsGroupKeysByChannel.push(key);
    }

    node.size = node.computeSize();
    node._iamccsLastGroupSig = groups.map(getGroupKey).join("|");
    node._iamccsApplyModes?.();
    graph.setDirtyCanvas(true, true);
}

function applyModes(node) {
    const graph = getGraph();
    if (!graph) return;

    const rows = (node.widgets || []).filter(w => w instanceof BusGroupRowWidget);

    // Persist widget state.
    node.properties = node.properties || {};
    node.properties.bus_group_state = node.properties.bus_group_state || {};
    for (const w of rows) {
        node.properties.bus_group_state[getGroupKey(w.group)] = { mute: !!w.value.mute, solo: !!w.value.solo };
    }

    const anySolo = rows.some(w => !!w.value.solo);

    // Expose channel state for receivers
    try {
        node._iamccsAnySolo = anySolo;
        node._iamccsChannelState = [];
        for (let i = 0; i < Math.min(rows.length, MAX_CHANNELS); i++) {
            const r = rows[i];
            node._iamccsChannelState[i] = { mute: !!r.value.mute, solo: !!r.value.solo, key: getGroupKey(r.group) };
        }
    } catch {}

    // Keep title stable (no match-title/match-color feature).
    try {
        if (node._iamccsBaseTitle == null) node._iamccsBaseTitle = node.title;
        node.title = node._iamccsBaseTitle;
    } catch {}

    const MUTE_MODE = getMuteMode();

    for (const w of rows) {
        const shouldEnable = anySolo ? !!w.value.solo : !w.value.mute;
        const key = getGroupKey(w.group);
        let members = null;
        try {
            const ids = node?._iamccsGroupMembersByKey?.get?.(key);
            if (Array.isArray(ids) && ids.length) members = _resolveNodesByIds(graph, ids);
        } catch {}
        if (!members) {
            // Fallback: compute stable membership on-demand and cache it.
            const computed = getGroupNodesStable(w.group, graph);
            members = computed;
            try {
                const ids = computed.map(n => n?.id).filter(id => id != null);
                node._iamccsGroupMembersByKey = node._iamccsGroupMembersByKey || new Map();
                node._iamccsGroupMembersByKey.set(key, ids);
            } catch {}
        }
        const nodes = members || [];
        for (const n of nodes) {
            try {
                n.mode = shouldEnable ? ENABLED_MODE : MUTE_MODE;
            } catch {}
        }
        try { w.group?.graph?.setDirtyCanvas(true, false); } catch {}
    }

    try { graph.setDirtyCanvas(true, true); } catch {}
}

function applyReceiverModes(node) {
    const graph = getGraph();
    if (!graph) return;

    const rows = (node.widgets || []).filter(w => w instanceof BusGroupRowWidget);

    // Receiver is now the same socket-less controller as main.
    // Keep for backward compatibility but do not mirror via links.
    try { graph.setDirtyCanvas(true, true); } catch {}
}

function _patchRowAlignedSockets(nodeType, isReceiver) {
    const originalGetConnectionPos = nodeType.prototype.getConnectionPos;
    nodeType.prototype.getConnectionPos = function (isInput, slotNumber, out) {
        try {
            const wantInput = !!isReceiver;
            if (isInput !== wantInput) {
                return originalGetConnectionPos ? originalGetConnectionPos.call(this, isInput, slotNumber, out) : out;
            }

            out = out || new Float32Array(2);

            const localY = this._iamccsRowCentersLocalY?.[slotNumber];
            if (!Number.isFinite(localY)) {
                return originalGetConnectionPos ? originalGetConnectionPos.call(this, isInput, slotNumber, out) : out;
            }
            const y = this.pos[1] + localY;

            // Place sockets *inside* the row so they visually sit in the same line.
            if (isInput) {
                out[0] = this.pos[0] + 8; // slightly inside left edge, aligned with stripe
            } else {
                out[0] = this.pos[0] + this.size[0] - 8; // near right edge (inline with row)
            }
            out[1] = y;
            return out;
        } catch {
            return originalGetConnectionPos ? originalGetConnectionPos.call(this, isInput, slotNumber, out) : out;
        }
    };
}

function _attachRowAlignedSocketsToInstance(node, isReceiver) {
    if (!node || node._iamccsRowAlignedSocketsAttached) return;
    node._iamccsRowAlignedSocketsAttached = true;

    const originalGetConnectionPos = node.getConnectionPos;
    const wantInput = !!isReceiver;

    node.getConnectionPos = function (isInput, slotNumber, out) {
        try {
            if (isInput !== wantInput) {
                return originalGetConnectionPos ? originalGetConnectionPos.call(this, isInput, slotNumber, out) : out;
            }

            out = out || new Float32Array(2);
            const localY = this._iamccsRowCentersLocalY?.[slotNumber];
            if (!Number.isFinite(localY)) {
                return originalGetConnectionPos ? originalGetConnectionPos.call(this, isInput, slotNumber, out) : out;
            }

            // Place sockets near the row (slightly inside node) so they visually sit on the same line.
            out[0] = isInput ? (this.pos[0] + 8) : (this.pos[0] + this.size[0] - 8);
            out[1] = this.pos[1] + localY;
            return out;
        } catch {
            return originalGetConnectionPos ? originalGetConnectionPos.call(this, isInput, slotNumber, out) : out;
        }
    };

    // Some extensions (e.g. rgthree connection layout) override getInputPos/getOutputPos.
    // Ensure our node instance always routes through our connection-position logic.
    node.getInputPos = function (slotNumber, out) {
        return this.getConnectionPos(true, slotNumber, out || new Float32Array(2));
    };
    node.getOutputPos = function (slotNumber, out) {
        return this.getConnectionPos(false, slotNumber, out || new Float32Array(2));
    };

    const originalOnResize = node.onResize;
    node.onResize = function () {
        const r = originalOnResize ? originalOnResize.apply(this, arguments) : undefined;
        try { _applyRowSlotPositions(this); } catch {}
        return r;
    };
}

app.registerExtension({
    name: "iamccs.bus_group",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_TYPE) return;

        const isReceiver = false;

        // Prevent ComfyUI/LiteGraph global hotkeys from interfering while interacting with Macro UI.
        // In particular, 'a'/'m' can trigger canvas-level actions; we want those keys to be inert
        // while this node is selected and macro features are in use.
        const origOnKeyDown = nodeType.prototype.onKeyDown;
        nodeType.prototype.onKeyDown = function (e) {
            try {
                const key = String(e?.key || "").toLowerCase();
                const noMods = !e?.ctrlKey && !e?.metaKey && !e?.altKey;
                const macroContext = !!this?.properties?.iamccs_bus_group_macro_mode ||
                    !!String(this?.properties?.iamccs_bus_group_macro_selected || "").trim() ||
                    !!String(this?.properties?.iamccs_bus_group_macro_name || "").trim();

                if (noMods && macroContext && (key === "a" || key === "m")) {
                    try { e.preventDefault?.(); } catch {}
                    try { e.stopPropagation?.(); } catch {}
                    return true;
                }
            } catch {}

            return origOnKeyDown ? origOnKeyDown.apply(this, arguments) : false;
        };

        // Reduce bottom padding without affecting spacing between header widgets and rows.
        // We keep the inter-widget margin but shrink the final bottom pad.
        const origComputeSize = nodeType.prototype.computeSize;
        nodeType.prototype.computeSize = function () {
            const base = origComputeSize ? origComputeSize.apply(this, arguments) : (this.size || [260, 60]);
            try {
                const LG = window?.LiteGraph;
                const titleH = Number(LG?.NODE_TITLE_HEIGHT ?? 30);
                const defaultWidgetH = Number(LG?.NODE_WIDGET_HEIGHT ?? 20);
                const topPad = 4;
                const betweenPad = 4;
                const bottomPad = 2; // independent from topPad (not "linked")

                const allWidgets = Array.isArray(this.widgets) ? this.widgets : [];
                const visibleWidgets = allWidgets.filter(w => {
                    if (!w) return false;
                    if (w.hidden) return false;
                    if (w._iamccsHiddenByOptions) return false;
                    if (w._iamccsHiddenByShowMode) return false;
                    return true;
                });

                let y = titleH + topPad;

                for (let i = 0; i < visibleWidgets.length; i++) {
                    const w = visibleWidgets[i];
                    let h = defaultWidgetH;
                    try {
                        const sz = w?.computeSize?.(this.size?.[0] || base?.[0] || 260);
                        if (Array.isArray(sz) && Number.isFinite(sz[1])) h = Math.max(0, sz[1]);
                    } catch {}

                    y += h;
                    y += (i === visibleWidgets.length - 1) ? bottomPad : betweenPad;
                }

                // If no widgets are visible, keep a tiny bottom so the title bar doesn't hug the border.
                if (visibleWidgets.length === 0) y += bottomPad;

                const wOut = Array.isArray(base) ? (base[0] ?? (this.size?.[0] || 260)) : (this.size?.[0] || 260);
                const hOut = Math.max(y, titleH + 10);
                return [wOut, hOut];
            } catch {
                return base;
            }
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);

            this.properties = this.properties || {};
            if (!this.properties.bus_group_state) this.properties.bus_group_state = {};

            this.properties._iamccs_is_receiver = isReceiver;

            // Socket-less: ensure no header inputs/outputs.
            try { _removeAllSlots(this); } catch {}

            // Methods used by widgets.
            this._iamccsApplyModes = () => (isReceiver ? applyReceiverModes(this) : applyModes(this));
            this._iamccsRefresh = () => rebuildWidgets(this);

            // Options
            if (this.properties.iamccs_bus_group_safety_shift == null) this.properties.iamccs_bus_group_safety_shift = false;
            if (this.properties.iamccs_bus_group_lock_groups == null) this.properties.iamccs_bus_group_lock_groups = false;
            if (this.properties.iamccs_bus_group_go_to == null) this.properties.iamccs_bus_group_go_to = false;
            if (this.properties.iamccs_bus_group_show_mode == null) this.properties.iamccs_bus_group_show_mode = "both";
            // Back-compat: older workflows may have iamccs_bus_group_show_options.
            if (this.properties.iamccs_bus_group_hide_options == null && this.properties.iamccs_bus_group_show_options != null) {
                this.properties.iamccs_bus_group_hide_options = !!this.properties.iamccs_bus_group_show_options;
            }
            if (this.properties.iamccs_bus_group_hide_options == null) this.properties.iamccs_bus_group_hide_options = false;
            if (this.properties.iamccs_bus_group_widget_colors == null) this.properties.iamccs_bus_group_widget_colors = false;
            if (this.properties.iamccs_bus_group_filter_title == null) this.properties.iamccs_bus_group_filter_title = "";
            if (this.properties.iamccs_bus_group_filter_color == null) this.properties.iamccs_bus_group_filter_color = "";

            // Persisted UI state
            if (this.properties.iamccs_bus_group_hide_options == null) this.properties.iamccs_bus_group_hide_options = false;

            if (!Array.isArray(this.properties.iamccs_bus_group_macros)) this.properties.iamccs_bus_group_macros = [];
            if (this.properties.iamccs_bus_group_macro_mode == null) this.properties.iamccs_bus_group_macro_mode = false;
            if (this.properties.iamccs_bus_group_macro_name == null) this.properties.iamccs_bus_group_macro_name = "";
            if (this.properties.iamccs_bus_group_macro_selected == null) this.properties.iamccs_bus_group_macro_selected = "";

            this._iamccsMacroSelection = this._iamccsMacroSelection || new Set();

            // Header widgets visibility helper (stable across repeated toggles).
            const wrapWidgetHide = (w) => {
                if (!w || w._iamccsHideWrapped) return;
                w._iamccsHideWrapped = true;

                const orig = typeof w.computeSize === "function" ? w.computeSize.bind(w) : null;
                w._iamccsOrigComputeSize = orig;

                w.computeSize = function (width) {
                    if (this._iamccsHiddenByOptions) return [0, 0];
                    if (this._iamccsOrigComputeSize) return this._iamccsOrigComputeSize(width);
                    const w0 = Number.isFinite(width) ? width : 260;
                    return [w0, 20];
                };
            };

            const setWidgetHidden = (w, hidden) => {
                if (!w) return;
                wrapWidgetHide(w);
                w._iamccsHiddenByOptions = !!hidden;
                w.hidden = !!hidden;
            };

            this._iamccsHeaderWidgets = [];

            // (1a) Show mode dropdown
            const showModeWidget = this.addWidget(
                "combo",
                "Show",
                String(this.properties.iamccs_bus_group_show_mode || "both"),
                (v) => {
                    this.properties.iamccs_bus_group_show_mode = String(v || "both");
                    this._iamccsRefresh?.();
                },
                { values: ["macro", "groups", "both"] }
            );
            this._iamccsShowModeWidget = showModeWidget;

            // (1b) Hide options toggle (false=show full header, true=hide everything except Show/Hide/Colors)
            const hideOptsWidget = this.addWidget("toggle", "Hide options", !!this.properties.iamccs_bus_group_hide_options, (v) => {
                this.properties.iamccs_bus_group_hide_options = !!v;
                this._iamccsUpdateHeaderVisibility?.();
                this._iamccsRefresh?.();
            });
            this._iamccsShowOptionsWidget = hideOptsWidget;

            // (1c) Widget colors toggle: when ON, group boxes use the group color.
            const widgetColorsToggle = this.addWidget("toggle", "Colors", !!this.properties.iamccs_bus_group_widget_colors, (v) => {
                this.properties.iamccs_bus_group_widget_colors = !!v;
                try { this.graph?.setDirtyCanvas(true, true); } catch {}
            });
            this._iamccsWidgetColorsBtn = widgetColorsToggle;

            // Inline filter widgets (no modal prompt)
            this.addWidget("text", "Set title filter", String(this.properties.iamccs_bus_group_filter_title || ""), (v) => {
                this.properties.iamccs_bus_group_filter_title = String(v ?? "");
                this._iamccsRefresh?.();
            }, { serialize: true });
            this.addWidget("text", "Set color filter", String(this.properties.iamccs_bus_group_filter_color || ""), (v) => {
                this.properties.iamccs_bus_group_filter_color = String(v ?? "");
                this._iamccsRefresh?.();
            }, { serialize: true });

            // Macro workflow
            const macroToggle = this.addWidget("toggle", "Macro", !!this.properties.iamccs_bus_group_macro_mode, (v) => {
                this.properties.iamccs_bus_group_macro_mode = !!v;
                if (!this.properties.iamccs_bus_group_macro_mode) {
                    try { this._iamccsMacroSelection?.clear?.(); } catch {}
                }
                try { this.graph?.setDirtyCanvas(true, true); } catch {}
            });
            this._iamccsMacroModeWidget = macroToggle;

            this.addWidget("button", "Set macro", null, () => {
                const name = String(this.properties.iamccs_bus_group_macro_name || "").trim();
                const keys = Array.from(this._iamccsMacroSelection || []);
                if (!name || keys.length === 0) return;

                const macros = Array.isArray(this.properties.iamccs_bus_group_macros) ? this.properties.iamccs_bus_group_macros : [];
                const idx = macros.findIndex(m => String(m?.name || "") === name);
                const entry = { name, keys };
                if (idx >= 0) macros[idx] = entry;
                else macros.push(entry);
                this.properties.iamccs_bus_group_macros = macros;

                try {
                    this._iamccsMacroSelection?.clear?.();
                    // Ensure selection is fully cleared for draw() even if Set isn't supported.
                    this._iamccsMacroSelection = new Set();
                } catch {
                    this._iamccsMacroSelection = new Set();
                }

                this.properties.iamccs_bus_group_macro_mode = false;
                this.properties.iamccs_bus_group_macro_name = "";
                // Do not auto-select the new macro: after setting it, no groups should be highlighted.
                this.properties.iamccs_bus_group_macro_selected = "";

                // Sync widgets immediately
                try {
                    if (this._iamccsMacroModeWidget) this._iamccsMacroModeWidget.value = false;
                    if (this._iamccsMacroNameWidget) this._iamccsMacroNameWidget.value = "";
                    if (this._iamccsMacroComboWidget) this._iamccsMacroComboWidget.value = "none";
                } catch {}

                this._iamccsRefresh?.();
            });
            const macroName = this.addWidget("text", "Macro name", String(this.properties.iamccs_bus_group_macro_name || ""), (v) => {
                this.properties.iamccs_bus_group_macro_name = String(v ?? "");
            }, { serialize: true });
            this._iamccsMacroNameWidget = macroName;

            // Macro dropdown + delete
            const macroCombo = this.addWidget(
                "combo",
                "Macro select",
                String(this.properties.iamccs_bus_group_macro_selected || "") || "none",
                (v) => {
                    const next = String(v ?? "");
                    this.properties.iamccs_bus_group_macro_selected = next === "none" ? "" : next;
                    // Selecting a macro also selects its group set (useful when Show: macro).
                    try {
                        const sel = String(this.properties.iamccs_bus_group_macro_selected || "");
                        const macros = Array.isArray(this.properties.iamccs_bus_group_macros) ? this.properties.iamccs_bus_group_macros : [];
                        const m = macros.find(mm => String(mm?.name || "") === sel);
                        const keys = Array.isArray(m?.keys) ? m.keys : [];
                        this._iamccsMacroSelection = new Set(keys);
                    } catch {}
                    try { this.graph?.setDirtyCanvas(true, true); } catch {}
                },
                { values: [] }
            );
            this._iamccsMacroComboWidget = macroCombo;

            this.addWidget("button", "Delete macro", null, () => {
                const sel = String(this.properties.iamccs_bus_group_macro_selected || "");
                if (!sel) return;
                const macros = Array.isArray(this.properties.iamccs_bus_group_macros) ? this.properties.iamccs_bus_group_macros : [];
                const next = macros.filter(m => String(m?.name || "") !== sel);
                this.properties.iamccs_bus_group_macros = next;
                this.properties.iamccs_bus_group_macro_selected = "";
                try {
                    const names = next.map(m => String(m?.name || "")).filter(Boolean);
                    if (this._iamccsMacroComboWidget?.options) {
                        this._iamccsMacroComboWidget.options.values = ["none", ...names];
                    }
                    if (this._iamccsMacroComboWidget) this._iamccsMacroComboWidget.value = "none";
                } catch {}
                this._iamccsRefresh?.();
            });

            this.addWidget("toggle", "Safety (Shift-click)", !!this.properties.iamccs_bus_group_safety_shift, (v) => {
                this.properties.iamccs_bus_group_safety_shift = !!v;
            });
            this.addWidget("toggle", "Go to", !!this.properties.iamccs_bus_group_go_to, (v) => {
                this.properties.iamccs_bus_group_go_to = !!v;
            });
            this.addWidget("toggle", "Lock groups", !!this.properties.iamccs_bus_group_lock_groups, (v) => {
                this.properties.iamccs_bus_group_lock_groups = !!v;
            });

            if (!isReceiver) this.addWidget("button", "Mute all", null, () => {
                if (this.properties?.iamccs_bus_group_lock_groups) return;
                for (const w of this.widgets || []) {
                    if (w instanceof BusGroupRowWidget) {
                        w.value.mute = true;
                        w.value.solo = false;
                    }
                }
                this._iamccsApplyModes?.();
            });
            if (!isReceiver) this.addWidget("button", "Enable all", null, () => {
                if (this.properties?.iamccs_bus_group_lock_groups) return;
                for (const w of this.widgets || []) {
                    if (w instanceof BusGroupRowWidget) {
                        w.value.mute = false;
                        w.value.solo = false;
                    }
                }
                this._iamccsApplyModes?.();
            });

            this.addWidget("button", "↻ Refresh groups", null, () => this._iamccsRefresh());

            // A bit of padding after the header widgets.
            const headerPad = new BusGroupSpacerWidget(5);
            this.addCustomWidget(headerPad);
            this._iamccsHeaderPadWidget = headerPad;

            // Track which widgets are header options (to hide/show them).
            try {
                // Everything currently in widgets is header stuff (rows are added in rebuildWidgets).
                this._iamccsHeaderWidgets = Array.from(this.widgets || []);
            } catch {
                this._iamccsHeaderWidgets = [];
            }

            this._iamccsUpdateHeaderVisibility = () => {
                const hide = !!this.properties.iamccs_bus_group_hide_options;
                for (const w of this._iamccsHeaderWidgets || []) {
                    // Never hide the three new controls.
                    if (w === this._iamccsShowModeWidget || w === this._iamccsShowOptionsWidget || w === this._iamccsWidgetColorsBtn) {
                        setWidgetHidden(w, false);
                        continue;
                    }
                    // Keep a tiny padding below header even when options are hidden.
                    if (w === this._iamccsHeaderPadWidget) {
                        setWidgetHidden(w, false);
                        continue;
                    }
                    setWidgetHidden(w, hide);
                }
                try { this.graph?.setDirtyCanvas(true, true); } catch {}
            };

            // Apply initial visibility
            this._iamccsUpdateHeaderVisibility();

            // Rebuild now.
            setTimeout(() => this._iamccsRefresh(), 0);

            return r;
        };

        const onAdded = nodeType.prototype.onAdded;
        nodeType.prototype.onAdded = function (graph) {
            const r = onAdded?.apply(this, arguments);
            setTimeout(() => this._iamccsRefresh?.(), 0);
            return r;
        };

        // Restore persisted header UI state on workflow load.
        // When loading a workflow, widgets are created before properties are applied by LiteGraph.
        // Without this, the toggle can visually reset even if the property is correctly serialized.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure?.apply(this, arguments);
            try {
                if (this.properties.iamccs_bus_group_hide_options == null) {
                    this.properties.iamccs_bus_group_hide_options = false;
                }
                if (this._iamccsShowOptionsWidget) {
                    this._iamccsShowOptionsWidget.value = !!this.properties.iamccs_bus_group_hide_options;
                }
                this._iamccsUpdateHeaderVisibility?.();
            } catch {}
            return r;
        };

        // Auto-refresh if groups change (best-effort, cheap signature).
        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            try {
                const graph = getGraph();
                const sig = getGroups(graph).map(getGroupKey).join("|");
                if (sig !== this._iamccsLastGroupSig) {
                    // Debounce via timeout to avoid rebuild inside draw.
                    clearTimeout(this._iamccsRefreshTimeout);
                    this._iamccsRefreshTimeout = setTimeout(() => this._iamccsRefresh?.(), 50);
                }

                // Receiver indicator
                if (isReceiver && this.properties?.iamccs_receiver_active) {
                    ctx.save();
                    ctx.fillStyle = "rgba(255,255,255,0.85)";
                    ctx.font = "bold 12px sans-serif";
                    ctx.textAlign = "left";
                    ctx.textBaseline = "top";
                    ctx.fillText("R", 8, 2);
                    ctx.restore();
                }
            } catch {}
            return onDrawForeground?.apply(this, arguments);
        };
    },
});
