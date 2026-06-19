// Canvas widget for the IAMCCS SCAIL Identity Tracker node.
// Draw ordered per-person markers on the reference image and the driving video's
// first frame. Two marker kinds:
//   - box: selectable, draggable, resizable.
//   - point identity: one or more clicks (positive/negative) that SAM3 unions into
//     the whole person. Click = new identity; Shift+click = add positive point;
//     Alt+click = add negative point. Two+ points make SAM return the whole object
//     instead of the sub-part under a single click.
// Right-click removes a box / a single point (or the identity if it was its last
// point); select + Delete also works. Placement order = identity (colour) order,
// matching the model palette in nodes_scail.py. Markers serialise into the hidden
// "markers" STRING widget as {"reference":[...], "driving":[...]}.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Must match DEFAULT_PALETTE order in comfy_extras/nodes_scail.py
const PALETTE = ["#0000ff", "#ff0000", "#00ff00", "#ff00ff", "#00ffff", "#ffff00"];
const CANVAS_H = 320;
const HANDLE = 5;   // half-size of a resize handle, screen px
const HIT_PX = 12;  // point hit radius, screen px

function viewURL(info) {
    const qs = `filename=${encodeURIComponent(info.filename)}&type=${info.type}` +
               `&subfolder=${encodeURIComponent(info.subfolder || "")}&rand=${Math.random()}`;
    const path = `/view?${qs}`;
    return (api && typeof api.apiURL === "function") ? api.apiURL(path) : path;
}

// 8 resize handles for a box: corners + edge midpoints. ax/ay in {0,0.5,1}.
const HANDLES = [
    [0, 0], [0.5, 0], [1, 0],
    [0, 0.5],         [1, 0.5],
    [0, 1], [0.5, 1], [1, 1],
];

// Legacy {type:'point',x,y} -> {type:'point',points:[[x,y,1]]}
function normMarker(m) {
    if (m && m.type === "point" && !Array.isArray(m.points)) {
        return { type: "point", points: [[m.x, m.y, 1]] };
    }
    return m;
}

function setupNode(node) {
    const markersWidget = node.widgets?.find((w) => w.name === "markers");
    if (markersWidget) {
        markersWidget.hidden = true;
        markersWidget.computeSize = () => [0, -4];
        markersWidget.draw = function () {};
    }

    const state = {
        side: "reference",
        mode: "box",
        markers: { reference: [], driving: [] },
        imgs: { reference: null, driving: null },
        view: { scale: 1, ox: 0, oy: 0 },
        selected: -1,
        action: null, // {type:'draw'|'move'|'resize', ...}
    };
    node._scail = state;

    const syncFromWidget = () => {
        if (!markersWidget) return;
        try {
            const parsed = JSON.parse(markersWidget.value || "{}");
            state.markers.reference = (Array.isArray(parsed.reference) ? parsed.reference : []).map(normMarker);
            state.markers.driving = (Array.isArray(parsed.driving) ? parsed.driving : []).map(normMarker);
        } catch (e) { /* keep current */ }
    };
    syncFromWidget();

    const writeMarkers = () => {
        if (markersWidget) markersWidget.value = JSON.stringify(state.markers);
        updateWarning();
        node.graph?.setDirtyCanvas(true, true);
    };

    // --- DOM ---
    const container = document.createElement("div");
    container.style.cssText = "display:flex;flex-direction:column;gap:4px;width:100%;";
    container.tabIndex = 0;

    const bar = document.createElement("div");
    bar.style.cssText = "display:flex;flex-wrap:wrap;gap:4px;font-size:11px;align-items:center;";
    container.appendChild(bar);

    const mkBtn = (label, on) => {
        const b = document.createElement("button");
        b.textContent = label;
        b.style.cssText = "padding:2px 6px;cursor:pointer;border-radius:4px;border:1px solid #555;background:#2a2a2a;color:#ddd;";
        b.onclick = (e) => { e.preventDefault(); e.stopPropagation(); on(b); redraw(); };
        bar.appendChild(b);
        return b;
    };

    const refBtn = mkBtn("Reference", () => { state.side = "reference"; state.selected = -1; });
    const drvBtn = mkBtn("Driving", () => { state.side = "driving"; state.selected = -1; });
    const sep = document.createElement("span"); sep.style.cssText = "width:8px;"; bar.appendChild(sep);
    const boxBtn = mkBtn("Box", () => (state.mode = "box"));
    const ptBtn = mkBtn("Point", () => (state.mode = "point"));
    const sep2 = document.createElement("span"); sep2.style.cssText = "width:8px;"; bar.appendChild(sep2);
    mkBtn("Undo", () => { state.markers[state.side].pop(); state.selected = -1; writeMarkers(); });
    mkBtn("Clear", () => { state.markers[state.side] = []; state.selected = -1; writeMarkers(); });

    const hint = document.createElement("div");
    hint.style.cssText = "font-size:10px;color:#999;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;height:14px;line-height:14px;";
    container.appendChild(hint);

    const canvas = document.createElement("canvas");
    canvas.style.cssText = "width:100%;height:" + CANVAS_H + "px;background:#1a1a1a;border-radius:4px;display:block;touch-action:none;";
    container.appendChild(canvas);
    const ctx = canvas.getContext("2d");

    const warningDiv = document.createElement("div");
    warningDiv.style.cssText = "font-size:10px;color:#e66;height:30px;line-height:13px;overflow:hidden;margin-top:2px;";
    container.appendChild(warningDiv);

    function updateWarning() {
        const ad = node.widgets?.find((w) => w.name === "auto_detect");
        const autoDetect = ad ? ad.value : true;
        const refN = state.markers.reference.length;
        const drvN = state.markers.driving.length;
        if (!autoDetect && refN < drvN) {
            warningDiv.textContent = `⚠ ${refN} reference vs ${drvN} driving subject(s): ${drvN - refN} driving subject(s) have no reference to map to (auto_detect is off).`;
        } else {
            warningDiv.textContent = "";
        }
    }

    const activeImg = () => state.imgs[state.side];
    const activeMarks = () => state.markers[state.side];

    function refreshView() {
        const w = Math.max(16, canvas.clientWidth || node.size[0] - 20);
        if (canvas.width !== w) canvas.width = w;
        if (canvas.height !== CANVAS_H) canvas.height = CANVAS_H;
        const img = activeImg();
        if (!img) { state.view = { scale: 1, ox: 0, oy: 0 }; return; }
        const scale = Math.min(canvas.width / img.naturalWidth, canvas.height / img.naturalHeight);
        state.view = {
            scale,
            ox: (canvas.width - img.naturalWidth * scale) / 2,
            oy: (canvas.height - img.naturalHeight * scale) / 2,
        };
    }

    const imgToScreen = (x, y) => [state.view.ox + x * state.view.scale, state.view.oy + y * state.view.scale];
    function eventToCanvas(e) {
        const r = canvas.getBoundingClientRect();
        return [(e.clientX - r.left) * (canvas.width / r.width), (e.clientY - r.top) * (canvas.height / r.height)];
    }
    const canvasToImg = (px, py) => [(px - state.view.ox) / state.view.scale, (py - state.view.oy) / state.view.scale];

    function handlePositions(box) {
        return HANDLES.map(([ax, ay]) => {
            const [sx, sy] = imgToScreen(box.x + ax * box.w, box.y + ay * box.h);
            return { ax, ay, sx, sy };
        });
    }
    function hitHandle(box, px, py) {
        for (const h of handlePositions(box)) {
            if (Math.abs(px - h.sx) <= HANDLE + 3 && Math.abs(py - h.sy) <= HANDLE + 3) return h;
        }
        return null;
    }
    // nearest point index within a point marker, or -1
    function nearestPoint(m, px, py) {
        let best = -1, bestD = HIT_PX;
        m.points.forEach((p, pi) => {
            const [sx, sy] = imgToScreen(p[0], p[1]);
            const d = Math.hypot(px - sx, py - sy);
            if (d <= bestD) { bestD = d; best = pi; }
        });
        return best;
    }
    function markerAt(ix, iy, px, py) {
        const marks = activeMarks();
        for (let i = marks.length - 1; i >= 0; i--) {
            const m = marks[i];
            if (m.type === "box") {
                if (ix >= m.x && ix <= m.x + m.w && iy >= m.y && iy <= m.y + m.h) return i;
            } else if (nearestPoint(m, px, py) >= 0) {
                return i;
            }
        }
        return -1;
    }

    function normalizeBox(b) {
        if (b.w < 0) { b.x += b.w; b.w = -b.w; }
        if (b.h < 0) { b.y += b.h; b.h = -b.h; }
    }

    // --- drawing ---
    function redraw() {
        refreshView();
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        for (const b of [refBtn, drvBtn]) b.style.outline = "";
        (state.side === "reference" ? refBtn : drvBtn).style.outline = "2px solid #6cf";
        for (const b of [boxBtn, ptBtn]) b.style.outline = "";
        (state.mode === "box" ? boxBtn : ptBtn).style.outline = "2px solid #6cf";

        const img = activeImg();
        if (!img) {
            ctx.fillStyle = "#777"; ctx.font = "12px sans-serif";
            ctx.fillText("Press the node's ▶ play button to load this frame", 12, 24);
            hint.textContent = "Box: drag. Point: click=new, Shift+click=add, Alt+click=negative. Right-click/Delete removes.";
            return;
        }
        ctx.drawImage(img, state.view.ox, state.view.oy, img.naturalWidth * state.view.scale, img.naturalHeight * state.view.scale);

        const marks = activeMarks();
        marks.forEach((m, i) => {
            const color = PALETTE[i % PALETTE.length];
            const sel = i === state.selected;
            ctx.lineWidth = 2;
            ctx.strokeStyle = color;
            ctx.fillStyle = color;
            if (m.type === "box") {
                const [sx, sy] = imgToScreen(m.x, m.y);
                ctx.strokeRect(sx, sy, m.w * state.view.scale, m.h * state.view.scale);
                drawLabel(i + 1, color, sx + 2, sy + 2);
                if (sel) {
                    ctx.fillStyle = "#fff";
                    for (const h of handlePositions(m)) {
                        ctx.fillRect(h.sx - HANDLE, h.sy - HANDLE, HANDLE * 2, HANDLE * 2);
                        ctx.strokeRect(h.sx - HANDLE, h.sy - HANDLE, HANDLE * 2, HANDLE * 2);
                    }
                }
            } else {
                m.points.forEach((p) => {
                    const [sx, sy] = imgToScreen(p[0], p[1]);
                    const positive = p[2] !== 0;
                    const r = sel ? 6 : 5;
                    ctx.beginPath(); ctx.arc(sx, sy, r, 0, Math.PI * 2);
                    ctx.fillStyle = positive ? color : "#222";
                    ctx.fill();
                    ctx.lineWidth = 2;
                    ctx.strokeStyle = positive ? (sel ? "#fff" : "#000") : color;
                    ctx.stroke();
                    if (!positive) { // white minus for negative points
                        ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5;
                        ctx.beginPath(); ctx.moveTo(sx - 3, sy); ctx.lineTo(sx + 3, sy); ctx.stroke();
                    }
                });
                const [lx, ly] = imgToScreen(m.points[0][0], m.points[0][1]);
                drawLabel(i + 1, color, lx + 8, ly - 8);
            }
        });

        if (state.action && state.action.type === "draw") {
            const a = state.action;
            const [sx, sy] = imgToScreen(Math.min(a.x0, a.x1), Math.min(a.y0, a.y1));
            ctx.setLineDash([4, 3]);
            ctx.strokeStyle = PALETTE[marks.length % PALETTE.length];
            ctx.strokeRect(sx, sy, Math.abs(a.x1 - a.x0) * state.view.scale, Math.abs(a.y1 - a.y0) * state.view.scale);
            ctx.setLineDash([]);
        }
        const smsg = state.selected >= 0 ? `, #${state.selected + 1} selected` : "";
        hint.textContent = `${state.side}: ${marks.length} id(s)${smsg}. Point: Shift+click add, Alt+click negative. Right-click/Delete removes.`;
    }

    function drawLabel(num, color, x, y) {
        ctx.font = "bold 13px sans-serif";
        ctx.fillStyle = "#000"; ctx.fillText(String(num), x + 1, y + 14);
        ctx.fillStyle = color; ctx.fillText(String(num), x, y + 13);
    }
    node._scailRedraw = redraw;

    // --- interaction ---
    canvas.addEventListener("pointerdown", (e) => {
        if (e.button !== 0 || !activeImg()) return;
        container.focus();
        refreshView();
        const [px, py] = eventToCanvas(e);
        const [ix, iy] = canvasToImg(px, py);
        const marks = activeMarks();
        const addMod = e.shiftKey || e.altKey;

        // Add a point to the selected point identity (Shift = positive, Alt = negative)
        if (addMod && state.selected >= 0 && marks[state.selected]?.type === "point") {
            marks[state.selected].points.push([ix, iy, e.altKey ? 0 : 1]);
            writeMarkers();
            redraw();
            return;
        }

        // 1. resize handle of the selected box
        if (state.selected >= 0 && marks[state.selected]?.type === "box") {
            const h = hitHandle(marks[state.selected], px, py);
            if (h) {
                state.action = { type: "resize", handle: h, orig: { ...marks[state.selected] } };
                canvas.setPointerCapture(e.pointerId);
                return;
            }
        }
        // 2. select + move an existing marker
        const idx = markerAt(ix, iy, px, py);
        if (idx >= 0) {
            state.selected = idx;
            state.action = { type: "move", start: [ix, iy], orig: JSON.parse(JSON.stringify(marks[idx])) };
            canvas.setPointerCapture(e.pointerId);
            redraw();
            return;
        }
        // 3. empty space -> deselect + start a new marker
        state.selected = -1;
        if (state.mode === "box") {
            state.action = { type: "draw", x0: ix, y0: iy, x1: ix, y1: iy };
        } else {
            marks.push({ type: "point", points: [[ix, iy, 1]] });
            state.selected = marks.length - 1;
            state.action = { type: "move", start: [ix, iy], orig: JSON.parse(JSON.stringify(marks[state.selected])) };
        }
        canvas.setPointerCapture(e.pointerId);
        redraw();
    });

    canvas.addEventListener("pointermove", (e) => {
        if (!state.action) return;
        const [px, py] = eventToCanvas(e);
        const [ix, iy] = canvasToImg(px, py);
        const a = state.action;
        const marks = activeMarks();

        if (a.type === "draw") {
            a.x1 = ix; a.y1 = iy;
        } else if (a.type === "move") {
            const dx = ix - a.start[0], dy = iy - a.start[1];
            const m = marks[state.selected];
            if (m.type === "box") {
                m.x = a.orig.x + dx; m.y = a.orig.y + dy;
            } else {
                m.points = a.orig.points.map((p) => [p[0] + dx, p[1] + dy, p[2]]);
            }
        } else if (a.type === "resize") {
            const m = marks[state.selected], o = a.orig, { ax, ay } = a.handle;
            let left = o.x, right = o.x + o.w, top = o.y, bottom = o.y + o.h;
            if (ax === 0) left = ix; else if (ax === 1) right = ix;
            if (ay === 0) top = iy; else if (ay === 1) bottom = iy;
            m.x = Math.min(left, right); m.w = Math.abs(right - left);
            m.y = Math.min(top, bottom); m.h = Math.abs(bottom - top);
        }
        redraw();
    });

    function endAction(e) {
        if (!state.action) return;
        const a = state.action;
        const marks = activeMarks();
        if (a.type === "draw") {
            const x = Math.min(a.x0, a.x1), y = Math.min(a.y0, a.y1);
            const w = Math.abs(a.x1 - a.x0), h = Math.abs(a.y1 - a.y0);
            if (w > 3 && h > 3) { marks.push({ type: "box", x, y, w, h }); state.selected = marks.length - 1; }
        } else if (a.type === "resize" || a.type === "move") {
            if (marks[state.selected]?.type === "box") normalizeBox(marks[state.selected]);
        }
        state.action = null;
        if (e && canvas.hasPointerCapture?.(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
        writeMarkers();
        redraw();
    }
    canvas.addEventListener("pointerup", endAction);
    canvas.addEventListener("pointercancel", () => { state.action = null; redraw(); });

    canvas.addEventListener("contextmenu", (e) => {
        e.preventDefault();
        if (!activeImg()) return;
        refreshView();
        const [px, py] = eventToCanvas(e);
        const [ix, iy] = canvasToImg(px, py);
        const marks = activeMarks();
        const idx = markerAt(ix, iy, px, py);
        if (idx < 0) return;
        const m = marks[idx];
        if (m.type === "point" && m.points.length > 1) {
            const pi = nearestPoint(m, px, py);  // remove just the clicked point
            if (pi >= 0) m.points.splice(pi, 1);
        } else {
            marks.splice(idx, 1);
            state.selected = -1;
        }
        writeMarkers();
        redraw();
    });

    container.addEventListener("keydown", (e) => {
        if ((e.key === "Delete" || e.key === "Backspace") && state.selected >= 0) {
            e.preventDefault(); e.stopPropagation();
            activeMarks().splice(state.selected, 1);
            state.selected = -1;
            writeMarkers();
            redraw();
        }
    });

    const widget = node.addDOMWidget("scail_canvas", "scail_canvas", container, { serialize: false, hideOnZoom: false });
    widget.computeSize = function () { return [node.size[0], CANVAS_H + 90]; };

    const adWidget = node.widgets?.find((w) => w.name === "auto_detect");
    if (adWidget) {
        const origCb = adWidget.callback;
        adWidget.callback = function () { const r = origCb?.apply(this, arguments); updateWarning(); return r; };
    }

    node._scailOnExecuted = (message) => {
        const load = (side, info) => {
            if (!info) return;
            const im = new Image();
            im.onload = () => { state.imgs[side] = im; if (state.side === side) redraw(); };
            im.src = viewURL(info);
        };
        load("reference", message?.reference_preview?.[0]);
        load("driving", message?.driving_preview?.[0]);
    };

    node._scailSync = () => { syncFromWidget(); redraw(); };

    if (node.size[1] < CANVAS_H + 170) node.size[1] = CANVAS_H + 170;
    setTimeout(() => { redraw(); updateWarning(); }, 50);
}

app.registerExtension({
    name: "iamccs.scailIdentityTracker",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "IAMCCS_ScailIdentityTracker") return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            setupNode(this);
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            this._scailOnExecuted?.(message);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            this._scailSync?.();
            return r;
        };
    },
});
