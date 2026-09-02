import { app } from "../../scripts/app.js";

const CLASS_NAME = "IAMCCS_CineShotboardPlannerV3B";
const COLORS = ["#71e2ff", "#ff7896", "#ffd166", "#8cff98", "#c6a0ff", "#ff9f43"];

function klass(node) { return String(node?.comfyClass || node?.type || ""); }
function widget(node, name) { return (node.widgets || []).find((item) => item.name === name); }
function clamp(v, a = 0, b = 1) { return Math.max(a, Math.min(b, Number(v) || 0)); }
function uid() { return `bbox_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`; }
function parseProject(node) {
    const timelineWidget = widget(node, "timeline_data");
    try {
        const timeline = JSON.parse(String(timelineWidget?.value || "{}"));
        const embedded = timeline?.bboxControl || timeline?.bbox_control;
        if (embedded && typeof embedded === "object") return JSON.parse(JSON.stringify(embedded));
    } catch (_) {}
    try {
        const value = JSON.parse(String(widget(node, "bbox_project_json")?.value || "{}"));
        if (value && typeof value === "object") return value;
    } catch (_) {}
    return { schema: "iamccs.ltx25.shotboard.v3b.bbox", version: 1, style_prompt: "", scene_prompt: "", objects: [] };
}
function saveProject(node, project) {
    const target = widget(node, "bbox_project_json");
    if (target) target.value = JSON.stringify(project);
    const timelineWidget = widget(node, "timeline_data");
    try {
        const liveTimeline = node._iamccsV3BTimelineController?.getTimeline?.();
        const timeline = liveTimeline && typeof liveTimeline === "object"
            ? liveTimeline
            : JSON.parse(String(timelineWidget?.value || "{}"));
        timeline.bboxControl = JSON.parse(JSON.stringify(project));
        timeline.bbox_control = timeline.bboxControl;
        if (timelineWidget) timelineWidget.value = JSON.stringify(timeline);
    } catch (_) {}
    node._iamccsV3BTimelineController?.writeTimeline?.();
    node.graph?.setDirtyCanvas(true, true);
}
function boxAt(object, time) {
    const frames = [...(object?.keyframes || [])].sort((a, b) => a.time - b.time);
    if (!frames.length) return null;
    if (time <= frames[0].time) return [...frames[0].box];
    if (time >= frames.at(-1).time) return [...frames.at(-1).box];
    for (let i = 0; i < frames.length - 1; i++) {
        const a = frames[i], b = frames[i + 1];
        if (time >= a.time && time <= b.time) {
            const r = (time - a.time) / Math.max(1e-6, b.time - a.time);
            return a.box.map((v, j) => v + (b.box[j] - v) * r);
        }
    }
    return [...frames[0].box];
}
function setKeyframe(object, time, box) {
    object.keyframes ||= [];
    const rounded = Math.round(time * 10000) / 10000;
    const existing = object.keyframes.find((item) => Math.abs(item.time - rounded) < 0.0002);
    if (existing) existing.box = box.map(Number);
    else object.keyframes.push({ time: rounded, box: box.map(Number) });
    object.keyframes.sort((a, b) => a.time - b.time);
}
function el(tag, attrs = {}, ...children) {
    const out = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
        if (key === "style") out.style.cssText = value;
        else if (key.startsWith("on")) out.addEventListener(key.slice(2), value);
        else if (key === "class") out.className = value;
        else out[key] = value;
    }
    for (const child of children) out.append(child);
    return out;
}

function openDirector(node) {
    const project = parseProject(node);
    project.objects ||= [];
    const controller = node._iamccsV3BTimelineController;
    const totalFrames = Math.max(1, Number(controller?.getTotalFrames?.() || 1));
    let selected = project.objects[0]?.id || null;
    let time = clamp(Number(controller?.getPlayFrame?.() || 0) / totalFrames);
    let imageOverride = "";
    let playing = false;
    let viewMode = "overlay";
    let playTimer = 0;
    let drag = null;

    const shade = el("div", { style: "position:fixed;inset:0;z-index:100000;background:#080a0edc;display:flex;align-items:center;justify-content:center;font:13px Inter,Arial;color:#eef2f7" });
    const shell = el("div", { style: "width:min(1460px,96vw);height:min(920px,94vh);background:#141821;border:1px solid #394150;border-radius:12px;display:grid;grid-template-rows:52px 1fr;overflow:hidden;box-shadow:0 28px 90px #000" });
    const bar = el("div", { style: "display:flex;align-items:center;gap:10px;padding:0 16px;background:#0e1118;border-bottom:1px solid #303744" });
    const title = el("strong", {}, "SHOTBOARD V3B · LTX 2.5 BBOX DIRECTOR");
    const status = el("span", { style: "margin-left:auto;color:#8da0b8" }, "Draw, move or resize the selected object at the playhead");
    const button = (text, fn, primary = false) => el("button", { onclick: fn, style: `border:1px solid ${primary ? "#e89b35" : "#465064"};background:${primary ? "#bd711d" : "#242b37"};color:white;border-radius:6px;padding:7px 12px;cursor:pointer` }, text);
    const close = (commit) => { clearInterval(playTimer); if (commit) saveProject(node, project); shade.remove(); };
    bar.append(title, status, button("Cancel", () => close(false)), button("Save & Close", () => close(true), true));

    const body = el("div", { style: "min-height:0;display:grid;grid-template-columns:300px minmax(500px,1fr) 330px" });
    const left = el("div", { style: "padding:12px;border-right:1px solid #303744;overflow:auto" });
    const center = el("div", { style: "padding:14px;min-width:0;display:flex;flex-direction:column;gap:10px;background:#0b0e13" });
    const right = el("div", { style: "padding:12px;border-left:1px solid #303744;overflow:auto" });
    body.append(left, center, right); shell.append(bar, body); shade.append(shell); document.body.append(shade);

    const field = (label, value, oninput, multiline = false) => {
        const control = multiline
            ? el("textarea", { value, oninput: e => oninput(e.target.value), style: "width:100%;height:82px;box-sizing:border-box;background:#0c1017;color:#edf2f8;border:1px solid #384252;border-radius:6px;padding:8px;resize:vertical" })
            : el("input", { value, oninput: e => oninput(e.target.value), style: "width:100%;box-sizing:border-box;background:#0c1017;color:#edf2f8;border:1px solid #384252;border-radius:6px;padding:7px" });
        return el("label", { style: "display:block;margin-bottom:10px;color:#aeb9c9" }, el("div", { style: "margin-bottom:5px;font-weight:600" }, label), control);
    };
    const referencePaths = () => controller?.getReferencePaths?.() || String(widget(node, "image_paths")?.value || "").split(/\r?\n/).map(v => v.trim()).filter(Boolean);
    const segmentPathAt = (normalizedTime) => {
        const frameIndex = Math.round(normalizedTime * totalFrames);
        const timeline = controller?.getTimeline?.() || {};
        const segment = (timeline.segments || []).find(seg => String(seg?.type || "image") !== "text" && frameIndex >= Number(seg.start || 0) && frameIndex < Number(seg.start || 0) + Number(seg.length || 1));
        if (!segment) return referencePaths()[0] || "";
        const explicit = String(segment.imageTruthPath || segment.image_file || segment.imageFile || segment.path || "").trim();
        return explicit || referencePaths()[Math.max(0, Number(segment.ref || 1) - 1)] || "";
    };
    const imageSelect = el("select", { onchange: e => { imageOverride = e.target.value; render(); }, style: "width:100%;box-sizing:border-box;background:#0c1017;color:#edf2f8;border:1px solid #384252;border-radius:6px;padding:7px;margin-bottom:10px" });
    imageSelect.append(el("option", { value: "" }, "Follow active timeline image"));
    referencePaths().forEach((path, index) => imageSelect.append(el("option", { value: path }, `Slot ${index + 1} · ${path.split(/[\\/]/).pop()}`)));
    const activeMedia = el("div", { style: "padding:8px;margin-bottom:10px;border:1px solid #384252;border-radius:6px;background:#0c1017;color:#91a3b8;font-size:11px;word-break:break-all" });
    left.append(
        field("STYLE · shared lighting/camera/colors", project.style_prompt || "", v => project.style_prompt = v, true),
        field("SCENE · environment + object count/action", project.scene_prompt || "", v => project.scene_prompt = v, true),
        el("div", { style: "margin-bottom:5px;color:#aeb9c9;font-weight:600" }, "CANVAS SOURCE · Shotboard image slots"),
        imageSelect,
        activeMedia
    );
    left.append(el("p", { style: "color:#748398;font-size:11px;line-height:1.45" }, "Add images with the normal V3B slots and drag their clips on the main timeline. This canvas follows the image under the same playhead automatically."));

    const stage = el("div", { style: "position:relative;flex:1;min-height:0;display:flex;align-items:center;justify-content:center;overflow:hidden;border:1px solid #303744;border-radius:8px;background:#030405" });
    const frame = el("div", { style: "position:relative;width:min(100%,960px);aspect-ratio:768/448;background:#05070a;overflow:hidden" });
    const bg = el("img", { src: "", style: "position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:.72;pointer-events:none" });
    const canvas = el("canvas", { style: "position:absolute;inset:0;width:100%;height:100%;touch-action:none;cursor:crosshair" });
    frame.append(bg, canvas); stage.append(frame); center.append(stage);
    const transport = el("div", { style: "display:grid;grid-template-columns:auto auto auto 1fr auto;align-items:center;gap:8px" });
    const play = button("▶ Play", () => {
        playing = !playing; play.textContent = playing ? "■ Stop" : "▶ Play";
        clearInterval(playTimer);
        if (playing) playTimer = setInterval(() => { time += 1 / Math.max(1, totalFrames); if (time > 1) time = 0; timeline.value = String(time); controller?.setPlayFrame?.(Math.round(time * totalFrames)); render(); }, 1000 / Math.max(1, Number(controller?.getFps?.() || 24)));
    });
    const timeline = el("input", { type: "range", min: 0, max: 1, step: 0.001, value: time, oninput: e => { time = Number(e.target.value); controller?.setPlayFrame?.(Math.round(time * totalFrames)); render(); } });
    const timeLabel = el("span", { style: "font-variant-numeric:tabular-nums;color:#e5a34a" }, "0.0%");
    const overlayMode = button("Overlay", () => { viewMode = "overlay"; render(); });
    const signalMode = button("Control signal", () => { viewMode = "signal"; render(); });
    transport.append(play, overlayMode, signalMode, timeline, timeLabel); center.append(transport);

    const objectHeader = el("div", { style: "display:flex;align-items:center;gap:8px;margin-bottom:10px" }, el("strong", {}, "OBJECT TRACKS"));
    objectHeader.append(button("+ Object", () => {
        const object = { id: uid(), name: `Object ${project.objects.length + 1}`, prompt: "", color: COLORS[project.objects.length % COLORS.length], strength: 1, enabled: true, start_time: 0, end_time: 1, keyframes: [{ time: 0, box: [.1, .15, .35, .85] }, { time: 1, box: [.55, .15, .8, .85] }] };
        project.objects.push(object); selected = object.id; rebuildObjects(); render();
    })); right.append(objectHeader);
    const list = el("div"); right.append(list);
    const inspector = el("div", { style: "margin-top:14px;padding-top:12px;border-top:1px solid #303744" }); right.append(inspector);

    function selectedObject() { return project.objects.find(o => o.id === selected); }
    function rebuildObjects() {
        list.replaceChildren();
        for (const object of project.objects) {
            const row = el("button", { onclick: () => { selected = object.id; rebuildObjects(); rebuildInspector(); render(); }, style: `width:100%;display:grid;grid-template-columns:10px 1fr auto;gap:8px;align-items:center;text-align:left;margin:4px 0;padding:9px;border-radius:6px;border:1px solid ${selected === object.id ? object.color : "#343d4c"};background:${selected === object.id ? "#26303d" : "#191e27"};color:white;cursor:pointer` },
                el("span", { style: `width:9px;height:28px;background:${object.color};border-radius:3px` }),
                el("span", {}, object.name || "Object"),
                el("span", { style: "color:#8c9aab" }, `${object.keyframes?.length || 0} KF`)
            ); list.append(row);
        }
        rebuildInspector();
    }
    function rebuildInspector() {
        inspector.replaceChildren(); const object = selectedObject(); if (!object) return;
        inspector.append(
            field("Object name", object.name || "", v => { object.name = v; rebuildObjects(); }),
            field("Regional prompt · appearance/material only", object.prompt || "", v => object.prompt = v, true)
        );
        const strength = el("input", { type: "number", min: 0, max: 5, step: .05, value: object.strength ?? 1, oninput: e => object.strength = Number(e.target.value) });
        const enabled = el("input", { type: "checkbox", checked: object.enabled !== false, onchange: e => { object.enabled = e.target.checked; render(); } });
        inspector.append(el("label", {}, " Strength ", strength), el("label", { style: "margin-left:12px" }, enabled, " enabled"));
        const start = el("input", { type: "number", min: 0, max: 1, step: .01, value: object.start_time ?? 0, style: "width:64px", oninput: e => { object.start_time = clamp(e.target.value); render(); } });
        const end = el("input", { type: "number", min: 0, max: 1, step: .01, value: object.end_time ?? 1, style: "width:64px", oninput: e => { object.end_time = clamp(e.target.value); render(); } });
        inspector.append(el("div", { style: "margin-top:10px;color:#aeb9c9" }, "Active range ", start, " → ", end));
        inspector.append(el("div", { style: "display:flex;gap:7px;margin-top:12px;flex-wrap:wrap" },
            button("Set keyframe", () => { const b = boxAt(object, time) || [.1,.1,.4,.8]; setKeyframe(object, time, b); rebuildObjects(); render(); }),
            button("Delete KF", () => { object.keyframes = (object.keyframes || []).filter(k => Math.abs(k.time - time) > .006); render(); }),
            button("Delete object", () => { project.objects = project.objects.filter(o => o.id !== object.id); selected = project.objects[0]?.id || null; rebuildObjects(); render(); })
        ));
    }
    function canvasPoint(e) {
        const rect = canvas.getBoundingClientRect(); return [clamp((e.clientX - rect.left) / rect.width, -2, 3), clamp((e.clientY - rect.top) / rect.height, -2, 3)];
    }
    function hitMode(box, p) {
        if (!box) return "draw"; const d = .025;
        if (Math.hypot(p[0] - box[2], p[1] - box[3]) < d) return "resize";
        if (p[0] >= box[0] && p[0] <= box[2] && p[1] >= box[1] && p[1] <= box[3]) return "move";
        return "draw";
    }
    canvas.addEventListener("pointerdown", e => {
        const object = selectedObject(); if (!object) return;
        const p = canvasPoint(e), box = boxAt(object, time);
        drag = { mode: hitMode(box, p), start: p, box: box ? [...box] : [p[0], p[1], p[0], p[1]] };
        canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", e => {
        if (!drag) return; const object = selectedObject(), p = canvasPoint(e); if (!object) return;
        let b = [...drag.box];
        if (drag.mode === "move") { const dx = p[0] - drag.start[0], dy = p[1] - drag.start[1]; b = [b[0]+dx,b[1]+dy,b[2]+dx,b[3]+dy]; }
        else if (drag.mode === "resize") { b[2] = p[0]; b[3] = p[1]; }
        else b = [Math.min(drag.start[0],p[0]),Math.min(drag.start[1],p[1]),Math.max(drag.start[0],p[0]),Math.max(drag.start[1],p[1])];
        if (b[2] - b[0] > .005 && b[3] - b[1] > .005) setKeyframe(object, time, b);
        render();
    });
    canvas.addEventListener("pointerup", () => { drag = null; rebuildObjects(); });

    function render() {
        const activePath = imageOverride || segmentPathAt(time);
        const activeUrl = activePath ? (controller?.previewUrlForPath?.(activePath) || activePath) : "";
        if (bg.dataset.path !== activePath) { bg.dataset.path = activePath; bg.src = activeUrl; }
        bg.style.display = viewMode === "overlay" ? "block" : "none";
        overlayMode.style.background = viewMode === "overlay" ? "#36516b" : "#242b37";
        signalMode.style.background = viewMode === "signal" ? "#36516b" : "#242b37";
        activeMedia.textContent = activePath ? `Active: ${activePath}` : "No image under playhead · add an image slot in Shotboard V3B";
        const dpr = devicePixelRatio || 1, rect = canvas.getBoundingClientRect();
        canvas.width = Math.max(1, Math.round(rect.width * dpr)); canvas.height = Math.max(1, Math.round(rect.height * dpr));
        const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr); ctx.clearRect(0,0,rect.width,rect.height);
        if (viewMode === "signal") { ctx.fillStyle = "#000"; ctx.fillRect(0, 0, rect.width, rect.height); }
        const visible = [];
        for (const object of project.objects) {
            if (object.enabled === false || time < (object.start_time ?? 0) || time > (object.end_time ?? 1)) continue;
            const b = boxAt(object, time); if (!b) continue;
            visible.push([object, b]);
            const x=b[0]*rect.width,y=b[1]*rect.height,w=(b[2]-b[0])*rect.width,h=(b[3]-b[1])*rect.height;
            if (viewMode === "signal") {
                ctx.save(); ctx.strokeStyle="#fff"; ctx.fillStyle="#fff"; ctx.lineWidth=2; ctx.setLineDash([]);
                ctx.strokeRect(x,y,w,h); ctx.beginPath(); ctx.arc(x+w/2,y+h/2,2.5,0,Math.PI*2); ctx.fill();
                ctx.beginPath(); ctx.rect(x,y,w,h); ctx.clip(); ctx.globalAlpha=.48;
                for (let trail=4; trail>=1; trail--) {
                    const previous = boxAt(object, clamp(time - trail / totalFrames)); if (!previous) continue;
                    const px=(previous[0]+previous[2])*.5*rect.width, py=(previous[1]+previous[3])*.5*rect.height;
                    ctx.beginPath(); ctx.arc(px,py,2.1,0,Math.PI*2); ctx.fill();
                }
                ctx.restore();
            } else {
                ctx.strokeStyle=object.color||"#fff";ctx.lineWidth=object.id===selected?3:1.5;ctx.setLineDash(object.id===selected?[]:[7,5]);ctx.strokeRect(x,y,w,h);
                ctx.fillStyle=object.color||"#fff";ctx.fillRect(x,y-20,Math.max(70,ctx.measureText(object.name||"Object").width+14),20);ctx.fillStyle="#081018";ctx.fillText(object.name||"Object",x+6,y-6);
                if(object.id===selected){ctx.fillStyle="#fff";ctx.fillRect(x+w-5,y+h-5,10,10);}
            }
        }
        let overlaps = 0;
        for (let i=0;i<visible.length;i++) for(let j=i+1;j<visible.length;j++) {
            const a=visible[i][1],b=visible[j][1];
            if (Math.min(a[2],b[2]) > Math.max(a[0],b[0]) && Math.min(a[3],b[3]) > Math.max(a[1],b[1])) overlaps++;
        }
        const emptyPrompts = project.objects.filter(o => o.enabled !== false && !String(o.prompt || "").trim()).length;
        status.textContent = overlaps ? `Warning: ${overlaps} overlapping BBox pair(s) · training excluded overlaps` : emptyPrompts ? `Warning: ${emptyPrompts} enabled object prompt(s) empty` : viewMode === "signal" ? "Control-signal monitor · exact render is also exposed by the backend" : "BBox project ready · drag inside to move, bottom-right handle to resize";
        status.style.color = overlaps || emptyPrompts ? "#ffad66" : "#89d6a3";
        ctx.setLineDash([]); timeLabel.textContent=`${(time*100).toFixed(1)}%`;
    }
    new ResizeObserver(render).observe(frame); rebuildObjects(); render();
}

app.registerExtension({
    name: "iamccs.ltx25.shotboard.v3b.ui",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (String(nodeData?.name || "") !== CLASS_NAME) return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = original?.apply(this, args);
            const raw = widget(this, "bbox_project_json"); if (raw) { raw.hidden = true; raw.computeSize = () => [0, -4]; }
            this._iamccsOpenLTX25BBoxDirector = () => openDirector(this);
            this.color = "#263445"; this.bgcolor = "#111923";
            return result;
        };
    },
    async nodeCreated(node) { if (klass(node) === CLASS_NAME) node.graph?.setDirtyCanvas(true, true); }
});
