import { app } from "../../scripts/app.js";

const TARGET_CLASS = "IAMCCS_CineMotionSketch";
const STATE_KEY = "__iamccs_motion_sketch_ui";
const UI_W = 1160;
const UI_H = 760;

function widget(node, name) {
    return (node?.widgets || []).find((w) => w?.name === name || w?.label === name) || null;
}

function safeParse(value, fallback = {}) {
    if (value && typeof value === "object") return value;
    try {
        const text = String(value || "").trim();
        return text ? JSON.parse(text) : fallback;
    } catch {
        return fallback;
    }
}

function writeWidget(node, name, value) {
    const w = widget(node, name);
    if (!w) return;
    w.value = typeof value === "string" ? value : JSON.stringify(value, null, 2);
    try { w.callback?.(w.value); } catch {}
    try { app.graph?.setDirtyCanvas?.(true, true); } catch {}
}

function graphNodeById(id) {
    try {
        return app.graph?._nodes_by_id?.[id] || app.graph?.getNodeById?.(id) || null;
    } catch {
        return null;
    }
}

function linkedSourceNode(node, inputName) {
    const input = (node.inputs || []).find((inp) => inp?.name === inputName);
    if (!input || input.link == null) return null;
    const link = app.graph?.links?.[input.link];
    if (!link) return null;
    return graphNodeById(link.origin_id);
}

function normalizeSegment(raw, index) {
    const start = Math.max(0, Math.round(Number(raw?.start ?? raw?.frame ?? 0) || 0));
    const length = Math.max(1, Math.round(Number(raw?.length ?? raw?.frames ?? raw?.len ?? 24) || 24));
    const id = String(raw?.id || raw?.segment_id || raw?.slot_id || `shot_${index + 1}`);
    const imageFile = String(raw?.imageFile || raw?.image_file || raw?.file || raw?.filename || raw?.path || "");
    return {
        id,
        label: String(raw?.label || raw?.name || raw?.refLabel || raw?.image_label || imageFile || id || `Shot ${index + 1}`),
        start,
        length,
        imageFile,
        prompt: String(raw?.prompt || raw?.local_prompt || raw?.localPrompt || ""),
        raw: raw && typeof raw === "object" ? raw : {},
    };
}

function extractSegmentsFromData(data) {
    if (!data || typeof data !== "object") return [];
    const candidates = [
        data.visual_segments,
        data.segments,
        data.timeline_segments,
        data.rows,
        data.timeline?.segments,
        data.payload?.visual_segments,
        data.resources?.cine_payload?.visual_segments,
        data.resources?.cine_payload?.segments,
    ];
    for (const candidate of candidates) {
        if (Array.isArray(candidate) && candidate.length) {
            return candidate.map(normalizeSegment).sort((a, b) => a.start - b.start);
        }
    }
    return [];
}

function collectShotboardCandidates(source) {
    const out = [];
    const push = (value) => {
        if (value == null) return;
        out.push(value);
    };
    for (const name of ["timeline_data", "timeline_json", "shotboard_data", "board_data"]) {
        push(widget(source, name)?.value);
    }
    for (const w of source?.widgets || []) {
        if (typeof w?.value === "string" && /segments|visual_segments|timeline/i.test(w.value)) push(w.value);
    }
    const props = source?.properties || {};
    for (const key of Object.keys(props)) {
        if (/timeline|board|segment|payload/i.test(key)) push(props[key]);
    }
    push(source?.timeline_data);
    push(source?.timelineData);
    push(source?.shotboard_data);
    push(source?.iamccs_timeline_data);
    return out;
}

function readLinkedShotboardSegments(node) {
    const source = linkedSourceNode(node, "cine_linx");
    if (!source) return { segments: [], source: "not connected" };
    for (const candidate of collectShotboardCandidates(source)) {
        const segments = extractSegmentsFromData(safeParse(candidate, null));
        if (segments.length) return { segments, source: `${source.type || source.comfyClass || "node"}:${source.id}` };
    }
    return { segments: [], source: `${source.type || source.comfyClass || "node"}:${source.id} no timeline found` };
}

function imageUrl(file) {
    if (!file) return "";
    const raw = String(file).replace(/\\/g, "/").trim();
    if (/^(https?:|blob:|data:)/i.test(raw)) return raw;
    if (/^[A-Z]:\//i.test(raw)) {
        return `/view?filename=${encodeURIComponent(raw.split("/").pop())}&type=input`;
    }
    return `/view?filename=${encodeURIComponent(raw)}&type=input`;
}

function ensureStyle() {
    if (document.getElementById("iamccs-motion-sketch-style")) return;
    const style = document.createElement("style");
    style.id = "iamccs-motion-sketch-style";
    style.textContent = `
.iamccs-motion-sketch{font-family:Inter,Arial,sans-serif;color:#eaf7f5;background:#071011;border:1px solid #2e4948;border-radius:8px;padding:12px;box-sizing:border-box;width:${UI_W}px;min-width:${UI_W}px;height:${UI_H}px;min-height:${UI_H}px;overflow:hidden}
.iamccs-ms-top{display:grid;grid-template-columns:1fr auto;gap:12px;align-items:center;height:42px;margin-bottom:10px}
.iamccs-ms-status{border:1px solid #284241;border-radius:6px;background:#081516;color:#9fe3e8;font:700 11px/1.2 monospace;padding:7px 10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.iamccs-ms-actions{display:flex;gap:8px;align-items:center}
.iamccs-ms-btn,.iamccs-ms-select{height:30px;border-radius:6px;border:1px solid #3f6c6c;background:#102b2e;color:#e9ffff;font-weight:800;font-size:11px;padding:0 10px}
.iamccs-ms-btn.on{background:#dcb66e;color:#111;border-color:#ffe3a0;box-shadow:0 0 0 1px rgba(255,232,170,.35) inset}
.iamccs-ms-grid{display:grid;grid-template-columns:1fr 292px;grid-template-rows:174px 1fr;gap:10px;height:682px}
.iamccs-ms-timeline{grid-column:1/3;border:1px solid #263d3c;border-radius:7px;background:#071112;overflow:hidden;display:grid;grid-template-columns:128px 1fr}
.iamccs-ms-legend{border-right:1px solid #284241;background:#0c191a;color:#ffe2a6;font-size:11px;font-weight:900;padding:10px}
.iamccs-ms-lane-label{height:28px;display:flex;align-items:center;border-top:1px solid rgba(255,255,255,.05);color:#a9dadd;cursor:pointer;padding-left:2px;border-left:3px solid transparent;box-sizing:border-box}
.iamccs-ms-lane-label.on{color:#fff0bc;border-left-color:#ffd579;background:rgba(255,213,121,.08)}
.iamccs-ms-time{position:relative;overflow-x:auto;overflow-y:hidden;background:#050b0c}
.iamccs-ms-ruler{position:relative;height:28px;border-bottom:1px solid #263d3c;background:linear-gradient(90deg,rgba(255,255,255,.12) 1px,transparent 1px);background-size:48px 100%}
.iamccs-ms-shotrow{position:relative;height:42px;border-bottom:1px solid #1d3030;background:#081112}
.iamccs-ms-trackrow{position:relative;height:28px;border-bottom:1px solid #132323;background:#061011;cursor:pointer}
.iamccs-ms-trackrow.on{background:linear-gradient(90deg,rgba(255,213,121,.08),rgba(6,16,17,.98))}
.iamccs-ms-shotblock{position:absolute;top:4px;height:32px;border:1px solid #486d6a;border-radius:5px;background:#122b2d;color:#f5e6bd;font-size:10px;font-weight:800;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:8px 6px;box-sizing:border-box;cursor:pointer}
.iamccs-ms-shotblock.on{background:#4a3617;border-color:#ffd579;color:#fff5ce}
.iamccs-ms-part{position:absolute;top:5px;height:18px;border-radius:999px;border:1px solid rgba(255,255,255,.35);box-shadow:0 0 10px rgba(0,0,0,.35);cursor:pointer}
.iamccs-ms-part.on{outline:2px solid #ffe7a2}
.iamccs-ms-playhead{position:absolute;top:0;width:2px;height:154px;background:#ffe89d;box-shadow:0 0 0 1px rgba(0,0,0,.4),0 0 10px rgba(255,232,157,.75);pointer-events:none;z-index:8}
.iamccs-ms-playhead::before{content:"";position:absolute;left:-5px;top:0;border-left:6px solid transparent;border-right:6px solid transparent;border-top:8px solid #ffe89d}
.iamccs-ms-stage{position:relative;background:#020607;border:1px solid #315452;border-radius:7px;overflow:hidden;min-height:498px}
.iamccs-ms-canvas,.iamccs-ms-img{position:absolute;inset:0;width:100%;height:100%}
.iamccs-ms-img{object-fit:contain;background:radial-gradient(circle at 50% 40%,#152323,#030707 68%)}
.iamccs-ms-canvas{touch-action:none;cursor:crosshair}
.iamccs-ms-noimg{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#5f8b8d;font-weight:900;font-size:13px;pointer-events:none}
.iamccs-ms-side{border:1px solid #263d3c;border-radius:7px;background:#0b1718;padding:10px;overflow:auto;min-height:498px}
.iamccs-ms-side h4{font-size:11px;color:#f4deb0;margin:8px 0 6px}
.iamccs-ms-field{display:grid;grid-template-columns:92px 1fr;gap:6px;align-items:center;margin:7px 0;font-size:10px;color:#a8d8da}
.iamccs-ms-field input,.iamccs-ms-field select{width:100%;height:26px;border-radius:5px;border:1px solid #315452;background:#071011;color:#eff;font-weight:700}
.iamccs-ms-list{display:flex;flex-direction:column;gap:5px;margin-top:8px}
.iamccs-ms-stroke{border:1px solid #315452;background:#102124;border-radius:5px;padding:7px;font-size:10px;color:#d7ffff;text-align:left}
.iamccs-ms-stroke.on{border-color:#ffd579;background:#332914}
.iamccs-ms-hint{font-size:10px;color:#8fc9cf;line-height:1.45;margin-top:8px}
`;
    document.head.appendChild(style);
}

function newSketchData(node) {
    const data = safeParse(widget(node, "motion_sketch_data")?.value, {});
    if (!Array.isArray(data.strokes)) data.strokes = [];
    data.schema = data.schema || "iamccs.shotboard_v4.motion_sketch";
    data.schema_version = data.schema_version || 1;
    data.view = data.view && typeof data.view === "object" ? data.view : {};
    return data;
}

function durationFrames(state) {
    return Math.max(24, ...state.segments.map((s) => s.start + s.length), ...state.strokes.map((s) => (s.start_frame || 0) + (s.length_frames || 1)));
}

function segmentsSignature(segments) {
    return (segments || []).map((seg) => [
        seg.id,
        seg.start,
        seg.length,
        seg.imageFile,
        seg.label,
        seg.prompt,
    ].join(":")).join("|");
}

function alignStrokesToSegments(state) {
    const byId = new Map((state.segments || []).map((seg) => [String(seg.id), seg]));
    for (const stroke of state.strokes || []) {
        const seg = byId.get(String(stroke.segment_id || ""));
        if (!seg) continue;
        stroke.start_frame = seg.start;
        stroke.length_frames = seg.length;
    }
}

function selectedSegment(state) {
    return state.segments.find((s) => s.id === state.selectedSegmentId) || state.segments[0] || null;
}

function persist(node, state) {
    const data = newSketchData(node);
    data.frame_rate = Number(widget(node, "frame_rate")?.value || data.frame_rate || 24);
    data.control_family = String(widget(node, "control_family")?.value || data.control_family || "auto");
    data.shotboard_segments = state.segments;
    data.strokes = state.strokes;
    data.motionParts = buildMotionParts(state);
    data.view = { ...(data.view || {}), selected_segment_id: state.selectedSegmentId, selected_stroke_id: state.selectedStrokeId };
    data.duration_frames = durationFrames(state);
    writeWidget(node, "motion_sketch_data", data);
}

function strokeColor(track) {
    const map = {
        camera_path: "#4ed6ff",
        subject_path: "#78f287",
        object_path: "#ffd267",
        background_lock: "#ff6c83",
        attention_mask: "#b389ff",
    };
    return map[track] || "#4ed6ff";
}

function buildMotionParts(state) {
    return state.strokes.map((s, i) => ({
        id: s.id || `motion_part_${i + 1}`,
        type: "motion_control",
        segment_id: s.segment_id || "",
        track: s.track || "camera_path",
        mode: s.mode || "motion_track",
        scope: s.scope || "slot_only",
        start: Math.max(0, Math.round(Number(s.start_frame || 0))),
        length: Math.max(1, Math.round(Number(s.length_frames || 24))),
        trimStart: Math.max(0, Math.round(Number(s.start_frame || 0))),
        videoStrength: Number(s.strength ?? 0.75),
        videoAttentionStrength: Number(s.attention_strength ?? 0.65),
        resampleMode: "nearest",
    }));
}

function repaintCanvas(state) {
    const canvas = state.canvas;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (const stroke of state.strokes) {
        if (stroke.segment_id && stroke.segment_id !== state.selectedSegmentId) continue;
        const points = Array.isArray(stroke.points) ? stroke.points : [];
        if (!points.length) continue;
        ctx.globalAlpha = stroke.id === state.selectedStrokeId ? 1 : 0.82;
        ctx.strokeStyle = strokeColor(stroke.track);
        ctx.lineWidth = Math.max(3, Number(stroke.radius || 20) / 6);
        ctx.shadowColor = strokeColor(stroke.track);
        ctx.shadowBlur = 10;
        ctx.beginPath();
        points.forEach((p, i) => {
            const x = Number(p[0] || 0) * rect.width;
            const y = Number(p[1] || 0) * rect.height;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.shadowBlur = 0;
        const last = points[points.length - 1];
        if (last) {
            ctx.fillStyle = strokeColor(stroke.track);
            ctx.beginPath();
            ctx.arc(Number(last[0]) * rect.width, Number(last[1]) * rect.height, 5, 0, Math.PI * 2);
            ctx.fill();
        }
    }
    ctx.globalAlpha = 1;
}

function renderStrokeList(state) {
    state.list.innerHTML = "";
    const visible = state.strokes.filter((s) => !s.segment_id || s.segment_id === state.selectedSegmentId);
    for (const stroke of visible) {
        const item = document.createElement("button");
        item.className = `iamccs-ms-stroke${stroke.id === state.selectedStrokeId ? " on" : ""}`;
        item.type = "button";
        item.textContent = `${stroke.track || "path"} / ${stroke.mode || "motion"} / ${stroke.scope || "slot_only"} / ${stroke.start_frame || 0}f + ${stroke.length_frames || 0}f / ${stroke.points?.length || 0} pts`;
        item.onclick = () => {
            state.selectedStrokeId = stroke.id;
            renderStrokeList(state);
            renderTimeline(state);
            repaintCanvas(state);
        };
        state.list.appendChild(item);
    }
}

function renderLaneHighlights(state) {
    const active = state.trackSelect?.value || "camera_path";
    state.root?.querySelectorAll?.(".iamccs-ms-lane-label[data-track]")?.forEach((label) => {
        label.classList.toggle("on", label.dataset.track === active);
    });
}

function setActiveTrack(state, track) {
    if (!track || !state.trackSelect) return;
    state.trackSelect.value = track;
    renderLaneHighlights(state);
    renderTimeline(state);
    state.status.textContent = `TRACK ${track.replace(/_/g, " ")} / draw on the selected shot`;
    persist(state.node, state);
}

function frameFromPointer(state, event) {
    const innerRect = state.timeInner.getBoundingClientRect();
    const x = Math.max(0, event.clientX - innerRect.left);
    return Math.max(0, Math.round(x / Math.max(0.001, state.pxPerFrame || 1)));
}

function setPlayheadFrame(state, frame, selectShot = true) {
    const total = durationFrames(state);
    const next = Math.max(0, Math.min(total, Math.round(Number(frame) || 0)));
    state.playheadFrame = next;
    if (selectShot) {
        const seg = state.segments.find((item) => next >= item.start && next < item.start + item.length) || state.segments[state.segments.length - 1];
        if (seg && seg.id !== state.selectedSegmentId) {
            state.selectedSegmentId = seg.id;
            state.noImg.style.display = seg?.imageFile ? "none" : "flex";
            state.currentImage.src = seg?.imageFile ? imageUrl(seg.imageFile) : "";
            renderStrokeList(state);
            repaintCanvas(state);
        }
    }
    state.status.textContent = `SCRUB ${next}f / ${total}f / shot ${selectedSegment(state)?.label || "none"}`;
    renderTimeline(state);
    persist(state.node, state);
}

function setPlayheadFromEvent(state, event) {
    setPlayheadFrame(state, frameFromPointer(state, event), true);
}

function setupScrubbing(state) {
    let scrubbing = false;
    state.timeScroll.onpointerdown = (event) => {
        if (event.target?.closest?.(".iamccs-ms-shotblock,.iamccs-ms-part")) return;
        event.preventDefault();
        scrubbing = true;
        state.timeScroll.setPointerCapture?.(event.pointerId);
        setPlayheadFromEvent(state, event);
    };
    state.timeScroll.onpointermove = (event) => {
        if (!scrubbing) return;
        event.preventDefault();
        setPlayheadFromEvent(state, event);
    };
    const stop = () => {
        scrubbing = false;
    };
    state.timeScroll.onpointerup = stop;
    state.timeScroll.onpointercancel = stop;
}

function renderTimeline(state) {
    const total = durationFrames(state);
    const pxPerFrame = Math.max(3, Math.min(10, 920 / total));
    const width = Math.max(960, Math.ceil(total * pxPerFrame) + 20);
    state.pxPerFrame = pxPerFrame;
    state.timelineWidth = width;
    state.timeInner.style.width = `${width}px`;
    state.timeInner.innerHTML = "";
    const ruler = document.createElement("div");
    ruler.className = "iamccs-ms-ruler";
    ruler.style.width = `${width}px`;
    const fps = Number(widget(state.node, "frame_rate")?.value || 24);
    for (let f = 0; f <= total; f += Math.max(1, Math.round(fps))) {
        const mark = document.createElement("div");
        mark.style.position = "absolute";
        mark.style.left = `${f * pxPerFrame}px`;
        mark.style.top = "7px";
        mark.style.font = "800 10px monospace";
        mark.style.color = "#d2f1f3";
        mark.textContent = `${Math.round(f / fps)}s`;
        ruler.appendChild(mark);
    }
    state.timeInner.appendChild(ruler);

    const shotRow = document.createElement("div");
    shotRow.className = "iamccs-ms-shotrow";
    shotRow.style.width = `${width}px`;
    for (const seg of state.segments) {
        const block = document.createElement("button");
        block.type = "button";
        block.className = `iamccs-ms-shotblock${seg.id === state.selectedSegmentId ? " on" : ""}`;
        block.style.left = `${seg.start * pxPerFrame}px`;
        block.style.width = `${Math.max(24, seg.length * pxPerFrame)}px`;
        block.textContent = seg.label;
        block.title = `${seg.label} ${seg.start}f + ${seg.length}f`;
        block.onclick = () => selectSegment(state, seg.id);
        shotRow.appendChild(block);
    }
    state.timeInner.appendChild(shotRow);

    for (const track of state.trackOrder) {
        const row = document.createElement("div");
        row.className = `iamccs-ms-trackrow${state.trackSelect?.value === track ? " on" : ""}`;
        row.dataset.track = track;
        row.style.width = `${width}px`;
        row.onclick = (event) => {
            if (event.target?.closest?.(".iamccs-ms-part")) return;
            setActiveTrack(state, track);
            setPlayheadFromEvent(state, event);
        };
        for (const stroke of state.strokes.filter((s) => (s.track || "camera_path") === track)) {
            const part = document.createElement("button");
            part.type = "button";
            part.className = `iamccs-ms-part${stroke.id === state.selectedStrokeId ? " on" : ""}`;
            part.style.left = `${Math.max(0, Number(stroke.start_frame || 0)) * pxPerFrame}px`;
            part.style.width = `${Math.max(18, Number(stroke.length_frames || 1) * pxPerFrame)}px`;
            part.style.background = strokeColor(track);
            part.title = `${track} ${stroke.start_frame || 0}f + ${stroke.length_frames || 0}f`;
            part.onclick = (event) => {
                event?.stopPropagation?.();
                state.selectedStrokeId = stroke.id;
                if (stroke.segment_id) selectSegment(state, stroke.segment_id, false);
                renderTimeline(state);
                renderStrokeList(state);
                repaintCanvas(state);
            };
            row.appendChild(part);
        }
        state.timeInner.appendChild(row);
    }
    const playhead = document.createElement("div");
    playhead.className = "iamccs-ms-playhead";
    playhead.style.left = `${Math.max(0, Math.min(total, state.playheadFrame || 0)) * pxPerFrame}px`;
    state.timeInner.appendChild(playhead);
    renderLaneHighlights(state);
}

function selectSegment(state, id, persistNow = true) {
    state.selectedSegmentId = id;
    const seg = selectedSegment(state);
    if (seg) state.playheadFrame = Math.max(0, Math.round(Number(seg.start || 0)));
    state.noImg.style.display = seg?.imageFile ? "none" : "flex";
    state.currentImage.src = seg?.imageFile ? imageUrl(seg.imageFile) : "";
    state.status.textContent = `SYNC ${state.syncSource} / shot ${seg?.label || "none"} / ${seg?.start || 0}f + ${seg?.length || 0}f`;
    if (persistNow) persist(state.node, state);
    renderTimeline(state);
    renderStrokeList(state);
    repaintCanvas(state);
}

function refreshFromShotboard(state, options = {}) {
    const linked = readLinkedShotboardSegments(state.node);
    const data = newSketchData(state.node);
    const fallback = extractSegmentsFromData(data);
    state.syncSource = linked.source;
    const nextSegments = linked.segments.length ? linked.segments : (fallback.length ? fallback : state.segments);
    const nextSignature = segmentsSignature(nextSegments);
    if (options.onlyIfChanged && nextSignature === state.lastSegmentsSignature) return false;
    state.segments = nextSegments;
    state.lastSegmentsSignature = nextSignature;
    state.strokes = Array.isArray(data.strokes) ? data.strokes : [];
    alignStrokesToSegments(state);
    const saved = data.view?.selected_segment_id;
    if (saved && state.segments.some((s) => s.id === saved)) state.selectedSegmentId = saved;
    if (!state.selectedSegmentId || !state.segments.some((s) => s.id === state.selectedSegmentId)) {
        state.selectedSegmentId = state.segments[0]?.id || "";
    }
    selectSegment(state, state.selectedSegmentId || "", false);
    persist(state.node, state);
    return true;
}

function setupDrawing(state) {
    let active = null;
    state.canvas.onpointerdown = (event) => {
        event.preventDefault();
        state.canvas.setPointerCapture?.(event.pointerId);
        const seg = selectedSegment(state);
        if (!seg) return;
        state.isDrawing = true;
        const rect = state.canvas.getBoundingClientRect();
        const stroke = {
            id: `motion_part_${Date.now().toString(36)}`,
            segment_id: seg.id,
            track: state.trackSelect.value,
            mode: state.modeSelect.value,
            scope: state.scopeSelect.value,
            start_frame: seg.start,
            length_frames: seg.length,
            strength: Number(state.strength.value || 0.75),
            attention_strength: Number(state.attention.value || 0.65),
            radius: Number(state.radius.value || 28),
            falloff: Number(state.falloff.value || 0.35),
            easing: "ease_in_out",
            points: [[(event.clientX - rect.left) / rect.width, (event.clientY - rect.top) / rect.height]],
        };
        state.strokes.push(stroke);
        state.selectedStrokeId = stroke.id;
        active = stroke;
        persist(state.node, state);
        renderTimeline(state);
        renderStrokeList(state);
        repaintCanvas(state);
    };
    state.canvas.onpointermove = (event) => {
        if (!active) return;
        const rect = state.canvas.getBoundingClientRect();
        active.points.push([
            Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
            Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
        ]);
        persist(state.node, state);
        repaintCanvas(state);
    };
    const finish = () => {
        if (!active) return;
        active = null;
        state.isDrawing = false;
        persist(state.node, state);
        renderTimeline(state);
        renderStrokeList(state);
        repaintCanvas(state);
    };
    state.canvas.onpointerup = finish;
    state.canvas.onpointercancel = finish;
}

function enforceNodeSize(node) {
    node.size = [Math.max(node.size?.[0] || 0, UI_W + 30), Math.max(node.size?.[1] || 0, UI_H + 92)];
    node.min_size = [UI_W + 30, UI_H + 92];
}

function createUi(node) {
    ensureStyle();
    enforceNodeSize(node);
    const root = document.createElement("div");
    root.className = "iamccs-motion-sketch";
    root.innerHTML = `
        <div class="iamccs-ms-top">
            <div class="iamccs-ms-status">Connect Shotboard V3 cine_linx -> Motion Sketch cine_linx</div>
            <div class="iamccs-ms-actions">
                <button class="iamccs-ms-btn" data-act="refresh">Sync Shotboard</button>
                <button class="iamccs-ms-btn" data-act="clear">Clear Shot</button>
                <button class="iamccs-ms-btn on" data-act="publish">Publish</button>
            </div>
        </div>
        <div class="iamccs-ms-grid">
            <div class="iamccs-ms-timeline">
                <div class="iamccs-ms-legend">
                    <div style="height:28px">TIME</div>
                    <div class="iamccs-ms-lane-label">SHOT</div>
                    <div class="iamccs-ms-lane-label" data-track="camera_path">CAMERA</div>
                    <div class="iamccs-ms-lane-label" data-track="subject_path">SUBJECT</div>
                    <div class="iamccs-ms-lane-label" data-track="object_path">OBJECT</div>
                    <div class="iamccs-ms-lane-label" data-track="background_lock">LOCK</div>
                </div>
                <div class="iamccs-ms-time"><div class="iamccs-ms-time-inner"></div></div>
            </div>
            <div class="iamccs-ms-stage">
                <img class="iamccs-ms-img" alt="">
                <canvas class="iamccs-ms-canvas"></canvas>
                <div class="iamccs-ms-noimg">NO SHOT IMAGE</div>
            </div>
            <div class="iamccs-ms-side">
                <h4>MOTION TRACK</h4>
                <label class="iamccs-ms-field"><span>track</span><select data-field="track"><option value="camera_path">camera path</option><option value="subject_path">subject path</option><option value="object_path">object path</option><option value="background_lock">background lock</option><option value="attention_mask">attention mask</option></select></label>
                <label class="iamccs-ms-field"><span>mode</span><select data-field="mode"><option value="motion_track">motion track</option><option value="orbit">orbit</option><option value="dolly">dolly</option><option value="pan">pan</option><option value="lock">lock</option></select></label>
                <label class="iamccs-ms-field"><span>scope</span><select data-field="scope"><option value="slot_only">slot only</option><option value="continue_to_next">continue to next</option><option value="hold_last">hold last</option><option value="cut_reset">cut reset</option></select></label>
                <label class="iamccs-ms-field"><span>strength</span><input data-field="strength" type="number" min="0" max="1" step="0.01" value="0.75"></label>
                <label class="iamccs-ms-field"><span>attention</span><input data-field="attention" type="number" min="0" max="1" step="0.01" value="0.65"></label>
                <label class="iamccs-ms-field"><span>brush</span><input data-field="radius" type="number" min="1" step="1" value="28"></label>
                <label class="iamccs-ms-field"><span>falloff</span><input data-field="falloff" type="number" min="0" max="1" step="0.01" value="0.35"></label>
                <h4>MOTION PARTS</h4>
                <div class="iamccs-ms-list"></div>
                <div class="iamccs-ms-hint">Disegna sull'immagine selezionata. Ogni tratto diventa una motionPart sincronizzata allo start/length dello shot nella timeline.</div>
            </div>
        </div>`;

    const state = {
        node,
        root,
        status: root.querySelector(".iamccs-ms-status"),
        timeScroll: root.querySelector(".iamccs-ms-time"),
        timeInner: root.querySelector(".iamccs-ms-time-inner"),
        canvas: root.querySelector(".iamccs-ms-canvas"),
        currentImage: root.querySelector(".iamccs-ms-img"),
        noImg: root.querySelector(".iamccs-ms-noimg"),
        list: root.querySelector(".iamccs-ms-list"),
        trackSelect: root.querySelector("[data-field='track']"),
        modeSelect: root.querySelector("[data-field='mode']"),
        scopeSelect: root.querySelector("[data-field='scope']"),
        strength: root.querySelector("[data-field='strength']"),
        attention: root.querySelector("[data-field='attention']"),
        radius: root.querySelector("[data-field='radius']"),
        falloff: root.querySelector("[data-field='falloff']"),
        trackOrder: ["camera_path", "subject_path", "object_path", "background_lock"],
        segments: [],
        strokes: [],
        selectedSegmentId: "",
        selectedStrokeId: "",
        playheadFrame: 0,
        lastSegmentsSignature: "",
        autoSyncTimer: null,
        syncSource: "not connected",
    };
    node[STATE_KEY] = state;

    root.querySelector("[data-act='refresh']").onclick = () => refreshFromShotboard(state);
    root.querySelector("[data-act='clear']").onclick = () => {
        state.strokes = state.strokes.filter((s) => s.segment_id !== state.selectedSegmentId);
        state.selectedStrokeId = "";
        persist(node, state);
        renderTimeline(state);
        renderStrokeList(state);
        repaintCanvas(state);
    };
    root.querySelector("[data-act='publish']").onclick = (event) => {
        event.currentTarget.classList.toggle("on");
        persist(node, state);
    };
    root.querySelectorAll(".iamccs-ms-lane-label[data-track]").forEach((label) => {
        label.onclick = () => setActiveTrack(state, label.dataset.track);
    });
    state.trackSelect.onchange = () => setActiveTrack(state, state.trackSelect.value);
    for (const input of [state.strength, state.attention, state.radius, state.falloff, state.modeSelect, state.scopeSelect]) {
        input.onchange = () => persist(node, state);
    }
    setupDrawing(state);
    setupScrubbing(state);
    state.currentImage.onload = () => repaintCanvas(state);
    new ResizeObserver(() => {
        enforceNodeSize(node);
        repaintCanvas(state);
    }).observe(root);

    const domWidget = node.addDOMWidget("CineMotionSketch", "iamccs_cine_motion_sketch", root, { serialize: false });
    domWidget.computeSize = () => [UI_W, UI_H];
    const originalOnResize = node.onResize;
    node.onResize = function (...args) {
        const out = originalOnResize?.apply(this, args);
        enforceNodeSize(node);
        setTimeout(() => repaintCanvas(state), 0);
        return out;
    };
    setTimeout(() => refreshFromShotboard(state), 80);
    state.autoSyncTimer = setInterval(() => {
        if (!document.body.contains(root) || node[STATE_KEY] !== state) {
            clearInterval(state.autoSyncTimer);
            return;
        }
        if (state.isDrawing) return;
        refreshFromShotboard(state, { onlyIfChanged: true });
    }, 700);
}

function setupNode(node) {
    if (!node || (node.comfyClass || node.type) !== TARGET_CLASS || node[STATE_KEY]) return;
    if (typeof node.addDOMWidget !== "function") {
        setTimeout(() => setupNode(node), 150);
        return;
    }
    const jsonWidget = widget(node, "motion_sketch_data");
    if (jsonWidget) {
        jsonWidget.type = "hidden";
        jsonWidget.computeSize = () => [0, -4];
    }
    createUi(node);
}

app.registerExtension({
    name: "iamccs.cine.motion.sketch.v4",
    nodeCreated(node) {
        setupNode(node);
    },
    loadedGraphNode(node) {
        setupNode(node);
    },
});
