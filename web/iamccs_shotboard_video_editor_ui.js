import { app } from "../../scripts/app.js";

const STYLE_ID = "iamccs-shotboard-video-editor-style-v4";
const FIXED_SIZE = [1380, 1100];
const NODE_BODY_HEIGHT = 1048;

function nodeType(node) {
    return String(node?.type || node?.comfyClass || node?.constructor?.type || "");
}
function isEditorNode(node) {
    return nodeType(node) === "IAMCCS_ShotboardVideoEditor";
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
    if (item.inputEl) item.inputEl.style.display = "none";
}
function hideRawWidgets(node) {
    [
        "editor_mode", "selected_take", "take_order", "audio_policy", "fps_mode", "override_fps",
        "global_trim_in_seconds", "global_trim_out_seconds", "concat_plan_json", "clip_edits_json",
        "editor_manifest_json",
    ].forEach((name) => hideWidget(widget(node, name)));
}
function clamp(value, min, max) {
    return Math.max(min, Math.min(max, Number(value) || 0));
}
function parseJson(value, fallback) {
    try {
        const out = JSON.parse(String(value || ""));
        return out && typeof out === "object" ? out : fallback;
    } catch {
        return fallback;
    }
}
function parseOrder(text, maxTake = 8) {
    const nums = String(text || "1,2")
        .split(/[,;\s]+/)
        .map((v) => Math.round(Number(v)))
        .filter((v) => v >= 1 && v <= maxTake);
    return nums.length ? nums : [1, 2].filter((v) => v <= maxTake);
}
function formatTime(seconds, fps = 24) {
    const safe = Math.max(0, Number(seconds) || 0);
    const frame = Math.round(safe * fps);
    const s = Math.floor(safe % 60).toString().padStart(2, "0");
    const m = Math.floor(safe / 60).toString().padStart(2, "0");
    return `${m}:${s}.${String(frame % Math.max(1, Math.round(fps))).padStart(2, "0")}`;
}
function formatBigClock(seconds, fps = 24) {
    const safe = Math.max(0, Number(seconds) || 0);
    const frame = Math.round(safe * fps);
    const hh = Math.floor(safe / 3600).toString().padStart(2, "0");
    const mm = Math.floor((safe % 3600) / 60).toString().padStart(2, "0");
    const ss = Math.floor(safe % 60).toString().padStart(2, "0");
    const ff = String(frame % Math.max(1, Math.round(fps))).padStart(2, "0");
    return `${hh}:${mm}:${ss}:${ff}`;
}
function flash(el, text) {
    if (!el) return;
    el.textContent = text;
    el.classList.add("pulse");
    window.setTimeout(() => el.classList.remove("pulse"), 220);
}
function readClipEdits(node) {
    return parseJson(widget(node, "clip_edits_json")?.value, {});
}
function writeClipEdits(node, patch) {
    const current = readClipEdits(node);
    const next = { ...(current || {}), ...patch, updated_at: Date.now() };
    setWidget(node, "clip_edits_json", JSON.stringify(next, null, 2));
}
function manifestData(node) {
    const direct = parseJson(widget(node, "editor_manifest_json")?.value, {});
    const plan = parseJson(widget(node, "concat_plan_json")?.value, {});
    const videos = Array.isArray(direct.video_manifest) ? direct.video_manifest : [];
    const audios = Array.isArray(direct.audio_manifest) ? direct.audio_manifest : [];
    const takes = Array.isArray(plan.takes) ? plan.takes : [];
    const maxTake = Math.max(1, videos.length, audios.length, takes.length, 8);
    const takeDurations = {};
    takes.forEach((take, idx) => {
        const takeIndex = Math.max(1, Math.round(Number(take.take_index || idx + 1)));
        const seconds = Number(take.duration_seconds || take.duration || 0);
        if (seconds > 0) takeDurations[takeIndex] = seconds;
    });
    videos.forEach((item, idx) => {
        const takeIndex = Math.max(1, Math.round(Number(item.timeline_id || item.slot || idx + 1).toString().replace(/\D/g, "") || idx + 1));
        const seconds = Number(item.duration_seconds || item.duration || 0);
        if (seconds > 0) takeDurations[takeIndex] = seconds;
    });
    audios.forEach((item, idx) => {
        const takeIndex = Math.max(1, Math.round(Number(item.audio_lane || item.slot || idx + 1).toString().replace(/\D/g, "") || idx + 1));
        const seconds = Number(item.duration_seconds || item.duration || 0);
        if (seconds > 0 && !takeDurations[takeIndex]) takeDurations[takeIndex] = seconds;
    });
    return { direct, plan, videos, audios, takes, maxTake, takeDurations };
}
function ensureStyle() {
    document.getElementById(STYLE_ID)?.remove();
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
        .iamccs-sve { box-sizing:border-box; width:100%; height:${NODE_BODY_HEIGHT}px; padding:12px; border:1px solid rgba(244,212,158,.42); border-radius:10px; background:#071012; color:#ebffff; font:12px Inter,Arial,sans-serif; overflow:hidden; pointer-events:auto; box-shadow:inset 0 0 0 1px rgba(255,255,255,.04); }
        .iamccs-sve * { box-sizing:border-box; }
        .iamccs-sve button { min-height:34px; border-radius:7px; border:1px solid rgba(126,198,196,.58); background:linear-gradient(180deg,#27545a,#153338); color:#efffff; cursor:pointer; font-weight:950; font-size:11px; padding:0 13px; white-space:nowrap; box-shadow:inset 0 1px 0 rgba(255,255,255,.10); }
        .iamccs-sve button:hover { border-color:#b5f4ef; filter:brightness(1.08); }
        .iamccs-sve button.active,.iamccs-sve button.primary { background:linear-gradient(180deg,#f8dda4,#c99950); color:#171207; border-color:#ffe3a4; }
        .iamccs-sve input,.iamccs-sve select { min-width:0; height:34px; border-radius:7px; border:1px solid rgba(94,161,161,.72); background:#061012; color:#efffff; font-size:11px; font-weight:850; padding:0 9px; }
        .iamccs-sve.open { position:fixed; z-index:99999; inset:18px; width:auto; height:auto; padding:16px; border-radius:12px; box-shadow:0 24px 90px rgba(0,0,0,.72), inset 0 0 0 1px rgba(255,255,255,.06); }
        .iamccs-sve-shell { height:100%; display:grid; grid-template-rows:104px 54px minmax(310px, 36vh) 1fr 38px 38px; gap:10px; min-height:0; }
        .iamccs-sve:not(.open) .iamccs-sve-shell { grid-template-rows:104px 54px 330px 430px 38px 38px; }
        .iamccs-sve-top { display:grid; grid-template-columns:210px 1fr 470px 245px; gap:10px; min-height:0; }
        .iamccs-sve-brand,.iamccs-sve-transport,.iamccs-sve-clock-panel { border:1px solid rgba(255,255,255,.10); border-radius:9px; background:linear-gradient(180deg,#13272a,#081112); overflow:hidden; }
        .iamccs-sve-brand { padding:12px 13px; }
        .iamccs-sve-title { color:#fff0b8; font-size:15px; font-weight:950; }
        .iamccs-sve-sub { color:#8fb4b3; font-size:10px; font-weight:850; margin-top:4px; line-height:1.35; }
        .iamccs-sve-transport { padding:9px; display:grid; grid-template-columns:1fr; grid-template-rows:36px 36px; gap:7px; }
        .iamccs-sve-transport-row,.iamccs-sve-edit-row { display:flex; align-items:center; justify-content:center; gap:7px; min-width:0; }
        .iamccs-sve-clock-panel { display:grid; grid-template-rows:26px 1fr; padding:8px; background:linear-gradient(180deg,#142322,#0b1010); }
        .iamccs-sve-clock-label { color:#fff0b8; font-size:10px; font-weight:950; letter-spacing:.02em; }
        .iamccs-sve-clock { display:flex; align-items:center; justify-content:center; border-radius:8px; background:#f3ffe9; border:2px solid #a7d59b; color:#0a7d23; font:25px Consolas,monospace; text-align:center; font-weight:950; box-shadow:inset 0 0 12px rgba(0,0,0,.16); }
        .iamccs-sve-sourcebar { display:grid; grid-template-columns:80px 1fr 220px; align-items:center; gap:8px; padding:8px 10px; border:1px solid rgba(255,255,255,.10); border-radius:9px; background:#081112; min-height:0; }
        .iamccs-sve-sourcebar strong { color:#fff0b8; font-size:11px; }
        .iamccs-sve-takes { display:flex; gap:6px; overflow:auto hidden; padding-bottom:2px; }
        .iamccs-sve-take { min-width:62px; height:32px; padding:0 8px; font-size:10px; }
        .iamccs-sve-imports { display:flex; gap:7px; justify-content:flex-end; }
        .iamccs-sve-monitor-row { display:grid; grid-template-columns:1fr 1fr; gap:10px; min-height:0; }
        .iamccs-sve.one-monitor .iamccs-sve-monitor-row { grid-template-columns:1fr; }
        .iamccs-sve.one-monitor .iamccs-sve-monitor.source { display:none; }
        .iamccs-sve-monitor { border:1px solid rgba(255,255,255,.12); border-radius:9px; background:linear-gradient(180deg,#0d1718,#030607); overflow:hidden; min-height:0; display:grid; grid-template-rows:28px 1fr; }
        .iamccs-sve-monitor-head { display:flex; align-items:center; justify-content:space-between; padding:0 10px; background:linear-gradient(180deg,#172d30,#0d181a); border-bottom:1px solid rgba(255,255,255,.09); color:#ffe5a8; font-weight:950; font-size:10px; text-transform:uppercase; }
        .iamccs-sve-screen { margin:10px; border-radius:8px; border:1px solid rgba(126,198,196,.32); background:radial-gradient(circle at 52% 45%,rgba(99,160,170,.22),transparent 34%),linear-gradient(135deg,#111b20,#06090b 52%,#141008); position:relative; overflow:hidden; min-height:230px; }
        .iamccs-sve.open .iamccs-sve-screen { min-height:320px; }
        .iamccs-sve-safe { position:absolute; inset:24px 36px; border:1px dashed rgba(255,255,255,.20); border-radius:5px; }
        .iamccs-sve-screen-label { position:absolute; left:12px; bottom:10px; color:#eaffff; background:rgba(0,0,0,.56); border:1px solid rgba(255,255,255,.14); border-radius:5px; padding:6px 8px; font:11px Consolas,monospace; }
        .iamccs-sve-controls { display:grid; grid-template-columns:1.15fr 74px 1fr 1.1fr 76px; gap:7px; border:1px solid rgba(255,255,255,.10); border-radius:9px; padding:8px; background:linear-gradient(180deg,#102224,#071011); min-height:0; overflow:hidden; }
        .iamccs-sve-controls label { display:grid; gap:3px; min-width:0; color:#9db7b8; font-size:8px; text-transform:uppercase; font-weight:950; }
        .iamccs-sve-timeline { border:1px solid rgba(255,255,255,.12); border-radius:10px; overflow:hidden; background:#05090a; min-height:0; display:grid; grid-template-rows:42px 1fr; }
        .iamccs-sve-ruler { margin-left:128px; position:relative; border-bottom:1px solid rgba(255,255,255,.12); background:linear-gradient(180deg,#132427,#091113); cursor:ew-resize; user-select:none; }
        .iamccs-sve-ruler-tick { position:absolute; top:0; bottom:0; width:1px; background:rgba(255,255,255,.15); }
        .iamccs-sve-ruler-tick span { position:absolute; top:6px; left:5px; color:#b7d0cf; font-size:10px; font-weight:900; line-height:1.1; }
        .iamccs-sve-tracks { overflow:auto; position:relative; min-height:0; }
        .iamccs-sve-playhead { position:absolute; top:0; bottom:0; width:2px; background:#ffe08d; z-index:20; box-shadow:0 0 10px rgba(255,224,141,.75); pointer-events:none; }
        .iamccs-sve-track { min-height:82px; display:grid; grid-template-columns:128px 1fr; border-bottom:1px solid rgba(255,255,255,.14); }
        .iamccs-sve-track.audio { min-height:76px; }
        .iamccs-sve-label { padding:9px 8px; border-right:1px solid rgba(255,255,255,.11); background:linear-gradient(90deg,rgba(36,68,65,.58),rgba(9,17,18,.94)); }
        .iamccs-sve-label strong { color:#fff0b8; font-size:13px; display:block; }
        .iamccs-sve-lanehint { color:#8fb4b3; font-size:9px; margin-top:2px; font-weight:850; }
        .iamccs-sve-chiprow { display:flex; gap:5px; margin-top:7px; flex-wrap:wrap; }
        .iamccs-sve-chip { height:22px; min-width:25px; border-radius:5px; border:1px solid rgba(126,198,196,.50); color:#dfffff; background:#102f34; font-size:9px; font-weight:950; display:flex; align-items:center; justify-content:center; cursor:pointer; }
        .iamccs-sve-chip.active { background:#f3d08d; color:#171207; border-color:#ffe3a4; }
        .iamccs-sve-lane { position:relative; min-width:1120px; border-left:1px solid rgba(255,224,141,.18); background:repeating-linear-gradient(90deg,rgba(255,255,255,.07) 0 1px,transparent 1px 54px),linear-gradient(180deg,rgba(24,31,29,.50),rgba(8,10,11,.90)); }
        .iamccs-sve-track:nth-child(even) .iamccs-sve-lane { background:repeating-linear-gradient(90deg,rgba(255,255,255,.06) 0 1px,transparent 1px 54px),linear-gradient(180deg,rgba(30,22,28,.48),rgba(8,10,11,.92)); }
        .iamccs-sve-clip { position:absolute; top:12px; height:56px; border-radius:6px; border:1px solid rgba(255,224,160,.70); background:linear-gradient(180deg,#31677a,#16343f); box-shadow:inset 0 0 0 1px rgba(255,255,255,.08); overflow:hidden; cursor:grab; touch-action:none; }
        .iamccs-sve-clip.audio { height:50px; background:linear-gradient(180deg,#386d9b,#173b63); border-color:#b9dcff; }
        .iamccs-sve-clip.dragging { cursor:grabbing; box-shadow:0 0 0 2px rgba(255,224,141,.62),0 8px 20px rgba(0,0,0,.35); z-index:10; }
        .iamccs-sve-clip-title { position:absolute; left:13px; top:6px; z-index:4; color:#fff1ba; font-weight:950; font-size:11px; text-shadow:0 1px 2px #000; pointer-events:none; }
        .iamccs-sve-handle { position:absolute; top:0; bottom:0; width:10px; background:#ffe08d; z-index:5; cursor:ew-resize; }
        .iamccs-sve-handle.left { left:0; }
        .iamccs-sve-handle.right { right:0; }
        .iamccs-sve-wave { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }
        .iamccs-sve-clip:not(.audio)::before { content:""; position:absolute; inset:0; background:repeating-linear-gradient(90deg, rgba(255,255,255,.13) 0 2px, transparent 2px 9px, rgba(0,0,0,.24) 9px 38px), linear-gradient(90deg, rgba(93,151,164,.34), rgba(23,54,64,.78), rgba(190,145,78,.28)); opacity:.95; }
        .iamccs-sve-clip:not(.audio)::after { content:""; position:absolute; left:0; right:0; top:0; height:8px; background:repeating-linear-gradient(90deg,#071012 0 6px,#e8d8a3 6px 9px); opacity:.82; }
        .iamccs-sve-actions { display:flex; gap:8px; align-items:center; justify-content:flex-end; min-height:0; }
        .iamccs-sve-ledger { padding:8px 10px; border:1px solid rgba(255,255,255,.08); border-radius:7px; background:#030708; color:#b8fff1; font:11px Consolas,monospace; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        .iamccs-sve-ledger.pulse { color:#171207; background:#f3d08d; }
    `;
    document.head.appendChild(style);
}
function drawWave(canvas, seed = 1) {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    const mid = h * 0.58;
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, "rgba(240,250,255,.98)");
    grad.addColorStop(1, "rgba(137,205,236,.86)");
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.moveTo(0, mid);
    for (let x = 0; x <= w; x += 2) {
        const t = (x / Math.max(1, w)) * 28 + seed;
        const amp = (Math.sin(t * 1.3) * .20 + Math.sin(t * 4.1) * .18 + Math.sin(t * 13.7) * .09 + .46);
        const env = .22 + .78 * Math.abs(Math.sin(t * .27 + seed));
        ctx.lineTo(x, mid - amp * env * h * .38);
    }
    for (let x = w; x >= 0; x -= 2) {
        const t = (x / Math.max(1, w)) * 28 + seed;
        const amp = (Math.sin(t * 1.3) * .20 + Math.sin(t * 4.1) * .18 + Math.sin(t * 13.7) * .09 + .46);
        const env = .22 + .78 * Math.abs(Math.sin(t * .27 + seed));
        ctx.lineTo(x, mid + amp * env * h * .32);
    }
    ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = "rgba(255,238,181,.74)";
    ctx.lineWidth = Math.max(1, dpr);
    ctx.beginPath();
    ctx.moveTo(0, mid);
    ctx.lineTo(w, mid);
    ctx.stroke();
}
function installFixedNode(node) {
    node.size = [...FIXED_SIZE];
    node.resizable = false;
    const originalOnResize = node.onResize;
    node.onResize = function () {
        this.size = [...FIXED_SIZE];
        try { originalOnResize?.apply(this, arguments); } catch {}
    };
}
function installEditorUI(node, reason = "install") {
    if (!isEditorNode(node) || node._iamccsShotboardVideoEditorReady || typeof node.addDOMWidget !== "function") return;
    node._iamccsShotboardVideoEditorReady = true;
    ensureStyle();
    hideRawWidgets(node);
    installFixedNode(node);
    const root = document.createElement("div");
    root.className = "iamccs-sve";
    const overlaySlot = document.createComment("iamccs-shotboard-video-editor-overlay-slot");
    function syncOverlayMount() {
        if (state.open) {
            if (!root._iamccsHomeParent && root.parentElement && root.parentElement !== document.body) {
                root._iamccsHomeParent = root.parentElement;
                root._iamccsHomeParent.insertBefore(overlaySlot, root);
            }
            if (root.parentElement !== document.body) document.body.appendChild(root);
        } else if (root._iamccsHomeParent && overlaySlot.parentNode) {
            root._iamccsHomeParent.insertBefore(root, overlaySlot);
            overlaySlot.remove();
            root._iamccsHomeParent = null;
        }
    }
    const state = {
        playhead: 0,
        duration: 24,
        oneMonitor: false,
        open: false,
        clipMap: {},
        buttons: {},
        manualVideo: 0,
        manualAudio: 0,
    };
    const clipKey = (type, take, pos) => `${type}_${take}_${pos}`;
    function getData() {
        return manifestData(node);
    }
    function takeDuration(take, fallback = 8) {
        const data = getData();
        return Math.max(.25, Number(data.takeDurations?.[take] || fallback));
    }
    function initClipState(order) {
        let cursor = 0;
        order.forEach((take, pos) => {
            const dur = takeDuration(take, 8);
            ["v", "a"].forEach((type) => {
                const key = clipKey(type, take, pos);
                if (!state.clipMap[key]) state.clipMap[key] = { start: cursor, length: dur };
            });
            cursor += dur;
        });
        Object.entries(state.clipMap).forEach(([key, value]) => {
            if (!Number.isFinite(Number(value.start))) value.start = 0;
            if (!Number.isFinite(Number(value.length)) || Number(value.length) <= 0) value.length = 4;
        });
    }
    function writeEditorState() {
        writeClipEdits(node, {
            ui_state: {
                playhead: state.playhead,
                oneMonitor: state.oneMonitor,
                clips: state.clipMap,
                manualVideo: state.manualVideo,
                manualAudio: state.manualAudio,
            },
        });
    }
    function button(label, cb, className = "") {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = label;
        b.className = className;
        b.onclick = (event) => {
            event.preventDefault();
            event.stopPropagation();
            cb?.(b);
        };
        return b;
    }
    function render() {
        hideRawWidgets(node);
        installFixedNode(node);
        root.classList.toggle("one-monitor", Boolean(state.oneMonitor));
        root.classList.toggle("open", Boolean(state.open));
        const data = getData();
        const mode = String(widget(node, "editor_mode")?.value || "assemble_sequence");
        const selected = clamp(Math.round(Number(widget(node, "selected_take")?.value || 1)), 1, data.maxTake || 8);
        const orderText = String(widget(node, "take_order")?.value || "1,2");
        const order = parseOrder(orderText, data.maxTake || 8);
        const fps = Number(widget(node, "override_fps")?.value || data.plan?.frame_rate || 24);
        initClipState(order);
        state.duration = Math.max(6, ...Object.values(state.clipMap).map((clip) => Number(clip.start || 0) + Number(clip.length || 0)));
        state.playhead = clamp(state.playhead, 0, state.duration);
        root.innerHTML = "";
        const shell = document.createElement("div");
        shell.className = "iamccs-sve-shell";

        const top = document.createElement("div");
        top.className = "iamccs-sve-top";
        const brand = document.createElement("div");
        brand.className = "iamccs-sve-brand";
        brand.innerHTML = `<div class="iamccs-sve-title">Shotboard Video Editor</div><div class="iamccs-sve-sub">CineInfo3 owns video/audio inputs. This editor assembles rendered T/A takes.</div>`;
        const ledger = document.createElement("div");
        ledger.className = "iamccs-sve-ledger";
        const bigClock = document.createElement("div");
        bigClock.className = "iamccs-sve-clock";
        const clockPanel = document.createElement("div");
        clockPanel.className = "iamccs-sve-clock-panel";
        clockPanel.innerHTML = `<div class="iamccs-sve-clock-label">PROGRAM TIME / DURATION</div>`;
        clockPanel.appendChild(bigClock);
        const updateClock = () => {
            bigClock.textContent = `${formatBigClock(state.playhead, fps)} / ${formatBigClock(state.duration, fps)}`;
        };
        const transport = document.createElement("div");
        transport.className = "iamccs-sve-transport";
        const transportRow = document.createElement("div");
        transportRow.className = "iamccs-sve-transport-row";
        const editRow = document.createElement("div");
        editRow.className = "iamccs-sve-edit-row";
        let playTimer = 0;
        const stopPlayback = () => {
            if (playTimer) window.clearInterval(playTimer);
            playTimer = 0;
        };
        [
            ["|<", () => { state.playhead = 0; updatePlayhead(); flash(ledger, "Go to start"); }],
            ["<<", () => { state.playhead = Math.max(0, state.playhead - .5); updatePlayhead(); flash(ledger, "Step back"); }],
            ["Play", (btn) => {
                const active = !btn.classList.contains("active");
                btn.classList.toggle("active", active);
                if (!active) {
                    stopPlayback();
                    flash(ledger, "Playback stopped");
                    return;
                }
                flash(ledger, "Playback running");
                stopPlayback();
                playTimer = window.setInterval(() => {
                    state.playhead += 1 / Math.max(1, fps);
                    if (state.playhead >= state.duration) {
                        state.playhead = state.duration;
                        btn.classList.remove("active");
                        stopPlayback();
                    }
                    updatePlayhead();
                }, 1000 / Math.min(60, Math.max(1, fps)));
            }, "primary"],
            [">>", () => { state.playhead = Math.min(state.duration, state.playhead + .5); updatePlayhead(); flash(ledger, "Step forward"); }],
            [">|", () => { state.playhead = state.duration; updatePlayhead(); flash(ledger, "Go to end"); }],
        ].forEach(([label, cb, cls]) => transportRow.appendChild(button(label, cb, cls || "")));
        [["Cut", "cut"], ["Trim", "trim"], ["Razor", "razor"], ["Snap", "snap"], ["Ripple", "ripple"], ["1 Monitor", "oneMonitor"], [state.open ? "Close Editor" : "Open Editor", "open"]].forEach(([label, key]) => {
            const b = button(label, () => {
                if (key === "oneMonitor") state.oneMonitor = !state.oneMonitor;
                else if (key === "open") state.open = !state.open;
                else state.buttons[key] = !state.buttons[key];
                writeEditorState();
                flash(ledger, `${label}: ${(state.buttons[key] || (key === "oneMonitor" && state.oneMonitor) || (key === "open" && state.open)) ? "ON" : "OFF"}`);
                render();
            }, (state.buttons[key] || (key === "oneMonitor" && state.oneMonitor) || (key === "open" && state.open)) ? "active" : "");
            editRow.appendChild(b);
        });
        transport.append(transportRow, editRow);
        const sourcebar = document.createElement("div");
        sourcebar.className = "iamccs-sve-sourcebar";
        sourcebar.innerHTML = `<strong>T/A TAKES</strong>`;
        const takeRow = document.createElement("div");
        takeRow.className = "iamccs-sve-takes";
        const maxTakes = Math.max(2, Math.min(12, data.maxTake || 8));
        for (let take = 1; take <= maxTakes; take += 1) {
            const b = button(`T${take}/A${take}`, () => {
                setWidget(node, "selected_take", take);
                setWidget(node, "editor_mode", "preview_selected_take");
                try {
                    window.dispatchEvent(new CustomEvent("iamccs:multigeneration-active-take", {
                        detail: { activeTake: take, timelineId: `T${String(take).padStart(2, "0")}`, audioLane: `A${take}`, source: "video_editor" },
                    }));
                } catch {}
                flash(ledger, `Selected T${take}/A${take} and requested Shotboard sync`);
                render();
            }, take === selected ? "active" : "");
            b.className = `iamccs-sve-take ${b.className}`;
            takeRow.appendChild(b);
        }
        const imports = document.createElement("div");
        imports.className = "iamccs-sve-imports";
        imports.append(
            button("Add Video", () => {
                state.manualVideo += 1;
                const key = `manual_video_${state.manualVideo}`;
                state.clipMap[key] = { start: state.playhead, length: Math.min(5, Math.max(1, state.duration - state.playhead || 5)), manual: true, type: "video" };
                writeEditorState();
                flash(ledger, "Added standalone video placeholder. Connect real media through CineInfo3 for execution.");
                render();
            }, "primary"),
            button("Add Audio", () => {
                state.manualAudio += 1;
                const key = `manual_audio_${state.manualAudio}`;
                state.clipMap[key] = { start: state.playhead, length: Math.min(5, Math.max(1, state.duration - state.playhead || 5)), manual: true, type: "audio" };
                writeEditorState();
                flash(ledger, "Added standalone audio placeholder. Connect real media through CineInfo3/master_audio for execution.");
                render();
            }, "primary"),
            button(state.oneMonitor ? "Two Monitors" : "One Monitor", () => { state.oneMonitor = !state.oneMonitor; render(); })
        );
        sourcebar.append(takeRow, imports);

        const monitors = document.createElement("div");
        monitors.className = "iamccs-sve-monitor-row";
        const makeMonitor = (label, right, cls = "") => {
            const m = document.createElement("div");
            m.className = `iamccs-sve-monitor ${cls}`;
            m.innerHTML = `<div class="iamccs-sve-monitor-head"><span>${label}</span><span>${right}</span></div><div class="iamccs-sve-screen"><div class="iamccs-sve-safe"></div><div class="iamccs-sve-screen-label" data-monitor-label>${label}: ${mode === "preview_selected_take" ? `T${selected}/A${selected}` : order.join(" -> ")}</div></div>`;
            return m;
        };
        monitors.append(makeMonitor("Source Monitor", `T${selected}/A${selected}`, "source"), makeMonitor("Program Monitor", formatTime(state.playhead, fps), "program"));

        const controls = document.createElement("div");
        controls.className = "iamccs-sve-controls";
        const labelWrap = (label, child) => { const w = document.createElement("label"); w.textContent = label; w.appendChild(child); return w; };
        const modeSelect = document.createElement("select");
        [["assemble_sequence", "Assemble sequence"], ["preview_selected_take", "Manual selected take"]].forEach(([v, l]) => { const o = document.createElement("option"); o.value = v; o.textContent = l; modeSelect.appendChild(o); });
        modeSelect.value = mode;
        modeSelect.onchange = () => { setWidget(node, "editor_mode", modeSelect.value); render(); };
        const selectedInput = document.createElement("input");
        selectedInput.type = "number"; selectedInput.min = "1"; selectedInput.max = String(maxTakes); selectedInput.value = String(selected);
        selectedInput.onchange = () => { setWidget(node, "selected_take", clamp(selectedInput.value, 1, maxTakes)); render(); };
        const orderInput = document.createElement("input");
        orderInput.value = orderText;
        orderInput.onchange = () => { setWidget(node, "take_order", orderInput.value || "1,2"); render(); };
        const audioSelect = document.createElement("select");
        [["concat_clip_audio", "Concat clip audio"], ["use_master_audio", "Use master audio"], ["first_selected_audio", "Selected clip audio"], ["mix_editor_audio_tracks", "Mix CineInfo3 audio"], ["concat_editor_audio_tracks", "Concat CineInfo3 audio"], ["silent", "Silent"]].forEach(([v, l]) => { const o = document.createElement("option"); o.value = v; o.textContent = l; audioSelect.appendChild(o); });
        audioSelect.value = String(widget(node, "audio_policy")?.value || "concat_clip_audio");
        audioSelect.onchange = () => { setWidget(node, "audio_policy", audioSelect.value); flash(ledger, `Audio policy: ${audioSelect.value}`); };
        const fpsInput = document.createElement("input");
        fpsInput.type = "number"; fpsInput.step = "0.01"; fpsInput.value = String(fps);
        fpsInput.onchange = () => setWidget(node, "override_fps", Math.max(1, Number(fpsInput.value || 24)));
        controls.append(labelWrap("Mode", modeSelect), labelWrap("T/A", selectedInput), labelWrap("Order", orderInput), labelWrap("Audio", audioSelect), labelWrap("FPS", fpsInput));
        top.append(brand, transport, controls, clockPanel);

        const timeline = document.createElement("div");
        timeline.className = "iamccs-sve-timeline";
        const ruler = document.createElement("div");
        ruler.className = "iamccs-sve-ruler";
        const tickStep = state.duration > 60 ? 5 : state.duration > 30 ? 2 : 1;
        for (let sec = 0; sec <= state.duration + .001; sec += tickStep) {
            const tick = document.createElement("div");
            tick.className = "iamccs-sve-ruler-tick";
            tick.style.left = `${(sec / state.duration) * 100}%`;
            tick.innerHTML = `<span>${Math.round(sec)}s<br>${Math.round(sec * fps)}f</span>`;
            ruler.appendChild(tick);
        }
        const tracks = document.createElement("div");
        tracks.className = "iamccs-sve-tracks";
        const playhead = document.createElement("div");
        playhead.className = "iamccs-sve-playhead";
        const updatePlayhead = () => {
            const width = tracks.querySelector(".iamccs-sve-lane")?.getBoundingClientRect().width || 1120;
            playhead.style.left = `${128 + (state.playhead / Math.max(.01, state.duration)) * width}px`;
            updateClock();
            root.querySelectorAll("[data-monitor-label]").forEach((el) => {
                el.textContent = `${el.textContent.split(":")[0]}: scrub ${formatTime(state.playhead, fps)} / ${mode === "preview_selected_take" ? `T${selected}/A${selected}` : order.join(" -> ")}`;
            });
        };
        const scrub = (ev) => {
            const box = ruler.getBoundingClientRect();
            state.playhead = clamp(((ev.clientX - box.left) / Math.max(1, box.width)) * state.duration, 0, state.duration);
            updatePlayhead();
        };
        ruler.onpointerdown = (ev) => {
            ev.preventDefault();
            scrub(ev);
            const move = (e) => scrub(e);
            const up = () => { window.removeEventListener("pointermove", move, true); window.removeEventListener("pointerup", up, true); writeEditorState(); };
            window.addEventListener("pointermove", move, true);
            window.addEventListener("pointerup", up, true);
        };
        tracks.appendChild(playhead);
        const secondsToPct = (seconds) => `${(seconds / Math.max(.01, state.duration)) * 100}%`;
        const clipDrag = (clip, key, lane, dragMode, pointerEvent) => {
            const startBox = lane.getBoundingClientRect();
            const startX = pointerEvent.clientX;
            const original = { ...state.clipMap[key] };
            clip.classList.add("dragging");
            const onMove = (ev) => {
                ev.preventDefault();
                const delta = ((ev.clientX - startX) / Math.max(1, startBox.width)) * state.duration;
                if (dragMode === "left") {
                    const nextStart = clamp(original.start + delta, 0, original.start + original.length - .25);
                    state.clipMap[key].start = nextStart;
                    state.clipMap[key].length = Math.max(.25, original.length + (original.start - nextStart));
                } else if (dragMode === "right") {
                    state.clipMap[key].length = Math.max(.25, original.length + delta);
                } else {
                    state.clipMap[key].start = clamp(original.start + delta, 0, Math.max(0, state.duration - original.length));
                }
                clip.style.left = secondsToPct(state.clipMap[key].start);
                clip.style.width = secondsToPct(state.clipMap[key].length);
            };
            const onUp = () => {
                clip.classList.remove("dragging");
                window.removeEventListener("pointermove", onMove, true);
                window.removeEventListener("pointerup", onUp, true);
                writeEditorState();
                flash(ledger, `${key} ${state.clipMap[key].start.toFixed(2)}s + ${state.clipMap[key].length.toFixed(2)}s`);
            };
            window.addEventListener("pointermove", onMove, true);
            window.addEventListener("pointerup", onUp, true);
        };
        const addClipEl = (lane, key, title, isAudio, seed) => {
            const clipState = state.clipMap[key] || { start: 0, length: 4 };
            const clip = document.createElement("div");
            clip.className = `iamccs-sve-clip${isAudio ? " audio" : ""}`;
            clip.style.left = secondsToPct(clipState.start);
            clip.style.width = secondsToPct(clipState.length);
            clip.innerHTML = `<div class="iamccs-sve-handle left"></div><div class="iamccs-sve-handle right"></div>${isAudio ? '<canvas class="iamccs-sve-wave"></canvas>' : ""}<div class="iamccs-sve-clip-title">${title}</div>`;
            clip.onpointerdown = (ev) => {
                ev.preventDefault();
                ev.stopPropagation();
                const isLeft = ev.target?.classList?.contains("left");
                const isRight = ev.target?.classList?.contains("right");
                clipDrag(clip, key, lane, isLeft ? "left" : isRight ? "right" : "move", ev);
            };
            lane.appendChild(clip);
            if (isAudio) requestAnimationFrame(() => drawWave(clip.querySelector("canvas"), seed));
        };
        const makeTrack = (name, type, trackNo) => {
            const row = document.createElement("div");
            row.className = `iamccs-sve-track ${type}`;
            const label = document.createElement("div");
            label.className = "iamccs-sve-label";
            label.innerHTML = `<strong>${name}</strong><div class="iamccs-sve-lanehint">${type === "audio" ? "audio lane" : "video lane"}</div><div class="iamccs-sve-chiprow"></div>`;
            const chipRow = label.querySelector(".iamccs-sve-chiprow");
            ["M", "S", "L"].forEach((txt) => {
                const c = document.createElement("span");
                c.className = "iamccs-sve-chip";
                c.textContent = txt;
                c.onclick = () => { c.classList.toggle("active"); flash(ledger, `${name} ${txt}: ${c.classList.contains("active") ? "ON" : "OFF"}`); };
                chipRow.appendChild(c);
            });
            const lane = document.createElement("div");
            lane.className = "iamccs-sve-lane";
            order.forEach((take, pos) => {
                if (type === "video" && trackNo === 2 && pos % 2 === 0) return;
                if (type === "video" && trackNo === 1 && pos % 2 === 1) return;
                if (type === "audio" && trackNo > 1 && take !== trackNo) return;
                const key = clipKey(type === "audio" ? "a" : "v", take, pos);
                addClipEl(lane, key, `${type === "audio" ? `A${take}` : `T${String(take).padStart(2, "0")}`} ${type}`, type === "audio", take + pos * 2 + trackNo);
            });
            Object.entries(state.clipMap).forEach(([key, clip]) => {
                if (!clip?.manual) return;
                if (type === "video" && trackNo === 2 && clip.type === "video") addClipEl(lane, key, `Manual V${key.replace(/\D/g, "")}`, false, 1);
                if (type === "audio" && trackNo === 3 && clip.type === "audio") addClipEl(lane, key, `Manual A${key.replace(/\D/g, "")}`, true, 12);
            });
            row.append(label, lane);
            return row;
        };
        tracks.append(makeTrack("V1", "video", 1), makeTrack("V2", "video", 2), makeTrack("A1", "audio", 1), makeTrack("A2", "audio", 2), makeTrack("A3", "audio", 3));
        timeline.append(ruler, tracks);
        const actions = document.createElement("div");
        actions.className = "iamccs-sve-actions";
        actions.append(
            button("Assemble Hard Cut", () => { setWidget(node, "editor_mode", "assemble_sequence"); flash(ledger, "Assemble mode"); render(); }, mode === "assemble_sequence" ? "primary" : ""),
            button(`Manual T${selected}/A${selected}`, () => { setWidget(node, "editor_mode", "preview_selected_take"); flash(ledger, `Manual T${selected}/A${selected}`); render(); }, mode === "preview_selected_take" ? "primary" : ""),
            button("Use Master Audio", () => { setWidget(node, "audio_policy", "use_master_audio"); flash(ledger, "Master audio selected for final editor output"); render(); }),
            button("Write Edit Metadata", () => { writeEditorState(); flash(ledger, "Edit metadata written"); }, "primary")
        );
        ledger.textContent = `Ready. ${data.videos.length} CineInfo3 videos / ${data.audios.length} CineInfo3 audios. Drag clip center to move; drag yellow handles to trim.`;
        shell.append(top, sourcebar, monitors, timeline, actions, ledger);
        root.appendChild(shell);
        syncOverlayMount();
        requestAnimationFrame(updatePlayhead);
    }
    render();
    const uiWidget = node.addDOMWidget("Shotboard Video Editor", "iamccs_shotboard_video_editor_ui", root, { serialize: false });
    uiWidget.computeSize = () => [FIXED_SIZE[0] - 24, NODE_BODY_HEIGHT + 16];
    console.info("[IAMCCS Shotboard Video Editor UI] full NLE installed", { nodeId: node?.id, reason });
}
app.registerExtension({
    name: "IAMCCS.ShotboardVideoEditorUI.NLEFull",
    setup() {
        [700, 1800, 3600].forEach((delay) => setTimeout(() => {
            const nodes = Array.isArray(app?.graph?._nodes) ? app.graph._nodes : [];
            nodes.forEach((node) => installEditorUI(node, `scan+${delay}`));
        }, delay));
    },
    nodeCreated(node) { [0, 250, 900].forEach((delay) => setTimeout(() => installEditorUI(node, `nodeCreated+${delay}`), delay)); },
    loadedGraphNode(node) { [0, 250, 900].forEach((delay) => setTimeout(() => installEditorUI(node, `loadedGraphNode+${delay}`), delay)); },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "IAMCCS_ShotboardVideoEditor") return;
        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            originalOnNodeCreated?.apply(this, arguments);
            setTimeout(() => installEditorUI(this, "prototype.onNodeCreated"), 0);
        };
    },
});
