import { app } from "../../scripts/app.js";

const STYLE_ID = "iamccs-shotboard-video-editor-style-v3";
const FIXED_SIZE = [1260, 940];

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
function parseOrder(text) {
    const nums = String(text || "1,2").split(/[,;\s]+/).map((v) => Math.round(Number(v))).filter((v) => v >= 1 && v <= 8);
    return nums.length ? nums : [1, 2];
}
function formatTime(seconds, fps = 24) {
    const safe = Math.max(0, Number(seconds) || 0);
    const frame = Math.round(safe * fps);
    const s = Math.floor(safe % 60).toString().padStart(2, "0");
    const m = Math.floor(safe / 60).toString().padStart(2, "0");
    return `${m}:${s}.${String(frame % Math.round(fps)).padStart(2, "0")}`;
}
function flash(el, text) {
    if (!el) return;
    el.textContent = text;
    el.classList.add("pulse");
    setTimeout(() => el.classList.remove("pulse"), 220);
}
function readClipEdits(node) {
    try { return JSON.parse(String(widget(node, "clip_edits_json")?.value || "{}")); } catch { return {}; }
}
function writeClipEdits(node, patch) {
    const current = readClipEdits(node);
    const next = { ...(current || {}), ...patch, updated_at: Date.now() };
    setWidget(node, "clip_edits_json", JSON.stringify(next, null, 2));
}
function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
        .iamccs-sve { box-sizing:border-box; width:100%; height:884px; padding:9px; border:1px solid rgba(244,212,158,.34); border-radius:8px; background:#071012; color:#ebffff; font:11px Inter,Arial,sans-serif; overflow:hidden; pointer-events:auto; }
        .iamccs-sve * { box-sizing:border-box; }
        .iamccs-sve button { min-height:28px; border-radius:5px; border:1px solid rgba(126,198,196,.52); background:linear-gradient(180deg,#24484d,#142e32); color:#efffff; cursor:pointer; font-weight:950; font-size:10px; padding:0 9px; }
        .iamccs-sve button.active,.iamccs-sve button.primary { background:linear-gradient(180deg,#f5d89c,#c5964e); color:#171207; border-color:#ffe3a4; }
        .iamccs-sve input,.iamccs-sve select { min-width:0; height:28px; border-radius:5px; border:1px solid rgba(94,161,161,.72); background:#061012; color:#efffff; font-size:10px; font-weight:850; padding:0 7px; }
        .iamccs-sve-top { display:grid; grid-template-columns:220px 1fr; grid-template-rows:42px 42px; gap:7px; height:91px; margin-bottom:7px; }
        .iamccs-sve-brand,.iamccs-sve-transport,.iamccs-sve-tools { border:1px solid rgba(255,255,255,.09); border-radius:7px; background:linear-gradient(180deg,#132326,#081112); overflow:hidden; }
        .iamccs-sve-brand { padding:8px 10px; grid-row:1 / 3; }
        .iamccs-sve-title { color:#fff0b8; font-size:13px; font-weight:950; }
        .iamccs-sve-sub { color:#8fb4b3; font-size:9px; font-weight:850; margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        .iamccs-sve-transport,.iamccs-sve-tools { display:flex; align-items:center; justify-content:center; gap:6px; padding:6px; min-width:0; }
        .iamccs-sve-clock { min-width:104px; padding:5px 7px; border-radius:5px; background:#f3ffe9; border:2px solid #a7d59b; color:#0a7d23; font:12px Consolas,monospace; text-align:center; font-weight:950; }
        .iamccs-sve-sourcebar { height:38px; display:grid; grid-template-columns:70px 1fr 130px; align-items:center; gap:7px; padding:6px 8px; border:1px solid rgba(255,255,255,.09); border-radius:7px; background:#081112; margin-bottom:7px; }
        .iamccs-sve-sourcebar strong { color:#fff0b8; font-size:10px; }
        .iamccs-sve-takes { display:flex; gap:5px; overflow:hidden; }
        .iamccs-sve-take { min-width:50px; height:24px; padding:0 6px; font-size:9px; }
        .iamccs-sve-monitor-row { display:grid; grid-template-columns:1fr 1fr; gap:8px; height:260px; margin-bottom:7px; }
        .iamccs-sve.one-monitor .iamccs-sve-monitor-row { grid-template-columns:1fr; }
        .iamccs-sve.one-monitor .iamccs-sve-monitor.source { display:none; }
        .iamccs-sve-monitor { border:1px solid rgba(255,255,255,.10); border-radius:7px; background:linear-gradient(180deg,#0d1718,#030607); overflow:hidden; }
        .iamccs-sve-monitor-head { height:24px; display:flex; align-items:center; justify-content:space-between; padding:0 8px; background:linear-gradient(180deg,#15292b,#0d181a); border-bottom:1px solid rgba(255,255,255,.08); color:#ffe5a8; font-weight:950; font-size:9px; text-transform:uppercase; }
        .iamccs-sve-screen { height:216px; margin:9px; border-radius:6px; border:1px solid rgba(126,198,196,.28); background:radial-gradient(circle at 52% 45%,rgba(99,160,170,.18),transparent 32%),linear-gradient(135deg,#111b20,#06090b 52%,#141008); position:relative; overflow:hidden; }
        .iamccs-sve-safe { position:absolute; inset:15px 24px; border:1px dashed rgba(255,255,255,.18); border-radius:4px; }
        .iamccs-sve-screen-label { position:absolute; left:10px; bottom:8px; color:#eaffff; background:rgba(0,0,0,.50); border:1px solid rgba(255,255,255,.12); border-radius:4px; padding:4px 6px; font:10px Consolas,monospace; }
        .iamccs-sve-controls { display:grid; grid-template-columns:1fr 80px 1fr 1fr 70px; gap:7px; margin-bottom:7px; border:1px solid rgba(255,255,255,.08); border-radius:7px; padding:7px; background:#080f10; }
        .iamccs-sve-controls label { display:grid; gap:3px; min-width:0; color:#9db7b8; font-size:8px; text-transform:uppercase; font-weight:950; }
        .iamccs-sve-timeline { height:344px; border:1px solid rgba(255,255,255,.10); border-radius:8px; overflow:hidden; background:#05090a; }
        .iamccs-sve-ruler { height:34px; margin-left:116px; position:relative; border-bottom:1px solid rgba(255,255,255,.11); background:linear-gradient(180deg,#122022,#091113); cursor:ew-resize; user-select:none; }
        .iamccs-sve-ruler-tick { position:absolute; top:0; bottom:0; width:1px; background:rgba(255,255,255,.14); }
        .iamccs-sve-ruler-tick span { position:absolute; top:5px; left:4px; color:#b7d0cf; font-size:9px; font-weight:900; }
        .iamccs-sve-tracks { height:310px; overflow:auto; position:relative; }
        .iamccs-sve-playhead { position:absolute; top:0; bottom:0; width:2px; background:#ffe08d; z-index:20; box-shadow:0 0 10px rgba(255,224,141,.65); pointer-events:none; }
        .iamccs-sve-track { min-height:74px; display:grid; grid-template-columns:116px 1fr; border-bottom:1px solid rgba(255,255,255,.12); }
        .iamccs-sve-track.audio { min-height:68px; }
        .iamccs-sve-label { padding:7px; border-right:1px solid rgba(255,255,255,.10); background:linear-gradient(90deg,rgba(36,68,65,.55),rgba(9,17,18,.92)); }
        .iamccs-sve-label strong { color:#fff0b8; font-size:12px; display:block; }
        .iamccs-sve-chiprow { display:flex; gap:4px; margin-top:5px; flex-wrap:wrap; }
        .iamccs-sve-chip { height:18px; min-width:22px; border-radius:4px; border:1px solid rgba(126,198,196,.45); color:#dfffff; background:#102f34; font-size:8px; font-weight:950; display:flex; align-items:center; justify-content:center; cursor:pointer; }
        .iamccs-sve-chip.active { background:#f3d08d; color:#171207; border-color:#ffe3a4; }
        .iamccs-sve-lane { position:relative; min-width:980px; border-left:1px solid rgba(255,224,141,.18); background:repeating-linear-gradient(90deg,rgba(255,255,255,.07) 0 1px,transparent 1px 48px),linear-gradient(180deg,rgba(24,31,29,.50),rgba(8,10,11,.90)); }
        .iamccs-sve-track:nth-child(even) .iamccs-sve-lane { background:repeating-linear-gradient(90deg,rgba(255,255,255,.06) 0 1px,transparent 1px 48px),linear-gradient(180deg,rgba(30,22,28,.48),rgba(8,10,11,.92)); }
        .iamccs-sve-clip { position:absolute; top:10px; height:52px; border-radius:5px; border:1px solid rgba(255,224,160,.65); background:linear-gradient(180deg,#31677a,#16343f); box-shadow:inset 0 0 0 1px rgba(255,255,255,.08); overflow:hidden; cursor:grab; touch-action:none; }
        .iamccs-sve-clip.audio { height:46px; background:linear-gradient(180deg,#386d9b,#173b63); border-color:#b9dcff; }
        .iamccs-sve-clip.dragging { cursor:grabbing; box-shadow:0 0 0 2px rgba(255,224,141,.55),0 8px 20px rgba(0,0,0,.35); z-index:10; }
        .iamccs-sve-clip-title { position:absolute; left:11px; top:5px; z-index:2; color:#fff1ba; font-weight:950; font-size:10px; text-shadow:0 1px 2px #000; pointer-events:none; }
        .iamccs-sve-handle { position:absolute; top:0; bottom:0; width:9px; background:#ffe08d; z-index:3; cursor:ew-resize; }
        .iamccs-sve-handle.left { left:0; }
        .iamccs-sve-handle.right { right:0; }
        .iamccs-sve-wave { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }
        .iamccs-sve-clip:not(.audio)::before { content:""; position:absolute; inset:0; background:repeating-linear-gradient(90deg, rgba(255,255,255,.12) 0 2px, transparent 2px 9px, rgba(0,0,0,.25) 9px 34px), linear-gradient(90deg, rgba(93,151,164,.34), rgba(23,54,64,.78), rgba(190,145,78,.28)); opacity:.92; }
        .iamccs-sve-clip:not(.audio)::after { content:""; position:absolute; left:0; right:0; top:0; height:7px; background:repeating-linear-gradient(90deg,#071012 0 6px,#e8d8a3 6px 9px); opacity:.75; }
        .iamccs-sve-actions { display:flex; gap:6px; align-items:center; justify-content:flex-end; margin-top:7px; }
        .iamccs-sve-ledger { margin-top:6px; padding:6px 8px; border:1px solid rgba(255,255,255,.08); border-radius:5px; background:#030708; color:#b8fff1; font:10px Consolas,monospace; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
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
    const mid = h * 0.56;
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, "rgba(235,249,255,.98)");
    grad.addColorStop(1, "rgba(130,200,232,.82)");
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.moveTo(0, mid);
    for (let x = 0; x <= w; x += 2) {
        const t = (x / Math.max(1, w)) * 22 + seed;
        const amp = (Math.sin(t * 1.7) * .22 + Math.sin(t * 4.6) * .16 + Math.sin(t * 11.3) * .07 + .45);
        const env = .28 + .72 * Math.abs(Math.sin(t * .31 + seed));
        ctx.lineTo(x, mid - amp * env * h * .38);
    }
    for (let x = w; x >= 0; x -= 2) {
        const t = (x / Math.max(1, w)) * 22 + seed;
        const amp = (Math.sin(t * 1.7) * .22 + Math.sin(t * 4.6) * .16 + Math.sin(t * 11.3) * .07 + .45);
        const env = .28 + .72 * Math.abs(Math.sin(t * .31 + seed));
        ctx.lineTo(x, mid + amp * env * h * .34);
    }
    ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = "rgba(255,238,181,.72)";
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
    const state = {
        playhead: 0,
        duration: 24,
        oneMonitor: false,
        clipMap: {},
        buttons: {},
    };
    const clipKey = (type, take, pos) => `${type}_${take}_${pos}`;
    function initClipState(order, perTake) {
        order.forEach((take, pos) => {
            ["v", "a"].forEach((type) => {
                const key = clipKey(type, take, pos);
                if (!state.clipMap[key]) state.clipMap[key] = { start: pos * perTake, length: perTake };
            });
        });
    }
    function writeEditorState(node) {
        writeClipEdits(node, {
            ui_state: {
                playhead: state.playhead,
                oneMonitor: state.oneMonitor,
                clips: state.clipMap,
            },
        });
    }
    function render() {
        hideRawWidgets(node);
        installFixedNode(node);
        root.classList.toggle("one-monitor", Boolean(state.oneMonitor));
        const mode = String(widget(node, "editor_mode")?.value || "assemble_sequence");
        const selected = clamp(Math.round(Number(widget(node, "selected_take")?.value || 1)), 1, 8);
        const orderText = String(widget(node, "take_order")?.value || "1,2");
        const order = parseOrder(orderText);
        const fps = Number(widget(node, "override_fps")?.value || 24);
        const perTake = 8;
        initClipState(order, perTake);
        state.duration = Math.max(12, ...Object.values(state.clipMap).map((clip) => Number(clip.start || 0) + Number(clip.length || perTake)));
        state.playhead = clamp(state.playhead, 0, state.duration);
        root.innerHTML = "";

        const top = document.createElement("div");
        top.className = "iamccs-sve-top";
        const brand = document.createElement("div");
        brand.className = "iamccs-sve-brand";
        brand.innerHTML = `<div class="iamccs-sve-title">Shotboard Video Editor</div><div class="iamccs-sve-sub">timeline editor / monitor / T-A takes</div>`;
        const transport = document.createElement("div");
        transport.className = "iamccs-sve-transport";
        const ledger = document.createElement("div");
        ledger.className = "iamccs-sve-ledger";
        const clock = document.createElement("div");
        clock.className = "iamccs-sve-clock";
        const updateClock = () => { clock.textContent = formatTime(state.playhead, fps); };
        updateClock();
        [
            ["|<", () => { state.playhead = 0; updatePlayhead(); flash(ledger, "Go to start"); }],
            ["<<", () => { state.playhead = Math.max(0, state.playhead - .5); updatePlayhead(); flash(ledger, "Step back"); }],
            ["Play", (btn) => { btn.classList.toggle("active"); flash(ledger, btn.classList.contains("active") ? "Playback armed" : "Playback stopped"); }],
            [">>", () => { state.playhead = Math.min(state.duration, state.playhead + .5); updatePlayhead(); flash(ledger, "Step forward"); }],
            [">|", () => { state.playhead = state.duration; updatePlayhead(); flash(ledger, "Go to end"); }],
        ].forEach(([label, cb]) => {
            const b = document.createElement("button");
            b.type = "button";
            b.textContent = label;
            if (label === "Play") b.className = "primary";
            b.onclick = () => cb(b);
            transport.appendChild(b);
        });
        transport.appendChild(clock);
        const tools = document.createElement("div");
        tools.className = "iamccs-sve-tools";
        [["Cut", "cut"], ["Trim", "trim"], ["Razor", "razor"], ["Snap", "snap"], ["1 Monitor", "oneMonitor"]].forEach(([label, key]) => {
            const b = document.createElement("button");
            b.type = "button";
            b.textContent = label;
            b.className = state.buttons[key] ? "active" : "";
            b.onclick = () => {
                state.buttons[key] = !state.buttons[key];
                if (key === "oneMonitor") state.oneMonitor = !state.oneMonitor;
                b.classList.toggle("active", Boolean(state.buttons[key]));
                writeEditorState(node);
                flash(ledger, `${label}: ${state.buttons[key] ? "ON" : "OFF"}`);
                if (key === "oneMonitor") render();
            };
            tools.appendChild(b);
        });
        top.append(brand, transport, tools);

        const sourcebar = document.createElement("div");
        sourcebar.className = "iamccs-sve-sourcebar";
        sourcebar.innerHTML = `<strong>T/A TAKES</strong>`;
        const takeRow = document.createElement("div");
        takeRow.className = "iamccs-sve-takes";
        for (let take = 1; take <= 8; take += 1) {
            const b = document.createElement("button");
            b.type = "button";
            b.className = `iamccs-sve-take${take === selected ? " active" : ""}`;
            b.textContent = `T${take}/A${take}`;
            b.onclick = () => {
                setWidget(node, "selected_take", take);
                setWidget(node, "editor_mode", "preview_selected_take");
                flash(ledger, `Selected T${take}/A${take}`);
                render();
            };
            takeRow.appendChild(b);
        }
        const monitorToggle = document.createElement("button");
        monitorToggle.type = "button";
        monitorToggle.textContent = state.oneMonitor ? "Two Monitors" : "One Monitor";
        monitorToggle.onclick = () => { state.oneMonitor = !state.oneMonitor; render(); };
        sourcebar.append(takeRow, monitorToggle);

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
        selectedInput.type = "number"; selectedInput.min = "1"; selectedInput.max = "8"; selectedInput.value = String(selected);
        selectedInput.onchange = () => { setWidget(node, "selected_take", clamp(selectedInput.value, 1, 8)); render(); };
        const orderInput = document.createElement("input");
        orderInput.value = orderText;
        orderInput.onchange = () => { setWidget(node, "take_order", orderInput.value || "1,2"); render(); };
        const audioSelect = document.createElement("select");
        [["concat_clip_audio", "Concat clip audio"], ["use_master_audio", "Use master audio"], ["first_selected_audio", "Selected clip audio"], ["mix_editor_audio_tracks", "Mix editor audio tracks"], ["concat_editor_audio_tracks", "Concat editor audio tracks"], ["silent", "Silent"]].forEach(([v, l]) => { const o = document.createElement("option"); o.value = v; o.textContent = l; audioSelect.appendChild(o); });
        audioSelect.value = String(widget(node, "audio_policy")?.value || "concat_clip_audio");
        audioSelect.onchange = () => { setWidget(node, "audio_policy", audioSelect.value); flash(ledger, `Audio policy: ${audioSelect.value}`); };
        const fpsInput = document.createElement("input");
        fpsInput.type = "number"; fpsInput.step = "0.01"; fpsInput.value = String(fps);
        fpsInput.onchange = () => setWidget(node, "override_fps", Math.max(1, Number(fpsInput.value || 24)));
        controls.append(labelWrap("Mode", modeSelect), labelWrap("T/A", selectedInput), labelWrap("Order", orderInput), labelWrap("Audio", audioSelect), labelWrap("FPS", fpsInput));

        const timeline = document.createElement("div");
        timeline.className = "iamccs-sve-timeline";
        const ruler = document.createElement("div");
        ruler.className = "iamccs-sve-ruler";
        for (let sec = 0; sec <= state.duration; sec += 1) {
            const tick = document.createElement("div");
            tick.className = "iamccs-sve-ruler-tick";
            tick.style.left = `${(sec / state.duration) * 100}%`;
            if (sec % 2 === 0) tick.innerHTML = `<span>${sec}s<br>${Math.round(sec * fps)}f</span>`;
            ruler.appendChild(tick);
        }
        const tracks = document.createElement("div");
        tracks.className = "iamccs-sve-tracks";
        const playhead = document.createElement("div");
        playhead.className = "iamccs-sve-playhead";
        const updatePlayhead = () => {
            const width = tracks.querySelector(".iamccs-sve-lane")?.getBoundingClientRect().width || 980;
            playhead.style.left = `${116 + (state.playhead / state.duration) * width}px`;
            updateClock();
            root.querySelectorAll("[data-monitor-label]").forEach((el) => { el.textContent = `${el.textContent.split(":")[0]}: scrub ${formatTime(state.playhead, fps)} / ${mode === "preview_selected_take" ? `T${selected}/A${selected}` : order.join(" -> ")}`; });
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
            const up = () => { window.removeEventListener("pointermove", move, true); window.removeEventListener("pointerup", up, true); writeEditorState(node); };
            window.addEventListener("pointermove", move, true);
            window.addEventListener("pointerup", up, true);
        };
        tracks.appendChild(playhead);
        const secondsToPct = (seconds) => `${(seconds / state.duration) * 100}%`;
        const clipDrag = (clip, key, lane, mode, pointerEvent) => {
            const startBox = lane.getBoundingClientRect();
            const startX = pointerEvent.clientX;
            const original = { ...state.clipMap[key] };
            clip.classList.add("dragging");
            const onMove = (ev) => {
                ev.preventDefault();
                const delta = ((ev.clientX - startX) / Math.max(1, startBox.width)) * state.duration;
                if (mode === "left") {
                    const nextStart = clamp(original.start + delta, 0, original.start + original.length - .25);
                    state.clipMap[key].start = nextStart;
                    state.clipMap[key].length = Math.max(.25, original.length + (original.start - nextStart));
                } else if (mode === "right") {
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
                writeEditorState(node);
                flash(ledger, `${key} updated ${state.clipMap[key].start.toFixed(2)}s + ${state.clipMap[key].length.toFixed(2)}s`);
            };
            window.addEventListener("pointermove", onMove, true);
            window.addEventListener("pointerup", onUp, true);
        };
        const makeTrack = (name, type, trackNo) => {
            const row = document.createElement("div");
            row.className = `iamccs-sve-track ${type}`;
            const label = document.createElement("div");
            label.className = "iamccs-sve-label";
            label.innerHTML = `<strong>${name}</strong><div class="iamccs-sve-chiprow"></div>`;
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
                const clipState = state.clipMap[key] || { start: pos * perTake, length: perTake };
                const clip = document.createElement("div");
                clip.className = `iamccs-sve-clip${type === "audio" ? " audio" : ""}`;
                clip.style.left = secondsToPct(clipState.start);
                clip.style.width = secondsToPct(clipState.length);
                clip.innerHTML = `<div class="iamccs-sve-handle left"></div><div class="iamccs-sve-handle right"></div>${type === "audio" ? '<canvas class="iamccs-sve-wave"></canvas>' : ""}<div class="iamccs-sve-clip-title">${type === "audio" ? `A${take}` : `T${String(take).padStart(2, "0")}`} ${type === "audio" ? "audio" : "video"}</div>`;
                clip.onpointerdown = (ev) => {
                    ev.preventDefault();
                    ev.stopPropagation();
                    const isLeft = ev.target?.classList?.contains("left");
                    const isRight = ev.target?.classList?.contains("right");
                    clipDrag(clip, key, lane, isLeft ? "left" : isRight ? "right" : "move", ev);
                };
                lane.appendChild(clip);
                if (type === "audio") requestAnimationFrame(() => drawWave(clip.querySelector("canvas"), take + pos * 2 + trackNo));
            });
            row.append(label, lane);
            return row;
        };
        tracks.append(makeTrack("V1", "video", 1), makeTrack("V2", "video", 2), makeTrack("A1", "audio", 1), makeTrack("A2", "audio", 2), makeTrack("A3", "audio", 3));
        timeline.append(ruler, tracks);
        const actions = document.createElement("div");
        actions.className = "iamccs-sve-actions";
        [["Assemble Hard Cut", () => { setWidget(node, "editor_mode", "assemble_sequence"); flash(ledger, "Assemble mode"); render(); }], [`Manual T${selected}/A${selected}`, () => { setWidget(node, "editor_mode", "preview_selected_take"); flash(ledger, `Manual T${selected}/A${selected}`); render(); }], ["Write Edit Metadata", () => { writeEditorState(node); flash(ledger, "Edit metadata written"); }]].forEach(([label, cb], idx) => {
            const b = document.createElement("button");
            b.type = "button";
            b.textContent = label;
            if ((idx === 0 && mode === "assemble_sequence") || (idx === 1 && mode === "preview_selected_take")) b.className = "primary";
            b.onclick = cb;
            actions.appendChild(b);
        });
        ledger.textContent = "Ready. Drag clip center to move; drag yellow handles to trim. Ruler scrubs playhead.";
        root.append(top, sourcebar, monitors, controls, timeline, actions, ledger);
        requestAnimationFrame(updatePlayhead);
    }
    render();
    const uiWidget = node.addDOMWidget("Shotboard Video Editor", "iamccs_shotboard_video_editor_ui", root, { serialize: false });
    uiWidget.computeSize = () => [FIXED_SIZE[0] - 24, 900];
    console.info("[IAMCCS Shotboard Video Editor UI] interactive NLE installed", { nodeId: node?.id, reason });
}
app.registerExtension({
    name: "IAMCCS.ShotboardVideoEditorUI.NLEInteractive",
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
