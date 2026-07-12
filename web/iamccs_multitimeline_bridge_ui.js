import { app } from "../../scripts/app.js";

console.info("[IAMCCS MultiTimelineBridge UI] stable module loaded", { ts: new Date().toISOString() });

const STYLE_ID = "iamccs-multitimeline-bridge-stable-style";
const BRIDGE_FIXED_SIZE = [660, 470];
const BRIDGE_CHROME_HEIGHT = 126;
const BRIDGE_WIDGET_HEIGHT = BRIDGE_FIXED_SIZE[1] - BRIDGE_CHROME_HEIGHT;

function nodeType(node) {
    return String(node?.type || node?.comfyClass || node?.constructor?.type || "");
}

function isBridgeNode(node) {
    const type = nodeType(node);
    return type === "IAMCCS_MultiTimelineBridge" || type.includes("MultiTimelineBridge");
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
    item.computeSize = () => [0, 0];
    item.draw = () => {};
    item.options = { ...(item.options || {}), hidden: true };
    if (item.inputEl) {
        item.inputEl.style.display = "none";
        item.inputEl.style.height = "0";
        item.inputEl.style.opacity = "0";
    }
}

function hideRawWidgets(node) {
    [
        "chunk_template",
        "custom_chunk_seconds",
        "source_bus",
        "take_source_mode",
        "take_count_mode",
        "fixed_take_count",
        "max_takes",
        "active_take",
        "frame_rate",
        "take_track_layout",
        "bus_manifest_json",
        "master_out_json",
        "track_1_json",
        "track_2_json",
        "track_3_json",
        "track_4_json",
        "track_5_json",
        "visual_timelines_json",
    ].forEach((name) => hideWidget(widget(node, name)));
}

function num(value, fallback = 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
}

function maxTakes(node) {
    return Math.max(2, Math.min(12, Math.round(num(widget(node, "max_takes")?.value, widget(node, "fixed_take_count")?.value || 3))));
}

function activeTake(node) {
    return Math.max(1, Math.min(maxTakes(node), Math.round(num(widget(node, "active_take")?.value, 1))));
}

function selectedTakesCsv(node) {
    const raw = String(node?.properties?.iamccs_selected_takes_csv || "").trim();
    if (raw) return raw;
    const countMode = String(widget(node, "take_count_mode")?.value || "auto_from_audio");
    const desired = countMode === "fixed_take_count"
        ? Math.max(1, Math.round(num(widget(node, "fixed_take_count")?.value, 3)))
        : maxTakes(node);
    return Array.from({ length: Math.min(maxTakes(node), desired) }, (_, index) => String(index + 1)).join(",");
}

function selectedTakeList(node) {
    const max = maxTakes(node);
    const unique = [];
    String(selectedTakesCsv(node) || "")
        .split(/[,;\s]+/)
        .map((value) => Math.round(Number(value)))
        .filter((value) => value >= 1 && value <= max)
        .forEach((value) => {
            if (!unique.includes(value)) unique.push(value);
        });
    return unique.length ? unique : [activeTake(node)];
}

function setSelectedTakesCsv(node, value) {
    node.properties = node.properties || {};
    node.properties.iamccs_selected_takes_csv = String(value || "").trim();
    try { node.setDirtyCanvas?.(true, true); } catch {}
    try { app.graph?.setDirtyCanvas?.(true, true); } catch {}
}

function setTake(node, take) {
    const safeTake = Math.max(1, Math.round(Number(take) || 1));
    setWidget(node, "active_take", safeTake);
    try {
        window.dispatchEvent(new CustomEvent("iamccs:multigeneration-active-take", {
            detail: {
                nodeId: node?.id,
                activeTake: safeTake,
                timelineId: `T${String(safeTake).padStart(2, "0")}`,
                audioLane: `A${safeTake}`,
                source: "bridge",
            },
        }));
    } catch {}
}

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

async function queuePromptOnce() {
    if (typeof app?.queuePrompt === "function") {
        return await app.queuePrompt(0, 1);
    }
    const api = window?.comfyAPI?.api;
    if (typeof api?.queuePrompt === "function") {
        return await api.queuePrompt(0, 1);
    }
    throw new Error("ComfyUI queuePrompt API not available");
}

function waitForShotboardTakeReady(take, timeoutMs = 2200) {
    const wanted = Math.max(1, Math.round(Number(take) || 1));
    return new Promise((resolve) => {
        let finished = false;
        const finish = (detail, timedOut = false) => {
            if (finished) return;
            finished = true;
            window.removeEventListener("iamccs:multigeneration-shotboard-take-ready", onReady);
            clearTimeout(timer);
            resolve({ ok: !timedOut, detail: detail || null });
        };
        const onReady = (event) => {
            const detail = event?.detail || {};
            const eventTake = Math.max(1, Math.round(Number(detail.activeTake || String(detail.timelineId || "").replace(/\D/g, "") || 1)));
            if (eventTake === wanted) finish(detail, false);
        };
        const timer = setTimeout(() => finish(null, true), timeoutMs);
        window.addEventListener("iamccs:multigeneration-shotboard-take-ready", onReady);
    });
}

async function waitForQueueIdle(statusEl = null, label = "", timeoutMs = 1000 * 60 * 90) {
    const started = Date.now();
    let sawBusy = false;
    while (Date.now() - started < timeoutMs) {
        try {
            const response = await fetch("/queue", { cache: "no-store" });
            const data = await response.json();
            const running = Array.isArray(data?.queue_running) ? data.queue_running.length : 0;
            const pending = Array.isArray(data?.queue_pending) ? data.queue_pending.length : 0;
            const busy = running + pending > 0;
            sawBusy = sawBusy || busy;
            if (statusEl && (busy || sawBusy)) statusEl.textContent = `${label} running: ${running} / pending: ${pending}`;
            if (sawBusy && !busy) return true;
        } catch {
            await sleep(900);
        }
        await sleep(1200);
    }
    return false;
}

async function queueSingleBackendSequence(node, count, statusEl = null, takeList = null) {
    const sequence = Array.isArray(takeList) && takeList.length
        ? takeList.map((take) => Math.max(1, Math.min(maxTakes(node), Math.round(Number(take) || 1))))
        : Array.from({ length: Math.max(1, Math.min(maxTakes(node), Math.round(Number(count) || 1))) }, (_, index) => index + 1);
    setWidget(node, "take_source_mode", "auto_detect_multi_lanes");
    setWidget(node, "take_track_layout", "collapse_to_lane_1");
    for (let index = 0; index < sequence.length; index += 1) {
        const take = sequence[index];
        setTake(node, take);
        const takeLabel = `T${String(take).padStart(2, "0")} / A${take}`;
        if (statusEl) statusEl.textContent = `Preparing ${takeLabel} on the single backend (${index + 1}/${sequence.length})...`;
        try { app.graph?.setDirtyCanvas?.(true, true); } catch {}
        const ready = await waitForShotboardTakeReady(take);
        if (statusEl) {
            const segs = ready?.detail?.audioSegments ?? "?";
            statusEl.textContent = `${ready.ok ? "Ready" : "Timed wait"} ${takeLabel}: ${segs} audio segment(s). Queueing...`;
        }
        if (!ready.ok) {
            throw new Error(`${takeLabel} was not confirmed by Shotboard before queue. Open/select the Shotboard timeline first, then press Prepare or Queue again.`);
        }
        await sleep(260);
        await queuePromptOnce();
        await waitForQueueIdle(statusEl, `${takeLabel}`);
        await sleep(350);
    }
    if (statusEl) statusEl.textContent = `Completed sequential queue: ${sequence.map((take) => `T${String(take).padStart(2, "0")}/A${take}`).join(" -> ")}.`;
}

function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
        .iamccs-mtb {
            box-sizing: border-box;
            width: 100%;
            height: ${BRIDGE_WIDGET_HEIGHT}px;
            max-height: ${BRIDGE_WIDGET_HEIGHT}px;
            padding: 8px;
            border: 1px solid rgba(244, 212, 158, .28);
            border-radius: 4px;
            background: linear-gradient(180deg, #141b1d, #080d0f);
            color: #e8f7f3;
            font: 11px Inter, Arial, sans-serif;
            pointer-events: auto;
            overflow: hidden;
            display: grid;
            grid-template-rows: 34px 112px 50px 34px 1fr;
            gap: 7px;
        }
        .iamccs-mtb-head {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 10px;
            min-height: 0;
            overflow: hidden;
        }
        .iamccs-mtb-title {
            color: #fff1ba;
            font-size: 13px;
            font-weight: 950;
        }
        .iamccs-mtb-sub {
            color: #91b4b3;
            font-size: 9px;
            font-weight: 850;
            margin-top: 2px;
        }
        .iamccs-mtb-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 6px;
            min-height: 0;
            overflow: hidden;
        }
        .iamccs-mtb label {
            min-width: 0;
            display: grid;
            gap: 3px;
            color: #9fb9ba;
            font-size: 8px;
            font-weight: 950;
            text-transform: uppercase;
        }
        .iamccs-mtb select,
        .iamccs-mtb input {
            min-width: 0;
            height: 27px;
            border: 1px solid rgba(91, 151, 154, .72);
            border-radius: 5px;
            background: #071013;
            color: #eaffff;
            font-size: 10px;
            font-weight: 850;
            padding: 0 7px;
            box-sizing: border-box;
        }
        .iamccs-mtb button {
            min-height: 27px;
            border: 1px solid rgba(143, 208, 204, .42);
            border-radius: 5px;
            background: linear-gradient(180deg, #284a4e, #1a3034);
            color: #ecffff;
            cursor: pointer;
            font-size: 10px;
            font-weight: 900;
        }
        .iamccs-mtb button.is-primary,
        .iamccs-mtb-take.is-active {
            color: #171207;
            background: linear-gradient(180deg, #f2d79a, #c79e59);
            border-color: #ffe6ae;
        }
        .iamccs-mtb-takes {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 6px;
            min-height: 0;
            overflow: hidden;
        }
        .iamccs-mtb-take {
            min-width: 0;
            min-height: 48px;
            padding: 6px;
            border: 1px solid rgba(143,208,204,.26);
            border-radius: 6px;
            background: rgba(0,0,0,.24);
            color: #e8f7f3;
            text-align: left;
        }
        .iamccs-mtb-take strong {
            display: block;
            font-size: 12px;
            font-weight: 950;
        }
        .iamccs-mtb-take span {
            display: block;
            color: #8ba9aa;
            font-size: 9px;
            font-weight: 850;
        }
        .iamccs-mtb-actions {
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
            justify-content: flex-end;
            min-height: 0;
            overflow: hidden;
        }
        .iamccs-mtb-ledger {
            padding: 5px 7px;
            border: 1px solid rgba(255,255,255,.08);
            border-radius: 5px;
            background: #05090a;
            color: #b8fff1;
            font: 10px Consolas, monospace;
            min-height: 0;
            overflow: hidden;
        }
    `;
    document.head.appendChild(style);
}

function installFixedBridgeNode(node) {
    if (typeof node.setSize === "function") node.setSize([...BRIDGE_FIXED_SIZE]);
    else node.size = [...BRIDGE_FIXED_SIZE];
    node.resizable = false;
    node.resizeable = false;
    node.flags = { ...(node.flags || {}), resizable: false };
    node.min_size = [...BRIDGE_FIXED_SIZE];
    node.shape = 0;
    if (!node._iamccsMtbOriginalOnResize) node._iamccsMtbOriginalOnResize = node.onResize;
    node.onResize = function () {
        try { this._iamccsMtbOriginalOnResize?.apply(this, arguments); } catch {}
        this.size = [...BRIDGE_FIXED_SIZE];
        this.resizable = false;
        this.resizeable = false;
        this.flags = { ...(this.flags || {}), resizable: false };
        this.min_size = [...BRIDGE_FIXED_SIZE];
    };
    try { node.setDirtyCanvas?.(true, true); } catch {}
}

function select(options, value, onChange) {
    const el = document.createElement("select");
    options.forEach(([optionValue, optionLabel]) => {
        const option = document.createElement("option");
        option.value = String(optionValue);
        option.textContent = optionLabel;
        el.appendChild(option);
    });
    el.value = String(value);
    el.onchange = () => onChange(el.value);
    return el;
}

function field(label, child) {
    const wrap = document.createElement("label");
    wrap.textContent = label;
    wrap.appendChild(child);
    return wrap;
}

function installBridgeUI(node, reason = "install") {
    if (!isBridgeNode(node) || node._iamccsStableMultiTimelineBridgeReady || typeof node.addDOMWidget !== "function") return;
    node._iamccsStableMultiTimelineBridgeReady = true;
    ensureStyle();
    hideRawWidgets(node);
    installFixedBridgeNode(node);
    const root = document.createElement("div");
    root.className = "iamccs-mtb";

    const render = () => {
        hideRawWidgets(node);
        installFixedBridgeNode(node);
        const active = activeTake(node);
        const max = maxTakes(node);
        const chunkTemplate = String(widget(node, "chunk_template")?.value || "20s");
        const customSeconds = num(widget(node, "custom_chunk_seconds")?.value, 20);
        const fixedCount = Math.max(1, Math.round(num(widget(node, "fixed_take_count")?.value, 3)));
        root.innerHTML = "";

        const head = document.createElement("div");
        head.className = "iamccs-mtb-head";
        const title = document.createElement("div");
        title.innerHTML = `<div class="iamccs-mtb-title">IAMCCS MultiTimeline Bridge</div><div class="iamccs-mtb-sub">real indexed take bridge, no raw JSON editing</div>`;
        const refresh = document.createElement("button");
        refresh.type = "button";
        refresh.textContent = "Refresh UI";
        refresh.onclick = render;
        head.append(title, refresh);

        const grid = document.createElement("div");
        grid.className = "iamccs-mtb-grid";
        const templateSelect = select([["10s", "10 sec"], ["15s", "15 sec"], ["20s", "20 sec"], ["25s", "25 sec"], ["custom", "custom"]], chunkTemplate, (value) => {
            setWidget(node, "chunk_template", value);
            render();
        });
        const secondsInput = document.createElement("input");
        secondsInput.type = "number";
        secondsInput.min = "1";
        secondsInput.max = "300";
        secondsInput.step = "0.25";
        secondsInput.value = String(customSeconds);
        secondsInput.onchange = () => setWidget(node, "custom_chunk_seconds", Math.max(1, Number(secondsInput.value || 20)));
        const activeSelect = select(Array.from({ length: max }, (_, index) => {
            const take = index + 1;
            return [String(take), `T${String(take).padStart(2, "0")} / generation ${take}`];
        }), active, (value) => {
            setTake(node, value);
            render();
        });
        const sourceSelect = select([["master_out", "Master out"], ["track_1", "A1"], ["track_2", "A2"], ["track_3", "A3"], ["track_4", "A4"], ["track_5", "A5"]], widget(node, "source_bus")?.value || "master_out", (value) => setWidget(node, "source_bus", value));
        const sourceMode = select([["auto_detect_multi_lanes", "Use arranger T lanes"], ["chunk_source_bus", "Chunk source bus"]], widget(node, "take_source_mode")?.value || "auto_detect_multi_lanes", (value) => setWidget(node, "take_source_mode", value));
        const countMode = select([["auto_from_audio", "Auto"], ["fixed_take_count", "Fixed"]], widget(node, "take_count_mode")?.value || "auto_from_audio", (value) => setWidget(node, "take_count_mode", value));
        const fixedInput = document.createElement("input");
        fixedInput.type = "number";
        fixedInput.min = "1";
        fixedInput.max = "64";
        fixedInput.value = String(fixedCount);
        fixedInput.onchange = () => {
            const nextCount = Math.max(1, Math.round(Number(fixedInput.value || 1)));
            setWidget(node, "fixed_take_count", nextCount);
            if (!String(node?.properties?.iamccs_selected_takes_csv || "").trim()) {
                setSelectedTakesCsv(node, Array.from({ length: Math.min(maxTakes(node), nextCount) }, (_, index) => String(index + 1)).join(","));
            }
            render();
        };
        const selectedInput = document.createElement("input");
        selectedInput.type = "text";
        selectedInput.value = selectedTakesCsv(node);
        selectedInput.placeholder = "1,2";
        selectedInput.title = "Queue only these takes. Examples: 1,2 or 1,3,5";
        selectedInput.onchange = () => {
            setSelectedTakesCsv(node, selectedInput.value || "1");
            render();
        };
        const layoutSelect = select([["collapse_to_lane_1", "Send active chunk as A1"], ["preserve_bus_tracks", "Preserve bus lane"]], widget(node, "take_track_layout")?.value || "collapse_to_lane_1", (value) => setWidget(node, "take_track_layout", value));
        grid.append(
            field("Chunk template", templateSelect),
            field("Custom sec", secondsInput),
            field("Active timeline", activeSelect),
            field("Source bus", sourceSelect),
            field("Take source", sourceMode),
            field("Take count", countMode),
            field("Fixed takes", fixedInput),
            field("Selected takes", selectedInput),
            field("Shotboard audio layout", layoutSelect),
        );

        const takeRow = document.createElement("div");
        takeRow.className = "iamccs-mtb-takes";
        for (let take = 1; take <= Math.min(5, max); take += 1) {
            const card = document.createElement("button");
            card.type = "button";
            card.className = `iamccs-mtb-take${take === active ? " is-active" : ""}`;
            card.innerHTML = `<strong>T${String(take).padStart(2, "0")} / A${take}</strong><span>${take === active ? "prepared now" : "click to prepare"}</span>`;
            card.onclick = () => {
                setTake(node, take);
                render();
            };
            takeRow.appendChild(card);
        }

        const actions = document.createElement("div");
        actions.className = "iamccs-mtb-actions";
        const prepare = document.createElement("button");
        prepare.type = "button";
        prepare.className = "is-primary";
        prepare.textContent = `Prepare T${String(active).padStart(2, "0")}`;
        prepare.onclick = () => {
            setTake(node, active);
            render();
        };
        const auto = document.createElement("button");
        auto.type = "button";
        auto.textContent = "Auto From T Lanes";
        auto.onclick = () => {
            setWidget(node, "take_source_mode", "auto_detect_multi_lanes");
            setWidget(node, "source_bus", "master_out");
            setWidget(node, "take_count_mode", "auto_from_audio");
            setWidget(node, "take_track_layout", "collapse_to_lane_1");
            setSelectedTakesCsv(node, "");
            setTake(node, 1);
            render();
        };
        const queueSeq = document.createElement("button");
        queueSeq.type = "button";
        queueSeq.textContent = "Queue Sequence / 1 Backend";
        queueSeq.title = "Queues T1/A1, T2/A2, etc. by changing this bridge active_take on the same single backend workflow.";
        actions.append(prepare, auto, queueSeq);

        const ledger = document.createElement("div");
        ledger.className = "iamccs-mtb-ledger";
        ledger.textContent = `Active T${String(active).padStart(2, "0")} / A${active} | selected queue ${selectedTakeList(node).map((take) => `T${String(take).padStart(2, "0")}/A${take}`).join(" -> ")} | Contract: T1=A1, T2=A2, T3=A3.`;
        queueSeq.onclick = async () => {
            queueSeq.disabled = true;
            queueSeq.classList.add("is-primary");
            try {
                const countMode = String(widget(node, "take_count_mode")?.value || "auto_from_audio");
                const desired = countMode === "fixed_take_count"
                    ? Math.max(1, Math.round(num(widget(node, "fixed_take_count")?.value, 3)))
                    : maxTakes(node);
                const selected = selectedTakeList(node);
                setSelectedTakesCsv(node, selected.join(","));
                await queueSingleBackendSequence(node, desired, ledger, selected);
            } catch (err) {
                ledger.textContent = `Queue sequence failed: ${err?.message || err}`;
                console.warn("[IAMCCS MultiTimelineBridge UI] single-backend queue failed", err);
            } finally {
                queueSeq.disabled = false;
                queueSeq.classList.remove("is-primary");
            }
        };
        root.append(head, grid, takeRow, actions, ledger);
    };

    render();
    const uiWidget = node.addDOMWidget("IAMCCS MultiTimeline Bridge UI", "iamccs_multitimeline_bridge_ui", root, { serialize: false });
    uiWidget.computeSize = () => [BRIDGE_FIXED_SIZE[0] - 24, BRIDGE_WIDGET_HEIGHT];
    installFixedBridgeNode(node);
    console.info("[IAMCCS MultiTimelineBridge UI] installed", { nodeId: node?.id, reason });
}

app.registerExtension({
    name: "IAMCCS.MultiTimelineBridgeStableUI",
    setup() {
        [600, 1600, 3600].forEach((delay) => setTimeout(() => {
            const nodes = Array.isArray(app?.graph?._nodes) ? app.graph._nodes : [];
            nodes.forEach((node) => installBridgeUI(node, `scan+${delay}`));
        }, delay));
    },
    nodeCreated(node) {
        [0, 200, 700].forEach((delay) => setTimeout(() => installBridgeUI(node, `nodeCreated+${delay}`), delay));
    },
    loadedGraphNode(node) {
        [0, 200, 700].forEach((delay) => setTimeout(() => installBridgeUI(node, `loadedGraphNode+${delay}`), delay));
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "IAMCCS_MultiTimelineBridge") return;
        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            originalOnNodeCreated?.apply(this, arguments);
            setTimeout(() => installBridgeUI(this, "prototype.onNodeCreated"), 0);
        };
    },
});
