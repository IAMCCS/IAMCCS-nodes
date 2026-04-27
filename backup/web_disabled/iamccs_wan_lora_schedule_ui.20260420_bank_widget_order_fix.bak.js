import { app } from "../../scripts/app.js";

const NODE_NAME = "IAMCCS_WanLoRASchedule";
const NODE_NAMES = new Set([
    "IAMCCS_WanLoRASchedule",
    "IAMCCS_WanLoRAHookSchedule",
    "IAMCCS_ApplyScheduledWanLoRAFromConditioning",
    "IAMCCS_BuildScheduledWanModelBank",
]);
const MODE_WIDGET = "_iamccs_schedule_mode";
const LOAD_WIDGET = "_iamccs_schedule_load_preset";
const SAVE_WIDGET = "_iamccs_schedule_save_preset";
const ADD_WIDGET = "_iamccs_schedule_add_slot";
const DELETE_WIDGET = "_iamccs_schedule_delete_slot";
const RESET_WIDGET = "_iamccs_schedule_reset_rules";
const MODE_PROP = "iamccs_schedule_ui_mode";
const SLOTS_PROP = "iamccs_schedule_visible_slots";
const AUTO_PROP = "iamccs_schedule_auto_synced";
const INIT_PROP = "iamccs_schedule_inited_slots";
const DROP_HOVER_TIMEOUT_MS = 250;
const PRESET_KIND = "iamccs_wan_lora_schedule_preset";
const PRESET_VERSION = 1;
const MAX_SLOTS = 64;
const LEGACY_MAX_SLOTS = 6;
const UI_WIDGET_COUNT = 4;
const EXPECTED_SERIALIZED_VALUES = 3 + (MAX_SLOTS * 5);
const LEGACY_EXPECTED_SERIALIZED_VALUES = 3 + (LEGACY_MAX_SLOTS * 5);
const SIMPLE_WIDTH = 480;
const ADVANCED_WIDTH = 620;
const BASE_HEIGHT = 230;
const SIMPLE_SLOT_HEIGHT = 54;
const ADVANCED_SLOT_HEIGHT = 132;
const EXTRA_HEIGHT = 30;
const SLOT_BLOCK = 5;
const SLOTS_START_IDX = 3;
const VISIBLE_PRESET_OPTIONS = ["custom range", "all generations", "gen 0 only", "gen 1 onwards"];

const SLOT_META = [
    { slot: 1, label: "Gen 0", preset: "gen 0 only", start: 0, end: 0 },
    { slot: 2, label: "Gen 1", preset: "custom range", start: 1, end: 1 },
    { slot: 3, label: "Gen 2+", preset: "custom range", start: 2, end: -1 },
];

function getGraph() {
    return app?.canvas?.getCurrentGraph?.() || app?.graph || null;
}

function getGraphLink(graph, linkId) {
    if (!graph || linkId == null) return null;
    try {
        if (typeof graph.links?.get === "function") return graph.links.get(linkId) || graph.links.get(String(linkId)) || null;
        return graph.links?.[linkId] || graph.links?.[String(linkId)] || null;
    } catch {
        return null;
    }
}

function getGraphNode(graph, nodeId) {
    if (!graph || nodeId == null) return null;
    try {
        return graph.getNodeById?.(nodeId) || null;
    } catch {
        return null;
    }
}

function getWidget(node, name) {
    return node?.widgets?.find((widget) => widget?.name === name || widget?.label === name) || null;
}

function setButtonText(widget, text) {
    if (!widget) return;
    widget.label = text;
    widget.options = { ...(widget.options || {}), label: text };
    widget._iamccsButtonText = text;
}

function getButtonText(widget, fallback = "") {
    return String(widget?._iamccsButtonText || widget?.label || widget?.options?.label || fallback);
}

function waitForNextPaint() {
    return new Promise((resolve) => {
        if (typeof requestAnimationFrame === "function") {
            requestAnimationFrame(() => setTimeout(resolve, 0));
            return;
        }
        setTimeout(resolve, 0);
    });
}

async function runButtonAction(node, widget, busyText, action) {
    if (!widget || widget._iamccsBusy) return;
    const originalText = getButtonText(widget);
    widget._iamccsBusy = true;
    setButtonText(widget, busyText);
    node.setDirtyCanvas?.(true, true);
    await waitForNextPaint();
    try {
        await action();
    } finally {
        widget._iamccsBusy = false;
        setButtonText(widget, originalText);
        node.setDirtyCanvas?.(true, true);
    }
}

function beginBatchUpdate(node) {
    node._iamccsBatchDepth = Number(node._iamccsBatchDepth || 0) + 1;
}

function endBatchUpdate(node) {
    node._iamccsBatchDepth = Math.max(0, Number(node._iamccsBatchDepth || 0) - 1);
    if (!node._iamccsBatchDepth && node._iamccsPendingSyncLowNodes) {
        node._iamccsPendingSyncLowNodes = false;
        requestSyncAllLinkedLowNodes(node);
    }
}

function isBatchUpdating(node) {
    return Number(node?._iamccsBatchDepth || 0) > 0;
}

function isRestoringNodeState(node) {
    return Boolean(node?._iamccsRestoringState);
}

function requestSyncAllLinkedLowNodes(node) {
    if (!node) return;
    if (isBatchUpdating(node)) {
        node._iamccsPendingSyncLowNodes = true;
        return;
    }
    if (isRestoringNodeState(node)) {
        node._iamccsPendingSyncLowNodes = true;
        return;
    }
    if (node._iamccsSyncLowNodesTimer) return;
    node._iamccsSyncLowNodesTimer = setTimeout(() => {
        node._iamccsSyncLowNodesTimer = 0;
        void syncAllLinkedLowNodes(node).finally(() => {
            node.setDirtyCanvas?.(true, true);
        });
    }, 0);
}

function markUiWidgetNonSerializable(widget) {
    if (!widget) return widget;
    widget.options = { ...(widget.options || {}), serialize: false };
    widget.serializeValue = () => undefined;
    widget._iamccsUiOnly = true;
    return widget;
}

function isModeValue(value) {
    return value === "simple" || value === "advanced";
}

function isModelTypeValue(value) {
    return value === "wan2x" || value === "flow" || value === "standard";
}

// Correct serialized count: generation_index(1) + log_prefix(1) + model_type(1) + 6 slots × 5 = 33
// (generation_index stays as a widget even when wired, in ComfyUI's serialization model)
const EXPECTED_SERIALIZED_VALUES_CONNECTED = EXPECTED_SERIALIZED_VALUES; // 33 – gen_idx present as wired widget

function slotCountFromSerializedLength(length) {
    if (typeof length !== "number" || length < 3) return 0;
    return Math.max(0, Math.floor((length - 3) / SLOT_BLOCK));
}

function sanitizePresetValue(value) {
    const raw = String(value || "custom range");
    const aliases = {
        manual_range: "custom range",
        even_generations: "custom range",
        odd_generations: "custom range",
        every_2_from_start: "custom range",
        every_3_from_start: "custom range",
        "even gens (0,2,4...)": "custom range",
        "odd gens (1,3,5...)": "custom range",
        "every 2nd gen": "custom range",
        "every 3rd gen": "custom range",
    };
    return aliases[raw] || raw;
}

function normalizeLegacyWidgetValues(config) {
    const values = config?.widgets_values;
    if (!Array.isArray(values)) return;

    let modeValue = null;
    let normalized = null;

    // Pattern A (37 values): pre-fix format where 4 UI widgets were serialized at front,
    // gen_idx at [-2] and log_prefix at [-1].
    // "simple"/null/null/null  model_type  ...30 slots...  gen_idx  log_prefix
    if (
        values.length === LEGACY_EXPECTED_SERIALIZED_VALUES + UI_WIDGET_COUNT
        && isModeValue(values[0])
        && isModelTypeValue(values[UI_WIDGET_COUNT])
    ) {
        modeValue = values[0];
        const generationIndex = values[values.length - 2] ?? 0;
        const logPrefix = values[values.length - 1] ?? "WAN LoRA schedule";
        const modelType = values[UI_WIDGET_COUNT] ?? "flow";
        const slotValues = values.slice(UI_WIDGET_COUNT + 1, -2);
        // Correct order: gen_idx  log_prefix  model_type  ...30 slots...
        normalized = [generationIndex, logPrefix, modelType, ...slotValues];

    // Pattern B (33 values): intermediate format where model_type was widget[0],
    // gen_idx at [-2] and log_prefix at [-1].
    // model_type  ...30 slots...  gen_idx  log_prefix
    } else if (
        values.length === LEGACY_EXPECTED_SERIALIZED_VALUES
        && isModelTypeValue(values[0])
        && typeof values[values.length - 1] === "string"
    ) {
        const generationIndex = values[values.length - 2] ?? 0;
        const logPrefix = values[values.length - 1] ?? "WAN LoRA schedule";
        const modelType = values[0] ?? "flow";
        const slotValues = values.slice(1, -2);
        normalized = [generationIndex, logPrefix, modelType, ...slotValues];
    }

    // Determine what to work on: use normalized if we reordered, else work on original (in-place)
    // for preset-migration-only case (already-correct 33-value arrays with legacy preset names).
    const inferredSlots = slotCountFromSerializedLength(values.length);
    const target = normalized ?? (inferredSlots > 0 ? values : null);
    if (!target) return;

    // Migrate legacy internal preset names → current _PRESET_OPTIONS names.
    // "manual_range" was an internal alias for "custom range" in early builds.
    // ComfyUI server REJECTS it as an invalid combo value because it is not in _PRESET_OPTIONS.
    let presetPatched = false;
    for (let s = 0; s < slotCountFromSerializedLength(target.length); s++) {
        const presetIdx = SLOTS_START_IDX + s * SLOT_BLOCK + 2;
        const raw = target[presetIdx];
        const normalizedPreset = sanitizePresetValue(raw);
        if (raw != null && normalizedPreset !== raw) {
            target[presetIdx] = normalizedPreset;
            presetPatched = true;
        }
    }

    if (!normalized && !presetPatched) return; // nothing changed

    if (normalized) {
        config.widgets_values = normalized;
    } else if (presetPatched) {
        // values was modified in-place above (same array reference)
        config.widgets_values = values;
    }
    config.properties = config.properties || {};
    if (isModeValue(modeValue) && !config.properties[MODE_PROP]) {
        config.properties[MODE_PROP] = modeValue;
    }
}

function hideWidget(widget) {
    if (!widget) return;
    // Use `in` check so we save the REAL original only once, never overwrite with [0,0]
    if (!("_iamccsOriginalComputeSize" in widget)) {
        widget._iamccsOriginalComputeSize = widget.computeSize ?? null;
    }
    widget.hidden = true;
    widget.disabled = true;
    widget.computeSize = () => [0, 0];
}

function showWidget(widget) {
    if (!widget) return;
    widget.hidden = false;
    widget.disabled = false;
    if (widget._iamccsOriginalComputeSize) {
        widget.computeSize = widget._iamccsOriginalComputeSize;
    } else {
        delete widget.computeSize;
    }
}

function setWidgetValue(node, name, value, options = {}) {
    const { invokeCallback = true } = options;
    const widget = getWidget(node, name);
    if (!widget) return false;
    widget.value = value;
    if (invokeCallback && !isBatchUpdating(node)) {
        try {
            widget.callback?.(value, app.canvas, node);
        } catch {}
    }
    return true;
}

function ensureBoxShape(node) {
    try {
        if (typeof LiteGraph !== "undefined" && LiteGraph?.BOX_SHAPE != null) {
            node.shape = LiteGraph.BOX_SHAPE;
        } else {
            node.shape = 0;
        }
    } catch {}
}

function getVisibleSlots(node) {
    node.properties = node.properties || {};
    const raw = Number(node.properties[SLOTS_PROP] ?? 3);
    return Math.max(3, Math.min(MAX_SLOTS, Math.trunc(raw) || 3));
}

function setVisibleSlots(node, value) {
    node.properties = node.properties || {};
    node.properties[SLOTS_PROP] = Math.max(3, Math.min(MAX_SLOTS, Math.trunc(Number(value) || 3)));
}

function slotMeta(slot) {
    const base = SLOT_META.find((item) => item.slot === slot);
    if (base) return base;
    return { slot, label: `Gen ${slot - 1}`, preset: "custom range", start: slot - 1, end: slot - 1 };
}

function ensureSlotDefaults(node, slot) {
    node.properties = node.properties || {};
    node.properties[INIT_PROP] = node.properties[INIT_PROP] || {};
    if (node.properties[INIT_PROP][slot]) return; // already set; don't clobber user values
    const meta = slotMeta(slot);
    const prefix = `slot_${String(slot).padStart(2, "0")}`;
    setWidgetValue(node, `${prefix}_preset`, meta.preset);
    setWidgetValue(node, `${prefix}_start`, meta.start);
    setWidgetValue(node, `${prefix}_end`, meta.end);
    const strengthWidget = getWidget(node, `${prefix}_strength`);
    if (strengthWidget && (strengthWidget.value == null || Number(strengthWidget.value) === 0)) {
        strengthWidget.value = 1;
    }
    node.properties[INIT_PROP][slot] = true;
}

function ensureVisibleDefaults(node) {
    for (let slot = 1; slot <= getVisibleSlots(node); slot += 1) {
        ensureSlotDefaults(node, slot);
    }
}

function resetScheduleRules(node) {
    node.properties = node.properties || {};
    node.properties[INIT_PROP] = {};
    node.properties[AUTO_PROP] = {};
    setVisibleSlots(node, 3);
    beginBatchUpdate(node);
    for (let slot = 1; slot <= MAX_SLOTS; slot += 1) {
        const meta = slotMeta(slot);
        const prefix = `slot_${String(slot).padStart(2, "0")}`;
        setWidgetValue(node, `${prefix}_lora_name`, "no", { invokeCallback: false });
        setWidgetValue(node, `${prefix}_strength`, 1, { invokeCallback: false });
        setWidgetValue(node, `${prefix}_preset`, meta.preset, { invokeCallback: false });
        setWidgetValue(node, `${prefix}_start`, meta.start, { invokeCallback: false });
        setWidgetValue(node, `${prefix}_end`, meta.end, { invokeCallback: false });
        node.properties[INIT_PROP][slot] = true;
    }
    endBatchUpdate(node);
    clearDropHoverSlot(node);
}

function notifyScheduleUi(message, isError = false, node = null, options = {}) {
    const prefix = "[IAMCCS_WanLoRASchedule UI]";
    if (isError) {
        console.error(prefix, message);
    } else {
        console.info(prefix, message);
    }
    if (node) {
        showNodeToast(node, message, { isError, ...options });
    }
}

function showNodeToast(node, message, options = {}) {
    if (!node || !message) return;
    const { isError = false, anchorWidgetName = null, durationMs = 1800 } = options;
    node._iamccsToast = {
        message: String(message),
        isError: Boolean(isError),
        anchorWidgetName,
        until: Date.now() + Math.max(800, Number(durationMs) || 1800),
    };
    node.setDirtyCanvas?.(true, true);
}

function drawNodeToast(node, ctx) {
    const toast = node?._iamccsToast;
    if (!toast?.message) return;
    if ((Number(toast.until) || 0) <= Date.now()) {
        delete node._iamccsToast;
        node.setDirtyCanvas?.(true, true);
        return;
    }

    const message = toast.message.length > 72 ? `${toast.message.slice(0, 69)}...` : toast.message;
    const anchorWidget = toast.anchorWidgetName ? getWidget(node, toast.anchorWidgetName) : null;
    const fallbackWidth = Number(node?.size?.[0]) || SIMPLE_WIDTH;
    const toastHeight = 24;
    const paddingX = 10;
    const textWidth = Math.ceil(ctx.measureText(message).width);
    const toastWidth = Math.min(Math.max(textWidth + (paddingX * 2), 110), Math.max(120, fallbackWidth - 12));

    let anchorY = 18;
    if (anchorWidget) {
        anchorY = getWidgetBodyY(node, anchorWidget) + (getWidgetHeight(anchorWidget) / 2);
    }

    const x = Math.max(6, fallbackWidth - toastWidth - 8);
    const maxY = Math.max(8, (Number(node?.size?.[1]) || 200) - toastHeight - 8);
    const y = Math.min(Math.max(8, anchorY - (toastHeight / 2)), maxY);

    ctx.save();
    ctx.fillStyle = toast.isError ? "rgba(120, 24, 24, 0.96)" : "rgba(24, 30, 36, 0.96)";
    ctx.strokeStyle = toast.isError ? "rgba(255, 126, 126, 0.95)" : "rgba(110, 214, 156, 0.95)";
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    ctx.roundRect(x, y, toastWidth, toastHeight, 8);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#f5f7fa";
    ctx.font = "12px sans-serif";
    ctx.textBaseline = "middle";
    ctx.fillText(message, x + paddingX, y + (toastHeight / 2) + 0.5);
    ctx.restore();
}

function isAbortError(error) {
    return error?.name === "AbortError" || /aborted/i.test(String(error?.message || ""));
}

function sanitizeFilename(value) {
    return String(value || "")
        .trim()
        .replace(/[<>:"/\\|?*\x00-\x1F]+/g, "_")
        .replace(/\s+/g, "_")
        .replace(/^_+|_+$/g, "")
        .slice(0, 120);
}

function getScheduleSlotRecord(node, slot) {
    const meta = slotMeta(slot);
    const prefix = `slot_${String(slot).padStart(2, "0")}`;
    return {
        slot,
        lora_name: String(getWidget(node, `${prefix}_lora_name`)?.value || "no"),
        strength: Number(getWidget(node, `${prefix}_strength`)?.value ?? 1) || 1,
        preset: sanitizePresetValue(getWidget(node, `${prefix}_preset`)?.value || meta.preset),
        start: Math.trunc(Number(getWidget(node, `${prefix}_start`)?.value ?? meta.start) || meta.start),
        end: Math.trunc(Number(getWidget(node, `${prefix}_end`)?.value ?? meta.end) || meta.end),
    };
}

function buildSchedulePreset(node) {
    const mode = isModeValue(node?.properties?.[MODE_PROP]) ? node.properties[MODE_PROP] : "simple";
    const modelType = String(getWidget(node, "model_type")?.value || "flow");
    const logPrefix = String(getWidget(node, "log_prefix")?.value || "WAN LoRA schedule");
    return {
        kind: PRESET_KIND,
        version: PRESET_VERSION,
        node_type: node?.type || NODE_NAME,
        exported_at: new Date().toISOString(),
        mode,
        visible_slots: getVisibleSlots(node),
        model_type: isModelTypeValue(modelType) ? modelType : "flow",
        log_prefix: logPrefix,
        slots: Array.from({ length: MAX_SLOTS }, (_, index) => getScheduleSlotRecord(node, index + 1)),
    };
}

function normalizeLoadedSlotRecord(rawSlot, fallbackSlot) {
    const slotNum = Math.max(1, Math.min(MAX_SLOTS, Math.trunc(Number(rawSlot?.slot ?? fallbackSlot) || fallbackSlot)));
    const meta = slotMeta(slotNum);
    return {
        slot: slotNum,
        lora_name: String(rawSlot?.lora_name || rawSlot?.loraName || "no"),
        strength: Number(rawSlot?.strength ?? 1) || 1,
        preset: sanitizePresetValue(rawSlot?.preset || meta.preset),
        start: Math.trunc(Number(rawSlot?.start ?? meta.start) || meta.start),
        end: Math.trunc(Number(rawSlot?.end ?? meta.end) || meta.end),
    };
}

async function applySchedulePreset(node, payload) {
    if (!payload || typeof payload !== "object") {
        throw new Error("Preset file is empty or invalid.");
    }
    if (payload.kind && payload.kind !== PRESET_KIND) {
        throw new Error(`Unsupported preset kind: ${payload.kind}`);
    }

    const slotMap = new Map();
    if (Array.isArray(payload.slots)) {
        for (let index = 0; index < payload.slots.length; index += 1) {
            const normalized = normalizeLoadedSlotRecord(payload.slots[index], index + 1);
            slotMap.set(normalized.slot, normalized);
        }
    } else if (payload.slots && typeof payload.slots === "object") {
        for (const [key, value] of Object.entries(payload.slots)) {
            const numericSlot = Math.trunc(Number(String(key).replace(/[^0-9]/g, "")) || 0);
            if (!numericSlot) continue;
            const normalized = normalizeLoadedSlotRecord(value, numericSlot);
            slotMap.set(normalized.slot, normalized);
        }
    }

    const nextMode = isModeValue(payload.mode) ? payload.mode : "simple";
    const nextVisibleSlots = Math.max(3, Math.min(MAX_SLOTS, Math.trunc(Number(payload.visible_slots ?? payload.visibleSlots) || 3)));
    const nextModelType = isModelTypeValue(payload.model_type || payload.modelType) ? String(payload.model_type || payload.modelType) : "flow";
    const nextLogPrefix = String(payload.log_prefix || payload.logPrefix || "WAN LoRA schedule");

    beginBatchUpdate(node);
    resetScheduleRules(node);
    node.properties = node.properties || {};
    node.properties[MODE_PROP] = nextMode;
    setVisibleSlots(node, nextVisibleSlots);

    const modeWidget = getWidget(node, MODE_WIDGET);
    if (modeWidget) modeWidget.value = nextMode;
    const modelTypeWidget = getWidget(node, "model_type");
    if (modelTypeWidget) modelTypeWidget.value = nextModelType;
    const logPrefixWidget = getWidget(node, "log_prefix");
    if (logPrefixWidget) logPrefixWidget.value = nextLogPrefix;

    node.properties[INIT_PROP] = node.properties[INIT_PROP] || {};
    for (let slot = 1; slot <= MAX_SLOTS; slot += 1) {
        const record = slotMap.get(slot) || normalizeLoadedSlotRecord({}, slot);
        const prefix = `slot_${String(slot).padStart(2, "0")}`;
        const loraWidget = getWidget(node, `${prefix}_lora_name`);
        const strengthWidget = getWidget(node, `${prefix}_strength`);
        const presetWidget = getWidget(node, `${prefix}_preset`);
        const startWidget = getWidget(node, `${prefix}_start`);
        const endWidget = getWidget(node, `${prefix}_end`);
        if (loraWidget) loraWidget.value = record.lora_name;
        if (strengthWidget) strengthWidget.value = record.strength;
        if (presetWidget) presetWidget.value = record.preset;
        if (startWidget) startWidget.value = record.start;
        if (endWidget) endWidget.value = record.end;
        node.properties[INIT_PROP][slot] = true;
    }
    endBatchUpdate(node);

    clearDropHoverSlot(node);
    applyLayout(node);
    requestSyncAllLinkedLowNodes(node);
    node.setDirtyCanvas?.(true, true);
}

function getSuggestedPresetFilename(node) {
    const logPrefix = String(getWidget(node, "log_prefix")?.value || node?.title || "wan-lora-schedule");
    const filename = sanitizeFilename(logPrefix) || "wan-lora-schedule";
    return `${filename}.json`;
}

function downloadTextFile(filename, text) {
    const blob = new Blob([text], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.style.display = "none";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
}

async function saveSchedulePresetAs(node) {
    const preset = buildSchedulePreset(node);
    const text = JSON.stringify(preset, null, 2);
    const suggestedName = getSuggestedPresetFilename(node);

    if (typeof window !== "undefined" && typeof window.showSaveFilePicker === "function") {
        const handle = await window.showSaveFilePicker({
            suggestedName,
            types: [{
                description: "IAMCCS Wan LoRA Schedule preset",
                accept: { "application/json": [".json"] },
            }],
        });
        const writable = await handle.createWritable();
        await writable.write(text);
        await writable.close();
    } else {
        downloadTextFile(suggestedName, text);
    }

    notifyScheduleUi(`Preset saved: ${suggestedName}`, false, node, { anchorWidgetName: SAVE_WIDGET });
}

async function pickPresetTextFile() {
    if (typeof window !== "undefined" && typeof window.showOpenFilePicker === "function") {
        const [handle] = await window.showOpenFilePicker({
            multiple: false,
            types: [{
                description: "IAMCCS Wan LoRA Schedule preset",
                accept: { "application/json": [".json"] },
            }],
        });
        const file = await handle.getFile();
        return await file.text();
    }

    return await new Promise((resolve, reject) => {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = ".json,application/json";
        input.style.display = "none";
        input.addEventListener("change", async () => {
            try {
                const file = input.files?.[0];
                if (!file) {
                    reject(new DOMException("User aborted file selection.", "AbortError"));
                    return;
                }
                resolve(await file.text());
            } catch (error) {
                reject(error);
            } finally {
                input.remove();
            }
        }, { once: true });
        document.body.appendChild(input);
        input.click();
    });
}

async function loadSchedulePreset(node) {
    const text = await pickPresetTextFile();
    const payload = JSON.parse(text);
    await applySchedulePreset(node, payload);
    notifyScheduleUi("Preset loaded.", false, node, { anchorWidgetName: LOAD_WIDGET });
}

function applyLabels(node) {
    const modeWidget = getWidget(node, MODE_WIDGET);
    if (modeWidget) modeWidget.label = "Mode";
    const modelTypeWidget = getWidget(node, "model_type");
    if (modelTypeWidget) modelTypeWidget.label = "Model type";

    for (const input of node.inputs || []) {
        if (input?.name === "default_lora") input.label = "Always-on LoRA";
        if (input?.name === "linx") input.label = "Linx in";
        if (input?.name === "generation_index") input.label = "Generation";
    }
    for (const output of node.outputs || []) {
        if (output?.name === "linx") output.label = "Linx out";
    }

    for (let slot = 1; slot <= MAX_SLOTS; slot += 1) {
        const meta = slotMeta(slot);
        const prefix = `slot_${String(slot).padStart(2, "0")}`;
        const loraWidget = getWidget(node, `${prefix}_lora_name`);
        const strengthWidget = getWidget(node, `${prefix}_strength`);
        const presetWidget = getWidget(node, `${prefix}_preset`);
        const startWidget = getWidget(node, `${prefix}_start`);
        const endWidget = getWidget(node, `${prefix}_end`);
        if (loraWidget) loraWidget.label = `${meta.label} LoRA`;
        if (strengthWidget) strengthWidget.label = `${meta.label} Strength`;
        if (presetWidget) presetWidget.label = `${meta.label} Apply when`;
        if (startWidget) startWidget.label = `${meta.label} Start gen`;
        if (endWidget) endWidget.label = `${meta.label} End gen  (-1 = forever)`;
    }
}

function applyPresetOptions(node) {
    for (let slot = 1; slot <= MAX_SLOTS; slot += 1) {
        const presetWidget = getWidget(node, `slot_${String(slot).padStart(2, "0")}_preset`);
        if (!presetWidget) continue;
        presetWidget.options = { ...(presetWidget.options || {}), values: VISIBLE_PRESET_OPTIONS };
        const current = sanitizePresetValue(presetWidget.value);
        if (!VISIBLE_PRESET_OPTIONS.includes(current)) {
            presetWidget.value = "custom range";
        } else if (current !== presetWidget.value) {
            presetWidget.value = current;
        }
    }
}

function ensureModeWidget(node) {
    if (getWidget(node, MODE_WIDGET)) return;
    node.properties = node.properties || {};
    if (!node.properties[MODE_PROP]) {
        node.properties[MODE_PROP] = "simple";
    }
    const widget = node.addWidget(
        "combo",
        MODE_WIDGET,
        node.properties[MODE_PROP],
        (value) => {
            node.properties[MODE_PROP] = String(value || "simple");
            applyLayout(node);
            requestSyncAllLinkedLowNodes(node);
            node.setDirtyCanvas(true, true);
        },
        { values: ["simple", "advanced"], serialize: false }
    );
    widget.label = "Mode";
    markUiWidgetNonSerializable(widget);
}

function ensureActionWidgets(node) {
    if (!getWidget(node, LOAD_WIDGET)) {
        const widget = node.addWidget("button", "Load Preset", null, () => {
            void runButtonAction(node, widget, "Loading...", async () => {
                await loadSchedulePreset(node);
            }).catch((error) => {
                if (isAbortError(error)) return;
                notifyScheduleUi(`Load preset failed: ${error?.message || error}`, true, node, { anchorWidgetName: LOAD_WIDGET, durationMs: 2600 });
            });
        });
        widget.name = LOAD_WIDGET;
        setButtonText(widget, "Load Preset");
        markUiWidgetNonSerializable(widget);
    }
    if (!getWidget(node, SAVE_WIDGET)) {
        const widget = node.addWidget("button", "Save As...", null, () => {
            void runButtonAction(node, widget, "Saving...", async () => {
                await saveSchedulePresetAs(node);
            }).catch((error) => {
                if (isAbortError(error)) return;
                notifyScheduleUi(`Save preset failed: ${error?.message || error}`, true, node, { anchorWidgetName: SAVE_WIDGET, durationMs: 2600 });
            });
        });
        widget.name = SAVE_WIDGET;
        setButtonText(widget, "Save As...");
        markUiWidgetNonSerializable(widget);
    }
    if (!getWidget(node, ADD_WIDGET)) {
        const widget = node.addWidget("button", "+ Add Slot", null, () => {
            const newSlot = getVisibleSlots(node) + 1;
            node.properties[INIT_PROP] = node.properties[INIT_PROP] || {};
            delete node.properties[INIT_PROP][newSlot]; // allow fresh defaults for new slot
            setVisibleSlots(node, newSlot);
            ensureVisibleDefaults(node);
            applyLayout(node);
            requestSyncAllLinkedLowNodes(node);
            node.setDirtyCanvas(true, true);
        });
        widget.name = ADD_WIDGET;
        setButtonText(widget, "+ Add Slot");
        markUiWidgetNonSerializable(widget);
    }
    if (!getWidget(node, DELETE_WIDGET)) {
        const widget = node.addWidget("button", "- Delete Slot", null, () => {
            setVisibleSlots(node, getVisibleSlots(node) - 1);
            applyLayout(node);
            requestSyncAllLinkedLowNodes(node);
            node.setDirtyCanvas(true, true);
        });
        widget.name = DELETE_WIDGET;
        setButtonText(widget, "- Delete Slot");
        markUiWidgetNonSerializable(widget);
    }
    if (!getWidget(node, RESET_WIDGET)) {
        const widget = node.addWidget("button", "Reset Rules", null, () => {
            void runButtonAction(node, widget, "Resetting...", async () => {
                resetScheduleRules(node);
                applyLayout(node);
                requestSyncAllLinkedLowNodes(node);
                node.setDirtyCanvas(true, true);
                notifyScheduleUi("Rules reset.", false, node, { anchorWidgetName: RESET_WIDGET, durationMs: 1400 });
            });
        });
        widget.name = RESET_WIDGET;
        setButtonText(widget, "Reset Rules");
        markUiWidgetNonSerializable(widget);
    }
}

function getOutputIndexByName(node, name) {
    return (node?.outputs || []).findIndex((output) => output?.name === name);
}

function getInputIndexByName(node, name) {
    return (node?.inputs || []).findIndex((input) => input?.name === name);
}

function getLinkedLinxTargets(node) {
    const graph = getGraph();
    if (!graph) return [];
    const outIndex = getOutputIndexByName(node, "linx");
    if (outIndex < 0) return [];
    const output = node.outputs?.[outIndex];
    const links = Array.isArray(output?.links) ? output.links : [];
    const out = [];
    for (const linkId of links) {
        const link = getGraphLink(graph, linkId);
        const target = getGraphNode(graph, link?.target_id);
        if (target) out.push(target);
    }
    return out;
}

function candidateLowNames(name) {
    const source = String(name || "");
    if (!source || source === "no") return [];
    const pairs = [
        ["_HN_", "_LN_"],
        ["-HN_", "-LN_"],
        ["_HN-", "_LN-"],
        ["_HIGH_", "_LOW_"],
        ["-HIGH_", "-LOW_"],
        ["_HIGH-", "_LOW-"],
        ["HN", "LN"],
        ["Hn", "Ln"],
        ["hn", "ln"],
        ["HIGH", "LOW"],
        ["High", "Low"],
        ["high", "low"],
    ];
    const out = [];
    const seen = new Set();
    for (const [oldValue, newValue] of pairs) {
        if (!source.includes(oldValue)) continue;
        const candidate = source.replace(oldValue, newValue);
        if (!seen.has(candidate)) {
            out.push(candidate);
            seen.add(candidate);
        }
    }
    if (!seen.has(source)) {
        out.push(source);
    }
    return out;
}

function syncLowNodePresentation(highNode, lowNode) {
    lowNode.properties = lowNode.properties || {};
    lowNode.properties[MODE_PROP] = highNode?.properties?.[MODE_PROP] || "simple";
    setVisibleSlots(lowNode, getVisibleSlots(highNode));
    lowNode.flags = lowNode.flags || {};
    lowNode.flags.collapsed = !!highNode?.flags?.collapsed;
}

function normalizePathSeparators(value) {
    return String(value || "").replace(/\\/g, "/").replace(/^file:\/\//i, "").trim();
}

function basenameOfPath(value) {
    const normalized = normalizePathSeparators(value);
    const parts = normalized.split("/");
    return String(parts[parts.length - 1] || "").trim();
}

function collectDroppedLoraCandidates(event) {
    const out = [];
    const seen = new Set();
    const push = (value) => {
        const normalized = normalizePathSeparators(value);
        if (!normalized) return;
        const base = basenameOfPath(normalized);
        const variants = [normalized, base];
        for (const variant of variants) {
            const key = String(variant || "").trim();
            if (!key || !key.toLowerCase().endsWith(".safetensors") || seen.has(key)) continue;
            seen.add(key);
            out.push(key);
        }
    };

    const fileList = Array.from(event?.dataTransfer?.files || []);
    for (const file of fileList) {
        push(String(file?.name || ""));
        push(String(file?.path || ""));
    }

    const payloads = [
        String(event?.dataTransfer?.getData?.("text/uri-list") || ""),
        String(event?.dataTransfer?.getData?.("DownloadURL") || ""),
        String(event?.dataTransfer?.getData?.("text/x-moz-url") || ""),
        String(event?.dataTransfer?.getData?.("text/plain") || ""),
    ].filter(Boolean);

    for (const payload of payloads) {
        const lines = payload.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
        for (const line of lines) {
            const chunks = line.split(/\|/).map((part) => part.trim()).filter(Boolean);
            for (const chunk of chunks) {
                push(chunk);
            }
        }
    }

    return out;
}

function getOptionSetForNode(node) {
    const rawOpts = getWidget(node, "slot_01_lora_name")?.options?.values;
    if (Array.isArray(rawOpts) && rawOpts.length > 1) {
        return new Set(rawOpts);
    }
    if (_loraSetCache instanceof Set && _loraSetCache.size > 0) {
        return _loraSetCache;
    }
    return null;
}

function resolveDroppedLoraName(node, event) {
    const candidates = collectDroppedLoraCandidates(event);
    if (!candidates.length) return "";

    const optionSet = getOptionSetForNode(node);
    if (optionSet && optionSet.size > 0) {
        for (const candidate of candidates) {
            if (optionSet.has(candidate)) {
                return candidate;
            }
        }

        const optionEntries = Array.from(optionSet);
        for (const candidate of candidates) {
            const candidateBase = basenameOfPath(candidate).toLowerCase();
            const matched = optionEntries.find((entry) => basenameOfPath(entry).toLowerCase() === candidateBase);
            if (matched) {
                return matched;
            }
        }
    }

    return basenameOfPath(candidates[0]);
}

function isExternalFileDrag(event) {
    const dt = event?.dataTransfer;
    if (!dt) return false;

    try {
        const items = Array.from(dt.items || []);
        if (items.some((item) => item?.kind === "file")) {
            return true;
        }
    } catch {}

    try {
        const types = Array.from(dt.types || []).map((value) => String(value));
        return types.includes("Files") || types.includes("application/x-moz-file");
    } catch {}

    return false;
}

function hasDroppedLoraCandidate(event) {
    return collectDroppedLoraCandidates(event).length > 0;
}

// Returns {x, y} in the same coordinate space as widget.y / widget.last_y in LiteGraph.
// The canvas is translated to node.pos before drawNode/onDrawForeground, so the coordinate
// system for drawing starts at (0,0) = top-left of node (including title).
// widget.y is set by _arrangeWidgets and starts at NODE_TITLE_HEIGHT (approximately),
// so no extra correction is needed — just subtract node.pos to get node-local coords.
function resolveLocalDropPoint(node, event) {
    try {
        const canvas = app?.canvas?.canvas;
        const ds = app?.canvas?.ds;
        const rect = canvas?.getBoundingClientRect?.();
        if (rect && ds && Number.isFinite(event?.clientX) && Number.isFinite(event?.clientY)) {
            const scale = ds.scale || 1;
            const graphX = ((event.clientX - rect.left) / scale) - (ds.offset?.[0] || 0);
            const graphY = ((event.clientY - rect.top) / scale) - (ds.offset?.[1] || 0);
            return {
                x: graphX - (node?.pos?.[0] || 0),
                y: graphY - (node?.pos?.[1] || 0),
            };
        }
    } catch {}

    const explicitX = Number(event?.graphX ?? event?.canvasX ?? NaN);
    const explicitY = Number(event?.graphY ?? event?.canvasY ?? NaN);
    if (Number.isFinite(explicitX) && Number.isFinite(explicitY)) {
        return {
            x: explicitX - (node?.pos?.[0] || 0),
            y: explicitY - (node?.pos?.[1] || 0),
        };
    }

    const graphMouse = app?.canvas?.graph_mouse;
    if (Array.isArray(graphMouse) && graphMouse.length >= 2 && Number.isFinite(graphMouse[0]) && Number.isFinite(graphMouse[1])) {
        return {
            x: graphMouse[0] - (node?.pos?.[0] || 0),
            y: graphMouse[1] - (node?.pos?.[1] || 0),
        };
    }

    return null;
}

function resolveLocalDropY(node, event) {
    return resolveLocalDropPoint(node, event)?.y ?? null;
}

function getWidgetHeight(widget, nodeWidth) {
    // Use LiteGraph's already-computed height if available
    if (typeof widget?.computedHeight === "number" && Number.isFinite(widget.computedHeight) && widget.computedHeight > 0) {
        return widget.computedHeight;
    }
    try {
        const height = widget?.computeSize?.(nodeWidth)?.[1];
        if (typeof height === "number" && Number.isFinite(height)) return Math.max(0, height);
    } catch {}
    try {
        if (typeof LiteGraph !== "undefined" && LiteGraph?.NODE_WIDGET_HEIGHT != null) {
            return LiteGraph.NODE_WIDGET_HEIGHT;
        }
    } catch {}
    return 20;
}

// Get widget local-body Y using LiteGraph's own properties (set during arrange/draw)
// widget.y is set by _arrangeWidgets(), widget.last_y is set during drawWidgets().
// Both are relative to the node body (not including the title bar).
function getWidgetBodyY(widget) {
    // widget.y is set by _arrangeWidgets() — always > 0 for arranged widgets (title adds ~30px offset).
    // A value of 0 means the widget hasn't been arranged yet (default from addWidget), treat as unknown.
    const y = widget?.y;
    if (typeof y === "number" && Number.isFinite(y) && y > 0) return y;
    const ly = widget?.last_y;
    if (typeof ly === "number" && Number.isFinite(ly) && ly > 0) return ly;
    return null;
}

function getWidgetAtBodyY(node, bodyY) {
    for (const widget of node.widgets || []) {
        if (!widget || widget.hidden) continue;
        const wy = getWidgetBodyY(widget);
        if (wy === null) continue;
        const wh = getWidgetHeight(widget, node.size?.[0] || SIMPLE_WIDTH);
        if (bodyY >= wy && bodyY <= wy + wh) return widget;
    }
    return null;
}

function getSlotFromWidgetName(name) {
    const match = /^slot_(\d+)_/.exec(String(name || ""));
    return match ? Number(match[1]) : null;
}

function getWidgetLayoutRows(node) {
    const rows = [];
    const nodeWidth = node.size?.[0] || SIMPLE_WIDTH;
    for (const widget of node.widgets || []) {
        if (!widget || widget.hidden) continue;
        const y = getWidgetBodyY(widget);
        if (y === null) continue;
        const height = getWidgetHeight(widget, nodeWidth);
        rows.push({ widget, y, height });
    }
    // Sort by Y in case order is wrong
    rows.sort((a, b) => a.y - b.y);
    return rows;
}

function getSlotRect(node, slot) {
    if (!slot || slot < 1) return null;
    const rows = getWidgetLayoutRows(node);
    let minY = Infinity;
    let maxY = -Infinity;

    for (const row of rows) {
        const rowSlot = getSlotFromWidgetName(row.widget?.name);
        if (rowSlot !== slot) continue;
        minY = Math.min(minY, row.y);
        maxY = Math.max(maxY, row.y + row.height);
    }

    if (!Number.isFinite(minY) || !Number.isFinite(maxY)) return null;
    return {
        x: 8,
        y: minY - 2,
        width: Math.max(0, (node?.size?.[0] || SIMPLE_WIDTH) - 16),
        height: Math.max(0, maxY - minY + 4),
    };
}

function getFallbackEmptySlot(node) {
    const visibleSlots = getVisibleSlots(node);
    for (let slot = 1; slot <= visibleSlots; slot += 1) {
        const widget = getWidget(node, `slot_${String(slot).padStart(2, "0")}_lora_name`);
        if (widget && !widget.hidden && String(widget.value || "no") === "no") {
            return slot;
        }
    }
    return visibleSlots > 0 ? 1 : null;
}

function getDropTargetSlot(node, event, options = {}) {
    const { fallbackToEmpty = false } = options;
    const visibleSlots = getVisibleSlots(node);
    const localPoint = resolveLocalDropPoint(node, event);
    const localX = Number(localPoint?.x);
    const localY = Number(localPoint?.y);
    if (Number.isFinite(localY)) {
        const slotRects = [];
        for (let slot = 1; slot <= visibleSlots; slot += 1) {
            const rect = getSlotRect(node, slot);
            if (rect) slotRects.push({ slot, rect });
        }

        for (const { slot, rect } of slotRects) {
            const insideX = !Number.isFinite(localX) || (localX >= rect.x && localX <= rect.x + rect.width);
            if (insideX && localY >= rect.y && localY <= rect.y + rect.height) {
                return slot;
            }
        }

        const hit = getWidgetAtBodyY(node, localY);
        const hitSlot = getSlotFromWidgetName(hit?.name);
        if (hitSlot != null) return hitSlot;

        for (const row of getWidgetLayoutRows(node)) {
            if (localY >= row.y && localY <= row.y + row.height + 4) {
                const rowSlot = getSlotFromWidgetName(row.widget?.name);
                if (rowSlot != null) return rowSlot;
            }
        }

        if (slotRects.length > 0) {
            let bestSlot = slotRects[0].slot;
            let bestDistance = Infinity;
            for (const { slot, rect } of slotRects) {
                const centerY = rect.y + (rect.height / 2);
                const distance = Math.abs(localY - centerY);
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestSlot = slot;
                }
            }
            return bestSlot;
        }
    }
    return fallbackToEmpty ? getFallbackEmptySlot(node) : null;
}

function setDropHoverSlot(node, slot) {
    node._iamccsDropHoverSlot = Number(slot) || 0;
    node._iamccsDropHoverAt = Date.now();
}

function clearDropHoverSlot(node) {
    if (!node) return;
    if (!node._iamccsDropHoverSlot && !node._iamccsDropHoverAt) return;
    node._iamccsDropHoverSlot = 0;
    node._iamccsDropHoverAt = 0;
    node.setDirtyCanvas?.(true, true);
}

function drawDropHover(node, ctx) {
    const slot = Number(node?._iamccsDropHoverSlot) || 0;
    const at = Number(node?._iamccsDropHoverAt) || 0;
    if (!slot || !at) return;
    if ((Date.now() - at) > DROP_HOVER_TIMEOUT_MS) {
        node._iamccsDropHoverSlot = 0;
        node._iamccsDropHoverAt = 0;
        return;
    }

    const rect = getSlotRect(node, slot);
    if (!rect) return;

    ctx.save();
    ctx.fillStyle = "rgba(94, 181, 120, 0.14)";
    ctx.strokeStyle = "rgba(94, 181, 120, 0.95)";
    ctx.lineWidth = 2;
    ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
    ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
    ctx.restore();
}

function assignDroppedLora(node, event) {
    const loraName = resolveDroppedLoraName(node, event);
    if (!loraName) return false;

    const targetSlot = Number(node?._iamccsDropHoverSlot) || getDropTargetSlot(node, event, { fallbackToEmpty: true });
    const targetWidget = targetSlot != null
        ? getWidget(node, `slot_${String(targetSlot).padStart(2, "0")}_lora_name`)
        : null;
    if (!targetWidget) return false;

    const optionValues = Array.isArray(targetWidget?.options?.values) ? targetWidget.options.values : null;
    if (optionValues && !optionValues.includes(loraName)) {
        targetWidget.options.values = [loraName, ...optionValues];
    }

    targetWidget.value = loraName;
    try {
        targetWidget.callback?.(loraName, app.canvas, node);
    } catch {}
    requestSyncAllLinkedLowNodes(node);
    clearDropHoverSlot(node);
    node.setDirtyCanvas?.(true, true);
    return true;
}

// Cache of LoRA filenames from server API (loaded once).
let _loraSetCache = null;

async function getLoraSet() {
    if (_loraSetCache !== null) return _loraSetCache;
    try {
        const resp = await app.api.fetchApi("/object_info/IAMCCS_WanLoRASchedule");
        if (resp.ok) {
            const data = await resp.json();
            const req = data?.IAMCCS_WanLoRASchedule?.input?.required;
            const slot1 = req?.slot_01_lora_name;
            if (Array.isArray(slot1?.[0])) {
                _loraSetCache = new Set(slot1[0]);
                return _loraSetCache;
            }
        }
    } catch {}
    // Fallback: try widget.options.values from any lora_name combo on an existing node
    try {
        const nodeType = Object.keys(app.graph?._nodes_by_id || {});
        for (const id of nodeType) {
            const n = app.graph.getNodeById(id);
            if (!NODE_NAMES.has(n?.type)) continue;
            const w = getWidget(n, "slot_01_lora_name");
            const v = w?.options?.values ?? (Array.isArray(w?.options) ? w.options : null);
            if (Array.isArray(v) && v.length > 1) {
                _loraSetCache = new Set(v);
                return _loraSetCache;
            }
        }
    } catch {}
    _loraSetCache = new Set();
    return _loraSetCache;
}

async function syncSlotToLowNode(highNode, lowNode, slot, providedOptionSet = null) {
    const prefix = `slot_${String(slot).padStart(2, "0")}`;
    const highWidget = getWidget(highNode, `${prefix}_lora_name`);
    const lowWidget = getWidget(lowNode, `${prefix}_lora_name`);
    if (!highWidget || !lowWidget) return;

    lowNode.properties = lowNode.properties || {};
    lowNode.properties[AUTO_PROP] = lowNode.properties[AUTO_PROP] || {};
    const autoKey = `${prefix}_lora_name`;
    const lastAuto = String(lowNode.properties[AUTO_PROP][autoKey] || "");
    const lowValue = String(lowWidget.value || "no");
    const highValue = String(highWidget.value || "no");
    if (!highValue || highValue === "no") {
        if (lowValue === lastAuto || lowValue === "no") {
            lowWidget.value = "no";
            delete lowNode.properties[AUTO_PROP][autoKey];
            try {
                lowWidget.callback?.("no", app.canvas, lowNode);
            } catch {}
        }
        return;
    }

    const candidates = candidateLowNames(highValue);
    if (!candidates.length) return; // no _HIGH_/_LOW_ pattern in name

    // Build option set: prefer widget options, then server cache
    let optionSet = providedOptionSet;
    const rawOpts = lowWidget?.options?.values ?? (Array.isArray(lowWidget?.options) ? lowWidget.options : null);
    if (Array.isArray(rawOpts) && rawOpts.length > 1) {
        optionSet = new Set(rawOpts);
    } else {
        optionSet = await getLoraSet();
    }

    const suggested = (optionSet && optionSet.size > 0)
        ? candidates.find((c) => optionSet.has(c))
        : candidates[0]; // no list available: try first candidate blindly
    if (!suggested) return;
    // If we have a list and the candidate is not in it, don't set an invalid value
    if (optionSet && optionSet.size > 0 && !optionSet.has(suggested)) return;

    // Don't overwrite if user manually chose something other than "no" or our last auto-set value
    if (lowValue !== "no" && lowValue !== lastAuto) return;

    lowWidget.value = suggested;
    lowNode.properties[AUTO_PROP][autoKey] = suggested;
    try {
        lowWidget.callback?.(suggested, app.canvas, lowNode);
    } catch {}
}

function syncSlotSettingsToLowNode(highNode, lowNode, slot) {
    const prefix = `slot_${String(slot).padStart(2, "0")}`;
    const fieldNames = ["strength", "preset", "start", "end"];
    for (const fieldName of fieldNames) {
        const highWidget = getWidget(highNode, `${prefix}_${fieldName}`);
        const lowWidget = getWidget(lowNode, `${prefix}_${fieldName}`);
        if (!highWidget || !lowWidget) continue;
        lowWidget.value = highWidget.value;
        try {
            lowWidget.callback?.(highWidget.value, app.canvas, lowNode);
        } catch {}
    }
}

async function syncAllLinkedLowNodes(highNode) {
    const targets = getLinkedLinxTargets(highNode);
    if (!targets.length) return;
    const optionSet = await getLoraSet();
    for (const lowNode of targets) {
        syncLowNodePresentation(highNode, lowNode);
        ensureVisibleDefaults(lowNode);
        applyLayout(lowNode);
        for (let slot = 1; slot <= getVisibleSlots(highNode); slot += 1) {
            await syncSlotToLowNode(highNode, lowNode, slot, optionSet);
            syncSlotSettingsToLowNode(highNode, lowNode, slot);
        }
        applyLayout(lowNode);
        lowNode.setDirtyCanvas?.(true, true);
    }
}

function wrapSlotCallbacks(node) {
    if (node._iamccsScheduleWrappedCallbacks) return;
    for (let slot = 1; slot <= MAX_SLOTS; slot += 1) {
        const prefix = `slot_${String(slot).padStart(2, "0")}`;
        // Wrap lora_name → triggers linx sync
        const loraWidget = getWidget(node, `${prefix}_lora_name`);
        if (loraWidget && !loraWidget._iamccsWrapped) {
            const origLora = loraWidget.callback;
            loraWidget.callback = (value, canvas, targetNode) => {
                try { origLora?.(value, canvas, targetNode); } finally {
                    requestSyncAllLinkedLowNodes(node);
                }
            };
            loraWidget._iamccsWrapped = true;
        }
        // Wrap preset → refreshes start/end visibility
        const presetWidget = getWidget(node, `${prefix}_preset`);
        if (presetWidget && !presetWidget._iamccsWrapped) {
            const origPreset = presetWidget.callback;
            presetWidget.callback = (value, canvas, targetNode) => {
                try { origPreset?.(value, canvas, targetNode); } finally {
                    setTimeout(() => {
                        applyLayout(node);
                        requestSyncAllLinkedLowNodes(node);
                        node.setDirtyCanvas(true, true);
                    }, 0);
                }
            };
            presetWidget._iamccsWrapped = true;
        }
        for (const fieldName of ["strength", "start", "end"]) {
            const widget = getWidget(node, `${prefix}_${fieldName}`);
            if (!widget || widget._iamccsWrapped) continue;
            const origCallback = widget.callback;
            widget.callback = (value, canvas, targetNode) => {
                try { origCallback?.(value, canvas, targetNode); } finally {
                    requestSyncAllLinkedLowNodes(node);
                }
            };
            widget._iamccsWrapped = true;
        }
    }
    node._iamccsScheduleWrappedCallbacks = true;
}

function presetNeedsRange(preset) {
    const s = String(preset || "");
    return !s || s === "custom range" || s === "manual_range";
}

function setSlotVisibility(node, slot, visible, advanced) {
    const prefix = `slot_${String(slot).padStart(2, "0")}`;
    const loraWidget = getWidget(node, `${prefix}_lora_name`);
    const strengthWidget = getWidget(node, `${prefix}_strength`);
    const presetWidget = getWidget(node, `${prefix}_preset`);
    const startWidget = getWidget(node, `${prefix}_start`);
    const endWidget = getWidget(node, `${prefix}_end`);
    if (visible) {
        showWidget(loraWidget);
        showWidget(strengthWidget);
        if (advanced) {
            showWidget(presetWidget);
            const currentPreset = String(presetWidget?.value ?? "");
            if (presetNeedsRange(currentPreset)) {
                showWidget(startWidget);
                showWidget(endWidget);
            } else {
                hideWidget(startWidget);
                hideWidget(endWidget);
            }
        } else {
            hideWidget(presetWidget);
            hideWidget(startWidget);
            hideWidget(endWidget);
        }
    } else {
        hideWidget(loraWidget);
        hideWidget(strengthWidget);
        hideWidget(presetWidget);
        hideWidget(startWidget);
        hideWidget(endWidget);
    }
}

function reorderWidgets(node) {
    const orderedNames = [
        "generation_index",
        "log_prefix",
        "model_type",
        MODE_WIDGET,
        LOAD_WIDGET,
        SAVE_WIDGET,
        ADD_WIDGET,
        DELETE_WIDGET,
        RESET_WIDGET,
    ];
    for (let slot = 1; slot <= MAX_SLOTS; slot += 1) {
        const prefix = `slot_${String(slot).padStart(2, "0")}`;
        orderedNames.push(`${prefix}_lora_name`);
        orderedNames.push(`${prefix}_strength`);
        orderedNames.push(`${prefix}_preset`);
        orderedNames.push(`${prefix}_start`);
        orderedNames.push(`${prefix}_end`);
    }
    const byName = new Map((node.widgets || []).map((widget) => [widget?.name, widget]));
    const ordered = [];
    for (const name of orderedNames) {
        const widget = byName.get(name);
        if (widget) {
            ordered.push(widget);
            byName.delete(name);
        }
    }
    node.widgets = [...ordered, ...Array.from(byName.values())];
}

function targetWidth(node) {
    return (node?.properties?.[MODE_PROP] || "simple") === "advanced" ? ADVANCED_WIDTH : SIMPLE_WIDTH;
}

function targetHeight(node) {
    const mode = node?.properties?.[MODE_PROP] || "simple";
    const count = getVisibleSlots(node);
    const slotHeight = mode === "advanced" ? ADVANCED_SLOT_HEIGHT : SIMPLE_SLOT_HEIGHT;
    return BASE_HEIGHT + (count * slotHeight) + EXTRA_HEIGHT;
}

function applyLayout(node) {
    ensureBoxShape(node);
    ensureVisibleDefaults(node);
    applyLabels(node);
    applyPresetOptions(node);
    wrapSlotCallbacks(node);

    hideWidget(getWidget(node, "generation_index"));
    hideWidget(getWidget(node, "log_prefix"));

    const advanced = (node?.properties?.[MODE_PROP] || "simple") === "advanced";
    if (advanced) {
        showWidget(getWidget(node, "model_type"));
    } else {
        hideWidget(getWidget(node, "model_type"));
    }

    const visibleSlots = getVisibleSlots(node);
    for (let slot = 1; slot <= MAX_SLOTS; slot += 1) {
        setSlotVisibility(node, slot, slot <= visibleSlots, advanced);
    }

    reorderWidgets(node);
    const width = targetWidth(node);
    const height = targetHeight(node);
    const computed = node.computeSize?.() || node.size || [width, height];
    node.size = [Math.max(width, computed[0] || width), Math.max(height, computed[1] || height)];
}

app.registerExtension({
    name: "iamccs.wan_lora_schedule_ui",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_NAMES.has(nodeData?.name)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            this.properties = this.properties || {};
            if (!this.properties[MODE_PROP]) this.properties[MODE_PROP] = "simple";
            if (!this.properties[SLOTS_PROP]) this.properties[SLOTS_PROP] = 3;
            ensureModeWidget(this);
            ensureActionWidgets(this);
            setTimeout(() => {
                applyLayout(this);
                this.setDirtyCanvas(true, true);
            }, 0);
            return result;
        };

        const onDragOver = nodeType.prototype.onDragOver;
        nodeType.prototype.onDragOver = function (event) {
            if (isExternalFileDrag(event)) {
                try {
                    app?.canvas?.adjustMouseEvent?.(event);
                } catch {}
                const slot = getDropTargetSlot(this, event, { fallbackToEmpty: false });
                if (slot != null) {
                    setDropHoverSlot(this, slot);
                } else {
                    clearDropHoverSlot(this);
                }
                try {
                    event?.preventDefault?.();
                    event?.stopPropagation?.();
                    if (event?.dataTransfer) event.dataTransfer.dropEffect = "copy";
                } catch {}
                this.setDirtyCanvas?.(true, true);
                return true;
            }
            clearDropHoverSlot(this);
            return onDragOver?.apply(this, arguments);
        };

        const onDragDrop = nodeType.prototype.onDragDrop;
        nodeType.prototype.onDragDrop = function (event) {
            try {
                app?.canvas?.adjustMouseEvent?.(event);
            } catch {}
            try {
                event?.preventDefault?.();
                event?.stopPropagation?.();
            } catch {}
            if (hasDroppedLoraCandidate(event)) {
                assignDroppedLora(this, event);
                return true;
            }
            clearDropHoverSlot(this);
            return onDragDrop?.apply(this, arguments);
        };

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            const result = onDrawForeground?.apply(this, arguments);
            drawDropHover(this, ctx);
            drawNodeToast(this, ctx);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (config) {
            this._iamccsRestoringState = true;
            // 1. Normalize legacy corrupt widget-value arrays (modifies config in place)
            normalizeLegacyWidgetValues(config);

            // 2. Re-apply normalized values to widgets explicitly.
            //    LiteGraph assigns widgets_values → widget.value BEFORE calling onConfigure,
            //    so if the values were corrupt, they are already set wrong.  Re-assigning here
            //    (in the same iteration order, skipping non-serializable UI widgets) corrects them.
            if (Array.isArray(config.widgets_values) && this.widgets) {
                let j = 0;
                for (const widget of this.widgets) {
                    if (!widget || widget.options?.serialize === false || widget._iamccsUiOnly) continue;
                    if (j < config.widgets_values.length) {
                        widget.value = config.widgets_values[j++];
                    }
                }
            }

            const result = onConfigure?.apply(this, arguments);
            this.properties = this.properties || {};
            if (!this.properties[MODE_PROP]) this.properties[MODE_PROP] = "simple";
            if (!this.properties[SLOTS_PROP]) this.properties[SLOTS_PROP] = 3;
            ensureModeWidget(this);
            ensureActionWidgets(this);
            setTimeout(() => {
                // Mark ALL slots as already-initialized so ensureSlotDefaults won't clobber
                // values that were restored from the saved workflow.
                this.properties[INIT_PROP] = this.properties[INIT_PROP] || {};
                for (let s = 1; s <= MAX_SLOTS; s++) {
                    this.properties[INIT_PROP][s] = true;
                }
                applyLayout(this);
                this._iamccsRestoringState = false;
                this._iamccsPendingSyncLowNodes = false;
                this.setDirtyCanvas(true, true);
            }, 0);
            return result;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = onConnectionsChange?.apply(this, arguments);
            setTimeout(() => {
                applyLayout(this);
                if (!isRestoringNodeState(this)) {
                    requestSyncAllLinkedLowNodes(this);
                }
                this.setDirtyCanvas(true, true);
            }, 0);
            return result;
        };

        const collapse = nodeType.prototype.collapse;
        if (typeof collapse === "function") {
            nodeType.prototype.collapse = function () {
                const result = collapse?.apply(this, arguments);
                setTimeout(() => {
                    if (!isRestoringNodeState(this)) {
                        requestSyncAllLinkedLowNodes(this);
                    }
                    this.setDirtyCanvas(true, true);
                }, 0);
                return result;
            };
        }
    },
});
