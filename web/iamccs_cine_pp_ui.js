// IAMCCS Cine-PP UI helpers
// Keeps AntiAlias preset changes visible in the ComfyUI node widgets.

import { app } from "../../scripts/app.js";

const AA_TYPES = new Set(["IAMCCS_CinePPAntiAlias", "IAMCCS_CineUTAntiAlias"]);
const APPLYING_PROP = "__iamccs_cinepp_aa_applying";
const HOOKED_PROP = "__iamccs_cinepp_aa_hooked";
const MANUAL_PROP = "__iamccs_cinepp_aa_manual";

const CONTROLLED_WIDGETS = [
    "edge_threshold",
    "blur_radius",
    "temporal_stability",
    "detail_protection",
    "chroma_cleanup",
    "max_frames_per_chunk",
];

const PRESETS = {
    Preview: { edge_threshold: 0.18, blur_radius: 1, temporal_stability: 0.00, detail_protection: 0.70, chroma_cleanup: 0.00, max_frames_per_chunk: 1 },
    Light: { edge_threshold: 0.14, blur_radius: 1, temporal_stability: 0.06, detail_protection: 0.60, chroma_cleanup: 0.05, max_frames_per_chunk: 0 },
    Balanced: { edge_threshold: 0.10, blur_radius: 1, temporal_stability: 0.12, detail_protection: 0.50, chroma_cleanup: 0.10, max_frames_per_chunk: 0 },
    Strong: { edge_threshold: 0.08, blur_radius: 2, temporal_stability: 0.18, detail_protection: 0.42, chroma_cleanup: 0.18, max_frames_per_chunk: 0 },
    "Crisp Lines": { edge_threshold: 0.07, blur_radius: 1, temporal_stability: 0.10, detail_protection: 0.82, chroma_cleanup: 0.12, max_frames_per_chunk: 0 },
    "Low VRAM": { edge_threshold: 0.13, blur_radius: 1, temporal_stability: 0.04, detail_protection: 0.65, chroma_cleanup: 0.04, max_frames_per_chunk: 2 },
};

function nodeClass(node) {
    return node?.comfyClass || node?.type || "";
}

function widget(node, name) {
    return (node?.widgets || []).find((w) => w?.name === name || w?.label === name) || null;
}

function isSentinelValue(name, value) {
    if (name === "blur_radius" || name === "max_frames_per_chunk") return Number(value) <= 0;
    return Number(value) < 0;
}

function choosePreset(node) {
    const preset = String(widget(node, "preset")?.value || "Auto");
    const hardware = String(widget(node, "hardware_mode")?.value || "auto");
    if (preset !== "Auto") return preset;
    if (hardware === "low_vram" || hardware === "cpu_safe") return "Low VRAM";
    if (hardware === "quality") return "Strong";
    return "Balanced";
}

function valuesFor(node) {
    const selected = choosePreset(node);
    const values = { ...(PRESETS[selected] || PRESETS.Balanced) };
    const hardware = String(widget(node, "hardware_mode")?.value || "auto");
    if (hardware === "low_vram" || hardware === "cpu_safe") {
        values.blur_radius = Math.min(Number(values.blur_radius) || 1, 1);
        values.temporal_stability = Math.min(Number(values.temporal_stability) || 0, 0.08);
        values.max_frames_per_chunk = Math.max(Number(values.max_frames_per_chunk) || 0, 2);
    }
    return { selected, values };
}

function setWidgetValue(node, name, value) {
    const w = widget(node, name);
    if (!w) return;
    w.value = value;
    try {
        w.callback?.(value);
    } catch {
        // Best effort UI sync only.
    }
}

function markDirty() {
    try {
        app.graph?.setDirtyCanvas?.(true, true);
    } catch {
        // ignore
    }
}

function applyPreset(node, { resetManual = false, force = false, onlySentinel = false } = {}) {
    if (!node || !AA_TYPES.has(nodeClass(node))) return;
    const manual = node[MANUAL_PROP] || new Set();
    if (resetManual) manual.clear();
    node[MANUAL_PROP] = manual;

    const { values } = valuesFor(node);
    node[APPLYING_PROP] = true;
    try {
        for (const name of CONTROLLED_WIDGETS) {
            const w = widget(node, name);
            if (!w) continue;
            if (!force && manual.has(name)) continue;
            if (onlySentinel && !isSentinelValue(name, w.value)) continue;
            setWidgetValue(node, name, values[name]);
        }
    } finally {
        node[APPLYING_PROP] = false;
    }
    markDirty();
}

function wrapCallback(node, name, fn) {
    const w = widget(node, name);
    if (!w || w.__iamccs_cinepp_wrapped) return;
    const original = w.callback;
    w.callback = function(value, ...args) {
        const result = original?.apply(this, [value, ...args]);
        fn(value, ...args);
        return result;
    };
    w.__iamccs_cinepp_wrapped = true;
}

function setupNode(node) {
    if (!node || !AA_TYPES.has(nodeClass(node)) || node[HOOKED_PROP]) return;
    node[HOOKED_PROP] = true;
    node[MANUAL_PROP] = new Set();

    wrapCallback(node, "preset", () => {
        applyPreset(node, { resetManual: true, force: true });
    });

    wrapCallback(node, "hardware_mode", () => {
        applyPreset(node, { resetManual: false, force: false, onlySentinel: false });
    });

    for (const name of CONTROLLED_WIDGETS) {
        wrapCallback(node, name, () => {
            if (node[APPLYING_PROP]) return;
            node[MANUAL_PROP].add(name);
        });
    }

    applyPreset(node, { resetManual: false, force: false, onlySentinel: true });
}

app.registerExtension({
    name: "iamccs.cinepp.antialias.ui",
    nodeCreated(node) {
        setupNode(node);
    },
    loadedGraphNode(node) {
        setupNode(node);
    },
});
