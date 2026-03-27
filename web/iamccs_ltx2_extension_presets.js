import { app } from "../../scripts/app.js";

const IAMCCS_LTX2_EXTENSION_PRESETS_VERSION = "2026-03-17-1";

console.log(
    `[IAMCCS LTX2] Loading ExtensionModule presets... v=${IAMCCS_LTX2_EXTENSION_PRESETS_VERSION}`
);

const PRESETS = {
    // Target workflow parity: matches `tarket extension ltx2.json` stitch settings
    // - overlap = 10
    // - overlap_side = source
    // - overlap_mode = linear_blend
    // - start_images = last (overlap-1) frames from the overlap window
    target_extension_ltx2: {
        overlap_frames: 10,
        overlap_mode: "linear_blend",
        overlap_side: "source",

        // Match the original graph math: num_frames = overlap - 1
        enable_math: true,
        math_operation: "a-1",
        math_value_b: 1,
        safe_mode: "none",
        start_frames_rule: "none",

        // Keep quality upgrades disabled (parity)
        seam_search_mode: "none",
        k_search: 0,
        color_match_mode: "none",
        color_match_strength: 0.0,
        color_reference_window: 8,

        // Keep defaults explicit for stability
        metric_weight_color: 1.0,
        metric_weight_edges: 0.5,
    },

    // Recommended current settings for concert / videoclip audio-sync workflows.
    videoclip_audio_24fps: {
        overlap_frames: 9,
        overlap_mode: "cut",
        overlap_side: "source",

        enable_math: true,
        math_operation: "none",
        math_value_b: 1,
        safe_mode: "none",
        start_frames_rule: "none",

        seam_search_mode: "best_of_k",
        k_search: 16,
        color_match_mode: "luma_only",
        color_match_strength: 0.25,
        color_reference_window: 8,

        metric_weight_color: 1.0,
        metric_weight_edges: 0.5,
    },

    monologue_audio_24fps: {
        overlap_frames: 13,
        overlap_mode: "cut",
        overlap_side: "source",

        enable_math: true,
        math_operation: "none",
        math_value_b: 1,
        safe_mode: "none",
        start_frames_rule: "none",

        seam_search_mode: "best_of_k",
        k_search: 16,
        color_match_mode: "luma_only",
        color_match_strength: 0.15,
        color_reference_window: 8,

        metric_weight_color: 1.0,
        metric_weight_edges: 0.5,
    },

    // Prova 1: cut seam + best_of_k (no crossfade)
    cut_bestofk_16: {
        overlap_frames: 10,
        overlap_mode: "cut",
        overlap_side: "new_images",
        seam_search_mode: "best_of_k",
        k_search: 16,
        color_match_mode: "none",
        color_match_strength: 0.0,
        color_reference_window: 8,
    },

    // Prova 2: cut seam + luma match
    cut_bestofk_16_luma: {
        overlap_frames: 10,
        overlap_mode: "cut",
        overlap_side: "new_images",
        seam_search_mode: "best_of_k",
        k_search: 16,
        color_match_mode: "luma_only",
        color_match_strength: 0.25,
        color_reference_window: 8,
    },

    // Prova 3: stronger seam search window
    cut_bestofk_32: {
        overlap_frames: 16,
        overlap_mode: "cut",
        overlap_side: "new_images",
        seam_search_mode: "best_of_k",
        k_search: 32,
        color_match_mode: "none",
        color_match_strength: 0.0,
        color_reference_window: 8,
    },

    // Alternative: minimal crossfade duration
    micro_crossfade_3: {
        overlap_frames: 3,
        overlap_mode: "perceptual_crossfade",
        overlap_side: "source",
        seam_search_mode: "none",
        k_search: 0,
        color_match_mode: "none",
        color_match_strength: 0.0,
        color_reference_window: 8,
    },
};

console.log(`[IAMCCS LTX2] Extension presets keys: ${Object.keys(PRESETS).join(", ")}`);

function getWidget(node, name) {
    return node?.widgets?.find((w) => w?.name === name);
}

function getWidgetIndex(node, name) {
    if (!node?.widgets?.length) return -1;
    return node.widgets.findIndex((w) => w?.name === name);
}

function setWidgetValue(node, name, value) {
    const widget = getWidget(node, name);
    if (!widget) return false;

    widget.value = value;

    // Try to trigger downstream UI logic if present.
    try {
        if (typeof widget.callback === "function") {
            widget.callback(value, app.canvas, node);
        }
    } catch {
        // ignore
    }

    return true;
}

function applyPreset(node, presetKey) {
    const key = String(presetKey || "custom");
    if (key === "custom") return;

    const cfg = PRESETS[key];
    if (!cfg) return;

    for (const [name, value] of Object.entries(cfg)) {
        setWidgetValue(node, name, value);
    }

    try {
        node.setDirtyCanvas(true, true);
    } catch {
        // ignore
    }
}

function hideWidget(widget) {
    if (!widget) return;
    // ComfyUI/LiteGraph widgets don't have a universal "hidden" API.
    // These hints work in most builds.
    try {
        widget.hidden = true;
        widget.disabled = true;
        widget.computeSize = () => [0, -4];
    } catch {
        // ignore
    }
}

const PRESET_UI_WIDGET = "preset_ui";

const ALLOWED_PRESET_KEYS = [
    "custom",
    "target_extension_ltx2",
    "videoclip_audio_24fps",
    "monologue_audio_24fps",
    "cut_bestofk_16",
    "cut_bestofk_16_luma",
    "cut_bestofk_32",
    "micro_crossfade_3",
];

const ALLOWED_MATH_OPERATION = [
    "none",
    "a-b",
    "a-1",
    "a+b",
    "a*b",
    "a/b",
    "min(a,b)",
    "max(a,b)",
];

const ALLOWED_SAFE_MODE = ["none", "native_workflow_safe"];

function _coerceInt(value, fallback) {
    const n = Number(value);
    if (!Number.isFinite(n)) return fallback;
    return Math.trunc(n);
}

function _sanitizeNodeValues(node) {
    // If a workflow was saved while widgets_values were index-shifted,
    // ComfyUI may restore impossible values (e.g. strings into INT widgets).
    // This causes prompt validation to fail and prevents the node from running.
    let changed = false;

    const setCombo = (name, allowed, def) => {
        const w = getWidget(node, name);
        if (!w) return;
        const v = String(w.value ?? "");
        if (!allowed.includes(v)) {
            w.value = def;
            changed = true;
        }
    };

    const setInt = (name, def, min, max) => {
        const w = getWidget(node, name);
        if (!w) return;
        const n = _coerceInt(w.value, def);
        const clamped = Math.max(min, Math.min(max, n));
        if (w.value !== clamped) {
            w.value = clamped;
            changed = true;
        }
    };

    // Fix the exact fields seen in prompt validation errors.
    setInt("math_value_b", 1, 0, 256);
    setCombo("math_operation", ALLOWED_MATH_OPERATION, "a-b");
    setCombo("safe_mode", ALLOWED_SAFE_MODE, "none");
    setCombo("preset", ALLOWED_PRESET_KEYS, "custom");

    if (changed) {
        try {
            node.setDirtyCanvas(true, true);
        } catch {
            // ignore
        }
    }
}

function _removePresetUiWidget(node) {
    const idx = getWidgetIndex(node, PRESET_UI_WIDGET);
    if (idx < 0) return null;
    try {
        const [item] = node.widgets.splice(idx, 1);
        return item || null;
    } catch {
        return null;
    }
}

function _movePresetUiWidgetToTop(node) {
    try {
        const idx = getWidgetIndex(node, PRESET_UI_WIDGET);
        if (idx > 0) {
            const [item] = node.widgets.splice(idx, 1);
            node.widgets.unshift(item);
        }
    } catch {
        // ignore
    }
}

function _createPresetUiWidgetAtEnd(node) {
    const existing = getWidget(node, PRESET_UI_WIDGET);
    if (existing) return existing;

    const backendPreset = getWidget(node, "preset");
    node.properties = node.properties || {};
    const saved = node.properties.iamccs_extension_preset;
    const initial = String(backendPreset?.value ?? saved ?? "custom");
    const values = [
        "custom",
        "target_extension_ltx2",
        "videoclip_audio_24fps",
        "monologue_audio_24fps",
        "cut_bestofk_16",
        "cut_bestofk_16_luma",
        "cut_bestofk_32",
        "micro_crossfade_3",
    ];

    const w = node.addWidget(
        "combo",
        "Preset",
        initial,
        (v) => {
            const key = String(v || "custom");
            node.properties = node.properties || {};
            node.properties.iamccs_extension_preset = key;

            // Sync backend widget if present.
            const backend = getWidget(node, "preset");
            if (backend) {
                backend.value = key;
                try {
                    backend.callback?.(key, app.canvas, node);
                } catch {
                    // ignore
                }
            }

            // Apply overrides (updates other widgets live)
            applyPreset(node, key);
        },
        { values }
    );

    w.name = PRESET_UI_WIDGET;
    w.serialize = false;

    // Hide the backend preset widget if it exists (avoid duplicate controls).
    if (backendPreset) hideWidget(backendPreset);

    return w;
}

app.registerExtension({
    name: "iamccs.ltx2.extension.presets",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        const nodeName = nodeData?.name;
        if (nodeName !== "IAMCCS_LTX2_ExtensionModule" && nodeName !== "IAMCCS_LTX2_ExtensionModule_simple") return;

        const serialize = nodeType.prototype.serialize;
        nodeType.prototype.serialize = function () {
            // Strip the UI-only widget while serializing to avoid any chance
            // of widgets_values index shifting across ComfyUI/LiteGraph builds.
            const node = this;
            const removed = _removePresetUiWidget(node);
            const r = serialize?.apply(this, arguments);
            if (removed) {
                node.widgets.push(removed);
                _movePresetUiWidgetToTop(node);
            } else {
                _createPresetUiWidgetAtEnd(node);
                _movePresetUiWidgetToTop(node);
            }
            return r;
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const node = this;

            // Create the UI-only preset widget at the end.
            // DO NOT move to top here: on loaded workflows, configure() will run after onNodeCreated,
            // and any widget index shift before widgets_values restore can corrupt persistence.
            _removePresetUiWidget(node);
            _createPresetUiWidgetAtEnd(node);

            // For brand-new nodes (no configure path), move it to top in a microtask.
            queueMicrotask(() => {
                try {
                    if (!node._iamccs_ltx2_did_configure) {
                        _movePresetUiWidgetToTop(node);
                    }
                } catch {
                    // ignore
                }
            });

            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            const node = this;
            node._iamccs_ltx2_did_configure = true;

            // widgets_values have been restored at this point.
            const backendPreset = getWidget(node, "preset");
            node.properties = node.properties || {};
            const restored = String(backendPreset?.value ?? node.properties.iamccs_extension_preset ?? "custom");
            node.properties.iamccs_extension_preset = restored;

            if (backendPreset) backendPreset.value = restored;

            // Guard against previously-corrupted widgets_values causing invalid prompt inputs.
            _sanitizeNodeValues(node);

            _removePresetUiWidget(node);
            const ui = _createPresetUiWidgetAtEnd(node);
            ui.value = restored;
            _movePresetUiWidgetToTop(node);

            // Apply preset AFTER restore so UI reflects backend-enforced behavior.
            applyPreset(node, restored);

            return r;
        };
    },
});
