import { app } from "../../scripts/app.js";

console.log("[IAMCCS LTX2] Loading ExtensionModule presets...");

const PRESETS = {
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

function ensureTopPresetWidget(node) {
    // Create a visible preset control at the top, without changing serialization order.
    // This avoids breaking older workflows that rely on widgets_values ordering.
    const existingTop = getWidget(node, "preset_ui");
    if (existingTop) return;

    const backendPreset = getWidget(node, "preset");
    node.properties = node.properties || {};
    const saved = node.properties.iamccs_extension_preset;
    const initial = backendPreset?.value ?? saved ?? "custom";
    const values = [
        "custom",
        "cut_bestofk_16",
        "cut_bestofk_16_luma",
        "cut_bestofk_32",
        "micro_crossfade_3",
    ];

    // Insert widget at the very top.
    const w = node.addWidget(
        "combo",
        "Preset",
        initial,
        (v) => {
            node.properties.iamccs_extension_preset = String(v || "custom");
            // Sync backend widget if present.
            if (backendPreset) {
                backendPreset.value = v;
                try {
                    backendPreset.callback?.(v, app.canvas, node);
                } catch {
                    // ignore
                }
            }
            // Apply overrides (updates other widgets live)
            applyPreset(node, v);
        },
        { values }
    );
    w.name = "preset_ui";
    w.serialize = false;

    // Move it to index 0 in the widgets array (top of UI)
    try {
        const idx = getWidgetIndex(node, "preset_ui");
        if (idx > 0) {
            const [item] = node.widgets.splice(idx, 1);
            node.widgets.unshift(item);
        }
    } catch {
        // ignore
    }

    // Hide the backend preset widget if it exists (avoid duplicate controls).
    if (backendPreset) hideWidget(backendPreset);

    // Apply immediately for loaded workflows.
    applyPreset(node, w.value);
}

app.registerExtension({
    name: "iamccs.ltx2.extension.presets",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        const nodeName = nodeData?.name;
        if (nodeName !== "IAMCCS_LTX2_ExtensionModule" && nodeName !== "IAMCCS_LTX2_ExtensionModule_simple") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const node = this;

            // Always install top preset widget (even if backend widget isn't present yet).
            ensureTopPresetWidget(node);

            return r;
        };
    },
});
