// iamccs_wan_motion_presets.js
// ==========================================================
// IAMCCS WanImageMotion + WanImageMotionPro — Quick-Preset UI
// Injects a combo widget at the top of both nodes.
// Changing it updates all other widgets in real-time.
// serialize = false → never breaks saved workflows.
// ==========================================================

import { app } from "../../scripts/app.js";

console.log("[IAMCCS WanMotion] Loading motion presets...");

// ─── Preset definitions ────────────────────────────────────
// Keys must match widget *name* as declared in INPUT_TYPES.
// COMBO values must be the exact option strings used in Python.

const PRESETS_MOTION = {
    // ── no-op ──────────────────────────────────────────────
    "[custom]": null,

    // ── 1:1 con WanImageToVideoSVIProFLF ───────────────────
    // Zero amplitude processing. Use to verify reference parity.
    "parity 1:1": {
        motion_latent_count: 1,
        motion: 1.0,
        motion_mode: "motion_only (prev_samples)",
        add_reference_latents: false,
        latent_precision: "fp32",
        vram_profile: "normal",
        include_padding_in_motion: false,
        safety_preset: "base",
        lock_start_slots: 1,
        use_prev_samples: true,
    },

    // ── Diagnostic (no-op + extra logs) ────────────────────
    // Same as parity, but enables backend diagnostic_log.
    "diagnostic (no-op logs)": {
        motion_latent_count: 1,
        motion: 1.0,
        motion_mode: "motion_only (prev_samples)",
        add_reference_latents: false,
        latent_precision: "fp32",
        vram_profile: "normal",
        include_padding_in_motion: false,
        safety_preset: "base",
        lock_start_slots: 1,
        diagnostic_log: true,
        use_prev_samples: true,
    },

    // ── First segment helper ──────────────────────────────
    // For workflows where prev_samples is always connected by autolink.
    "first segment (ignore prev)": {
        motion_latent_count: 1,
        motion: 1.15,
        motion_mode: "motion_only (prev_samples)",
        add_reference_latents: false,
        latent_precision: "fp32",
        vram_profile: "normal",
        include_padding_in_motion: false,
        safety_preset: "safe",
        lock_start_slots: 1,
        use_prev_samples: false,
        diagnostic_log: true,
    },

    // ── Default recommended ─────────────────────────────────
    // Good for single-chunk generation with modest motion.
    standard: {
        motion_latent_count: 1,
        motion: 1.15,
        motion_mode: "motion_only (prev_samples)",
        add_reference_latents: false,
        latent_precision: "fp32",
        vram_profile: "normal",
        include_padding_in_motion: false,
        safety_preset: "safe",
        lock_start_slots: 1,
        use_prev_samples: true,
    },

    // ── Multi-chunk chaining ────────────────────────────────
    // include_padding_in_motion keeps the boost range consistent
    // even when prev_samples is connected but small.
    "smooth chain": {
        motion_latent_count: 1,
        motion: 1.3,
        motion_mode: "motion_only (prev_samples)",
        add_reference_latents: false,
        latent_precision: "fp32",
        vram_profile: "normal",
        include_padding_in_motion: false,
        safety_preset: "safer",
        lock_start_slots: 1,
        use_prev_samples: true,
    },

    // ── Large/dynamic motion ────────────────────────────────
    "high motion": {
        motion_latent_count: 1,
        motion: 1.6,
        motion_mode: "all_nonfirst (anchor+motion)",
        add_reference_latents: false,
        latent_precision: "fp32",
        vram_profile: "normal",
        include_padding_in_motion: false,
        safety_preset: "safer",
        // Keep the first frame anchored to the provided input.
        lock_start_slots: 1,
        use_prev_samples: true,
    },

    // ── Low VRAM safe ───────────────────────────────────────
    "low vram": {
        motion_latent_count: 1,
        motion: 1.15,
        motion_mode: "motion_only (prev_samples)",
        add_reference_latents: false,
        latent_precision: "fp32",
        vram_profile: "chunked_blocks_2",
        include_padding_in_motion: false,
        safety_preset: "safe",
        lock_start_slots: 1,
        use_prev_samples: true,
    },
};

const PRESETS_MOTION_PRO = {
    "[custom]": null,

    // ── 1:1 con WanImageToVideoSVIProFLF ───────────────────
    "parity 1:1": {
        motion_latent_count: 1,
        motion: 1.0,
        motion_mode: "motion_only (prev_samples)",
        add_reference_latents: false,
        latent_precision: "fp32",
        vram_profile: "normal",
        include_padding_in_motion: false,
        safety_preset: "base",
        // Match original FLF behavior: end_samples is honored and hard-locked.
        use_end_frame: true,
        // Original node has no end-transition blend.
        end_transition_frames: 0,
        // Match: end_t_fix = min(T_end, total_latents)
        end_lock_slots: 16,
        lock_start_slots: 1,
        use_prev_samples: true,
    },

    // ── Diagnostic (no-op + extra logs) ────────────────────
    // Same as parity, but enables backend diagnostic_log.
    "diagnostic (no-op logs)": {
        motion_latent_count: 1,
        motion: 1.0,
        motion_mode: "motion_only (prev_samples)",
        add_reference_latents: false,
        latent_precision: "fp32",
        vram_profile: "normal",
        include_padding_in_motion: false,
        safety_preset: "base",
        use_end_frame: true,
        end_transition_frames: 0,
        end_lock_slots: 16,
        lock_start_slots: 1,
        diagnostic_log: true,
        use_prev_samples: true,
    },

    // ── First segment helper ──────────────────────────────
    // For workflows where prev_samples is always connected by autolink.
    "first segment (ignore prev)": {
        motion_latent_count: 1,
        motion: 1.15,
        motion_mode: "motion_only (prev_samples)",
        add_reference_latents: false,
        latent_precision: "fp32",
        vram_profile: "normal",
        include_padding_in_motion: false,
        safety_preset: "safe",
        use_end_frame: true,
        end_transition_frames: 0,
        end_lock_slots: 1,
        lock_start_slots: 1,
        use_prev_samples: false,
        diagnostic_log: true,
    },

    // ── Start frame only (no end lock) ────────────────────
    "start only": {
        motion_latent_count: 1,
        motion: 1.15,
        motion_mode: "motion_only (prev_samples)",
        add_reference_latents: false,
        latent_precision: "fp32",
        vram_profile: "normal",
        include_padding_in_motion: false,
        safety_preset: "safe",
        // Always honor end_samples when connected (first/last reference).
        use_end_frame: true,
        end_transition_frames: 0,
        end_lock_slots: 1,
        lock_start_slots: 1,
        use_prev_samples: true,
    },

    // ── Standard FLF (start + end, 1 locked slot) ─────────
    "flf standard": {
        motion_latent_count: 1,
        motion: 1.15,
        motion_mode: "motion_only (prev_samples)",
        add_reference_latents: false,
        latent_precision: "fp32",
        vram_profile: "normal",
        include_padding_in_motion: false,
        safety_preset: "safe",
        use_end_frame: true,
        end_transition_frames: 0,
        end_lock_slots: 1,
        lock_start_slots: 1,
        use_prev_samples: true,
    },

    // ── Multi-chunk chain (no end lock) ───────────────────
    "smooth chain": {
        motion_latent_count: 1,
        motion: 1.3,
        motion_mode: "motion_only (prev_samples)",
        add_reference_latents: false,
        latent_precision: "fp32",
        vram_profile: "normal",
        include_padding_in_motion: false,
        safety_preset: "safer",
        // Always honor end_samples when connected (first/last reference).
        use_end_frame: true,
        end_transition_frames: 0,
        end_lock_slots: 1,
        lock_start_slots: 1,
        use_prev_samples: true,
    },

    // ── Smooth FLF chain (start + end + smooth transition) ─
    // end_transition_frames=4: safe for clips as short as 33F.
    // The Python backend additionally clamps trans_start >= T_anchor+1.
    "smooth flf": {
        motion_latent_count: 1,
        motion: 1.3,
        motion_mode: "motion_only (prev_samples)",
        add_reference_latents: false,
        latent_precision: "fp32",
        vram_profile: "normal",
        include_padding_in_motion: false,
        safety_preset: "safer",
        use_end_frame: true,
        end_transition_frames: 0,
        end_lock_slots: 1,
        lock_start_slots: 1,
        use_prev_samples: true,
    },

    // ── High motion FLF (big movement, longer convergence) ─
    // end_lock_slots=1 (not 2): avoids locking 8 extra video frames.
    // end_transition_frames=4: safe. Python guard prevents frozen-dissolve.
    "high motion flf": {
        motion_latent_count: 1,
        motion: 1.6,
        motion_mode: "all_nonfirst (anchor+motion)",
        add_reference_latents: false,
        latent_precision: "fp32",
        vram_profile: "normal",
        include_padding_in_motion: false,
        safety_preset: "safer",
        use_end_frame: true,
        end_transition_frames: 0,
        end_lock_slots: 1,
        // Keep the first frame anchored to the provided input.
        lock_start_slots: 1,
        use_prev_samples: true,
    },

    // ── Low VRAM + FLF ─────────────────────────────────────
    "low vram flf": {
        motion_latent_count: 1,
        motion: 1.15,
        motion_mode: "motion_only (prev_samples)",
        add_reference_latents: false,
        latent_precision: "fp32",
        vram_profile: "chunked_blocks_2",
        include_padding_in_motion: false,
        safety_preset: "safe",
        use_end_frame: true,
        end_transition_frames: 0,
        end_lock_slots: 1,
        lock_start_slots: 1,
        use_prev_samples: true,
    },
};

// ─── Utility helpers ───────────────────────────────────────

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

    // Fire callback so downstream UI updates (e.g. colour-coded combos).
    try {
        if (typeof widget.callback === "function") {
            widget.callback(value, app.canvas, node);
        }
    } catch {
        // ignore
    }

    return true;
}

function applyPreset(node, presetKey, presetMap) {
    const key = String(presetKey ?? "[custom]");
    if (key === "[custom]") return;

    const cfg = presetMap[key];
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

// ─── Top-preset widget injection ──────────────────────────

function ensureTopPresetWidget(node, presetMap) {
    // Guard: only once per node instance.
    if (getWidget(node, "_iamccs_preset_ui")) return;

    const presetKeys = Object.keys(presetMap);
    const initial = "[custom]";

    node.properties = node.properties || {};

    // Restore saved choice from node properties (survives Save→Load because
    // properties ARE serialized in ComfyUI, unlike widget values for non-backend widgets).
    const saved = node.properties._iamccs_wan_preset ?? initial;

    const w = node.addWidget(
        "combo",
        "⚡ Quick Preset",
        saved,
        (v) => {
            const chosen = String(v ?? "[custom]");
            node.properties._iamccs_wan_preset = chosen;
            applyPreset(node, chosen, presetMap);
            // After applying, reset the combo back to "[custom]" so users see
            // it as a "fire-once" action and are free to tweak further.
            // Comment out the next line if you prefer it to stay on the chosen value.
            // w.value = "[custom]";
        },
        { values: presetKeys }
    );

    // Critical: serialize=false prevents this widget from appearing in
    // widgets_values, keeping full backward-compat with saved workflows.
    w.serialize = false;
    w.name = "_iamccs_preset_ui";

    // Move to index 0 so it renders at the very top of the node.
    try {
        const idx = getWidgetIndex(node, "_iamccs_preset_ui");
        if (idx > 0) {
            const [item] = node.widgets.splice(idx, 1);
            node.widgets.unshift(item);
        }
    } catch {
        // ignore
    }

    // Apply saved preset immediately (for loaded workflows).
    if (saved !== "[custom]") {
        applyPreset(node, saved, presetMap);
    }
}

// ─── Extension registration ────────────────────────────────

app.registerExtension({
    name: "iamccs.wan.motion.presets",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        const nodeName = nodeData?.name;

        // IAMCCS_WanImageMotion — standard motion node
        if (nodeName === "IAMCCS_WanImageMotion") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated?.apply(this, arguments);
                ensureTopPresetWidget(this, PRESETS_MOTION);
                return r;
            };
            return;
        }

        // WanImageMotionPro — extended FLF node
        if (nodeName === "WanImageMotionPro") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated?.apply(this, arguments);
                ensureTopPresetWidget(this, PRESETS_MOTION_PRO);
                return r;
            };
        }
    },
});
