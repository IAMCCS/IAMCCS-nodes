// iamccs_wan_motion_presets.js
// ==========================================================
// IAMCCS WanImageMotion + WanImageMotionPro — Quick-Preset UI
// Injects a combo widget at the top of both nodes.
// Changing it updates all other widgets in real-time.
// serialize = false → never breaks saved workflows.
// ==========================================================

import { app } from "../../scripts/app.js";

console.log("[IAMCCS WanMotion] Loading motion presets... (persist-fix v3)");

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
        end_overshoot_slots: 0,
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
        end_overshoot_slots: 0,
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
        end_overshoot_slots: 0,
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
        use_end_frame: true,
        end_transition_frames: 0,
        end_lock_slots: 1,
        lock_start_slots: 1,
        use_prev_samples: true,
        end_overshoot_slots: 0,
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
        end_overshoot_slots: 0,
    },

    // ── FLF with overshoot (soft convergence, no freeze) ──
    // end_overshoot_slots=1: adds 4 hidden frames so the end-lock zone is trimmed
    // away after sampling. Wire trim_slots → Cut Latent Frames.
    "flf overshoot": {
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
        end_overshoot_slots: 1,
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
        use_end_frame: true,
        end_transition_frames: 0,
        end_lock_slots: 1,
        lock_start_slots: 1,
        use_prev_samples: true,
        end_overshoot_slots: 0,
    },

    // ── Smooth FLF chain ───────────────────────────────────
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
        end_overshoot_slots: 0,
    },

    // ── High motion FLF ────────────────────────────────────
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
        lock_start_slots: 1,
        use_prev_samples: true,
        end_overshoot_slots: 1,
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
        end_overshoot_slots: 0,
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

function _movePresetWidgetToTop(node) {
    try {
        const idx = getWidgetIndex(node, "_iamccs_preset_ui");
        if (idx > 0) {
            const [item] = node.widgets.splice(idx, 1);
            node.widgets.unshift(item);
        }
    } catch {
        // ignore
    }
}

function _removePresetWidget(node) {
    try {
        const idx = getWidgetIndex(node, "_iamccs_preset_ui");
        if (idx >= 0) {
            node.widgets.splice(idx, 1);
        }
    } catch {
        // ignore
    }
}

function _detachPresetWidget(node) {
    try {
        const idx = getWidgetIndex(node, "_iamccs_preset_ui");
        if (idx >= 0) {
            const [item] = node.widgets.splice(idx, 1);
            return item;
        }
    } catch {
        // ignore
    }
    return null;
}

function _attachPresetWidgetToTop(node, widget) {
    if (!widget) return;
    try {
        if (getWidget(node, "_iamccs_preset_ui")) return;
        node.widgets = node.widgets || [];
        node.widgets.unshift(widget);
    } catch {
        // ignore
    }
}

/**
 * Create and append the preset combo widget at the END of node.widgets.
 * Reads the saved label from node.properties (which configure() has already
 * restored from the workflow JSON by the time onConfigure fires).
 * Always creates a fresh widget — callers are responsible for removing any
 * stale instance first via _removePresetWidget().
 */
function _createPresetWidgetAtEnd(node, presetMap) {
    const presetKeys = Object.keys(presetMap);
    node.properties = node.properties || {};
    const savedLabel = node.properties._iamccs_wan_preset_label;
    const initial =
        typeof savedLabel === "string" && presetKeys.includes(savedLabel)
            ? savedLabel
            : "[custom]";

    const w = node.addWidget(
        "combo",
        "⚡ Quick Preset",
        initial,
        (v) => {
            const chosen = String(v ?? "[custom]");
            node.properties._iamccs_wan_preset_label = chosen;
            applyPreset(node, chosen, presetMap);
        },
        { values: presetKeys, serialize: false }
    );

    // Belt-and-suspenders: set serialize=false both on the widget and in options
    // so it is excluded from widgets_values regardless of which LiteGraph version
    // checks which property.
    w.serialize = false;
    w.name = "_iamccs_preset_ui";
    return w;
}

function _installPresetHooks(nodeType, presetMap) {
    // Idempotent: protect against multiple extension reloads.
    if (nodeType?.prototype?._iamccs_preset_hooks_installed) return;
    nodeType.prototype._iamccs_preset_hooks_installed = true;

    // ── onNodeCreated ──────────────────────────────────────────────────────────
    // Fires when a fresh node instance is constructed (both for new nodes dropped
    // from the menu AND for nodes recreated during workflow load before configure).
    // We add the preset widget at the END so it never sits at position 0 while
    // configure() / widgets_values assignment is running.
    // For the workflow-load path, onConfigure (below) will move it to the top
    // after widgets_values are safely restored.
    // For the "new node" path (no configure follows), we schedule a microtask to
    // move it to position 0 — the microtask runs after any synchronous configure
    // that may immediately follow (so it never races with widgets_values).
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const r = onNodeCreated?.apply(this, arguments);
        if (!getWidget(this, "_iamccs_preset_ui")) {
            _createPresetWidgetAtEnd(this, presetMap);
        }
        // Deferred: runs after any synchronous configure() that follows in the
        // same call stack (e.g. during graph.configure / loadGraphData).
        // onConfigure will have already moved it to top if a configure happened;
        // _movePresetWidgetToTop is a no-op when idx === 0.
        const self = this;
        Promise.resolve().then(() => {
            if (!getWidget(self, "_iamccs_preset_ui")) {
                _createPresetWidgetAtEnd(self, presetMap);
            }
            _movePresetWidgetToTop(self);
        });
        return r;
    };

    // ── onConfigure ───────────────────────────────────────────────────────────
    // LiteGraph calls onConfigure() at the END of LGraphNode.prototype.configure(),
    // AFTER widgets_values have been applied and AFTER node.properties has been
    // restored from the saved workflow JSON.
    // This is the canonical place to re-insert the preset widget: it is guaranteed
    // to fire after the real widget values are in place, so re-adding the widget
    // here can never corrupt the index mapping.
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (data) {
        const r = onConfigure?.apply(this, arguments);
        // Strip any existing preset widget (added by onNodeCreated at END, or
        // possibly left at position 0 from a previous configure cycle).
        _removePresetWidget(this);
        // Re-create at END, reading the fresh node.properties label.
        _createPresetWidgetAtEnd(this, presetMap);
        // Move to position 0 for the UX (preset combo always visible at top).
        _movePresetWidgetToTop(this);
        return r;
    };

    // ── serialize ─────────────────────────────────────────────────────────────
    // Belt-and-suspenders: strip the preset widget before serialization so it
    // never appears in widgets_values, even on LiteGraph builds that don't check
    // w.serialize === false consistently.
    const serialize = nodeType.prototype.serialize;
    nodeType.prototype.serialize = function () {
        const detached = _detachPresetWidget(this);
        const r = serialize?.apply(this, arguments);
        if (detached) {
            _attachPresetWidgetToTop(this, detached);
            _movePresetWidgetToTop(this);
        }
        return r;
    };
}

// Legacy alias — kept for any call sites that may still reference it.
function ensureTopPresetWidget(node, presetMap) {
    if (getWidget(node, "_iamccs_preset_ui")) return;
    _createPresetWidgetAtEnd(node, presetMap);
}

// ─── Extension registration ────────────────────────────────

app.registerExtension({
    name: "iamccs.wan.motion.presets",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        const nodeName = nodeData?.name;

        // IAMCCS_WanImageMotion — standard motion node
        if (nodeName === "IAMCCS_WanImageMotion") {
            _installPresetHooks(nodeType, PRESETS_MOTION);
            return;
        }

        // WanImageMotionPro — extended FLF node
        if (nodeName === "WanImageMotionPro") {
            _installPresetHooks(nodeType, PRESETS_MOTION_PRO);
        }
    },
});
