// IAMCCS Detail Atelier UI
// Frontend helper for live loop estimates and opt-in "Auto" on the Advanced LINX node.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const ADVANCED_TYPE = "IAMCCS_DetailAtelierAdvanced";
const SAMPLER_TYPE = "IAMCCS_DetailAtelierSampler";
const PREVIEW_WIDGET = "iamccs_detail_atelier_live_preview";
const APPLY_BUTTON = "iamccs_detail_atelier_apply_auto";
const REFRESH_BUTTON = "iamccs_detail_atelier_refresh";
const CONTROLS_WIDGET = "iamccs_detail_atelier_auto_controls";
const AUTO_ACTIVE_PROP = "iamccs_detail_atelier_auto_active";
const AUTO_HOOK_PROP = "__iamccs_detail_atelier_auto_hooked";
const AUTO_WATCH_PROP = "__iamccs_detail_atelier_auto_watch";
const AUTO_SIG_PROP = "__iamccs_detail_atelier_auto_sig";
const AUTO_CONTROLS_PROP = "__iamccs_detail_atelier_auto_controls";
const AUTO_SETTING_PROP = "__iamccs_detail_atelier_auto_setting";
const AUTO_MANUAL_OVERRIDES_PROP = "__iamccs_detail_atelier_manual_overrides";
const AUTO_MANUAL_HOOKED_PROP = "__iamccs_detail_atelier_manual_hooked";
const AUTO_ADV_VALUES_PROP = "__iamccs_detail_atelier_adv_values";

const AUTO_CONTROLLED_WIDGETS = [
    "enabled",
    "temporal_tile_size",
    "temporal_overlap",
    "guiding_strength",
    "temporal_overlap_cond_strength",
    "cond_image_strength",
    "horizontal_tiles",
    "vertical_tiles",
    "spatial_overlap",
    "spatial_tiling_mode",
];

const PRESETS = {
    "8GB": {
        preview: { temporal_tile_size: 32, temporal_overlap: 16, guiding_strength: 0.85, temporal_overlap_cond_strength: 0.45, cond_image_strength: 1.0 },
        balanced: { temporal_tile_size: 40, temporal_overlap: 16, guiding_strength: 0.90, temporal_overlap_cond_strength: 0.50, cond_image_strength: 1.0 },
        quality: { temporal_tile_size: 40, temporal_overlap: 16, guiding_strength: 0.95, temporal_overlap_cond_strength: 0.55, cond_image_strength: 1.0 },
    },
    "12GB": {
        preview: { temporal_tile_size: 40, temporal_overlap: 16, guiding_strength: 0.90, temporal_overlap_cond_strength: 0.45, cond_image_strength: 1.0 },
        balanced: { temporal_tile_size: 56, temporal_overlap: 24, guiding_strength: 1.00, temporal_overlap_cond_strength: 0.50, cond_image_strength: 1.0 },
        quality: { temporal_tile_size: 56, temporal_overlap: 24, guiding_strength: 1.00, temporal_overlap_cond_strength: 0.55, cond_image_strength: 1.0 },
    },
    "16GB": {
        preview: { temporal_tile_size: 48, temporal_overlap: 16, guiding_strength: 0.95, temporal_overlap_cond_strength: 0.45, cond_image_strength: 1.0 },
        balanced: { temporal_tile_size: 56, temporal_overlap: 24, guiding_strength: 1.00, temporal_overlap_cond_strength: 0.50, cond_image_strength: 1.0 },
        quality: { temporal_tile_size: 64, temporal_overlap: 24, guiding_strength: 1.00, temporal_overlap_cond_strength: 0.60, cond_image_strength: 1.0 },
    },
    "24GB": {
        preview: { temporal_tile_size: 56, temporal_overlap: 24, guiding_strength: 1.00, temporal_overlap_cond_strength: 0.45, cond_image_strength: 1.0 },
        balanced: { temporal_tile_size: 64, temporal_overlap: 24, guiding_strength: 1.00, temporal_overlap_cond_strength: 0.55, cond_image_strength: 1.0 },
        quality: { temporal_tile_size: 80, temporal_overlap: 32, guiding_strength: 1.00, temporal_overlap_cond_strength: 0.60, cond_image_strength: 1.0 },
    },
};

function nodeClass(node) {
    return node?.comfyClass || node?.type || "";
}

function widget(node, name) {
    return (node?.widgets || []).find((w) => w?.name === name || w?.label === name) || null;
}

function setWidget(node, name, value) {
    const w = widget(node, name);
    if (!w) return false;
    w.value = value;
    try {
        w.callback?.(value);
    } catch {
        // ignore
    }
    return true;
}

function clamp(v, min, max) {
    const n = Number(v);
    if (!Number.isFinite(n)) return min;
    return Math.max(min, Math.min(max, n));
}

function effectiveVramPreset(vramGb, fallback = "12GB") {
    const n = Number(vramGb);
    if (!Number.isFinite(n)) return fallback;
    if (n <= 8.5) return "8GB";
    if (n <= 12.5) return "12GB";
    if (n <= 16.5) return "16GB";
    return "24GB";
}

function getGraphNodeById(id) {
    try {
        return app.graph?.getNodeById?.(id) || app.graph?._nodes_by_id?.[id] || null;
    } catch {
        return null;
    }
}

function linkedTarget(node) {
    try {
        const out = node.outputs?.find((o) => o?.type === "IAMCCS_DETAIL_ATELIER_LINX") || node.outputs?.[0];
        const linkId = Array.isArray(out?.links) ? out.links[0] : null;
        if (linkId == null) return null;
        const link = app.graph?.links?.[linkId];
        if (!link) return null;
        const targetId = link.target_id ?? link[3];
        return getGraphNodeById(targetId);
    } catch {
        return null;
    }
}

function linkedAdvancedNode(node) {
    try {
        const inp = (node.inputs || []).find((i) => i?.type === "IAMCCS_DETAIL_ATELIER_LINX");
        const linkId = inp?.link;
        if (linkId == null) return null;
        const link = app.graph?.links?.[linkId];
        if (!link) return null;
        const sourceId = link.origin_id ?? link[1];
        return getGraphNodeById(sourceId);
    } catch {
        return null;
    }
}

function detailControlTarget(node) {
    const linked = linkedTarget(node);
    if (widget(linked, "quality_mode") || widget(linked, "vram_preset")) return linked;

    try {
        const nodes = app.graph?._nodes || [];
        const linkedSampler = nodes.find((n) => nodeClass(n) === SAMPLER_TYPE && linkedAdvancedNode(n) === node);
        if (linkedSampler) return linkedSampler;

        const detailNodes = nodes.filter((n) => {
            const cls = nodeClass(n);
            return cls.includes("DetailAtelier") && n !== node && (widget(n, "quality_mode") || widget(n, "vram_preset"));
        });
        if (detailNodes.length === 1) return detailNodes[0];
    } catch {
        // best effort only
    }
    return linked;
}

function targetSignature(target) {
    const quality = widget(target, "quality_mode")?.value || "balanced";
    const vram = widget(target, "vram_preset")?.value || "12GB";
    return `${quality}|${vram}`;
}

function readLoaderContext() {
    const ctx = { frames: null, fps: null, width: null, height: null };
    try {
        const nodes = app.graph?._nodes || [];
        const loader = nodes.find((n) => nodeClass(n) === "VHS_LoadVideoPath");
        const info = nodes.find((n) => nodeClass(n) === "VHS_VideoInfoLoaded");
        const scaler = nodes.find((n) => nodeClass(n) === "ImageScaleToMaxDimension");

        const loaderFrames = widget(loader, "frame_load_cap")?.value ?? loader?.widgets_values?.frame_load_cap;
        const loaderFps = widget(loader, "force_rate")?.value ?? loader?.widgets_values?.force_rate;
        ctx.frames = Number(loaderFrames) > 0 ? Number(loaderFrames) : null;
        ctx.fps = Number(loaderFps) > 0 ? Number(loaderFps) : null;

        const infoFps = widget(info, "fps")?.value ?? widget(info, "frame_rate")?.value;
        if (!ctx.fps && Number(infoFps) > 0) ctx.fps = Number(infoFps);

        const largest = widget(scaler, "largest_size")?.value ?? widget(scaler, "max_dimension")?.value;
        if (Number(largest) > 0) {
            ctx.width = Number(largest);
            ctx.height = Math.round(Number(largest) * 9 / 16);
        }
    } catch {
        // best effort only
    }
    return ctx;
}

function estimateChunks(frames, temporalTileSize, temporalOverlap, timeScale = 8) {
    const f = Number(frames);
    if (!Number.isFinite(f) || f <= 0) return null;
    const latentFrames = Math.max(1, Math.ceil(f / timeScale));
    const latentTile = Math.max(1, Math.floor(Number(temporalTileSize) / timeScale));
    let latentOverlap = Math.max(0, Math.floor(Number(temporalOverlap) / timeScale));
    if (latentOverlap >= latentTile) latentOverlap = Math.max(0, latentTile - 1);
    const step = Math.max(1, latentTile - latentOverlap);
    const chunks = latentFrames <= latentTile ? 1 : Math.ceil((latentFrames - latentTile) / step) + 1;
    return { chunks, latentFrames, latentTile, latentOverlap, step };
}

async function probeHardware(ctx) {
    try {
        const params = new URLSearchParams();
        if (ctx.width) params.set("width", String(Math.round(ctx.width)));
        if (ctx.height) params.set("height", String(Math.round(ctx.height)));
        if (ctx.frames) params.set("frames", String(Math.round(ctx.frames)));
        if (ctx.fps) params.set("fps", String(ctx.fps));
        const res = await api.fetchApi(`/api/iamccs/hw_probe?${params.toString()}`);
        if (!res.ok) return null;
        return await res.json();
    } catch {
        return null;
    }
}

function chooseAutoValues(hw, ctx, quality = "balanced", fallbackVram = "12GB") {
    const vramGb = hw?.hardware?.cuda_total_vram_gb ?? hw?.cuda_total_vram_gb ?? null;
    const ramGb = hw?.hardware?.system_ram_gb ?? hw?.system_ram_gb ?? null;
    const vram = effectiveVramPreset(vramGb, fallbackVram);
    const q = PRESETS[vram]?.[quality] ? quality : "balanced";
    const base = { ...(PRESETS[vram]?.[q] || PRESETS["12GB"].balanced) };

    const lowRam = Number(ramGb) > 0 && Number(ramGb) <= 40 && (vram === "8GB" || vram === "12GB");
    if (lowRam || vram === "8GB" || vram === "12GB") {
        base.horizontal_tiles = 1;
        base.vertical_tiles = 1;
        base.spatial_overlap = 1;
        base.spatial_tiling_mode = "force_off";
        if (vram === "12GB") {
            base.temporal_tile_size = Math.min(base.temporal_tile_size, 56);
            base.temporal_overlap = Math.min(base.temporal_overlap, 24);
        }
        if (vram === "8GB") {
            base.temporal_tile_size = Math.min(base.temporal_tile_size, 32);
            base.temporal_overlap = Math.min(base.temporal_overlap, 16);
        }
    } else {
        base.horizontal_tiles = 0;
        base.vertical_tiles = 0;
        base.spatial_overlap = 0;
        base.spatial_tiling_mode = "inherit";
    }

    const chunks = estimateChunks(ctx.frames, base.temporal_tile_size, base.temporal_overlap);
    return {
        vram,
        quality: q,
        vramGb,
        ramGb,
        lowRam,
        chunks,
        values: base,
    };
}

function ensurePreview(node) {
    let w = widget(node, PREVIEW_WIDGET);
    if (w) return w;
    w = node.addWidget("text", "Atelier Live", "", () => {}, { multiline: true });
    w.name = PREVIEW_WIDGET;
    w.serialize = false;
    try {
        w.inputEl?.setAttribute?.("readonly", true);
    } catch {
        // ignore
    }
    return w;
}

function setPreview(node, text) {
    const w = ensurePreview(node);
    w.value = text;
    try {
        app.graph.setDirtyCanvas(true, false);
    } catch {
        // ignore
    }
}

function autoIsActive(node) {
    return Boolean(node?.[AUTO_ACTIVE_PROP]);
}

function setAutoActive(node, active) {
    if (node) node[AUTO_ACTIVE_PROP] = Boolean(active);
}

function updatePreviewIdle(node) {
    try {
        if (node?.properties) delete node.properties[PREVIEW_WIDGET];
    } catch {
        // ignore
    }
    setPreview(
        node,
        [
            "Auto is idle.",
            "Press Auto to probe hardware and write Advanced loop overrides.",
            "Manual edits always override Auto values.",
        ].join("\n")
    );
}

function debounce(fn, delay = 120) {
    let timer = null;
    return (...args) => {
        if (timer) clearTimeout(timer);
        timer = setTimeout(() => fn(...args), delay);
    };
}

function manualOverrides(node) {
    if (!node[AUTO_MANUAL_OVERRIDES_PROP]) node[AUTO_MANUAL_OVERRIDES_PROP] = {};
    return node[AUTO_MANUAL_OVERRIDES_PROP];
}

function recordAdvancedValue(node, name) {
    if (!node[AUTO_ADV_VALUES_PROP]) node[AUTO_ADV_VALUES_PROP] = {};
    const w = widget(node, name);
    if (w) node[AUTO_ADV_VALUES_PROP][name] = w.value;
}

function setAutoWidget(node, name, value) {
    if (manualOverrides(node)[name]) return false;
    const ok = setWidget(node, name, value);
    if (ok) recordAdvancedValue(node, name);
    return ok;
}

function hookManualOverrides(node) {
    if (!node || node[AUTO_MANUAL_HOOKED_PROP]) return;
    for (const name of AUTO_CONTROLLED_WIDGETS) {
        const w = widget(node, name);
        if (!w) continue;
        const previous = w.callback;
        w.callback = function () {
            if (autoIsActive(node) && !node[AUTO_SETTING_PROP]) {
                manualOverrides(node)[name] = true;
                recordAdvancedValue(node, name);
            }
            return previous?.apply(this, arguments);
        };
    }
    node[AUTO_MANUAL_HOOKED_PROP] = true;
}

function scanManualOverrides(node) {
    if (!autoIsActive(node) || node[AUTO_SETTING_PROP]) return;
    const last = node[AUTO_ADV_VALUES_PROP] || {};
    for (const name of AUTO_CONTROLLED_WIDGETS) {
        const w = widget(node, name);
        if (!w) continue;
        if (Object.prototype.hasOwnProperty.call(last, name) && w.value !== last[name]) {
            manualOverrides(node)[name] = true;
        }
        recordAdvancedValue(node, name);
    }
}

async function refreshAdvanced(node, apply = false, options = {}) {
    const ctx = readLoaderContext();
    const target = detailControlTarget(node);
    const fallbackVram = widget(target, "vram_preset")?.value || "12GB";
    const quality = widget(target, "quality_mode")?.value || "balanced";
    const hw = await probeHardware(ctx);
    const plan = chooseAutoValues(hw, ctx, quality, fallbackVram);
    node[AUTO_SIG_PROP] = targetSignature(target);

    if (apply) {
        if (options.clearManualOverrides) node[AUTO_MANUAL_OVERRIDES_PROP] = {};
        setAutoActive(node, true);
        node[AUTO_SETTING_PROP] = true;
        try {
            setAutoWidget(node, "enabled", true);
            setAutoWidget(node, "temporal_tile_size", plan.values.temporal_tile_size);
            setAutoWidget(node, "temporal_overlap", plan.values.temporal_overlap);
            setAutoWidget(node, "guiding_strength", plan.values.guiding_strength);
            setAutoWidget(node, "temporal_overlap_cond_strength", plan.values.temporal_overlap_cond_strength);
            setAutoWidget(node, "cond_image_strength", plan.values.cond_image_strength);
            setAutoWidget(node, "horizontal_tiles", plan.values.horizontal_tiles ?? 0);
            setAutoWidget(node, "vertical_tiles", plan.values.vertical_tiles ?? 0);
            setAutoWidget(node, "spatial_overlap", plan.values.spatial_overlap ?? 0);
            setAutoWidget(node, "spatial_tiling_mode", plan.values.spatial_tiling_mode || "inherit");
        } finally {
            node[AUTO_SETTING_PROP] = false;
        }
    }

    const seconds = ctx.frames && ctx.fps ? (ctx.frames / ctx.fps).toFixed(2) : "?";
    const hwLine = plan.vramGb ? `${plan.vram} detected (${Number(plan.vramGb).toFixed(1)}GB VRAM)` : `${plan.vram} fallback`;
    const ramLine = plan.ramGb ? `${Number(plan.ramGb).toFixed(1)}GB RAM` : "RAM ?";
    const chunkLine = plan.chunks ? `${plan.chunks.chunks} chunks approx, latent tile ${plan.chunks.latentTile}, step ${plan.chunks.step}` : "chunks ?";
    const manual = Object.keys(manualOverrides(node));
    setPreview(
        node,
        [
            `Auto: ${hwLine}, ${ramLine}`,
            `Preset: ${plan.quality} / ${fallbackVram}`,
            `Video: ${ctx.frames || "?"} frames @ ${ctx.fps || "?"} fps (${seconds}s)`,
            `Loop: ${plan.values.temporal_tile_size}/${plan.values.temporal_overlap}, ${chunkLine}`,
            `Tiles: ${plan.values.horizontal_tiles ?? 0}x${plan.values.vertical_tiles ?? 0}, overlap ${plan.values.spatial_overlap ?? 0}`,
            manual.length ? `Manual override: ${manual.join(", ")}` : "Manual override: none",
            apply ? "Auto applied to Advanced overrides." : "Auto estimate only; press Auto to write these values.",
        ].join("\n")
    );
}

function hookTargetAutoRefresh(node) {
    const target = detailControlTarget(node);
    if (!target) return;
    const refreshLive = debounce(() => {
        if (autoIsActive(node)) refreshAdvanced(node, true);
    });
    for (const name of ["quality_mode", "vram_preset"]) {
        const w = widget(target, name);
        if (!w || w[AUTO_HOOK_PROP]) continue;
        const previous = w.callback;
        w.callback = function () {
            const result = previous?.apply(this, arguments);
            refreshLive();
            return result;
        };
        w[AUTO_HOOK_PROP] = true;
    }
}

function startAutoWatcher(node) {
    if (!node || node[AUTO_WATCH_PROP]) return;
    node[AUTO_WATCH_PROP] = setInterval(() => {
        scanManualOverrides(node);
        if (!autoIsActive(node)) return;
        const target = detailControlTarget(node);
        if (!target) return;
        const sig = targetSignature(target);
        if (sig === node[AUTO_SIG_PROP]) return;
        node[AUTO_SIG_PROP] = sig;
        refreshAdvanced(node, true);
    }, 300);
}

function removeLegacyButtonWidgets(node) {
    if (!Array.isArray(node?.widgets)) return;
    node.widgets = node.widgets.filter((w) => w?.name !== APPLY_BUTTON && w?.name !== REFRESH_BUTTON);
}

function makeButton(label, onClick, primary = false) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = label;
    btn.style.cursor = "pointer";
    btn.style.border = "1px solid #4d5668";
    btn.style.borderRadius = "6px";
    btn.style.padding = "6px 10px";
    btn.style.fontSize = "12px";
    btn.style.fontWeight = primary ? "700" : "600";
    btn.style.color = primary ? "#0b1020" : "#d7deea";
    btn.style.background = primary ? "#b8d8ff" : "#1d2430";
    btn.style.minWidth = primary ? "78px" : "128px";
    btn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        onClick();
    });
    return btn;
}

function ensureAutoControls(node) {
    if (!node || node[AUTO_CONTROLS_PROP]) return;

    const runAuto = () => {
        hookTargetAutoRefresh(node);
        startAutoWatcher(node);
        refreshAdvanced(node, true, { clearManualOverrides: true });
    };

    if (typeof node.addDOMWidget === "function") {
        const root = document.createElement("div");
        root.style.display = "flex";
        root.style.gap = "8px";
        root.style.alignItems = "center";
        root.style.padding = "4px 0";
        root.appendChild(makeButton("Auto", runAuto, true));
        const domWidget = node.addDOMWidget("Atelier Auto", CONTROLS_WIDGET, root, { serialize: false });
        if (domWidget) domWidget.serialize = false;
        node[AUTO_CONTROLS_PROP] = domWidget || root;
        return;
    }

    const btn = node.addWidget("button", "Auto", "Auto", runAuto, { serialize: false });
    btn.name = APPLY_BUTTON;
    btn.value = "Auto";
    btn.callback = runAuto;
    btn.serialize = false;
}

function installAdvanced(node) {
    ensurePreview(node);
    hookManualOverrides(node);
    startAutoWatcher(node);
    removeLegacyButtonWidgets(node);
    ensureAutoControls(node);
    setTimeout(() => {
        hookTargetAutoRefresh(node);
        if (autoIsActive(node)) refreshAdvanced(node, true);
        else updatePreviewIdle(node);
    }, 250);
}

function installSampler(node) {
    const advanced = linkedAdvancedNode(node);
    if (advanced) {
        setTimeout(() => {
            hookTargetAutoRefresh(advanced);
            startAutoWatcher(advanced);
            if (autoIsActive(advanced)) refreshAdvanced(advanced, true);
        }, 250);
    }
}

app.registerExtension({
    name: "iamccs.detail_atelier.ui",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        const name = nodeData?.name;
        if (name !== ADVANCED_TYPE && name !== SAMPLER_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            if (name === ADVANCED_TYPE) installAdvanced(this);
            if (name === SAMPLER_TYPE) installSampler(this);
            return r;
        };
    },

    async nodeCreated(node) {
        if (nodeClass(node) === ADVANCED_TYPE) installAdvanced(node);
        if (nodeClass(node) === SAMPLER_TYPE) installSampler(node);
    },
});
