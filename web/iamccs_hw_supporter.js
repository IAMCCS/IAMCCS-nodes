// IAMCCS HW Supporter - preset->widget synchronization
// Goal: when profile changes, propagate recommended values into other widgets
// so the UI reflects what the backend will effectively apply.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPES = new Set(["IAMCCS_HwSupporter", "IAMCCS_HwSupporterAny"]);
const VAE_DECODE_TYPES = new Set(["IAMCCS_VAEDecodeTiledSafe"]);
const GGUF_TYPES = new Set(["IAMCCS_GGUF_accelerator"]);
const SAMPLER_TYPES = new Set(["IAMCCS_SamplerAdvancedVersion1"]);

function getNodeClass(node) {
    return node?.comfyClass || node?.type || "";
}

function findWidget(node, name) {
    return (node?.widgets || []).find(w => w?.name === name) || null;
}

function setWidgetValue(widget, value) {
    if (!widget) return;
    widget.value = value;
    // Some widgets expose a callback that we should fire to keep internal state consistent.
    try {
        widget.callback?.(value);
    } catch {
        // ignore
    }
}

function getProp(node, key, fallback) {
    try {
        if (!node.properties) node.properties = {};
        const v = node.properties[key];
        return v == null ? fallback : v;
    } catch {
        return fallback;
    }
}

function setProp(node, key, value) {
    try {
        if (!node.properties) node.properties = {};
        node.properties[key] = value;
    } catch {
        // ignore
    }
}

function shouldFillMissing(widget) {
    if (!widget) return false;
    const v = widget.value;
    return v === null || v === undefined || v === "";
}

function applyWidgetKV(node, k, v, applyMode) {
    const w = findWidget(node, k);
    if (!w) return;
    if (applyMode === "fill_missing" && !shouldFillMissing(w)) return;
    setWidgetValue(w, v);
}

function isWindows() {
    try {
        return String(navigator?.platform || "").toLowerCase().includes("win");
    } catch {
        return false;
    }
}

function presetFor(profile) {
    // Keep this aligned with backend _recommend_for_profile() defaults.
    if (profile === "12GB_VRAM_32GB_RAM") {
        return {
            reserved_vram_gb: 1.25,
            reserved_vram_mode: "manual",
            sage_attention: isWindows() ? "sageattn_qk_int8_pv_fp16_cuda" : "auto",
            allow_sageattention_torch_compile: false,
            torch_compile_mode: "off",
            fp16_accumulation: "auto",
            tf32: "auto",
        };
    }
    if (profile === "low_vram") {
        return {
            reserved_vram_gb: 1.5,
            reserved_vram_mode: "manual",
            sage_attention: "auto",
            allow_sageattention_torch_compile: false,
            torch_compile_mode: "off",
            fp16_accumulation: "auto",
            tf32: "auto",
        };
    }
    if (profile === "balanced") {
        return {
            reserved_vram_gb: 1.0,
            reserved_vram_mode: "manual",
            sage_attention: "auto",
            allow_sageattention_torch_compile: false,
            torch_compile_mode: "off",
            fp16_accumulation: "auto",
            tf32: "auto",
        };
    }
    if (profile === "max_speed") {
        return {
            reserved_vram_gb: 0.5,
            reserved_vram_mode: "manual",
            sage_attention: "auto",
            allow_sageattention_torch_compile: true,
            torch_compile_mode: "reduce-overhead",
            fp16_accumulation: "auto",
            tf32: "auto",
        };
    }
    // auto
    return {
        // keep UI in a state that matches backend auto logic
        reserved_vram_gb: 0.0,
        reserved_vram_mode: "manual",
        sage_attention: "auto",
        allow_sageattention_torch_compile: false,
        torch_compile_mode: "off",
        fp16_accumulation: "auto",
        tf32: "auto",
    };
}

function findFirstNodeByType(typeName) {
    try {
        const nodes = app?.graph?._nodes || [];
        return nodes.find(n => getNodeClass(n) === typeName) || null;
    } catch {
        return null;
    }
}

function tryGetLtxContext() {
    // Best-effort: looks for LTX nodes commonly present in this workflow.
    // EmptyLTXVLatentVideo widgets: [width, height, length, ...]
    // LTXVConditioning widgets: [fps]
    const ctx = { width: null, height: null, frames: null, fps: null };

    try {
        const latentNode = findFirstNodeByType("EmptyLTXVLatentVideo");
        const fpsNode = findFirstNodeByType("LTXVConditioning");

        if (latentNode?.widgets?.length) {
            const wW = latentNode.widgets.find(w => w?.name === "width") || latentNode.widgets[0];
            const wH = latentNode.widgets.find(w => w?.name === "height") || latentNode.widgets[1];
            const wL = latentNode.widgets.find(w => w?.name === "length") || latentNode.widgets[2];
            ctx.width = wW?.value != null ? Number(wW.value) : null;
            ctx.height = wH?.value != null ? Number(wH.value) : null;
            ctx.frames = wL?.value != null ? Number(wL.value) : null;
        }

        if (fpsNode?.widgets?.length) {
            const wFps = fpsNode.widgets.find(w => w?.name === "frame_rate") || fpsNode.widgets[0];
            ctx.fps = wFps?.value != null ? Number(wFps.value) : null;
        }
    } catch {
        // ignore
    }

    // Normalize
    if (!Number.isFinite(ctx.width)) ctx.width = null;
    if (!Number.isFinite(ctx.height)) ctx.height = null;
    if (!Number.isFinite(ctx.frames)) ctx.frames = null;
    if (!Number.isFinite(ctx.fps)) ctx.fps = null;
    return ctx;
}

function installProfileSync(node) {
    const wProfile = findWidget(node, "profile");
    if (!wProfile) return;

    // Default ON to keep previous behavior, but let users disable persistence.
    if (getProp(node, "iamccs_profile_sync", null) == null) {
        setProp(node, "iamccs_profile_sync", true);
    }

    // Frontend-only toggle: when off, profile changes do not overwrite user-tuned widgets.
    if (!node._iamccsPresetSyncWidget) {
        try {
            const wSync = node.addWidget(
                "toggle",
                "Preset sync (profile → widgets)",
                !!getProp(node, "iamccs_profile_sync", true),
                (v) => {
                    setProp(node, "iamccs_profile_sync", !!v);
                }
            );
            node._iamccsPresetSyncWidget = wSync;
        } catch {
            // ignore
        }
    }

    let updating = false;

    function applyPreset() {
        if (!getProp(node, "iamccs_profile_sync", true)) return;
        const p = presetFor(String(wProfile.value || "auto"));
        setWidgetValue(findWidget(node, "reserved_vram_gb"), p.reserved_vram_gb);
        setWidgetValue(findWidget(node, "reserved_vram_mode"), p.reserved_vram_mode);
        setWidgetValue(findWidget(node, "sage_attention"), p.sage_attention);
        setWidgetValue(findWidget(node, "allow_sageattention_torch_compile"), p.allow_sageattention_torch_compile);
        setWidgetValue(findWidget(node, "torch_compile_mode"), p.torch_compile_mode);
        setWidgetValue(findWidget(node, "fp16_accumulation"), p.fp16_accumulation);
        setWidgetValue(findWidget(node, "tf32"), p.tf32);

        try {
            node.setDirtyCanvas(true, true);
            app.graph?.setDirtyCanvas(true, true);
        } catch {
            // ignore
        }
    }

    const prev = wProfile.callback;
    wProfile.callback = function () {
        const r = prev?.apply(this, arguments);
        if (updating) return r;
        updating = true;
        try {
            applyPreset();
        } finally {
            updating = false;
        }
        return r;
    };

    // Apply once on creation only if preset sync is enabled.
    applyPreset();

    // Ensure persistence across workflow reloads.
    const prevOnConfigure = node.onConfigure;
    node.onConfigure = function () {
        const r = prevOnConfigure?.apply(this, arguments);
        try {
            if (getProp(this, "iamccs_profile_sync", null) == null) {
                setProp(this, "iamccs_profile_sync", true);
            }
            if (this._iamccsPresetSyncWidget) {
                this._iamccsPresetSyncWidget.value = !!getProp(this, "iamccs_profile_sync", true);
            }
        } catch {
            // ignore
        }
        return r;
    };
}

function formatSummaryReport(data) {
    try {
        const hw = data?.hardware || {};
        const rec = data?.recommendations || {};
        const gpu = hw?.cuda_device_name ? `${hw.cuda_device_name} (${(hw.cuda_total_vram_gb ?? 0).toFixed?.(2) ?? hw.cuda_total_vram_gb} GB)` : "no CUDA";
        const ram = hw?.system_ram_gb != null ? `${hw.system_ram_gb.toFixed?.(2) ?? hw.system_ram_gb} GB RAM` : "RAM ?";

        const profile = rec?.hw_supporter?.profile || "auto";
        const headroom = rec?.hw_supporter?.reserved_vram_auto_headroom_gb;
        const reservedEff = rec?.hw_supporter?.reserved_vram_effective_gb ?? rec?.hw_supporter?.reserved_vram_gb;

        const vae = rec?.vae_decode || {};
        const vaeTile = vae?.tile_size;
        const vaeOv = vae?.overlap;
        const vaeT = vae?.temporal_size;
        const vaeTo = vae?.temporal_overlap;

        const gguf = rec?.gguf_accelerator || {};
        const ggufPolicy = gguf?.move_policy;
        const ggufLeave = gguf?.leave_free_vram_mb;

        const sampler = rec?.sampler || {};
        const samplerCleanup = sampler?.cleanup;

        // Multiline on purpose (textarea becomes scrollable if needed)
        return [
            `GPU: ${gpu}`,
            `RAM: ${ram}`,
            `HW profile: ${profile} | reserved headroom: ${headroom} GB | reserved(est): ${reservedEff} GB`,
            `VAE decode: tile=${vaeTile} overlap=${vaeOv} temporal=${vaeT} t_overlap=${vaeTo}`,
            `GGUF: move_policy=${ggufPolicy} leave_free_vram_mb=${ggufLeave} | Sampler: cleanup=${samplerCleanup}`,
        ].join("\n");
    } catch {
        return "(failed to format report)";
    }
}

async function probeHardware(context) {
    const qs = new URLSearchParams();
    if (context?.width != null) qs.set("width", String(context.width));
    if (context?.height != null) qs.set("height", String(context.height));
    if (context?.frames != null) qs.set("frames", String(context.frames));
    if (context?.fps != null) qs.set("fps", String(context.fps));
    const url = qs.toString() ? `/api/iamccs/hw_probe?${qs.toString()}` : "/api/iamccs/hw_probe";
    const r = await api.fetchApi(url, { method: "GET" });
    if (!r?.ok) {
        const t = await r.text();
        throw new Error(`hw_probe failed: ${r.status} ${t}`);
    }
    return await r.json();
}

function ensureReportWidgets(node) {
    if (findWidget(node, "hw_probe_summary")) return;

    // Persistent defaults.
    if (getProp(node, "iamccs_hw_probe_apply_mode", null) == null) {
        setProp(node, "iamccs_hw_probe_apply_mode", "overwrite");
    }

    try {
        // Apply mode: overwrite vs fill-missing (non-destructive).
        const wMode = node.addWidget(
            "combo",
            "HW probe apply mode",
            String(getProp(node, "iamccs_hw_probe_apply_mode", "overwrite")),
            (v) => {
                setProp(node, "iamccs_hw_probe_apply_mode", String(v || "overwrite"));
            },
            { values: ["overwrite", "fill_missing"] }
        );
        node._iamccsHwProbeApplyModeWidget = wMode;

        node.addWidget("text", "hw_probe_summary", "(not probed)", () => {}, { multiline: true });
        node.addWidget("button", "Probe HW & Apply", "run", async () => {
            const wSummary = findWidget(node, "hw_probe_summary");
            try {
                setWidgetValue(wSummary, "(probing...)");
                const ctx = tryGetLtxContext();
                const data = await probeHardware(ctx);

                const applyMode = String(getProp(node, "iamccs_hw_probe_apply_mode", "overwrite"));

                // Apply to HW Supporter nodes
                const nodeClass = getNodeClass(node);

                if (NODE_TYPES.has(nodeClass)) {
                    const hw = data?.recommendations?.hw_supporter || {};
                    if (hw.profile != null) {
                        applyWidgetKV(node, "profile", hw.profile, applyMode);
                    }
                    for (const [k, v] of Object.entries(hw)) {
                        if (k === "profile") continue;
                        applyWidgetKV(node, k, v, applyMode);
                    }
                }

                // Apply to VAE decode
                if (VAE_DECODE_TYPES.has(nodeClass)) {
                    const vae = data?.recommendations?.vae_decode || {};
                    // Ensure manual so the values we set are used.
                    if (vae.tiling_mode != null) applyWidgetKV(node, "tiling_mode", vae.tiling_mode, applyMode);
                    for (const [k, v] of Object.entries(vae)) {
                        if (k === "context") continue;
                        applyWidgetKV(node, k, v, applyMode);
                    }
                }

                // Apply to GGUF accelerator
                if (GGUF_TYPES.has(nodeClass)) {
                    const gguf = data?.recommendations?.gguf_accelerator || {};
                    for (const [k, v] of Object.entries(gguf)) {
                        applyWidgetKV(node, k, v, applyMode);
                    }
                }

                // Apply to sampler wrapper
                if (SAMPLER_TYPES.has(nodeClass)) {
                    const sampler = data?.recommendations?.sampler || {};
                    for (const [k, v] of Object.entries(sampler)) {
                        applyWidgetKV(node, k, v, applyMode);
                    }
                }

                // Always set summary + store full JSON for copy
                node.__iamccs_hw_probe_json = data;
                setWidgetValue(wSummary, formatSummaryReport(data));

                console.log("[IAMCCS HW Probe]", data);
                try {
                    node.setDirtyCanvas(true, true);
                    app.graph?.setDirtyCanvas(true, true);
                } catch {
                    // ignore
                }
            } catch (e) {
                console.warn("[IAMCCS HW Probe] failed", e);
                setWidgetValue(wSummary, `ERROR: ${String(e?.message || e)}`);
            }
        });

        node.addWidget("button", "Copy HW report", "copy", async () => {
            const data = node.__iamccs_hw_probe_json;
            if (!data) {
                console.warn("[IAMCCS HW Probe] nothing to copy yet");
                return;
            }
            const txt = JSON.stringify(data, null, 2);
            try {
                await navigator.clipboard.writeText(txt);
            } catch {
                // fallback: best-effort
                console.log(txt);
            }
        });

        // Restore persisted UI state on workflow load.
        const prevOnConfigure = node.onConfigure;
        node.onConfigure = function () {
            const r = prevOnConfigure?.apply(this, arguments);
            try {
                if (getProp(this, "iamccs_hw_probe_apply_mode", null) == null) {
                    setProp(this, "iamccs_hw_probe_apply_mode", "overwrite");
                }
                if (this._iamccsHwProbeApplyModeWidget) {
                    this._iamccsHwProbeApplyModeWidget.value = String(getProp(this, "iamccs_hw_probe_apply_mode", "overwrite"));
                }
            } catch {
                // ignore
            }
            return r;
        };
    } catch {
        // ignore
    }
}

app.registerExtension({
    name: "iamccs.hw_supporter.preset_sync",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        const name = nodeData?.name;
        if (!name || (!NODE_TYPES.has(name) && !VAE_DECODE_TYPES.has(name))) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            ensureReportWidgets(this);
            if (NODE_TYPES.has(name)) {
                installProfileSync(this);
            }
            return r;
        };
    },
});
