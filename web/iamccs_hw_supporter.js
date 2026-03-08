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

function _readWidgetValue(node, ...names) {
    // Try each name in order; also fall back to widget index if name is a number.
    for (const n of names) {
        let w;
        if (typeof n === "number") {
            w = node?.widgets?.[n];
        } else {
            w = findWidget(node, n);
        }
        if (w?.value != null) {
            const v = Number(w.value);
            if (Number.isFinite(v)) return v;
        }
    }
    return null;
}

function _followLinkedInputByName(node, widgetName) {
    // When a widget is wired via link, ComfyUI converts it to an input that retains
    // the original widget name.  This helper finds that input by name, follows the
    // link to the source node, and returns the first numeric widget value found there.
    // Works reliably for INTConstant, PrimitiveInt, PrimitiveFloat, etc.
    try {
        const inp = (node?.inputs || []).find(i => i?.name === widgetName);
        if (!inp || inp.link == null) return null;
        const linkData = app.graph?.links?.[inp.link];
        if (!linkData) return null;
        // LiteGraph stores links as LLink objects with .origin_id (named property).
        // Guard with array fallback for older serialized plain-array links.
        const srcNodeId = linkData.origin_id ?? linkData[1];
        if (srcNodeId == null) return null;
        const srcNode = app.graph?.getNodeById(srcNodeId);
        if (!srcNode) return null;
        // Try every widget; return the first that is a finite number.
        for (const w of srcNode.widgets || []) {
            const v = Number(w?.value);
            if (Number.isFinite(v)) return v;
        }
        return null;
    } catch {
        return null;
    }
}

function tryGetLtxContext(hintNode) {
    // Best-effort: looks for LTX nodes commonly present in this workflow.
    // EmptyLTXVLatentVideo widgets: [width, height, length, ...]
    // LTXVConditioning widgets: [fps]
    const ctx = { width: null, height: null, frames: null, fps: null };

    try {
        const latentNode = findFirstNodeByType("EmptyLTXVLatentVideo");
        const fpsNode = findFirstNodeByType("LTXVConditioning");

        if (latentNode) {
            // Try named widgets first (direct values), then follow link by name if widget was converted.
            ctx.width  = _readWidgetValue(latentNode, "width",  0) ?? _followLinkedInputByName(latentNode, "width");
            ctx.height = _readWidgetValue(latentNode, "height", 1) ?? _followLinkedInputByName(latentNode, "height");
            // length = frames; widget may be hidden if wired via link.
            ctx.frames = _readWidgetValue(latentNode, "length", 2) ?? _followLinkedInputByName(latentNode, "length");
        }

        if (fpsNode?.widgets?.length) {
            ctx.fps = _readWidgetValue(fpsNode, "frame_rate", "fps", 0);
        }

        // Fallback: look for IAMCCS-style INTConstant nodes with recognizable titles.
        if (ctx.frames == null) {
            // Scan all nodes for a "length" constant; prefer ones with title containing "second"
            const all = app?.graph?._nodes || [];
            const lenNode = all.find(n => {
                const t = (n.title || "").toLowerCase();
                return (t.includes("length") || t.includes("duration") || t.includes("second"));
            });
            if (lenNode) {
                const val = _readWidgetValue(lenNode, "value", 0);
                if (val != null) {
                    // Treat as seconds if title contains "second" or "duration", else frames.
                    const t = (lenNode.title || "").toLowerCase();
                    if (t.includes("second") || t.includes("duration")) {
                        // convert to frames below after fps is resolved
                        ctx._length_seconds = val;
                    } else {
                        ctx.frames = val;
                    }
                }
            }
        }

        if (ctx.fps == null) {
            const all = app?.graph?._nodes || [];
            const fNode = all.find(n => (n.title || "").toLowerCase() === "fps");
            if (fNode) ctx.fps = _readWidgetValue(fNode, "value", 0);
        }

        // Convert length_seconds → frames if we got one
        if (ctx.frames == null && ctx._length_seconds != null) {
            const fps = ctx.fps || 24;
            ctx.frames = Math.round(ctx._length_seconds * fps);
        }
    } catch {
        // ignore
    }

    // --- Override from the probe node's own duration_hint_s widget (most reliable) ---
    if (hintNode) {
        // _readWidgetValue works when the value is typed directly;
        // _followLinkedInputByName works when a link was connected (widget → input).
        const dur = _readWidgetValue(hintNode, "duration_hint_s") ??
                    _followLinkedInputByName(hintNode, "duration_hint_s");
        if (dur != null && dur > 0) {
            const fps = ctx.fps || 24;
            ctx.frames = Math.round(dur * fps);
            ctx._duration_hint_s = dur;
        }
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
        const vaeCtx = vae?.context || {};
        const vaeDurS = vaeCtx?.duration_s;
        const vaeMp = vaeCtx?.megapixels;
        const vaeFrames = vaeCtx?.frames;
        const vaeFps = vaeCtx?.fps;

        const durStr = vaeDurS != null
            ? `${vaeDurS.toFixed(1)}s`
            : (vaeFrames != null && vaeFps != null)
                ? `${(vaeFrames / vaeFps).toFixed(1)}s`
                : "? s";
        const mpStr = vaeMp != null ? ` ${vaeMp.toFixed(2)}MP` : "";

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
            `VAE decode: tile=${vaeTile} overlap=${vaeOv} temporal=${vaeT} t_overlap=${vaeTo} | duration=${durStr}${mpStr}`,
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
                const ctx = tryGetLtxContext(node);
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
