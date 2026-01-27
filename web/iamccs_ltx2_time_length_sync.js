// IAMCCS LTX-2 seconds <-> length sync
// Frontend-only helper for:
// - IAMCCS_LTX2_TimeFrameCount
// - IAMCCS_LTX2_Validator
// Uses FPS to convert:
//   length(frames) = 1 + seconds * fps
//   seconds = (length-1) / fps

import { app } from "../../scripts/app.js";

const TIMEFRAME_TYPE = "IAMCCS_LTX2_TimeFrameCount";
const VALIDATOR_TYPE = "IAMCCS_LTX2_Validator";
const FRAMERATE_SYNC_TYPE = "IAMCCS_LTX2_FrameRateSync";

console.log("[IAMCCS LTX2] Loading seconds/length sync...");

function getWidget(node, name) {
    if (!node?.widgets?.length) return null;
    return node.widgets.find(w => w?.name === name || w?.label === name) || null;
}

function clampNumber(v, min, max) {
    const n = Number(v);
    if (!Number.isFinite(n)) return min;
    return Math.max(min, Math.min(max, n));
}

function getGraphFpsOrDefault(defaultFps = 25) {
    try {
        const nodes = app?.graph?._nodes || [];
        const fr = nodes.find(n => n?.type === FRAMERATE_SYNC_TYPE);
        if (!fr) return defaultFps;

        const wFps = getWidget(fr, "fps");
        const wMode = getWidget(fr, "int_mode");
        const fpsIn = clampNumber(wFps?.value ?? defaultFps, 1.0, 240.0);
        const mode = String(wMode?.value || "round");

        if (mode === "floor") return Math.max(1, Math.floor(fpsIn));
        if (mode === "ceil") return Math.max(1, Math.ceil(fpsIn));
        if (mode === "fixed") return Math.max(1, Math.round(fpsIn));
        // round
        return Math.max(1, Math.round(fpsIn));
    } catch (e) {
        return defaultFps;
    }
}

function snapLengthToLtx2RuleUp(length) {
    // LTX-2 constraint: 1 + 8*x frames
    const n = Math.max(1, Math.round(Number(length) || 1));
    const rem = (n - 1) % 8;
    if (rem === 0) return n;
    return n + (8 - rem);
}

function installSecondsLengthSync(node, { snapRule = false } = {}) {
    const wSeconds = getWidget(node, "seconds");
    const wLength = getWidget(node, "length");
    if (!wSeconds || !wLength) return;

    let updating = false;

    const updateLengthFromSeconds = () => {
        const fps = getGraphFpsOrDefault(25);
        const seconds = clampNumber(wSeconds.value, 0.01, 3600);
        let length = clampNumber(1 + Math.round(seconds * fps), 1, 16385);
        if (snapRule) length = snapLengthToLtx2RuleUp(length);
        wLength.value = length;
    };

    const updateSecondsFromLength = () => {
        const fps = getGraphFpsOrDefault(25);
        const length = clampNumber(wLength.value, 1, 16385);
        const seconds = (length - 1) / fps;
        // keep precision consistent with widget step 0.01
        wSeconds.value = Math.round(clampNumber(seconds, 0.01, 3600) * 100) / 100;
    };

    const hookWidget = (widget, fn) => {
        const prev = widget.callback;
        widget.callback = function () {
            const r = prev?.apply(this, arguments);
            if (updating) return r;
            updating = true;
            try {
                fn();
                node.setDirtyCanvas(true, true);
            } finally {
                updating = false;
            }
            return r;
        };
    };

    hookWidget(wSeconds, updateLengthFromSeconds);
    hookWidget(wLength, updateSecondsFromLength);
}

app.registerExtension({
    name: "iamccs.ltx2.time_length_sync",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData?.name) return;
        const name = nodeData.name;

        if (name !== TIMEFRAME_TYPE && name !== VALIDATOR_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            // Snap to 8n+1 on both nodes, since LTX-2 VAE encode requires it.
            installSecondsLengthSync(this, { snapRule: true });
            return r;
        };
    },
});
