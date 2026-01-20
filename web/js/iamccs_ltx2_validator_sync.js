import { app } from "/scripts/app.js";

function findWidget(node, name) {
    return node?.widgets?.find((w) => w?.name === name);
}

function clampNumber(value, min, max) {
    const v = Number(value);
    if (!Number.isFinite(v)) return min;
    return Math.min(max, Math.max(min, v));
}

function getNearestFrameRateSyncFpsOrFallback(node, fallback = 24.0) {
    try {
        const graph = node?.graph;
        const nodes = graph?._nodes || graph?.nodes || [];
        const frNodes = nodes.filter((n) => n?.comfyClass === "IAMCCS_LTX2_FrameRateSync");
        if (!frNodes.length) return fallback;

        const pos = node?.pos || [0, 0];
        let best = frNodes[0];
        let bestD2 = Infinity;

        for (const n of frNodes) {
            const p = n?.pos || [0, 0];
            const dx = Number(p[0]) - Number(pos[0]);
            const dy = Number(p[1]) - Number(pos[1]);
            const d2 = dx * dx + dy * dy;
            if (d2 < bestD2) {
                bestD2 = d2;
                best = n;
            }
        }

        const fpsWidget = findWidget(best, "fps") || findWidget(best, "value");
        const fps = Number(fpsWidget?.value);
        if (!Number.isFinite(fps) || fps <= 0) return fallback;
        return fps;
    } catch (e) {
        return fallback;
    }
}

function framesTo8n1(frames, mode) {
    let f = Math.max(1, Math.round(Number(frames) || 1));
    const rem = (f - 1) % 8;
    if (rem === 0) return f;

    const down = Math.max(1, f - rem);
    const up = f + (8 - rem);

    if (mode === "down") return down;
    if (mode === "nearest") return (up - f) <= (f - down) ? up : down;
    return up;
}

function shouldEnable8n1Autofix(node) {
    const autofixWidget = findWidget(node, "autofix");
    const lengthFixWidget = findWidget(node, "length_fix");

    return {
        hasAutofixWidgets: Boolean(autofixWidget && lengthFixWidget),
        getAutofix: () => Boolean(autofixWidget?.value),
        getLengthFix: () => String(lengthFixWidget?.value || "up"),
        autofixWidget,
        lengthFixWidget,
    };
}

function syncLtx2SecondsLengthNode(node) {
    const secondsWidget = findWidget(node, "seconds");
    const lengthWidget = findWidget(node, "length");

    if (!secondsWidget || !lengthWidget) return;

    const {
        hasAutofixWidgets,
        getAutofix,
        getLengthFix,
        autofixWidget,
        lengthFixWidget,
    } = shouldEnable8n1Autofix(node);

    let isUpdating = false;

    const applyAutofix = (len) => {
        const safeLen = Math.max(1, Math.round(Number(len) || 1));
        if (!hasAutofixWidgets) return safeLen;
        if (!getAutofix()) return safeLen;
        return framesTo8n1(safeLen, getLengthFix());
    };

    const setWidgetValue = (widget, value) => {
        widget.value = value;
        // Ensure UI refresh
        app.graph?.setDirtyCanvas(true, false);
    };

    const syncFromSeconds = () => {
        const fps = getNearestFrameRateSyncFpsOrFallback(node, 24.0);
        const seconds = clampNumber(secondsWidget.value, 0.0, 3600.0);

        const rawLength = Math.round(seconds * fps) + 1;
        const fixedLength = applyAutofix(rawLength);
        const fixedSeconds = (fixedLength - 1) / fps;

        setWidgetValue(lengthWidget, fixedLength);
        // Keep both consistent with what the node will actually output.
        setWidgetValue(secondsWidget, Number(fixedSeconds.toFixed(2)));
    };

    const syncFromLength = () => {
        const fps = getNearestFrameRateSyncFpsOrFallback(node, 24.0);
        const len = Math.max(1, Math.round(Number(lengthWidget.value) || 1));
        const fixedLength = applyAutofix(len);
        const fixedSeconds = (fixedLength - 1) / fps;

        setWidgetValue(lengthWidget, fixedLength);
        setWidgetValue(secondsWidget, Number(fixedSeconds.toFixed(2)));
    };

    const withGuard = (fn) => {
        if (isUpdating) return;
        isUpdating = true;
        try {
            fn();
        } finally {
            isUpdating = false;
        }
    };

    const originalSecondsCb = secondsWidget.callback;
    secondsWidget.callback = function (value) {
        originalSecondsCb?.call(this, value);
        withGuard(syncFromSeconds);
    };

    const originalLengthCb = lengthWidget.callback;
    lengthWidget.callback = function (value) {
        originalLengthCb?.call(this, value);
        withGuard(syncFromLength);
    };

    const originalAutofixCb = autofixWidget?.callback;
    if (autofixWidget) {
        autofixWidget.callback = function (value) {
            originalAutofixCb?.call(this, value);
            withGuard(syncFromLength);
        };
    }

    const originalLengthFixCb = lengthFixWidget?.callback;
    if (lengthFixWidget) {
        lengthFixWidget.callback = function (value) {
            originalLengthFixCb?.call(this, value);
            withGuard(syncFromLength);
        };
    }

    // Initial sync after creation/configure.
    withGuard(syncFromLength);
}

app.registerExtension({
    name: "IAMCCS.LTX2Validator.SecondsLengthSync",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const supported = new Set([
            "IAMCCS_LTX2_Validator",
            "IAMCCS_LTX2_TimeFrameCount",
        ]);
        if (!supported.has(nodeData?.name)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const node = this;
            setTimeout(() => syncLtx2SecondsLengthNode(node), 10);
            return r;
        };

        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function () {
            const r = configure?.apply(this, arguments);
            const node = this;
            setTimeout(() => syncLtx2SecondsLengthNode(node), 10);
            return r;
        };
    },
});
