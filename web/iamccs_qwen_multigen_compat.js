import { app } from "../../scripts/app.js";

const SEED_MODES = new Set(["fixed", "randomize", "increment", "decrement"]);
const NODE_DEFAULTS = {
    IAMCCS_QwenMultiGen: [0, "randomize", 4, 1, "euler", "simple", 1, "\\n", "index_timestep_zero", "qwen_multi", true],
    IAMCCS_FluxKleinMultiGen: [0, "randomize", 8, 1, "euler", "\\n", "workflow_default", 1, "nearest-exact", "", "flux_klein_multi", false, 1, 720, 1024, false, "flux_klein_debug"],
};

function isFiniteNumber(value) {
    return typeof value === "number" && Number.isFinite(value);
}

function normalizeWidgetsValues(nodeName, values) {
    const defaults = NODE_DEFAULTS[nodeName];
    if (!defaults) {
        return Array.isArray(values) ? [...values] : [];
    }

    const normalized = Array.isArray(values) ? [...values] : [];

    if (normalized.length > 0 && typeof normalized[0] === "string" && SEED_MODES.has(normalized[0])) {
        normalized.unshift(defaults[0]);
    }

    if (normalized.length === defaults.length - 1 && !SEED_MODES.has(String(normalized[1]))) {
        normalized.splice(1, 0, defaults[1]);
    }

    while (normalized.length < defaults.length) {
        normalized.push(defaults[normalized.length]);
    }

    normalized.length = defaults.length;

    normalized[0] = isFiniteNumber(normalized[0]) ? normalized[0] : defaults[0];
    normalized[1] = SEED_MODES.has(String(normalized[1])) ? normalized[1] : defaults[1];

    for (let index = 2; index < defaults.length; index += 1) {
        const defaultValue = defaults[index];
        if (typeof defaultValue === "number") {
            if (!isFiniteNumber(normalized[index])) normalized[index] = defaultValue;
            continue;
        }

        if (typeof defaultValue === "boolean") {
            normalized[index] = Boolean(normalized[index]);
            continue;
        }

        if (typeof normalized[index] !== "string") {
            normalized[index] = defaultValue;
            continue;
        }

        if (defaultValue !== "" && !normalized[index]) {
            normalized[index] = defaultValue;
        }
    }

    return normalized;
}

function syncNodeWidgets(node) {
    const nodeName = node?.comfyClass ?? node?.type;
    const values = normalizeWidgetsValues(nodeName, node.widgets_values ?? []);
    node.widgets_values = values;
    if (!Array.isArray(node.widgets)) return;
    for (let index = 0; index < Math.min(node.widgets.length, values.length); index += 1) {
        if (node.widgets[index]) {
            node.widgets[index].value = values[index];
        }
    }
}

app.registerExtension({
    name: "iamccs.qwen_multigen.compat",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_DEFAULTS[nodeData?.name]) return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            try {
                syncNodeWidgets(this);
            } catch {
                // ignore compat patch failures
            }
            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            if (info && Array.isArray(info.widgets_values)) {
                info.widgets_values = normalizeWidgetsValues(nodeData.name, info.widgets_values);
            }
            const result = originalOnConfigure?.apply(this, arguments);
            try {
                syncNodeWidgets(this);
            } catch {
                // ignore compat patch failures
            }
            return result;
        };

        const originalOnSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (info) {
            const result = originalOnSerialize?.apply(this, arguments);
            try {
                const target = info ?? {};
                target.widgets_values = normalizeWidgetsValues(nodeData.name, target.widgets_values ?? this.widgets_values ?? []);
            } catch {
                // ignore compat patch failures
            }
            return result;
        };
    },
});