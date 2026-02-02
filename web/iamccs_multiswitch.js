// IAMCCS MultiSwitch
// Base behavior matches an "Any Switch" style: first non-empty input wins.
// Extras: optional label helpers.

import { app } from "../../scripts/app.js";

const NODE_TYPE_MAIN = "IAMCCS_MultiSwitch";

const INPUT_PREFIX = "input_";
const MIN_INPUTS = 5;

function getGraph() {
    return app?.canvas?.getCurrentGraph?.() || app?.graph || null;
}

function clampInt(v, min, max) {
    const n = Number(v);
    if (!Number.isFinite(n)) return min;
    return Math.max(min, Math.min(max, Math.trunc(n)));
}

function getInputIndexFromName(name) {
    const m = String(name || "").match(/^input_(\d{2,})$/);
    if (!m) return null;
    return parseInt(m[1], 10);
}

function getConnectedInputIndices(node) {
    const out = [];
    for (const inp of node?.inputs || []) {
        const idx = getInputIndexFromName(inp?.name);
        if (!idx) continue;
        // Only consider a link "connected" if it resolves to a valid graph link.
        const graph = getGraph();
        const linkId = inp?.link;
        const link = (graph && linkId != null) ? _getGraphLink(graph, linkId) : null;
        if (link) out.push(idx);
    }
    return out;
}

function _getGraphLink(graph, linkId) {
    try {
        const links = graph?.links;
        if (!links) return null;
        if (typeof links.get === "function") return links.get(linkId) || links.get(String(linkId)) || null;
        return links?.[linkId] || links?.[String(linkId)] || null;
    } catch {
        return null;
    }
}

function listSwitchInputs(node) {
    return (node?.inputs || []).filter(Boolean);
}

function renameSwitchInputsSequential(node) {
    const inputs = listSwitchInputs(node);
    for (let i = 0; i < inputs.length; i++) {
        inputs[i].name = `${INPUT_PREFIX}${String(i + 1).padStart(2, "0")}`;
    }
}

function addOneInput(node) {
    const nextIndex = listSwitchInputs(node).length + 1;
    node.addInput(`${INPUT_PREFIX}${String(nextIndex).padStart(2, "0")}`, "*");
}

function removeTrailingUnusedInputs(node, minKeep) {
    const keep = Math.max(MIN_INPUTS, minKeep || 0);
    while (true) {
        const inputs = listSwitchInputs(node);
        if (inputs.length <= keep) break;
        const last = inputs[inputs.length - 1];
        if (last?.link != null) break;
        // Remove last input slot.
        node.removeInput?.(node.inputs.indexOf(last));
        if (!node.removeInput) {
            // Fallback for older LiteGraph builds
            node.inputs.splice(node.inputs.indexOf(last), 1);
        }
    }
}

function stabilizeDynamicInputs(node) {
    // Ensure at least MIN_INPUTS inputs.
    while (listSwitchInputs(node).length < MIN_INPUTS) addOneInput(node);

    // If the last input is connected, add another empty one.
    const inputs = listSwitchInputs(node);
    const last = inputs[inputs.length - 1];
    if (last?.link != null) addOneInput(node);

    // Clean up extra trailing empties (but keep one spare).
    removeTrailingUnusedInputs(node, MIN_INPUTS + 1);

    // Make names stable and sequential.
    renameSwitchInputsSequential(node);
}

function pad2(n) {
    return String(n).padStart(2, "0");
}

function getSlotIndexForInputNumber(node, n) {
    const name = `${INPUT_PREFIX}${pad2(n)}`;
    const idx = (node?.inputs || []).findIndex(i => i?.name === name);
    return idx >= 0 ? idx : null;
}

function swapLabelEntries(node, n1, n2) {
    node.properties = node.properties || {};
    node.properties.iamccs_labels = node.properties.iamccs_labels || {};
    const labels = node.properties.iamccs_labels;
    const k1 = `${INPUT_PREFIX}${pad2(n1)}`;
    const k2 = `${INPUT_PREFIX}${pad2(n2)}`;
    const t = labels[k1];
    labels[k1] = labels[k2];
    labels[k2] = t;
}

function swapInputConnectionsByNumber(node, n1, n2) {
    if (n1 === n2) return;

    const graph = getGraph();
    if (!graph) return;

    // Ensure enough dynamic inputs exist.
    stabilizeDynamicInputs(node);
    while (listSwitchInputs(node).length < Math.max(n1, n2)) {
        addOneInput(node);
        stabilizeDynamicInputs(node);
    }

    const slot1 = getSlotIndexForInputNumber(node, n1);
    const slot2 = getSlotIndexForInputNumber(node, n2);
    if (slot1 == null || slot2 == null) return;

    const inp1 = node.inputs[slot1];
    const inp2 = node.inputs[slot2];

    const link1 = inp1?.link;
    const link2 = inp2?.link;

    // Move link1 to slot2
    if (link1 != null && graph.links?.[link1]) {
        graph.links[link1].target_slot = slot2;
    }
    // Move link2 to slot1
    if (link2 != null && graph.links?.[link2]) {
        graph.links[link2].target_slot = slot1;
    }

    inp1.link = link2 ?? null;
    inp2.link = link1 ?? null;

    swapLabelEntries(node, n1, n2);
    applyLabels(node);

    try {
        graph.setDirtyCanvas(true, true);
        node.setDirtyCanvas(true, true);
    } catch {}
}

function tryInferLabelFromSource(node, inputIndex) {
    try {
        const graph = getGraph();
        if (!graph) return null;
        const wantedName = `${INPUT_PREFIX}${String(inputIndex).padStart(2, "0")}`;
        const slot = (node?.inputs || []).findIndex(i => i?.name === wantedName);
        if (slot < 0) return null;
        const inp = node?.inputs?.[slot];
        if (!inp || inp.link == null) return null;

        const link = _getGraphLink(graph, inp.link);
        if (!link) return null;

        // Guard against stale/aliased links: only accept if it targets this node+slot.
        if (String(link.target_id) !== String(node.id) || Number(link.target_slot) !== Number(slot)) return null;

        const originNode = graph.getNodeById?.(link.origin_id);
        if (!originNode) return null;

        // Preferred: use the connected output's label/name.
        try {
            const outSlot = Number(link.origin_slot);
            const out = originNode?.outputs?.[outSlot];
            const outLabel = String(out?.label ?? "").trim();
            const outName = String(out?.name ?? "").trim();
            const chosen = outLabel || outName;
            if (chosen) return chosen;
        } catch {}

        // Heuristic: common widget names holding filenames/models.
        const widgetNames = [
            "ckpt_name",
            "checkpoint",
            "model_name",
            "vae_name",
            "lora_name",
            "clip_name",
            "unet_name",
        ];

        const widgets = originNode.widgets || [];
        for (const wn of widgetNames) {
            const w = widgets.find(x => String(x?.name || x?.label || "").toLowerCase() === wn);
            if (w && w.value != null) {
                const txt = String(w.value).trim();
                if (txt) return txt;
            }
        }

        // Fallback: origin title.
        const t = String(originNode.title || "").trim();
        return t || null;
    } catch {
        return null;
    }
}

function applyLabels(node) {
    node.properties = node.properties || {};
    const labels = node.properties.iamccs_labels || {};

    for (const inp of node.inputs || []) {
        const idx = getInputIndexFromName(inp?.name);
        if (!idx) continue;

        const custom = labels[inp.name];
        if (custom && String(custom).trim()) {
            inp.label = String(custom).trim();
        } else {
            inp.label = inp.name;
        }
    }

    // Ensure UI refresh.
    try {
        getGraph()?.setDirtyCanvas(true, true);
        node.setDirtyCanvas?.(true, true);
        app?.canvas?.setDirty?.(true, true);
    } catch {}
}

function _getLocalPosFromEvent(node, e, pos) {
    // In ComfyUI/LiteGraph, the `pos` argument is not always in node-local space across builds.
    // Use canvas coordinates when available for consistent hit-testing.
    try {
        if (e && typeof e.canvasX === "number" && typeof e.canvasY === "number") {
            return [e.canvasX - (node?.pos?.[0] || 0), e.canvasY - (node?.pos?.[1] || 0)];
        }
    } catch {}
    return [pos?.[0] ?? 0, pos?.[1] ?? 0];
}

function _getInputSocketLocalPos(node, slot, out) {
    const res = out || [0, 0];
    try {
        if (typeof node?.getConnectionPos === "function") {
            const p = node.getConnectionPos(true, slot, [0, 0]);
            res[0] = (p[0] - (node?.pos?.[0] || 0));
            res[1] = (p[1] - (node?.pos?.[1] || 0));
            return res;
        }
    } catch {}

    // Fallback approximation.
    const LG = window?.LiteGraph;
    const titleH = Number(LG?.NODE_TITLE_HEIGHT ?? 30);
    const slotH = Number(LG?.NODE_SLOT_HEIGHT ?? 20);
    res[0] = 0;
    res[1] = titleH + (slot + 0.5) * slotH;
    return res;
}

function _isNodeEnabledForExecution(n) {
    try {
        const mode = n?.mode;
        // Bus Group enables with LiteGraph.ALWAYS (0) and disables with a mute-mode.
        // We treat only ALWAYS (0) (or unset) as "enabled" for this indicator.
        return mode == null || mode === 0;
    } catch {
        return true;
    }
}

function _drawActiveLinkIndicators(node, ctx) {
    try {
        const graph = getGraph();
        if (!graph) return;

        // MultiSwitch semantics: only the first valid connected input is "active".
        // After AutoLink conversion, all sources can be "enabled" (Get nodes), so
        // showing all connected inputs as active is misleading.
        let activeSlot = null;
        for (let slot = 0; slot < (node.inputs || []).length; slot++) {
            const inp = node.inputs[slot];
            const idx = getInputIndexFromName(inp?.name);
            if (!idx) continue;
            if (inp?.link == null) continue;
            const link = _getGraphLink(graph, inp.link);
            if (!link) continue;
            if (String(link.target_id) !== String(node.id) || Number(link.target_slot) !== Number(slot)) continue;
            const originNode = graph.getNodeById?.(link.origin_id);
            if (!originNode) continue;
            if (!_isNodeEnabledForExecution(originNode)) continue;
            activeSlot = slot;
            break;
        }

        if (activeSlot == null) return;

        const slot = activeSlot;
        if (typeof node.getConnectionPos !== "function") return;
        const p = node.getConnectionPos(true, slot, [0, 0]);
        const localX = (p[0] - node.pos[0]);
        const localY = (p[1] - node.pos[1]);

        // Draw a ring around the socket (no overlap with the label).
        ctx.save();
        ctx.globalAlpha = 0.95;
        ctx.strokeStyle = "#2ecc71";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(localX, localY, 6, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
    } catch {}
}

function _computeInputLinkSignature(node) {
    try {
        const parts = [];
        for (const inp of (node?.inputs || [])) {
            parts.push(String(inp?.name || "") + ":" + String(inp?.link ?? ""));
        }
        return parts.join("|");
    } catch {
        return "";
    }
}

function _syncAutoLabels(node) {
    try {
        if (!node?.properties?.iamccs_auto_label) return;

        stabilizeDynamicInputs(node);

        const n = listSwitchInputs(node).length;
        for (let i = 1; i <= n; i++) {
            const inferred = tryInferLabelFromSource(node, i);
            if (!inferred) continue;
            const key = `${INPUT_PREFIX}${String(i).padStart(2, "0")}`;
            if (!node.properties.iamccs_labels[key] || !String(node.properties.iamccs_labels[key]).trim()) {
                node.properties.iamccs_labels[key] = inferred;
            }
        }
        applyLabels(node);
    } catch {}
}

app.registerExtension({
    name: "iamccs.multiswitch",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_TYPE_MAIN) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);

            this.properties = this.properties || {};
            if (!this.properties.iamccs_labels) this.properties.iamccs_labels = {};
            if (this.properties.iamccs_auto_label == null) this.properties.iamccs_auto_label = true;
            if (this.properties.iamccs_rename_target == null) this.properties.iamccs_rename_target = 0;
            if (this.properties.iamccs_rename_text == null) this.properties.iamccs_rename_text = "";

            const renameText = this.addWidget("text", "Rename", String(this.properties.iamccs_rename_text || ""), (v) => {
                this.properties.iamccs_rename_text = String(v ?? "");

                // Apply on commit (Enter)
                const idx = clampInt(this.properties.iamccs_rename_target || 0, 0, 999);
                if (!idx) return;
                const key = `${INPUT_PREFIX}${String(idx).padStart(2, "0")}`;
                this.properties.iamccs_labels[key] = String(this.properties.iamccs_rename_text || "").trim();
                applyLabels(this);
                this.setDirtyCanvas(true, true);
            }, { serialize: true });
            this._iamccsRenameTextWidget = renameText;

            this.addWidget("toggle", "Auto label from source", !!this.properties.iamccs_auto_label, v => {
                this.properties.iamccs_auto_label = !!v;
            });

            // Only quick-action button: clear all labels.
            this.addWidget("button", "Clear all labels", null, () => {
                try {
                    this.properties = this.properties || {};
                    this.properties.iamccs_labels = {};
                    this.properties.iamccs_rename_target = 0;
                    this.properties.iamccs_rename_text = "";
                    if (this._iamccsRenameTextWidget) {
                        this._iamccsRenameTextWidget.value = "";
                        this._iamccsRenameTextWidget.label = "Rename";
                    }
                    applyLabels(this);
                    this.setDirtyCanvas(true, true);
                } catch {}
            });

            // Dynamic inputs init
            setTimeout(() => {
                try {
                    stabilizeDynamicInputs(this);
                    applyLabels(this);
                    this.setDirtyCanvas(true, true);
                } catch {}
            }, 0);

            return r;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const r = onConnectionsChange?.apply(this, arguments);

            try {
                stabilizeDynamicInputs(this);

                _syncAutoLabels(this);

                this.setDirtyCanvas(true, true);
            } catch {}

            return r;
        };

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            try {
                _drawActiveLinkIndicators(this, ctx);

                // Some builds don't reliably fire onConnectionsChange (or fire before links settle).
                // Debounce a cheap signature check and sync labels in the background.
                const sig = _computeInputLinkSignature(this);
                if (sig !== this._iamccsLastLinkSig) {
                    this._iamccsLastLinkSig = sig;
                    clearTimeout(this._iamccsAutoLabelTimeout);
                    this._iamccsAutoLabelTimeout = setTimeout(() => {
                        try {
                            this.properties = this.properties || {};
                            if (!this.properties.iamccs_labels) this.properties.iamccs_labels = {};
                            _syncAutoLabels(this);
                            this.setDirtyCanvas(true, true);
                        } catch {}
                    }, 30);
                }
            } catch {}
            return onDrawForeground?.apply(this, arguments);
        };

        // Click an input title to target rename (but do not interfere with dragging).
        const onMouseDown = nodeType.prototype.onMouseDown;
        nodeType.prototype.onMouseDown = function (e, pos, canvas) {
            try {
                if (e?.button != null && e.button !== 0) return onMouseDown?.apply(this, arguments);

                const [localX, localY] = _getLocalPosFromEvent(this, e, pos);

                const downCanvasX = (typeof e?.canvasX === "number") ? e.canvasX : ((this.pos?.[0] || 0) + localX);
                const downCanvasY = (typeof e?.canvasY === "number") ? e.canvasY : ((this.pos?.[1] || 0) + localY);

                // Only react when clicking inside the node body (avoid titlebar interactions).
                if (typeof e?.canvasY === "number" && (e.canvasY - this.pos[1]) < 0) {
                    return onMouseDown?.apply(this, arguments);
                }

                for (let slot = 0; slot < (this.inputs || []).length; slot++) {
                    const inp = this.inputs[slot];
                    if (!inp || !inp.name) continue;
                    const idx = getInputIndexFromName(inp.name);
                    if (!idx) continue;

                    const sp = _getInputSocketLocalPos(this, slot, [0, 0]);
                    const x = sp[0];
                    const y = sp[1];

                    // Slightly larger hitbox: clicking the label should be reliable.
                    const inY = Math.abs(localY - y) <= 14;
                    // Title/label region: give generous width so clicks reliably register.
                    const nodeW = (this.size?.[0] || 260);
                    const minX = Math.max(0, x + 10);
                    const maxX = Math.max(minX + 120, nodeW - 10);
                    const inX = localX >= minX && localX <= maxX;
                    if (!inY || !inX) continue;

                    // Double-click to open rename. Do it on mousedown because some builds
                    // don't deliver node.onMouseUp reliably if the canvas starts dragging.
                    const now = Date.now();
                    const last = this._iamccsLastRenameClick;
                    const isSame = last && String(last.inputName || "") === String(inp.name || "");
                    const dt = last ? Math.abs(now - (last.t || 0)) : 9999;
                    const dd = last ? Math.hypot((downCanvasX ?? 0) - (last.x ?? 0), (downCanvasY ?? 0) - (last.y ?? 0)) : 9999;

                    // Save this click as potential first click.
                    this._iamccsLastRenameClick = { inputName: inp.name, t: now, x: downCanvasX, y: downCanvasY };

                    if (isSame && dt <= 420 && dd <= 6) {
                        // Confirmed double-click: select rename target + prefill text.
                        this.properties = this.properties || {};
                        this.properties.iamccs_rename_target = idx;

                        const inferred = tryInferLabelFromSource(this, idx);
                        const current = this.properties.iamccs_labels?.[inp.name] || (this.inputs?.[slot]?.label) || inp.name;
                        this.properties.iamccs_rename_text = String((inferred || current) || "");
                        try {
                            if (this._iamccsRenameTextWidget) {
                                this._iamccsRenameTextWidget.value = this.properties.iamccs_rename_text;
                                this._iamccsRenameTextWidget.label = `Rename ${inp.name}`;
                            }
                        } catch {}
                        try {
                            this.setDirtyCanvas(true, true);
                            this.graph?.setDirtyCanvas(true, true);
                        } catch {}

                        // Consume so a double-click doesn't start dragging.
                        return true;
                    }

                    // Single click: do nothing special; allow default behavior (dragging etc).
                    return onMouseDown?.apply(this, arguments);
                }
            } catch {}

            return onMouseDown?.apply(this, arguments);
        };
    },
});
