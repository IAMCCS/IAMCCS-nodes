import { app } from "../../../scripts/app.js";

const PRESET_CONFIGS = {
    "IAMCCS-SuperNodes AU+IMG2VID Exec Render": {
        presetWidget: "ui_preset",
        defaultPreset: "balanced",
        values: {
            low_ram_safe: { modular_decode: "low_ram", steps: 16, image_compression: 40, continuity_anchor_mode: "off", anchor_refresh_interval: 2, anti_drift_mode: "off", anti_drift_strength: 0.0, identity_persistence_strength: 0.0 },
            balanced: { modular_decode: "normal", steps: 20, image_compression: 33, continuity_anchor_mode: "off", anchor_refresh_interval: 3, anti_drift_mode: "off", anti_drift_strength: 0.0, identity_persistence_strength: 0.0 },
            high_quality: { modular_decode: "high", steps: 24, image_compression: 28, continuity_anchor_mode: "off", anchor_refresh_interval: 1, anti_drift_mode: "off", anti_drift_strength: 0.0, identity_persistence_strength: 0.0 },
            fast_preview: { modular_decode: "low_ram", steps: 12, image_compression: 45, continuity_anchor_mode: "off", anchor_refresh_interval: 2, anti_drift_mode: "off", anti_drift_strength: 0.0, identity_persistence_strength: 0.0 },
        },
        visibility: {
            low_ram_safe: { anchor_image_strength: false, anti_drift_mode: false, anti_drift_strength: false, identity_persistence_strength: false, output_root: true },
            balanced: { anchor_image_strength: false, anti_drift_mode: false, anti_drift_strength: false, identity_persistence_strength: false, output_root: true },
            high_quality: { anchor_image_strength: false, anti_drift_mode: false, anti_drift_strength: false, identity_persistence_strength: false, output_root: true },
            fast_preview: { anchor_image_strength: false, anti_drift_mode: false, anti_drift_strength: false, identity_persistence_strength: false, output_root: false },
        },
    },
    "IAMCCS-SuperNodes AU+IMG2VID Exec VAE": {
        presetWidget: "ui_preset",
        defaultPreset: "balanced",
        values: {
            low_ram_safe: { decode_mode: "low_ram", jpg_quality: 92, crf: 21, tiled_tile_size: 512, tiled_overlap: 64 },
            balanced: { decode_mode: "normal", jpg_quality: 95, crf: 19, tiled_tile_size: 512, tiled_overlap: 64 },
            high_quality: { decode_mode: "high", jpg_quality: 100, crf: 16, tiled_tile_size: 768, tiled_overlap: 96 },
            fast_preview: { decode_mode: "low_ram", jpg_quality: 88, crf: 24, tiled_tile_size: 384, tiled_overlap: 48 },
        },
        visibility: {
            low_ram_safe: { frames_subdir: true, image_format: true, jpg_quality: true },
            balanced: { frames_subdir: true, image_format: true, jpg_quality: true },
            high_quality: { frames_subdir: true, image_format: true, jpg_quality: true },
            fast_preview: { frames_subdir: true, image_format: true, jpg_quality: true },
        },
    },
};

const NODE_GROUPS = {
    "IAMCCS-SuperNodes AU+IMG2VID Exec Planner": [
        { key: "timeline", label: "Timeline", color: "#2d6f63", widgets: ["fps", "segment_seconds"] },
        { key: "planning", label: "Planning", color: "#6d5d2a", widgets: ["planning_mode", "segment_preset", "overlap_frames", "ltx_round_mode"] },
    ],
    "IAMCCS-SuperNodes AU+IMG2VID Exec Render": [
        { key: "preset", label: "Preset", color: "#445c75", widgets: ["ui_preset"] },
        { key: "prompts", label: "Prompting", color: "#6b4f2a", widgets: ["positive_text", "negative_text"] },
        { key: "video", label: "Video", color: "#2f4d7a", widgets: ["width", "height"] },
        { key: "sampling", label: "Sampling", color: "#5b3a73", widgets: ["steps", "cfg", "sampler_name", "seed", "max_shift", "base_shift", "sigma_terminal", "manual_sigmas", "image_strength", "image_compression"] },
        { key: "audio", label: "Audio", color: "#2f6a47", widgets: ["audio_context_mode", "audio_left_context_s", "audio_right_context_s"] },
        { key: "stitch", label: "Stitch", color: "#7a4f2f", widgets: ["stitch_preset", "overlap_side", "overlap_mode", "start_frames_rule"] },
        { key: "anchor", label: "Anchor", color: "#8a4d4d", widgets: ["continuity_anchor_mode", "anchor_refresh_interval", "anchor_image_strength", "anti_drift_mode", "anti_drift_strength", "identity_persistence_strength"] },
        { key: "modular", label: "Modular", color: "#35566e", widgets: ["modular_decode", "downstream_stage_mode", "output_root"] },
        { key: "debug", label: "Debug", color: "#5b5b2f", widgets: ["segment_overlay_mode", "segment_overlay_text"] },
    ],
    "IAMCCS-SuperNodes Second Stage": [
        { key: "stage2", label: "Second Stage", color: "#6f4a7f", widgets: ["second_stage_mode", "stage2_model_policy", "second_stage_upscale_model", "second_stage_reinject_strength", "second_stage_cfg", "second_stage_manual_sigmas"] },
    ],
    IAMCCS_GC_AudioConcatSupernode: [
        { key: "concat", label: "Audio Concat", color: "#3e6f55", widgets: ["concat_mode", "clip_durations_seconds", "gap_seconds", "intro_seconds", "outro_seconds"] },
    ],
    "IAMCCS-SuperNodes AU+IMG2VID Exec VAE": [
        { key: "preset", label: "Preset", color: "#445c75", widgets: ["ui_preset"] },
        { key: "decode", label: "Decode", color: "#3e5f82", widgets: ["frame_rate", "decode_mode", "output_root", "frames_subdir", "image_format", "jpg_quality", "tiled_tile_size", "tiled_overlap"] },
        { key: "final", label: "Finalize", color: "#6c5330", widgets: ["filename_prefix", "crf", "pix_fmt", "trim_to_audio", "save_metadata"] },
    ],
    "IAMCCS-SuperNodes AU+IMG2VID Exec Finalize": [
        { key: "output", label: "Output", color: "#556677", widgets: ["frame_rate", "filename_prefix", "crf", "pix_fmt", "trim_to_audio"] },
    ],
};

function findWidget(node, name) {
    return node.widgets?.find((widget) => widget?.name === name);
}

function setWidgetVisibility(widget, visible) {
    if (!widget || widget.type === "converted-widget") {
        return;
    }

    widget.hidden = !visible;
    widget.disabled = !visible;

    if (widget.element) {
        widget.element.style.display = visible ? "" : "none";
    }
    if (widget.inputEl) {
        widget.inputEl.style.display = visible ? "" : "none";
    }

    if (visible) {
        if (Object.prototype.hasOwnProperty.call(widget, "__iamccsOrigComputeSize")) {
            widget.computeSize = widget.__iamccsOrigComputeSize;
        } else {
            delete widget.computeSize;
        }
    } else {
        if (!Object.prototype.hasOwnProperty.call(widget, "__iamccsOrigComputeSize")) {
            widget.__iamccsOrigComputeSize = widget.computeSize;
        }
        widget.computeSize = () => [0, -4];
        widget.y = undefined;
        widget.last_y = undefined;
    }
}

function fitNodeToWidgets(node) {
    const size = node.computeSize?.() || node.size;
    node.setSize([
        Math.max(node.size?.[0] || 0, size[0]),
        Math.max(size[1], size[1] + 8),
    ]);
}

function insertWidget(node, widget, index) {
    const widgets = node.widgets || [];
    const currentIndex = widgets.indexOf(widget);
    if (currentIndex >= 0) {
        widgets.splice(currentIndex, 1);
    }
    widgets.splice(index, 0, widget);
}

function setWidgetValue(node, widgetName, value) {
    const widget = findWidget(node, widgetName);
    if (!widget || widget.value === value) {
        return;
    }
    widget.value = value;
    widget.callback?.(value);
}

function getLinxTargets(node) {
    const results = [];
    for (const output of node.outputs || []) {
        if (output?.type !== "IAMCCS_SUPERNODE_LINX") {
            continue;
        }
        for (const linkId of output.links || []) {
            const link = app.graph.links?.[linkId];
            if (!link) {
                continue;
            }
            const targetNode = app.graph.getNodeById?.(link.target_id);
            if (targetNode) {
                results.push(targetNode);
            }
        }
    }
    return results;
}

function applyVaeDecodeModeVisibility(node) {
    if (node.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec VAE") {
        return;
    }
    const decodeMode = String(findWidget(node, "decode_mode")?.value || "low_ram");
    const isLowRam = decodeMode === "low_ram";
    const isNormal = decodeMode === "normal";
    setWidgetVisibility(findWidget(node, "frames_subdir"), isLowRam);
    setWidgetVisibility(findWidget(node, "image_format"), isLowRam);
    setWidgetVisibility(findWidget(node, "jpg_quality"), isLowRam);
    setWidgetVisibility(findWidget(node, "tiled_tile_size"), isNormal);
    setWidgetVisibility(findWidget(node, "tiled_overlap"), isNormal);
    fitNodeToWidgets(node);
}

function syncDownstreamVaeDecodeModes(renderNode) {
    const renderDecode = String(findWidget(renderNode, "modular_decode")?.value || "low_ram");
    const visited = new Set();
    const queue = [...getLinxTargets(renderNode)];
    while (queue.length > 0) {
        const node = queue.shift();
        if (!node || visited.has(node.id)) {
            continue;
        }
        visited.add(node.id);
        if (node.comfyClass === "IAMCCS-SuperNodes AU+IMG2VID Exec VAE") {
            setWidgetValue(node, "decode_mode", renderDecode);
            setWidgetValue(node, "ui_preset", "custom");
            applyVaeDecodeModeVisibility(node);
            app.graph.setDirtyCanvas(true, true);
            continue;
        }
        if (node.comfyClass === "IAMCCS-SuperNodes Second Stage") {
            queue.push(...getLinxTargets(node));
        }
    }
}

function applyPresetConfig(node, nodeName) {
    const config = PRESET_CONFIGS[nodeName];
    if (!config) {
        return;
    }
    const presetWidget = findWidget(node, config.presetWidget);
    if (!presetWidget) {
        return;
    }
    const presetName = String(presetWidget.value || config.defaultPreset || "custom");
    if (presetName !== "custom") {
        const values = config.values?.[presetName] || {};
        Object.entries(values).forEach(([widgetName, widgetValue]) => setWidgetValue(node, widgetName, widgetValue));
    }
    const visibilityMap = config.visibility?.[presetName] || {};
    Object.entries(visibilityMap).forEach(([widgetName, isVisible]) => setWidgetVisibility(findWidget(node, widgetName), !!isVisible));
    if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec VAE") {
        applyVaeDecodeModeVisibility(node);
    }
    fitNodeToWidgets(node);
}

function getSerializableWidgets(node) {
    return (node.widgets || []).filter((widget) => widget && widget.serialize !== false);
}

function rehydrateSerializedWidgets(node, serializedValues) {
    if (!Array.isArray(serializedValues) || !node.widgets?.length) {
        return;
    }

    const widgets = getSerializableWidgets(node);
    const count = Math.min(widgets.length, serializedValues.length);
    for (let index = 0; index < count; index += 1) {
        const widget = widgets[index];
        if (!widget) {
            continue;
        }
        widget.value = serializedValues[index];
    }
}

function addSectionButton(node, group, index) {
    // Duplicate guard: if a button for this section key is already in the widget list,
    // return it immediately.  This prevents double-insertion if onNodeCreated or
    // enhanceNodeLayout is called more than once on the same node instance.
    const existing = node.widgets?.find((w) => w._iamccsSectionKey === group.key);
    if (existing) {
        return existing;
    }

    node.properties = node.properties || {};
    const propKey = `iamccs_section_${group.key}`;
    if (node.properties[propKey] === undefined) {
        node.properties[propKey] = true;
    }

    const sectionButton = node.addWidget("button", "", "", () => {
        node.properties[propKey] = !node.properties[propKey];
        applyGroupVisibility(node, group, propKey, sectionButton);
    });
    // IMPORTANT: do NOT set serialize = false.
    // Section buttons must occupy a slot in widgets_values so that ComfyUI's
    // index-based configure() assignment keeps real widget values aligned.
    // Their value (empty string / null) is harmless.
    sectionButton._iamccsSectionKey = group.key;
    sectionButton.computeSize = (width) => [width || 280, 28];
    sectionButton.label = group.label;
    sectionButton.options = { bgcolor: group.color || "#405060", color: "#f4f7fa" };

    applyGroupVisibility(node, group, propKey, sectionButton);
    insertWidget(node, sectionButton, index);
    return sectionButton;
}

function applyGroupVisibility(node, group, propKey, button) {
    const isExpanded = !!node.properties[propKey];
    button.name = `${isExpanded ? "▼" : "▶"} ${group.label}`;
    for (const widgetName of group.widgets) {
        setWidgetVisibility(findWidget(node, widgetName), isExpanded);
    }
    fitNodeToWidgets(node);
    app.graph.setDirtyCanvas(true, true);
}

function applyPlannerModeVisibility(node) {
    if (node.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
        return;
    }
    const planningMode = String(findWidget(node, "planning_mode")?.value || "manual_segment_seconds");
    const planningExpanded = !!node.properties?.iamccs_section_planning;
    const showSegmentPreset = planningExpanded && planningMode === "explicit_preset_seconds";
    setWidgetVisibility(findWidget(node, "segment_preset"), showSegmentPreset);
    fitNodeToWidgets(node);
}

function refreshNodeLayoutState(node, nodeName) {
    const groups = NODE_GROUPS[nodeName] || [];
    for (const group of groups) {
        const button = node.widgets?.find((widget) => widget?._iamccsSectionKey === group.key);
        if (!button) {
            continue;
        }
        applyGroupVisibility(node, group, `iamccs_section_${group.key}`, button);
    }

    const config = PRESET_CONFIGS[nodeName];
    if (config) {
        const presetWidget = findWidget(node, config.presetWidget);
        const presetName = String(presetWidget?.value || config.defaultPreset || "custom");
        const visibilityMap = config.visibility?.[presetName] || {};
        Object.entries(visibilityMap).forEach(([widgetName, isVisible]) => setWidgetVisibility(findWidget(node, widgetName), !!isVisible));
    }

    if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec VAE") {
        applyVaeDecodeModeVisibility(node);
    } else if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
        applyPlannerModeVisibility(node);
    } else {
        fitNodeToWidgets(node);
    }
}

function addPlannerChip(node) {
    if (!node.widgets || node.widgets.some((widget) => widget?.name === "planner_chip_preview")) {
        return;
    }
    const chipWidget = {
        type: "custom",
        name: "planner_chip_preview",
        serialize: false,
        computeSize(width) {
            return [Math.max(220, (width || 260) - 20), 30];
        },
        draw(ctx, widget, nodeRef, widgetWidth, y) {
            const text = nodeRef.properties?.iamccsPlannerChip || "";
            if (!text) {
                return;
            }
            const x = 12;
            const w = Math.max(180, widgetWidth - 24);
            const h = 22;
            const top = y + 4;

            ctx.save();
            ctx.fillStyle = "#18312b";
            ctx.strokeStyle = "#6ea88d";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.roundRect(x, top, w, h, 9);
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = "#d8efe2";
            ctx.font = "12px Segoe UI";
            ctx.textBaseline = "middle";
            ctx.fillText(text, x + 10, top + h / 2 + 0.5);
            ctx.restore();
        },
    };
    node.widgets.push(chipWidget);
}

function addStatusBox(node, propertyName, widgetName, fill, stroke, textColor) {
    if (!node.widgets || node.widgets.some((widget) => widget?.name === widgetName)) {
        return;
    }
    const statusWidget = {
        type: "custom",
        name: widgetName,
        serialize: false,
        computeSize(width) {
            return [Math.max(220, (width || 260) - 20), 42];
        },
        draw(ctx, widget, nodeRef, widgetWidth, y) {
            const text = nodeRef.properties?.[propertyName] || "";
            if (!text) {
                return;
            }
            const x = 12;
            const w = Math.max(180, widgetWidth - 24);
            const h = 34;
            const top = y + 4;

            ctx.save();
            ctx.fillStyle = fill;
            ctx.strokeStyle = stroke;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.roundRect(x, top, w, h, 10);
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = textColor;
            ctx.font = "12px Segoe UI";
            ctx.textBaseline = "middle";
            ctx.fillText(text, x + 10, top + h / 2 + 0.5);
            ctx.restore();
        },
    };
    node.widgets.push(statusWidget);
}

function addMultiLineStatusBox(node, propertyName, widgetName, fill, stroke, textColor) {
    if (!node.widgets || node.widgets.some((widget) => widget?.name === widgetName)) {
        return;
    }
    const statusWidget = {
        type: "custom",
        name: widgetName,
        serialize: false,
        computeSize(width) {
            return [Math.max(220, (width || 260) - 20), 78];
        },
        draw(ctx, widget, nodeRef, widgetWidth, y) {
            const text = nodeRef.properties?.[propertyName] || "";
            if (!text) {
                return;
            }
            const lines = String(text).split("\n").filter(Boolean).slice(0, 4);
            const x = 12;
            const w = Math.max(180, widgetWidth - 24);
            const h = 70;
            const top = y + 4;

            ctx.save();
            ctx.fillStyle = fill;
            ctx.strokeStyle = stroke;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.roundRect(x, top, w, h, 10);
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = textColor;
            ctx.font = "12px Segoe UI";
            ctx.textBaseline = "top";
            lines.forEach((line, index) => {
                ctx.fillText(line, x + 10, top + 8 + index * 15);
            });
            ctx.restore();
        },
    };
    node.widgets.push(statusWidget);
}

function enhanceNodeLayout(node, nodeName) {
    const groups = NODE_GROUPS[nodeName] || [];
    if (!groups.length || !node.widgets?.length) {
        return;
    }

    if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
        addPlannerChip(node);
        addMultiLineStatusBox(node, "iamccsPlannerDetails", "planner_details_preview", "#24362d", "#72ab8e", "#e4f2e8");
    }
    if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        addStatusBox(node, "iamccsRenderStatus", "render_status_preview", "#1e2f3f", "#6f93ba", "#e4edf7");
    }
    if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec VAE") {
        applyVaeDecodeModeVisibility(node);
    }

    for (let index = groups.length - 1; index >= 0; index -= 1) {
        const group = groups[index];
        const firstWidgetIndex = node.widgets.findIndex((widget) => group.widgets.includes(widget?.name));
        if (firstWidgetIndex >= 0) {
            addSectionButton(node, group, firstWidgetIndex);
        }
    }

    applyPresetConfig(node, nodeName);
    fitNodeToWidgets(node);
}

app.registerExtension({
    name: "IAMCCS.SuperNodesExecUI",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        const nodeName = nodeData?.name;
        if (!NODE_GROUPS[nodeName]) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            // MUST run synchronously, BEFORE configure() assigns widgets_values by index.
            // Section buttons inserted here occupy their index slots so that the null
            // placeholders in widgets_values land on buttons, not on real widgets.
            // (Using setTimeout here was the root cause of NaN on load / after undo.)
            enhanceNodeLayout(this, nodeName);
            const config = PRESET_CONFIGS[nodeName];
            const presetWidget = config ? findWidget(this, config.presetWidget) : null;
            if (presetWidget) {
                const originalCallback = presetWidget.callback;
                presetWidget.callback = (...args) => {
                    originalCallback?.apply(presetWidget, args);
                    applyPresetConfig(this, nodeName);
                    if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
                        syncDownstreamVaeDecodeModes(this);
                    }
                    if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec VAE") {
                        applyVaeDecodeModeVisibility(this);
                    }
                    app.graph.setDirtyCanvas(true, true);
                };
            }
            if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
                const modularDecodeWidget = findWidget(this, "modular_decode");
                if (modularDecodeWidget) {
                    const originalCallback = modularDecodeWidget.callback;
                    modularDecodeWidget.callback = (...args) => {
                        originalCallback?.apply(modularDecodeWidget, args);
                        syncDownstreamVaeDecodeModes(this);
                        app.graph.setDirtyCanvas(true, true);
                    };
                    syncDownstreamVaeDecodeModes(this);
                }
            }
            if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
                const planningModeWidget = findWidget(this, "planning_mode");
                if (planningModeWidget) {
                    const originalCallback = planningModeWidget.callback;
                    planningModeWidget.callback = (...args) => {
                        originalCallback?.apply(planningModeWidget, args);
                        applyPlannerModeVisibility(this);
                        app.graph.setDirtyCanvas(true, true);
                    };
                }
                applyPlannerModeVisibility(this);
            }
            if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec VAE") {
                const decodeModeWidget = findWidget(this, "decode_mode");
                if (decodeModeWidget) {
                    const originalCallback = decodeModeWidget.callback;
                    decodeModeWidget.callback = (...args) => {
                        originalCallback?.apply(decodeModeWidget, args);
                        applyVaeDecodeModeVisibility(this);
                        app.graph.setDirtyCanvas(true, true);
                    };
                }
            }
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const result = onConfigure?.apply(this, arguments);
            enhanceNodeLayout(this, nodeName);
            rehydrateSerializedWidgets(this, info?.widgets_values);
            refreshNodeLayoutState(this, nodeName);
            return result;
        };

        if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                this.properties = this.properties || {};
                if (Array.isArray(message?.planner_chip) && message.planner_chip.length > 0) {
                    this.properties.iamccsPlannerChip = String(message.planner_chip[0] || "");
                }
                const details = [];
                const duration = Array.isArray(message?.planned_duration_seconds) && message.planned_duration_seconds.length > 0
                    ? message.planned_duration_seconds[0]
                    : Array.isArray(message?.duration_seconds) && message.duration_seconds.length > 0
                        ? message.duration_seconds[0]
                        : null;
                const totalFrames = Array.isArray(message?.planned_total_frames) && message.planned_total_frames.length > 0
                    ? message.planned_total_frames[0]
                    : Array.isArray(message?.total_frames) && message.total_frames.length > 0
                        ? message.total_frames[0]
                        : null;
                const segments = Array.isArray(message?.planned_segment_count) && message.planned_segment_count.length > 0
                    ? message.planned_segment_count[0]
                    : Array.isArray(message?.segment_count) && message.segment_count.length > 0
                        ? message.segment_count[0]
                        : null;
                if (duration !== null) {
                    details.push(`duration ${Number(duration || 0).toFixed(2)}s`);
                }
                if (totalFrames !== null) {
                    details.push(`total ${totalFrames}f`);
                }
                if (segments !== null) {
                    details.push(`segments ${segments}`);
                }
                if (Array.isArray(message?.recommended_overlap_frames) && message.recommended_overlap_frames.length > 0) {
                    details.push(`overlap ${message.recommended_overlap_frames[0]}f`);
                }
                if (Array.isArray(message?.recommended_audio_left_context_s) && message.recommended_audio_left_context_s.length > 0) {
                    details.push(`left ctx ${Number(message.recommended_audio_left_context_s[0] || 0).toFixed(2)}s`);
                }
                if (Array.isArray(message?.planning_report) && message.planning_report.length > 0) {
                    details.push(String(message.planning_report[0] || ""));
                }
                if (details.length > 0) {
                    this.properties.iamccsPlannerDetails = details.join("\n");
                    app.graph.setDirtyCanvas(true, true);
                }
            };
        }

        if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                if (Array.isArray(message?.report) && message.report.length > 0) {
                    this.properties = this.properties || {};
                    this.properties.iamccsRenderStatus = String(message.report[0] || "").split("\n")[0];
                    app.graph.setDirtyCanvas(true, true);
                }
            };
        }
    },
});
