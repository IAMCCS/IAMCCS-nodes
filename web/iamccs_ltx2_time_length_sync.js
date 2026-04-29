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
const SEGMENT_PLANNER_TYPE = "IAMCCS_SegmentPlanner";
const SEGMENT_PLANNER_LINKED_TYPE = "IAMCCS_SegmentPlannerLinked";
const SEGMENT_PLANNER_SETTINGS_TYPE = "IAMCCS_SegmentPlannerSettings";
const SEGMENT_PLANNER_PREVIEW_WIDGET = "iamccs_segmentplanner_live_preview";
const SEGMENT_PLANNER_DETAILS_WIDGET = "iamccs_segmentplanner_live_details";
const SEGMENT_PLANNER_COPY_BUTTON = "iamccs_segmentplanner_copy_button";
const SEGMENT_PLANNER_SETTINGS_REPORT_WIDGET = "iamccs_segmentplanner_settings_live_report";

console.log("[IAMCCS LTX2] Loading seconds/length sync...");

function getWidget(node, name) {
    if (!node?.widgets?.length) return null;
    return node.widgets.find(w => w?.name === name || w?.label === name) || null;
}

function getInput(node, name) {
    if (!node?.inputs?.length) return null;
    return node.inputs.find(i => i?.name === name) || null;
}

function getNodeById(nodeId) {
    if (nodeId == null) return null;
    return app?.graph?._nodes_by_id?.[nodeId] || app?.graph?._nodes?.find(n => n?.id === nodeId) || null;
}

function getLinkedSource(node, inputName) {
    const input = getInput(node, inputName);
    const linkId = input?.link;
    if (linkId == null) return null;

    const link = app?.graph?.links?.[linkId];
    if (!link) return null;

    const originId = Array.isArray(link) ? link[1] : link.origin_id;
    const originSlot = Array.isArray(link) ? link[2] : link.origin_slot;
    const sourceNode = getNodeById(originId);
    if (!sourceNode) return null;

    // Defensive: outputs must exist and be an array, originSlot must be valid
    let sourceOutput = null;
    if (Array.isArray(sourceNode.outputs) && originSlot != null && originSlot >= 0 && originSlot < sourceNode.outputs.length) {
        sourceOutput = sourceNode.outputs[originSlot];
    }

    return {
        sourceNode,
        sourceOutput,
    };
}

function getLinkedPrimitiveValue(node, inputName) {
    const source = getLinkedSource(node, inputName);
    if (!source?.sourceNode) return null;

    const sourceNode = source.sourceNode;
    const outputName = String(source.sourceOutput?.name || "");
    const candidateWidgetNames = [outputName, inputName, "value", "float", "int", "string", "text"];

    // Defensive: try to get value from output if present
    if (source.sourceOutput && typeof source.sourceOutput.value !== "undefined") {
        return source.sourceOutput.value;
    }

    for (const widgetName of candidateWidgetNames) {
        const widget = getWidget(sourceNode, widgetName);
        if (widget?.value != null) return widget.value;
    }

    if (Array.isArray(sourceNode.widgets) && sourceNode.widgets.length > 0) {
        const widgetValue = sourceNode.widgets[0]?.value;
        if (widgetValue != null) return widgetValue;
    }

    return null;
}

function clampNumber(v, min, max) {
    const n = Number(v);
    if (!Number.isFinite(n)) return min;
    return Math.max(min, Math.min(max, n));
}

function numbersClose(a, b, epsilon = 0.0001) {
    return Math.abs(Number(a) - Number(b)) <= epsilon;
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

function snapLengthToLtx2Rule(length, mode = "up") {
    const n = Math.max(1, Math.round(Number(length) || 1));
    const rem = (n - 1) % 8;
    if (rem === 0) return n;
    const down = Math.max(1, n - rem);
    const up = n + (8 - rem);
    if (mode === "down") return down;
    if (mode === "nearest") return (up - n) <= (n - down) ? up : down;
    return up;
}

function ensurePlannerPreviewWidget(node) {
    let widget = getWidget(node, SEGMENT_PLANNER_PREVIEW_WIDGET);
    if (widget) return widget;

    widget = node.addWidget("text", "Live Preview", "", () => {}, { multiline: true });
    widget.name = SEGMENT_PLANNER_PREVIEW_WIDGET;
    widget.serialize = false;
    try {
        widget.inputEl?.setAttribute?.("readonly", true);
    } catch {
        // ignore
    }
    return widget;
}

function ensurePlannerDetailsWidget(node) {
    let widget = getWidget(node, SEGMENT_PLANNER_DETAILS_WIDGET);
    if (widget) return widget;

    widget = node.addWidget("text", "Plan Details", "", () => {}, { multiline: true });
    widget.name = SEGMENT_PLANNER_DETAILS_WIDGET;
    widget.serialize = false;
    try {
        widget.inputEl?.setAttribute?.("readonly", true);
    } catch {
        // ignore
    }
    return widget;
}

function ensurePlannerCopyButton(node) {
    let widget = getWidget(node, SEGMENT_PLANNER_COPY_BUTTON);
    if (widget) return widget;

    widget = node.addWidget("button", "Copy Planner Report", null, async () => {
        try {
            const summary = String(node.properties?.iamccs_segmentplanner_live_preview || "");
            const details = String(node.properties?.iamccs_segmentplanner_live_details || "");
            const text = [summary, details].filter(Boolean).join("\n\n");
            await navigator.clipboard.writeText(text);
        } catch {
            // ignore
        }
    });
    widget.name = SEGMENT_PLANNER_COPY_BUTTON;
    widget.serialize = false;
    return widget;
}

function ensurePlannerSettingsReportWidget(node) {
    let widget = getWidget(node, SEGMENT_PLANNER_SETTINGS_REPORT_WIDGET);
    if (widget) {
        widget.label = "Live Preview";
        return widget;
    }

    widget = node.addWidget("text", "Live Preview", "", () => {}, { multiline: true });
    widget.name = SEGMENT_PLANNER_SETTINGS_REPORT_WIDGET;
    widget.serialize = false;
    try {
        widget.inputEl?.setAttribute?.("readonly", true);
    } catch {
        // ignore
    }
    return widget;
}

function setWidgetVisibility(widget, visible) {
    if (!widget || widget.type === "converted-widget") return;

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

function applySegmentPlannerSettingsVisibility(node) {
    const wPlanning = getWidget(node, "planning_mode");
    const wSeg = getWidget(node, "segment_duration_s");
    const wPreset = getWidget(node, "segment_preset") || getWidget(node, "content_profile");
    if (!wPlanning || !wSeg || !wPreset) return;

    const planningMode = String(wPlanning.value || "manual_segment_seconds");
    const explicitPresetMode = planningMode === "explicit_preset_seconds" || planningMode === "auto_profile";

    setWidgetVisibility(wSeg, !explicitPresetMode);
    setWidgetVisibility(wPreset, explicitPresetMode);
}

function ensurePlannerNodeSize(node) {
    try {
        const width = Math.max(460, Number(node.size?.[0] || 0));
        const height = Math.max(node?.type === SEGMENT_PLANNER_SETTINGS_TYPE ? 300 : 520, Number(node.size?.[1] || 0));
        node.size = [width, height];
    } catch {
        // ignore
    }
}

function refreshLinkedPlannerPreviews() {
    try {
        const nodes = app?.graph?._nodes || [];
        for (const node of nodes) {
            if (node?.type === SEGMENT_PLANNER_LINKED_TYPE) {
                updateSegmentPlannerPreview(node);
            }
        }
    } catch {
        // ignore
    }
}

function updateSegmentPlannerSettingsReport(node) {
    const wSeg = getWidget(node, "segment_duration_s");
    const wPlanning = getWidget(node, "planning_mode");
    const wProfile = getWidget(node, "segment_preset") || getWidget(node, "content_profile");
    const wOverlap = getWidget(node, "overlap_frames");
    const wAutoSync = getWidget(node, "auto_sync_overlap");
    if (!wSeg || !wPlanning || !wProfile || !wOverlap || !wAutoSync) return;

    applySegmentPlannerSettingsVisibility(node);

    const segmentDuration = clampNumber(wSeg.value, 0.01, 3600.0);
    const planningMode = String(wPlanning.value || "manual_segment_seconds");
    const segmentPreset = String(wProfile.value || "10sec");
    const autoSyncOverlap = Boolean(wAutoSync.value);

    const rec = getPlannerProfile(segmentPreset);
    const explicitPresetMode = planningMode === "explicit_preset_seconds" || planningMode === "auto_profile";
    const autoDurationProfile = !explicitPresetMode ? getAutoOverlapProfileForDuration(segmentDuration) : null;
    const effectiveSegmentDuration = explicitPresetMode ? rec.segmentSeconds : segmentDuration;
    const effectiveOverlapFrames = clampNumber(wOverlap.value, 0, 4096);

    if (explicitPresetMode && !numbersClose(wSeg.value, effectiveSegmentDuration)) {
        wSeg.value = effectiveSegmentDuration;
    }

    const previewFps = getGraphFpsOrDefault(24);
    const uniqueFrames = Math.max(1, Math.round(effectiveSegmentDuration * previewFps));
    const firstRawFrames = snapLengthToLtx2Rule(uniqueFrames, "up");
    const continuationRawFrames = snapLengthToLtx2Rule(uniqueFrames + effectiveOverlapFrames, "up");

    const report = [
        `segment_duration_s = ${effectiveSegmentDuration.toFixed(3)}`,
        `planning_mode = ${planningMode}`,
        `segment_preset = ${segmentPreset}`,
        `overlap_frames = ${effectiveOverlapFrames}`,
        `preview_fps = ${previewFps}`,
        `unique_segment_frames = ${uniqueFrames}`,
        `first_segment_raw_frames = ${firstRawFrames}`,
        `continuation_raw_frames = ${continuationRawFrames}`,
        `auto_sync_overlap = ${autoSyncOverlap}`,
        `auto_duration_profile = ${autoDurationProfile?.profileLabel || "none"}`,
        `overlap_policy = manual/default_9`,
        `recommended_extension_preset = ${rec.preset}`,
    ].join("\n");

    const widget = ensurePlannerSettingsReportWidget(node);
    widget.value = report;
    node.properties = node.properties || {};
    node.properties.iamccs_segmentplanner_settings_live_report = report;
    ensurePlannerNodeSize(node);
    try {
        node.setDirtyCanvas(true, true);
    } catch {
        // ignore
    }
}

function getPlannerProfile(presetOrLegacyProfile) {
    const value = String(presetOrLegacyProfile || "15sec");
    if (value === "20sec") {
        return { segmentSeconds: 20.0, overlap: 9, leftContext: 1.0, preset: "monologue_audio_24fps", profileLabel: "20sec" };
    }
    if (value === "15sec" || value === "monologue") {
        return { segmentSeconds: 15.0, overlap: 9, leftContext: 0.75, preset: "monologue_audio_24fps", profileLabel: "15sec" };
    }
    if (value === "5sec") {
        return { segmentSeconds: 5.0, overlap: 9, leftContext: 0.25, preset: "videoclip_audio_24fps", profileLabel: "5sec" };
    }
    return { segmentSeconds: 10.0, overlap: 9, leftContext: 0.5, preset: "videoclip_audio_24fps", profileLabel: "10sec" };
}

function getAutoOverlapProfileForDuration(segmentDuration) {
    const duration = Number(segmentDuration);
    if (!Number.isFinite(duration)) return null;
    const profiles = [
        getPlannerProfile("5sec"),
        getPlannerProfile("10sec"),
        getPlannerProfile("15sec"),
        getPlannerProfile("20sec"),
    ];
    return profiles.find(profile => Math.abs(duration - profile.segmentSeconds) <= 0.1) || null;
}

function updateSegmentPlannerPreview(node) {
    const wSong = getWidget(node, "song_duration_s");
    const wFps = getWidget(node, "fps");
    const wSeg = getWidget(node, "segment_duration_s");
    const wPlanning = getWidget(node, "planning_mode") || getWidget(node, "planning_mode_in");
    const wProfile = getWidget(node, "segment_preset") || getWidget(node, "segment_preset_in") || getWidget(node, "content_profile");
    const wOverlap = getWidget(node, "overlap_frames") || getWidget(node, "overlap_frames_in");
    const wRound = getWidget(node, "ltx_round_mode");
    const wIndex = getWidget(node, "segment_index");
    if (!wSong || !wFps || !wSeg || !wPlanning || !wProfile || !wOverlap || !wRound || !wIndex) return;

    const songDuration = clampNumber(wSong.value, 0.01, 36000);
    const fps = clampNumber(wFps.value, 0.001, 240.0);
    const linkedSegmentDuration = node?.type === SEGMENT_PLANNER_LINKED_TYPE ? getLinkedPrimitiveValue(node, "segment_duration_s") : null;
    const linkedPlanningMode = node?.type === SEGMENT_PLANNER_LINKED_TYPE ? getLinkedPrimitiveValue(node, "planning_mode_in") : null;
    const linkedSegmentPreset = node?.type === SEGMENT_PLANNER_LINKED_TYPE ? getLinkedPrimitiveValue(node, "segment_preset_in") : null;
    const linkedOverlapFrames = node?.type === SEGMENT_PLANNER_LINKED_TYPE ? getLinkedPrimitiveValue(node, "overlap_frames_in") : null;

    const segmentDuration = clampNumber(linkedSegmentDuration ?? wSeg.value, 0.01, 3600.0);
    const planningMode = String(linkedPlanningMode ?? wPlanning.value || "manual_segment_seconds");
    const segmentPreset = String(linkedSegmentPreset ?? wProfile.value || "15sec");
    const inputOverlapFrames = clampNumber(linkedOverlapFrames ?? wOverlap.value, 0, 4096);
    const roundMode = String(wRound.value || "up");
    const segmentIndex = Math.max(0, Math.trunc(Number(wIndex.value || 0)));

    const rec = getPlannerProfile(segmentPreset);
    const explicitPresetMode = planningMode === "explicit_preset_seconds" || planningMode === "auto_profile";
    const autoDurationProfile = !explicitPresetMode ? getAutoOverlapProfileForDuration(segmentDuration) : null;
    const effectiveSegmentDuration = explicitPresetMode ? rec.segmentSeconds : segmentDuration;
    const effectiveOverlapFrames = inputOverlapFrames;

    if (explicitPresetMode && node?.type !== SEGMENT_PLANNER_LINKED_TYPE && !numbersClose(wSeg.value, effectiveSegmentDuration)) {
        wSeg.value = effectiveSegmentDuration;
    }
    const overlapFrames = clampNumber(wOverlap.value, 0, 4096);

    const totalFrames = Math.max(1, Math.round(songDuration * fps));
    const uniqueFrames = Math.max(1, Math.round(effectiveSegmentDuration * fps));
    const firstRaw = snapLengthToLtx2Rule(uniqueFrames, roundMode);
    const nextRaw = snapLengthToLtx2Rule(uniqueFrames + effectiveOverlapFrames, roundMode);
    const segments = Math.max(1, Math.ceil(totalFrames / uniqueFrames));
    const loops = Math.max(0, segments - 1);
    const usedBeforeLast = uniqueFrames * Math.max(0, segments - 1);
    const lastUnique = Math.max(1, totalFrames - usedBeforeLast);
    const clampedIndex = Math.min(segmentIndex, Math.max(0, segments - 1));
    const currentStart = uniqueFrames * clampedIndex;
    const currentUnique = Math.max(1, Math.min(uniqueFrames, totalFrames - currentStart));
    const currentEnd = Math.min(totalFrames, currentStart + currentUnique);
    const currentRemaining = Math.max(0, totalFrames - currentEnd);
    const lastRaw = segments <= 1 ? firstRaw : snapLengthToLtx2Rule(lastUnique + effectiveOverlapFrames, roundMode);
    const currentRaw = clampedIndex === 0 ? firstRaw : (clampedIndex >= segments - 1 ? lastRaw : nextRaw);
    const currentStartS = currentStart / fps;
    const currentEndS = currentEnd / fps;

    const summary = [
        `total=${totalFrames}f`,
        `unique=${uniqueFrames}f`,
        `mode=${planningMode}`,
        `profile=${segmentPreset}`,
        `first_raw=${firstRaw}f`,
        `next_raw=${nextRaw}f`,
        `segments=${segments}`,
        `loops=${loops}`,
        `last_unique=${lastUnique}f`,
        `seg=${clampedIndex}`,
        `cur_raw=${currentRaw}f`,
        `cur_unique=${currentUnique}f`,
        `cur=[${currentStart}..${currentEnd})`,
        `remain=${currentRemaining}f`,
    ].join(" | ");

    const details = [
        `song_duration_s = ${songDuration.toFixed(3)}`,
        `fps = ${fps.toFixed(3)}`,
        `segment_duration_s = ${segmentDuration.toFixed(3)}`,
        `effective_segment_duration_s = ${effectiveSegmentDuration.toFixed(3)}`,
        `planning_mode = ${planningMode}`,
        `segment_preset = ${segmentPreset}`,
        `overlap_frames = ${overlapFrames}`,
        `effective_overlap_frames = ${effectiveOverlapFrames}`,
        `overlap_policy = manual/default_9`,
        `ltx_round_mode = ${roundMode}`,
        `segment_index = ${clampedIndex}`,
        `total_frames = ${totalFrames}`,
        `unique_segment_frames = ${uniqueFrames}`,
        `first_segment_raw_frames = ${firstRaw}`,
        `continuation_raw_frames = ${nextRaw}`,
        `last_segment_raw_frames = ${lastRaw}`,
        `estimated_segments = ${segments}`,
        `continuation_loops = ${loops}`,
        `last_segment_unique_frames = ${lastUnique}`,
        `current_segment_raw_frames = ${currentRaw}`,
        `current_segment_unique_frames = ${currentUnique}`,
        `current_segment_start_frames = ${currentStart}`,
        `current_segment_end_frames = ${currentEnd}`,
        `current_remaining_frames_after = ${currentRemaining}`,
        `current_segment_start_s = ${currentStartS.toFixed(3)}`,
        `current_segment_end_s = ${currentEndS.toFixed(3)}`,
        `recommended_audio_left_context_s = ${rec.leftContext.toFixed(2)}`,
        `recommended_extension_preset = ${rec.preset}`,
        `auto_duration_profile = ${autoDurationProfile?.profileLabel || "none"}`,
    ].join("\n");

    const widget = ensurePlannerPreviewWidget(node);
    widget.value = summary;
    const detailsWidget = ensurePlannerDetailsWidget(node);
    detailsWidget.value = details;
    ensurePlannerCopyButton(node);
    ensurePlannerNodeSize(node);
    node.properties = node.properties || {};
    node.properties.iamccs_segmentplanner_live_preview = summary;
    node.properties.iamccs_segmentplanner_live_details = details;
    try {
        node.setDirtyCanvas(true, true);
    } catch {
        // ignore
    }
}

function installSegmentPlannerSync(node) {
    const widgets = [
        getWidget(node, "song_duration_s"),
        getWidget(node, "fps"),
        getWidget(node, "segment_duration_s"),
        getWidget(node, "planning_mode") || getWidget(node, "planning_mode_in"),
        getWidget(node, "segment_preset") || getWidget(node, "segment_preset_in") || getWidget(node, "content_profile"),
        getWidget(node, "overlap_frames") || getWidget(node, "overlap_frames_in"),
        getWidget(node, "ltx_round_mode"),
        getWidget(node, "segment_index"),
    ].filter(Boolean);

    if (!widgets.length || node._iamccsSegmentPlannerSyncInstalled) {
        updateSegmentPlannerPreview(node);
        return;
    }

    node._iamccsSegmentPlannerSyncInstalled = true;
    ensurePlannerPreviewWidget(node);
    ensurePlannerDetailsWidget(node);
    ensurePlannerCopyButton(node);
    ensurePlannerNodeSize(node);

    for (const widget of widgets) {
        const prev = widget.callback;
        widget.callback = function () {
            const r = prev?.apply(this, arguments);
            updateSegmentPlannerPreview(node);
            return r;
        };
    }

    updateSegmentPlannerPreview(node);
}

function installSegmentPlannerSettingsSync(node) {
    const widgets = [
        getWidget(node, "segment_duration_s"),
        getWidget(node, "planning_mode"),
        getWidget(node, "segment_preset"),
        getWidget(node, "overlap_frames"),
        getWidget(node, "auto_sync_overlap"),
    ].filter(Boolean);

    if (!widgets.length || node._iamccsSegmentPlannerSettingsSyncInstalled) {
        applySegmentPlannerSettingsVisibility(node);
        updateSegmentPlannerSettingsReport(node);
        return;
    }

    node._iamccsSegmentPlannerSettingsSyncInstalled = true;
    ensurePlannerSettingsReportWidget(node);
    ensurePlannerNodeSize(node);

    for (const widget of widgets) {
        const prev = widget.callback;
        widget.callback = function () {
            const r = prev?.apply(this, arguments);
            updateSegmentPlannerSettingsReport(node);
            refreshLinkedPlannerPreviews();
            return r;
        };
    }

    updateSegmentPlannerSettingsReport(node);
    refreshLinkedPlannerPreviews();
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

        if (name !== TIMEFRAME_TYPE && name !== VALIDATOR_TYPE && name !== SEGMENT_PLANNER_TYPE && name !== SEGMENT_PLANNER_LINKED_TYPE && name !== SEGMENT_PLANNER_SETTINGS_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            if (name === SEGMENT_PLANNER_TYPE || name === SEGMENT_PLANNER_LINKED_TYPE) {
                installSegmentPlannerSync(this);
            } else if (name === SEGMENT_PLANNER_SETTINGS_TYPE) {
                installSegmentPlannerSettingsSync(this);
            } else {
                // Snap to 8n+1 on both nodes, since LTX-2 VAE encode requires it.
                installSecondsLengthSync(this, { snapRule: true });
            }
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            if (name === SEGMENT_PLANNER_TYPE || name === SEGMENT_PLANNER_LINKED_TYPE) {
                installSegmentPlannerSync(this);
                updateSegmentPlannerPreview(this);
            } else if (name === SEGMENT_PLANNER_SETTINGS_TYPE) {
                installSegmentPlannerSettingsSync(this);
                updateSegmentPlannerSettingsReport(this);
            }
            return r;
        };
    },
});
