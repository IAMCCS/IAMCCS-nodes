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
const SEGMENT_PLANNER_PREVIEW_WIDGET = "iamccs_segmentplanner_live_preview";
const SEGMENT_PLANNER_DETAILS_WIDGET = "iamccs_segmentplanner_live_details";
const SEGMENT_PLANNER_COPY_BUTTON = "iamccs_segmentplanner_copy_button";

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

function ensurePlannerNodeSize(node) {
    try {
        const width = Math.max(460, Number(node.size?.[0] || 0));
        const height = Math.max(520, Number(node.size?.[1] || 0));
        node.size = [width, height];
    } catch {
        // ignore
    }
}

function updateSegmentPlannerPreview(node) {
    const wSong = getWidget(node, "song_duration_s");
    const wFps = getWidget(node, "fps");
    const wSeg = getWidget(node, "segment_duration_s");
    const wPlanning = getWidget(node, "planning_mode");
    const wProfile = getWidget(node, "content_profile");
    const wOverlap = getWidget(node, "overlap_frames");
    const wRound = getWidget(node, "ltx_round_mode");
    const wIndex = getWidget(node, "segment_index");
    if (!wSong || !wFps || !wSeg || !wPlanning || !wProfile || !wOverlap || !wRound || !wIndex) return;

    const songDuration = clampNumber(wSong.value, 0.01, 36000);
    const fps = clampNumber(wFps.value, 0.001, 240.0);
    const segmentDuration = clampNumber(wSeg.value, 0.01, 3600.0);
    const planningMode = String(wPlanning.value || "manual_segment_seconds");
    const contentProfile = String(wProfile.value || "videoclip");
    const overlapFrames = clampNumber(wOverlap.value, 0, 4096);
    const roundMode = String(wRound.value || "up");
    const segmentIndex = Math.max(0, Math.trunc(Number(wIndex.value || 0)));

    const rec = contentProfile === "monologue"
        ? { baseTargetS: 15.0, minS: 8.0, maxS: 15.0, overlap: 13, leftContext: 0.75, preset: "monologue_audio_24fps" }
        : { baseTargetS: 10.0, minS: 5.0, maxS: 10.0, overlap: 9, leftContext: 0.5, preset: "videoclip_audio_24fps" };

    const effectiveSegmentDuration = planningMode === "auto_profile"
        ? Math.max(rec.minS, Math.min(rec.maxS, songDuration / Math.max(1, Math.ceil(songDuration / rec.baseTargetS))))
        : segmentDuration;
    const effectiveOverlapFrames = planningMode === "auto_profile" ? rec.overlap : overlapFrames;

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
    const currentRaw = clampedIndex === 0 ? firstRaw : nextRaw;
    const currentStartS = currentStart / fps;
    const currentEndS = currentEnd / fps;

    const summary = [
        `total=${totalFrames}f`,
        `unique=${uniqueFrames}f`,
        `mode=${planningMode}`,
        `profile=${contentProfile}`,
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
        `content_profile = ${contentProfile}`,
        `overlap_frames = ${overlapFrames}`,
        `effective_overlap_frames = ${effectiveOverlapFrames}`,
        `ltx_round_mode = ${roundMode}`,
        `segment_index = ${clampedIndex}`,
        `total_frames = ${totalFrames}`,
        `unique_segment_frames = ${uniqueFrames}`,
        `first_segment_raw_frames = ${firstRaw}`,
        `continuation_raw_frames = ${nextRaw}`,
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
        getWidget(node, "planning_mode"),
        getWidget(node, "content_profile"),
        getWidget(node, "overlap_frames"),
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

        if (name !== TIMEFRAME_TYPE && name !== VALIDATOR_TYPE && name !== SEGMENT_PLANNER_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            if (name === SEGMENT_PLANNER_TYPE) {
                installSegmentPlannerSync(this);
            } else {
                // Snap to 8n+1 on both nodes, since LTX-2 VAE encode requires it.
                installSecondsLengthSync(this, { snapRule: true });
            }
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            if (name === SEGMENT_PLANNER_TYPE) {
                installSegmentPlannerSync(this);
                updateSegmentPlannerPreview(this);
            }
            return r;
        };
    },
});
