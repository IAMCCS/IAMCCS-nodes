import { app } from "../../../scripts/app.js";

const PRESET_CONFIGS = {
    "IAMCCS-SuperNodes AU+IMG2VID Exec Render": {
        presetWidget: "ui_preset",
        defaultPreset: "custom",
        values: {
            low_ram_safe: { backend_mode: "auto", vae_mode: "very_low_ram_disk", steps: 16, image_compression: 40, stitch_preset: "custom", overlap_side: "source", overlap_mode: "cut", start_frames_rule: "none", continuity_anchor_mode: "tail_only", anchor_refresh_interval: 2, anchor_image_strength: 0.2, anti_drift_mode: "rolling_adain", anti_drift_strength: 0.1, identity_persistence_strength: 0.0, generation_mode: "img2vid", media_mode: "auto_from_generation_mode", vram_flush: true, motion_intensity: 1.0, second_stage_mode: "off", stage2_model_policy: "stage2_model_if_connected", second_stage_reinject_strength: 0.0 },
            balanced: { backend_mode: "auto", vae_mode: "normal_tiled_vhs", steps: 20, image_compression: 28, stitch_preset: "custom", overlap_side: "source", overlap_mode: "cut", start_frames_rule: "none", continuity_anchor_mode: "tail_only", anchor_refresh_interval: 2, anchor_image_strength: 0.25, anti_drift_mode: "rolling_adain", anti_drift_strength: 0.12, identity_persistence_strength: 0.0, generation_mode: "img2vid", media_mode: "auto_from_generation_mode", vram_flush: false, motion_intensity: 1.0, second_stage_mode: "off", stage2_model_policy: "stage2_model_if_connected", second_stage_reinject_strength: 0.0 },
            high_quality: { backend_mode: "auto", vae_mode: "high_vram", steps: 24, image_compression: 20, stitch_preset: "custom", overlap_side: "source", overlap_mode: "cut", start_frames_rule: "none", continuity_anchor_mode: "tail_only", anchor_refresh_interval: 1, anchor_image_strength: 0.3, anti_drift_mode: "dual_reference_adain", anti_drift_strength: 0.16, identity_persistence_strength: 0.06, generation_mode: "img2vid", media_mode: "auto_from_generation_mode", vram_flush: false, motion_intensity: 1.0, second_stage_mode: "off", stage2_model_policy: "stage2_model_if_connected", second_stage_reinject_strength: 0.0 },
            fast_preview: { backend_mode: "single_best", vae_mode: "low_ram_disk", steps: 12, image_compression: 45, stitch_preset: "custom", overlap_side: "source", overlap_mode: "cut", start_frames_rule: "none", continuity_anchor_mode: "off", anchor_refresh_interval: 2, anti_drift_mode: "off", anti_drift_strength: 0.0, identity_persistence_strength: 0.0, generation_mode: "img2vid", media_mode: "auto_from_generation_mode", vram_flush: true, motion_intensity: 1.0, second_stage_mode: "off", stage2_model_policy: "stage2_model_if_connected", second_stage_reinject_strength: 0.0 },
            loop_lipsync_safe: { backend_mode: "loop_normal_vram", vae_mode: "normal_tiled_vhs", stitch_preset: "custom", overlap_side: "source", overlap_mode: "cut", start_frames_rule: "none", continuity_anchor_mode: "tail_only", anchor_refresh_interval: 2, anchor_image_strength: 0.2, anti_drift_mode: "rolling_adain", anti_drift_strength: 0.1, generation_mode: "img2vid", media_mode: "input_audio_img2vid", vram_flush: false, motion_intensity: 1.0, second_stage_mode: "off" },
            img2vid_generated_audio: { backend_mode: "single_best", generation_mode: "img2vid", media_mode: "generated_audio_img2vid", generated_media_duration_seconds: 10.0, continuity_anchor_mode: "off", stitch_preset: "custom", start_frames_rule: "none", vram_flush: false, motion_intensity: 1.0, second_stage_mode: "off" },
            t2v_generated_audio: { backend_mode: "single_best", generation_mode: "t2v", media_mode: "generated_audio_t2v", generated_media_duration_seconds: 10.0, continuity_anchor_mode: "off", stitch_preset: "custom", start_frames_rule: "none", vram_flush: false, motion_intensity: 1.0, second_stage_mode: "off" },
            img2vid_pure: { backend_mode: "single_best", generation_mode: "img2vid", media_mode: "img2vid_pure", generated_media_duration_seconds: 10.0, continuity_anchor_mode: "off", stitch_preset: "custom", start_frames_rule: "none", vram_flush: false, motion_intensity: 1.0, second_stage_mode: "off" },
            t2v_pure: { backend_mode: "single_best", generation_mode: "t2v", media_mode: "t2v_pure", generated_media_duration_seconds: 10.0, continuity_anchor_mode: "off", stitch_preset: "custom", start_frames_rule: "none", vram_flush: false, motion_intensity: 1.0, second_stage_mode: "off" },
            loop_img2vid_pure_normal_vram: { backend_mode: "loop_normal_vram", vae_mode: "normal_tiled_vhs", generation_mode: "img2vid", media_mode: "img2vid_pure", generated_media_duration_seconds: 20.0, continuity_anchor_mode: "tail_only", stitch_preset: "custom", overlap_side: "source", overlap_mode: "cut", start_frames_rule: "none", vram_flush: false, motion_intensity: 1.0, second_stage_mode: "off", anti_drift_mode: "rolling_adain", anti_drift_strength: 0.08 },
            loop_t2v_pure_normal_vram: { backend_mode: "loop_normal_vram", vae_mode: "normal_tiled_vhs", generation_mode: "t2v", media_mode: "t2v_pure", generated_media_duration_seconds: 20.0, continuity_anchor_mode: "tail_only", stitch_preset: "custom", overlap_side: "source", overlap_mode: "cut", start_frames_rule: "none", vram_flush: false, motion_intensity: 1.0, second_stage_mode: "off", anti_drift_mode: "rolling_adain", anti_drift_strength: 0.08 },
            loop_img2vid_pure_low_ram: { backend_mode: "loop_low_ram_disk", vae_mode: "low_ram_disk", generation_mode: "img2vid", media_mode: "img2vid_pure", generated_media_duration_seconds: 20.0, continuity_anchor_mode: "tail_only", stitch_preset: "custom", overlap_side: "source", overlap_mode: "cut", start_frames_rule: "none", vram_flush: true, motion_intensity: 1.0, second_stage_mode: "off", anti_drift_mode: "rolling_adain", anti_drift_strength: 0.08 },
            motion_controlled: { backend_mode: "auto", stitch_preset: "custom", overlap_side: "source", overlap_mode: "cut", start_frames_rule: "none", anchor_image_strength: 0.18, anti_drift_mode: "rolling_adain", anti_drift_strength: 0.08, motion_intensity: 1.35, image_strength: 0.78, image_compression: 32, second_stage_mode: "off" },
        },
        visibility: {
            low_ram_safe: { anchor_image_strength: true, anti_drift_mode: true, anti_drift_strength: true, identity_persistence_strength: false, output_root: true },
            balanced: { anchor_image_strength: true, anti_drift_mode: true, anti_drift_strength: true, identity_persistence_strength: false, output_root: true },
            high_quality: { anchor_image_strength: true, anti_drift_mode: true, anti_drift_strength: true, identity_persistence_strength: true, output_root: true },
            fast_preview: { anchor_image_strength: false, anti_drift_mode: false, anti_drift_strength: false, identity_persistence_strength: false, output_root: false },
            loop_lipsync_safe: { anchor_image_strength: true, anti_drift_mode: true, anti_drift_strength: true, identity_persistence_strength: false, output_root: true },
            img2vid_generated_audio: { anchor_image_strength: false, anti_drift_mode: false, anti_drift_strength: false, identity_persistence_strength: false, output_root: true },
            t2v_generated_audio: { anchor_image_strength: false, anti_drift_mode: false, anti_drift_strength: false, identity_persistence_strength: false, output_root: true },
            img2vid_pure: { anchor_image_strength: false, anti_drift_mode: false, anti_drift_strength: false, identity_persistence_strength: false, output_root: true },
            t2v_pure: { anchor_image_strength: false, anti_drift_mode: false, anti_drift_strength: false, identity_persistence_strength: false, output_root: true },
            loop_img2vid_pure_normal_vram: { anchor_image_strength: false, anti_drift_mode: true, anti_drift_strength: true, identity_persistence_strength: false, output_root: true },
            loop_t2v_pure_normal_vram: { anchor_image_strength: false, anti_drift_mode: true, anti_drift_strength: true, identity_persistence_strength: false, output_root: true },
            loop_img2vid_pure_low_ram: { anchor_image_strength: false, anti_drift_mode: true, anti_drift_strength: true, identity_persistence_strength: false, output_root: true },
            motion_controlled: { anchor_image_strength: true, anti_drift_mode: true, anti_drift_strength: true, identity_persistence_strength: false, output_root: true },
        },
    },
    "IAMCCS-SuperNodes AU+IMG2VID Exec VAE": {
        presetWidget: "ui_preset",
        defaultPreset: "custom",
        values: {
            very_low_ram_decode: { decode_mode: "very_low_ram_disk", vram_flush: true, jpg_quality: 90, crf: 22, tiled_tile_size: 256, tiled_overlap: 32 },
            low_ram_safe: { decode_mode: "very_low_ram_disk", vram_flush: true, jpg_quality: 92, crf: 21, tiled_tile_size: 256, tiled_overlap: 32 },
            balanced: { decode_mode: "normal_tiled_vhs", vram_flush: false, jpg_quality: 95, crf: 19, tiled_tile_size: 512, tiled_overlap: 64 },
            high_quality: { decode_mode: "high_vram", vram_flush: false, jpg_quality: 100, crf: 16, tiled_tile_size: 768, tiled_overlap: 96 },
            fast_preview: { decode_mode: "low_ram_disk", vram_flush: true, jpg_quality: 88, crf: 24, tiled_tile_size: 384, tiled_overlap: 48 },
        },
        visibility: {
            very_low_ram_decode: { frames_subdir: true, image_format: true, jpg_quality: true },
            low_ram_safe: { frames_subdir: true, image_format: true, jpg_quality: true },
            balanced: { frames_subdir: true, image_format: true, jpg_quality: true },
            high_quality: { frames_subdir: true, image_format: true, jpg_quality: true },
            fast_preview: { frames_subdir: true, image_format: true, jpg_quality: true },
        },
    },
};

const NODE_GROUPS = {
    "IAMCCS-SuperNodes AU+IMG2VID Exec Planner": [
        { key: "timeline", label: "Timeline", color: "#2f8f80", widgets: ["fps", "segment_seconds"] },
        { key: "planning", label: "Planning", color: "#a07b2c", widgets: ["planning_mode", "segment_preset", "overlap_frames", "ltx_round_mode"] },
    ],
    "IAMCCS-SuperNodes AU+IMG2VID Exec Render": [
        { key: "generation", label: "Generation Type", color: "#8a5ca0", widgets: ["generation_type", "generated_media_duration_seconds"] },
        { key: "preset", label: "Preset", color: "#557ea6", widgets: ["ui_preset"] },
        { key: "prompts", label: "Prompting", color: "#9a6b2f", widgets: ["positive_text", "negative_text"] },
        { key: "video", label: "Video", color: "#3f6fb0", widgets: ["width", "height"] },
        { key: "sampling", label: "Sampling", color: "#8352a6", widgets: ["steps", "cfg", "sampler_name", "seed", "max_shift", "base_shift", "sigma_terminal", "manual_sigmas", "image_strength", "motion_intensity", "image_compression"] },
        { key: "audio", label: "Audio", color: "#3d8a5c", widgets: ["audio_context_mode", "audio_left_context_s", "audio_right_context_s"] },
        { key: "stitch", label: "Stitch", color: "#a4673d", widgets: ["stitch_preset", "overlap_side", "overlap_mode", "start_frames_rule"] },
        { key: "anchor", label: "Anchor", color: "#b35c5c", widgets: ["continuity_anchor_mode", "anchor_refresh_interval", "anchor_image_strength", "anti_drift_mode", "anti_drift_strength", "identity_persistence_strength"] },
        { key: "modular", label: "Modular", color: "#46769a", widgets: ["vae_mode", "vram_flush", "downstream_stage_mode", "output_root"] },
        { key: "debug", label: "Debug", color: "#8a8a36", widgets: ["segment_overlay_mode", "segment_overlay_text"] },
        { key: "stage2", label: "Second Stage", color: "#8a5ca0", widgets: ["second_stage_mode", "stage2_model_policy", "second_stage_upscale_model", "second_stage_reinject_strength", "second_stage_cfg", "second_stage_manual_sigmas"] },
    ],
    "IAMCCS-SuperNodes Second Stage": [
        { key: "stage2", label: "Second Stage", color: "#8a5ca0", widgets: ["second_stage_mode", "stage2_model_policy", "second_stage_upscale_model", "second_stage_reinject_strength", "second_stage_cfg", "second_stage_manual_sigmas"] },
    ],
    IAMCCS_GC_AudioConcatSupernode: [
        { key: "concat", label: "Audio Concat", color: "#4e8f6b", widgets: ["concat_mode", "clip_durations_seconds", "gap_seconds", "intro_seconds", "outro_seconds"] },
    ],
    "IAMCCS-SuperNodes AU+IMG2VID Exec VAE": [
        { key: "preset", label: "Preset", color: "#557ea6", widgets: ["ui_preset"] },
        { key: "decode", label: "Decode", color: "#4b79a6", widgets: ["frame_rate", "decode_mode", "output_root", "frames_subdir", "image_format", "jpg_quality", "tiled_tile_size", "tiled_overlap"] },
        { key: "final", label: "Finalize", color: "#9b7441", widgets: ["filename_prefix", "crf", "pix_fmt", "trim_to_audio", "save_metadata", "vram_flush"] },
    ],
    "IAMCCS-SuperNodes AU+IMG2VID Exec Finalize": [
        { key: "output", label: "Output", color: "#6a85a0", widgets: ["frame_rate", "filename_prefix", "crf", "pix_fmt", "trim_to_audio"] },
    ],
};

const SECTION_BUTTON_COLLAPSED_BG = "#1f2935";
const SECTION_BUTTON_COLLAPSED_BORDER = "#6d7785";
const SECTION_BUTTON_RADIUS = 6;
const SECTION_BUTTON_TEXT = "#f8fbff";
const SECTION_BUTTON_HEIGHT = 30;
const RENDER_INTERNAL_WIDGETS = new Set(["backend_mode", "generation_mode", "media_mode"]);
const GENERATED_DURATION_TYPES = new Set(["img2video", "text2video"]);
const GENERATION_TYPE_CONFIGS = {
    "aud+img2video_simple": { generation_mode: "img2vid", backend_mode: "single_best", media_mode: "input_audio_img2vid" },
    "aud+img2video_2_segments": { generation_mode: "img2vid", backend_mode: "two_segments_normal_vram", media_mode: "input_audio_img2vid" },
    "aud+img2video_infinite": { generation_mode: "img2vid", backend_mode: "loop_normal_vram", media_mode: "input_audio_img2vid" },
    "text+audio2video": { generation_mode: "t2v", backend_mode: "loop_normal_vram", media_mode: "input_audio_t2v" },
    img2video: { generation_mode: "img2vid", backend_mode: "single_best", media_mode: "img2vid_pure" },
    text2video: { generation_mode: "t2v", backend_mode: "single_best", media_mode: "t2v_pure" },
};
const RENDER_UI_PRESET_VALUES = new Set(["custom", ...Object.keys(PRESET_CONFIGS["IAMCCS-SuperNodes AU+IMG2VID Exec Render"]?.values || {})]);
const RENDER_UI_PRESET_VISIBLE_VALUES = ["custom", "low_ram_safe", "balanced", "high_quality", "fast_preview", "loop_lipsync_safe", "motion_controlled"];
const RENDER_GENERATION_TYPE_VALUES = new Set(Object.keys(GENERATION_TYPE_CONFIGS));
const RENDER_LEGACY_WIDGET_ORDER = [
    "generation_mode", "backend_mode", "positive_text", "negative_text", "width", "height", "steps", "cfg", "sampler_name", "seed", "control_after_generate",
    "max_shift", "base_shift", "sigma_terminal", "manual_sigmas", "image_strength", "image_compression",
    "audio_context_mode", "audio_left_context_s", "audio_right_context_s",
    "stitch_preset", "overlap_side", "overlap_mode", "start_frames_rule",
    "continuity_anchor_mode", "anchor_refresh_interval", "anchor_image_strength", "anti_drift_mode", "anti_drift_strength", "identity_persistence_strength",
    "vae_mode", "downstream_stage_mode", "output_root", "segment_overlay_mode", "segment_overlay_text",
    "second_stage_mode", "stage2_model_policy", "second_stage_upscale_model", "second_stage_reinject_strength", "second_stage_cfg", "second_stage_manual_sigmas",
    "media_mode", "vram_flush", "motion_intensity", "ui_preset", "generated_media_duration_seconds", "generation_type",
];

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

function setWidgetLabel(node, widgetName, label) {
    const widget = findWidget(node, widgetName);
    if (!widget) {
        return;
    }
    widget.label = label;
}

function sectionButtonCaption(group, isExpanded) {
    return `${isExpanded ? "[-]" : "[+]"} ${group.label}`;
}

function sectionButtonStyle(group, isExpanded) {
    const expandedBg = group.color || "#4f6f8f";
    return {
        background: isExpanded ? expandedBg : SECTION_BUTTON_COLLAPSED_BG,
        border: isExpanded ? expandedBg : SECTION_BUTTON_COLLAPSED_BORDER,
        rail: expandedBg,
        text: SECTION_BUTTON_TEXT,
    };
}

function roundedRectPath(ctx, x, y, width, height, radius) {
    const r = Math.max(0, Math.min(radius, height * 0.5, width * 0.5));
    if (typeof ctx.roundRect === "function") {
        ctx.roundRect(x, y, width, height, r);
        return;
    }
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + width - r, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + r);
    ctx.lineTo(x + width, y + height - r);
    ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
    ctx.lineTo(x + r, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
}

function installSectionButtonDraw(button) {
    if (!button || button._iamccsCustomSectionDraw) {
        return;
    }
    button._iamccsCustomSectionDraw = true;
    button.draw = function(ctx, node, widgetWidth, y, widgetHeight) {
        const style = this._iamccsButtonStyle || sectionButtonStyle({ color: "#4f6f8f" }, true);
        const x = 8;
        const height = Math.max(SECTION_BUTTON_HEIGHT - 4, (widgetHeight || SECTION_BUTTON_HEIGHT) - 4);
        const width = Math.max(40, (widgetWidth || node?.size?.[0] || 280) - 16);
        const top = y + 2;
        const label = String(this.label || this.name || "");

        ctx.save();
        ctx.beginPath();
        roundedRectPath(ctx, x, top, width, height, SECTION_BUTTON_RADIUS);
        ctx.fillStyle = style.background;
        ctx.fill();
        ctx.lineWidth = 1;
        ctx.strokeStyle = style.border;
        ctx.stroke();

        ctx.beginPath();
        roundedRectPath(ctx, x, top, 6, height, SECTION_BUTTON_RADIUS);
        ctx.fillStyle = style.rail;
        ctx.fill();

        ctx.fillStyle = style.text;
        ctx.font = "bold 12px sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(label, x + 14, top + height * 0.5);
        ctx.restore();
        return true;
    };
}

function insertWidget(node, widget, index) {
    const widgets = node.widgets || [];
    const currentIndex = widgets.indexOf(widget);
    if (currentIndex >= 0) {
        widgets.splice(currentIndex, 1);
    }
    widgets.splice(Math.max(0, Math.min(index, widgets.length)), 0, widget);
}

function moveWidgetsAfter(node, anchorName, widgetNames) {
    if (!node.widgets?.length) {
        return;
    }
    const moving = [];
    for (const widgetName of widgetNames) {
        const index = node.widgets.findIndex((widget) => widget?.name === widgetName);
        if (index >= 0) {
            moving.push(node.widgets.splice(index, 1)[0]);
        }
    }
    if (!moving.length) {
        return;
    }
    const anchorIndex = node.widgets.findIndex((widget) => widget?.name === anchorName);
    const insertAt = anchorIndex >= 0 ? anchorIndex + 1 : node.widgets.length;
    node.widgets.splice(insertAt, 0, ...moving);
}

function normalizeRenderWidgetOrder(node) {
    if (node.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        return;
    }
    moveWidgetsAfter(node, "second_stage_mode", ["stage2_model_policy", "second_stage_upscale_model", "second_stage_reinject_strength", "second_stage_cfg", "second_stage_manual_sigmas"]);
}

function setWidgetValue(node, widgetName, value) {
    const widget = findWidget(node, widgetName);
    if (!widget || widget.value === value) {
        return;
    }
    widget.value = value;
    widget.callback?.(value);
}

const EXEC_PLANNER_PRESETS = {
    "5sec": { seconds: 5.0, overlap: 9 },
    "10sec": { seconds: 10.0, overlap: 9 },
    "15sec": { seconds: 15.0, overlap: 9 },
    "20sec": { seconds: 20.0, overlap: 9 },
    videoclip: { seconds: 10.0, overlap: 9 },
    monologue: { seconds: 15.0, overlap: 9 },
};

function getExecPlannerPreset(value) {
    return EXEC_PLANNER_PRESETS[String(value || "15sec")] || EXEC_PLANNER_PRESETS["15sec"];
}

function getExecPlannerPresetForSeconds(seconds) {
    const value = Number(seconds);
    if (!Number.isFinite(value)) {
        return null;
    }
    for (const presetName of ["5sec", "10sec", "15sec", "20sec"]) {
        const rec = EXEC_PLANNER_PRESETS[presetName];
        if (Math.abs(value - rec.seconds) <= 0.001) {
            return rec;
        }
    }
    return null;
}

function ltxSafeFramesAtLeast(frameCount) {
    const frames = Math.max(1, Math.round(Number(frameCount) || 1));
    const remainder = (frames - 1) % 8;
    return remainder === 0 ? frames : frames + (8 - remainder);
}

function readPlannerNumber(node, widgetName, fallback) {
    const value = Number(findWidget(node, widgetName)?.value);
    return Number.isFinite(value) ? value : fallback;
}

function getExecPlannerLiveConfig(node) {
    const fps = Math.max(0.001, readPlannerNumber(node, "fps", 24.0));
    const planningMode = String(findWidget(node, "planning_mode")?.value || "manual_segment_seconds");
    const segmentPreset = String(findWidget(node, "segment_preset")?.value || "15sec");
    const presetRec = getExecPlannerPreset(segmentPreset);
    let seconds = Math.max(0.01, readPlannerNumber(node, "segment_seconds", presetRec.seconds));
    const overlap = Math.max(0, Math.round(readPlannerNumber(node, "overlap_frames", 9)));
    let overlapSource = "manual/default";

    if (planningMode === "explicit_preset_seconds") {
        seconds = presetRec.seconds;
        overlapSource = `${segmentPreset} preset/manual`;
    } else {
        const secondsRec = getExecPlannerPresetForSeconds(seconds);
        if (secondsRec) {
            overlapSource = `${secondsRec.seconds}s duration`;
        }
    }

    return { fps, planningMode, segmentPreset, seconds, overlap, overlapSource };
}

function updateExecPlannerLivePreview(node) {
    if (node.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
        return;
    }
    node.properties = node.properties || {};
    const config = getExecPlannerLiveConfig(node);
    const uniqueFrames = Math.max(1, Math.round(config.seconds * config.fps));
    const firstRaw = ltxSafeFramesAtLeast(uniqueFrames);
    const continuationTarget = uniqueFrames + config.overlap;
    const continuationRaw = ltxSafeFramesAtLeast(continuationTarget);
    const lines = [
        `live ${config.seconds.toFixed(2)}s @ ${config.fps.toFixed(2)}fps -> ${uniqueFrames}f unique`,
        `overlap ${config.overlap}f (${config.overlapSource}) | first raw ${firstRaw}f | cont raw ${continuationRaw}f`,
    ];

    const storedDuration = Number(node.properties.iamccsPlannerDuration);
    const storedTotalFrames = Number(node.properties.iamccsPlannerTotalFrames);
    const hasDuration = Number.isFinite(storedDuration) && storedDuration > 0;
    const totalFrames = Number.isFinite(storedTotalFrames) && storedTotalFrames > 0
        ? Math.round(storedTotalFrames)
        : hasDuration
            ? Math.round(storedDuration * config.fps)
            : 0;

    if (totalFrames > 0) {
        const segmentCount = Math.max(1, Math.ceil(totalFrames / uniqueFrames));
        lines.push(`audio ${hasDuration ? storedDuration.toFixed(2) : "cached"}s -> ${totalFrames}f | segments ${segmentCount}`);
        const parts = [];
        for (let index = 0; index < Math.min(segmentCount, 4); index += 1) {
            const remaining = Math.max(0, totalFrames - index * uniqueFrames);
            const effective = Math.max(1, Math.min(uniqueFrames, remaining));
            const target = index === 0 ? effective : effective + config.overlap;
            parts.push(`s${index + 1}:${effective}${index === 0 ? "" : `+${config.overlap}`}->${ltxSafeFramesAtLeast(target)}`);
        }
        lines.push(`${parts.join(" ")}${segmentCount > 4 ? " ..." : ""}`);
    } else {
        lines.push("audio duration unknown until planner runs once");
    }

    node.properties.iamccsPlannerLivePreview = lines.join("\n");
}

function syncExecPlannerExplicitPreset(node, sourceWidgetName = "") {
    if (node.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
        return;
    }
    const planningMode = String(findWidget(node, "planning_mode")?.value || "manual_segment_seconds");
    if (planningMode === "explicit_preset_seconds") {
        const rec = getExecPlannerPreset(findWidget(node, "segment_preset")?.value);
        setWidgetValue(node, "segment_seconds", rec.seconds);
        if (sourceWidgetName === "planning_mode" || sourceWidgetName === "segment_preset") {
            setWidgetValue(node, "overlap_frames", rec.overlap);
        }
    }
    updateExecPlannerLivePreview(node);
}

function installExecPlannerExplicitPresetSync(node) {
    if (node._iamccsExecPlannerExplicitPresetSyncInstalled) {
        syncExecPlannerExplicitPreset(node);
        updateExecPlannerLivePreview(node);
        return;
    }
    node._iamccsExecPlannerExplicitPresetSyncInstalled = true;
    for (const widgetName of ["fps", "segment_seconds", "planning_mode", "segment_preset", "overlap_frames", "ltx_round_mode"]) {
        const widget = findWidget(node, widgetName);
        if (!widget) {
            continue;
        }
        const originalCallback = widget.callback;
        widget.callback = (...args) => {
            originalCallback?.apply(widget, args);
            if (widgetName === "planning_mode") {
                applyPlannerModeVisibility(node);
            } else {
                syncExecPlannerExplicitPreset(node, widgetName);
            }
            updateExecPlannerLivePreview(node);
            app.graph.setDirtyCanvas(true, true);
        };
    }
    syncExecPlannerExplicitPreset(node);
    updateExecPlannerLivePreview(node);
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
    const decodeMode = String(findWidget(node, "decode_mode")?.value || "low_ram_disk");
    const isLowRam = decodeMode === "low_ram_disk" || decodeMode === "very_low_ram_disk";
    const isNormal = decodeMode === "normal_tiled_vhs" || decodeMode === "custom_mode" || decodeMode === "inherit_render_backend";
    setWidgetVisibility(findWidget(node, "frames_subdir"), isLowRam);
    setWidgetVisibility(findWidget(node, "image_format"), isLowRam);
    setWidgetVisibility(findWidget(node, "jpg_quality"), isLowRam);
    setWidgetVisibility(findWidget(node, "tiled_tile_size"), isNormal);
    setWidgetVisibility(findWidget(node, "tiled_overlap"), isNormal);
    fitNodeToWidgets(node);
}

function applyRenderAnchorVisibility(node) {
    if (node.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        return;
    }
    const anchorMode = String(findWidget(node, "continuity_anchor_mode")?.value || "off");
    const antiDriftMode = String(findWidget(node, "anti_drift_mode")?.value || "off");
    const anchorsEnabled = anchorMode !== "off";
    const periodicAnchor = anchorMode.startsWith("periodic_");
    setWidgetVisibility(findWidget(node, "anchor_refresh_interval"), anchorsEnabled && periodicAnchor);
    setWidgetVisibility(findWidget(node, "anchor_image_strength"), anchorsEnabled);
    setWidgetVisibility(findWidget(node, "anti_drift_strength"), antiDriftMode !== "off");
    setWidgetVisibility(findWidget(node, "identity_persistence_strength"), antiDriftMode === "dual_reference_adain");
    fitNodeToWidgets(node);
}

function applyRenderAnchorLabels(node) {
    if (node.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        return;
    }
    setWidgetLabel(node, "backend_mode", "Render Backend");
    setWidgetLabel(node, "generation_mode", "Generation Mode");
    setWidgetLabel(node, "media_mode", "Media Mode");
    setWidgetLabel(node, "generation_type", "Generation Type");
    setWidgetLabel(node, "generated_media_duration_seconds", "Generated Duration");
    setWidgetLabel(node, "continuity_anchor_mode", "Anchor Refresh");
    setWidgetLabel(node, "anchor_refresh_interval", "Refresh Interval");
    setWidgetLabel(node, "anchor_image_strength", "Anchor Guidance Strength");
    setWidgetLabel(node, "motion_intensity", "Motion Intensity");
    setWidgetLabel(node, "vram_flush", "VRAM Flush");
    setWidgetLabel(node, "second_stage_mode", "Second Stage");
    setWidgetLabel(node, "stage2_model_policy", "Stage2 Model Policy");
    setWidgetLabel(node, "second_stage_upscale_model", "2x Upscale Model");
    setWidgetLabel(node, "second_stage_reinject_strength", "Anchor Reinject Strength");
}

function syncRenderGenerationType(node) {
    if (node.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        return;
    }
    const generationType = String(findWidget(node, "generation_type")?.value || "aud+img2video_infinite");
    const config = GENERATION_TYPE_CONFIGS[generationType] || GENERATION_TYPE_CONFIGS["aud+img2video_infinite"];
    Object.entries(config).forEach(([widgetName, widgetValue]) => setWidgetValue(node, widgetName, widgetValue));
}

function renderUsesGeneratedDuration(node) {
    const generationType = String(findWidget(node, "generation_type")?.value || "");
    return GENERATED_DURATION_TYPES.has(generationType);
}

function applyRenderGeneratedDurationVisibility(node) {
    if (node.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        return;
    }
    const generationExpanded = node.properties?.iamccs_section_generation !== false;
    setWidgetVisibility(findWidget(node, "generated_media_duration_seconds"), generationExpanded && renderUsesGeneratedDuration(node));
    fitNodeToWidgets(node);
}

function inferRenderGenerationType(valuesByName) {
    const generationMode = String(valuesByName.generation_mode || "img2vid");
    const backendMode = String(valuesByName.backend_mode || "auto");
    const mediaMode = String(valuesByName.media_mode || "");
    if (mediaMode === "t2v_pure") {
        return "text2video";
    }
    if (mediaMode === "img2vid_pure" || mediaMode === "generated_audio_img2vid") {
        return "img2video";
    }
    if (generationMode === "t2v" || mediaMode === "input_audio_t2v" || mediaMode === "generated_audio_t2v") {
        return "text+audio2video";
    }
    if (backendMode === "two_segments_normal_vram") {
        return "aud+img2video_2_segments";
    }
    if (backendMode === "loop_normal_vram" || backendMode === "loop_low_ram_disk" || backendMode === "auto") {
        return "aud+img2video_infinite";
    }
    return "aud+img2video_simple";
}

function applyRenderPresetDropdownOptions(node) {
    if (node.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        return;
    }
    const widget = findWidget(node, "ui_preset");
    if (widget?.options) {
        widget.options.values = RENDER_UI_PRESET_VISIBLE_VALUES;
    }
}

function applyRenderInternalWidgetVisibility(node) {
    if (node.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        return;
    }
    for (const widgetName of RENDER_INTERNAL_WIDGETS) {
        setWidgetVisibility(findWidget(node, widgetName), false);
    }
}

function applyRenderSecondStageVisibility(node) {
    if (node.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        return;
    }

    const mode = String(findWidget(node, "second_stage_mode")?.value || "off");
    const stageExpanded = node.properties?.iamccs_section_stage2 !== false;
    const enabled = stageExpanded && mode !== "off";
    setWidgetVisibility(findWidget(node, "stage2_model_policy"), enabled);
    setWidgetVisibility(findWidget(node, "second_stage_reinject_strength"), enabled);
    setWidgetVisibility(findWidget(node, "second_stage_cfg"), enabled);
    setWidgetVisibility(findWidget(node, "second_stage_manual_sigmas"), enabled);
    setWidgetVisibility(findWidget(node, "second_stage_upscale_model"), enabled && mode === "latent_upscale_refine_x2_beta");
    fitNodeToWidgets(node);
}

function syncDownstreamVaeDecodeModes(renderNode) {
    const renderDecode = String(findWidget(renderNode, "vae_mode")?.value || "low_ram_disk");
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
    Object.entries(visibilityMap).forEach(([widgetName, isVisible]) => {
        setWidgetVisibility(findWidget(node, widgetName), !!isVisible);
    });
    if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec VAE") {
        applyVaeDecodeModeVisibility(node);
    }
    if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        applyRenderAnchorLabels(node);
        applyRenderAnchorVisibility(node);
        applyRenderSecondStageVisibility(node);
        syncRenderGenerationType(node);
        applyRenderGeneratedDurationVisibility(node);
        applyRenderPresetDropdownOptions(node);
    }
    fitNodeToWidgets(node);
}

function getSerializableWidgets(node) {
    return (node.widgets || []).filter((widget) => widget && widget.serialize !== false && !widget._iamccsSectionKey);
}

function isSectionButtonCaption(value) {
    const text = String(value ?? "");
    return /^\[[+-]\]\s/.test(text) || text.startsWith("▶ ") || text.startsWith("▼ ");
}

function sanitizeSerializedValues(values) {
    return (values || []).filter((value) => value !== null && value !== undefined && !isSectionButtonCaption(value));
}

function sanitizeRenderCombo(node, widgetName, fallback, validValues) {
    const widget = findWidget(node, widgetName);
    if (!widget) {
        return;
    }
    const text = String(widget.value ?? "");
    if (isSectionButtonCaption(text) || (validValues?.size && !validValues.has(text))) {
        widget.value = fallback;
    }
}

function sanitizeRenderWidgetValues(node) {
    if (node.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        return;
    }
    applyRenderPresetDropdownOptions(node);
    sanitizeRenderCombo(node, "ui_preset", "custom", RENDER_UI_PRESET_VALUES);
    sanitizeRenderCombo(node, "generation_type", "aud+img2video_infinite", RENDER_GENERATION_TYPE_VALUES);

    const durationWidget = findWidget(node, "generated_media_duration_seconds");
    if (durationWidget && (isSectionButtonCaption(durationWidget.value) || !Number.isFinite(Number(durationWidget.value)))) {
        durationWidget.value = 10.0;
    }

    syncRenderGenerationType(node);
    applyRenderGeneratedDurationVisibility(node);
}

function rehydrateSerializedWidgets(node, serializedValues) {
    if (!Array.isArray(serializedValues) || !node.widgets?.length) {
        return;
    }

    const widgets = getSerializableWidgets(node);
    const cleanValues = sanitizeSerializedValues(serializedValues);
    if (node.comfyClass === "IAMCCS-SuperNodes AU+IMG2VID Exec Render" && ["img2vid", "t2v"].includes(String(cleanValues[0] || ""))) {
        const valuesByName = {};
        RENDER_LEGACY_WIDGET_ORDER.forEach((widgetName, index) => {
            if (index < cleanValues.length) {
                valuesByName[widgetName] = cleanValues[index];
            }
        });
        valuesByName.generation_type = RENDER_GENERATION_TYPE_VALUES.has(String(valuesByName.generation_type || ""))
            ? valuesByName.generation_type
            : inferRenderGenerationType(valuesByName);
        valuesByName.ui_preset = RENDER_UI_PRESET_VALUES.has(String(valuesByName.ui_preset || "")) ? valuesByName.ui_preset : "custom";
        valuesByName.generated_media_duration_seconds = Number.isFinite(Number(valuesByName.generated_media_duration_seconds))
            ? Number(valuesByName.generated_media_duration_seconds)
            : 10.0;
        const generationConfig = GENERATION_TYPE_CONFIGS[String(valuesByName.generation_type)] || GENERATION_TYPE_CONFIGS["aud+img2video_infinite"];
        Object.assign(valuesByName, generationConfig);
        for (const widget of widgets) {
            if (Object.prototype.hasOwnProperty.call(valuesByName, widget.name)) {
                widget.value = valuesByName[widget.name];
            }
        }
        return;
    }

    const count = Math.min(widgets.length, cleanValues.length);
    for (let index = 0; index < count; index += 1) {
        const widget = widgets[index];
        if (!widget) {
            continue;
        }
        widget.value = cleanValues[index];
    }
}

function addSectionButton(node, group, index) {
    // Duplicate guard: if a button for this section key is already in the widget list,
    // return it immediately.  This prevents double-insertion if onNodeCreated or
    // enhanceNodeLayout is called more than once on the same node instance.
    const existing = node.widgets?.find((w) => w._iamccsSectionKey === group.key);
    if (existing) {
        installSectionButtonDraw(existing);
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
    // Visual-only section buttons must never enter widgets_values/prompt payloads.
    sectionButton.serialize = false;
    sectionButton._iamccsSectionKey = group.key;
    sectionButton.computeSize = (width) => [width || 280, SECTION_BUTTON_HEIGHT];
    sectionButton.label = group.label;
    installSectionButtonDraw(sectionButton);
    sectionButton.options = {
        bgcolor: group.color || "#4f6f8f",
        background_color: group.color || "#4f6f8f",
        color: SECTION_BUTTON_TEXT,
    };
    sectionButton.value = "";

    applyGroupVisibility(node, group, propKey, sectionButton);
    insertWidget(node, sectionButton, index);
    return sectionButton;
}

function applyGroupVisibility(node, group, propKey, button) {
    const isExpanded = !!node.properties[propKey];
    const caption = sectionButtonCaption(group, isExpanded);
    button.name = caption;
    button.label = caption;
    button.value = "";
    const style = sectionButtonStyle(group, isExpanded);
    button._iamccsButtonStyle = style;
    button.options = {
        bgcolor: style.background,
        background_color: style.background,
        border_color: style.border,
        color: SECTION_BUTTON_TEXT,
    };
    for (const widgetName of group.widgets) {
        setWidgetVisibility(findWidget(node, widgetName), isExpanded);
    }
    if (node.comfyClass === "IAMCCS-SuperNodes AU+IMG2VID Exec Render" && group.key === "generation") {
        applyRenderGeneratedDurationVisibility(node);
    }
    if (node.comfyClass === "IAMCCS-SuperNodes AU+IMG2VID Exec Render" && group.key === "stage2") {
        applyRenderSecondStageVisibility(node);
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
    syncExecPlannerExplicitPreset(node);
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
        Object.entries(visibilityMap).forEach(([widgetName, isVisible]) => {
            setWidgetVisibility(findWidget(node, widgetName), !!isVisible);
        });
    }

    if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec VAE") {
        applyVaeDecodeModeVisibility(node);
    } else if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
        applyPlannerModeVisibility(node);
    } else if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        sanitizeRenderWidgetValues(node);
        applyRenderSecondStageVisibility(node);
        applyRenderInternalWidgetVisibility(node);
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

    if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        normalizeRenderWidgetOrder(node);
    }

    if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
        addPlannerChip(node);
        addMultiLineStatusBox(node, "iamccsPlannerLivePreview", "planner_live_preview", "#1d2f38", "#67a7bb", "#e5f3f7");
        addMultiLineStatusBox(node, "iamccsPlannerDetails", "planner_details_preview", "#24362d", "#72ab8e", "#e4f2e8");
    }
    if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        addStatusBox(node, "iamccsRenderStatus", "render_status_preview", "#1e2f3f", "#6f93ba", "#e4edf7");
    }
    if (nodeName === "IAMCCS-SuperNodes Second Stage") {
        addStatusBox(node, "iamccsSecondStageStatus", "second_stage_status_preview", "#30243d", "#9f77bf", "#f1e7f8");
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
            this.properties = this.properties || {};
            // MUST run synchronously, BEFORE configure() assigns widgets_values by index.
            // Section buttons inserted here occupy their index slots so that the null
            // placeholders in widgets_values land on buttons, not on real widgets.
            // (Using setTimeout here was the root cause of NaN on load / after undo.)
            enhanceNodeLayout(this, nodeName);
            if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
                sanitizeRenderWidgetValues(this);
                applyRenderInternalWidgetVisibility(this);
            }
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
                const vaeModeWidget = findWidget(this, "vae_mode");
                if (vaeModeWidget) {
                    const originalCallback = vaeModeWidget.callback;
                    vaeModeWidget.callback = (...args) => {
                        originalCallback?.apply(vaeModeWidget, args);
                        syncDownstreamVaeDecodeModes(this);
                        app.graph.setDirtyCanvas(true, true);
                    };
                    syncDownstreamVaeDecodeModes(this);
                }
                for (const widgetName of ["generation_type", "generated_media_duration_seconds", "second_stage_mode", "continuity_anchor_mode", "anti_drift_mode"]) {
                    const widget = findWidget(this, widgetName);
                    if (!widget) {
                        continue;
                    }
                    const originalCallback = widget.callback;
                    widget.callback = (...args) => {
                        originalCallback?.apply(widget, args);
                        sanitizeRenderWidgetValues(this);
                        applyRenderAnchorVisibility(this);
                        applyRenderSecondStageVisibility(this);
                        applyRenderInternalWidgetVisibility(this);
                        app.graph.setDirtyCanvas(true, true);
                    };
                }
                sanitizeRenderWidgetValues(this);
                applyRenderSecondStageVisibility(this);
                applyRenderInternalWidgetVisibility(this);
            }
            if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
                installExecPlannerExplicitPresetSync(this);
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
            this.properties = this.properties || {};
            enhanceNodeLayout(this, nodeName);
            if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
                sanitizeRenderWidgetValues(this);
                applyRenderInternalWidgetVisibility(this);
            }
            rehydrateSerializedWidgets(this, info?.widgets_values);
            if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
                sanitizeRenderWidgetValues(this);
                applyRenderInternalWidgetVisibility(this);
            }
            if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
                installExecPlannerExplicitPresetSync(this);
            }
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
                    this.properties.iamccsPlannerDuration = Number(duration || 0);
                }
                if (totalFrames !== null) {
                    this.properties.iamccsPlannerTotalFrames = Number(totalFrames || 0);
                }
                if (segments !== null) {
                    this.properties.iamccsPlannerSegmentCount = Number(segments || 0);
                }
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
                const plannerReport = Array.isArray(message?.planning_report) && message.planning_report.length > 0
                    ? message.planning_report[0]
                    : Array.isArray(message?.report) && message.report.length > 0
                        ? message.report[0]
                        : null;
                if (plannerReport !== null) {
                    details.push(String(plannerReport || ""));
                }
                updateExecPlannerLivePreview(this);
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

        if (nodeName === "IAMCCS-SuperNodes Second Stage") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                if (Array.isArray(message?.report) && message.report.length > 0) {
                    this.properties = this.properties || {};
                    this.properties.iamccsSecondStageStatus = String(message.report[0] || "").split("\n")[0];
                    app.graph.setDirtyCanvas(true, true);
                }
            };
        }
    },
});










