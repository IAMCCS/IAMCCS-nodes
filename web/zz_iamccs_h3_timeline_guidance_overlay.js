import { app } from "../../scripts/app.js";

// IAMCCS H3 Timeline Guidance Overlay
// UI-only authoring aid. It never changes Shotboard timeline data, prompts,
// guide positions, conditioning, planner output, audio, seeds or rendering.

const TARGET_NODE = "IAMCCS_MiniMaxH3ShotPlanner";
const PROP_ENABLED = "iamccs_h3_timeline_guidance_enabled";
const OVERLAY_ATTR = "data-iamccs-h3-guidance-overlay";
const FPS = 24;

// Editorial recommendations, not MiniMax hard limits.
const PRESERVE_FRAMES = 72;     // 3.00 s: prefer not to introduce another major checkpoint.
const PREFERRED_FRAMES = 96;    // 4.00 s: practical working target for substantial visual changes.
const PREFERRED_END_FRAMES = 120; // 5.00 s: upper edge of the preferred target band.
const H3_NATIVE_WINDOW = 362;   // 15.08 s native trained/sample window used by the R42 planner.

const COLORS = {
    cyanFill: "rgba(80, 194, 216, .11)",
    cyanEdge: "rgba(92, 207, 228, .72)",
    greenFill: "rgba(112, 204, 112, .14)",
    green: "#82D67A",
    amber: "#F2C75C",
    red: "#E46B6B",
    text: "#F4F1E7",
    dark: "rgba(8, 12, 14, .86)",
};

function widget(node, name) {
    return node?.widgets?.find((item) => item?.name === name || item?.label === name) || null;
}

function parseTimeline(node) {
    const raw = String(widget(node, "timeline_data")?.value || "").trim();
    let data = {};
    if (raw) {
        try { data = JSON.parse(raw); } catch { data = {}; }
    }
    return data && typeof data === "object" && !Array.isArray(data) ? data : {};
}

function rowsFromTimeline(data) {
    for (const key of ["rows", "segments", "slots", "shots"]) {
        if (Array.isArray(data?.[key])) return data[key].filter((row) => row && typeof row === "object");
    }
    if (data?.timeline && typeof data.timeline === "object") return rowsFromTimeline(data.timeline);
    return [];
}

function timelineFps(data) {
    const value = Number(data?.fps ?? data?.frame_rate ?? data?.frameRate ?? FPS);
    return Number.isFinite(value) && value > 0 ? value : FPS;
}

function isFrameTimeline(data) {
    const schema = String(data?.schema || "").toLowerCase();
    return schema.includes("filmmaker_timeline") || schema.includes("shotboard_timeline") || data?.fps != null || data?.frame_rate != null;
}

function rowStartFrame(row, data, fps) {
    for (const key of ["frame", "start_frame", "startFrame", "global_frame", "globalFrame"]) {
        const value = Number(row?.[key]);
        if (Number.isFinite(value)) return Math.max(0, Math.round(value));
    }
    const seconds = Number(row?.second ?? row?.start_seconds ?? row?.startSeconds ?? row?.time_seconds);
    if (Number.isFinite(seconds)) return Math.max(0, Math.round(seconds * fps));
    const start = Number(row?.start);
    if (Number.isFinite(start)) return Math.max(0, Math.round(isFrameTimeline(data) ? start : start * fps));
    return 0;
}

function rowEndFrame(row, data, fps) {
    for (const key of ["end_frame", "endFrame"]) {
        const value = Number(row?.[key]);
        if (Number.isFinite(value)) return Math.max(0, Math.round(value));
    }
    const endSecond = Number(row?.end_second ?? row?.end_seconds ?? row?.endSeconds);
    if (Number.isFinite(endSecond)) return Math.max(0, Math.round(endSecond * fps));
    const start = rowStartFrame(row, data, fps);
    const lengthFrames = Number(row?.length_frames ?? row?.duration_frames ?? row?.frame_count);
    if (Number.isFinite(lengthFrames)) return start + Math.max(0, Math.round(lengthFrames));
    const durationSeconds = Number(row?.duration_seconds ?? row?.length_seconds ?? row?.duration);
    if (Number.isFinite(durationSeconds)) return start + Math.max(0, Math.round(durationSeconds * fps));
    const length = Number(row?.length);
    if (Number.isFinite(length)) return start + Math.max(0, Math.round(isFrameTimeline(data) ? length : length * fps));
    return start;
}

function isImageGuideRow(row) {
    const type = String(row?.type || "image").trim().toLowerCase();
    if (["audio", "motion", "video", "text"].includes(type)) return false;
    if (row?.placeholder === true) return false;
    if (row?.use_guide === false || row?.use_keyframe === false) return false;
    const hasImage = Boolean(
        row?.imageFile || row?.image_file || row?.imageTruthPath || row?.image_truth_path ||
        row?.path || row?.image_path || row?.image || Number(row?.ref || 0) > 0
    );
    return type === "image" ? hasImage || row?.use_guide === true || row?.use_keyframe === true : hasImage;
}

function totalFrames(node, data, fps, rows) {
    const seconds = Number(data?.duration_seconds ?? data?.durationSeconds ?? widget(node, "duration_seconds")?.value);
    if (Number.isFinite(seconds) && seconds > 0) return Math.max(1, Math.round(seconds * fps));
    let end = 0;
    for (const row of rows) end = Math.max(end, rowEndFrame(row, data, fps));
    return Math.max(1, end);
}

function effectiveTaskMode(node, data) {
    const timelineMode = String(data?.task_mode || data?.taskMode || "").trim().toLowerCase();
    const widgetMode = String(widget(node, "task_mode")?.value || "").trim().toLowerCase();
    if (timelineMode && timelineMode !== "auto" && timelineMode !== "auto_from_timeline") return timelineMode;
    return widgetMode || timelineMode;
}

function guidanceModel(node) {
    const data = parseTimeline(node);
    const fps = timelineFps(data);
    const rows = rowsFromTimeline(data);
    const total = totalFrames(node, data, fps, rows);
    const guides = rows
        .filter(isImageGuideRow)
        .map((row) => rowStartFrame(row, data, fps))
        .filter((frame) => frame >= 0 && frame <= total)
        .sort((a, b) => a - b)
        .filter((frame, index, all) => index === 0 || frame !== all[index - 1]);
    return { data, fps, rows, total, guides, mode: effectiveTaskMode(node, data) };
}

function candidateScore(el) {
    if (!(el instanceof HTMLElement)) return -Infinity;
    const style = el.style || {};
    let score = 0;
    const height = Number.parseFloat(style.height || "0");
    if (style.position === "relative") score += 3;
    if (String(style.cursor || "").includes("ew-resize")) score += 6;
    if (height >= 32 && height <= 42) score += 6;
    if (/\b0s\b/.test(String(el.textContent || ""))) score += 2;
    if (String(style.overflow || "").includes("hidden")) score += 1;
    return score;
}

function widgetRoots(node) {
    const roots = [];
    for (const item of node?.widgets || []) {
        for (const candidate of [item?.element, item?.inputEl, item?.container]) {
            if (candidate instanceof HTMLElement && !roots.includes(candidate)) roots.push(candidate);
        }
    }
    return roots;
}

function findRuler(node) {
    let best = null;
    let bestScore = -Infinity;
    for (const root of widgetRoots(node)) {
        const candidates = [root, ...root.querySelectorAll("div")];
        for (const el of candidates) {
            const score = candidateScore(el);
            if (score > bestScore) { best = el; bestScore = score; }
        }
    }
    return bestScore >= 10 ? best : null;
}

function el(tag, cssText = "", text = "") {
    const node = document.createElement(tag);
    if (cssText) node.style.cssText = cssText;
    if (text) node.textContent = text;
    return node;
}

function percent(frame, total) {
    return Math.max(0, Math.min(100, (Number(frame) / Math.max(1, Number(total))) * 100));
}

function marker(layer, frame, total, color, title, label = "", z = 5) {
    if (frame <= 0 || frame >= total) return;
    const x = percent(frame, total);
    const line = el("div", `position:absolute;left:calc(${x}% - 1px);top:0;bottom:0;width:2px;background:${color};opacity:.95;pointer-events:none;z-index:${z};box-shadow:0 0 5px ${color};`);
    line.title = title;
    layer.appendChild(line);
    if (label) {
        const tag = el("div", `position:absolute;left:calc(${x}% + 3px);bottom:2px;padding:1px 3px;border-radius:3px;background:${COLORS.dark};color:${color};font:800 8px/1.15 system-ui;white-space:nowrap;pointer-events:none;z-index:${z + 1};`, label);
        tag.title = title;
        layer.appendChild(tag);
    }
}

function band(layer, startFrame, endFrame, total, background, borderColor, title, z = 1) {
    const start = Math.max(0, Math.min(total, startFrame));
    const end = Math.max(start, Math.min(total, endFrame));
    if (end <= start) return;
    const left = percent(start, total);
    const width = Math.max(0, percent(end, total) - left);
    const zone = el("div", `position:absolute;left:${left}%;width:${width}%;top:0;bottom:0;background:${background};border-left:1px solid ${borderColor};border-right:1px solid ${borderColor};pointer-events:none;z-index:${z};box-sizing:border-box;`);
    zone.title = title;
    layer.appendChild(zone);
}

function pairGap(layer, start, end, total, fps) {
    if (!(end > start)) return;
    const gap = end - start;
    const color = gap < PRESERVE_FRAMES ? COLORS.red : gap < PREFERRED_FRAMES ? COLORS.amber : COLORS.green;
    const left = percent(start, total);
    const right = percent(end, total);
    const width = Math.max(0, right - left);
    const line = el("div", `position:absolute;left:${left}%;width:${width}%;bottom:0;height:3px;background:${color};opacity:.92;pointer-events:none;z-index:8;`);
    const seconds = gap / fps;
    const status = gap < PRESERVE_FRAMES ? "tight" : gap < PREFERRED_FRAMES ? "caution" : "preferred";
    line.title = `${gap}f / ${seconds.toFixed(2)}s between image guides · ${status} spacing (IAMCCS authoring guidance, not a MiniMax hard limit)`;
    layer.appendChild(line);
    if (width >= 8) {
        const label = el("div", `position:absolute;left:${left + width / 2}%;transform:translateX(-50%);top:2px;color:${color};background:${COLORS.dark};padding:1px 3px;border-radius:3px;font:800 8px/1.15 system-ui;white-space:nowrap;pointer-events:none;z-index:9;`, `${gap}f`);
        label.title = line.title;
        layer.appendChild(label);
    }
}

function enabledFor(node, model) {
    if (node?.properties?.[PROP_ENABLED] === false) return false;
    // The spacing recommendations below are intentionally scoped to Positioned Guides.
    return model.mode === "longvid_guides" || model.mode === "longvid_positioned_guides";
}

function renderOverlay(node, ruler, layer) {
    const model = guidanceModel(node);
    layer.replaceChildren();

    const active = enabledFor(node, model);
    const chip = el("button",
        `position:absolute;left:5px;top:4px;height:18px;padding:1px 6px;border:1px solid ${active ? COLORS.cyanEdge : "#777"};border-radius:4px;background:${COLORS.dark};color:${active ? COLORS.text : "#AAA"};font:800 8px/1 system-ui;letter-spacing:.04em;pointer-events:auto;cursor:pointer;z-index:30;`,
        active ? "H3 GUIDANCE ON" : "H3 GUIDANCE OFF"
    );
    chip.type = "button";
    chip.title = active
        ? "Visual authoring overlay only. Click to hide. It does not modify prompts, guides, planner, conditioning or rendering."
        : "Click to show H3 Positioned Guides authoring overlay.";
    chip.onclick = (event) => {
        event.preventDefault();
        event.stopPropagation();
        node.properties ||= {};
        node.properties[PROP_ENABLED] = !active;
        try { node.graph?.change?.(); app.graph?.change?.(); } catch {}
        renderOverlay(node, ruler, layer);
    };
    layer.appendChild(chip);

    if (!active) return;
    const { total, guides, fps } = model;
    if (total < 2) return;

    for (const guide of guides) {
        const preserveEnd = Math.min(total, guide + PRESERVE_FRAMES);
        band(
            layer, guide, preserveEnd, total,
            COLORS.cyanFill, COLORS.cyanEdge,
            `${PRESERVE_FRAMES}f / ${(PRESERVE_FRAMES / fps).toFixed(2)}s motion-preserve recommendation after this guide. Prefer not to add another major visual checkpoint inside this zone.`
        );
        const prefStart = guide + PREFERRED_FRAMES;
        const prefEnd = Math.min(total, guide + PREFERRED_END_FRAMES);
        if (prefStart < total) {
            band(
                layer, prefStart, prefEnd, total,
                COLORS.greenFill, COLORS.green,
                `${PREFERRED_FRAMES}–${PREFERRED_END_FRAMES}f / ${(PREFERRED_FRAMES / fps).toFixed(2)}–${(PREFERRED_END_FRAMES / fps).toFixed(2)}s preferred working zone for the next substantial visual checkpoint.`
            );
            marker(
                layer, prefStart, total, COLORS.green,
                `${PREFERRED_FRAMES}f / ${(PREFERRED_FRAMES / fps).toFixed(2)}s preferred spacing marker from the previous guide.`,
                "4s", 4
            );
        }
    }

    for (let i = 0; i + 1 < guides.length; i++) pairGap(layer, guides[i], guides[i + 1], total, fps);

    for (let frame = H3_NATIVE_WINDOW; frame < total; frame += H3_NATIVE_WINDOW) {
        marker(
            layer, frame, total, COLORS.red,
            `H3 native technical window boundary · ${frame}f / ${(frame / fps).toFixed(2)}s. R42 may need a continuation chunk beyond this point.`,
            frame === H3_NATIVE_WINDOW ? "362f" : `${frame}f`, 12
        );
    }
}

function installOnRuler(node, ruler) {
    if (!ruler || ruler._iamccsH3GuidanceInstalled) return;
    ruler._iamccsH3GuidanceInstalled = true;

    let layer = null;
    const ensure = () => {
        if (!ruler.isConnected) return;
        layer = ruler.querySelector(`:scope > [${OVERLAY_ATTR}]`);
        if (!layer) {
            layer = el("div", "position:absolute;inset:0;overflow:hidden;pointer-events:none;z-index:14;");
            layer.setAttribute(OVERLAY_ATTR, "1");
            ruler.appendChild(layer);
        }
        renderOverlay(node, ruler, layer);
    };

    const observer = new MutationObserver(() => {
        if (!ruler.querySelector(`:scope > [${OVERLAY_ATTR}]`)) requestAnimationFrame(ensure);
    });
    observer.observe(ruler, { childList: true });

    let signature = "";
    const poll = window.setInterval(() => {
        if (!ruler.isConnected || !node?.graph) {
            window.clearInterval(poll);
            observer.disconnect();
            return;
        }
        const model = guidanceModel(node);
        const next = JSON.stringify([
            model.mode, model.total, model.guides,
            node?.properties?.[PROP_ENABLED] !== false,
            ruler.clientWidth,
        ]);
        if (next !== signature) {
            signature = next;
            ensure();
        }
    }, 500);

    node._iamccsH3GuidanceCleanup = () => {
        window.clearInterval(poll);
        observer.disconnect();
        try { ruler.querySelector(`:scope > [${OVERLAY_ATTR}]`)?.remove(); } catch {}
    };

    ensure();
}

function tryAttach(node) {
    const ruler = findRuler(node);
    if (!ruler) return false;
    installOnRuler(node, ruler);
    return true;
}

function scheduleAttach(node) {
    const attempts = [0, 100, 300, 750, 1500, 3000];
    for (const delay of attempts) {
        window.setTimeout(() => {
            if (!node?._iamccsH3GuidanceCleanup) tryAttach(node);
        }, delay);
    }
}

app.registerExtension({
    name: "iamccs.h3.timeline_guidance_overlay",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== TARGET_NODE) return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            scheduleAttach(this);
            return result;
        };

        const originalRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            try { this._iamccsH3GuidanceCleanup?.(); } catch {}
            this._iamccsH3GuidanceCleanup = null;
            return originalRemoved?.apply(this, arguments);
        };
    },
});
