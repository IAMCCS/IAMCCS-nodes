import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "IAMCCS_ShotboardPlannerV2V";
const VERSION = "2026-07-12-shotboard-v2v-settings-meter-ui5";
const WIDTH = 1840;
const WIDGET_HEIGHT = 980;
const NODE_HEIGHT = 1040;

function removeExistingShotboardDom(node) {
    if (!Array.isArray(node.widgets)) return;
    for (let i = node.widgets.length - 1; i >= 0; i--) {
        const item = node.widgets[i];
        if (item?.type === "iamccs_shotboard_planner_v2v" || item?.name === "Shotboard Planner V2V") {
            try { item.element?.remove?.(); } catch {}
            node.widgets.splice(i, 1);
        }
    }
}

function nodeName(node) {
    return String(node?.comfyClass || node?.type || node?.constructor?.type || "");
}

function widget(node, name) {
    return (node.widgets || []).find((item) => item?.name === name);
}

function read(node, name, fallback = "") {
    const value = widget(node, name)?.value;
    return value === undefined || value === null || String(value) === "" ? fallback : value;
}

function write(node, name, value) {
    const item = widget(node, name);
    if (!item) return;
    item.value = value;
    try { item.callback?.(value, null, node); } catch {}
    node.setDirtyCanvas?.(true, true);
}

function hideWidget(item) {
    if (!item || item._iamccsV2VShotboardHidden) return;
    item._iamccsV2VShotboardHidden = true;
    item.hidden = true;
    item.disabled = true;
    item.computeSize = () => [0, 0];
    item.draw = () => {};
    item.type = "hidden";
    item.options = Object.assign({}, item.options || {}, { hidden: true });
}

function setWidgetOnNode(target, names, value) {
    for (const name of names) {
        const item = widget(target, name);
        if (!item) continue;
        item.value = value;
        try { item.callback?.(value, null, target); } catch {}
        target.setDirtyCanvas?.(true, true);
        return true;
    }
    return false;
}

function nearestNode(source, predicate) {
    const nodes = Array.isArray(app?.graph?._nodes) ? app.graph._nodes : [];
    const candidates = nodes.filter((item) => item !== source && predicate(item));
    candidates.sort((a, b) => {
        const ax = Number(a.pos?.[0] || 0) - Number(source.pos?.[0] || 0);
        const ay = Number(a.pos?.[1] || 0) - Number(source.pos?.[1] || 0);
        const bx = Number(b.pos?.[0] || 0) - Number(source.pos?.[0] || 0);
        const by = Number(b.pos?.[1] || 0) - Number(source.pos?.[1] || 0);
        return (ax * ax + ay * ay) - (bx * bx + by * by);
    });
    return candidates[0] || null;
}

function syncVideoBackend(node, value) {
    const target = nearestNode(node, (item) => nodeName(item) === "VHS_LoadVideo");
    return target ? setWidgetOnNode(target, ["video"], value) : false;
}

function syncImageBackend(node, value) {
    const target = nearestNode(node, (item) => nodeName(item) === "LoadImage");
    return target ? setWidgetOnNode(target, ["image", "image_upload"], value) || setWidgetOnNode(target, [0], value) : false;
}

function graphNodes() {
    return Array.isArray(app?.graph?._nodes) ? app.graph._nodes : [];
}

function syncBackendGraphWidgets(node, payload) {
    const mode = payload.backend_mode || resolveBackendMode(node);
    const family = payload.backend_family || BACKEND_MODES[mode]?.family || "ltx";
    let touched = 0;
    for (const item of graphNodes()) {
        if (item === node) continue;
        const type = nodeName(item);
        const title = String(item.title || "");
        const prefix = String(widgetValue(item, "filename_prefix") || "");
        if (type === "VHS_LoadVideo") {
            if (writeWidgetObjectValue(item, "video", payload.source_video_path)) touched++;
            if (writeWidgetObjectValue(item, "frame_load_cap", payload.frame_load_cap)) touched++;
            if (writeWidgetObjectValue(item, "force_rate", Math.round(Number(payload.fps || 0)))) touched++;
            continue;
        }
        if (type === "LoadImage") {
            if (writeWidgetObjectValue(item, "image", payload.source_image_path)) touched++;
            continue;
        }
        if (type === "CLIPTextEncode" || type === "CLIPTextEncodeFlux") {
            const lower = title.toLowerCase();
            const text = lower.includes("negative") ? payload.negative_prompt : payload.global_prompt;
            if (setWidgetOnNode(item, ["text", 0], text)) touched++;
            continue;
        }
        if (type === "VHS_VideoCombine") {
            const shouldOwn =
                (family === "scail2" && (prefix.includes("SCAIL2") || title.includes("SCAIL"))) ||
                (family === "wananimate" && (prefix.includes("WAN") || title.includes("WAN"))) ||
                (family === "pose_transfer" && (prefix.includes("POSE") || title.includes("POSE")));
            if (shouldOwn) {
                if (writeWidgetObjectValue(item, "filename_prefix", payload.output_prefix || prefix)) touched++;
                if (writeWidgetObjectValue(item, "frame_rate", Number(payload.fps || 24))) touched++;
            }
        }
        if (type === "SaveImage" || type === "SaveImageKJ") {
            if (family === "pose_transfer" && setWidgetOnNode(item, ["filename_prefix", 0], payload.output_prefix || "IAMCCS/POSE_TRANSFER_SHOTBOARD")) touched++;
        }
    }
    return touched;
}

function viewUrl(filename, type = "input") {
    const file = String(filename || "").trim();
    if (!file) return "";
    const parts = file.split("/");
    const name = parts.pop();
    const subfolder = parts.join("/");
    const query = new URLSearchParams({ filename: name, type });
    if (subfolder) query.set("subfolder", subfolder);
    return `/view?${query.toString()}`;
}

function viewUrlFromPreviewParams(params) {
    if (!params?.filename) return "";
    const query = new URLSearchParams({ filename: params.filename, type: params.type || "output" });
    if (params.subfolder) query.set("subfolder", params.subfolder);
    if (params.format) query.set("format", params.format);
    return `/view?${query.toString()}`;
}

function widgetValue(target, name) {
    const item = widget(target, name);
    if (item?.value !== undefined) return item.value;
    const raw = target?.widgets_values;
    if (raw && typeof raw === "object" && !Array.isArray(raw)) return raw[name];
    return undefined;
}

function writeWidgetObjectValue(target, name, value) {
    const item = widget(target, name);
    if (item) {
        item.value = value;
        try { item.callback?.(value, null, target); } catch {}
        target.setDirtyCanvas?.(true, true);
        return true;
    }
    if (target?.widgets_values && typeof target.widgets_values === "object" && !Array.isArray(target.widgets_values)) {
        target.widgets_values[name] = value;
        target.setDirtyCanvas?.(true, true);
        return true;
    }
    return false;
}

async function uploadFile(file) {
    const body = new FormData();
    body.append("image", file);
    const response = await api.fetchApi("/upload/image", { method: "POST", body });
    if (!response || response.status !== 200) throw new Error(`upload failed: ${response?.status || "no response"}`);
    const data = await response.json();
    const name = data?.name || file.name;
    const subfolder = data?.subfolder || "";
    return subfolder ? `${subfolder}/${name}` : name;
}

function injectStyle() {
    const existing = document.getElementById("iamccs-shotboard-v2v-style");
    const style = existing || document.createElement("style");
    style.id = "iamccs-shotboard-v2v-style";
    style.textContent = `
.iamccs-v2v-board{position:relative;box-sizing:border-box;width:100%;height:100%;padding:10px 12px;border:1px solid rgba(95,198,218,.58);border-radius:6px;background:#0f1419;color:#eef5f8;font-family:Inter,Arial,sans-serif;font-size:12px;overflow:hidden;box-shadow:inset 0 1px 0 rgba(255,255,255,.05)}
.iamccs-v2v-board *{box-sizing:border-box;letter-spacing:0}
.iamccs-v2v-board.is-full-editor{position:fixed;left:18px;right:18px;top:18px;bottom:18px;width:auto!important;height:auto!important;z-index:999999;border-color:#d8b860;background:#10151a;box-shadow:0 24px 80px rgba(0,0,0,.72),inset 0 1px 0 rgba(255,255,255,.08)}
.iamccs-v2v-board.is-full-editor .iamccs-v2v-main{grid-template-columns:430px minmax(0,1fr) 360px}
.iamccs-v2v-board.is-full-editor .iamccs-v2v-timebar{min-height:calc(100vh - 300px)}
.iamccs-v2v-board.is-full-editor .iamccs-v2v-track{height:clamp(210px,36vh,360px)}
.iamccs-v2v-board.is-full-editor .iamccs-v2v-handle{height:100%}
.iamccs-v2v-head{display:grid;grid-template-columns:auto minmax(0,1fr) auto auto;align-items:center;gap:10px;height:34px;margin:0 0 10px;border-bottom:1px solid rgba(95,198,218,.28)}
.iamccs-v2v-title{font-size:18px;font-weight:900;color:#fff;white-space:nowrap}
.iamccs-v2v-path{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;text-align:right;color:#9fb3c1;font-size:11px}
.iamccs-v2v-head-editor{height:24px;min-width:112px;padding:0 12px;border-color:#62c99e;background:#1e5b4c;color:#f3fff9}
.iamccs-v2v-head-settings{height:24px;min-width:92px;padding:0 12px;border-color:#8db5e4;background:#203a59;color:#edf6ff}
.iamccs-v2v-settings-drawer{display:none;position:absolute;z-index:40;top:50px;right:10px;bottom:10px;width:min(560px,calc(100% - 20px));min-height:0;flex-direction:column;border:1px solid rgba(141,181,228,.74);border-radius:6px;background:#111821;box-shadow:0 20px 60px rgba(0,0,0,.7),inset 0 1px 0 rgba(255,255,255,.07);overflow:hidden}
.iamccs-v2v-settings-drawer.is-open{display:flex}
.iamccs-v2v-settings-head{height:42px;display:flex;align-items:center;justify-content:space-between;gap:10px;padding:0 10px;border-bottom:1px solid rgba(141,181,228,.32);background:#17283b;color:#eef7ff;font-size:12px;font-weight:900;text-transform:uppercase}
.iamccs-v2v-settings-head small{color:#9cb4ca;font-size:10px;font-weight:800}
.iamccs-v2v-settings-close{height:26px;padding:0 10px;border:1px solid #9eb8d2;border-radius:3px;background:#29445f;color:#f4fbff;font-size:10px;font-weight:900;cursor:pointer}
.iamccs-v2v-settings-drawer .iamccs-v2v-panel-body{padding:10px;overflow-y:auto}
.iamccs-v2v-main{display:grid;grid-template-columns:minmax(410px,430px) minmax(0,1fr) minmax(300px,360px);gap:12px;height:calc(100% - 44px);min-height:0}
.iamccs-v2v-panel{min-height:0;display:flex;flex-direction:column;border:1px solid rgba(160,184,205,.28);border-radius:5px;background:#141c24;overflow:hidden;box-shadow:inset 0 1px 0 rgba(255,255,255,.04)}
.iamccs-v2v-panel.preview{border-color:rgba(112,202,222,.44);background:#121921}
.iamccs-v2v-panel.timeline{border-color:rgba(218,184,96,.46);background:#14120f}
.iamccs-v2v-panel.controls{border-color:rgba(172,139,219,.46);background:#15131d}
.iamccs-v2v-panel-head{height:34px;display:flex;align-items:center;justify-content:space-between;padding:0 10px;border-bottom:1px solid rgba(160,184,205,.2);background:#192630;font-size:11px;font-weight:900;text-transform:uppercase;color:#e9f7fb}
.iamccs-v2v-panel-body{flex:1;min-height:0;padding:8px;overflow-y:auto;overflow-x:hidden;scrollbar-width:thin;scrollbar-color:#668292 #111820}
.iamccs-v2v-panel-body::-webkit-scrollbar{width:8px}
.iamccs-v2v-panel-body::-webkit-scrollbar-thumb{background:#668292;border-radius:8px}
.iamccs-v2v-media-stack{display:flex;flex-direction:column;gap:10px;height:auto;min-height:100%}
.iamccs-v2v-media-card{position:relative;z-index:1;flex:0 0 auto;border:1px solid rgba(160,184,205,.24);border-radius:5px;background:#101820;overflow:hidden;min-height:0}.iamccs-v2v-media-card.source-video{padding-bottom:2px}
.iamccs-v2v-media-title{height:26px;display:flex;align-items:center;justify-content:space-between;padding:0 9px;color:#b7c7d1;font-size:11px;font-weight:900;background:#16232d}
.iamccs-v2v-media{height:var(--iamccs-v2v-video-h,210px);min-height:145px;max-height:300px;background:#05090d;display:flex;align-items:center;justify-content:center;color:#7f919d;font-size:12px;overflow:hidden}
.iamccs-v2v-media.ref{height:132px;min-height:110px;max-height:170px}
.iamccs-v2v-media.result{height:118px;min-height:96px;max-height:150px;border-top:1px solid rgba(160,184,205,.12)}
.iamccs-v2v-media video,.iamccs-v2v-media img{width:100%;height:100%;object-fit:contain;background:#05090d}
.iamccs-v2v-media-actions{display:flex;gap:7px;flex-wrap:wrap;padding:8px;border-top:1px solid rgba(160,184,205,.14)}
.iamccs-v2v-media-actions.is-hidden{display:none}
.iamccs-v2v-preview-monitor{height:268px;border:1px solid rgba(118,214,235,.38);border-radius:5px;background:#05090d;display:flex;align-items:center;justify-content:center;overflow:hidden;color:#8ea8b6;position:relative}
.iamccs-v2v-preview-monitor::before{content:"SOURCE";position:absolute;left:10px;top:8px;padding:3px 7px;border:1px solid rgba(118,214,235,.45);border-radius:3px;background:rgba(8,17,23,.86);color:#d9fbff;font-size:10px;font-weight:900;z-index:2}
.iamccs-v2v-preview-monitor img,.iamccs-v2v-preview-monitor video{width:100%;height:100%;object-fit:contain;background:#05090d}
.iamccs-v2v-backend-preview-panel{border-color:rgba(118,214,235,.46);background:#101a22}
.iamccs-v2v-backend-preview-strip{display:flex;flex-direction:column;gap:9px;margin:0}
.iamccs-v2v-backend-preview{height:clamp(132px,18vh,188px);min-height:118px;border:1px solid rgba(160,184,205,.3);border-radius:4px;background:#05090d;overflow:hidden;display:flex;flex-direction:column;color:#7f919d;font-size:10px;font-weight:800;text-transform:uppercase}
.iamccs-v2v-preview-slot-head{height:25px;flex:0 0 25px;display:flex;align-items:center;padding:0 8px;border-bottom:1px solid rgba(160,184,205,.18);background:#142531;color:#d9f6ff;font-size:10px;font-weight:900;letter-spacing:.04em}
.iamccs-v2v-preview-slot-media{position:relative;flex:1;min-height:0;display:flex;align-items:center;justify-content:center;overflow:hidden}
.iamccs-v2v-backend-preview video,.iamccs-v2v-backend-preview img{width:100%;height:100%;object-fit:contain;background:#05090d}
.iamccs-v2v-preview-slot-media .iamccs-v2v-preview-placeholder{min-height:100%}
.iamccs-v2v-preview-placeholder{width:100%;height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;background:radial-gradient(circle at 50% 36%,#143442,#061016 62%);color:#bfd8e1;text-align:center;font-size:12px;font-weight:800}
.iamccs-v2v-preview-placeholder span{color:#7f9ba8;font-size:11px;font-weight:700}
.iamccs-v2v-resize{height:13px;display:flex;align-items:center;justify-content:center;border-top:1px solid rgba(160,184,205,.14);background:#111c25;cursor:row-resize;color:#7fa6b6;font-size:9px;font-weight:900}
.iamccs-v2v-resize::before{content:"";width:64px;height:3px;border-radius:3px;background:#527b8b}
.iamccs-v2v-actions{display:flex;gap:7px;flex-wrap:wrap;padding:8px;border-top:1px solid rgba(160,184,205,.14)}
.iamccs-v2v-btn{height:30px;padding:0 12px;border:1px solid #52687a;border-radius:3px;background:#243341;color:#f3fbff;font-size:11px;font-weight:900;cursor:pointer;white-space:nowrap;box-shadow:inset 0 1px 0 rgba(255,255,255,.07)}
.iamccs-v2v-btn:hover{background:#2d4051}
.iamccs-v2v-btn.is-active{border-color:#9ee9ff;background:#2b6074;box-shadow:inset 0 0 0 1px rgba(255,255,255,.16),0 0 0 1px rgba(95,198,218,.16)}
.iamccs-v2v-btn.good{background:#1e5b4c;border-color:#62c99e}
.iamccs-v2v-btn.warn{background:#50375b;border-color:#bd89d9}
.iamccs-v2v-mode-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.iamccs-v2v-mode-grid.backend{grid-template-columns:1fr 1fr}
.iamccs-v2v-mode-grid.profile{grid-template-columns:1fr 1fr}
.iamccs-v2v-mode-grid.posemodes{grid-template-columns:1fr 1fr}
.iamccs-v2v-mode-grid .iamccs-v2v-btn{width:100%;height:36px;text-align:center;border-radius:3px}
.iamccs-v2v-mode-hint{margin-top:8px;padding:8px;border:1px solid rgba(160,184,205,.2);border-radius:3px;background:#0c151d;color:#9fb2bf;font-size:11px;line-height:1.35}
.iamccs-v2v-mode-panel{display:flex;flex-direction:column;gap:8px}
.iamccs-v2v-mode-section{display:flex;flex-direction:column;gap:8px;border:1px solid rgba(160,184,205,.22);border-radius:4px;background:#0d151d;padding:8px}
.iamccs-v2v-mode-section.is-hidden{display:none}
.iamccs-v2v-mode-title{display:flex;align-items:center;justify-content:space-between;gap:8px;color:#eaf7ff;font-size:11px;font-weight:900;text-transform:uppercase}
.iamccs-v2v-mode-title span:last-child{color:#8fa5b5;font-size:10px}
.iamccs-v2v-mini-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.iamccs-v2v-mini-slot{min-height:42px;border:1px solid rgba(160,184,205,.25);border-radius:3px;background:#091118;padding:7px 8px;display:flex;flex-direction:column;justify-content:center;gap:2px}
.iamccs-v2v-mini-slot b{font-size:10px;color:#eaf7ff;text-transform:uppercase}
.iamccs-v2v-mini-slot span{font-size:10px;color:#8fa5b5;line-height:1.25}
.iamccs-v2v-toggle-row{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.iamccs-v2v-toggle-row .iamccs-v2v-btn{height:30px}
.iamccs-v2v-channel-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-top:8px}
.iamccs-v2v-channel{height:30px;border:1px solid rgba(160,184,205,.26);border-radius:3px;background:#111a22;color:#8fa3b2;font-size:10px;font-weight:900;text-transform:uppercase;display:flex;align-items:center;justify-content:center}
.iamccs-v2v-channel.is-active{border-color:#8ee6ff;background:#174253;color:#e8fbff;box-shadow:inset 0 0 0 1px rgba(255,255,255,.08)}
.iamccs-v2v-editor-row{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}
.iamccs-v2v-editor-drawer{display:none;margin-top:8px;padding:9px;border:1px solid rgba(218,184,96,.34);border-radius:6px;background:#1f1b12;color:#efd89a;font-size:11px;line-height:1.35}
.iamccs-v2v-board.is-editor-open .iamccs-v2v-editor-drawer{display:block}
.iamccs-v2v-field{display:grid;grid-template-columns:94px minmax(0,1fr);gap:7px;align-items:center;margin:6px 0}
.iamccs-v2v-field label{font-size:11px;color:#adbdc8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.iamccs-v2v-field input,.iamccs-v2v-field select,.iamccs-v2v-field textarea{width:100%;min-width:0;border:1px solid rgba(160,184,205,.34);border-radius:3px;background:#0b1219;color:#f7fbff;padding:7px 8px;font-size:12px;outline:none}
.iamccs-v2v-field textarea{resize:vertical;min-height:94px;line-height:1.32;font-family:Consolas,monospace;background:#f8f4ea;color:#111;border-color:#d7c797;font-weight:700}
.iamccs-v2v-field input[type="number"]{cursor:ew-resize;font-variant-numeric:tabular-nums}
.iamccs-v2v-drag-number{border-color:#7897ad!important;background:#101b25!important}
.iamccs-v2v-path-field{padding:2px 8px 10px}
.iamccs-v2v-mid{display:grid;grid-template-rows:minmax(0,1fr) auto;gap:8px;height:100%;min-height:0}
.iamccs-v2v-prompt-grid{display:grid;grid-template-columns:1fr;gap:7px}
.iamccs-v2v-prompt-grid textarea{height:58px}
.iamccs-v2v-timeline-card{min-height:0;border:1px solid rgba(218,184,96,.34);border-radius:7px;background:#0b1117;overflow:hidden;display:flex;flex-direction:column}
.iamccs-v2v-timeline-head{height:34px;display:flex;align-items:center;justify-content:space-between;padding:0 10px;background:#271f13;color:#f6e7b4;font-size:11px;font-weight:900;text-transform:uppercase}
.iamccs-v2v-timebar{position:relative;flex:1;min-height:0;display:flex;flex-direction:column;background:#091016;overflow:hidden;padding:0 0 2px}
.iamccs-v2v-ruler{position:relative;left:auto;right:auto;top:auto;height:34px;flex:0 0 34px;margin:0 16px;border-bottom:1px solid rgba(151,181,201,.24);color:#9aaebd;font-size:10px}
.iamccs-v2v-tick{position:absolute;top:19px;width:1px;height:12px;background:rgba(151,181,201,.34)}
.iamccs-v2v-tick span{position:absolute;top:-16px;left:-12px;color:#8ea3b2;font-size:10px}
.iamccs-v2v-track{position:relative;left:auto;right:auto;top:auto;height:clamp(180px,32vh,310px);flex:0 1 310px;margin:12px 16px 0;border:1px solid rgba(218,184,96,.46);border-radius:4px;background:#060a0f;overflow:hidden;cursor:crosshair}
.iamccs-v2v-timeline-media{position:absolute;inset:0;background:linear-gradient(90deg,#111b24,#182838);overflow:hidden}
.iamccs-v2v-timeline-media video{width:100%;height:100%;object-fit:cover;filter:saturate(.9) contrast(.92);opacity:.9}
.iamccs-v2v-timeline-media-empty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#758a99;font-size:12px;background:repeating-linear-gradient(90deg,#101822 0,#101822 34px,#142130 34px,#142130 68px)}
.iamccs-v2v-timeline-shade{position:absolute;top:0;bottom:0;background:rgba(0,0,0,.56);backdrop-filter:saturate(.7)}
.iamccs-v2v-selected{position:absolute;top:0;bottom:0;background:linear-gradient(180deg,rgba(52,160,135,.38),rgba(61,103,154,.42));box-shadow:inset 0 0 0 2px rgba(167,245,255,.55)}
.iamccs-v2v-selected::after{content:"";position:absolute;inset:0;background:linear-gradient(180deg,rgba(255,255,255,.08),rgba(255,255,255,0));pointer-events:none}
.iamccs-v2v-handle{position:absolute;top:0;width:20px;height:100%;margin-left:-10px;border:1px solid #f4e6b8;border-radius:3px;background:#f3ead1;color:#0d1a22;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:900;cursor:ew-resize;box-shadow:0 8px 22px rgba(0,0,0,.34);z-index:6}
.iamccs-v2v-handle.is-active{background:#d6b65d;border-color:#fff1b4;color:#060606;box-shadow:0 0 0 2px rgba(214,182,93,.24),0 8px 22px rgba(0,0,0,.34)}
.iamccs-v2v-playhead{position:absolute;top:0;bottom:0;width:2px;background:#d6b65d;box-shadow:0 0 0 1px rgba(0,0,0,.55),0 0 12px rgba(214,182,93,.7);pointer-events:none;z-index:7}
.iamccs-v2v-playhead::before{content:"";position:absolute;top:-9px;left:-6px;width:14px;height:14px;background:#d6b65d;clip-path:polygon(50% 100%,0 0,100% 0)}
.iamccs-v2v-playhead-time{position:absolute;top:8px;transform:translateX(-50%);padding:3px 6px;border-radius:4px;background:rgba(0,0,0,.72);color:#fff0b4;font-size:10px;font-weight:900;z-index:8;pointer-events:none}
.iamccs-v2v-frame-bubble{display:none}
.iamccs-v2v-frame-bubble canvas{width:100%;height:58px;display:block;background:#020405}
.iamccs-v2v-frame-bubble span{display:block;height:20px;line-height:20px;padding:0 6px;background:#211a10;color:#f8e8b5;font-size:10px;font-weight:900;text-align:center;font-variant-numeric:tabular-nums}
.iamccs-v2v-segment-head{position:relative;left:auto;right:auto;top:auto;height:24px;flex:0 0 24px;margin:10px 16px 0;display:flex;justify-content:space-between;color:#9fb5c6;font-size:10px;font-weight:900;text-transform:uppercase}
.iamccs-v2v-segments{position:relative;left:auto;right:auto;top:auto;height:78px;flex:0 0 78px;display:flex;gap:6px;overflow-x:auto;overflow-y:hidden;scrollbar-width:thin;margin:0 16px;padding:6px;border:1px solid rgba(160,184,205,.18);border-radius:4px;background:#081018}
.iamccs-v2v-segment{height:100%;min-width:126px;flex:1 0 126px;border-radius:3px;background:#273a50;border:1px solid rgba(196,220,236,.24);display:flex;flex-direction:column;align-items:stretch;justify-content:center;gap:5px;padding:7px 8px;color:#e6f4fb;font-size:11px;font-weight:900;overflow:hidden}
.iamccs-v2v-segment-code{display:flex;justify-content:space-between;gap:5px;color:#f7e6ae;font-size:10px}
.iamccs-v2v-segment-meter{height:7px;border:1px solid rgba(196,220,236,.25);border-radius:2px;background:#101c27;overflow:hidden}
.iamccs-v2v-segment-meter i{display:block;height:100%;background:linear-gradient(90deg,#d6b65d,#71d4e7);box-shadow:0 0 10px rgba(113,212,231,.22)}
.iamccs-v2v-playbar{position:relative;left:auto;right:auto;top:auto;height:46px;flex:0 0 46px;margin:10px 16px 12px;border:1px solid rgba(218,184,96,.24);border-radius:4px;background:#12100d;display:grid;grid-template-columns:112px minmax(0,1fr) 74px;gap:10px;align-items:center;padding:0 10px}
.iamccs-v2v-playbar button{height:28px;border:1px solid rgba(218,184,96,.48);border-radius:3px;background:#2a2115;color:#f7e6ae;font-size:11px;font-weight:900;cursor:pointer}
.iamccs-v2v-transfer-controls{display:grid;grid-template-columns:30px 46px 30px;gap:3px}
.iamccs-v2v-analog{height:6px;border-radius:8px;background:linear-gradient(90deg,#8d7440,#2c3d4e);position:relative;cursor:ew-resize;box-shadow:inset 0 1px 0 rgba(255,255,255,.15)}
.iamccs-v2v-analog-thumb{position:absolute;top:50%;width:14px;height:14px;border-radius:50%;background:#d6b65d;border:1px solid #fff2b9;transform:translate(-50%,-50%);box-shadow:0 1px 8px rgba(0,0,0,.5)}
.iamccs-v2v-time-label{font-size:11px;color:#f0d991;font-weight:900;text-align:right;font-variant-numeric:tabular-nums}
.iamccs-v2v-scrub-wrap{height:0;padding:0;border:0;overflow:hidden}
.iamccs-v2v-scrub-row{display:grid;grid-template-columns:60px minmax(0,1fr) 66px;gap:8px;align-items:center}
.iamccs-v2v-scrub-row span{color:#a9bac5;font-size:11px;font-weight:800}
.iamccs-v2v-scrub{width:100%;accent-color:#66d2e7}
.iamccs-v2v-readout{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.iamccs-v2v-chip{border:1px solid rgba(160,184,205,.24);border-radius:3px;background:#101820;padding:8px 9px;min-width:0}
.iamccs-v2v-chip span{display:block;color:#91a7b6;font-size:10px;text-transform:uppercase;font-weight:900;margin-bottom:3px}
.iamccs-v2v-chip b{display:block;font-size:13px;color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.iamccs-v2v-control-group{border:1px solid rgba(160,184,205,.24);border-radius:4px;background:#101820;margin-bottom:8px;overflow:hidden}
.iamccs-v2v-control-group.run{border-color:rgba(118,214,235,.38);background:#0f1d24}
.iamccs-v2v-control-group.prompts{border-color:rgba(216,199,151,.46);background:#252014}
.iamccs-v2v-control-group.timing{border-color:rgba(106,167,230,.38);background:#111b2a}
.iamccs-v2v-control-group.frame{border-color:rgba(114,198,154,.38);background:#102018}
.iamccs-v2v-control-group.pose{border-color:rgba(190,135,217,.38);background:#201326}
.iamccs-v2v-control-group.audio{border-color:rgba(218,129,103,.38);background:#241712}
.iamccs-v2v-control-title{height:28px;display:flex;align-items:center;justify-content:space-between;padding:0 9px;background:#172530;color:#dceff6;font-size:11px;font-weight:900;text-transform:uppercase}
.iamccs-v2v-control-group.run .iamccs-v2v-control-title{background:#16313b;color:#d8fbff}
.iamccs-v2v-control-group.prompts .iamccs-v2v-control-title{background:#4a3e21;color:#fff1bd}
.iamccs-v2v-control-group.timing .iamccs-v2v-control-title{background:#172d4b;color:#d7eaff}
.iamccs-v2v-control-group.frame .iamccs-v2v-control-title{background:#173b2b;color:#dfffe9}
.iamccs-v2v-control-group.pose .iamccs-v2v-control-title{background:#3b1c48;color:#f5d7ff}
.iamccs-v2v-control-group.audio .iamccs-v2v-control-title{background:#4a2419;color:#ffd9c7}
.iamccs-v2v-control-body{padding:8px}
.iamccs-v2v-two{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.iamccs-v2v-two .iamccs-v2v-field{grid-template-columns:68px minmax(0,1fr)}
.iamccs-v2v-note{color:#91a4b3;font-size:11px;line-height:1.35;padding:8px;border-top:1px solid rgba(160,184,205,.14)}
.iamccs-v2v-drop{outline:2px solid #9ed7ff;outline-offset:-4px}
`;
    if (!existing) document.head.appendChild(style);
}

function field(labelText, control) {
    const row = document.createElement("div");
    row.className = "iamccs-v2v-field";
    const label = document.createElement("label");
    label.textContent = labelText;
    row.append(label, control);
    return row;
}

function button(label, tone = "") {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = label;
    btn.className = `iamccs-v2v-btn ${tone}`.trim();
    return btn;
}

function input(type, value = "") {
    const el = document.createElement("input");
    el.type = type;
    el.value = value;
    return el;
}

function select(options, value) {
    const el = document.createElement("select");
    for (const item of options) {
        const option = document.createElement("option");
        option.value = item;
        option.textContent = item.replaceAll("_", " ");
        el.appendChild(option);
    }
    el.value = value;
    return el;
}

function miniSlot(title, subtitle = "") {
    const slot = document.createElement("div");
    slot.className = "iamccs-v2v-mini-slot";
    slot.innerHTML = `<b>${title}</b><span>${subtitle}</span>`;
    return slot;
}

const BACKEND_MODES = {
    ltx_simple: {
        label: "LTX Simple",
        family: "ltx",
        backendProfile: "ltx23_v2v_infinite_lipsync",
        poseMode: "dwpose_openpose",
        outputPrefix: "IAMCCS/LTX23_V2V_SHOTBOARD",
        hint: "Current LTX V2V backend. Stable general video-to-video, DW pose, audio bridge and segment planning.",
    },
    scail2: {
        label: "SCAIL-2",
        family: "scail2",
        backendProfile: "scail2_single_person",
        poseMode: "source_pose_only",
        outputPrefix: "IAMCCS/SCAIL2_SHOTBOARD",
        hint: "SCAIL-2 backend from the attached single/multi identity workflows. Uses SAM 3.1 previews, masks and IAMCCS_ScailExtends.",
    },
    wananimate: {
        label: "WanAnimate",
        family: "wananimate",
        backendProfile: "wananimate_extension",
        poseMode: "source_pose_only",
        outputPrefix: "IAMCCS/WANANIMATE_SHOTBOARD",
        hint: "WanAnimate backend from the extension workflow. Prepared for SAM identity mask, pose/face/control previews and background lock.",
    },
    pose_transfer: {
        label: "Pose Transfer",
        family: "pose_transfer",
        backendProfile: "flux_klein_pose_transfer",
        poseMode: "image_pose_transfer",
        outputPrefix: "IAMCCS/POSE_TRANSFER_SHOTBOARD",
        hint: "FLUX Klein Pose Transfer mode. Canvas-style image + driver video + result asset, ready to hand off into V2V.",
    },
};

function resolveBackendMode(node) {
    const raw = String(read(node, "backend_mode", "") || read(node, "backend_profile", "ltx23_v2v_infinite_lipsync"));
    if (BACKEND_MODES[raw]) return raw;
    if (raw.includes("scail")) return "scail2";
    if (raw.includes("wan")) return "wananimate";
    if (raw.includes("pose") || raw.includes("flux")) return "pose_transfer";
    return "ltx_simple";
}

function controlGroup(title, subtitle, children, tone = "") {
    const group = document.createElement("section");
    group.className = `iamccs-v2v-control-group ${tone}`.trim();
    const head = document.createElement("div");
    head.className = "iamccs-v2v-control-title";
    head.innerHTML = `<span>${title}</span><span>${subtitle || ""}</span>`;
    const body = document.createElement("div");
    body.className = "iamccs-v2v-control-body";
    children.forEach((item) => body.appendChild(item));
    group.append(head, body);
    return group;
}

function enableNumberDrag(el, { step = 1, min = -Infinity, max = Infinity, precision = null, onCommit = null, onPreview = null } = {}) {
    if (!el || el._iamccsNumberDrag) return;
    el._iamccsNumberDrag = true;
    el.classList.add("iamccs-v2v-drag-number");
    el.title = "Drag left/right to adjust. Click to type.";
    el.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;
        const startX = event.clientX;
        const startValue = Number(el.value || 0);
        let dragging = false;
        const format = (value) => {
            const clamped = Math.max(min, Math.min(max, value));
            if (precision != null) return clamped.toFixed(precision);
            return String(Math.round(clamped));
        };
        const move = (moveEvent) => {
            const dx = moveEvent.clientX - startX;
            if (Math.abs(dx) < 3 && !dragging) return;
            dragging = true;
            moveEvent.preventDefault();
            el.value = format(startValue + dx * step);
            onPreview?.();
        };
        const up = () => {
            window.removeEventListener("pointermove", move);
            window.removeEventListener("pointerup", up);
            if (dragging) onCommit?.();
        };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up);
    });
}

function renderShotboardV2V(node) {
    if (node._iamccsShotboardV2VReady === VERSION) return;
    removeExistingShotboardDom(node);
    node._iamccsShotboardV2VReady = VERSION;
    injectStyle();
    (node.widgets || []).forEach(hideWidget);

    const root = document.createElement("div");
    root.className = "iamccs-v2v-board";
    const state = {
        videoObjectUrl: "",
        imageObjectUrl: "",
        playheadSec: 0,
        videoHeight: Number(read(node, "ui_video_height", 245)) || 245,
        timelineVideo: null,
        timelineVideoSrc: "",
        pendingSeek: 0,
        seekRaf: 0,
        autoDurationSyncedForSrc: "",
        activeTrim: "end",
    };
    root.style.setProperty("--iamccs-v2v-video-h", `${Math.max(155, Math.min(340, state.videoHeight))}px`);

    const status = document.createElement("div");
    status.className = "iamccs-v2v-path";
    const videoInput = document.createElement("input");
    videoInput.type = "file";
    videoInput.accept = "video/*";
    videoInput.style.display = "none";
    const imageInput = document.createElement("input");
    imageInput.type = "file";
    imageInput.accept = "image/*";
    imageInput.style.display = "none";
    const poseResultInput = document.createElement("input");
    poseResultInput.type = "file";
    poseResultInput.accept = "image/*,video/*";
    poseResultInput.style.display = "none";
    const poseUploadResult = button("Upload Result", "warn");
    root.append(videoInput, imageInput, poseResultInput);

    const head = document.createElement("div");
    head.className = "iamccs-v2v-head";
    const title = document.createElement("div");
    title.className = "iamccs-v2v-title";
    title.textContent = "IAMCCS Shotboard Planner V2V";
    const settingsBtn = button("Settings");
    settingsBtn.classList.add("iamccs-v2v-head-settings");
    const openEditorBtn = button("Open Editor", "good");
    openEditorBtn.classList.add("iamccs-v2v-head-editor");
    head.append(title, status, settingsBtn, openEditorBtn);
    root.appendChild(head);

    const main = document.createElement("div");
    main.className = "iamccs-v2v-main";
    root.appendChild(main);

    const videoPath = input("text", read(node, "source_video_path", "IMG_4145 2.mp4"));
    const imagePath = input("text", read(node, "source_image_path", "QWEN2509_FIRST_FRAME_DWPOSE_OPENPOSE_CONTROL_00001_.png"));

    const mediaPanel = document.createElement("section");
    mediaPanel.className = "iamccs-v2v-panel preview";
    mediaPanel.innerHTML = `<div class="iamccs-v2v-panel-head"><span>Media</span><span class="iamccs-v2v-media-mode-badge">source + reference</span></div>`;
    const mediaBody = document.createElement("div");
    mediaBody.className = "iamccs-v2v-panel-body";
    const mediaStack = document.createElement("div");
    mediaStack.className = "iamccs-v2v-media-stack";

    const previewBox = document.createElement("div");
    previewBox.className = "iamccs-v2v-preview-monitor";

    const sourceCard = document.createElement("section");
    sourceCard.className = "iamccs-v2v-media-card source-video";
    sourceCard.innerHTML = `<div class="iamccs-v2v-media-title"><span class="iamccs-v2v-source-title">Source video</span><span class="iamccs-v2v-source-subtitle">driver / timeline</span></div>`;
    const videoBtns = document.createElement("div");
    videoBtns.className = "iamccs-v2v-actions";
    const addVideo = button("Upload Video", "good");
    const syncVideo = button("Apply Path");
    videoBtns.append(addVideo, syncVideo);
    const videoPathWrap = document.createElement("div");
    videoPathWrap.className = "iamccs-v2v-path-field";
    videoPathWrap.appendChild(field("video", videoPath));
    sourceCard.append(previewBox, videoBtns, videoPathWrap);

    const imageCard = document.createElement("section");
    imageCard.className = "iamccs-v2v-media-card";
    imageCard.innerHTML = `<div class="iamccs-v2v-media-title"><span class="iamccs-v2v-image-title">Reference image</span><span class="iamccs-v2v-image-subtitle">identity / pose</span></div>`;
    const imageBox = document.createElement("div");
    imageBox.className = "iamccs-v2v-media ref";
    const imageBtns = document.createElement("div");
    imageBtns.className = "iamccs-v2v-actions";
    const addImage = button("Upload Image", "good");
    const syncImage = button("Apply Path");
    imageBtns.append(addImage, syncImage);
    const imagePathWrap = document.createElement("div");
    imagePathWrap.className = "iamccs-v2v-path-field";
    imagePathWrap.appendChild(field("image", imagePath));
    imageCard.append(imageBox, imageBtns, imagePathWrap);

    const resultCard = document.createElement("section");
    resultCard.className = "iamccs-v2v-media-card";
    resultCard.innerHTML = `<div class="iamccs-v2v-media-title"><span class="iamccs-v2v-result-title">Backend result</span><span class="iamccs-v2v-result-subtitle">final / transfer</span></div>`;
    const resultBox = document.createElement("div");
    resultBox.className = "iamccs-v2v-media result";
    const resultActions = document.createElement("div");
    resultActions.className = "iamccs-v2v-media-actions is-hidden";
    resultActions.appendChild(poseUploadResult);
    const backendPreviewStrip = document.createElement("div");
    backendPreviewStrip.className = "iamccs-v2v-backend-preview-strip";
    resultCard.append(resultBox, resultActions);
    mediaStack.append(sourceCard, imageCard, resultCard);
    mediaBody.appendChild(mediaStack);
    mediaPanel.appendChild(mediaBody);
    main.appendChild(mediaPanel);

    const mediaModeBadge = mediaPanel.querySelector(".iamccs-v2v-media-mode-badge");
    const mediaSourceTitle = sourceCard.querySelector(".iamccs-v2v-source-title");
    const mediaSourceSubtitle = sourceCard.querySelector(".iamccs-v2v-source-subtitle");
    const mediaImageTitle = imageCard.querySelector(".iamccs-v2v-image-title");
    const mediaImageSubtitle = imageCard.querySelector(".iamccs-v2v-image-subtitle");
    const mediaResultTitle = resultCard.querySelector(".iamccs-v2v-result-title");
    const mediaResultSubtitle = resultCard.querySelector(".iamccs-v2v-result-subtitle");
    const mediaLabels = {
        ltx_simple: {
            badge: "LTX source + reference",
            source: "Source video",
            sourceSub: "timeline / audio driver",
            image: "Reference / pose image",
            imageSub: "first frame / DW Pose",
            result: "LTX generated video",
            resultSub: "selected output stage",
        },
        scail2: {
            badge: "SCAIL driver + identity",
            source: "Driver video",
            sourceSub: "pose / SAM track source",
            image: "Identity reference",
            imageSub: "single or multi identity",
            result: "SCAIL generated video",
            resultSub: "16 FPS / 32 FPS / preview",
        },
        wananimate: {
            badge: "WanAnimate driver + character",
            source: "Driver video",
            sourceSub: "face / pose / background source",
            image: "Character reference",
            imageSub: "identity / background lock",
            result: "WanAnimate generated video",
            resultSub: "selected output stage",
        },
        pose_transfer: {
            badge: "Pose Transfer canvas",
            source: "Pose driver video",
            sourceSub: "motion source",
            image: "Character reference image",
            imageSub: "image to animate",
            result: "Pose transfer result image",
            resultSub: "reference handoff / export",
        },
    };
    function refreshMediaDock(mode = activeBackendMode()) {
        const labels = mediaLabels[mode] || mediaLabels.ltx_simple;
        if (mediaModeBadge) mediaModeBadge.textContent = labels.badge;
        if (mediaSourceTitle) mediaSourceTitle.textContent = labels.source;
        if (mediaSourceSubtitle) mediaSourceSubtitle.textContent = labels.sourceSub;
        if (mediaImageTitle) mediaImageTitle.textContent = labels.image;
        if (mediaImageSubtitle) mediaImageSubtitle.textContent = labels.imageSub;
        if (mediaResultTitle) mediaResultTitle.textContent = labels.result;
        if (mediaResultSubtitle) mediaResultSubtitle.textContent = labels.resultSub;
        resultActions.classList.toggle("is-hidden", mode !== "pose_transfer");
    }

    const timelinePanel = document.createElement("section");
    timelinePanel.className = "iamccs-v2v-panel timeline";
    timelinePanel.innerHTML = `<div class="iamccs-v2v-panel-head"><span>Video Timeline</span><span>drag I/O on source</span></div>`;
    const timelineBody = document.createElement("div");
    timelineBody.className = "iamccs-v2v-panel-body";
    const middle = document.createElement("div");
    middle.className = "iamccs-v2v-mid";
    const prompt = document.createElement("textarea");
    prompt.value = read(node, "global_prompt", "");
    const negative = document.createElement("textarea");
    negative.value = read(node, "negative_prompt", "");
    const timelineCard = document.createElement("section");
    timelineCard.className = "iamccs-v2v-timeline-card";
    timelineCard.innerHTML = `<div class="iamccs-v2v-timeline-head"><span>Source video range</span><span>1:1 trim timeline</span></div>`;
    const timelineBox = document.createElement("div");
    timelineBox.className = "iamccs-v2v-timebar";
    const scrubWrap = document.createElement("div");
    scrubWrap.className = "iamccs-v2v-scrub-wrap";
    const scrubRow = document.createElement("div");
    scrubRow.className = "iamccs-v2v-scrub-row";
    const scrubStart = document.createElement("span");
    const scrubEnd = document.createElement("span");
    const scrub = document.createElement("input");
    scrub.type = "range";
    scrub.className = "iamccs-v2v-scrub";
    scrub.min = "0";
    scrub.max = "1000";
    scrub.step = "1";
    scrub.value = "0";
    scrubRow.append(scrubStart, scrub, scrubEnd);
    scrubWrap.appendChild(scrubRow);
    timelineCard.append(timelineBox, scrubWrap);
    const readout = document.createElement("div");
    readout.className = "iamccs-v2v-readout";
    middle.append(timelineCard, readout);
    timelineBody.appendChild(middle);
    timelinePanel.appendChild(timelineBody);
    main.appendChild(timelinePanel);

    const optionsPanel = document.createElement("section");
    optionsPanel.className = "iamccs-v2v-settings-drawer controls";
    const settingsHead = document.createElement("div");
    settingsHead.className = "iamccs-v2v-settings-head";
    settingsHead.innerHTML = `<span>V2V Settings <small>active mode controls</small></span>`;
    const settingsCloseBtn = button("Close Settings");
    settingsCloseBtn.classList.add("iamccs-v2v-settings-close");
    settingsHead.appendChild(settingsCloseBtn);
    optionsPanel.appendChild(settingsHead);
    const optionsBody = document.createElement("div");
    optionsBody.className = "iamccs-v2v-panel-body";
    const normalBtn = button("Normal VRAM");
    const lowBtn = button("Low VRAM");
    const dwposeBtn = button("DW Pose");
    const ltxBtn = button(BACKEND_MODES.ltx_simple.label);
    const scailBtn = button(BACKEND_MODES.scail2.label);
    const wanBtn = button(BACKEND_MODES.wananimate.label);
    const poseTransferBtn = button(BACKEND_MODES.pose_transfer.label);
    const backendButtons = {
        ltx_simple: ltxBtn,
        scail2: scailBtn,
        wananimate: wanBtn,
        pose_transfer: poseTransferBtn,
    };

    const duration = input("number", read(node, "duration_seconds", 10));
    duration.step = "0.01";
    const fps = input("number", read(node, "fps", 24));
    fps.step = "0.01";
    const frameCap = input("number", read(node, "frame_load_cap", 241));
    frameCap.step = "1";
    const widthInput = input("number", read(node, "generation_width", 1280));
    widthInput.step = "8";
    const heightInput = input("number", read(node, "generation_height", 720));
    heightInput.step = "8";
    const segment = input("number", read(node, "segment_seconds", 10));
    segment.step = "0.01";
    const overlap = input("number", read(node, "overlap_frames", 9));
    overlap.step = "1";
    const preset = select(["5sec", "10sec", "15sec", "20sec", "videoclip", "monologue"], read(node, "segment_preset", "5sec"));
    const planning = select(["manual_segment_seconds", "explicit_preset_seconds"], read(node, "planning_mode", "explicit_preset_seconds"));
    const round = select(["up", "nearest", "down"], read(node, "ltx_round_mode", "up"));
    const audioVae = input("text", read(node, "audio_vae_name", "ltx-2.3-22b-dev_audio_vae.safetensors"));
    const audioDevice = select(["main_device", "cpu"], read(node, "audio_vae_device", "main_device"));
    const audioDtype = select(["bf16", "fp16", "fp32"], read(node, "audio_vae_dtype", "bf16"));
    const pose = select(["none", "dwpose_openpose", "source_pose_only", "image_pose_transfer"], read(node, "pose_mode", "dwpose_openpose"));
    const strength = input("number", read(node, "dwpose_strength", 0.75));
    strength.step = "0.01";
    const outputPrefix = input("text", read(node, "output_prefix", "IAMCCS/LTX23_V2V_SHOTBOARD"));
    const outputStage = select(["final", "draft", "sam_preview", "mask_preview", "pose_preview", "reference_preview", "result_image"], read(node, "output_stage", "final"));
    const previewStage = select(["final", "draft", "sam_preview", "mask_preview", "pose_preview", "reference_preview", "result_image", "source"], read(node, "preview_stage", "final"));
    const backendVariant = select(["ltx_simple", "scail2_single_person", "scail2_multi_person_identity", "wananimate_bg_locked", "flux_klein_pose_transfer"], read(node, "backend_variant", "ltx_simple"));
    const scailIdentity = select(["single_person", "multi_person_identity"], read(node, "scail_identity_mode", "single_person"));
    const scailOutputStage = select(["final_32fps_upscaled", "generated_16fps", "both"], read(node, "scail_output_stage", "final_32fps_upscaled"));
    const sam31PreviewBtn = button("SAM 3.1 Preview");
    const wanBackgroundLockBtn = button("BG Locked");
    const wanMaskMode = select(["sam31_identity_mask", "uploaded_character_mask", "none"], read(node, "wan_character_mask_mode", "sam31_identity_mask"));
    const wanControlPreviewBtn = button("Control Preview");
    const poseResultMode = select(["preview_only", "use_as_reference", "export_result"], read(node, "pose_transfer_result_mode", "use_as_reference"));
    const poseImagePath = input("text", read(node, "pose_transfer_image_path", read(node, "source_image_path", "")));
    const poseVideoPath = input("text", read(node, "pose_transfer_video_path", read(node, "source_video_path", "")));
    const poseResultPath = input("text", read(node, "pose_transfer_result_path", ""));
    const poseUploadImage = button("Upload Image", "good");
    const poseUploadVideo = button("Upload Video", "good");
    const poseUseSource = button("Use Source");
    const poseUseReference = button("Use Reference");
    const mapBackendBtn = button("Map Backend", "good");
    const refreshBackendPreviewsBtn = button("Refresh Previews");

    const backendModeRow = document.createElement("div");
    backendModeRow.className = "iamccs-v2v-mode-grid backend";
    backendModeRow.append(ltxBtn, scailBtn, wanBtn, poseTransferBtn);
    const backendHint = document.createElement("div");
    backendHint.className = "iamccs-v2v-mode-hint";
    const channelRow = document.createElement("div");
    channelRow.className = "iamccs-v2v-channel-grid";
    const samChannel = Object.assign(document.createElement("div"), { className: "iamccs-v2v-channel", textContent: "SAM 3.1" });
    const controlChannel = Object.assign(document.createElement("div"), { className: "iamccs-v2v-channel", textContent: "Control" });
    const poseChannel = Object.assign(document.createElement("div"), { className: "iamccs-v2v-channel", textContent: "Pose" });
    channelRow.append(samChannel, controlChannel, poseChannel);
    const modeRow = document.createElement("div");
    modeRow.className = "iamccs-v2v-mode-grid profile";
    modeRow.append(normalBtn, lowBtn, dwposeBtn);
    const editorDrawer = document.createElement("div");
    editorDrawer.className = "iamccs-v2v-editor-drawer";
    editorDrawer.textContent = "Timeline editor ready: source-video strip, trim handles, realtime scrub and segment lane. Backend modes write explicit planner metadata for LTX, SCAIL-2, WanAnimate and Pose Transfer.";

    const modePanel = document.createElement("div");
    modePanel.className = "iamccs-v2v-mode-panel";
    const ltxSection = document.createElement("section");
    ltxSection.className = "iamccs-v2v-mode-section";
    ltxSection.dataset.mode = "ltx_simple";
    ltxSection.innerHTML = `<div class="iamccs-v2v-mode-title"><span>LTX Simple</span><span>audio guard</span></div>`;
    ltxSection.appendChild(Object.assign(document.createElement("div"), {
        className: "iamccs-v2v-mini-grid",
        innerHTML: "",
    }));
    ltxSection.querySelector(".iamccs-v2v-mini-grid").append(
        miniSlot("Source V2V", "Timeline trim, frame cap, LTX rounding"),
        miniSlot("DW Pose", "Optional pose guide from source"),
        miniSlot("Audio VAE", "KJ/LTX audio loader metadata"),
        miniSlot("Output", "LTX23 V2V shotboard prefix")
    );

    const scailSection = document.createElement("section");
    scailSection.className = "iamccs-v2v-mode-section is-hidden";
    scailSection.dataset.mode = "scail2";
    scailSection.innerHTML = `<div class="iamccs-v2v-mode-title"><span>SCAIL-2 Backend</span><span>SAM masks</span></div>`;
    scailSection.append(
        field("identity", scailIdentity),
        field("output", scailOutputStage),
        Object.assign(document.createElement("div"), { className: "iamccs-v2v-toggle-row" })
    );
    scailSection.querySelector(".iamccs-v2v-toggle-row").append(sam31PreviewBtn, miniSlot("Core", "IAMCCS_ScailExtends"));
    const scailGrid = document.createElement("div");
    scailGrid.className = "iamccs-v2v-mini-grid";
    scailGrid.append(
        miniSlot("Single", "reference replacement"),
        miniSlot("Multi", "identity tracker + colored masks"),
        miniSlot("Preview", "SAM3_TrackPreview"),
        miniSlot("Outputs", "16 FPS and 32 FPS upscaled")
    );
    scailSection.appendChild(scailGrid);

    const wanSection = document.createElement("section");
    wanSection.className = "iamccs-v2v-mode-section is-hidden";
    wanSection.dataset.mode = "wananimate";
    wanSection.innerHTML = `<div class="iamccs-v2v-mode-title"><span>WanAnimate Backend</span><span>bg lock</span></div>`;
    const wanToggles = document.createElement("div");
    wanToggles.className = "iamccs-v2v-toggle-row";
    wanToggles.append(wanBackgroundLockBtn, wanControlPreviewBtn);
    const wanGrid = document.createElement("div");
    wanGrid.className = "iamccs-v2v-mini-grid";
    wanGrid.append(
        miniSlot("Face/Pose", "PoseAndFaceDetection + ViTPose"),
        miniSlot("Mask", "SAM3 track to character mask"),
        miniSlot("Background", "DrawMaskOnImage background video"),
        miniSlot("Core", "IAMCCS_WanAnimateExtends")
    );
    wanSection.append(wanToggles, field("mask", wanMaskMode), wanGrid);

    const poseTransferSection = document.createElement("section");
    poseTransferSection.className = "iamccs-v2v-mode-section is-hidden";
    poseTransferSection.dataset.mode = "pose_transfer";
    poseTransferSection.innerHTML = `<div class="iamccs-v2v-mode-title"><span>Pose Transfer Canvas</span><span>image + video + result</span></div>`;
    const poseMediaNote = document.createElement("div");
    poseMediaNote.className = "iamccs-v2v-mode-hint";
    poseMediaNote.textContent = "Driver video, character image and result upload are managed in the three Media panels on the left.";
    poseTransferSection.append(
        poseMediaNote,
        miniSlot("Core", "FLUX Klein pose transfer"),
        field("result mode", poseResultMode)
    );
    modePanel.append(ltxSection, scailSection, wanSection, poseTransferSection);
    const backendActionRow = document.createElement("div");
    backendActionRow.className = "iamccs-v2v-toggle-row";
    backendActionRow.append(mapBackendBtn, refreshBackendPreviewsBtn);

    optionsBody.append(
        controlGroup("Backend mode", "edition", [backendModeRow, backendHint, channelRow], "run"),
        controlGroup("Mode backend", "workflow", [modePanel, backendActionRow], "run"),
        controlGroup("Run mode", "profile", [modeRow, editorDrawer], "run"),
        controlGroup("Prompts", "conditioning", [
            field("positive", prompt),
            field("negative", negative),
        ], "prompts"),
        controlGroup("Timing", "timeline", [
            field("duration", duration),
            field("fps", fps),
            field("frame cap", frameCap),
            field("segment s", segment),
            field("overlap f", overlap),
            field("preset", preset),
            field("planning", planning),
            field("ltx round", round),
        ], "timing"),
        controlGroup("Frame", "resolution", [
            field("width", widthInput),
            field("height", heightInput),
        ], "frame"),
        controlGroup("Pose", "control", [
            field("pose", pose),
            field("pose str", strength),
        ], "pose"),
        controlGroup("Audio + output", "backend", [
            field("audio vae", audioVae),
            field("device", audioDevice),
            field("dtype", audioDtype),
            field("output", outputPrefix),
            field("save", outputStage),
            field("preview", previewStage),
        ], "audio"),
        Object.assign(document.createElement("div"), { className: "iamccs-v2v-note", textContent: "Mode metadata is now carried by the planner. SCAIL-2, WanAnimate and Pose Transfer are prepared for their backend bridge/workflow wiring." })
    );
    optionsPanel.appendChild(optionsBody);

    const backendPreviewPanel = document.createElement("section");
    backendPreviewPanel.className = "iamccs-v2v-panel preview iamccs-v2v-backend-preview-panel";
    backendPreviewPanel.innerHTML = `<div class="iamccs-v2v-panel-head"><span>Mode Preview</span><span class="iamccs-v2v-preview-mode-badge">active backend</span></div>`;
    const backendPreviewBody = document.createElement("div");
    backendPreviewBody.className = "iamccs-v2v-panel-body";
    const backendPreviewStatus = document.createElement("div");
    backendPreviewStatus.className = "iamccs-v2v-mode-hint";
    const backendPreviewModeBadge = backendPreviewPanel.querySelector(".iamccs-v2v-preview-mode-badge");
    backendPreviewBody.append(backendPreviewStatus, backendPreviewStrip);
    backendPreviewPanel.appendChild(backendPreviewBody);
    main.appendChild(backendPreviewPanel);
    root.appendChild(optionsPanel);

    function numberValue(el, fallback) {
        const n = Number(el.value);
        return Number.isFinite(n) ? n : fallback;
    }

    function boolWidget(name, fallback) {
        const value = read(node, name, fallback);
        if (typeof value === "boolean") return value;
        return String(value).toLowerCase() === "true" || String(value) === "1";
    }

    function activeBackendMode() {
        return resolveBackendMode(node);
    }

    function boolControl(name, fallback) {
        return boolWidget(name, fallback);
    }

    function selectedBackendProfile() {
        const mode = activeBackendMode();
        if (mode === "scail2") {
            return scailIdentity.value === "multi_person_identity" ? "scail2_multi_person_identity" : "scail2_single_person";
        }
        if (mode === "wananimate") return "wananimate_extension";
        if (mode === "pose_transfer") return "flux_klein_pose_transfer";
        return "ltx23_v2v_infinite_lipsync";
    }

    function selectedBackendVariant() {
        const mode = activeBackendMode();
        if (mode === "scail2") return selectedBackendProfile();
        if (mode === "wananimate") return boolControl("wan_background_lock", true) ? "wananimate_bg_locked" : "wananimate_extension";
        if (mode === "pose_transfer") return "flux_klein_pose_transfer";
        return "ltx_simple";
    }

    function setActiveButtons() {
        const vram = String(read(node, "vram_profile", "normal_vram"));
        const mode = activeBackendMode();
        Object.entries(backendButtons).forEach(([key, btn]) => btn.classList.toggle("is-active", key === mode));
        backendHint.textContent = BACKEND_MODES[mode]?.hint || BACKEND_MODES.ltx_simple.hint;
        samChannel.classList.toggle("is-active", boolControl("enable_sam31_preview", true) && (mode === "scail2" || mode === "wananimate"));
        controlChannel.classList.toggle("is-active", (mode === "wananimate" && boolControl("wan_control_preview", true)) || mode === "pose_transfer");
        poseChannel.classList.toggle("is-active", mode !== "ltx_simple");
        normalBtn.classList.toggle("is-active", vram === "normal_vram");
        lowBtn.classList.toggle("is-active", vram === "low_vram");
        dwposeBtn.classList.toggle("is-active", boolWidget("dwpose_enabled", true));
        sam31PreviewBtn.classList.toggle("is-active", boolControl("enable_sam31_preview", true));
        wanBackgroundLockBtn.classList.toggle("is-active", boolControl("wan_background_lock", true));
        wanControlPreviewBtn.classList.toggle("is-active", boolControl("wan_control_preview", true));
        backendVariant.value = selectedBackendVariant();
        refreshMediaDock(mode);
        modePanel.querySelectorAll(".iamccs-v2v-mode-section").forEach((section) => {
            section.classList.toggle("is-hidden", section.dataset.mode !== mode);
        });
        renderBackendPreviews();
    }

    function trimValues() {
        const dur = Math.max(0.01, numberValue(duration, 10));
        const start = Math.max(0, Number(read(node, "trim_start_s", 0)) || 0);
        const endRaw = Number(read(node, "trim_end_s", 0)) || dur;
        const end = Math.max(start + 0.01, Math.min(dur, endRaw));
        return { dur, start, end };
    }

    function timelinePayload() {
        const { dur, start, end } = trimValues();
        return {
            schema: "iamccs.v2v.shotboard.timeline",
            schema_version: 4,
            backend_mode: activeBackendMode(),
            backend_family: BACKEND_MODES[activeBackendMode()]?.family || "ltx",
            backend_profile: selectedBackendProfile(),
            backend_variant: selectedBackendVariant(),
            scail_identity_mode: scailIdentity.value,
            scail_output_stage: scailOutputStage.value,
            enable_sam31_preview: boolControl("enable_sam31_preview", true),
            wan_background_lock: boolControl("wan_background_lock", true),
            wan_character_mask_mode: wanMaskMode.value,
            wan_control_preview: boolControl("wan_control_preview", true),
            pose_transfer_image_path: poseImagePath.value,
            pose_transfer_video_path: poseVideoPath.value,
            pose_transfer_result_path: poseResultPath.value,
            pose_transfer_result_mode: poseResultMode.value,
            output_stage: outputStage.value,
            preview_stage: previewStage.value,
            preview_channels: {
                taeltx: false,
                source: true,
                pose: pose.value !== "none",
                sam31: boolControl("enable_sam31_preview", true) && (activeBackendMode() === "scail2" || activeBackendMode() === "wananimate"),
                controlnet: (activeBackendMode() === "wananimate" && boolControl("wan_control_preview", true)) || activeBackendMode() === "pose_transfer",
                result: activeBackendMode() === "pose_transfer",
            },
            source_video_path: videoPath.value,
            source_image_path: imagePath.value,
            duration_seconds: end - start,
            source_duration_seconds: dur,
            fps: numberValue(fps, 24),
            trim_start_s: start,
            trim_end_s: end,
            frame_load_cap: Math.max(1, Math.round(numberValue(frameCap, 241))),
            generation_width: Math.max(64, Math.round(numberValue(widthInput, 1280))),
            generation_height: Math.max(64, Math.round(numberValue(heightInput, 720))),
            segment_seconds: Math.max(0.01, numberValue(segment, 10)),
            overlap_frames: Math.max(0, Math.round(numberValue(overlap, 9))),
            segment_preset: preset.value,
            planning_mode: planning.value,
            ltx_round_mode: round.value,
            vram_profile: String(read(node, "vram_profile", "normal_vram")),
            audio_vae_name: audioVae.value,
            audio_vae_device: audioDevice.value,
            audio_vae_dtype: audioDtype.value,
            pose_mode: pose.value,
            dwpose_enabled: boolWidget("dwpose_enabled", true),
            dwpose_strength: Math.max(0, numberValue(strength, 0.75)),
            global_prompt: prompt.value,
            negative_prompt: negative.value,
        };
    }

    function autoCapFromDuration() {
        const dur = Math.max(0.01, numberValue(duration, 10));
        const rate = Math.max(1, numberValue(fps, 24));
        frameCap.value = String(Math.max(1, Math.round(dur * rate) + 1));
    }

    function commit() {
        const effectiveVideo = String(videoPath.value || "").trim() || "IMG_4145 2.mp4";
        const effectiveImage = String(imagePath.value || "").trim() || "QWEN2509_FIRST_FRAME_DWPOSE_OPENPOSE_CONTROL_00001_.png";
        videoPath.value = effectiveVideo;
        imagePath.value = effectiveImage;
        write(node, "source_video_path", effectiveVideo);
        write(node, "source_image_path", effectiveImage);
        write(node, "duration_seconds", Math.max(0.01, numberValue(duration, 10)));
        write(node, "fps", Math.max(1, numberValue(fps, 24)));
        write(node, "frame_load_cap", Math.max(1, Math.round(numberValue(frameCap, 241))));
        write(node, "generation_width", Math.max(64, Math.round(numberValue(widthInput, 1280))));
        write(node, "generation_height", Math.max(64, Math.round(numberValue(heightInput, 720))));
        write(node, "segment_seconds", Math.max(0.01, numberValue(segment, 10)));
        write(node, "planning_mode", planning.value);
        write(node, "segment_preset", preset.value);
        write(node, "overlap_frames", Math.max(0, Math.round(numberValue(overlap, 9))));
        write(node, "ltx_round_mode", round.value);
        write(node, "backend_mode", activeBackendMode());
        write(node, "backend_family", BACKEND_MODES[activeBackendMode()]?.family || "ltx");
        write(node, "backend_profile", selectedBackendProfile());
        write(node, "backend_variant", selectedBackendVariant());
        write(node, "scail_identity_mode", scailIdentity.value);
        write(node, "scail_output_stage", scailOutputStage.value);
        write(node, "enable_sam31_preview", boolControl("enable_sam31_preview", true));
        write(node, "wan_background_lock", boolControl("wan_background_lock", true));
        write(node, "wan_character_mask_mode", wanMaskMode.value);
        write(node, "wan_control_preview", boolControl("wan_control_preview", true));
        write(node, "pose_transfer_image_path", poseImagePath.value);
        write(node, "pose_transfer_video_path", poseVideoPath.value);
        write(node, "pose_transfer_result_path", poseResultPath.value);
        write(node, "pose_transfer_result_mode", poseResultMode.value);
        write(node, "output_stage", outputStage.value);
        write(node, "preview_stage", previewStage.value);
        write(node, "audio_vae_name", audioVae.value);
        write(node, "audio_vae_device", audioDevice.value);
        write(node, "audio_vae_dtype", audioDtype.value);
        write(node, "pose_mode", pose.value);
        write(node, "dwpose_strength", Math.max(0, numberValue(strength, 0.75)));
        write(node, "taeltx_preview_enabled", false);
        write(node, "global_prompt", prompt.value);
        write(node, "negative_prompt", negative.value);
        write(node, "output_prefix", outputPrefix.value);
        write(node, "ui_video_height", Math.round(state.videoHeight));
        const payload = timelinePayload();
        write(node, "timeline_data", JSON.stringify(payload, null, 2));
        const touched = syncBackendGraphWidgets(node, payload);
        if (touched) status.textContent = `backend mapped: ${touched} widget values synced`;
        setActiveButtons();
        drawTimeline();
        renderBackendPreviews();
    }

    function drawPreview() {
        const video = String(videoPath.value || "").trim();
        const image = String(imagePath.value || "").trim();
        const result = String(poseResultPath?.value || "").trim();
        previewBox.innerHTML = "";
        imageBox.innerHTML = "";
        resultBox.innerHTML = "";
        if (state.videoObjectUrl || video) {
            const monitor = document.createElement("video");
            monitor.src = state.videoObjectUrl || viewUrl(video, "input");
            monitor.muted = true;
            monitor.playsInline = true;
            monitor.preload = "metadata";
            monitor.controls = false;
            previewBox.appendChild(monitor);
            try { monitor.currentTime = Math.max(0, Number(state.playheadSec || 0)); } catch {}
        } else {
            const preview = document.createElement("div");
            preview.className = "iamccs-v2v-preview-placeholder";
            preview.innerHTML = `Source Monitor<span>drop or upload a video to inspect the source range</span>`;
            previewBox.appendChild(preview);
        }
        if (state.imageObjectUrl || image) {
            const el = document.createElement("img");
            el.src = state.imageObjectUrl || viewUrl(image, "input");
            imageBox.appendChild(el);
        } else {
            imageBox.textContent = "Drop or upload reference / first-frame image";
        }
        if (result) {
            const lower = result.toLowerCase();
            if (/\.(mp4|webm|mov|mkv)$/i.test(lower)) {
                const el = document.createElement("video");
                el.src = viewUrl(result, "output");
                el.muted = true;
                el.playsInline = true;
                el.loop = true;
                el.preload = "metadata";
                resultBox.appendChild(el);
                resultBox.onmouseenter = () => el.play?.().catch?.(() => {});
                resultBox.onmouseleave = () => el.pause?.();
            } else {
                const el = document.createElement("img");
                el.src = viewUrl(result, "output");
                resultBox.appendChild(el);
            }
        } else {
            const best = backendPreviewItems()[0];
            if (best?.url) {
                appendPreviewMedia(resultBox, best, true);
            } else {
                resultBox.textContent = "Final image / video preview";
            }
        }
        renderBackendPreviews();
        status.textContent = `${video || "no video"} | ${image || "no image"}`;
    }

    function setVideoCurrentTime(seconds) {
        if (!Number.isFinite(seconds)) return;
        state.pendingSeek = Math.max(0, seconds);
        const el = state.timelineVideo;
        if (!el) return;
        const applySeek = () => {
            state.seekRaf = 0;
            try {
                el.pause?.();
                const next = Math.max(0, state.pendingSeek);
                if (Math.abs(Number(el.currentTime || 0) - next) > 0.015) el.currentTime = next;
            } catch {}
        };
        if (el.readyState >= 1) {
            if (!state.seekRaf) state.seekRaf = requestAnimationFrame(applySeek);
        } else {
            el.addEventListener("loadedmetadata", applySeek, { once: true });
        }
    }

    function drawCurrentFrameCanvas() {
        const el = state.timelineVideo;
        const canvas = timelineBox.querySelector(".iamccs-v2v-frame-bubble canvas");
        if (!el || !canvas || el.readyState < 2) return;
        try {
            const ctx = canvas.getContext("2d");
            canvas.width = 236;
            canvas.height = 116;
            ctx.drawImage(el, 0, 0, canvas.width, canvas.height);
        } catch {}
    }

    function timelineVideoUrl() {
        const video = String(videoPath.value || "").trim();
        return state.videoObjectUrl || (video ? viewUrl(video, "input") : "");
    }

    function previewParamsForNode(item) {
        const candidates = [
            widgetValue(item, "videopreview"),
            widgetValue(item, "preview"),
            item?.imgs?.[0],
            item?.image,
            item?.image_url,
        ];
        for (const candidate of candidates) {
            const params = candidate?.params || candidate;
            if (params?.filename) return params;
        }
        return null;
    }

    function backendPreviewItems() {
        const mode = activeBackendMode();
        const items = [];
        const supportedTypes = new Set(["VHS_VideoCombine", "PreviewAnimation", "PreviewImage", "SaveImage", "SaveImageKJ"]);
        for (const item of graphNodes()) {
            if (item === node || !supportedTypes.has(nodeName(item))) continue;
            const title = String(item.title || "");
            const prefix = String(widgetValue(item, "filename_prefix") || "");
            const haystack = `${title} ${prefix}`.toUpperCase();
            const params = previewParamsForNode(item);
            const url = viewUrlFromPreviewParams(params);
            if (!url) continue;
            const unifiedOutput = haystack.includes("IAMCCS V2V FINAL OUTPUT") || haystack.includes("IAMCCS V2V PREVIEW OUTPUT");
            const matches = unifiedOutput ||
                (mode === "scail2" && (haystack.includes("SCAIL") || haystack.includes("SAM") || haystack.includes("POSE VIDEO"))) ||
                (mode === "wananimate" && (haystack.includes("WAN") || haystack.includes("SAM3") || haystack.includes("BLOCKIFY") || haystack.includes("POSE"))) ||
                (mode === "pose_transfer" && (haystack.includes("POSE") || haystack.includes("FLUX"))) ||
                (mode === "ltx_simple" && (haystack.includes("LTX") || haystack.includes("SHOTBOARD")));
            if (!matches) continue;
            items.push({
                title: title || prefix || "preview",
                url,
                kind: nodeName(item) === "PreviewImage" || nodeName(item) === "SaveImage" || nodeName(item) === "SaveImageKJ" ? "image" : "video",
                type: params?.type || "output",
            });
        }
        items.sort((a, b) => {
            const rank = (x) => /FINAL|32FPS|OUTPUT/.test(x.title.toUpperCase()) ? 0 : /SAM|MASK|POSE|REFERENCE/.test(x.title.toUpperCase()) ? 1 : 2;
            return rank(a) - rank(b);
        });
        return items.slice(0, 3);
    }

    function appendPreviewMedia(container, item, hoverPlay = true) {
        container.innerHTML = "";
        if (!item?.url) return;
        const media = document.createElement(item.kind === "image" ? "img" : "video");
        media.src = item.url;
        media.title = item.title;
        if (media.tagName === "VIDEO") {
            media.muted = true;
            media.playsInline = true;
            media.loop = true;
            media.preload = "metadata";
            if (hoverPlay) {
                container.onmouseenter = () => media.play?.().catch?.(() => {});
                container.onmouseleave = () => media.pause?.();
            }
        }
        container.appendChild(media);
    }

    function renderBackendPreviews() {
        const mode = activeBackendMode();
        const items = backendPreviewItems();
        const labels = {
            ltx_simple: ["Final video", "Draft / source", "Pose / reference"],
            scail2: ["SCAIL final", "SAM 3.1 preview", "Mask / pose"],
            wananimate: ["WanAnimate final", "SAM 3.1 preview", "Control / pose"],
            pose_transfer: ["Result image", "Driver pose", "Reference"],
        }[mode] || ["Final", "Preview", "Control"];
        const modeLabel = BACKEND_MODES[mode]?.label || mode;
        if (backendPreviewModeBadge) backendPreviewModeBadge.textContent = modeLabel;
        backendPreviewStatus.textContent = `${modeLabel} · previews follow the selected mode`;
        backendPreviewStrip.innerHTML = "";
        for (let i = 0; i < 3; i++) {
            const cell = document.createElement("div");
            cell.className = "iamccs-v2v-backend-preview";
            const slotHead = document.createElement("div");
            slotHead.className = "iamccs-v2v-preview-slot-head";
            slotHead.textContent = labels[i];
            const slotMedia = document.createElement("div");
            slotMedia.className = "iamccs-v2v-preview-slot-media";
            const item = items[i];
            if (item?.url) {
                appendPreviewMedia(slotMedia, item);
            } else {
                slotMedia.innerHTML = `<div class="iamccs-v2v-preview-placeholder"><b>${labels[i]}</b><span>waiting for backend preview</span></div>`;
            }
            cell.append(slotHead, slotMedia);
            backendPreviewStrip.appendChild(cell);
        }
    }

    function ensureTimelineVideo(media) {
        const src = timelineVideoUrl();
        if (!src) {
            state.timelineVideo = null;
            state.timelineVideoSrc = "";
            const empty = document.createElement("div");
            empty.className = "iamccs-v2v-timeline-media-empty";
            empty.textContent = "source video";
            media.appendChild(empty);
            return;
        }
        let el = state.timelineVideo;
        if (!el || state.timelineVideoSrc !== src) {
            el = document.createElement("video");
            el.muted = true;
            el.playsInline = true;
            el.preload = "auto";
            el.controls = false;
            el.disablePictureInPicture = true;
            el.src = src;
            state.timelineVideo = el;
            state.timelineVideoSrc = src;
            state.autoDurationSyncedForSrc = "";
            el.addEventListener("loadedmetadata", () => {
                const metaDuration = Number(el.duration || 0);
                if (Number.isFinite(metaDuration) && metaDuration > 0 && state.autoDurationSyncedForSrc !== src) {
                    state.autoDurationSyncedForSrc = src;
                    if (Math.abs(metaDuration - numberValue(duration, 10)) > 0.05) {
                        duration.value = metaDuration.toFixed(2);
                        write(node, "trim_end_s", metaDuration);
                        autoCapFromDuration();
                        commit();
                        return;
                    }
                }
                setVideoCurrentTime(state.playheadSec);
            });
            el.addEventListener("seeked", () => {
                updatePlayheadVisual(false);
                drawCurrentFrameCanvas();
            });
            el.addEventListener("loadeddata", drawCurrentFrameCanvas);
        }
        media.appendChild(el);
        setVideoCurrentTime(state.playheadSec);
    }

    function drawTimeline() {
        const data = timelinePayload();
        const dur = Math.max(0.01, Number(data.source_duration_seconds || data.duration_seconds || 10));
        const start = Math.max(0, Number(data.trim_start_s || 0));
        const end = Math.max(start + 0.01, Math.min(dur, Number(data.trim_end_s || dur)));
        state.playheadSec = Math.max(start, Math.min(end, Number(state.playheadSec || start)));
        const left = (start / dur) * 100;
        const width = Math.max(0.5, ((end - start) / dur) * 100);
        const playLeft = (state.playheadSec / dur) * 100;
        const activeStart = state.activeTrim === "start";
        const activeEnd = state.activeTrim !== "start";
        const segmentSeconds = Math.max(0.01, Number(data.segment_seconds || 10));
        const segCount = Math.max(1, Math.ceil((end - start) / segmentSeconds));
        const ticks = [];
        const tickCount = Math.min(8, Math.max(2, Math.ceil(dur)));
        for (let i = 0; i <= tickCount; i++) {
            const pct = (i / tickCount) * 100;
            const sec = (dur * i) / tickCount;
            ticks.push(`<div class="iamccs-v2v-tick" style="left:${pct}%"><span>${sec.toFixed(sec >= 10 ? 0 : 1)}s</span></div>`);
        }
        timelineBox.innerHTML = `
            <div class="iamccs-v2v-ruler">${ticks.join("")}</div>
            <div class="iamccs-v2v-track">
                <div class="iamccs-v2v-timeline-media"></div>
                <div class="iamccs-v2v-timeline-shade" style="left:0;width:${left}%"></div>
                <div class="iamccs-v2v-timeline-shade" style="left:${left + width}%;right:0"></div>
                <div class="iamccs-v2v-selected" style="left:${left}%;width:${width}%"></div>
                <div class="iamccs-v2v-handle ${activeStart ? "is-active" : ""}" data-handle="start" style="left:${left}%">I</div>
                <div class="iamccs-v2v-handle ${activeEnd ? "is-active" : ""}" data-handle="end" style="left:${left + width}%">O</div>
            </div>
            <div class="iamccs-v2v-segment-head"><span>Segments / Chunks</span><span>${segCount} planned</span></div>
            <div class="iamccs-v2v-segments"></div>
            <div class="iamccs-v2v-playbar">
                <div class="iamccs-v2v-transfer-controls">
                    <button type="button" class="iamccs-v2v-stepbtn" data-step="-1">-</button>
                    <button type="button" class="iamccs-v2v-playbtn">Play</button>
                    <button type="button" class="iamccs-v2v-stepbtn" data-step="1">+</button>
                </div>
                <div class="iamccs-v2v-analog"><div class="iamccs-v2v-analog-thumb" style="left:${playLeft}%"></div></div>
                <div class="iamccs-v2v-time-label">${state.playheadSec.toFixed(2)}s</div>
            </div>
        `;
        const media = timelineBox.querySelector(".iamccs-v2v-timeline-media");
        ensureTimelineVideo(media);
        const segs = timelineBox.querySelector(".iamccs-v2v-segments");
        for (let i = 0; i < segCount; i++) {
            const item = document.createElement("div");
            item.className = "iamccs-v2v-segment";
            item.textContent = `S${i + 1} · ${Number(data.segment_seconds).toFixed(2)}s`;
            const segmentStart = start + (i * segmentSeconds);
            const segmentEnd = Math.min(end, segmentStart + segmentSeconds);
            const meterLeft = Math.max(0, Math.min(100, (segmentStart / dur) * 100));
            const meterWidth = Math.max(1, Math.min(100 - meterLeft, ((segmentEnd - segmentStart) / dur) * 100));
            item.innerHTML = `<div class="iamccs-v2v-segment-code"><span>S${i + 1}</span><span>${segmentStart.toFixed(2)}-${segmentEnd.toFixed(2)}s</span></div><div class="iamccs-v2v-segment-meter"><i style="width:${meterWidth}%;margin-left:${meterLeft}%"></i></div>`;
            segs.appendChild(item);
        }
        scrubStart.textContent = `${start.toFixed(2)}s`;
        scrubEnd.textContent = `${end.toFixed(2)}s`;
        scrub.value = String(Math.round((state.playheadSec / dur) * 1000));
        readout.innerHTML = "";
        [
            ["Trim", `${start.toFixed(2)} - ${end.toFixed(2)}s`],
            ["Frames", `${data.frame_load_cap} @ ${Number(data.fps).toFixed(2)}`],
            ["Segments", `${segCount} x ${Number(data.segment_seconds).toFixed(2)}s`],
            ["Size", `${data.generation_width} x ${data.generation_height}`],
        ].forEach(([k, v]) => {
            const chip = document.createElement("div");
            chip.className = "iamccs-v2v-chip";
            chip.innerHTML = `<span>${k}</span><b>${v}</b>`;
            readout.appendChild(chip);
        });
        bindHandles();
        bindTrackScrub();
        bindPlaybarScrub();
    }

    function updatePlayheadVisual(seekVideo = true) {
        const { dur, start, end } = trimValues();
        state.playheadSec = Math.max(start, Math.min(end, Number(state.playheadSec || start)));
        const pct = (state.playheadSec / Math.max(0.01, dur)) * 100;
        timelineBox.querySelectorAll(".iamccs-v2v-frame-bubble").forEach((item) => { item.style.left = `${pct}%`; const label = item.querySelector("span"); if (label) label.textContent = `${state.playheadSec.toFixed(2)}s`; });
        timelineBox.querySelectorAll(".iamccs-v2v-analog-thumb").forEach((item) => { item.style.left = `${pct}%`; });
        timelineBox.querySelectorAll(".iamccs-v2v-time-label").forEach((item) => { item.textContent = `${state.playheadSec.toFixed(2)}s`; });
        scrub.value = String(Math.round((state.playheadSec / Math.max(0.01, dur)) * 1000));
        if (seekVideo) setVideoCurrentTime(state.playheadSec);
    }

    function bindHandles() {
        timelineBox.querySelectorAll(".iamccs-v2v-handle").forEach((handle) => {
            handle.onpointerdown = (event) => {
                event.preventDefault();
                event.stopPropagation();
                handle.setPointerCapture?.(event.pointerId);
                const kind = handle.dataset.handle;
                const track = timelineBox.querySelector(".iamccs-v2v-track");
                const rect = track.getBoundingClientRect();
                const dur = Math.max(0.01, numberValue(duration, 10));
                const move = (moveEvent) => {
                    const ratio = Math.max(0, Math.min(1, (moveEvent.clientX - rect.left) / Math.max(1, rect.width)));
                    const seconds = ratio * dur;
                    const start = Number(read(node, "trim_start_s", 0)) || 0;
                    const end = Number(read(node, "trim_end_s", dur)) || dur;
                    state.activeTrim = kind === "start" ? "start" : "end";
                    if (kind === "start") {
                        const next = Math.max(0, Math.min(seconds, end - 0.01));
                        write(node, "trim_start_s", next);
                        state.playheadSec = next;
                    } else {
                        const next = Math.max(start + 0.01, Math.min(dur, seconds));
                        write(node, "trim_end_s", next);
                        state.playheadSec = next;
                    }
                    drawTimeline();
                };
                const up = () => {
                    window.removeEventListener("pointermove", move);
                    window.removeEventListener("pointerup", up);
                    commit();
                };
                window.addEventListener("pointermove", move);
                window.addEventListener("pointerup", up);
            };
        });
    }

    function bindTrackScrub() {
        const track = timelineBox.querySelector(".iamccs-v2v-track");
        if (!track) return;
        track.onpointerdown = (event) => {
            if (event.target?.classList?.contains("iamccs-v2v-handle")) return;
            event.preventDefault();
            const rect = track.getBoundingClientRect();
            const dur = Math.max(0.01, numberValue(duration, 10));
            const apply = (moveEvent) => {
                const ratio = Math.max(0, Math.min(1, (moveEvent.clientX - rect.left) / Math.max(1, rect.width)));
                const seconds = ratio * dur;
                const { start, end } = trimValues();
                if (state.activeTrim === "start") {
                    const next = Math.max(0, Math.min(seconds, end - 0.01));
                    write(node, "trim_start_s", next);
                    state.playheadSec = next;
                } else {
                    const next = Math.max(start + 0.01, Math.min(dur, seconds));
                    write(node, "trim_end_s", next);
                    state.playheadSec = next;
                }
                drawTimeline();
            };
            const up = () => {
                window.removeEventListener("pointermove", apply);
                window.removeEventListener("pointerup", up);
                commit();
            };
            apply(event);
            window.addEventListener("pointermove", apply);
            window.addEventListener("pointerup", up);
        };
    }

    function bindPlaybarScrub() {
        const analog = timelineBox.querySelector(".iamccs-v2v-analog");
        if (!analog) return;
        const apply = (event) => {
            const rect = analog.getBoundingClientRect();
            const dur = Math.max(0.01, numberValue(duration, 10));
            const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / Math.max(1, rect.width)));
            const { start, end } = trimValues();
            state.playheadSec = Math.max(start, Math.min(end, ratio * dur));
            updatePlayheadVisual();
        };
        analog.onpointerdown = (event) => {
            event.preventDefault();
            const move = (moveEvent) => apply(moveEvent);
            const up = () => {
                window.removeEventListener("pointermove", move);
                window.removeEventListener("pointerup", up);
            };
            apply(event);
            window.addEventListener("pointermove", move);
            window.addEventListener("pointerup", up);
        };
        const playBtn = timelineBox.querySelector(".iamccs-v2v-playbtn");
        if (playBtn) playBtn.onclick = () => {
            const el = state.timelineVideo;
            if (!el) return;
            const { start, end } = trimValues();
            if (el.paused) {
                setVideoCurrentTime(state.playheadSec);
                el.play?.().catch?.(() => {});
                playBtn.textContent = "Pause";
                const tick = () => {
                    if (!state.timelineVideo || state.timelineVideo.paused) {
                        playBtn.textContent = "Play";
                        return;
                    }
                    state.playheadSec = Math.max(start, Math.min(end, Number(state.timelineVideo.currentTime || state.playheadSec)));
                    if (state.playheadSec >= end - 0.01) {
                        state.timelineVideo.pause?.();
                        playBtn.textContent = "Play";
                        state.playheadSec = end;
                    }
                    updatePlayheadVisual(false);
                    requestAnimationFrame(tick);
                };
                requestAnimationFrame(tick);
            } else {
                el.pause?.();
                playBtn.textContent = "Play";
            }
        };
        timelineBox.querySelectorAll(".iamccs-v2v-stepbtn").forEach((btn) => {
            btn.onclick = () => {
                const step = Number(btn.dataset.step || 1);
                const frame = 1 / Math.max(1, numberValue(fps, 24));
                const { start, end } = trimValues();
                state.playheadSec = Math.max(start, Math.min(end, state.playheadSec + frame * step));
                updatePlayheadVisual();
            };
        });
    }

    async function loadVideo(file) {
        if (!file) return;
        if (state.videoObjectUrl) URL.revokeObjectURL(state.videoObjectUrl);
        state.videoObjectUrl = URL.createObjectURL(file);
        try {
            const uploaded = await uploadFile(file);
            videoPath.value = uploaded;
            if (activeBackendMode() === "pose_transfer") poseVideoPath.value = uploaded;
            write(node, "source_video_path", uploaded);
            syncVideoBackend(node, uploaded);
            status.textContent = `video uploaded: ${uploaded}`;
        } catch (err) {
            videoPath.value = file.name;
            if (activeBackendMode() === "pose_transfer") poseVideoPath.value = file.name;
            write(node, "source_video_path", file.name);
            status.textContent = `video preview only: ${err?.message || err}`;
        }
        drawPreview();
        commit();
    }

    async function loadImage(file) {
        if (!file) return;
        if (state.imageObjectUrl) URL.revokeObjectURL(state.imageObjectUrl);
        state.imageObjectUrl = URL.createObjectURL(file);
        try {
            const uploaded = await uploadFile(file);
            imagePath.value = uploaded;
            if (activeBackendMode() === "pose_transfer") poseImagePath.value = uploaded;
            write(node, "source_image_path", uploaded);
            syncImageBackend(node, uploaded);
            status.textContent = `image uploaded: ${uploaded}`;
        } catch (err) {
            imagePath.value = file.name;
            if (activeBackendMode() === "pose_transfer") poseImagePath.value = file.name;
            write(node, "source_image_path", file.name);
            status.textContent = `image preview only: ${err?.message || err}`;
        }
        drawPreview();
        commit();
    }

    async function loadPoseTransferAsset(kind, file) {
        if (!file) return;
        try {
            const uploaded = await uploadFile(file);
            if (kind === "image") {
                poseImagePath.value = uploaded;
                imagePath.value = uploaded;
                write(node, "source_image_path", uploaded);
                syncImageBackend(node, uploaded);
                status.textContent = `pose image uploaded: ${uploaded}`;
            } else if (kind === "video") {
                poseVideoPath.value = uploaded;
                videoPath.value = uploaded;
                write(node, "source_video_path", uploaded);
                syncVideoBackend(node, uploaded);
                status.textContent = `pose video uploaded: ${uploaded}`;
            } else {
                poseResultPath.value = uploaded;
                status.textContent = `pose result uploaded: ${uploaded}`;
            }
        } catch (err) {
            const name = file.name;
            if (kind === "image") poseImagePath.value = name;
            else if (kind === "video") poseVideoPath.value = name;
            else poseResultPath.value = name;
            status.textContent = `pose ${kind} preview only: ${err?.message || err}`;
        }
        drawPreview();
        commit();
    }

    addVideo.onclick = () => { videoInput.dataset.poseTransferKind = ""; videoInput.click(); };
    addImage.onclick = () => { imageInput.dataset.poseTransferKind = ""; imageInput.click(); };
    syncVideo.onclick = () => {
        commit();
        status.textContent = `video path applied: ${videoPath.value}`;
    };
    syncImage.onclick = () => {
        commit();
        status.textContent = `image path applied: ${imagePath.value}`;
    };
    function chooseBackendMode(mode) {
        const config = BACKEND_MODES[mode] || BACKEND_MODES.ltx_simple;
        write(node, "backend_mode", mode);
        write(node, "backend_family", config.family);
        pose.value = config.poseMode;
        outputPrefix.value = config.outputPrefix;
        if (mode === "pose_transfer") write(node, "dwpose_enabled", true);
        if (mode === "scail2" && !String(scailIdentity.value || "").trim()) scailIdentity.value = "single_person";
        if (mode === "wananimate") {
            write(node, "wan_background_lock", true);
            write(node, "wan_control_preview", true);
        }
        setActiveButtons();
        commit();
    }
    Object.entries(backendButtons).forEach(([mode, btn]) => {
        btn.onclick = () => chooseBackendMode(mode);
    });
    normalBtn.onclick = () => { write(node, "vram_profile", "normal_vram"); setActiveButtons(); commit(); };
    lowBtn.onclick = () => { write(node, "vram_profile", "low_vram"); setActiveButtons(); commit(); };
    dwposeBtn.onclick = () => { write(node, "dwpose_enabled", !dwposeBtn.classList.contains("is-active")); setActiveButtons(); commit(); };
    sam31PreviewBtn.onclick = () => { write(node, "enable_sam31_preview", !sam31PreviewBtn.classList.contains("is-active")); setActiveButtons(); commit(); };
    wanBackgroundLockBtn.onclick = () => { write(node, "wan_background_lock", !wanBackgroundLockBtn.classList.contains("is-active")); setActiveButtons(); commit(); };
    wanControlPreviewBtn.onclick = () => { write(node, "wan_control_preview", !wanControlPreviewBtn.classList.contains("is-active")); setActiveButtons(); commit(); };
    const toggleSettings = (open = !optionsPanel.classList.contains("is-open")) => {
        optionsPanel.classList.toggle("is-open", Boolean(open));
        settingsBtn.classList.toggle("is-active", Boolean(open));
        settingsBtn.textContent = open ? "Close Settings" : "Settings";
    };
    settingsBtn.onclick = () => toggleSettings();
    settingsCloseBtn.onclick = () => toggleSettings(false);
    openEditorBtn.onclick = () => {
        const open = !root.classList.contains("is-full-editor");
        root.classList.toggle("is-full-editor", open);
        root.classList.toggle("is-editor-open", open);
        openEditorBtn.textContent = open ? "Close Editor" : "Open Editor";
        if (open) {
            state.editorParent = root.parentElement;
            document.body.appendChild(root);
        } else {
            (state.editorParent || domWidget.element?.parentElement)?.appendChild?.(root);
        }
        drawTimeline();
    };
    videoInput.onchange = (event) => {
        const kind = videoInput.dataset.poseTransferKind;
        const file = event.target.files?.[0];
        const task = kind === "video" ? loadPoseTransferAsset("video", file) : loadVideo(file);
        task.finally(() => { videoInput.value = ""; videoInput.dataset.poseTransferKind = ""; });
    };
    imageInput.onchange = (event) => {
        const kind = imageInput.dataset.poseTransferKind;
        const file = event.target.files?.[0];
        const task = kind === "image" ? loadPoseTransferAsset("image", file) : loadImage(file);
        task.finally(() => { imageInput.value = ""; imageInput.dataset.poseTransferKind = ""; });
    };
    poseUploadImage.onclick = () => { videoInput.dataset.poseTransferKind = ""; imageInput.dataset.poseTransferKind = "image"; imageInput.click(); };
    poseUploadVideo.onclick = () => { imageInput.dataset.poseTransferKind = ""; videoInput.dataset.poseTransferKind = "video"; videoInput.click(); };
    poseUploadResult.onclick = () => poseResultInput.click();
    poseResultInput.onchange = (event) => loadPoseTransferAsset("result", event.target.files?.[0]).finally(() => { poseResultInput.value = ""; });
    poseUseSource.onclick = () => {
        poseVideoPath.value = videoPath.value;
        poseImagePath.value = imagePath.value;
        commit();
        status.textContent = "pose transfer uses current source video and reference image";
    };
    poseUseReference.onclick = () => {
        poseImagePath.value = imagePath.value;
        commit();
        status.textContent = `pose transfer image set: ${poseImagePath.value || "empty"}`;
    };
    mapBackendBtn.onclick = () => {
        const touched = syncBackendGraphWidgets(node, timelinePayload());
        status.textContent = touched ? `backend mapped: ${touched} widget values synced` : "backend map: no compatible widgets found";
        renderBackendPreviews();
    };
    refreshBackendPreviewsBtn.onclick = () => {
        drawPreview();
        status.textContent = "backend previews refreshed from graph";
    };
    scrub.oninput = () => {
        const { dur, start, end } = trimValues();
        const sec = Math.max(start, Math.min(end, (Number(scrub.value || 0) / 1000) * dur));
        state.playheadSec = sec;
        updatePlayheadVisual();
    };
    duration.onchange = () => { autoCapFromDuration(); write(node, "trim_end_s", Math.max(0.01, numberValue(duration, 10))); commit(); };
    duration.oninput = () => { autoCapFromDuration(); drawTimeline(); };
    fps.onchange = () => { autoCapFromDuration(); commit(); };
    fps.oninput = () => { autoCapFromDuration(); drawTimeline(); };
    enableNumberDrag(duration, { step: 0.02, min: 0.1, max: 600, precision: 2, onPreview: () => { autoCapFromDuration(); drawTimeline(); }, onCommit: () => { write(node, "trim_end_s", Math.max(0.01, numberValue(duration, 10))); commit(); } });
    enableNumberDrag(fps, { step: 0.05, min: 1, max: 120, precision: 2, onPreview: () => { autoCapFromDuration(); drawTimeline(); }, onCommit: commit });
    enableNumberDrag(frameCap, { step: 1, min: 1, max: 10000, onPreview: drawTimeline, onCommit: commit });
    enableNumberDrag(widthInput, { step: 8, min: 64, max: 4096, onPreview: drawTimeline, onCommit: commit });
    enableNumberDrag(heightInput, { step: 8, min: 64, max: 4096, onPreview: drawTimeline, onCommit: commit });
    enableNumberDrag(segment, { step: 0.02, min: 0.1, max: 120, precision: 2, onPreview: drawTimeline, onCommit: commit });
    enableNumberDrag(overlap, { step: 1, min: 0, max: 240, onPreview: drawTimeline, onCommit: commit });
    enableNumberDrag(strength, { step: 0.005, min: 0, max: 2, precision: 2, onCommit: commit });
    [videoPath, imagePath, frameCap, widthInput, heightInput, segment, overlap, preset, planning, round, audioVae, audioDevice, audioDtype, pose, strength, outputPrefix, backendVariant, scailIdentity, scailOutputStage, wanMaskMode, poseImagePath, poseVideoPath, poseResultPath, poseResultMode, prompt, negative].forEach((el) => {
        el.onchange = commit;
        el.oninput = () => {
            if (el === segment || el === overlap || el === frameCap || el === widthInput || el === heightInput) drawTimeline();
        };
    });
    root.addEventListener("dragover", (event) => {
        if (!event.dataTransfer?.files?.length) return;
        event.preventDefault();
        root.classList.add("iamccs-v2v-drop");
    });
    root.addEventListener("dragleave", () => root.classList.remove("iamccs-v2v-drop"));
    root.addEventListener("drop", async (event) => {
        const files = Array.from(event.dataTransfer?.files || []);
        if (!files.length) return;
        event.preventDefault();
        root.classList.remove("iamccs-v2v-drop");
        const video = files.find((file) => String(file.type || "").startsWith("video/"));
        const image = files.find((file) => String(file.type || "").startsWith("image/"));
        if (video) await loadVideo(video);
        if (image) await loadImage(image);
    });

    const domWidget = node.addDOMWidget("", "iamccs_shotboard_planner_v2v", root, { serialize: false });
    domWidget.name = "";
    domWidget.label = "";
    domWidget.computeSize = (width) => [Math.max(WIDTH, Number(width || WIDTH)), WIDGET_HEIGHT];
    node.size = [WIDTH, NODE_HEIGHT];
    node.setSize?.([WIDTH, NODE_HEIGHT]);
    node.resizeable = true;

    const originalSerialize = node.onSerialize;
    node.onSerialize = function(serialized) {
        commit();
        return originalSerialize?.call?.(this, serialized);
    };

    drawPreview();
    setActiveButtons();
    commit();
    setTimeout(() => {
        node.size = [WIDTH, NODE_HEIGHT];
        node.setSize?.([WIDTH, NODE_HEIGHT]);
        node.setDirtyCanvas?.(true, true);
    }, 0);
}

app.registerExtension({
    name: `IAMCCS.ShotboardPlannerV2V.${VERSION}`,
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_CLASS) return;
        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = originalCreated?.apply?.(this, arguments);
            renderShotboardV2V(this);
            return result;
        };
        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info) {
            const result = originalConfigure?.apply?.(this, arguments);
            this._iamccsShotboardV2VReady = "";
            setTimeout(() => renderShotboardV2V(this), 0);
            return result;
        };
    },
});
