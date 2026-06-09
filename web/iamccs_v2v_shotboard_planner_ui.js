import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "IAMCCS_ShotboardPlannerV2V";
const VERSION = "2026-06-07-shotboard-v2v-ui-polish1";
const WIDTH = 1660;
const WIDGET_HEIGHT = 705;
const NODE_HEIGHT = 765;

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
.iamccs-v2v-board{box-sizing:border-box;width:100%;height:100%;padding:10px 12px;border:1px solid rgba(95,198,218,.58);border-radius:8px;background:#0f1419;color:#eef5f8;font-family:Inter,Arial,sans-serif;font-size:12px;overflow:hidden;box-shadow:inset 0 1px 0 rgba(255,255,255,.05)}
.iamccs-v2v-board *{box-sizing:border-box;letter-spacing:0}
.iamccs-v2v-board.is-full-editor{position:fixed;left:18px;right:18px;top:18px;bottom:18px;width:auto!important;height:auto!important;z-index:999999;border-color:#d8b860;background:#10151a;box-shadow:0 24px 80px rgba(0,0,0,.72),inset 0 1px 0 rgba(255,255,255,.08)}
.iamccs-v2v-board.is-full-editor .iamccs-v2v-main{grid-template-columns:380px minmax(0,1fr) 460px}
.iamccs-v2v-board.is-full-editor .iamccs-v2v-timebar{min-height:calc(100vh - 300px)}
.iamccs-v2v-board.is-full-editor .iamccs-v2v-track{height:36vh}
.iamccs-v2v-board.is-full-editor .iamccs-v2v-handle{height:36vh}
.iamccs-v2v-head{display:grid;grid-template-columns:auto minmax(0,1fr) auto;align-items:center;gap:14px;height:34px;margin:0 0 10px;border-bottom:1px solid rgba(95,198,218,.28)}
.iamccs-v2v-title{font-size:18px;font-weight:900;color:#fff;white-space:nowrap}
.iamccs-v2v-path{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;text-align:right;color:#9fb3c1;font-size:11px}
.iamccs-v2v-head-editor{height:24px;min-width:112px;padding:0 12px;border-color:#62c99e;background:#1e5b4c;color:#f3fff9}
.iamccs-v2v-main{display:grid;grid-template-columns:350px minmax(0,1fr) 450px;gap:10px;height:calc(100% - 44px);min-height:0}
.iamccs-v2v-panel{min-height:0;display:flex;flex-direction:column;border:1px solid rgba(160,184,205,.28);border-radius:7px;background:#141c24;overflow:hidden;box-shadow:inset 0 1px 0 rgba(255,255,255,.04)}
.iamccs-v2v-panel.preview{border-color:rgba(112,202,222,.44);background:#121921}
.iamccs-v2v-panel.timeline{border-color:rgba(218,184,96,.46);background:#14120f}
.iamccs-v2v-panel.controls{border-color:rgba(172,139,219,.46);background:#15131d}
.iamccs-v2v-panel-head{height:34px;display:flex;align-items:center;justify-content:space-between;padding:0 10px;border-bottom:1px solid rgba(160,184,205,.2);background:linear-gradient(90deg,#192a36,#14202a);font-size:11px;font-weight:900;text-transform:uppercase;color:#e9f7fb}
.iamccs-v2v-panel-body{flex:1;min-height:0;padding:8px;overflow-y:auto;overflow-x:hidden;scrollbar-width:thin;scrollbar-color:#668292 #111820}
.iamccs-v2v-panel-body::-webkit-scrollbar{width:8px}
.iamccs-v2v-panel-body::-webkit-scrollbar-thumb{background:#668292;border-radius:8px}
.iamccs-v2v-media-stack{display:flex;flex-direction:column;gap:10px;height:100%;min-height:0}
.iamccs-v2v-media-card{border:1px solid rgba(160,184,205,.24);border-radius:7px;background:#101820;overflow:hidden;min-height:0}.iamccs-v2v-media-card.source-video{flex:0 0 auto;padding-bottom:2px}
.iamccs-v2v-media-title{height:26px;display:flex;align-items:center;justify-content:space-between;padding:0 9px;color:#b7c7d1;font-size:11px;font-weight:900;background:#16232d}
.iamccs-v2v-media{height:var(--iamccs-v2v-video-h,210px);min-height:145px;max-height:300px;background:#05090d;display:flex;align-items:center;justify-content:center;color:#7f919d;font-size:12px;overflow:hidden}
.iamccs-v2v-media.ref{height:132px;min-height:110px;max-height:170px}
.iamccs-v2v-media video,.iamccs-v2v-media img{width:100%;height:100%;object-fit:contain;background:#05090d}
.iamccs-v2v-preview-monitor{height:268px;border:1px solid rgba(118,214,235,.38);border-radius:7px;background:#05090d;display:flex;align-items:center;justify-content:center;overflow:hidden;color:#8ea8b6;position:relative}
.iamccs-v2v-preview-monitor::before{content:"TAELTX2";position:absolute;left:10px;top:8px;padding:3px 7px;border:1px solid rgba(118,214,235,.45);border-radius:4px;background:rgba(8,17,23,.86);color:#d9fbff;font-size:10px;font-weight:900;z-index:2}
.iamccs-v2v-preview-monitor img,.iamccs-v2v-preview-monitor video{width:100%;height:100%;object-fit:contain;background:#05090d}
.iamccs-v2v-preview-placeholder{width:100%;height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;background:radial-gradient(circle at 50% 36%,#143442,#061016 62%);color:#bfd8e1;text-align:center;font-size:12px;font-weight:800}
.iamccs-v2v-preview-placeholder span{color:#7f9ba8;font-size:11px;font-weight:700}
.iamccs-v2v-resize{height:13px;display:flex;align-items:center;justify-content:center;border-top:1px solid rgba(160,184,205,.14);background:#111c25;cursor:row-resize;color:#7fa6b6;font-size:9px;font-weight:900}
.iamccs-v2v-resize::before{content:"";width:64px;height:3px;border-radius:3px;background:#527b8b}
.iamccs-v2v-actions{display:flex;gap:7px;flex-wrap:wrap;padding:8px;border-top:1px solid rgba(160,184,205,.14)}
.iamccs-v2v-btn{height:30px;padding:0 12px;border:1px solid #52687a;border-radius:5px;background:#243341;color:#f3fbff;font-size:11px;font-weight:900;cursor:pointer;white-space:nowrap;box-shadow:inset 0 1px 0 rgba(255,255,255,.07)}
.iamccs-v2v-btn:hover{background:#2d4051}
.iamccs-v2v-btn.is-active{border-color:#9ee9ff;background:#2b6074;box-shadow:inset 0 0 0 1px rgba(255,255,255,.16),0 0 0 1px rgba(95,198,218,.16)}
.iamccs-v2v-btn.good{background:#1e5b4c;border-color:#62c99e}
.iamccs-v2v-btn.warn{background:#50375b;border-color:#bd89d9}
.iamccs-v2v-mode-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.iamccs-v2v-mode-grid .iamccs-v2v-btn{width:100%;height:36px;text-align:center;border-radius:6px}
.iamccs-v2v-editor-row{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}
.iamccs-v2v-editor-drawer{display:none;margin-top:8px;padding:9px;border:1px solid rgba(218,184,96,.34);border-radius:6px;background:#1f1b12;color:#efd89a;font-size:11px;line-height:1.35}
.iamccs-v2v-board.is-editor-open .iamccs-v2v-editor-drawer{display:block}
.iamccs-v2v-field{display:grid;grid-template-columns:94px minmax(0,1fr);gap:7px;align-items:center;margin:6px 0}
.iamccs-v2v-field label{font-size:11px;color:#adbdc8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.iamccs-v2v-field input,.iamccs-v2v-field select,.iamccs-v2v-field textarea{width:100%;min-width:0;border:1px solid rgba(160,184,205,.34);border-radius:5px;background:#0b1219;color:#f7fbff;padding:7px 8px;font-size:12px;outline:none}
.iamccs-v2v-field textarea{resize:vertical;min-height:94px;line-height:1.32;font-family:Consolas,monospace;background:#f8f4ea;color:#111;border-color:#d7c797;font-weight:700}
.iamccs-v2v-field input[type="number"]{cursor:ew-resize;font-variant-numeric:tabular-nums}
.iamccs-v2v-drag-number{border-color:#7897ad!important;background:#101b25!important}
.iamccs-v2v-path-field{padding:2px 8px 10px}
.iamccs-v2v-mid{display:grid;grid-template-rows:minmax(0,1fr) auto;gap:8px;height:100%;min-height:0}
.iamccs-v2v-prompt-grid{display:grid;grid-template-columns:1fr;gap:7px}
.iamccs-v2v-prompt-grid textarea{height:58px}
.iamccs-v2v-timeline-card{min-height:0;border:1px solid rgba(218,184,96,.34);border-radius:7px;background:#0b1117;overflow:hidden;display:flex;flex-direction:column}
.iamccs-v2v-timeline-head{height:34px;display:flex;align-items:center;justify-content:space-between;padding:0 10px;background:#271f13;color:#f6e7b4;font-size:11px;font-weight:900;text-transform:uppercase}
.iamccs-v2v-timebar{position:relative;flex:1;min-height:520px;background:linear-gradient(180deg,#100e0b,#091016);overflow:hidden}
.iamccs-v2v-ruler{position:absolute;left:26px;right:26px;top:18px;height:32px;border-bottom:1px solid rgba(151,181,201,.24);color:#9aaebd;font-size:10px}
.iamccs-v2v-tick{position:absolute;top:18px;width:1px;height:12px;background:rgba(151,181,201,.34)}
.iamccs-v2v-tick span{position:absolute;top:-16px;left:-12px;color:#8ea3b2;font-size:10px}
.iamccs-v2v-track{position:absolute;left:34px;right:34px;top:82px;height:264px;border:1px solid rgba(218,184,96,.46);border-radius:7px;background:#060a0f;overflow:hidden;cursor:crosshair}
.iamccs-v2v-timeline-media{position:absolute;inset:0;background:linear-gradient(90deg,#111b24,#182838);overflow:hidden}
.iamccs-v2v-timeline-media video{width:100%;height:100%;object-fit:cover;filter:saturate(.9) contrast(.92);opacity:.9}
.iamccs-v2v-timeline-media-empty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#758a99;font-size:12px;background:repeating-linear-gradient(90deg,#101822 0,#101822 34px,#142130 34px,#142130 68px)}
.iamccs-v2v-timeline-shade{position:absolute;top:0;bottom:0;background:rgba(0,0,0,.56);backdrop-filter:saturate(.7)}
.iamccs-v2v-selected{position:absolute;top:0;bottom:0;background:linear-gradient(180deg,rgba(52,160,135,.38),rgba(61,103,154,.42));box-shadow:inset 0 0 0 2px rgba(167,245,255,.55)}
.iamccs-v2v-selected::after{content:"";position:absolute;inset:0;background:linear-gradient(180deg,rgba(255,255,255,.08),rgba(255,255,255,0));pointer-events:none}
.iamccs-v2v-handle{position:absolute;top:0;width:20px;height:264px;margin-left:-10px;border:1px solid #f4e6b8;border-radius:5px;background:#f3ead1;color:#0d1a22;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:900;cursor:ew-resize;box-shadow:0 8px 22px rgba(0,0,0,.34);z-index:6}
.iamccs-v2v-handle.is-active{background:#d6b65d;border-color:#fff1b4;color:#060606;box-shadow:0 0 0 2px rgba(214,182,93,.24),0 8px 22px rgba(0,0,0,.34)}
.iamccs-v2v-playhead{position:absolute;top:0;bottom:0;width:2px;background:#d6b65d;box-shadow:0 0 0 1px rgba(0,0,0,.55),0 0 12px rgba(214,182,93,.7);pointer-events:none;z-index:7}
.iamccs-v2v-playhead::before{content:"";position:absolute;top:-9px;left:-6px;width:14px;height:14px;background:#d6b65d;clip-path:polygon(50% 100%,0 0,100% 0)}
.iamccs-v2v-playhead-time{position:absolute;top:8px;transform:translateX(-50%);padding:3px 6px;border-radius:4px;background:rgba(0,0,0,.72);color:#fff0b4;font-size:10px;font-weight:900;z-index:8;pointer-events:none}
.iamccs-v2v-frame-bubble{display:none}
.iamccs-v2v-frame-bubble canvas{width:100%;height:58px;display:block;background:#020405}
.iamccs-v2v-frame-bubble span{display:block;height:20px;line-height:20px;padding:0 6px;background:#211a10;color:#f8e8b5;font-size:10px;font-weight:900;text-align:center;font-variant-numeric:tabular-nums}
.iamccs-v2v-segments{position:absolute;left:34px;right:34px;top:372px;height:52px;display:flex;gap:6px;overflow-x:auto;overflow-y:hidden;scrollbar-width:thin;padding-bottom:3px}
.iamccs-v2v-segment{height:100%;border-radius:5px;background:#273a50;border:1px solid rgba(196,220,236,.24);display:flex;align-items:center;justify-content:center;color:#e6f4fb;font-size:11px;font-weight:900;min-width:110px;flex:1 0 110px;overflow:hidden}
.iamccs-v2v-playbar{position:absolute;left:34px;right:34px;top:436px;height:42px;border:1px solid rgba(218,184,96,.24);border-radius:7px;background:#12100d;display:grid;grid-template-columns:74px minmax(0,1fr) 74px;gap:10px;align-items:center;padding:0 10px}
.iamccs-v2v-playbar button{height:28px;border:1px solid rgba(218,184,96,.48);border-radius:5px;background:#2a2115;color:#f7e6ae;font-size:11px;font-weight:900;cursor:pointer}
.iamccs-v2v-analog{height:6px;border-radius:8px;background:linear-gradient(90deg,#8d7440,#2c3d4e);position:relative;cursor:ew-resize;box-shadow:inset 0 1px 0 rgba(255,255,255,.15)}
.iamccs-v2v-analog-thumb{position:absolute;top:50%;width:14px;height:14px;border-radius:50%;background:#d6b65d;border:1px solid #fff2b9;transform:translate(-50%,-50%);box-shadow:0 1px 8px rgba(0,0,0,.5)}
.iamccs-v2v-time-label{font-size:11px;color:#f0d991;font-weight:900;text-align:right;font-variant-numeric:tabular-nums}
.iamccs-v2v-taeltx-stage{position:absolute;left:34px;right:34px;bottom:18px;height:44px;border:1px dashed rgba(112,202,222,.42);border-radius:7px;background:#0c141b;display:grid;grid-template-columns:160px minmax(0,1fr) 82px;align-items:center;gap:10px;padding:0 12px;color:#8da5b3}
.iamccs-v2v-taeltx-stage.is-active{border-style:solid;border-color:#78d7ef;background:#102230;color:#d8f8ff}
.iamccs-v2v-taeltx-title{font-size:11px;font-weight:900;text-transform:uppercase;color:#d8f8ff}
.iamccs-v2v-taeltx-strip{height:34px;border-radius:5px;background:repeating-linear-gradient(90deg,#172534 0,#172534 40px,#1e3144 40px,#1e3144 80px);box-shadow:inset 0 0 0 1px rgba(255,255,255,.05)}
.iamccs-v2v-taeltx-state{font-size:10px;font-weight:900;text-align:right;text-transform:uppercase}
.iamccs-v2v-scrub-wrap{height:0;padding:0;border:0;overflow:hidden}
.iamccs-v2v-scrub-row{display:grid;grid-template-columns:60px minmax(0,1fr) 66px;gap:8px;align-items:center}
.iamccs-v2v-scrub-row span{color:#a9bac5;font-size:11px;font-weight:800}
.iamccs-v2v-scrub{width:100%;accent-color:#66d2e7}
.iamccs-v2v-readout{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.iamccs-v2v-chip{border:1px solid rgba(160,184,205,.24);border-radius:6px;background:#101820;padding:8px 9px;min-width:0}
.iamccs-v2v-chip span{display:block;color:#91a7b6;font-size:10px;text-transform:uppercase;font-weight:900;margin-bottom:3px}
.iamccs-v2v-chip b{display:block;font-size:13px;color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.iamccs-v2v-control-group{border:1px solid rgba(160,184,205,.24);border-radius:7px;background:#101820;margin-bottom:8px;overflow:hidden}
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
    root.append(videoInput, imageInput);

    const head = document.createElement("div");
    head.className = "iamccs-v2v-head";
    const title = document.createElement("div");
    title.className = "iamccs-v2v-title";
    title.textContent = "IAMCCS Shotboard Planner V2V";
    const openEditorBtn = button("Open Editor", "good");
    openEditorBtn.classList.add("iamccs-v2v-head-editor");
    head.append(title, status, openEditorBtn);
    root.appendChild(head);

    const main = document.createElement("div");
    main.className = "iamccs-v2v-main";
    root.appendChild(main);

    const videoPath = input("text", read(node, "source_video_path", "IMG_4145 2.mp4"));
    const imagePath = input("text", read(node, "source_image_path", "QWEN2509_FIRST_FRAME_DWPOSE_OPENPOSE_CONTROL_00001_.png"));

    const mediaPanel = document.createElement("section");
    mediaPanel.className = "iamccs-v2v-panel preview";
    mediaPanel.innerHTML = `<div class="iamccs-v2v-panel-head"><span>Preview</span><span>TAELTX2 override</span></div>`;
    const mediaBody = document.createElement("div");
    mediaBody.className = "iamccs-v2v-panel-body";
    const mediaStack = document.createElement("div");
    mediaStack.className = "iamccs-v2v-media-stack";

    const previewCard = document.createElement("section");
    previewCard.className = "iamccs-v2v-media-card";
    previewCard.innerHTML = `<div class="iamccs-v2v-media-title"><span>TAELTX2 sampler preview</span><span>preview override</span></div>`;
    const previewBox = document.createElement("div");
    previewBox.className = "iamccs-v2v-preview-monitor";
    previewCard.appendChild(previewBox);

    const sourceCard = document.createElement("section");
    sourceCard.className = "iamccs-v2v-media-card source-video";
    sourceCard.innerHTML = `<div class="iamccs-v2v-media-title"><span>Source video</span><span>loaded into timeline</span></div>`;
    const videoBtns = document.createElement("div");
    videoBtns.className = "iamccs-v2v-actions";
    const addVideo = button("Upload Video", "good");
    const syncVideo = button("Apply Path");
    videoBtns.append(addVideo, syncVideo);
    const videoPathWrap = document.createElement("div");
    videoPathWrap.className = "iamccs-v2v-path-field";
    videoPathWrap.appendChild(field("video", videoPath));
    sourceCard.append(videoBtns, videoPathWrap);

    const imageCard = document.createElement("section");
    imageCard.className = "iamccs-v2v-media-card";
    imageCard.innerHTML = `<div class="iamccs-v2v-media-title"><span>Reference / pose image</span><span>first frame</span></div>`;
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
    mediaStack.append(previewCard, sourceCard, imageCard);
    mediaBody.appendChild(mediaStack);
    mediaPanel.appendChild(mediaBody);
    main.appendChild(mediaPanel);

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
    optionsPanel.className = "iamccs-v2v-panel controls";
    optionsPanel.innerHTML = `<div class="iamccs-v2v-panel-head"><span>V2V Controls</span><span>backend values</span></div>`;
    const optionsBody = document.createElement("div");
    optionsBody.className = "iamccs-v2v-panel-body";
    const normalBtn = button("Normal VRAM");
    const lowBtn = button("Low VRAM");
    const taeltxBtn = button("TAELTX Preview", "warn");
    const dwposeBtn = button("DW Pose");

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
    const previewFrames = input("number", read(node, "taeltx_preview_max_frames", 17));
    previewFrames.step = "1";
    const previewFps = input("number", read(node, "taeltx_preview_fps", 8));
    previewFps.step = "1";
    const outputPrefix = input("text", read(node, "output_prefix", "IAMCCS/LTX23_V2V_SHOTBOARD"));

    const modeRow = document.createElement("div");
    modeRow.className = "iamccs-v2v-mode-grid";
    modeRow.append(normalBtn, lowBtn, taeltxBtn, dwposeBtn);
    const editorDrawer = document.createElement("div");
    editorDrawer.className = "iamccs-v2v-editor-drawer";
    editorDrawer.textContent = "Timeline editor ready: source-video strip, trim handles, playhead scratch and segment lane. This space is reserved for future multi-timeline field/counterfield generation.";
    optionsBody.append(
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
        controlGroup("Pose + preview", "control", [
            field("pose", pose),
            field("pose str", strength),
            field("preview f", previewFrames),
            field("preview fps", previewFps),
        ], "pose"),
        controlGroup("Audio + output", "backend", [
            field("audio vae", audioVae),
            field("device", audioDevice),
            field("dtype", audioDtype),
        ], "audio"),
        Object.assign(document.createElement("div"), { className: "iamccs-v2v-note", textContent: "Media, planner, source ranges, audio VAE and segment audio are generated inside the CineInfo bridge." })
    );
    optionsPanel.appendChild(optionsBody);
    main.appendChild(optionsPanel);

    function numberValue(el, fallback) {
        const n = Number(el.value);
        return Number.isFinite(n) ? n : fallback;
    }

    function boolWidget(name, fallback) {
        const value = read(node, name, fallback);
        if (typeof value === "boolean") return value;
        return String(value).toLowerCase() === "true" || String(value) === "1";
    }

    function setActiveButtons() {
        const vram = String(read(node, "vram_profile", "normal_vram"));
        normalBtn.classList.toggle("is-active", vram === "normal_vram");
        lowBtn.classList.toggle("is-active", vram === "low_vram");
        taeltxBtn.classList.toggle("is-active", boolWidget("taeltx_preview_enabled", false));
        dwposeBtn.classList.toggle("is-active", boolWidget("dwpose_enabled", true));
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
            schema_version: 1,
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
        write(node, "audio_vae_name", audioVae.value);
        write(node, "audio_vae_device", audioDevice.value);
        write(node, "audio_vae_dtype", audioDtype.value);
        write(node, "pose_mode", pose.value);
        write(node, "dwpose_strength", Math.max(0, numberValue(strength, 0.75)));
        write(node, "taeltx_preview_max_frames", Math.max(1, Math.round(numberValue(previewFrames, 17))));
        write(node, "taeltx_preview_fps", Math.max(1, Math.round(numberValue(previewFps, 8))));
        write(node, "global_prompt", prompt.value);
        write(node, "negative_prompt", negative.value);
        write(node, "output_prefix", outputPrefix.value);
        write(node, "ui_video_height", Math.round(state.videoHeight));
        write(node, "timeline_data", JSON.stringify(timelinePayload(), null, 2));
        setActiveButtons();
        drawTimeline();
    }

    function drawPreview() {
        const video = String(videoPath.value || "").trim();
        const image = String(imagePath.value || "").trim();
        previewBox.innerHTML = "";
        imageBox.innerHTML = "";
        const preview = document.createElement("div");
        preview.className = "iamccs-v2v-preview-placeholder";
        preview.innerHTML = boolWidget("taeltx_preview_enabled", false)
            ? `TAELTX2 Preview Enabled<span>waiting for preview override frames</span>`
            : `TAELTX2 Preview Standby<span>enable preview to monitor sampler output</span>`;
        previewBox.appendChild(preview);
        if (state.imageObjectUrl || image) {
            const el = document.createElement("img");
            el.src = state.imageObjectUrl || viewUrl(image, "input");
            imageBox.appendChild(el);
        } else {
            imageBox.textContent = "Drop or upload reference / first-frame image";
        }
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
        const segCount = Math.max(1, Math.ceil((end - start) / Math.max(0.01, Number(data.segment_seconds || 10))));
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
            <div class="iamccs-v2v-segments"></div>
            <div class="iamccs-v2v-playbar">
                <button type="button" class="iamccs-v2v-playbtn">Preview</button>
                <div class="iamccs-v2v-analog"><div class="iamccs-v2v-analog-thumb" style="left:${playLeft}%"></div></div>
                <div class="iamccs-v2v-time-label">${state.playheadSec.toFixed(2)}s</div>
            </div>
            <div class="iamccs-v2v-taeltx-stage">
                <div class="iamccs-v2v-taeltx-title">TAELTX2 Preview</div>
                <div class="iamccs-v2v-taeltx-strip"></div>
                <div class="iamccs-v2v-taeltx-state">standby</div>
            </div>
        `;
        const media = timelineBox.querySelector(".iamccs-v2v-timeline-media");
        ensureTimelineVideo(media);
        const taeltxStage = timelineBox.querySelector(".iamccs-v2v-taeltx-stage");
        const taeltxActive = boolWidget("taeltx_preview_enabled", false);
        taeltxStage?.classList.toggle("is-active", taeltxActive);
        const taeltxState = timelineBox.querySelector(".iamccs-v2v-taeltx-state");
        if (taeltxState) taeltxState.textContent = taeltxActive ? "enabled" : "standby";
        const segs = timelineBox.querySelector(".iamccs-v2v-segments");
        for (let i = 0; i < segCount; i++) {
            const item = document.createElement("div");
            item.className = "iamccs-v2v-segment";
            item.textContent = `S${i + 1} · ${Number(data.segment_seconds).toFixed(2)}s`;
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
        if (playBtn) playBtn.onclick = () => setVideoCurrentTime(state.playheadSec);
    }

    async function loadVideo(file) {
        if (!file) return;
        if (state.videoObjectUrl) URL.revokeObjectURL(state.videoObjectUrl);
        state.videoObjectUrl = URL.createObjectURL(file);
        try {
            const uploaded = await uploadFile(file);
            videoPath.value = uploaded;
            write(node, "source_video_path", uploaded);
            syncVideoBackend(node, uploaded);
            status.textContent = `video uploaded: ${uploaded}`;
        } catch (err) {
            videoPath.value = file.name;
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
            write(node, "source_image_path", uploaded);
            syncImageBackend(node, uploaded);
            status.textContent = `image uploaded: ${uploaded}`;
        } catch (err) {
            imagePath.value = file.name;
            write(node, "source_image_path", file.name);
            status.textContent = `image preview only: ${err?.message || err}`;
        }
        drawPreview();
        commit();
    }

    addVideo.onclick = () => videoInput.click();
    addImage.onclick = () => imageInput.click();
    syncVideo.onclick = () => {
        commit();
        status.textContent = `video path applied: ${videoPath.value}`;
    };
    syncImage.onclick = () => {
        commit();
        status.textContent = `image path applied: ${imagePath.value}`;
    };
    normalBtn.onclick = () => { write(node, "vram_profile", "normal_vram"); setActiveButtons(); commit(); };
    lowBtn.onclick = () => { write(node, "vram_profile", "low_vram"); setActiveButtons(); commit(); };
    taeltxBtn.onclick = () => { write(node, "taeltx_preview_enabled", !taeltxBtn.classList.contains("is-active")); setActiveButtons(); drawPreview(); commit(); };
    dwposeBtn.onclick = () => { write(node, "dwpose_enabled", !dwposeBtn.classList.contains("is-active")); setActiveButtons(); commit(); };
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
    videoInput.onchange = (event) => loadVideo(event.target.files?.[0]).finally(() => { videoInput.value = ""; });
    imageInput.onchange = (event) => loadImage(event.target.files?.[0]).finally(() => { imageInput.value = ""; });
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
    enableNumberDrag(previewFrames, { step: 1, min: 1, max: 240, onCommit: commit });
    enableNumberDrag(previewFps, { step: 1, min: 1, max: 60, onCommit: commit });
    [videoPath, imagePath, frameCap, widthInput, heightInput, segment, overlap, preset, planning, round, audioVae, audioDevice, audioDtype, pose, strength, previewFrames, previewFps, outputPrefix, prompt, negative].forEach((el) => {
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
