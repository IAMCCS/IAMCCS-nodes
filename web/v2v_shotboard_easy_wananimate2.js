import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "IAMCCS_V2VShotboardEasyWanAnimate2";
const NODE_DISPLAY = "V2V Shotboard Easy - Wan Animate 2 Edition";
const VERSION = "1.0.0-easy-wananimate2";
const UNIFIED_WORKFLOW_PATH = "workflows/IAMCCS_V2V_SHOTBOARD_EASY/IAMCCS_V2V_SHOTBOARD_EASY_WANANIMATE2_GGUF.json";
const UNIFIED_WORKFLOW_NAME = "IAMCCS_V2V_SHOTBOARD_EASY_WANANIMATE2_GGUF";
const runtime = { session: null, eventsBound: false };

console.info(`[IAMCCS Wan Animate 2 Easy] frontend ${VERSION} loaded`);

const $ = (root, selector) => root?.querySelector?.(selector) || null;
const $$ = (root, selector) => Array.from(root?.querySelectorAll?.(selector) || []);

function nodeClass(node) {
    return String(node?.comfyClass || node?.type || node?.constructor?.type || "");
}

function isPlannerNode(node) {
    return nodeClass(node) === NODE_CLASS || String(node?.title || node?.constructor?.title || "") === NODE_DISPLAY;
}

function graphFor(source = null) {
    return source?.graph || app?.canvas?.graph || app?.graph || null;
}

function graphNodes(source = null) {
    const graph = graphFor(source);
    return Array.isArray(graph?._nodes) ? graph._nodes : [];
}

function widget(node, name) {
    return (node?.widgets || []).find((item) => item?.name === name);
}

function read(node, name, fallback = "") {
    const value = widget(node, name)?.value;
    return value === undefined || value === null || value === "" ? fallback : value;
}

function write(node, name, value) {
    const item = widget(node, name);
    if (!item) return false;
    item.value = value;
    try { item.callback?.(value, null, node); } catch {}
    node.setDirtyCanvas?.(true, true);
    return true;
}

function writeAny(node, names, value) {
    for (const name of names) {
        if (write(node, name, value)) return true;
    }
    return false;
}

function sameValue(actual, expected) {
    if (typeof expected === "number") {
        const value = Number(actual);
        return Number.isFinite(value) && Math.abs(value - expected) < 0.0001;
    }
    return String(actual ?? "") === String(expected ?? "");
}

function linkedFrom(target, inputName, sourceType) {
    const input = (target?.inputs || []).find((item) => item?.name === inputName);
    if (input?.link === undefined || input?.link === null) return false;
    const graph = graphFor(target);
    const links = graph?.links || graph?._links;
    const link = links?.get?.(input.link) || links?.[input.link];
    const sourceId = link?.origin_id ?? link?.originId;
    const source = graph?.getNodeById?.(sourceId) || graphNodes(target).find((item) => item?.id === sourceId);
    return nodeClass(source) === sourceType;
}

function expectWidget(issues, target, names, expected, label) {
    const item = (target?.widgets || []).find((candidate) => names.includes(candidate?.name));
    if (!item) {
        issues.push(`${label}: backend widget missing`);
    } else if (!sameValue(item.value, expected)) {
        issues.push(`${label}: UI=${expected}, backend=${item.value}`);
    }
}

function hideWidget(item) {
    if (!item || item._iamccsV2VFreeHidden) return;
    item._iamccsV2VFreeHidden = true;
    item.hidden = true;
    item.disabled = true;
    item.computeSize = () => [0, 0];
    item.draw = () => {};
    item.type = "hidden";
    item.options = { ...(item.options || {}), hidden: true };
}

function pathDistance(a, b) {
    const dx = Number(a?.pos?.[0] || 0) - Number(b?.pos?.[0] || 0);
    const dy = Number(a?.pos?.[1] || 0) - Number(b?.pos?.[1] || 0);
    return dx * dx + dy * dy;
}

function nodesByType(source, types) {
    const set = new Set(types);
    return graphNodes(source)
        .filter((item) => item !== source && set.has(nodeClass(item)))
        .sort((a, b) => pathDistance(source, a) - pathDistance(source, b));
}

function inputRelativePath(data) {
    const name = String(data?.name || "");
    const subfolder = String(data?.subfolder || "").replace(/\\/g, "/").replace(/^\/+|\/+$/g, "");
    return [subfolder, name].filter(Boolean).join("/");
}

function viewUrl(item) {
    if (!item?.filename) return "";
    const query = new URLSearchParams({
        filename: item.filename,
        subfolder: item.subfolder || "",
        type: item.type || "input",
    });
    if (item.format) query.set("format", item.format);
    query.set("_", String(item.cacheKey || Date.now()));
    return `/view?${query.toString()}`;
}

function posterUrl(item) {
    if (mediaKind(item) !== "video") return item?.url || viewUrl(item);
    const poster = String(item?.workflow || item?.poster || "").trim();
    if (!poster || !/\.(png|jpe?g|webp)$/i.test(poster)) return "";
    return viewUrl({
        filename: poster,
        subfolder: item.subfolder || "",
        type: item.type || "output",
        cacheKey: item.cacheKey,
    });
}

function itemFromPath(path, kind) {
    const clean = String(path || "").trim().replace(/\\/g, "/");
    if (!clean || /^[A-Za-z]:\//.test(clean)) return null;
    const parts = clean.split("/").filter(Boolean);
    const filename = parts.pop() || "";
    return filename ? { filename, subfolder: parts.join("/"), type: "input", kind, cacheKey: Date.now() } : null;
}

function mediaKind(item) {
    if (item?.kind) return item.kind;
    const text = `${item?.filename || ""} ${item?.format || ""}`.toLowerCase();
    return /\.(mp4|webm|mov|mkv|avi)|video\//.test(text) ? "video" : "image";
}

async function upload(file) {
    const body = new FormData();
    body.append("image", file);
    body.append("overwrite", "true");
    const response = await api.fetchApi("/upload/image", { method: "POST", body });
    const data = await response.json();
    if (!response.ok || data?.error) throw new Error(data?.error || `Upload failed (${response.status})`);
    return { path: inputRelativePath(data), item: { filename: data.name, subfolder: data.subfolder || "", type: data.type || "input", kind: String(file.type || "").startsWith("video/") ? "video" : "image", cacheKey: Date.now() } };
}

function installStyles() {
    if (document.getElementById("iamccs-v2v-free-style")) return;
    const style = document.createElement("style");
    style.id = "iamccs-v2v-free-style";
    style.textContent = `
        .iamccs-v2vf { --bg:#081016; --panel:#101d25; --panel2:#0b151b; --line:#35505e; --cyan:#59cee6; --mint:#59d5ae; --gold:#e4bd62; --rose:#dc6f91; --violet:#a48be8; --text:#edf5f7; --muted:#8fa5af; position:fixed; inset:0; z-index:100000; display:grid; grid-template-rows:58px minmax(0,1fr); color:var(--text); background:var(--bg); font-family:Inter,Segoe UI,Arial,sans-serif; letter-spacing:0; }
        .iamccs-v2vf * { box-sizing:border-box; }
        .iamccs-v2vf button,.iamccs-v2vf input,.iamccs-v2vf select,.iamccs-v2vf textarea { font:inherit; letter-spacing:0; }
        .iamccs-v2vf button { min-height:30px; border:1px solid #3a5664; border-radius:2px; background:#172b36; color:var(--text); padding:0 11px; font-size:10px; font-weight:800; cursor:pointer; }
        .iamccs-v2vf button:hover { border-color:var(--cyan); background:#1b3946; }
        .iamccs-v2vf button.active,.iamccs-v2vf button.primary { border-color:var(--cyan); background:#1b596a; color:#fff; }
        .iamccs-v2vf button.danger { border-color:#723d4d; background:#2b1820; color:#f1a8bc; }
        .iamccs-v2vf button:disabled { opacity:.38; cursor:not-allowed; }
        .iamccs-v2vf input,.iamccs-v2vf select,.iamccs-v2vf textarea { width:100%; border:1px solid #38515d; border-radius:2px; background:#071016; color:var(--text); padding:8px; font-size:11px; }
        .iamccs-v2vf textarea { min-height:78px; resize:vertical; }
        .iamccs-v2vf-header { display:grid; grid-template-columns:280px 1fr auto; align-items:center; gap:14px; border-bottom:1px solid var(--line); background:#0d1a22; padding:7px 12px; }
        .iamccs-v2vf-brand { display:flex; align-items:center; gap:10px; min-width:0; }
        .iamccs-v2vf-logo { width:42px; height:42px; display:grid; grid-template-rows:1fr 1fr; border:1px solid var(--gold); background:#071016; font-size:12px; font-weight:950; text-align:center; }
        .iamccs-v2vf-logo span { display:grid; place-items:center; }
        .iamccs-v2vf-logo span:last-child { color:var(--cyan); border-top:1px solid #32505e; }
        .iamccs-v2vf-brand b { display:block; font-size:14px; }
        .iamccs-v2vf-brand small { color:var(--muted); font-size:9px; }
        .iamccs-v2vf-modebar { display:flex; justify-content:center; gap:5px; }
        .iamccs-v2vf-actions { display:flex; align-items:center; gap:7px; }
        .iamccs-v2vf-status { display:flex; align-items:center; gap:6px; color:#9cdcc9; font-size:9px; font-weight:800; white-space:nowrap; }
        .iamccs-v2vf-status i { width:7px; height:7px; border-radius:50%; background:var(--mint); box-shadow:0 0 8px rgba(89,213,174,.65); }
        .iamccs-v2vf-shell { min-height:0; display:grid; grid-template-columns:minmax(245px,17vw) minmax(650px,1fr) minmax(315px,21vw); gap:8px; padding:8px; }
        .iamccs-v2vf-column { min-height:0; border:1px solid var(--line); background:var(--panel); overflow-y:auto; scrollbar-width:thin; scrollbar-color:#526c78 #0a1319; }
        .iamccs-v2vf-center { min-height:0; display:grid; grid-template-rows:minmax(270px,42vh) minmax(280px,1fr); gap:8px; }
        .iamccs-v2vf-section-head { min-height:42px; display:flex; align-items:center; justify-content:space-between; gap:8px; padding:8px 10px; border-bottom:1px solid var(--line); background:#14242e; }
        .iamccs-v2vf-section-head strong { font-size:12px; }
        .iamccs-v2vf-kicker { color:var(--cyan); font-size:8px; font-weight:900; }
        .iamccs-v2vf-card { margin:8px; border:1px solid #314955; background:var(--panel2); }
        .iamccs-v2vf-card-head { display:flex; justify-content:space-between; gap:8px; padding:7px 8px; border-bottom:1px solid #314955; font-size:9px; font-weight:900; }
        .iamccs-v2vf-card-head span:last-child { color:var(--muted); }
        .iamccs-v2vf-media { position:relative; display:grid; place-items:center; width:100%; aspect-ratio:16/9; min-height:110px; overflow:hidden; isolation:isolate; background:#03090d; color:var(--muted); text-align:center; }
        .iamccs-v2vf-media > img,.iamccs-v2vf-media > video { position:absolute!important; inset:0!important; display:block!important; width:100%!important; height:100%!important; min-width:0!important; min-height:0!important; max-width:none!important; max-height:none!important; object-fit:contain!important; object-position:50% 50%!important; background:#03090d!important; transform:none!important; }
        .iamccs-v2vf-media-placeholder b { display:block; color:var(--gold); margin-bottom:5px; font-size:11px; }
        .iamccs-v2vf-card-actions { display:grid; grid-template-columns:1fr 1fr; gap:6px; padding:7px; }
        .iamccs-v2vf-path { padding:0 7px 7px; }
        .iamccs-v2vf-monitors { min-height:0; border:1px solid var(--line); background:var(--panel); display:grid; grid-template-rows:42px minmax(0,1fr); }
        .iamccs-v2vf-monitor-grid { min-height:0; display:grid; grid-template-columns:1fr 1fr; gap:1px; background:var(--line); }
        .iamccs-v2vf-monitor { min-width:0; min-height:0; display:grid; grid-template-rows:24px minmax(0,1fr) 34px 22px; background:#050b0f; }
        .iamccs-v2vf-monitor-head,.iamccs-v2vf-monitor-foot { display:flex; align-items:center; justify-content:space-between; padding:0 7px; background:#11232c; color:#9ab0ba; font-size:8px; font-weight:850; }
        .iamccs-v2vf-monitor-foot { background:#0c1a21; }
        .iamccs-v2vf-monitor .iamccs-v2vf-media { min-height:0; aspect-ratio:auto; }
        .iamccs-v2vf-transport { display:flex; align-items:center; justify-content:center; gap:4px; border-top:1px solid #243e49; background:#0c171d; }
        .iamccs-v2vf-transport button { min-height:24px; height:24px; padding:0 9px; font-size:9px; }
        .iamccs-v2vf-full { min-height:21px!important; height:21px!important; color:#b8f1db!important; border-color:#417561!important; }
        .iamccs-v2vf-timeline { min-height:0; border:1px solid var(--line); background:var(--panel); display:grid; grid-template-rows:42px 28px minmax(0,1fr) 30px; overflow:hidden; }
        .iamccs-v2vf-timeline-head { display:grid; grid-template-columns:1fr auto 1fr; align-items:center; gap:8px; padding:6px 8px; border-bottom:1px solid var(--line); background:#14242e; }
        .iamccs-v2vf-timeline-head strong { font-size:12px; }
        .iamccs-v2vf-timeline-head .iamccs-v2vf-transport { border:0; background:transparent; }
        .iamccs-v2vf-timeline-head > div:last-child { display:flex; justify-content:flex-end; gap:5px; }
        .iamccs-v2vf-ruler { position:relative; margin-left:92px; border-bottom:1px solid #39505a; background:#091218; cursor:ew-resize; user-select:none; -webkit-user-select:none; touch-action:none; }
        .iamccs-v2vf-ruler.scrubbing { cursor:grabbing; }
        .iamccs-v2vf-tick { position:absolute; top:0; height:100%; border-left:1px solid #3a515c; color:#8fb5c4; font-size:7px; padding:5px 0 0 3px; white-space:nowrap; pointer-events:none; user-select:none; }
        .iamccs-v2vf-playhead { position:absolute; z-index:20; top:0; bottom:0; width:2px; background:#f4d57c; pointer-events:none; }
        .iamccs-v2vf-playhead::before { content:""; position:absolute; top:0; left:-4px; width:10px; height:8px; background:#f4d57c; }
        .iamccs-v2vf-lanes { min-height:0; overflow:auto; padding:5px; }
        .iamccs-v2vf-lane { min-height:49px; display:grid; grid-template-columns:86px minmax(0,1fr); border:1px solid #2d4651; margin-bottom:4px; background:#071016; }
        .iamccs-v2vf-lane.source { min-height:82px; }
        .iamccs-v2vf-lane-label { position:relative; padding:7px 6px 6px 28px; border-right:1px solid #2d4651; background:#11242d; }
        .iamccs-v2vf-lane-label b { display:block; color:var(--cyan); font-size:8px; }
        .iamccs-v2vf-lane-label small { color:var(--muted); font-size:7px; }
        .iamccs-v2vf-lane-select { position:absolute; left:6px; top:11px; width:14px; height:14px; min-height:14px!important; padding:0!important; border-radius:50%!important; }
        .iamccs-v2vf-lane-track { position:relative; min-width:0; overflow:hidden; }
        .iamccs-v2vf-strip { position:absolute; inset:5px; display:flex; overflow:hidden; border:1px dashed #46616d; background:#0b151b; }
        .iamccs-v2vf-lane.source .iamccs-v2vf-strip { bottom:25px; }
        .iamccs-v2vf-strip img,.iamccs-v2vf-strip video { display:block; height:100%; width:auto; min-width:72px; max-width:140px; object-fit:cover; border-right:1px solid #152a33; background:#03090d; }
        .iamccs-v2vf-strip-empty { margin:auto; color:#66808b; font-size:8px; }
        .iamccs-v2vf-segments { position:absolute; left:5px; right:5px; bottom:3px; height:19px; display:flex; gap:2px; }
        .iamccs-v2vf-segment { display:grid; place-items:center; min-width:30px; border:1px solid #6d5a24; background:#332b15; color:#f0cf72; font-size:7px; overflow:hidden; }
        .iamccs-v2vf-trim-handle { position:absolute; z-index:10; top:3px; bottom:3px; width:7px; background:var(--gold); cursor:ew-resize; }
        .iamccs-v2vf-trim-handle.out { transform:translateX(-7px); }
        .iamccs-v2vf-bottom { display:grid; grid-template-columns:repeat(4,1fr); border-top:1px solid var(--line); background:#0b171e; }
        .iamccs-v2vf-stat { display:flex; align-items:center; justify-content:center; gap:5px; border-right:1px solid #2e4752; color:var(--muted); font-size:8px; }
        .iamccs-v2vf-stat b { color:var(--text); }
        .iamccs-v2vf-form { display:grid; grid-template-columns:1fr 1fr; gap:7px; padding:8px; }
        .iamccs-v2vf-form label { display:flex; flex-direction:column; gap:4px; color:#9fb0b8; font-size:8px; font-weight:850; }
        .iamccs-v2vf-form .wide { grid-column:1/-1; }
        .iamccs-v2vf-toggle { grid-column:1/-1; display:flex!important; flex-direction:row!important; align-items:center; border:1px solid #3d5661; padding:8px; background:#0a141a; }
        .iamccs-v2vf-toggle input { width:16px; height:16px; accent-color:var(--cyan); }
        .iamccs-v2vf-control-actions { display:grid; grid-template-columns:1fr 1fr; gap:6px; padding:8px; border-top:1px solid var(--line); }
        .iamccs-v2vf-progress { margin:8px; border:1px solid #35505e; background:#081117; }
        .iamccs-v2vf-progress-head { display:flex; justify-content:space-between; padding:7px; color:var(--cyan); font-size:9px; font-weight:900; }
        .iamccs-v2vf-progress-track { height:7px; margin:0 7px 7px; background:#15242b; }
        .iamccs-v2vf-progress-fill { width:0; height:100%; background:var(--cyan); transition:width .15s linear; }
        .iamccs-v2vf-progress-copy { min-height:44px; padding:0 7px 7px; color:var(--muted); font-size:8px; line-height:1.45; }
        .iamccs-v2vf-preview-grid { display:grid; grid-template-columns:1fr 1fr; gap:6px; padding:8px; }
        .iamccs-v2vf-preview { border:1px solid #314955; background:#060d11; }
        .iamccs-v2vf-preview b { display:block; padding:5px; color:#8faeba; font-size:7px; }
        .iamccs-v2vf-preview .iamccs-v2vf-media { min-height:76px; aspect-ratio:16/10; }
        .iamccs-v2vf-models { grid-column:1/-1; border-top:1px solid #304954; margin-top:3px; padding-top:7px; }
        .iamccs-v2vf-model-row { display:grid; grid-template-columns:105px minmax(0,1fr); align-items:center; gap:6px; margin-bottom:5px; }
        .iamccs-v2vf-model-row span { color:#89a1ab; font-size:7px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .iamccs-v2vf-notice { margin:0 8px 8px; border:1px solid #5d4f29; background:#241f13; color:#e4c56e; padding:7px; font-size:8px; line-height:1.4; }
        .iamccs-v2vf-hidden-input { display:none; }
        .iamccs-v2vf-monitor:fullscreen { width:100vw; height:100vh; background:#000; }
        @media (max-width:1250px) { .iamccs-v2vf-shell { grid-template-columns:230px minmax(560px,1fr) 300px; } .iamccs-v2vf-header { grid-template-columns:220px 1fr auto; } }
    `;
    document.head.appendChild(style);
}

function placeholder(label, copy = "") {
    const box = document.createElement("div");
    box.className = "iamccs-v2vf-media-placeholder";
    box.innerHTML = `<b>${label}</b><span>${copy}</span>`;
    return box;
}

function preserveIntrinsicAspect(container, element) {
    const update = () => {
        const width = Number(element.videoWidth || element.naturalWidth || 0);
        const height = Number(element.videoHeight || element.naturalHeight || 0);
        if (!(width > 0 && height > 0)) return;
        element.style.setProperty("position", "absolute", "important");
        element.style.setProperty("inset", "0", "important");
        element.style.setProperty("width", "100%", "important");
        element.style.setProperty("height", "100%", "important");
        element.style.setProperty("min-width", "0", "important");
        element.style.setProperty("min-height", "0", "important");
        element.style.setProperty("max-width", "none", "important");
        element.style.setProperty("max-height", "none", "important");
        element.style.setProperty("aspect-ratio", "auto", "important");
        element.style.setProperty("object-fit", "contain", "important");
        element.style.setProperty("object-position", "center center", "important");
        container.dataset.mediaWidth = String(width);
        container.dataset.mediaHeight = String(height);
        container.dataset.mediaAspect = String(width / height);
    };
    element.addEventListener(element.tagName === "VIDEO" ? "loadedmetadata" : "load", update);
    if (element.tagName === "VIDEO") element.addEventListener("resize", update);
    requestAnimationFrame(update);
}

function installMedia(container, item, label = "PREVIEW") {
    if (!container) return null;
    container.replaceChildren();
    if (!item) {
        container.appendChild(placeholder(label, "Waiting for media"));
        return null;
    }
    const url = item.url || viewUrl(item);
    let element;
    if (mediaKind(item) === "video") {
        element = document.createElement("video");
        element.src = url;
        element.preload = "metadata";
        element.playsInline = true;
        element.controls = false;
        element.muted = false;
        element.volume = 1;
        const poster = posterUrl(item);
        if (poster) element.poster = poster;
    } else {
        element = document.createElement("img");
        element.src = url;
        element.alt = label;
    }
    container.appendChild(element);
    preserveIntrinsicAspect(container, element);
    return element;
}

function field(root, id) {
    return $ (root, `#${id}`);
}

function numberValue(root, id, fallback) {
    const value = Number(field(root, id)?.value);
    return Number.isFinite(value) ? value : fallback;
}

function frame41(value, minimum = 1) {
    const n = Math.max(minimum, Math.floor(Number(value) || minimum));
    return Math.max(minimum, Math.floor((n - 1) / 4) * 4 + 1);
}

function nodePayload(session) {
    const { root, node } = session;
    const duration = Math.max(.01, numberValue(root, "v2vfDuration", Number(read(node, "duration_seconds", 2.7))));
    const fps = Math.max(1, numberValue(root, "v2vfFps", Number(read(node, "fps", 30))));
    const trimIn = Math.max(0, Math.min(duration, session.trimIn));
    const trimOut = Math.max(trimIn + .01, Math.min(duration, session.trimOut));
    const chunk = frame41(numberValue(root, "v2vfChunk", 81), 9);
    return {
        source_video_path: String(field(root, "v2vfVideoPath")?.value || "").trim(),
        source_image_path: String(field(root, "v2vfImagePath")?.value || "").trim(),
        duration_seconds: duration,
        fps,
        trim_start_s: trimIn,
        trim_end_s: trimOut,
        frame_load_cap: Math.max(1, Math.round((trimOut - trimIn) * fps)),
        generation_width: Math.max(16, Math.floor(numberValue(root, "v2vfWidth", 832) / 16) * 16),
        generation_height: Math.max(16, Math.floor(numberValue(root, "v2vfHeight", 480) / 16) * 16),
        chunk_frames: chunk,
        overlap_frames: 1,
        generation_steps: Math.max(1, Math.floor(numberValue(root, "v2vfSteps", 6))),
        generation_cfg: Math.max(0, numberValue(root, "v2vfCfg", 1)),
        generation_seed: Math.max(0, Math.floor(numberValue(root, "v2vfSeed", 0))),
        seed_mode: String(field(root, "v2vfSeedMode")?.value || "fixed"),
        reference_image_strength: Math.max(0, numberValue(root, "v2vfReferenceStrength", 1)),
        pose_strength: Math.max(0, numberValue(root, "v2vfPoseStrength", 1)),
        pose_start_percent: Math.max(0, Math.min(1, numberValue(root, "v2vfPoseStart", 0))),
        pose_end_percent: Math.max(0, Math.min(1, numberValue(root, "v2vfPoseEnd", 1))),
        enable_context_windows: Boolean(field(root, "v2vfContext")?.checked),
        context_length_latents: Math.max(1, Math.floor(numberValue(root, "v2vfContextLength", 21))),
        context_overlap_latents: Math.max(0, Math.floor(numberValue(root, "v2vfContextOverlap", 8))),
        context_schedule: String(field(root, "v2vfContextSchedule")?.value || "standard_static"),
        context_fuse_method: String(field(root, "v2vfContextFuse")?.value || "pyramid"),
        enable_pose_cache: Boolean(field(root, "v2vfPoseCache")?.checked),
        cache_device: String(field(root, "v2vfCacheDevice")?.value || "cpu"),
        cache_dtype: String(field(root, "v2vfCacheDtype")?.value || "int8"),
        reference_background_mode: String(field(root, "v2vfReferenceBackground")?.value || "keep_reference_background"),
        output_background_mode: String(field(root, "v2vfOutputBackground")?.value || "native_generated"),
        live_chunk_preview: String(field(root, "v2vfLivePreview")?.value || "middle_frame"),
        empty_cache_each_chunk: Boolean(field(root, "v2vfEmptyCache")?.checked),
        lora_name: String(field(root, "v2vfLora")?.value || "auto / graph"),
        apply_distill_lora: Boolean(field(root, "v2vfApplyLora")?.checked),
        lora_strength: numberValue(root, "v2vfLoraStrength", 1),
        global_prompt: String(field(root, "v2vfPrompt")?.value || ""),
        pose_prompt: String(field(root, "v2vfPosePrompt")?.value || ""),
        negative_prompt: String(field(root, "v2vfNegative")?.value || ""),
        output_prefix: String(field(root, "v2vfPrefix")?.value || "IAMCCS/WAN_ANIMATE_2_EASY"),
        preview_stage: session.activeLane,
    };
}

function commit(session) {
    const payload = nodePayload(session);
    for (const [name, value] of Object.entries(payload)) write(session.node, name, value);
    const timeline = {
        schema: "iamccs.v2v.shotboard.easy.wananimate2",
        source_video_path: payload.source_video_path,
        source_image_path: payload.source_image_path,
        duration_seconds: payload.trim_end_s - payload.trim_start_s,
        source_duration_seconds: payload.duration_seconds,
        fps: payload.fps,
        trim_start_s: payload.trim_start_s,
        trim_end_s: payload.trim_end_s,
        frame_load_cap: payload.frame_load_cap,
        backend_mode: "wananimate2_native_extender",
        backend_profile: "wananimate2_gguf_easy",
        chunk_length_frames: payload.chunk_frames,
        chunk_overlap_frames: 1,
        reference_background_mode: payload.reference_background_mode,
        output_background_mode: payload.output_background_mode,
    };
    write(session.node, "timeline_data", JSON.stringify(timeline));
    return payload;
}

function backendSummary(session) {
    const workflowNodes = graphNodes(session.node);
    const types = workflowNodes.map(nodeClass);
    const required = [
        "IAMCCS_WanAnimate2Extends",
        "LoadVideo",
        "Video Slice",
        "GetVideoComponents",
        "LoadImage",
        "UnetLoaderGGUF",
        "CreateVideo",
        "SaveVideo",
    ];
    const missing = required.filter((type) => !types.includes(type));
    const text = missing.length
        ? `Missing backend: ${missing.join(", ")}`
        : "WAN ANIMATE 2 native/GGUF backend ready / source FPS and audio linked / IAMCCS extender detected";
    session.backendMissing = missing;
    session.backendIdentityMode = missing.length ? null : "wananimate2";
    session.backendUnified = !missing.length;
    session.backendMismatch = false;
    const notice = field(session.root, "v2vfBackendNotice");
    if (notice) notice.textContent = text;
}

async function loadUnifiedBackend(session, payload) {
    setProgress(session, "LOADING", 0, 1, "Loading the installed Wan-Animate-2 GGUF backend...");
    const response = await api.fetchApi(`/userdata/${encodeURIComponent(UNIFIED_WORKFLOW_PATH)}`);
    if (!response.ok) throw new Error(`Unified Easy workflow is not installed (${response.status})`);
    const workflow = await response.json();
    if (!workflow || !Array.isArray(workflow.nodes)) throw new Error("Installed Wan-Animate-2 Easy workflow is invalid");
    if (typeof app.loadGraphData !== "function") throw new Error("ComfyUI workflow loader API is unavailable");

    await app.loadGraphData(workflow, true, true, UNIFIED_WORKFLOW_NAME);
    await new Promise((resolve) => window.setTimeout(resolve, 50));
    const planner = graphNodes().find((item) => isPlannerNode(item));
    if (!planner) throw new Error("Wan-Animate-2 Easy workflow loaded without its planner node");

    for (const [name, value] of Object.entries(payload)) write(planner, name, value);
    openShotboard(planner);
    const loadedSession = runtime.session;
    if (!loadedSession || loadedSession.node !== planner) throw new Error("Unable to reopen the unified Easy planner");
    commit(loadedSession);
    backendSummary(loadedSession);
    return loadedSession;
}

function syncBackend(session, payload) {
    let touched = 0;
    let positivePromptTargets = 0;
    let negativePromptTargets = 0;
    let posePromptTargets = 0;
    let outputPrefixTargets = 0;
    let fpsTargets = 0;
    for (const target of nodesByType(session.node, ["LoadVideo"]).slice(0, 1)) {
        touched += Number(writeAny(target, ["file", "video"], payload.source_video_path));
    }
    for (const target of nodesByType(session.node, ["Video Slice"]).slice(0, 1)) {
        touched += Number(writeAny(target, ["start_time"], payload.trim_start_s));
        touched += Number(writeAny(target, ["duration"], payload.trim_end_s - payload.trim_start_s));
        touched += Number(writeAny(target, ["strict_duration"], false));
    }
    const imageNodes = nodesByType(session.node, ["LoadImage"]);
    if (imageNodes[0]) touched += Number(writeAny(imageNodes[0], ["image", "image_upload"], payload.source_image_path));
    for (const target of nodesByType(session.node, ["IAMCCS_WanAnimate2Extends"])) {
        touched += Number(writeAny(target, ["width"], payload.generation_width));
        touched += Number(writeAny(target, ["height"], payload.generation_height));
        touched += Number(writeAny(target, ["chunk_length"], payload.chunk_frames));
        touched += Number(writeAny(target, ["cfg"], payload.generation_cfg));
        touched += Number(writeAny(target, ["noise_seed", "seed"], payload.generation_seed));
        touched += Number(writeAny(target, ["seed_mode"], payload.seed_mode));
        touched += Number(writeAny(target, ["reference_image_strength"], payload.reference_image_strength));
        touched += Number(writeAny(target, ["pose_strength"], payload.pose_strength));
        touched += Number(writeAny(target, ["pose_start_percent"], payload.pose_start_percent));
        touched += Number(writeAny(target, ["pose_end_percent"], payload.pose_end_percent));
        touched += Number(writeAny(target, ["enable_context_windows"], payload.enable_context_windows));
        touched += Number(writeAny(target, ["context_length_latents"], payload.context_length_latents));
        touched += Number(writeAny(target, ["context_overlap_latents"], payload.context_overlap_latents));
        touched += Number(writeAny(target, ["context_schedule"], payload.context_schedule));
        touched += Number(writeAny(target, ["context_fuse_method"], payload.context_fuse_method));
        touched += Number(writeAny(target, ["enable_pose_cache"], payload.enable_pose_cache));
        touched += Number(writeAny(target, ["cache_device"], payload.cache_device));
        touched += Number(writeAny(target, ["cache_dtype"], payload.cache_dtype));
        touched += Number(writeAny(target, ["reference_background_mode"], payload.reference_background_mode));
        touched += Number(writeAny(target, ["output_background_mode"], payload.output_background_mode));
        touched += Number(writeAny(target, ["live_chunk_preview"], payload.live_chunk_preview));
        touched += Number(writeAny(target, ["empty_cache_each_chunk"], payload.empty_cache_each_chunk));
    }
    for (const target of nodesByType(session.node, ["BasicScheduler"])) touched += Number(writeAny(target, ["steps"], payload.generation_steps));
    for (const target of nodesByType(session.node, ["LoraLoaderModelOnly"])) {
        if (payload.lora_name && payload.lora_name !== "auto / graph") touched += Number(writeAny(target, ["lora_name"], payload.lora_name));
        touched += Number(writeAny(target, ["strength_model", "strength"], payload.apply_distill_lora ? payload.lora_strength : 0));
    }
    for (const target of graphNodes(session.node)) {
        const type = nodeClass(target);
        const title = String(target.title || "").toLowerCase();
        if ((type === "CLIPTextEncode" || type === "CLIPTextEncodeFlux") && title.includes("negative")) {
            negativePromptTargets++;
            touched += Number(writeAny(target, ["text"], payload.negative_prompt));
        } else if ((type === "CLIPTextEncode" || type === "CLIPTextEncodeFlux") && /pose|motion|driving/.test(title)) {
            posePromptTargets++;
            touched += Number(writeAny(target, ["text"], payload.pose_prompt));
        } else if (type === "CLIPTextEncode" || type === "CLIPTextEncodeFlux") {
            positivePromptTargets++;
            touched += Number(writeAny(target, ["text"], payload.global_prompt));
        }
        if (type === "PreviewAnimation" && writeAny(target, ["fps"], payload.fps)) {
            fpsTargets++;
            touched++;
        }
        if (type === "SaveVideo" && writeAny(target, ["filename_prefix"], payload.output_prefix)) {
            outputPrefixTargets++;
            touched++;
        }
    }
    session.backendSync = { positivePromptTargets, negativePromptTargets, posePromptTargets, outputPrefixTargets, fpsTargets, touched };
    return touched;
}

function syncMediaLoaders(session, payload) {
    let touched = 0;
    const video = nodesByType(session.node, ["LoadVideo"])[0];
    const image = nodesByType(session.node, ["LoadImage"])[0];
    if (video && payload.source_video_path) touched += Number(writeAny(video, ["file", "video"], payload.source_video_path));
    if (image && payload.source_image_path) touched += Number(writeAny(image, ["image", "image_upload"], payload.source_image_path));
    return touched;
}

function inputLinked(target, name) {
    return Boolean((target?.inputs || []).find((item) => item?.name === name)?.link !== null
        && (target?.inputs || []).find((item) => item?.name === name)?.link !== undefined);
}

function auditBackendTruth(session, payload) {
    const issues = [];
    const video = nodesByType(session.node, ["LoadVideo"])[0];
    if (!video) issues.push("native source video loader missing");
    else expectWidget(issues, video, ["file", "video"], payload.source_video_path, "source video path");

    const slice = nodesByType(session.node, ["Video Slice"])[0];
    if (!slice) issues.push("native Video Slice missing");
    else {
        expectWidget(issues, slice, ["start_time"], payload.trim_start_s, "trim start");
        expectWidget(issues, slice, ["duration"], payload.trim_end_s - payload.trim_start_s, "trim duration");
        if (!linkedFrom(slice, "video", "LoadVideo")) issues.push("LoadVideo to Video Slice link missing");
    }

    const components = nodesByType(session.node, ["GetVideoComponents"])[0];
    if (!components || !linkedFrom(components, "video", "Video Slice")) issues.push("trimmed video components link missing");

    const reference = nodesByType(session.node, ["LoadImage"])[0];
    if (!reference) issues.push("reference image loader missing");
    else expectWidget(issues, reference, ["image", "image_upload"], payload.source_image_path, "reference image path");

    const extenders = nodesByType(session.node, ["IAMCCS_WanAnimate2Extends"]);
    if (!extenders.length) issues.push("Wan-Animate-2 extender missing");
    for (const target of extenders) {
        for (const name of ["model", "positive", "negative", "vae", "sampler", "sigmas", "reference_image", "pose_video", "target_frames", "clip_vision_output", "positive_pose", "clip_vision_output_pose"]) {
            if (!inputLinked(target, name)) issues.push(`extender ${name} link missing`);
        }
        expectWidget(issues, target, ["width"], payload.generation_width, "generation width");
        expectWidget(issues, target, ["height"], payload.generation_height, "generation height");
        expectWidget(issues, target, ["cfg"], payload.generation_cfg, "CFG");
        expectWidget(issues, target, ["noise_seed", "seed"], payload.generation_seed, "seed");
        expectWidget(issues, target, ["chunk_length"], payload.chunk_frames, "chunk frames");
        expectWidget(issues, target, ["seed_mode"], payload.seed_mode, "seed mode");
        expectWidget(issues, target, ["reference_image_strength"], payload.reference_image_strength, "reference strength");
        expectWidget(issues, target, ["pose_strength"], payload.pose_strength, "pose strength");
        expectWidget(issues, target, ["pose_start_percent"], payload.pose_start_percent, "pose start");
        expectWidget(issues, target, ["pose_end_percent"], payload.pose_end_percent, "pose end");
        expectWidget(issues, target, ["enable_context_windows"], payload.enable_context_windows, "context windows");
        expectWidget(issues, target, ["context_length_latents"], payload.context_length_latents, "context length");
        expectWidget(issues, target, ["context_overlap_latents"], payload.context_overlap_latents, "context overlap");
        expectWidget(issues, target, ["context_schedule"], payload.context_schedule, "context schedule");
        expectWidget(issues, target, ["context_fuse_method"], payload.context_fuse_method, "context fuse");
        expectWidget(issues, target, ["enable_pose_cache"], payload.enable_pose_cache, "pose cache");
        expectWidget(issues, target, ["cache_device"], payload.cache_device, "cache device");
        expectWidget(issues, target, ["cache_dtype"], payload.cache_dtype, "cache dtype");
        expectWidget(issues, target, ["reference_background_mode"], payload.reference_background_mode, "reference background mode");
        expectWidget(issues, target, ["output_background_mode"], payload.output_background_mode, "output background mode");
        if (payload.reference_background_mode === "isolate_character" && !inputLinked(target, "reference_character_mask")) {
            issues.push("isolate_character requires a reference_character_mask backend link");
        }
        if (payload.output_background_mode !== "native_generated" && !inputLinked(target, "composite_mask")) {
            issues.push(`${payload.output_background_mode} requires a composite_mask backend link`);
        }
    }

    const promptNodes = graphNodes(session.node).filter((target) => ["CLIPTextEncode", "CLIPTextEncodeFlux"].includes(nodeClass(target)));
    const positive = promptNodes.filter((target) => !/negative|pose|motion|driving/.test(String(target.title || "").toLowerCase()));
    const negative = promptNodes.filter((target) => String(target.title || "").toLowerCase().includes("negative"));
    const pose = promptNodes.filter((target) => /pose|motion|driving/.test(String(target.title || "").toLowerCase()));
    if (!positive.length) issues.push("positive prompt node missing");
    if (!negative.length) issues.push("negative prompt node missing");
    if (!pose.length) issues.push("pose prompt node missing");
    for (const target of positive) expectWidget(issues, target, ["text"], payload.global_prompt, "positive prompt");
    for (const target of negative) expectWidget(issues, target, ["text"], payload.negative_prompt, "negative prompt");
    for (const target of pose) expectWidget(issues, target, ["text"], payload.pose_prompt, "pose prompt");

    const schedulers = nodesByType(session.node, ["BasicScheduler"]);
    if (!schedulers.length) issues.push("scheduler missing");
    for (const target of schedulers) expectWidget(issues, target, ["steps"], payload.generation_steps, "generation steps");

    for (const target of nodesByType(session.node, ["LoraLoaderModelOnly"])) {
        if (payload.lora_name && payload.lora_name !== "auto / graph") expectWidget(issues, target, ["lora_name"], payload.lora_name, "LoRA model");
        expectWidget(issues, target, ["strength_model", "strength"], payload.apply_distill_lora ? payload.lora_strength : 0, "LoRA strength");
    }
    const createVideo = nodesByType(session.node, ["CreateVideo"])[0];
    if (!createVideo) issues.push("CreateVideo missing");
    else {
        if (!linkedFrom(createVideo, "images", "IAMCCS_WanAnimate2Extends")) issues.push("extender images to CreateVideo link missing");
        if (!linkedFrom(createVideo, "audio", "GetVideoComponents")) issues.push("source audio to final video link missing");
        if (!linkedFrom(createVideo, "fps", "GetVideoComponents")) issues.push("source FPS to final video link missing");
    }
    const save = nodesByType(session.node, ["SaveVideo"])[0];
    if (!save || !linkedFrom(save, "video", "CreateVideo")) issues.push("CreateVideo to SaveVideo link missing");
    else expectWidget(issues, save, ["filename_prefix"], payload.output_prefix, "output prefix");
    if ((session.backendSync?.outputPrefixTargets || 0) < 1) issues.push("final output prefix was not mapped");
    return [...new Set(issues)];
}

function renderModelDeck(session) {
    const host = field(session.root, "v2vfModels");
    if (!host) return;
    host.replaceChildren();
    const candidates = graphNodes(session.node).filter((target) => {
        if (target === session.node) return false;
        const descriptor = `${nodeClass(target)} ${target.title || ""} ${(target.widgets || []).map((item) => item?.name || "").join(" ")}`;
        if (/rvc|voice|hubert|rmvpe|audio model/i.test(descriptor)) return false;
        return (target.widgets || []).some((item) => /(^|_)(model|unet|vae|clip|lora)(_|$)|model_name|lora_name/i.test(String(item.name || "")));
    });
    for (const target of candidates.slice(0, 8)) {
        const modelWidget = (target.widgets || []).find((item) => /(^|_)(model|unet|vae|clip|lora)(_|$)|model_name|lora_name/i.test(String(item.name || "")) && item.type !== "button");
        if (!modelWidget) continue;
        const row = document.createElement("div");
        row.className = "iamccs-v2vf-model-row";
        const label = document.createElement("span");
        label.textContent = String(target.title || nodeClass(target));
        label.title = label.textContent;
        let input;
        const values = Array.isArray(modelWidget.options?.values) ? modelWidget.options.values : [];
        if (values.length) {
            input = document.createElement("select");
            for (const value of values) input.appendChild(new Option(String(value), String(value)));
            input.value = String(modelWidget.value ?? values[0]);
        } else {
            input = document.createElement("input");
            input.value = String(modelWidget.value ?? "");
        }
        input.addEventListener("change", () => {
            modelWidget.value = input.value;
            try { modelWidget.callback?.(input.value, null, target); } catch {}
            target.setDirtyCanvas?.(true, true);
            const plannerField = {
                UnetLoaderGGUF: "model_name",
                LoraLoaderModelOnly: "lora_name",
                CLIPLoader: "clip_name",
                CLIPVisionLoader: "clip_vision_name",
                VAELoader: "vae_name",
            }[nodeClass(target)];
            if (plannerField) write(session.node, plannerField, input.value);
        });
        row.append(label, input);
        host.appendChild(row);
    }
    if (!host.children.length) host.textContent = "No editable model widgets detected in the current graph.";
}

function updateStats(session) {
    const payload = nodePayload(session);
    const duration = payload.trim_end_s - payload.trim_start_s;
    const set = (id, value) => { const item = field(session.root, id); if (item) item.textContent = String(value); };
    set("v2vfStatIn", time(payload.trim_start_s));
    set("v2vfStatOut", time(payload.trim_end_s));
    set("v2vfStatDuration", time(duration));
    set("v2vfStatFrames", `${payload.frame_load_cap} FR @ ${Number(payload.fps.toFixed(3))}`);
    field(session.root, "v2vfFrameCap").value = String(payload.frame_load_cap);
}

function time(seconds) {
    const value = Math.max(0, Number(seconds) || 0);
    const minutes = Math.floor(value / 60);
    const whole = Math.floor(value % 60);
    const ms = Math.floor((value - Math.floor(value)) * 1000);
    return `${String(minutes).padStart(2, "0")}:${String(whole).padStart(2, "0")}.${String(ms).padStart(3, "0")}`;
}

function renderTimeline(session) {
    const { root } = session;
    const payload = nodePayload(session);
    session.trimIn = payload.trim_start_s;
    session.trimOut = payload.trim_end_s;
    const ruler = field(root, "v2vfRuler");
    if (ruler) {
        ruler.querySelectorAll(".iamccs-v2vf-tick").forEach((item) => item.remove());
        for (let i = 0; i <= 10; i++) {
            const tick = document.createElement("span");
            tick.className = "iamccs-v2vf-tick";
            tick.style.left = `${i * 10}%`;
            tick.textContent = time(payload.duration_seconds * i / 10);
            ruler.appendChild(tick);
        }
    }
    updatePlayheadVisual(session);
    const sourceTrack = field(root, "v2vfSourceTrack");
    const left = payload.duration_seconds ? payload.trim_start_s / payload.duration_seconds * 100 : 0;
    const width = payload.duration_seconds ? (payload.trim_end_s - payload.trim_start_s) / payload.duration_seconds * 100 : 100;
    const sourceStrip = field(root, "v2vfStrip-source");
    if (sourceStrip) {
        sourceStrip.style.left = `${left}%`;
        sourceStrip.style.width = `${width}%`;
    }
    const trimIn = field(root, "v2vfTrimIn");
    const trimOut = field(root, "v2vfTrimOut");
    if (trimIn) trimIn.style.left = `${left}%`;
    if (trimOut) trimOut.style.left = `${left + width}%`;
    const segments = field(root, "v2vfSegments");
    if (!segments || !sourceTrack) return;
    segments.replaceChildren();
    const step = Math.max(1, payload.chunk_frames - payload.overlap_frames);
    const count = Math.max(1, Math.ceil((payload.frame_load_cap - payload.chunk_frames) / step) + 1);
    for (let index = 0; index < count; index++) {
        const part = document.createElement("div");
        part.className = "iamccs-v2vf-segment";
        part.style.flex = String(index === count - 1 ? Math.max(1, payload.frame_load_cap - index * step) : payload.chunk_frames);
        part.textContent = `S${String(index + 1).padStart(2, "0")}`;
        segments.appendChild(part);
    }
    if (sourceTrack) sourceTrack.title = `${payload.frame_load_cap} selected source frames`;
    updateStats(session);
    commit(session);
}

function renderStrip(session, lane, item) {
    const host = field(session.root, `v2vfStrip-${lane}`);
    if (!host) return;
    host.replaceChildren();
    if (!item) {
        const empty = document.createElement("span");
        empty.className = "iamccs-v2vf-strip-empty";
        empty.textContent = `AWAITING ${lane.toUpperCase()} PREVIEW`;
        host.appendChild(empty);
        return;
    }
    const url = item.url || viewUrl(item);
    const poster = posterUrl(item);
    for (let i = 0; i < 12; i++) {
        if (poster || mediaKind(item) !== "video") {
            const image = document.createElement("img");
            image.src = poster || url;
            image.alt = lane;
            host.appendChild(image);
        } else {
            const video = document.createElement("video");
            video.src = url;
            video.preload = "metadata";
            video.muted = true;
            video.playsInline = true;
            host.appendChild(video);
        }
    }
}

function lanePriority(node, item, lane) {
    const text = `${nodeClass(node)} ${node?.title || ""} ${item?.filename || ""}`.toLowerCase();
    const frames = Math.max(0, Number(item?.frameCount || 0));
    let score = 0;
    if (mediaKind(item) === "video") score += 300;
    if (item?.animated) score += 220;
    if (frames > 1) score += Math.min(180, Math.round(Math.log2(frames) * 24));
    if (String(item?.type || "").toLowerCase() === "output") score += 160;
    if (lane === "output") {
        if (/32\s*fps|32fps/.test(text)) score += 1000;
        else if (/final/.test(text)) score += 850;
        else if (/16\s*fps|16fps/.test(text)) score += 700;
        if (/preview|temp/.test(text)) score -= 120;
    }
    if (lane === "pose") {
        if (/active mode pose mask|pose video mask|pose[_ -]?mask/.test(text)) score += 1400;
        if (/object ids preview|sam3[_ -]?trackpreview|sam tracking/.test(text)) score -= 900;
    }
    if (lane === "mask") {
        if (/active mode reference mask|reference image mask|reference[_ -]?mask/.test(text)) score += 1500;
        else if (/colored mask|identity mask/.test(text)) score += 900;
        if (/object ids preview|sam3[_ -]?trackpreview|sam tracking/.test(text)) score -= 900;
    }
    if (lane === "intermediate" && /object ids preview|sam3[_ -]?trackpreview|sam tracking/.test(text)) score += 1000;
    if (lane === "intermediate" && frames === 1 && !item?.animated) score -= 500;
    return score;
}

function updateLane(session, lane, item, node = null, force = false) {
    const priority = item ? lanePriority(node, item, lane) : Number.NEGATIVE_INFINITY;
    const currentPriority = Number(session.mediaPriority?.[lane] ?? Number.NEGATIVE_INFINITY);
    if (!force && item && priority < currentPriority) return false;
    if (session.mediaPriority) session.mediaPriority[lane] = priority;
    session.media[lane] = item;
    renderStrip(session, lane, item);
    const preview = field(session.root, `v2vfPreview-${lane}`);
    if (preview) installMedia(preview, item, lane.toUpperCase());
    if (lane === "output") installMedia(field(session.root, "v2vfGenerated"), item, "GENERATED OUTPUT");
    if (session.activeLane === lane) renderProgram(session);
    return true;
}

function renderProgram(session) {
    const item = session.activeLane === "source" ? session.media.source : session.media[session.activeLane];
    session.programElement = installMedia(field(session.root, "v2vfProgram"), item, session.activeLane.toUpperCase());
    const label = field(session.root, "v2vfProgramLabel");
    if (label) label.textContent = `SELECTED: ${session.activeLane.toUpperCase()}`;
}

function selectLane(session, lane) {
    session.activeLane = lane;
    write(session.node, "preview_stage", lane);
    $$(session.root, ".iamccs-v2vf-lane-select").forEach((button) => button.classList.toggle("active", button.dataset.lane === lane));
    renderProgram(session);
}

function selectedVideo(session, monitor) {
    const container = field(session.root, monitor === "source" ? "v2vfSource" : "v2vfProgram");
    return $(container, "video");
}

function timelineDuration(session) {
    return Math.max(.01, numberValue(session.root, "v2vfDuration", 2.7));
}

function monitorOffset(session, monitor) {
    return monitor === "program" && session.activeLane !== "source" ? session.trimIn : 0;
}

function mediaTimeForPlayhead(session, monitor, video) {
    const requested = session.playhead - monitorOffset(session, monitor);
    const maximum = Number.isFinite(video?.duration) ? video.duration : timelineDuration(session);
    return Math.max(0, Math.min(maximum, requested));
}

function updatePlayheadVisual(session) {
    const duration = timelineDuration(session);
    const ratio = Math.max(0, Math.min(1, session.playhead / duration));
    const playhead = field(session.root, "v2vfPlayhead");
    if (playhead) playhead.style.left = `${ratio * 100}%`;
    const sourceTime = field(session.root, "v2vfSourceTime");
    if (sourceTime) sourceTime.textContent = time(session.playhead);
    const programTime = field(session.root, "v2vfProgramTime");
    if (programTime) programTime.textContent = time(Math.max(0, session.playhead - monitorOffset(session, "program")));
}

function syncPlayheadMedia(session) {
    for (const monitor of ["source", "program"]) {
        const video = selectedVideo(session, monitor);
        if (!video) continue;
        const nextTime = mediaTimeForPlayhead(session, monitor, video);
        try {
            if (Math.abs(Number(video.currentTime || 0) - nextTime) > .004) video.currentTime = nextTime;
        } catch (_) {
            // Metadata may still be loading; the next pointer move will retry the seek.
        }
    }
}

function setPlayhead(session, seconds, seekMedia = true) {
    session.playhead = Math.max(0, Math.min(timelineDuration(session), Number(seconds) || 0));
    updatePlayheadVisual(session);
    if (seekMedia) syncPlayheadMedia(session);
}

function bindPlaybackClock(session, video, monitor) {
    if (video.__iamccsPlaybackClock) video.removeEventListener("timeupdate", video.__iamccsPlaybackClock);
    const update = () => {
        if (session.scrubbing) return;
        session.playhead = Math.max(0, Math.min(timelineDuration(session), monitorOffset(session, monitor) + Number(video.currentTime || 0)));
        updatePlayheadVisual(session);
    };
    video.__iamccsPlaybackClock = update;
    video.addEventListener("timeupdate", update);
}

function play(session, monitor = "program") {
    const video = selectedVideo(session, monitor);
    if (!video) return;
    let startTime = mediaTimeForPlayhead(session, monitor, video);
    const fps = Math.max(1, numberValue(session.root, "v2vfFps", 30));
    const atEnd = Number.isFinite(video.duration) && startTime >= Math.max(0, video.duration - 1 / fps);
    if (video.ended || atEnd) {
        setPlayhead(session, monitorOffset(session, monitor), false);
        startTime = 0;
    }
    video.currentTime = startTime;
    bindPlaybackClock(session, video, monitor);
    video.play().catch(() => {});
}

function stop(session, monitor = "program") {
    const video = selectedVideo(session, monitor);
    if (video) video.pause();
}

function step(session, direction) {
    const fps = Math.max(1, numberValue(session.root, "v2vfFps", 30));
    setPlayhead(session, Math.max(session.trimIn, Math.min(session.trimOut, session.playhead + direction / fps)));
}

function setProgress(session, phase, value = 0, max = 1, copy = "") {
    const percent = phase === "COMPLETE" ? 100 : Math.max(0, Math.min(100, Math.round(Number(value || 0) / Math.max(1, Number(max || 1)) * 100)));
    field(session.root, "v2vfProgressPhase").textContent = phase;
    field(session.root, "v2vfProgressPercent").textContent = `${percent}%`;
    field(session.root, "v2vfProgressFill").style.width = `${percent}%`;
    field(session.root, "v2vfProgressCopy").textContent = copy || "Waiting for ComfyUI";
    const running = !["IDLE", "COMPLETE", "ERROR", "STOPPED"].includes(phase);
    field(session.root, "v2vfRender").disabled = running;
    field(session.root, "v2vfStop").disabled = !running;
}

function itemsFromExecuted(detail) {
    const output = detail?.output || detail?.detail?.output || {};
    const items = [];
    for (const key of ["images", "gifs", "videos"]) {
        const rawItems = Array.isArray(output?.[key]) ? output[key] : [];
        for (let index = 0; index < rawItems.length; index++) {
            const raw = rawItems[index];
            const description = String(output?.text?.[index] ?? output?.text?.[0] ?? "");
            const frameMatch = description.match(/(?:^|\s)(\d+)x\d+x\d+/i);
            items.push({
                ...raw,
                kind: mediaKind(raw),
                animated: Boolean(output?.animated?.[index]),
                frameCount: frameMatch ? Number(frameMatch[1]) : 0,
                cacheKey: Date.now(),
            });
        }
    }
    return items;
}

function classifyOutput(node, item) {
    const text = `${nodeClass(node)} ${node?.title || ""} ${item?.filename || ""}`.toLowerCase();
    if (/live chunk|chunk preview|wan animate 2 preview/.test(text)) return "intermediate";
    if (/object ids preview|sam3[_ -]?trackpreview|sam tracking/.test(text)) return "intermediate";
    if (/reference image mask|reference[_ -]?mask/.test(text)) return "mask";
    if (/pose video mask|pose[_ -]?mask/.test(text)) return "pose";
    if (/pose|dwpose|openpose|skeleton|vitpose/.test(text)) return "pose";
    if (/sam|mask|segment|track/.test(text)) return "mask";
    if (/final|output|combine|savevideo|video combine/.test(text)) return "output";
    return "intermediate";
}

function historyTimestamp(entry) {
    const messages = Array.isArray(entry?.status?.messages) ? entry.status.messages : [];
    for (let index = messages.length - 1; index >= 0; index--) {
        const timestamp = Number(messages[index]?.[1]?.timestamp || 0);
        if (timestamp) return timestamp;
    }
    return 0;
}

async function hydrateLatestOutputs(session) {
    try {
        const response = await api.fetchApi("/history?max_items=20");
        if (!response.ok) return;
        const history = await response.json();
        const nodes = new Map(graphNodes(session.node).map((node) => [String(node.id), node]));
        const entries = Object.values(history || {}).sort((a, b) => historyTimestamp(b) - historyTimestamp(a));
        for (const entry of entries) {
            if (entry?.status?.status_str && entry.status.status_str !== "success") continue;
            const best = new Map();
            let matched = false;
            for (const [nodeId, output] of Object.entries(entry?.outputs || {})) {
                const target = nodes.get(String(nodeId));
                if (!target) continue;
                matched = true;
                for (const item of itemsFromExecuted({ output })) {
                    const laneId = classifyOutput(target, item);
                    const priority = lanePriority(target, item, laneId);
                    if (!best.has(laneId) || priority >= best.get(laneId).priority) best.set(laneId, { item, target, priority });
                }
            }
            if (!matched || !best.size) continue;
            for (const [laneId, candidate] of best) updateLane(session, laneId, candidate.item, candidate.target);
            if (session.media.output) {
                selectLane(session, "output");
                setPlayhead(session, session.trimIn);
            }
            setProgress(session, "IDLE", 0, 1, session.media.output ? "Latest complete backend video loaded" : "Latest backend previews loaded");
            return;
        }
    } catch (error) {
        console.warn("IAMCCS V2V: unable to restore latest backend previews", error);
    }
}

function queueRowValue(row, index, key) {
    return Array.isArray(row) ? row[index] : row?.[key];
}

async function hydrateQueueState(session) {
    try {
        const response = await api.fetchApi("/queue");
        if (!response.ok || runtime.session !== session) return false;
        const queue = await response.json();
        const running = Array.isArray(queue?.queue_running) ? queue.queue_running[0] : null;
        const pending = Array.isArray(queue?.queue_pending) ? queue.queue_pending[0] : null;
        const row = running || pending;
        if (!row) return false;
        const promptId = String(queueRowValue(row, 1, "prompt_id") || "");
        const prompt = queueRowValue(row, 2, "prompt") || {};
        const extra = queueRowValue(row, 3, "extra_data") || {};
        const createTime = Number(extra?.create_time || 0);
        const elapsedSeconds = createTime > 0 ? Math.max(0, (Date.now() - createTime) / 1000) : 0;
        const elapsed = elapsedSeconds >= 60
            ? `${Math.floor(elapsedSeconds / 60)}m ${Math.floor(elapsedSeconds % 60)}s`
            : `${Math.floor(elapsedSeconds)}s`;
        const isShotboard = Object.values(prompt).some((item) => item?.class_type === NODE_CLASS);
        session.queueWasBusy = true;
        session.activePromptId = promptId;
        const phase = running ? "RUNNING" : "QUEUED";
        const position = running ? "processing" : "waiting in queue";
        setProgress(session, phase, session.progressValue, session.progressMax, `${isShotboard ? "Shotboard" : "ComfyUI"} prompt ${promptId.slice(0, 8)} ${position} / elapsed ${elapsed}`);
        return true;
    } catch (error) {
        console.warn("IAMCCS V2V: unable to recover active queue state", error);
        return false;
    }
}

function startQueueMonitor(session) {
    const poll = async () => {
        if (runtime.session !== session) return;
        const wasBusy = Boolean(session.queueWasBusy);
        const busy = await hydrateQueueState(session);
        if (!busy && wasBusy && runtime.session === session) {
            session.queueWasBusy = false;
            session.activePromptId = "";
            await hydrateLatestOutputs(session);
        }
        if (runtime.session === session) session.queueTimer = window.setTimeout(poll, 2500);
    };
    void poll();
}

function bindEvents() {
    if (runtime.eventsBound) return;
    runtime.eventsBound = true;
    api.addEventListener("execution_start", () => {
        if (runtime.session) {
            runtime.session.queueWasBusy = true;
            runtime.session.executionFailed = false;
            for (const laneId of ["pose", "mask", "intermediate", "output"]) runtime.session.mediaPriority[laneId] = Number.NEGATIVE_INFINITY;
            setProgress(runtime.session, "STARTING", 0, 1, "ComfyUI accepted the workflow");
        }
    });
    api.addEventListener("executing", (event) => {
        const session = runtime.session;
        if (!session) return;
        const nodeId = String(event?.detail ?? "");
        if (!nodeId || nodeId === "null") {
            if (session.executionFailed) return;
            setProgress(session, "COMPLETE", 1, 1, "Generation complete");
            if (session.media.output) {
                selectLane(session, "output");
                setPlayhead(session, session.trimIn);
            }
            return;
        }
        const target = graphNodes(session.node).find((item) => String(item.id) === nodeId);
        setProgress(session, "RUNNING", session.progressValue, session.progressMax, `Node ${nodeId} - ${target?.title || nodeClass(target) || "Executing"}`);
    });
    api.addEventListener("progress", (event) => {
        const session = runtime.session;
        if (!session) return;
        const detail = event?.detail || {};
        session.progressValue = Number(detail.value || 0);
        session.progressMax = Number(detail.max || 1);
        setProgress(session, "SAMPLING", session.progressValue, session.progressMax, field(session.root, "v2vfProgressCopy")?.textContent || "Sampling");
    });
    api.addEventListener("b_preview", (event) => {
        const session = runtime.session;
        if (!session || !event?.detail) return;
        if (session.livePreviewUrl) URL.revokeObjectURL(session.livePreviewUrl);
        session.livePreviewUrl = URL.createObjectURL(event.detail);
        updateLane(session, "intermediate", {
            url: session.livePreviewUrl,
            filename: "WAN_ANIMATE_2_LIVE_CHUNK.png",
            kind: "image",
            type: "temp",
            cacheKey: Date.now(),
        }, { title: "WAN ANIMATE 2 LIVE CHUNK", type: "PreviewAnimation" }, true);
    });
    api.addEventListener("executed", (event) => {
        const session = runtime.session;
        if (!session) return;
        const detail = event?.detail || {};
        const target = graphNodes(session.node).find((item) => String(item.id) === String(detail.node));
        for (const item of itemsFromExecuted(detail)) updateLane(session, classifyOutput(target, item), item, target);
    });
    api.addEventListener("execution_error", (event) => {
        if (runtime.session) {
            runtime.session.executionFailed = true;
            setProgress(runtime.session, "ERROR", 0, 1, String(event?.detail?.exception_message || "ComfyUI execution error"));
        }
    });
    api.addEventListener("execution_interrupted", () => {
        if (runtime.session) {
            runtime.session.executionFailed = true;
            setProgress(runtime.session, "STOPPED", 0, 1, "Generation interrupted");
        }
    });
}

function createMediaCard(title, role, mediaId, uploadId, pathId, accept) {
    const card = document.createElement("section");
    card.className = "iamccs-v2vf-card";
    card.innerHTML = `
        <div class="iamccs-v2vf-card-head"><span>${title}</span><span>${role}</span></div>
        <div id="${mediaId}" class="iamccs-v2vf-media"></div>
        <div class="iamccs-v2vf-card-actions"><button type="button" data-upload="${uploadId}">UPLOAD</button><button type="button" data-apply="${pathId}">APPLY PATH</button></div>
        <div class="iamccs-v2vf-path"><input id="${pathId}" type="text" placeholder="ComfyUI input path"></div>
        <input id="${uploadId}" class="iamccs-v2vf-hidden-input" type="file" accept="${accept}">
    `;
    return card;
}

function lane(label, copy, id, source = false) {
    const row = document.createElement("div");
    row.className = `iamccs-v2vf-lane ${source ? "source" : ""}`;
    row.innerHTML = `
        <div class="iamccs-v2vf-lane-label"><button type="button" class="iamccs-v2vf-lane-select" data-lane="${id}">o</button><b>${label}</b><small>${copy}</small></div>
        <div class="iamccs-v2vf-lane-track" id="v2vf${source ? "SourceTrack" : `${id}Track`}">
            <div id="v2vfStrip-${id}" class="iamccs-v2vf-strip"><span class="iamccs-v2vf-strip-empty">AWAITING ${label}</span></div>
            ${source ? '<div id="v2vfSegments" class="iamccs-v2vf-segments"></div><div id="v2vfTrimIn" class="iamccs-v2vf-trim-handle in"></div><div id="v2vfTrimOut" class="iamccs-v2vf-trim-handle out"></div>' : ""}
        </div>`;
    return row;
}

function controlMarkup(node) {
    return `
        <div class="iamccs-v2vf-section-head"><div><div class="iamccs-v2vf-kicker">05 / CONTROL ROOM</div><strong>Generation setup</strong></div><span>EASY</span></div>
        <div class="iamccs-v2vf-form">
            <label>WIDTH<input id="v2vfWidth" type="number" min="16" step="16" value="${read(node, "generation_width", 832)}"></label>
            <label>HEIGHT<input id="v2vfHeight" type="number" min="16" step="16" value="${read(node, "generation_height", 480)}"></label>
            <label>SOURCE FPS<input id="v2vfFps" type="number" min="1" step="0.01" value="${read(node, "fps", 30)}"></label>
            <label>FRAME CAP<input id="v2vfFrameCap" type="number" value="${read(node, "frame_load_cap", 81)}" disabled></label>
            <label>CHUNK FRAMES (4N+1)<input id="v2vfChunk" type="number" min="9" step="4" value="${read(node, "chunk_frames", 81)}"></label>
            <label>STEPS<input id="v2vfSteps" type="number" min="1" value="${read(node, "generation_steps", 6)}"></label>
            <label>CFG<input id="v2vfCfg" type="number" min="0" step="0.05" value="${read(node, "generation_cfg", 1)}"></label>
            <label>SEED<input id="v2vfSeed" type="number" min="0" value="${read(node, "generation_seed", 0)}"></label>
            <label>SEED MODE<select id="v2vfSeedMode"><option value="fixed" ${read(node, "seed_mode", "fixed") === "fixed" ? "selected" : ""}>FIXED</option><option value="increment" ${read(node, "seed_mode", "fixed") === "increment" ? "selected" : ""}>INCREMENT</option></select></label>
            <label>REFERENCE STRENGTH<input id="v2vfReferenceStrength" type="number" min="0" step="0.01" value="${read(node, "reference_image_strength", 1)}"></label>
            <label>POSE STRENGTH<input id="v2vfPoseStrength" type="number" min="0" step="0.01" value="${read(node, "pose_strength", 1)}"></label>
            <label>POSE START<input id="v2vfPoseStart" type="number" min="0" max="1" step="0.01" value="${read(node, "pose_start_percent", 0)}"></label>
            <label>POSE END<input id="v2vfPoseEnd" type="number" min="0" max="1" step="0.01" value="${read(node, "pose_end_percent", 1)}"></label>
            <label>CONTEXT LENGTH<input id="v2vfContextLength" type="number" min="1" value="${read(node, "context_length_latents", 21)}"></label>
            <label>CONTEXT OVERLAP<input id="v2vfContextOverlap" type="number" min="0" value="${read(node, "context_overlap_latents", 8)}"></label>
            <label>CONTEXT SCHEDULE<select id="v2vfContextSchedule"><option>standard_static</option><option>standard_uniform</option><option>looped_uniform</option><option>batched</option></select></label>
            <label>CONTEXT FUSE<select id="v2vfContextFuse"><option>pyramid</option><option>relative</option><option>flat</option><option>overlap-linear</option></select></label>
            <label>CACHE DEVICE<select id="v2vfCacheDevice"><option>cpu</option><option>gpu</option></select></label>
            <label>CACHE DTYPE<select id="v2vfCacheDtype"><option>int8</option><option>int4</option><option>default</option></select></label>
            <label>REFERENCE BG<select id="v2vfReferenceBackground"><option>keep_reference_background</option><option>isolate_character</option></select></label>
            <label>OUTPUT BG<select id="v2vfOutputBackground"><option>native_generated</option><option>source_video_composite</option><option>reference_image_composite</option><option>custom_background_composite</option></select></label>
            <label>LIVE PREVIEW<select id="v2vfLivePreview"><option>middle_frame</option><option>first_frame</option><option>last_frame</option><option>off</option></select></label>
            <label>LORA STRENGTH<input id="v2vfLoraStrength" type="number" step="0.01" value="${read(node, "lora_strength", 1)}"></label>
            <label class="wide">LORA<select id="v2vfLora"><option value="auto / graph">AUTO / GRAPH</option></select></label>
            <label class="iamccs-v2vf-toggle"><input id="v2vfApplyLora" type="checkbox" ${read(node, "apply_distill_lora", false) ? "checked" : ""}><span>APPLY DISTILL LORA</span></label>
            <label class="iamccs-v2vf-toggle"><input id="v2vfContext" type="checkbox" ${read(node, "enable_context_windows", true) ? "checked" : ""}><span>CONTEXT WINDOWS</span></label>
            <label class="iamccs-v2vf-toggle"><input id="v2vfPoseCache" type="checkbox" ${read(node, "enable_pose_cache", true) ? "checked" : ""}><span>WAN ANIMATE 2 POSE CACHE</span></label>
            <label class="iamccs-v2vf-toggle"><input id="v2vfEmptyCache" type="checkbox" ${read(node, "empty_cache_each_chunk", false) ? "checked" : ""}><span>EMPTY CACHE EACH CHUNK</span></label>
            <label class="wide">GENERATION PROMPT<textarea id="v2vfPrompt">${String(read(node, "global_prompt", "")).replaceAll("<", "&lt;")}</textarea></label>
            <label class="wide">DRIVING / MOTION PROMPT<textarea id="v2vfPosePrompt">${String(read(node, "pose_prompt", "")).replaceAll("<", "&lt;")}</textarea></label>
            <label class="wide">NEGATIVE PROMPT<textarea id="v2vfNegative">${String(read(node, "negative_prompt", "")).replaceAll("<", "&lt;")}</textarea></label>
            <label class="wide">OUTPUT PREFIX<input id="v2vfPrefix" value="${read(node, "output_prefix", "IAMCCS/WAN_ANIMATE_2_EASY")}"></label>
            <div class="iamccs-v2vf-models wide"><div class="iamccs-v2vf-kicker">LIVE BACKEND MODEL WIDGETS</div><div id="v2vfModels"></div></div>
        </div>
        <div id="v2vfBackendNotice" class="iamccs-v2vf-notice">Inspecting the Wan-Animate-2 graph...</div>
        <div class="iamccs-v2vf-control-actions"><button id="v2vfRender" class="primary" type="button">RENDER</button><button id="v2vfStop" class="danger" type="button" disabled>STOP</button></div>
        <div class="iamccs-v2vf-progress">
            <div class="iamccs-v2vf-progress-head"><span id="v2vfProgressPhase">IDLE</span><span id="v2vfProgressPercent">0%</span></div>
            <div class="iamccs-v2vf-progress-track"><div id="v2vfProgressFill" class="iamccs-v2vf-progress-fill"></div></div>
            <div id="v2vfProgressCopy" class="iamccs-v2vf-progress-copy">Waiting for ComfyUI</div>
        </div>
        <div class="iamccs-v2vf-preview-grid">
            <div class="iamccs-v2vf-preview"><b>DRIVING VIDEO</b><div id="v2vfPreview-pose" class="iamccs-v2vf-media"></div></div>
            <div class="iamccs-v2vf-preview"><b>REFERENCE IMAGE</b><div id="v2vfPreview-mask" class="iamccs-v2vf-media"></div></div>
            <div class="iamccs-v2vf-preview"><b>LIVE CHUNK</b><div id="v2vfPreview-intermediate" class="iamccs-v2vf-media"></div></div>
            <div class="iamccs-v2vf-preview"><b>FINAL OUTPUT</b><div id="v2vfPreview-output" class="iamccs-v2vf-media"></div></div>
        </div>`;
}

function openShotboard(node) {
    installStyles();
    bindEvents();
    runtime.session?.close?.();
    const root = document.createElement("div");
    root.className = "iamccs-v2vf";
    root.innerHTML = `
        <header class="iamccs-v2vf-header">
            <div class="iamccs-v2vf-brand"><div class="iamccs-v2vf-logo"><span>IAM</span><span>CCS</span></div><div><b>V2V SHOTBOARD EASY</b><small>WAN ANIMATE 2 EDITION / COMFYUI NODE</small></div></div>
            <div class="iamccs-v2vf-modebar"><button type="button" class="active">WAN ANIMATE 2</button></div>
            <div class="iamccs-v2vf-actions"><span class="iamccs-v2vf-status"><i></i> COMFYUI NATIVE</span><button id="v2vfClose" type="button">CLOSE</button></div>
        </header>
        <main class="iamccs-v2vf-shell">
            <aside id="v2vfLeft" class="iamccs-v2vf-column"><div class="iamccs-v2vf-section-head"><div><div class="iamccs-v2vf-kicker">01 / MEDIA DECK</div><strong>Inputs & output</strong></div><span>WAN 2</span></div></aside>
            <section class="iamccs-v2vf-center">
                <div class="iamccs-v2vf-monitors">
                    <div class="iamccs-v2vf-section-head"><div><div class="iamccs-v2vf-kicker">02 / SHOTBOARD VIEW</div><strong>Wan Animate 2 source and program</strong></div><span>LIVE</span></div>
                    <div class="iamccs-v2vf-monitor-grid">
                        <article class="iamccs-v2vf-monitor"><div class="iamccs-v2vf-monitor-head"><span>SOURCE</span><span>VIDEO / AUDIO</span></div><div id="v2vfSource" class="iamccs-v2vf-media"></div><div class="iamccs-v2vf-transport"><button data-source-step="-1">STEP -</button><button data-source-play>PLAY</button><button data-source-stop>STOP</button><button data-source-step="1">STEP +</button></div><div class="iamccs-v2vf-monitor-foot"><span>INPUT</span><span id="v2vfSourceTime">00:00.000</span></div></article>
                        <article id="v2vfProgramMonitor" class="iamccs-v2vf-monitor"><div class="iamccs-v2vf-monitor-head"><span>PROGRAM</span><span id="v2vfProgramLabel">SELECTED: SOURCE</span><button id="v2vfFull" class="iamccs-v2vf-full" type="button">FULL</button></div><div id="v2vfProgram" class="iamccs-v2vf-media"></div><div class="iamccs-v2vf-transport"><button data-program-step="-1">STEP -</button><button data-program-play>PLAY</button><button data-program-stop>STOP</button><button data-program-step="1">STEP +</button></div><div class="iamccs-v2vf-monitor-foot"><span>WAN ANIMATE 2</span><span id="v2vfProgramTime">00:00.000</span></div></article>
                    </div>
                </div>
                <div class="iamccs-v2vf-timeline">
                    <div class="iamccs-v2vf-timeline-head"><div><div class="iamccs-v2vf-kicker">03 / EDIT TIMELINE</div><strong>Layered shot timeline</strong></div><div class="iamccs-v2vf-transport"><button data-timeline-step="-1">STEP -</button><button data-timeline-play>PLAY</button><button data-timeline-stop>STOP</button><button data-timeline-step="1">STEP +</button></div><div><button id="v2vfSetIn">SET IN</button><button id="v2vfSetOut">SET OUT</button><button id="v2vfFit">FIT</button></div></div>
                    <div id="v2vfRuler" class="iamccs-v2vf-ruler"><div id="v2vfPlayhead" class="iamccs-v2vf-playhead"></div></div>
                    <div id="v2vfLanes" class="iamccs-v2vf-lanes"></div>
                    <div class="iamccs-v2vf-bottom"><div class="iamccs-v2vf-stat">IN <b id="v2vfStatIn"></b></div><div class="iamccs-v2vf-stat">OUT <b id="v2vfStatOut"></b></div><div class="iamccs-v2vf-stat">DURATION <b id="v2vfStatDuration"></b></div><div class="iamccs-v2vf-stat">FRAMES <b id="v2vfStatFrames"></b></div></div>
                </div>
            </section>
            <aside class="iamccs-v2vf-column">${controlMarkup(node)}</aside>
        </main>`;
    document.body.appendChild(root);
    const session = {
        node, root,
        identityMode: "wananimate2",
        trimIn: Number(read(node, "trim_start_s", 0)),
        trimOut: Number(read(node, "trim_end_s", read(node, "duration_seconds", 2.7))),
        playhead: Number(read(node, "trim_start_s", 0)),
        activeLane: String(read(node, "preview_stage", "source")),
        media: { source: itemFromPath(read(node, "source_video_path", ""), "video"), reference: itemFromPath(read(node, "source_image_path", ""), "image"), pose: itemFromPath(read(node, "source_video_path", ""), "video"), mask: itemFromPath(read(node, "source_image_path", ""), "image"), intermediate: null, output: null },
        mediaPriority: { source: 0, reference: 0, pose: Number.NEGATIVE_INFINITY, mask: Number.NEGATIVE_INFINITY, intermediate: Number.NEGATIVE_INFINITY, output: Number.NEGATIVE_INFINITY },
        progressValue: 0, progressMax: 1, executionFailed: false,
        close() { if (session.queueTimer) window.clearTimeout(session.queueTimer); if (session.livePreviewUrl) URL.revokeObjectURL(session.livePreviewUrl); root.remove(); if (runtime.session === session) runtime.session = null; document.body.style.overflow = session.oldOverflow; },
        oldOverflow: document.body.style.overflow,
    };
    session.modeDrafts = {};
    runtime.session = session;
    document.body.style.overflow = "hidden";

    const left = field(root, "v2vfLeft");
    left.append(createMediaCard("SOURCE VIDEO", "MOTION DRIVER", "v2vfSourceCard", "v2vfVideoUpload", "v2vfVideoPath", "video/*,.mp4,.mov,.mkv,.webm"));
    left.append(createMediaCard("REFERENCE IMAGE", "IDENTITY", "v2vfReferenceCard", "v2vfImageUpload", "v2vfImagePath", "image/*,.png,.jpg,.jpeg,.webp"));
    const outputCard = document.createElement("section");
    outputCard.className = "iamccs-v2vf-card";
    outputCard.innerHTML = '<div class="iamccs-v2vf-card-head"><span>GENERATED OUTPUT</span><span>FINAL VIDEO</span></div><div id="v2vfGenerated" class="iamccs-v2vf-media"></div><div class="iamccs-v2vf-card-actions"><button id="v2vfUseOutput" type="button">USE AS REFERENCE</button><button id="v2vfClear" type="button">CLEAR</button></div>';
    left.append(outputCard);
    field(root, "v2vfVideoPath").value = String(read(node, "source_video_path", ""));
    field(root, "v2vfImagePath").value = String(read(node, "source_image_path", ""));
    field(root, "v2vfDuration");

    const hiddenDuration = document.createElement("input");
    hiddenDuration.id = "v2vfDuration";
    hiddenDuration.type = "hidden";
    hiddenDuration.value = String(read(node, "duration_seconds", 2.7));
    root.appendChild(hiddenDuration);

    const lanes = field(root, "v2vfLanes");
    lanes.append(lane("SOURCE", "VIDEO + AUDIO", "source", true), lane("DRIVING", "raw motion video", "pose"), lane("REFERENCE", "identity / background", "mask"), lane("LIVE CHUNK", "sampling preview", "intermediate"), lane("OUTPUT", "final Wan Animate 2 video", "output"));

    installMedia(field(root, "v2vfSource"), session.media.source, "SOURCE");
    installMedia(field(root, "v2vfSourceCard"), session.media.source, "SOURCE VIDEO");
    installMedia(field(root, "v2vfReferenceCard"), session.media.reference, "REFERENCE IMAGE");
    installMedia(field(root, "v2vfGenerated"), null, "GENERATED OUTPUT");
    installMedia(field(root, "v2vfPreview-pose"), session.media.pose, "DRIVING VIDEO");
    installMedia(field(root, "v2vfPreview-mask"), session.media.mask, "REFERENCE IMAGE");
    renderStrip(session, "source", session.media.source);
    renderStrip(session, "pose", session.media.pose);
    renderStrip(session, "mask", session.media.mask);
    for (const laneId of ["intermediate", "output"]) renderStrip(session, laneId, null);
    renderProgram(session);

    const loraSelect = field(root, "v2vfLora");
    const loraNode = nodesByType(node, ["LoraLoaderModelOnly"])[0];
    const loraWidget = loraNode && widget(loraNode, "lora_name");
    for (const value of Array.isArray(loraWidget?.options?.values) ? loraWidget.options.values : []) loraSelect.appendChild(new Option(String(value), String(value)));
    const currentLora = String(read(node, "lora_name", "auto / graph"));
    if (![...loraSelect.options].some((option) => option.value === currentLora)) loraSelect.appendChild(new Option(currentLora, currentLora));
    loraSelect.value = currentLora;
    for (const [id, value] of [
        ["v2vfContextSchedule", read(node, "context_schedule", "standard_static")],
        ["v2vfContextFuse", read(node, "context_fuse_method", "pyramid")],
        ["v2vfCacheDevice", read(node, "cache_device", "cpu")],
        ["v2vfCacheDtype", read(node, "cache_dtype", "int8")],
        ["v2vfReferenceBackground", read(node, "reference_background_mode", "keep_reference_background")],
        ["v2vfOutputBackground", read(node, "output_background_mode", "native_generated")],
        ["v2vfLivePreview", read(node, "live_chunk_preview", "middle_frame")],
    ]) {
        if (field(root, id)) field(root, id).value = String(value);
    }
    $$(root, ".iamccs-v2vf-lane-select").forEach((button) => button.addEventListener("click", () => selectLane(session, button.dataset.lane)));
    selectLane(session, session.activeLane);

    async function loadFile(kind, file) {
        setProgress(session, "UPLOADING", 0, 1, `Uploading ${file.name}`);
        try {
            const result = await upload(file);
            const pathField = field(root, kind === "video" ? "v2vfVideoPath" : "v2vfImagePath");
            pathField.value = result.path;
            if (kind === "video") {
                session.media.source = result.item;
                session.media.pose = result.item;
                installMedia(field(root, "v2vfSource"), result.item, "SOURCE");
                installMedia(field(root, "v2vfSourceCard"), result.item, "SOURCE VIDEO");
                renderStrip(session, "source", result.item);
                updateLane(session, "pose", result.item, null, true);
                const probe = $(field(root, "v2vfSource"), "video");
                if (probe) probe.addEventListener("loadedmetadata", () => {
                    hiddenDuration.value = String(Math.max(.01, Number(probe.duration || 2.7)));
                    session.trimIn = 0; session.trimOut = Number(hiddenDuration.value); session.playhead = 0;
                    renderTimeline(session); renderProgram(session);
                }, { once: true });
            } else {
                session.media.reference = result.item;
                installMedia(field(root, "v2vfReferenceCard"), result.item, "REFERENCE IMAGE");
                updateLane(session, "mask", result.item, null, true);
            }
            const payload = commit(session);
            const synced = syncMediaLoaders(session, payload);
            setProgress(session, "IDLE", 0, 1, `${file.name} ready / backend media loaders updated: ${synced}`);
        } catch (error) {
            setProgress(session, "ERROR", 0, 1, error.message);
        }
    }
    $$(root, "[data-upload]").forEach((button) => button.addEventListener("click", () => field(root, button.dataset.upload)?.click()));
    field(root, "v2vfVideoUpload").addEventListener("change", (event) => event.target.files?.[0] && loadFile("video", event.target.files[0]));
    field(root, "v2vfImageUpload").addEventListener("change", (event) => event.target.files?.[0] && loadFile("image", event.target.files[0]));
    $$(root, "[data-apply]").forEach((button) => button.addEventListener("click", () => {
        const isVideo = button.dataset.apply === "v2vfVideoPath";
        const path = String(field(root, button.dataset.apply)?.value || "");
        const item = itemFromPath(path, isVideo ? "video" : "image");
        if (isVideo) {
            session.media.source = item;
            session.media.pose = item;
            installMedia(field(root, "v2vfSource"), item, "SOURCE");
            installMedia(field(root, "v2vfSourceCard"), item, "SOURCE VIDEO");
            renderStrip(session, "source", item);
            updateLane(session, "pose", item, null, true);
        } else {
            session.media.reference = item;
            installMedia(field(root, "v2vfReferenceCard"), item, "REFERENCE IMAGE");
            updateLane(session, "mask", item, null, true);
        }
        const payload = commit(session);
        const synced = syncMediaLoaders(session, payload);
        setProgress(session, "IDLE", 0, 1, `Media path applied / backend media loaders updated: ${synced}`);
        renderProgram(session);
    }));

    $$(root, "input,select,textarea").forEach((input) => input.addEventListener("change", () => renderTimeline(session)));
    const ruler = field(root, "v2vfRuler");
    const scrubAt = (event) => {
        const rect = ruler.getBoundingClientRect();
        if (!rect.width) return;
        const position = Math.max(0, Math.min(rect.width, event.clientX - rect.left));
        setPlayhead(session, position / rect.width * timelineDuration(session));
    };
    ruler.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;
        event.preventDefault();
        session.scrubbing = true;
        ruler.classList.add("scrubbing");
        ruler.setPointerCapture?.(event.pointerId);
        stop(session, "source");
        stop(session, "program");
        scrubAt(event);
        const move = (moveEvent) => {
            if (moveEvent.pointerId !== event.pointerId) return;
            moveEvent.preventDefault();
            scrubAt(moveEvent);
        };
        const finish = (finishEvent) => {
            if (finishEvent.pointerId !== event.pointerId) return;
            finishEvent.preventDefault();
            scrubAt(finishEvent);
            session.scrubbing = false;
            ruler.classList.remove("scrubbing");
            if (ruler.hasPointerCapture?.(event.pointerId)) ruler.releasePointerCapture(event.pointerId);
            ruler.removeEventListener("pointermove", move);
            ruler.removeEventListener("pointerup", finish);
            ruler.removeEventListener("pointercancel", finish);
        };
        ruler.addEventListener("pointermove", move);
        ruler.addEventListener("pointerup", finish);
        ruler.addEventListener("pointercancel", finish);
    });
    function dragTrim(which, event) {
        event.preventDefault();
        const track = field(root, "v2vfSourceTrack");
        const move = (moveEvent) => {
            const rect = track.getBoundingClientRect();
            const value = Math.max(0, Math.min(numberValue(root, "v2vfDuration", 2.7), (moveEvent.clientX - rect.left) / rect.width * numberValue(root, "v2vfDuration", 2.7)));
            if (which === "in") session.trimIn = Math.min(value, session.trimOut - .01);
            else session.trimOut = Math.max(value, session.trimIn + .01);
            session.playhead = value;
            renderTimeline(session);
        };
        const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up);
    }
    field(root, "v2vfTrimIn").addEventListener("pointerdown", (event) => dragTrim("in", event));
    field(root, "v2vfTrimOut").addEventListener("pointerdown", (event) => dragTrim("out", event));
    field(root, "v2vfSetIn").addEventListener("click", () => { session.trimIn = Math.min(session.playhead, session.trimOut - .01); renderTimeline(session); });
    field(root, "v2vfSetOut").addEventListener("click", () => { session.trimOut = Math.max(session.playhead, session.trimIn + .01); renderTimeline(session); });
    field(root, "v2vfFit").addEventListener("click", () => { session.trimIn = 0; session.trimOut = numberValue(root, "v2vfDuration", 2.7); session.playhead = 0; renderTimeline(session); });
    $$(root, "[data-source-play]").forEach((button) => button.addEventListener("click", () => play(session, "source")));
    $$(root, "[data-source-stop]").forEach((button) => button.addEventListener("click", () => stop(session, "source")));
    $$(root, "[data-program-play],[data-timeline-play]").forEach((button) => button.addEventListener("click", () => play(session, "program")));
    $$(root, "[data-program-stop],[data-timeline-stop]").forEach((button) => button.addEventListener("click", () => stop(session, "program")));
    $$(root, "[data-source-step],[data-program-step],[data-timeline-step]").forEach((button) => button.addEventListener("click", () => step(session, Number(button.dataset.sourceStep || button.dataset.programStep || button.dataset.timelineStep))));
    field(root, "v2vfFull").addEventListener("click", () => field(root, "v2vfProgramMonitor").requestFullscreen?.());
    field(root, "v2vfClose").addEventListener("click", () => { commit(session); session.close(); });
    field(root, "v2vfClear").addEventListener("click", () => {
        for (const laneId of ["intermediate", "output"]) updateLane(session, laneId, null, null, true);
        setProgress(session, "IDLE", 0, 1, "Preview outputs cleared");
    });
    field(root, "v2vfUseOutput").addEventListener("click", () => {
        const item = session.media.output;
        if (!item) return;
        const path = [item.subfolder, item.filename].filter(Boolean).join("/");
        field(root, "v2vfImagePath").value = path;
        session.media.reference = { ...item, type: item.type || "output" };
        installMedia(field(root, "v2vfReferenceCard"), session.media.reference, "REFERENCE IMAGE");
        updateLane(session, "mask", session.media.reference, null, true);
        commit(session);
    });
    field(root, "v2vfRender").addEventListener("click", async () => {
        let renderSession = session;
        let payload = commit(renderSession);
        backendSummary(renderSession);
        if (!payload.source_video_path || !payload.source_image_path) {
            setProgress(renderSession, "ERROR", 0, 1, "Load both source video and reference image before rendering");
            return;
        }
        if (renderSession.backendMissing.length) {
            try {
                renderSession = await loadUnifiedBackend(renderSession, payload);
                payload = commit(renderSession);
            } catch (error) {
                setProgress(runtime.session || renderSession, "ERROR", 0, 1, `Backend load failed: ${error.message}`);
                return;
            }
        }
        if (renderSession.backendMissing.length) {
            setProgress(renderSession, "ERROR", 0, 1, `Unified workflow is missing: ${renderSession.backendMissing.join(", ")}`);
            return;
        }
        if (graphFor(renderSession.node) !== app?.graph) {
            setProgress(renderSession, "ERROR", 0, 1, "The Wan-Animate-2 Easy backend is loaded but its workflow tab is not active. Select IAMCCS_V2V_SHOTBOARD_EASY_WANANIMATE2_GGUF and press Render again.");
            return;
        }
        const touched = syncBackend(renderSession, payload);
        const truthIssues = auditBackendTruth(renderSession, payload);
        if (truthIssues.length) {
            setProgress(renderSession, "ERROR", 0, 1, `UI truth check failed: ${truthIssues.join("; ")}`);
            return;
        }
        renderSession.executionFailed = false;
        setProgress(renderSession, "QUEUED", 0, 1, `UI truth verified: ${touched} backend values / prompts ${renderSession.backendSync.positivePromptTargets}+${renderSession.backendSync.posePromptTargets}+${renderSession.backendSync.negativePromptTargets} / source FPS+audio linked / outputs ${renderSession.backendSync.outputPrefixTargets}. Queueing workflow...`);
        try {
            if (typeof app.queuePrompt !== "function") throw new Error("ComfyUI queue API is unavailable");
            await app.queuePrompt(0, 1);
        } catch (error) {
            setProgress(renderSession, "ERROR", 0, 1, error.message);
        }
    });
    field(root, "v2vfStop").addEventListener("click", async () => {
        try {
            await api.fetchApi("/interrupt", { method: "POST" });
            setProgress(session, "STOPPED", 0, 1, "Interrupt sent to ComfyUI");
        } catch (error) {
            setProgress(session, "ERROR", 0, 1, error.message);
        }
    });

    renderModelDeck(session);
    backendSummary(session);
    renderTimeline(session);
    setProgress(session, "IDLE", 0, 1, "Ready. Load media and render the Wan-Animate-2 backend.");
    void (async () => {
        await hydrateLatestOutputs(session);
        if (runtime.session === session) startQueueMonitor(session);
    })();
}

function configureNode(node) {
    if (node._iamccsV2VEasyWanAnimate2Ready) return;
    node._iamccsV2VEasyWanAnimate2Ready = true;
    for (const item of node.widgets || []) hideWidget(item);
    const open = node.addWidget("button", "OPEN V2V SHOTBOARD", null, () => openShotboard(node));
    open.serialize = false;
    const info = node.addWidget("text", "edition", "EASY / WAN ANIMATE 2", () => {});
    info.disabled = true;
    node.size = [330, 112];
    node.setSize?.([330, 112]);
    node.color = "#123642";
    node.bgcolor = "#0b171e";
    const originalSerialize = node.onSerialize;
    node.onSerialize = function(serialized) {
        if (runtime.session?.node === node) commit(runtime.session);
        return originalSerialize?.call?.(this, serialized);
    };
    const originalRemoved = node.onRemoved;
    node.onRemoved = function() {
        if (runtime.session?.node === node) runtime.session.close();
        return originalRemoved?.apply?.(this, arguments);
    };
}

let configurationScannerStarted = false;

function startConfigurationScanner() {
    if (configurationScannerStarted) return;
    configurationScannerStarted = true;
    let lastScanSignature = "";
    const configureVisiblePlanner = () => {
        const nodes = graphNodes();
        const scanSignature = nodes.map((node) => `${nodeClass(node)}|${node?.title || ""}`).join(";");
        if (scanSignature && scanSignature !== lastScanSignature) {
            console.info("[IAMCCS Wan Animate 2 Easy] graph probe", scanSignature);
            lastScanSignature = scanSignature;
        }
        for (const node of nodes) {
            if (isPlannerNode(node)) configureNode(node);
        }
    };
    configureVisiblePlanner();
    window.setInterval(configureVisiblePlanner, 1000);
}

try {
    console.info("[IAMCCS Wan Animate 2 Easy] registering extension", typeof app?.registerExtension);
    const extensionRegistration = app.registerExtension({
        name: `IAMCCS.V2VShotboardEasyWanAnimate2.${VERSION}`,
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const definitionName = String(nodeData?.name || nodeData?.class_type || nodeData?.display_name || "");
        if (String(nodeData?.category || "").startsWith("IAMCCS/V2V/Easy")) {
            console.info("[IAMCCS Wan Animate 2 Easy] definition probe", JSON.stringify({
                definitionName,
                name: nodeData?.name,
                class_type: nodeData?.class_type,
                display_name: nodeData?.display_name,
                category: nodeData?.category,
                type: nodeType?.type,
                title: nodeType?.title,
            }));
        }
        if (definitionName !== NODE_CLASS && definitionName !== NODE_DISPLAY) return;
        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = originalCreated?.apply?.(this, arguments);
            configureNode(this);
            return result;
        };
        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info) {
            const result = originalConfigure?.apply?.(this, arguments);
            setTimeout(() => configureNode(this), 0);
            return result;
        };
    },
    async nodeCreated(node) {
        if (String(node?.title || "").includes("Shotboard Easy")) {
            console.info("[IAMCCS Wan Animate 2 Easy] node probe", JSON.stringify({
                comfyClass: node?.comfyClass,
                type: node?.type,
                title: node?.title,
                constructorType: node?.constructor?.type,
                constructorTitle: node?.constructor?.title,
            }));
        }
        if (!isPlannerNode(node)) return;
        configureNode(node);
    },
    async setup() {
        startConfigurationScanner();
    },
    });
    console.info("[IAMCCS Wan Animate 2 Easy] registration call returned", typeof extensionRegistration);
    Promise.resolve(extensionRegistration).then(
        () => console.info("[IAMCCS Wan Animate 2 Easy] extension registered"),
        (error) => console.error("[IAMCCS Wan Animate 2 Easy] extension registration rejected", error),
    );
} catch (error) {
    console.error("[IAMCCS Wan Animate 2 Easy] extension registration failed", error);
}

startConfigurationScanner();

globalThis.IAMCCSV2VShotboardEasyWanAnimate2 = {
    open() {
        const node = graphNodes().find((item) => isPlannerNode(item));
        if (!node) throw new Error("Load a V2V Shotboard Easy workflow first");
        openShotboard(node);
    },
    close() {
        runtime.session?.close?.();
    },
};
