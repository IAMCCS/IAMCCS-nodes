import { app } from "../../scripts/app.js";

const NAV_NODE_TYPE = "IAMCCS_Navigator";
const STORAGE_KEY = "iamccs.navigator.ui.v1";
const VERSION = "0.1.1";

const COLOR_MAP = {
    teal: "#22c7a9",
    amber: "#f2b84b",
    green: "#70c86b",
    red: "#ef6a6a",
    violet: "#a98bff",
    blue: "#6da7ff",
    gray: "#a7abb3",
};

const state = {
    panel: null,
    palette: null,
    quickInput: null,
    quickList: null,
    indexInput: null,
    indexList: null,
    helpModal: null,
    panelOpen: false,
    paletteOpen: false,
    quickSelected: 0,
    quickQuery: "",
    indexQuery: "",
    refreshQueued: false,
    dragging: false,
    dragStart: null,
    history: [],
    tempReturn: null,
    ui: loadUiState(),
};

function loadUiState() {
    const fallback = {
        persistent: false,
        layout: "float",
        collapsed: false,
        left: 72,
        top: 96,
        width: 320,
        height: 430,
        bottomHeight: 96,
    };
    try {
        return { ...fallback, ...JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}") };
    } catch (_) {
        return fallback;
    }
}

function saveUiState() {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(state.ui));
    } catch (_) {
        // Local storage can be disabled in hardened browser profiles.
    }
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function isTextTarget(target) {
    const tag = String(target?.tagName || "").toLowerCase();
    return tag === "input" || tag === "textarea" || target?.isContentEditable;
}

function getCanvas() {
    return app?.canvas || window?.LiteGraph?.LGraphCanvas?.active_canvas || null;
}

function getGraph() {
    return app?.graph || getCanvas()?.graph || null;
}

function getWidget(node, name) {
    return (node?.widgets || []).find((widget) => widget.name === name);
}

function getWidgetValue(node, name, fallback = "") {
    const widget = getWidget(node, name);
    const value = widget?.value;
    return value === undefined || value === null ? fallback : value;
}

function setWidgetValue(node, name, value) {
    const widget = getWidget(node, name);
    if (!widget) return false;
    widget.value = value;
    widget.callback?.(value, app?.canvas, node, null);
    markCanvasDirty();
    scheduleRefresh();
    return true;
}

function markCanvasDirty() {
    const canvas = getCanvas();
    canvas?.setDirty?.(true, true);
    canvas?.draw?.(true, true);
}

function readColor(node) {
    const color = String(getWidgetValue(node, "color", "teal")).trim();
    if (color === "custom") {
        const custom = String(getWidgetValue(node, "custom_color", "")).trim();
        if (/^#[0-9a-f]{3,8}$/i.test(custom)) return custom;
    }
    return COLOR_MAP[color] || COLOR_MAP.teal;
}

function isNavigatorNode(node) {
    return String(node?.type || node?.constructor?.type || "") === NAV_NODE_TYPE;
}

function asBookmark(node) {
    if (!isNavigatorNode(node)) return null;
    const showInIndex = getWidgetValue(node, "show_in_index", true);
    if (showInIndex === false || showInIndex === "false" || showInIndex === 0 || showInIndex === "0") return null;

    const name = String(getWidgetValue(node, "bookmark_name", "") || "").trim() || node.title || `Bookmark ${node.id}`;
    const category = String(getWidgetValue(node, "category", "") || "").trim();
    const icon = String(getWidgetValue(node, "icon", "") || "").trim();
    const note = String(getWidgetValue(node, "note", "") || "").trim();
    const order = Number(getWidgetValue(node, "order", 0)) || 0;
    const zoomMode = String(getWidgetValue(node, "zoom_mode", "keep current zoom"));
    const savedZoom = Number(getWidgetValue(node, "saved_zoom", 0.8)) || 0.8;
    const size = Array.isArray(node.size) ? node.size : [180, 80];
    const pos = Array.isArray(node.pos) ? node.pos : [0, 0];

    return {
        id: node.id,
        node,
        name,
        category,
        icon,
        note,
        order,
        zoomMode,
        savedZoom,
        color: readColor(node),
        x: pos[0] + size[0] * 0.5,
        y: pos[1] + size[1] * 0.5,
    };
}

function getBookmarks() {
    const nodes = getGraph()?._nodes || [];
    return nodes
        .map(asBookmark)
        .filter(Boolean)
        .sort((a, b) => {
            if (a.order !== b.order) return a.order - b.order;
            if (a.y !== b.y) return a.y - b.y;
            return a.x - b.x;
        });
}

function currentView() {
    const canvas = getCanvas();
    const ds = canvas?.ds;
    if (!canvas || !ds) return null;
    const scale = Number(ds.scale || 1);
    const width = Number(canvas.canvas?.width || window.innerWidth || 1);
    const height = Number(canvas.canvas?.height || window.innerHeight || 1);
    return {
        x: width / (2 * scale) - Number(ds.offset?.[0] || 0),
        y: height / (2 * scale) - Number(ds.offset?.[1] || 0),
        zoom: scale,
    };
}

function targetOffset(x, y, zoom) {
    const canvas = getCanvas();
    const width = Number(canvas?.canvas?.width || window.innerWidth || 1);
    const height = Number(canvas?.canvas?.height || window.innerHeight || 1);
    return [width / (2 * zoom) - x, height / (2 * zoom) - y];
}

function jumpToPosition(target, options = {}) {
    const canvas = getCanvas();
    const ds = canvas?.ds;
    if (!canvas || !ds || !target) return;

    const startZoom = Number(ds.scale || 1);
    const endZoom = Number(options.zoom || target.zoom || startZoom);
    const startOffset = [Number(ds.offset?.[0] || 0), Number(ds.offset?.[1] || 0)];
    const endOffset = targetOffset(target.x, target.y, endZoom);
    const duration = options.instant ? 0 : 230;
    const started = performance.now();

    function apply(offset, zoom) {
        ds.offset[0] = offset[0];
        ds.offset[1] = offset[1];
        ds.scale = zoom;
        markCanvasDirty();
    }

    if (!duration) {
        apply(endOffset, endZoom);
        return;
    }

    function tick(now) {
        const t = clamp((now - started) / duration, 0, 1);
        const eased = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) * 0.5;
        apply(
            [
                startOffset[0] + (endOffset[0] - startOffset[0]) * eased,
                startOffset[1] + (endOffset[1] - startOffset[1]) * eased,
            ],
            startZoom + (endZoom - startZoom) * eased,
        );
        if (t < 1) requestAnimationFrame(tick);
    }

    requestAnimationFrame(tick);
}

function jumpToBookmark(bookmark, pushHistory = true) {
    if (!bookmark) return;
    const previous = currentView();
    if (pushHistory && previous) state.history.push(previous);
    const zoom = bookmark.zoomMode === "restore saved zoom" ? bookmark.savedZoom : previous?.zoom;
    jumpToPosition({ x: bookmark.x, y: bookmark.y, zoom });
    closePalette();
}

function returnBack() {
    const target = state.history.pop() || state.tempReturn;
    if (target) jumpToPosition(target);
}

function saveTempReturn() {
    state.tempReturn = currentView();
    flashPanelMessage("Return point saved");
}

function scoreBookmark(bookmark, query) {
    const q = String(query || "").trim().toLowerCase();
    if (!q) return 1;
    const haystack = `${bookmark.name} ${bookmark.category} ${bookmark.note}`.toLowerCase();
    const direct = haystack.indexOf(q);
    if (direct >= 0) return 1000 - direct;

    let qi = 0;
    let score = 0;
    for (let i = 0; i < haystack.length && qi < q.length; i += 1) {
        if (haystack[i] === q[qi]) {
            score += 10 - Math.min(8, i);
            qi += 1;
        }
    }
    return qi === q.length ? score : 0;
}

function filterBookmarks(query) {
    return getBookmarks()
        .map((bookmark) => ({ bookmark, score: scoreBookmark(bookmark, query) }))
        .filter((item) => item.score > 0)
        .sort((a, b) => b.score - a.score)
        .map((item) => item.bookmark);
}

function injectStyles() {
    if (document.getElementById("iamccs-navigator-style")) return;
    const style = document.createElement("style");
    style.id = "iamccs-navigator-style";
    style.textContent = `
.iamccs-nav-panel,
.iamccs-nav-palette {
    color: #f4f2ec;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    letter-spacing: 0;
    z-index: 9999;
}
.iamccs-nav-panel {
    position: fixed;
    display: none;
    min-width: 220px;
    min-height: 44px;
    max-width: min(720px, calc(100vw - 24px));
    max-height: calc(100vh - 24px);
    background: rgba(24, 25, 27, 0.94);
    border: 1px solid rgba(255, 255, 255, 0.12);
    box-shadow: 0 16px 44px rgba(0, 0, 0, 0.36);
    backdrop-filter: blur(18px);
    border-radius: 8px;
    overflow: hidden;
}
.iamccs-nav-panel.is-open {
    display: grid;
    grid-template-rows: auto auto 1fr;
}
.iamccs-nav-panel[data-layout="float"] {
    resize: both;
}
.iamccs-nav-panel[data-layout="bottom"] {
    left: 14px !important;
    right: 14px !important;
    top: auto !important;
    bottom: 12px;
    width: auto !important;
    max-width: none;
    resize: vertical;
}
.iamccs-nav-panel[data-layout="rail"] {
    left: 14px !important;
    right: 14px !important;
    top: auto !important;
    bottom: 12px;
    width: auto !important;
    height: 48px !important;
    min-height: 48px;
    max-width: none;
    resize: none;
    grid-template-rows: 48px;
}
.iamccs-nav-panel.is-collapsed {
    height: 44px !important;
    min-height: 44px;
    resize: none;
    grid-template-rows: 44px;
}
.iamccs-nav-head {
    height: 44px;
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 0 8px 0 12px;
    background: linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02));
    cursor: grab;
    user-select: none;
}
.iamccs-nav-title {
    min-width: 0;
    flex: 1;
    font-size: 12px;
    font-weight: 720;
    text-transform: uppercase;
    color: #f6efe0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.iamccs-nav-count {
    color: #aeb4b8;
    font-size: 11px;
    font-weight: 620;
}
.iamccs-nav-button {
    height: 28px;
    min-width: 28px;
    border: 1px solid rgba(255, 255, 255, 0.11);
    border-radius: 7px;
    background: rgba(255, 255, 255, 0.06);
    color: #f4f2ec;
    font-size: 11px;
    font-weight: 720;
    padding: 0 8px;
    cursor: pointer;
}
.iamccs-nav-button:hover,
.iamccs-nav-button.is-active {
    border-color: rgba(34, 199, 169, 0.58);
    background: rgba(34, 199, 169, 0.15);
}
.iamccs-nav-search-row {
    padding: 8px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
}
.iamccs-nav-input {
    width: 100%;
    height: 32px;
    box-sizing: border-box;
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 7px;
    outline: none;
    background: rgba(0, 0, 0, 0.28);
    color: #f4f2ec;
    padding: 0 10px;
    font-size: 12px;
}
.iamccs-nav-input:focus {
    border-color: rgba(242, 184, 75, 0.72);
}
.iamccs-nav-list {
    min-height: 0;
    overflow: auto;
    padding: 6px 8px 10px;
}
.iamccs-nav-panel[data-layout="bottom"] .iamccs-nav-list,
.iamccs-nav-panel[data-layout="rail"] .iamccs-nav-list {
    display: flex;
    gap: 6px;
    align-items: center;
    overflow-x: auto;
    overflow-y: hidden;
    padding: 5px 8px;
}
.iamccs-nav-panel[data-layout="rail"] .iamccs-nav-search-row,
.iamccs-nav-panel.is-collapsed .iamccs-nav-search-row,
.iamccs-nav-panel.is-collapsed .iamccs-nav-list {
    display: none;
}
.iamccs-nav-group {
    margin: 8px 4px 4px;
    color: #9da3a7;
    font-size: 10px;
    font-weight: 820;
    text-transform: uppercase;
}
.iamccs-nav-panel[data-layout="bottom"] .iamccs-nav-group,
.iamccs-nav-panel[data-layout="rail"] .iamccs-nav-group {
    display: none;
}
.iamccs-nav-bookmark {
    width: 100%;
    min-height: 34px;
    display: grid;
    grid-template-columns: 12px 1fr auto;
    gap: 8px;
    align-items: center;
    border: 1px solid transparent;
    border-radius: 7px;
    background: transparent;
    color: #f4f2ec;
    padding: 6px 8px;
    text-align: left;
    cursor: pointer;
}
.iamccs-nav-bookmark:hover,
.iamccs-nav-bookmark.is-selected {
    border-color: rgba(255, 255, 255, 0.12);
    background: rgba(255, 255, 255, 0.07);
}
.iamccs-nav-panel[data-layout="bottom"] .iamccs-nav-bookmark,
.iamccs-nav-panel[data-layout="rail"] .iamccs-nav-bookmark {
    width: auto;
    min-width: 84px;
    max-width: 210px;
    grid-template-columns: 10px auto;
    white-space: nowrap;
}
.iamccs-nav-dot {
    width: 9px;
    height: 9px;
    border-radius: 999px;
    box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.05);
}
.iamccs-nav-name {
    min-width: 0;
    font-size: 12px;
    font-weight: 720;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.iamccs-nav-note {
    grid-column: 2 / 4;
    color: #b9bec1;
    font-size: 11px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.iamccs-nav-panel[data-layout="bottom"] .iamccs-nav-note,
.iamccs-nav-panel[data-layout="rail"] .iamccs-nav-note {
    display: none;
}
.iamccs-nav-order {
    color: #8d9499;
    font-size: 10px;
    font-weight: 760;
}
.iamccs-nav-empty {
    color: #b9bec1;
    font-size: 12px;
    padding: 16px 10px;
}
.iamccs-nav-palette {
    position: fixed;
    inset: 0;
    display: none;
    align-items: flex-start;
    justify-content: center;
    padding-top: min(15vh, 120px);
    background: rgba(0, 0, 0, 0.22);
}
.iamccs-nav-palette.is-open {
    display: flex;
}
.iamccs-nav-palette-box {
    width: min(560px, calc(100vw - 28px));
    max-height: min(620px, calc(100vh - 64px));
    display: grid;
    grid-template-rows: auto auto 1fr;
    background: rgba(24, 25, 27, 0.97);
    border: 1px solid rgba(255, 255, 255, 0.13);
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 24px 70px rgba(0, 0, 0, 0.48);
    backdrop-filter: blur(20px);
}
.iamccs-nav-palette-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 42px;
    padding: 0 12px;
    color: #f6efe0;
    font-size: 12px;
    font-weight: 820;
    text-transform: uppercase;
    background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
}
.iamccs-nav-hint {
    color: #aeb4b8;
    font-size: 11px;
    font-weight: 620;
    text-transform: none;
}
.iamccs-nav-palette .iamccs-nav-input {
    height: 38px;
    border-radius: 0;
    border-left: 0;
    border-right: 0;
}
.iamccs-nav-toast {
    position: fixed;
    right: 18px;
    bottom: 18px;
    padding: 8px 10px;
    border-radius: 7px;
    background: rgba(24, 25, 27, 0.96);
    color: #f4f2ec;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 10px 28px rgba(0,0,0,0.35);
    font: 720 12px Inter, ui-sans-serif, system-ui, sans-serif;
    z-index: 10000;
}
.iamccs-nav-help {
    position: fixed;
    inset: 0;
    display: none;
    align-items: center;
    justify-content: center;
    padding: 18px;
    background: rgba(0, 0, 0, 0.34);
    z-index: 10001;
}
.iamccs-nav-help.is-open {
    display: flex;
}
.iamccs-nav-help-box {
    width: min(680px, calc(100vw - 28px));
    max-height: min(720px, calc(100vh - 28px));
    overflow: auto;
    color: #f4f2ec;
    background: rgba(24, 25, 27, 0.98);
    border: 1px solid rgba(255, 255, 255, 0.13);
    border-radius: 8px;
    box-shadow: 0 24px 70px rgba(0, 0, 0, 0.52);
}
.iamccs-nav-help-head {
    position: sticky;
    top: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 12px 14px;
    background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.iamccs-nav-help-title {
    font-size: 13px;
    font-weight: 840;
    text-transform: uppercase;
}
.iamccs-nav-help-body {
    padding: 14px;
}
.iamccs-nav-help-section {
    margin: 0 0 16px;
}
.iamccs-nav-help-section h3 {
    margin: 0 0 8px;
    color: #f6efe0;
    font-size: 12px;
    font-weight: 840;
    text-transform: uppercase;
}
.iamccs-nav-help-section p,
.iamccs-nav-help-section li {
    color: #c9ced1;
    font-size: 12px;
    line-height: 1.52;
}
.iamccs-nav-help-section ol,
.iamccs-nav-help-section ul {
    margin: 0;
    padding-left: 20px;
}
.iamccs-nav-help-kbd {
    display: inline-block;
    min-width: 18px;
    padding: 1px 5px;
    border: 1px solid rgba(255,255,255,0.16);
    border-radius: 5px;
    background: rgba(255,255,255,0.07);
    color: #f4f2ec;
    font-weight: 760;
}
`;
    document.head.appendChild(style);
}

function makeButton(label, title, action) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "iamccs-nav-button";
    button.textContent = label;
    button.title = title;
    button.dataset.action = action;
    return button;
}

function createPalette() {
    if (state.palette) return;
    const palette = document.createElement("div");
    palette.className = "iamccs-nav-palette";
    palette.innerHTML = `
        <div class="iamccs-nav-palette-box">
            <div class="iamccs-nav-palette-title">
                <span>IAMCCS Navigator</span>
                <span class="iamccs-nav-hint">Enter jump / Esc close</span>
            </div>
            <input class="iamccs-nav-input" type="text" autocomplete="off" placeholder="Search bookmarks">
            <div class="iamccs-nav-list"></div>
        </div>
    `;
    document.body.appendChild(palette);
    state.palette = palette;
    state.quickInput = palette.querySelector("input");
    state.quickList = palette.querySelector(".iamccs-nav-list");

    palette.addEventListener("mousedown", (event) => {
        if (event.target === palette) closePalette();
    });
    state.quickInput.addEventListener("input", () => {
        state.quickQuery = state.quickInput.value;
        state.quickSelected = 0;
        renderPalette();
    });
    state.quickInput.addEventListener("keydown", handlePaletteKeydown);
}

function createPanel() {
    if (state.panel) return;
    const panel = document.createElement("div");
    panel.className = "iamccs-nav-panel";
    panel.innerHTML = `
        <div class="iamccs-nav-head">
            <div class="iamccs-nav-title">Navigator <span class="iamccs-nav-count"></span></div>
        </div>
        <div class="iamccs-nav-search-row">
            <input class="iamccs-nav-input" type="text" autocomplete="off" placeholder="Filter bookmarks">
        </div>
        <div class="iamccs-nav-list"></div>
    `;
    const head = panel.querySelector(".iamccs-nav-head");
    head.appendChild(makeButton("Find", "Open quick search", "palette"));
    head.appendChild(makeButton("Back", "Return to previous Navigator jump", "back"));
    head.appendChild(makeButton("Mode", "Cycle float, bottom strip, rail", "mode"));
    head.appendChild(makeButton("Pin", "Persist this index on reload", "pin"));
    head.appendChild(makeButton("?", "Navigator help", "help"));
    head.appendChild(makeButton("-", "Collapse index", "collapse"));
    head.appendChild(makeButton("x", "Close index", "close"));

    document.body.appendChild(panel);
    state.panel = panel;
    state.indexInput = panel.querySelector("input");
    state.indexList = panel.querySelector(".iamccs-nav-list");

    panel.addEventListener("click", handlePanelClick);
    state.indexInput.addEventListener("input", () => {
        state.indexQuery = state.indexInput.value;
        renderIndex();
    });
    head.addEventListener("pointerdown", beginPanelDrag);
    panel.addEventListener("pointerup", capturePanelSize);
    applyPanelLayout();
}

function openPalette() {
    createPalette();
    state.paletteOpen = true;
    state.quickQuery = "";
    state.quickSelected = 0;
    state.quickInput.value = "";
    state.palette.classList.add("is-open");
    renderPalette();
    requestAnimationFrame(() => state.quickInput.focus());
}

function closePalette() {
    if (!state.palette) return;
    state.paletteOpen = false;
    state.palette.classList.remove("is-open");
}

function togglePalette() {
    if (state.paletteOpen) closePalette();
    else openPalette();
}

function openPanel() {
    createPanel();
    state.panelOpen = true;
    state.panel.classList.add("is-open");
    renderIndex();
    saveUiState();
}

function closePanel() {
    if (!state.panel) return;
    state.panelOpen = false;
    state.panel.classList.remove("is-open");
    saveUiState();
}

function togglePanel() {
    if (state.panelOpen) closePanel();
    else openPanel();
}

function handlePaletteKeydown(event) {
    const bookmarks = filterBookmarks(state.quickQuery).slice(0, 30);
    if (event.key === "Escape") {
        event.preventDefault();
        closePalette();
        return;
    }
    if (event.key === "ArrowDown") {
        event.preventDefault();
        state.quickSelected = clamp(state.quickSelected + 1, 0, Math.max(0, bookmarks.length - 1));
        renderPalette();
        return;
    }
    if (event.key === "ArrowUp") {
        event.preventDefault();
        state.quickSelected = clamp(state.quickSelected - 1, 0, Math.max(0, bookmarks.length - 1));
        renderPalette();
        return;
    }
    if (event.key === "Enter") {
        event.preventDefault();
        jumpToBookmark(bookmarks[state.quickSelected]);
        return;
    }
    if (!state.quickQuery && /^[1-9]$/.test(event.key)) {
        const bookmark = bookmarks[Number(event.key) - 1];
        if (bookmark) {
            event.preventDefault();
            jumpToBookmark(bookmark);
        }
    }
}

function handleGlobalKeydown(event) {
    if (event.defaultPrevented) return;
    const key = String(event.key || "").toLowerCase();
    const command = event.ctrlKey || event.metaKey;
    if (key === "escape" && state.helpModal?.classList.contains("is-open")) {
        event.preventDefault();
        closeHelp();
        return;
    }
    if (isTextTarget(event.target) && !state.paletteOpen) return;

    if (command && event.code === "Space") {
        event.preventDefault();
        togglePalette();
        return;
    }
    if (command && event.altKey && key === "n") {
        event.preventDefault();
        togglePanel();
        return;
    }
    if (command && event.shiftKey && key === "m") {
        event.preventDefault();
        saveTempReturn();
        return;
    }
    if (event.altKey && event.key === "Backspace") {
        event.preventDefault();
        returnBack();
        return;
    }
}

function handlePanelClick(event) {
    const actionButton = event.target.closest("[data-action]");
    if (actionButton) {
        const action = actionButton.dataset.action;
        if (action === "palette") openPalette();
        if (action === "back") returnBack();
        if (action === "help") openHelp();
        if (action === "close") closePanel();
        if (action === "collapse") {
            state.ui.collapsed = !state.ui.collapsed;
            applyPanelLayout();
            saveUiState();
        }
        if (action === "pin") {
            state.ui.persistent = !state.ui.persistent;
            applyPanelLayout();
            saveUiState();
        }
        if (action === "mode") {
            state.ui.layout = state.ui.layout === "float" ? "bottom" : state.ui.layout === "bottom" ? "rail" : "float";
            state.ui.collapsed = false;
            applyPanelLayout();
            saveUiState();
        }
        return;
    }

}

function beginPanelDrag(event) {
    if (state.ui.layout !== "float") return;
    if (event.target.closest("button") || event.target.closest("input")) return;
    state.dragging = true;
    state.dragStart = {
        pointerId: event.pointerId,
        x: event.clientX,
        y: event.clientY,
        left: state.ui.left,
        top: state.ui.top,
    };
    state.panel.setPointerCapture?.(event.pointerId);
    document.addEventListener("pointermove", movePanelDrag);
    document.addEventListener("pointerup", endPanelDrag, { once: true });
}

function movePanelDrag(event) {
    if (!state.dragging || !state.dragStart) return;
    const dx = event.clientX - state.dragStart.x;
    const dy = event.clientY - state.dragStart.y;
    state.ui.left = clamp(state.dragStart.left + dx, 8, Math.max(8, window.innerWidth - 120));
    state.ui.top = clamp(state.dragStart.top + dy, 8, Math.max(8, window.innerHeight - 60));
    applyPanelLayout();
}

function endPanelDrag() {
    state.dragging = false;
    state.dragStart = null;
    document.removeEventListener("pointermove", movePanelDrag);
    capturePanelSize();
    saveUiState();
}

function capturePanelSize() {
    if (!state.panel || state.ui.layout === "rail") return;
    const rect = state.panel.getBoundingClientRect();
    if (state.ui.layout === "float") {
        state.ui.width = Math.round(rect.width);
        state.ui.height = Math.round(rect.height);
    } else if (state.ui.layout === "bottom") {
        state.ui.bottomHeight = Math.round(rect.height);
    }
    saveUiState();
}

function applyPanelLayout() {
    if (!state.panel) return;
    const panel = state.panel;
    panel.dataset.layout = state.ui.layout;
    panel.classList.toggle("is-collapsed", Boolean(state.ui.collapsed));
    panel.style.left = `${state.ui.left}px`;
    panel.style.top = `${state.ui.top}px`;
    panel.style.width = `${state.ui.width}px`;
    panel.style.height = `${state.ui.layout === "bottom" ? state.ui.bottomHeight : state.ui.height}px`;
    panel.querySelector('[data-action="pin"]')?.classList.toggle("is-active", Boolean(state.ui.persistent));
    panel.querySelector('[data-action="collapse"]')?.classList.toggle("is-active", Boolean(state.ui.collapsed));
}

function renderBookmarkButton(bookmark, index, selected = false) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `iamccs-nav-bookmark${selected ? " is-selected" : ""}`;
    button.dataset.bookmarkId = bookmark.id;
    button.title = bookmark.note ? `${bookmark.name} - ${bookmark.note}` : bookmark.name;

    const dot = document.createElement("span");
    dot.className = "iamccs-nav-dot";
    dot.style.background = bookmark.color;
    button.appendChild(dot);

    const name = document.createElement("span");
    name.className = "iamccs-nav-name";
    name.textContent = bookmark.icon ? `${bookmark.icon} ${bookmark.name}` : bookmark.name;
    button.appendChild(name);

    const order = document.createElement("span");
    order.className = "iamccs-nav-order";
    order.textContent = index < 9 ? String(index + 1) : "";
    button.appendChild(order);

    if (bookmark.note) {
        const note = document.createElement("span");
        note.className = "iamccs-nav-note";
        note.textContent = bookmark.note;
        button.appendChild(note);
    }
    button.addEventListener("click", () => jumpToBookmark(bookmark));
    return button;
}

function renderGroupedBookmarks(container, bookmarks, selectedIndex = -1) {
    container.replaceChildren();
    if (!bookmarks.length) {
        const empty = document.createElement("div");
        empty.className = "iamccs-nav-empty";
        empty.innerHTML = `
            <strong>No Navigator bookmarks yet.</strong><br>
            Add one <em>IAMCCS Navigator</em> node near each important workflow area, rename its
            <em>bookmark_name</em>, then reopen this index with <em>Ctrl+Alt+N</em>.
        `;
        container.appendChild(empty);
        return;
    }

    let lastCategory = null;
    bookmarks.forEach((bookmark, index) => {
        const category = bookmark.category || "General";
        const rail = state.ui.layout === "bottom" || state.ui.layout === "rail";
        if (!rail && category !== lastCategory) {
            const group = document.createElement("div");
            group.className = "iamccs-nav-group";
            group.textContent = category;
            container.appendChild(group);
            lastCategory = category;
        }
        container.appendChild(renderBookmarkButton(bookmark, index, index === selectedIndex));
    });
}

function renderPalette() {
    if (!state.quickList) return;
    const bookmarks = filterBookmarks(state.quickQuery).slice(0, 30);
    state.quickSelected = clamp(state.quickSelected, 0, Math.max(0, bookmarks.length - 1));
    renderGroupedBookmarks(state.quickList, bookmarks, state.quickSelected);
}

function renderIndex() {
    if (!state.indexList) return;
    const bookmarks = filterBookmarks(state.indexQuery);
    const count = state.panel?.querySelector(".iamccs-nav-count");
    if (count) count.textContent = `${getBookmarks().length}`;
    renderGroupedBookmarks(state.indexList, bookmarks);
    applyPanelLayout();
}

function scheduleRefresh() {
    if (state.refreshQueued) return;
    state.refreshQueued = true;
    requestAnimationFrame(() => {
        state.refreshQueued = false;
        if (state.paletteOpen) renderPalette();
        if (state.panelOpen) renderIndex();
    });
}

function flashPanelMessage(message) {
    const toast = document.createElement("div");
    toast.className = "iamccs-nav-toast";
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 1200);
}

function createHelpModal() {
    if (state.helpModal) return;
    const modal = document.createElement("div");
    modal.className = "iamccs-nav-help";
    modal.innerHTML = `
        <div class="iamccs-nav-help-box">
            <div class="iamccs-nav-help-head">
                <div class="iamccs-nav-help-title">IAMCCS Navigator Help</div>
                <button class="iamccs-nav-button" type="button" data-help-close="1">x</button>
            </div>
            <div class="iamccs-nav-help-body">
                <section class="iamccs-nav-help-section">
                    <h3>Concept</h3>
                    <p>Navigator is not a processing node. It is a marker. Use many small Navigator nodes as named places in the workflow: Storyboard, Prompt Builder, Sampler, Preview, Upscale, Export.</p>
                </section>
                <section class="iamccs-nav-help-section">
                    <h3>First Setup</h3>
                    <ol>
                        <li>Add an <strong>IAMCCS Navigator</strong> node near an important area of the canvas.</li>
                        <li>Set <strong>bookmark_name</strong> to the name you want in the index.</li>
                        <li>Set <strong>category</strong>, <strong>color</strong>, and optionally <strong>note</strong>.</li>
                        <li>Repeat this for every area you want to reach quickly.</li>
                        <li>Press <span class="iamccs-nav-help-kbd">Ctrl</span> + <span class="iamccs-nav-help-kbd">Alt</span> + <span class="iamccs-nav-help-kbd">N</span> to open the persistent index.</li>
                    </ol>
                </section>
                <section class="iamccs-nav-help-section">
                    <h3>Daily Use</h3>
                    <ul>
                        <li><strong>Click a bookmark</strong> in the index to jump there.</li>
                        <li><strong>Ctrl+Space</strong> opens quick search. Type part of a name, press Enter.</li>
                        <li><strong>Mode</strong> cycles between floating panel, bottom strip, and compact rail.</li>
                        <li><strong>Pin</strong> remembers the index after browser reload.</li>
                        <li><strong>Back</strong> returns to the previous Navigator jump.</li>
                    </ul>
                </section>
                <section class="iamccs-nav-help-section">
                    <h3>Zoom</h3>
                    <p>Default behavior keeps your current zoom. If a bookmark must restore a specific zoom, set <strong>zoom_mode</strong> to restore saved zoom, move to the desired zoom level, then press <strong>Capture Zoom</strong> on that bookmark node.</p>
                </section>
                <section class="iamccs-nav-help-section">
                    <h3>Good Pattern</h3>
                    <p>Put Navigator nodes at the top-left or center of each logical area. Do not connect them. They should stay small and act like chapter markers for the workflow.</p>
                </section>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
    state.helpModal = modal;
    modal.addEventListener("mousedown", (event) => {
        if (event.target === modal || event.target.closest("[data-help-close]")) closeHelp();
    });
}

function openHelp() {
    createHelpModal();
    state.helpModal.classList.add("is-open");
}

function closeHelp() {
    state.helpModal?.classList.remove("is-open");
}

function refreshNodeAppearance(node) {
    if (!isNavigatorNode(node)) return;
    const bookmark = asBookmark(node);
    if (!bookmark) return;
    node.title = bookmark.name;
    node.color = bookmark.color;
    node.bgcolor = "#202225";
    markCanvasDirty();
}

function installNavigatorNodeHooks(nodeType) {
    if (nodeType.prototype._iamccsNavigatorWrapped) return;
    nodeType.prototype._iamccsNavigatorWrapped = true;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
        const result = onNodeCreated?.apply(this, args);
        if (!this._iamccsNavigatorButtons) {
            this._iamccsNavigatorButtons = true;
            this.addWidget("button", "Open Index", null, () => openPanel());
            this.addWidget("button", "Help", null, () => openHelp());
            this.addWidget("button", "Capture Zoom", null, () => {
                const view = currentView();
                if (view) setWidgetValue(this, "saved_zoom", Number(view.zoom.toFixed(2)));
            });
            this.addWidget("button", "Test Jump Here", null, () => jumpToBookmark(asBookmark(this), false));
        }
        for (const widget of this.widgets || []) {
            if (widget._iamccsNavigatorWatched) continue;
            widget._iamccsNavigatorWatched = true;
            const original = widget.callback;
            const node = this;
            widget.callback = function (...widgetArgs) {
                const callbackResult = original?.apply(this, widgetArgs);
                refreshNodeAppearance(node);
                scheduleRefresh();
                return callbackResult;
            };
        }
        refreshNodeAppearance(this);
        scheduleRefresh();
        return result;
    };

    const onAdded = nodeType.prototype.onAdded;
    nodeType.prototype.onAdded = function (...args) {
        const result = onAdded?.apply(this, args);
        refreshNodeAppearance(this);
        scheduleRefresh();
        return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (...args) {
        const result = onConfigure?.apply(this, args);
        refreshNodeAppearance(this);
        scheduleRefresh();
        return result;
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function (...args) {
        scheduleRefresh();
        return onRemoved?.apply(this, args);
    };
}

app.registerExtension({
    name: "IAMCCS.Navigator",
    async setup() {
        injectStyles();
        createPalette();
        createPanel();
        document.addEventListener("keydown", handleGlobalKeydown, true);
        if (state.ui.persistent) openPanel();
        console.info(`[IAMCCS Navigator] frontend ${VERSION} ready`);
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const name = String(nodeData?.name || nodeData?.class_type || "");
        if (name === NAV_NODE_TYPE) installNavigatorNodeHooks(nodeType);
    },
    async nodeCreated(node) {
        if (isNavigatorNode(node)) {
            refreshNodeAppearance(node);
            scheduleRefresh();
        }
    },
    async loadedGraphNode(node) {
        if (isNavigatorNode(node)) {
            refreshNodeAppearance(node);
            scheduleRefresh();
        }
    },
});
