// SPDX-License-Identifier: GPL-3.0-or-later
import { app } from "/scripts/app.js";

const PANEL_SELECTOR = ".iamccs-h3pro";
const MAIN_SELECTOR = ".h3p-main";
const GRID_SELECTOR = "[data-grid]";
const ASSISTANT_SECTION = "assistant";
const STYLE_ID = "iamccs-h3-settings-pro-assistant-scroll-v3-style";
const MAIN_CLASS = "iamccs-h3p-assistant-scroll-owner";
const SCROLLER_CLASS = "iamccs-h3p-assistant-scroll";

const panelState = new WeakMap();

function ensureStyle() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    ${PANEL_SELECTOR} ${MAIN_SELECTOR}.${MAIN_CLASS} {
      display: flex !important;
      flex-direction: column !important;
      min-height: 0 !important;
      overflow: hidden !important;
    }
    ${PANEL_SELECTOR} ${MAIN_SELECTOR}.${MAIN_CLASS} > .h3p-section-title,
    ${PANEL_SELECTOR} ${MAIN_SELECTOR}.${MAIN_CLASS} > .h3p-section-note,
    ${PANEL_SELECTOR} ${MAIN_SELECTOR}.${MAIN_CLASS} > .h3p-context,
    ${PANEL_SELECTOR} ${MAIN_SELECTOR}.${MAIN_CLASS} > .h3p-recipes {
      flex: 0 0 auto;
    }
    ${PANEL_SELECTOR} ${MAIN_SELECTOR}.${MAIN_CLASS} > ${GRID_SELECTOR}.${SCROLLER_CLASS} {
      flex: 1 1 0 !important;
      height: 0 !important;
      max-height: 100% !important;
      min-height: 0 !important;
      overflow-y: auto !important;
      overflow-x: hidden !important;
      overscroll-behavior: contain !important;
      scrollbar-gutter: stable;
      pointer-events: auto !important;
      touch-action: pan-y;
    }
  `;
  (document.head || document.documentElement).append(style);
}

function wheelDeltaPixels(event, scroller) {
  if (event.deltaMode === WheelEvent.DOM_DELTA_LINE) return event.deltaY * 16;
  if (event.deltaMode === WheelEvent.DOM_DELTA_PAGE) return event.deltaY * Math.max(1, scroller.clientHeight);
  return event.deltaY;
}

function bindScroller(scroller, state) {
  if (state.scroller === scroller && state.bound) return;
  state.scroller = scroller;
  state.bound = true;

  scroller.addEventListener("scroll", () => {
    if (state.restoring) return;
    state.scrollTop = scroller.scrollTop;
  }, { passive: true });

  scroller.addEventListener("wheel", (event) => {
    if (!scroller.classList.contains(SCROLLER_CLASS)) return;

    // A vertical wheel/trackpad gesture belongs to the Mode Assistant, not to
    // LiteGraph's canvas zoom/pan handlers. Drive scrollTop ourselves so this
    // remains reliable even when the canvas installs capture-phase listeners.
    if (Math.abs(event.deltaY) < Math.abs(event.deltaX)) {
      event.stopPropagation();
      return;
    }

    const maxScroll = Math.max(0, scroller.scrollHeight - scroller.clientHeight);
    if (maxScroll <= 0) {
      event.stopPropagation();
      return;
    }

    const delta = wheelDeltaPixels(event, scroller);
    const next = Math.max(0, Math.min(maxScroll, scroller.scrollTop + delta));
    event.preventDefault();
    event.stopPropagation();
    scroller.scrollTop = next;
    state.scrollTop = next;
  }, { passive: false });
}

function restoreScroll(scroller, state) {
  const maxScroll = Math.max(0, scroller.scrollHeight - scroller.clientHeight);
  const target = Math.max(0, Math.min(maxScroll, Number(state.scrollTop || 0)));
  state.restoring = true;
  scroller.scrollTop = target;
  state.restoring = false;
}

function syncPanel(root) {
  const state = panelState.get(root);
  if (!state) return;

  const main = root.querySelector(MAIN_SELECTOR);
  const grid = root.querySelector(GRID_SELECTOR);
  if (!main || !grid) return;

  const assistantActive = String(main.dataset.section || "") === ASSISTANT_SECTION;
  main.classList.toggle(MAIN_CLASS, assistantActive);
  grid.classList.toggle(SCROLLER_CLASS, assistantActive);

  bindScroller(grid, state);

  if (assistantActive) {
    // renderAssistant() can replace all children synchronously. Restoring on
    // the next frame happens after layout has a real scrollHeight again.
    cancelAnimationFrame(state.restoreFrame || 0);
    state.restoreFrame = requestAnimationFrame(() => restoreScroll(grid, state));
  }
}

function installPanel(root) {
  if (!(root instanceof HTMLElement) || panelState.has(root)) return;

  const state = {
    scrollTop: 0,
    scroller: null,
    bound: false,
    restoreFrame: 0,
    observer: null,
  };
  panelState.set(root, state);

  state.observer = new MutationObserver(() => syncPanel(root));
  state.observer.observe(root, {
    subtree: true,
    childList: true,
    attributes: true,
    attributeFilter: ["class", "data-section"],
  });

  syncPanel(root);
}

function scanNode(node) {
  if (!(node instanceof Element)) return;
  if (node.matches?.(PANEL_SELECTOR)) installPanel(node);
  node.querySelectorAll?.(PANEL_SELECTOR).forEach(installPanel);
}

app.registerExtension({
  name: "IAMCCS.H3SettingsPro.AssistantScrollV3",
  async setup() {
    ensureStyle();
    document.querySelectorAll(PANEL_SELECTOR).forEach(installPanel);

    const host = document.body || document.documentElement;
    const observer = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        for (const node of mutation.addedNodes) scanNode(node);
      }
    });
    observer.observe(host, { childList: true, subtree: true });
  },
});
