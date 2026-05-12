/**
 * iamccs_narrative_planner_ui.js
 * By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
 *
 * UI extension for IAMCCS_CineNarrativePlanner node.
 * Adds three action buttons to the node:
 *   🔍 Match Images   — sends images to CLIP endpoint, updates ref_map preview
 *   ⚙  Generate Plan  — calls CPU narrative parser, shows preview panel
 *   →  Push to PlannerPro — writes generated rows into the connected PlannerPro's
 *                           timeline_data widget so they appear in the shotboard table
 *
 * By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NARRATIVE_VERSION = "2026-05-08-v1";
const NARRATIVE_NODE_CLASS = "IAMCCS_CineNarrativePlanner";
const PLANNER_NODE_CLASS = "IAMCCS_CineShotboardPlannerPro";
const PLANNER_NODE_CLASSES = new Set([PLANNER_NODE_CLASS, "IAMCCS_CineShotboardPlannerProLegacy"]);
const CINE_FILM_LAB = {
    header: "#2E2A24",
    nodeBg: "#171512",
    panelDark: "#15120F",
    border: "#4A4032",
    text: "#F1E8D2",
    muted: "#B8A98E",
    guide: "#D6A85A",
    relay: "#5FA8C7",
    active: "#7FAE7A",
};

// ── Helpers ──────────────────────────────────────────────────────────────────

function nodeClassName(node) {
    return String(node?.comfyClass || node?.type || node?.constructor?.comfyClass || "");
}

function getWidget(node, name) {
    return node?.widgets?.find((w) => w?.name === name || w?.label === name) || null;
}

function setWidgetValue(node, name, value) {
    const widget = getWidget(node, name);
    if (!widget) return false;
    widget.value = value;
    if (typeof widget.callback === "function") {
        try { widget.callback(value); } catch (_) {}
    }
    node.setDirtyCanvas(true, true);
    return true;
}

function parseImagePaths(rawValue) {
    const raw = String(rawValue || "").trim();
    if (!raw) return [];
    try {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) {
            return parsed.map((item) => String(item || "").trim()).filter(Boolean);
        }
        if (parsed && Array.isArray(parsed.paths)) {
            return parsed.paths.map((item) => String(item || "").trim()).filter(Boolean);
        }
        if (parsed && Array.isArray(parsed.images)) {
            return parsed.images.map((item) => String(item?.path || item || "").trim()).filter(Boolean);
        }
    } catch (_) {}
    return raw.split(/\n|,/).map((s) => s.trim()).filter(Boolean);
}

function narrativeRequestBody(node) {
    return {
        narrative: getWidget(node, "narrative")?.value || "",
        global_prompt: getWidget(node, "global_prompt")?.value || "",
        duration_seconds: parseFloat(getWidget(node, "duration_seconds")?.value) || 12,
        frame_rate: parseInt(getWidget(node, "frame_rate")?.value) || 24,
        guide_density: getWidget(node, "guide_density")?.value || "medium",
        force_profile: getWidget(node, "force_profile")?.value || "gradual_release",
        relay_mode: getWidget(node, "relay_mode")?.value || "disabled",
        relay_style: getWidget(node, "relay_style")?.value || "short_motion",
        visual_analysis_text: getWidget(node, "visual_analysis_text")?.value || "",
        vlm_prompt_target: getWidget(node, "vlm_prompt_target")?.value || "notes_only",
    };
}

/**
 * Find the PlannerPro node connected to the narrative node's output.
 * Follows any link on the timeline_rows_json output (output 1) or
 * falls back to searching the graph for any PlannerPro node.
 * By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
 */
function findConnectedPlannerPro(narrativeNode) {
    const graph = app.graph;
    if (!graph) return null;

    // Check output[1] = timeline_rows_json links
    const outputs = narrativeNode.outputs || [];
    for (const out of outputs) {
        const links = out.links || [];
        for (const linkId of links) {
            const link = graph.links ? graph.links[linkId] : null;
            if (!link) continue;
            const targetNode = graph.getNodeById(link.target_id);
            if (targetNode && PLANNER_NODE_CLASSES.has(nodeClassName(targetNode))) {
                return targetNode;
            }
        }
    }

    // Also check output[0] = cine_linx
    for (const out of outputs) {
        const links = out.links || [];
        for (const linkId of links) {
            const link = graph.links ? graph.links[linkId] : null;
            if (!link) continue;
            const targetNode = graph.getNodeById(link.target_id);
            if (targetNode && PLANNER_NODE_CLASSES.has(nodeClassName(targetNode))) {
                return targetNode;
            }
        }
    }

    // Fallback: find the first PlannerPro on the canvas
    if (graph._nodes) {
        for (const n of graph._nodes) {
            if (PLANNER_NODE_CLASSES.has(nodeClassName(n))) return n;
        }
    }
    return null;
}

/**
 * Find the CineReferenceBoard node that provides images for this narrative planner.
 * Strategy (in order):
 *  1. IMAGE input on the NarrativePlanner itself (direct connection)
 *  2. multi_input of the connected PlannerPro
 *  3. Any IAMCCS_CineReferenceBoard on the canvas
 * By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
 */
function findReferenceBoard(narrativeNode) {
    const graph = app.graph;
    if (!graph) return null;

    // 1. Direct IMAGE input on NarrativePlanner
    for (const inp of (narrativeNode.inputs || [])) {
        if (inp.type !== "IMAGE" || !inp.link) continue;
        const link = graph.links?.[inp.link];
        if (!link) continue;
        const src = graph.getNodeById(link.origin_id);
        if (src && getWidget(src, "image_paths")) return src;
    }

    // 2. multi_input of connected PlannerPro
    const planner = findConnectedPlannerPro(narrativeNode);
    if (planner) {
        for (const inp of (planner.inputs || [])) {
            if (inp.name !== "multi_input" || !inp.link) continue;
            const link = graph.links?.[inp.link];
            if (!link) continue;
            const src = graph.getNodeById(link.origin_id);
            if (src && getWidget(src, "image_paths")) return src;
        }
    }

    // 3. Any CineReferenceBoard on canvas
    if (graph._nodes) {
        for (const n of graph._nodes) {
            if (nodeClassName(n) === "IAMCCS_CineReferenceBoard") return n;
        }
    }
    return null;
}

/**
 * Read image_paths from a ReferenceBoard node and fetch each as base64 PNG.
 * Uses /api/iamccs/cine/view_image — the same endpoint used by the Shotboard UI.
 * By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
 */
async function collectImageBase64List(narrativeNode) {
    const b64List = [];
    const refBoard = findReferenceBoard(narrativeNode);
    if (!refBoard) return b64List;

    const pathsWidget = getWidget(refBoard, "image_paths");
    if (!pathsWidget || !pathsWidget.value) return b64List;

    const paths = parseImagePaths(pathsWidget.value);
    for (const path of paths) {
        try {
            const resp = await api.fetchApi(`/api/iamccs/cine/view_image?path=${encodeURIComponent(path)}`);
            if (!resp.ok) continue;
            const blob = await resp.blob();
            const b64 = await new Promise((resolve) => {
                const reader = new FileReader();
                reader.onloadend = () => resolve(reader.result);
                reader.readAsDataURL(blob);
            });
            b64List.push(b64);
        } catch (_) {}
    }
    return b64List;
}

/**
 * Get the raw image_paths string from the CineReferenceBoard connected to narrativeNode.
 * By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
 */
function getReferenceBoardImagePaths(narrativeNode) {
    const refBoard = findReferenceBoard(narrativeNode);
    if (!refBoard) return "";
    return parseImagePaths(getWidget(refBoard, "image_paths")?.value || "").join("\n");
}

// ── Button builders ───────────────────────────────────────────────────────────

function makeActionButton(label, title, colorHex) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = label;
    btn.title = title;
    btn.style.cssText = [
        "height:30px",
        "padding:0 10px",
        `border:1px solid ${colorHex}44`,
        "border-radius:5px",
        `background:${colorHex}22`,
        `color:${colorHex}`,
        "font-size:12px",
        "font-weight:600",
        "cursor:pointer",
        "white-space:nowrap",
        "flex:1",
        "min-width:0",
        "transition:background 0.15s",
    ].join(";");
    btn.onmouseenter = () => { btn.style.background = `${colorHex}44`; };
    btn.onmouseleave = () => { btn.style.background = `${colorHex}22`; };
    return btn;
}

function makeStatusText() {
    const el = document.createElement("div");
    el.style.cssText = `font-size:11px;color:${CINE_FILM_LAB.muted};padding:4px 2px;min-height:16px;word-break:break-word;`;
    return el;
}

// ── Preview panel ─────────────────────────────────────────────────────────────

/**
 * Build a compact rows preview showing the generated plan.
 * By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
 */
function buildPreviewPanel(rows) {
    const container = document.createElement("div");
    container.style.cssText = `margin-top:6px;font-size:11px;font-family:monospace;max-height:200px;overflow-y:auto;background:${CINE_FILM_LAB.panelDark};border:1px solid ${CINE_FILM_LAB.border};border-radius:5px;padding:6px;`;

    for (const row of rows) {
        const line = document.createElement("div");
        line.style.cssText = `display:grid;grid-template-columns:38px 26px 48px 1fr;gap:4px;align-items:start;padding:2px 0;border-bottom:1px solid ${CINE_FILM_LAB.border};`;

        const secEl = document.createElement("span");
        secEl.style.color = CINE_FILM_LAB.relay;
        secEl.textContent = `${Number(row.second || 0).toFixed(1)}s`;

        const refEl = document.createElement("span");
        refEl.style.color = CINE_FILM_LAB.guide;
        refEl.textContent = `r${row.ref || 1}`;

        const forceEl = document.createElement("span");
        forceEl.style.color = CINE_FILM_LAB.active;
        forceEl.textContent = `f${Number(row.force || 0).toFixed(2)}`;

        const noteEl = document.createElement("span");
        noteEl.style.color = CINE_FILM_LAB.text;
        noteEl.style.overflow = "hidden";
        noteEl.style.whiteSpace = "nowrap";
        noteEl.style.textOverflow = "ellipsis";
        noteEl.title = row.relay_prompt || row.note || "";
        noteEl.textContent = row.label || row.note || "";

        line.append(secEl, refEl, forceEl, noteEl);
        container.appendChild(line);
    }
    return container;
}

// ── Main extension ─────────────────────────────────────────────────────────────

app.registerExtension({
    name: "IAMCCS.CineNarrativePlannerUI",

    async nodeCreated(node) {
        if (nodeClassName(node) !== NARRATIVE_NODE_CLASS) return;
        node.color = CINE_FILM_LAB.header;
        node.bgcolor = CINE_FILM_LAB.nodeBg;
        node.boxcolor = CINE_FILM_LAB.active;

        // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        // State held on the node instance
        node._narrativePlan = null;   // last generated {rows, timeline_data_json, clauses}
        node._refAssignments = null;  // last CLIP match result

        // ── Container widget ──────────────────────────────────────────────────
        const container = document.createElement("div");
        container.style.cssText = "padding:6px 4px 4px 4px;display:flex;flex-direction:column;gap:5px;";

        // Button row
        const btnRow = document.createElement("div");
        btnRow.style.cssText = "display:flex;gap:5px;";

        const btnMatch    = makeActionButton("🔍 Match Images",     "Use CLIP to match each action clause to a reference image index", "#4fc3f7");
        const btnGenerate = makeActionButton("⚙  Generate Plan",    "Parse the narrative and generate shotboard rows (CPU only)", "#81c784");
        const btnPush     = makeActionButton("→ Push to PlannerPro","Write the generated rows into the connected PlannerPro timeline_data widget", "#ffb300");

        btnRow.append(btnMatch, btnGenerate, btnPush);
        container.appendChild(btnRow);

        // Status line
        const statusEl = makeStatusText();
        container.appendChild(statusEl);

        // Preview panel (initially hidden)
        let previewPanel = null;
        const clearPreview = () => {
            if (previewPanel && previewPanel.parentNode) previewPanel.parentNode.removeChild(previewPanel);
            previewPanel = null;
        };

        // ── Match Images ──────────────────────────────────────────────────────
        btnMatch.addEventListener("click", async (e) => {
            e.preventDefault();
            statusEl.textContent = "🔍 Collecting images…";
            btnMatch.disabled = true;
            try {
                const b64List = await collectImageBase64List(node);
                if (b64List.length === 0) {
                    const refBoard = findReferenceBoard(node);
                    if (!refBoard) {
                        statusEl.textContent = "⚠ No CineReferenceBoard found. Connect one to the images input, or to the PlannerPro multi_input.";
                    } else {
                        statusEl.textContent = "⚠ CineReferenceBoard found but has no image_paths. Load images in it first.";
                    }
                    return;
                }

                // Get current clauses via a quick generate call first (CPU)
                const genResp = await api.fetchApi("/api/iamccs/narrative/generate_plan", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(narrativeRequestBody(node)),
                });
                const genData = await genResp.json();
                if (!genData.ok) throw new Error(genData.error || "generate_plan failed");
                const clauses = genData.clauses || [];

                statusEl.textContent = `🔍 Matching ${clauses.length} clauses to ${b64List.length} images via CLIP…`;

                const matchResp = await api.fetchApi("/api/iamccs/narrative/match_images", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ clauses, image_b64_list: b64List }),
                });
                const matchData = await matchResp.json();
                if (!matchData.ok) throw new Error(matchData.error || "match_images failed");

                node._refAssignments = matchData.ref_assignments || [];
                const method = matchData.method || "unknown";

                const preview = clauses.map((c, i) => `ref:${node._refAssignments[i]} → "${c}"`).join("\n");
                statusEl.textContent = `✓ Matched (${method}). Press ⚙ Generate Plan to apply.`;
                statusEl.title = preview;
            } catch (err) {
                statusEl.textContent = `✗ Match failed: ${err.message}`;
            } finally {
                btnMatch.disabled = false;
            }
        });

        // ── Generate Plan ──────────────────────────────────────────────────────
        btnGenerate.addEventListener("click", async (e) => {
            e.preventDefault();
            statusEl.textContent = "⚙ Generating plan…";
            btnGenerate.disabled = true;
            clearPreview();
            try {
                const body = narrativeRequestBody(node);

                const resp = await api.fetchApi("/api/iamccs/narrative/generate_plan", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(body),
                });
                const data = await resp.json();
                if (!data.ok) throw new Error(data.error || "generate_plan failed");

                // Apply CLIP ref assignments if available
                let rows = data.rows || [];
                if (node._refAssignments && node._refAssignments.length === rows.length) {
                    rows = rows.map((row, i) => ({ ...row, ref: node._refAssignments[i] }));
                }

                // Rebuild timeline_data_json with updated refs
                const timelineMeta = {
                    ...(data.timeline_meta || {}),
                    duration_seconds: parseFloat(getWidget(node, "duration_seconds")?.value) || 12,
                    frame_rate: parseInt(getWidget(node, "frame_rate")?.value) || 24,
                    global_prompt: data.global_prompt || getWidget(node, "global_prompt")?.value || "",
                };
                const timeline_data_json = JSON.stringify({ rows, meta: timelineMeta }, null, 2);
                node._narrativePlan = {
                    rows,
                    timeline_data_json,
                    timeline_meta: timelineMeta,
                    clauses: data.clauses || [],
                    global_prompt: timelineMeta.global_prompt,
                };

                // Show preview
                previewPanel = buildPreviewPanel(rows);
                container.appendChild(previewPanel);
                node.setSize([Math.max(node.size[0], 500), node.size[1]]);

                statusEl.textContent = `✓ ${rows.length} rows generated. Press → Push to PlannerPro to apply.`;
            } catch (err) {
                statusEl.textContent = `✗ Generate failed: ${err.message}`;
            } finally {
                btnGenerate.disabled = false;
            }
        });

        // ── Push to PlannerPro ────────────────────────────────────────────────
        // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        btnPush.addEventListener("click", async (e) => {
            // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
            e.preventDefault();
            if (!node._narrativePlan) {
                statusEl.textContent = "⚠ No plan generated yet. Press ⚙ Generate Plan first.";
                return;
            }

            const plannerNode = findConnectedPlannerPro(node);
            if (!plannerNode) {
                statusEl.textContent = "⚠ No PlannerPro found. Connect cine_linx or timeline_rows_json output to a CineShotboardPlannerPro node.";
                return;
            }

            // ── guide_density → max_guides ────────────────────────────────────
            const GUIDE_DENSITY_MAP = { low: 3, medium: 5, high: 50 };
            const guideDensity = String(getWidget(node, "guide_density")?.value || "medium");
            const maxGuides = GUIDE_DENSITY_MAP[guideDensity] ?? 5;

            // ── force_profile → default_force (first anchor of the curve) ─────
            // The first anchor = the strength at second=0, the maximum grip.
            // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
            const FORCE_PROFILE_FIRST = {
                strong_anchors:  1.00,
                gradual_release: 0.95,
                soft_journey:    0.85,
                free_float:      0.78,
                flat:            0.38,
            };
            const forceProfile = String(getWidget(node, "force_profile")?.value || "gradual_release");
            const defaultForce = FORCE_PROFILE_FIRST[forceProfile] ?? 0.95;

            // ── Collect values from NarrativePlanner widgets ──────────────────
            const globalPrompt      = String(node._narrativePlan?.global_prompt || getWidget(node, "global_prompt")?.value || "");
            const durationSeconds   = parseFloat(getWidget(node, "duration_seconds")?.value) || 12.0;
            const frameRate         = parseInt(getWidget(node, "frame_rate")?.value) || 24;
            const imageWidth        = parseInt(getWidget(node, "image_width")?.value) || 768;
            const imageHeight       = parseInt(getWidget(node, "image_height")?.value) || 432;
            const relayEpsilon      = parseFloat(getWidget(node, "promptrelay_epsilon")?.value) || 0.65;
            const ltxRoundMode      = String(getWidget(node, "ltx_round_mode")?.value || "up_8n_plus_1");

            // ── Push timeline rows (already contains per-row relay_prompts) ───
            const { timeline_data_json, rows } = node._narrativePlan;
            const ok = setWidgetValue(plannerNode, "timeline_data", timeline_data_json);
            if (!ok) {
                statusEl.textContent = `✗ Could not write timeline_data on PlannerPro #${plannerNode.id}.`;
                return;
            }

            // ── Push all configuration widgets ────────────────────────────────
            setWidgetValue(plannerNode, "global_prompt",       globalPrompt);
            setWidgetValue(plannerNode, "duration_seconds",    durationSeconds);
            setWidgetValue(plannerNode, "frame_rate",          frameRate);
            setWidgetValue(plannerNode, "image_width",         imageWidth);
            setWidgetValue(plannerNode, "image_height",        imageHeight);
            setWidgetValue(plannerNode, "max_guides",          maxGuides);
            setWidgetValue(plannerNode, "default_force",       defaultForce);
            setWidgetValue(plannerNode, "promptrelay_epsilon", relayEpsilon);
            setWidgetValue(plannerNode, "ltx_round_mode",      ltxRoundMode);

            // ── Push image_paths from the connected ReferenceBoard ────────────
            // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
            const imagePaths = getReferenceBoardImagePaths(node);
            let imgMsg = "";
            if (imagePaths) {
                const nImgs = imagePaths.split("\n").filter(s => s.trim()).length;
                const imgOk = setWidgetValue(plannerNode, "image_paths", imagePaths);
                imgMsg = imgOk ? ` + ${nImgs} img` : "";
            }

            // ── Build summary line ────────────────────────────────────────────
            const rowForceRange = rows.length > 0
                ? `force ${rows[0].force?.toFixed(2)}→${rows[rows.length - 1].force?.toFixed(2)}`
                : "";
            statusEl.textContent = [
                `✓ Pushed ${rows.length} rows${imgMsg}`,
                `guides=${maxGuides}`,
                rowForceRange,
                `${imageWidth}×${imageHeight}`,
                `→ PlannerPro #${plannerNode.id}`,
            ].filter(Boolean).join(" | ");

            // ── Trigger PlannerPro table re-render ────────────────────────────
            plannerNode.domElement?.dispatchEvent(new CustomEvent("iamccs:narrative_push", {
                detail: { rows, source_node_id: node.id },
            }));
            document.dispatchEvent(new CustomEvent("iamccs:planner_rows_updated", {
                detail: { node_id: plannerNode.id, rows },
            }));
        });

        // ── Attach as DOM widget ───────────────────────────────────────────────
        // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        const domWidget = node.addDOMWidget("narrative_planner_buttons", "div", container, {
            getValue() { return ""; },
            setValue() {},
            serialize: false,
        });

        // Resize node to fit the buttons
        const origSize = node.size ? [...node.size] : [520, 300];
        node.setSize([Math.max(origSize[0], 520), Math.max(origSize[1], origSize[1] + 50)]);
    },
});
