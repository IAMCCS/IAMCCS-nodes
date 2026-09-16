const STATE = "h3_advisor_state";
const SAVED = "iamccs_minimax_h3_saved_settings_v1";
const widget = (node, name) => node?.widgets?.find(w => w.name === name);
const clone = value => JSON.parse(JSON.stringify(value));

export function h3TaskFamily(task) {
    if (["v2va_controlnet", "controlnet_v2v", "h3_fun_controlnet"].includes(String(task))) return "fl2va";
    return String(task).startsWith("ref2v") || ["v2va_object_swap", "v2va_face_swap"].includes(task) ? "ref2va" : "fl2va";
}

export function selectH3SpeedAsset(assets, task, role = "turbo", prefer8 = false) {
    return assets.filter(a => a.family === h3TaskFamily(task) && a.role === role && a.native && !a.error && !a.conflict)
        .sort((a, b) => Number(prefer8 && !String(a.recipe).endsWith("_8")) - Number(prefer8 && !String(b.recipe).endsWith("_8")))[0]?.name || "";
}

export function connectedH3Nodes(start) {
    const graph = start.graph;
    const visited = new Set(), pending = [start];
    while (pending.length) {
        const node = pending.pop();
        if (!node || visited.has(node)) continue;
        visited.add(node);
        for (const input of node.inputs || []) {
            const link = graph?.links?.[input.link];
            if (link) pending.push(graph.getNodeById(link.origin_id));
        }
        for (const output of node.outputs || []) for (const id of output.links || []) {
            const link = graph?.links?.[id];
            if (link) pending.push(graph.getNodeById(link.target_id));
        }
    }
    return [...visited];
}

export function advisorSnapshot(source, board) {
    const settings = {...(board?.properties?.[SAVED] || {})};
    for (const node of [board, source]) for (const w of node?.widgets || []) {
        if (w.name && w.value !== undefined && !w.element && w.type !== "button") settings[w.name] = clone(w.value);
    }
    delete settings[STATE];
    const connected = connectedH3Nodes(board || source).map(node => ({id: node.id, type: node.comfyClass || node.type, mode: node.mode || 0,
        values: Object.fromEntries((node.widgets || []).filter(w => /unet|model|vae|lora|clip|encoder|projection|task_override/.test(w.name) && w.value !== undefined).map(w => [w.name, clone(w.value)]))}));
    return {settings, connected, timeline: widget(board, "timeline_data")?.value || ""};
}

export function applyAdvisorValues(source, board, values, specs, notify) {
    const assignments = [];
    for (const [name, value] of Object.entries(values)) {
        const spec = specs[name];
        if (!spec) throw new Error(`Setting unavailable: ${name}`);
        if (Array.isArray(spec[0]) && !spec[0].includes(value)) throw new Error(`Option unavailable: ${name} = ${value}`);
        const options = spec[1] || {};
        if (typeof value === "number" && (!Number.isFinite(value) || (options.min != null && value < options.min) || (options.max != null && value > options.max))) throw new Error(`Value out of range: ${name}`);
        for (const node of new Set([source, board].filter(Boolean))) {
            const w = widget(node, name);
            if (w) assignments.push({node, w, name, value, before: clone(w.value)});
            else if (node === board && /^(h3_advisor_|h3_faceswap_|h3_sla_|h3_exact_|h3_clipproj_|h3_upres_|motion_context_window_frames$)/.test(name)) {
                assignments.push({node, name, value, before: node.properties?.[SAVED]?.[name], property: true});
            } else throw new Error(`Widget unavailable: ${name}`);
        }
    }
    const write = (item, value) => {
        if (item.property) {
            item.node.properties ||= {}; item.node.properties[SAVED] ||= {};
            if (value === undefined) delete item.node.properties[SAVED][item.name];
            else item.node.properties[SAVED][item.name] = clone(value);
        } else {
            item.w.value = clone(value);
            const index = item.node.widgets.indexOf(item.w);
            if (Array.isArray(item.node.widgets_values)) item.node.widgets_values[index] = clone(value);
        }
    };
    try {
        for (const item of assignments) write(item, item.value);
        notify();
    } catch (error) {
        for (const item of assignments) write(item, item.before);
        throw error;
    }
}

export function createH3AdvisorPanel({node, api, getSource, getBoard, getSpecs, notify}) {
    const panel = document.createElement("section");
    panel.className = "iamccs-h3-advisor";
    panel.style.cssText = "padding:20px;border:1px solid #6e6651;border-radius:16px;background:linear-gradient(125deg,#293137,#151b21);color:#eee9dd;font:13px/1.6 system-ui;grid-column:1/-1;box-shadow:0 12px 30px #0003;min-width:0";
    const eyebrow=document.createElement("div"); eyebrow.textContent="IAMCCS  /  PERFORMANCE STUDIO"; eyebrow.style.cssText="font-size:10px;letter-spacing:2px;color:#cfb982;margin-bottom:12px"; panel.append(eyebrow);
    const line = document.createElement("label"), toggle = document.createElement("input"), title = document.createElement("strong");
    toggle.type = "checkbox"; toggle.setAttribute("aria-label", "Hardware advisor for current mode");
    line.append(toggle, title);
    line.style.cssText="display:flex;align-items:center;gap:10px;font-size:16px;margin-bottom:14px";
    const body = document.createElement("div");
    const targetLabel = document.createElement("label"); targetLabel.textContent = "Final long edge (px) ";
    const target = document.createElement("input"); target.type = "number"; target.min = "256"; target.max = "5760"; target.step = "32"; target.style.width = "85px";
    target.setAttribute("aria-label", "Advisor final long edge"); targetLabel.append(target);
    const analyze = document.createElement("button"); analyze.textContent = "Analyze server / proposals";
    const undo = document.createElement("button"); undo.textContent = "Undo last application";
    for (const button of [analyze, undo]) { button.type = "button"; button.style.cssText = "margin:6px;padding:10px 14px;cursor:pointer;border:1px solid #8b7956;border-radius:8px;background:#38382e;color:#f6edda"; }
    const status = document.createElement("div"); status.setAttribute("role", "status"); status.style.cssText="padding:14px;border-radius:10px;background:#ffffff08;margin:12px 0;overflow-wrap:anywhere";
    const results = document.createElement("div");
    body.append(targetLabel, analyze, undo, status, results); panel.append(line, body);
    let proposal = null, busy = false, lastMode = "";
    const context = () => advisorSnapshot(getSource(), getBoard());
    const mode = () => `${context().settings.task_mode || "auto"}:${context().settings.audio_mode || "h3_native_generated"}`;
    const state = () => { try { return JSON.parse(widget(getSource(), STATE)?.value || getBoard()?.properties?.[SAVED]?.[STATE] || "{}"); } catch { return {}; } };
    const apply = values => applyAdvisorValues(getSource(), getBoard(), values, getSpecs(), notify);
    const saveState = value => apply({[STATE]: JSON.stringify(value)});
    const refresh = () => {
        const key = mode(), entry = state()[key] || {};
        title.textContent = ` Hardware Advisor · ${key.replace(":", " / ")}`;
        toggle.checked = !!entry.enabled; body.hidden = !toggle.checked;
        if (document.activeElement !== target) target.value = String(entry.target || Math.max(1280, Number(context().settings.width || 0), Number(context().settings.height || 0)));
        undo.disabled = busy || !entry.undo;
        if (lastMode && key !== lastMode) { proposal = null; results.replaceChildren(); status.textContent = "Mode changed. Analyze again."; }
        lastMode = key;
    };
    toggle.onchange = () => { const all = state(); all[mode()] = {...all[mode()], enabled: toggle.checked}; saveState(all); refresh(); };
    target.onchange = () => { const all = state(); all[mode()] = {...all[mode()], target: Number(target.value)}; saveState(all); proposal = null; results.replaceChildren(); refresh(); };
    const requestAdvice = async snapshot => {
        const response = await api.fetchApi("/api/iamccs/h3/advice", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({...snapshot, target_long_edge: Number(target.value)})});
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
        return data;
    };
    analyze.onclick = async () => {
        if (busy) return;
        busy = true; analyze.disabled = true; results.replaceChildren(); status.textContent = "Reading server hardware and asset headers…";
        try {
            const snapshot = context(), fingerprint = JSON.stringify(snapshot), key = mode();
            const data = await requestAdvice(snapshot);
            if (fingerprint !== JSON.stringify(context()) || key !== mode()) throw new Error("Settings changed during analysis. Analyze again.");
            proposal = {data, fingerprint, key};
            const hw = data.hardware, gib = bytes => (bytes / 2 ** 30).toFixed(1);
            status.replaceChildren();
            const hardwareGrid=document.createElement("div");hardwareGrid.style.cssText="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px";
            for (const [label,value] of [["GPU / SERVER",hw.name],["FREE / TOTAL VRAM",`${gib(hw.vram_free)} / ${gib(hw.vram_total)} GiB`],["FREE / TOTAL RAM",`${gib(hw.ram_free)} / ${gib(hw.ram_total)} GiB`],["MODEL FAMILY",data.family]]) {
                const tile=document.createElement("div");tile.style.cssText="padding:12px;background:#ffffff06;border-radius:8px";
                const key=document.createElement("div");key.textContent=label;key.style.cssText="color:#c4b68e;font-size:10px;letter-spacing:1px";
                const val=document.createElement("strong");val.textContent=value;tile.append(key,val);hardwareGrid.append(tile);
            }
            status.append(hardwareGrid);
            const availability = document.createElement("p");
            availability.textContent = `${data.proposals.length} proposals available · memory path: ${data.memory_backend || "see proposal"} · ${data.rtx_delivery_available ? "RTX branch available" : "native output"}. ${data.warnings.length ? "Review the warnings for the selected configuration." : "No configuration warning; output quality and render time still require a measured render."}`;
            results.append(availability);
            const inventory = document.createElement("details"), heading = document.createElement("summary"), listing = document.createElement("pre");
            heading.textContent = "Installed assets and connected model loaders";
            listing.style.cssText = "font-size:10px;white-space:pre-wrap;overflow-wrap:anywhere";
            listing.textContent = [
                ...snapshot.connected.filter(n => Object.keys(n.values).length).map(n => `${n.type}: ${JSON.stringify(n.values)}`),
                ...data.assets.map(a => `${a.name}: ${a.role} / ${a.family || "unverified family"} / ${a.recipe || "manual recipe"}${a.error ? " / " + a.error : ""}`),
                ...data.models.map(m => `${m.category}: ${m.name} (${gib(m.bytes)} GiB file)`),
            ].join("\n");
            inventory.append(heading, listing); results.append(inventory);
            if (data.compatibility_notes?.length) {
                const details = document.createElement("details"), summary = document.createElement("summary");
                summary.textContent = "Compatibility notes for alternatives and unselected components";
                details.append(summary);
                for (const message of data.compatibility_notes) { const text = document.createElement("p"); text.textContent = message; details.append(text); }
                results.append(details);
            }
            for (const warning of data.warnings) { const text = document.createElement("p"); text.textContent = warning; results.append(text); }
            for (const option of data.proposals) {
                const card = document.createElement("details"); card.style.cssText = "padding:16px;margin:12px 0;border:1px solid #616453;border-radius:12px;background:#11182088";
                const summary = document.createElement("summary"); summary.textContent = `${option.label} · ${option.values.width}×${option.values.height} native · sampling pixels ×${option.sampling_load_relative}`;
                const diff = document.createElement("pre"); diff.style.cssText = "font-size:11px;white-space:pre-wrap";
                diff.textContent = Object.entries(option.values).filter(([k, v]) => snapshot.settings[k] !== v).map(([k, v]) => `${k}: ${snapshot.settings[k] ?? "default"} → ${v}`).join("\n") || "Current values already match.";
                const notes = document.createElement("p"); notes.textContent = option.notes.join("\n");
                const accept = document.createElement("button"); accept.type = "button"; accept.textContent = "Apply these changes";
                accept.onclick = async () => {
                    if (busy) return;
                    busy = true; accept.disabled = true;
                    try {
                        if (!proposal || proposal.fingerprint !== JSON.stringify(context())) throw new Error("Proposal expired: settings, media or graph changed. Analyze again.");
                        const fresh = await requestAdvice(context());
                        if (fresh.inventory_revision !== data.inventory_revision || fresh.hardware.vram_free < data.hardware.vram_free * 0.8) throw new Error("Server assets or available memory changed. Analyze again.");
                        if (proposal.fingerprint !== JSON.stringify(context())) throw new Error("Settings changed during verification. Analyze again.");
                        const all = state(), before = {};
                        for (const name of Object.keys(option.values)) before[name] = snapshot.settings[name] ?? getSpecs()[name]?.[1]?.default;
                        all[mode()] = {...all[mode()], undo: {before, after: option.values}};
                        apply({...option.values, [STATE]: JSON.stringify(all)});
                        proposal = null; results.replaceChildren(); status.textContent = "Applied and synchronized. Run a short render to measure this configuration.";
                    } catch (error) { status.textContent = error.message; }
                    finally { busy = false; accept.disabled = false; refresh(); }
                };
                card.append(summary, diff, notes, accept); results.append(card);
            }
        } catch (error) { status.textContent = error.message; }
        finally { busy = false; analyze.disabled = false; refresh(); }
    };
    undo.onclick = () => {
        try {
            const all = state(), key = mode(), saved = all[key]?.undo;
            if (!saved) return;
            const now = context().settings;
            if (Object.entries(saved.after).some(([name, value]) => now[name] !== value)) throw new Error("A proposed field was edited after application; undo would overwrite that edit.");
            delete all[key].undo; apply({...saved.before, [STATE]: JSON.stringify(all)});
            proposal = null; results.replaceChildren(); status.textContent = "Previous settings restored and synchronized."; refresh();
        } catch (error) { status.textContent = error.message; }
    };
    node._iamccsRefreshH3Advisor = refresh;
    refresh();
    return panel;
}
