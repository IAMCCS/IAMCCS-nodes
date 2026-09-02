// VRAM classes are starting points, not benchmarks or a fit guarantee.
export const H3_WINDOW_PRESETS = [
    {id:"memory", label:"8 GB / memory first · 39f / 1.63s · overlap 5f / 0.21s · VAE 1", window:39, overlap:5, groups:1},
    {id:"speed12", label:"12 GB / speed trial · 56f / 2.33s · overlap 5f / 0.21s · VAE 2", window:56, overlap:5, groups:2},
    {id:"balanced12", label:"12 GB / balanced · 73f / 3.04s · overlap 22f / 0.92s · VAE 2", window:73, overlap:22, groups:2},
    {id:"balanced16", label:"16 GB / context · 107f / 4.46s · overlap 22f / 0.92s · VAE 3", window:107, overlap:22, groups:3},
    {id:"context24", label:"24 GB+ / long context · 158f / 6.58s · overlap 22f / 0.92s · VAE 4", window:158, overlap:22, groups:4},
    {id:"previous", label:"Previous R38B · 136 requested → 124f / 5.17s · VAE 1", window:136, overlap:22, groups:1},
];
export const windowPresetValues = p => ({h3_upres_window_frames:p.window, h3_upres_window_overlap:p.overlap, h3_upres_pixel_groups:p.groups});

// Same grid and standard_static count as the pinned provider / ComfyUI scheduler.
export function windowEstimate(totalFrames, windowFrames, overlapFrames) {
    const latent = f => Math.max(2, Math.floor((f - 5) / 17) * 5 + 2);
    const frames = t => 17 * Math.floor((t - 2) / 5) + 5;
    const total = latent(5 + 17 * Math.ceil((Math.max(5,totalFrames) - 5) / 17));
    const length = Math.min(total, latent(windowFrames));
    const overlap = Math.min(overlapFrames > 0 ? latent(overlapFrames) : 0, Math.max(0,length - 5));
    return {window:frames(length), overlap:overlap ? frames(overlap) : 0,
        count:1 + Math.ceil(Math.max(0,total - length) / (length - overlap))};
}

export function createH3WindowPanel(read, apply) {
    const root = document.createElement("section");
    root.style.cssText = "grid-column:1/-1;display:grid;gap:7px;min-width:0;padding:10px;border:1px solid #8d7ad2;background:#241f36;color:#e9e2ff";
    const title = document.createElement("strong"); title.textContent = "STAGE 2 · TEMPORAL WINDOWS / GPU BUDGET";
    title.style.cssText = "font-size:11px;white-space:normal";
    const select = document.createElement("select");
    select.setAttribute("aria-label", "H3 temporal window GPU preset");
    select.style.cssText = "width:100%;min-width:0;height:34px;background:#191725;color:#eee7ff;border:1px solid #8d7ad2;font-size:11px";
    select.append(new Option("CUSTOM · current boxes are the truth", "custom"));
    H3_WINDOW_PRESETS.forEach(p => select.append(new Option(p.label,p.id)));
    const summary = document.createElement("div"); summary.style.cssText = "font-size:11px;line-height:1.45;overflow-wrap:anywhere";
    const note = document.createElement("div"); note.style.cssText = "font-size:10px;line-height:1.4;color:#bdb2d7";
    note.textContent = "Smaller windows reduce attention memory but add passes. Larger VAE batches reduce repeated decode work but use more RAM/VRAM. These presets do not alter native chunks, prompts, audio or steps. Values remain editable below; Save/Import Settings includes them. GPU labels are starting points, not guarantees.";
    const refresh = () => {
        root.style.display = read("upscale_mode") === "h3_pixel_refine" ? "grid" : "none";
        const w = Number(read("h3_upres_window_frames") ?? 136), o = Number(read("h3_upres_window_overlap") ?? 22), g = Number(read("h3_upres_pixel_groups") ?? 1);
        const match = H3_WINDOW_PRESETS.find(p => p.window === w && p.overlap === o && p.groups === g);
        select.value = match?.id || "custom";
        root.style.borderColor = match ? "#d0b3ff" : "#8d7ad2";
        const estimate = windowEstimate(345,w,o);
        summary.textContent = `${match ? "● PRESET ACTIVE" : "● CUSTOM"} · effective window ${estimate.window}f (${(estimate.window/24).toFixed(2)}s), overlap ${estimate.overlap}f · VAE ${g} × 17f. Example: 345f / 14.38s → ${estimate.count} passes × ${read("h3_upres_steps") ?? 2} steps. Exact count is logged per native chunk.`;
    };
    select.onchange = () => {
        const preset = H3_WINDOW_PRESETS.find(p => p.id === select.value);
        if (preset) apply(preset.label, windowPresetValues(preset));
        refresh();
    };
    root.append(title,select,summary,note);
    return {root,refresh};
}

// Native Motion Context presets are independent from the R38B delivery
// windows above. GPU names describe broad VRAM classes, never a benchmark or
// a guarantee. Applying a preset writes named Settings boxes; users may then
// edit every value and those boxes remain the queue-time truth.
export const H3_MOTION_CONTEXT_PRESETS = [
    {
        id:"xx60", label:"RTX xx60 · 8–12 GB · 124f / 5.17s",
        window:124, profile:"rtx_xx60_8_12gb_124", chunkRows:2048,
        clipproj:"4b_v3.1", loadMode:"dynamic", attention:"Standard",
    },
    {
        id:"xx70", label:"RTX xx70 · 12–16 GB · 209f / 8.71s",
        window:209, profile:"rtx_xx70_12_16gb_209", chunkRows:2048,
        clipproj:"4b_v3.1", loadMode:"dynamic", attention:"Standard",
    },
    {
        id:"xx80", label:"RTX xx80 · 16–24 GB · 294f / 12.25s",
        window:294, profile:"rtx_xx80_16_24gb_294", chunkRows:4096,
        clipproj:"4b_v3.1", loadMode:"dynamic", attention:"Standard",
    },
    {
        id:"xx90", label:"RTX xx90 · 24 GB+ · 362f / 15.08s",
        window:362, profile:"rtx_xx90_24gb_362", chunkRows:8192,
        clipproj:"8b_v3.1", loadMode:"dynamic", attention:"Standard",
    },
];

export const motionContextPresetValues = (preset) => ({
    motion_context_window_frames:preset.window,
    h3_exact_profile:preset.profile,
    h3_exact_chunk_rows:preset.chunkRows,
    h3_clipproj_profile:preset.clipproj,
    h3_clipproj_load_mode:preset.loadMode,
    h3_exact_attention_memory:preset.attention,
});

export function motionContextEstimate(windowFrames, tailFrames = 22) {
    const window = Math.max(56, Math.min(362, Math.round(Number(windowFrames) || 124)));
    const tail = Math.max(0, Math.min(window - 1, Math.round(Number(tailFrames) || 0)));
    const visible = Math.max(1, window - tail);
    return {window,tail,visible,windowSeconds:window/24,tailSeconds:tail/24,visibleSeconds:visible/24};
}

export function createH3MotionContextPanel(read, apply) {
    const root = document.createElement("section");
    root.style.cssText = "grid-column:1/-1;display:grid;gap:7px;min-width:0;padding:10px;border:1px solid #8862bd;background:#241a35;color:#f0e7ff";
    const title = document.createElement("strong");
    title.textContent = "NATIVE MOTION CONTEXT WINDOWS · EDITABLE";
    title.style.cssText = "font-size:11px;letter-spacing:.055em;white-space:normal";
    const select = document.createElement("select");
    select.setAttribute("aria-label", "Native H3 Motion Context hardware preset");
    select.style.cssText = "width:100%;min-width:0;height:34px;background:#191324;color:#f3ebff;border:1px solid #9d7bcc;font-size:11px";
    select.append(new Option("CUSTOM · current boxes are the truth", "custom"));
    H3_MOTION_CONTEXT_PRESETS.forEach((preset) => select.append(new Option(preset.label,preset.id)));
    const summary = document.createElement("div");
    summary.style.cssText = "font-size:11px;line-height:1.45;overflow-wrap:anywhere";
    const clipNote = document.createElement("div");
    clipNote.style.cssText = "font-size:10px;line-height:1.45;color:#c8badf";
    clipNote.textContent = "ClipProj: 4B v3.1 is the default lower-memory conditioner; 8B is larger and heavier; OFF uses the workflow fallback CLIP. Presets are starting points, not guarantees. Applying one never changes prompts, image slots, audio, mode or duration.";
    const refresh = () => {
        const windowFrames = Number(read("motion_context_window_frames") ?? 124);
        const tailFrames = Number(read("flf_continuity_tail_frames") ?? 22);
        const profile = String(read("h3_exact_profile") || "");
        const match = H3_MOTION_CONTEXT_PRESETS.find((preset) => preset.window === windowFrames && preset.profile === profile)
            || H3_MOTION_CONTEXT_PRESETS.find((preset) => preset.window === windowFrames);
        select.value = match?.id || "custom";
        const estimate = motionContextEstimate(windowFrames,tailFrames);
        const clipproj = String(read("h3_clipproj_profile") || "off").toUpperCase();
        summary.textContent = `${match ? "● PRESET ACTIVE" : "● CUSTOM"} · native window ${estimate.window}f / ${estimate.windowSeconds.toFixed(2)}s · carried AV tail ${estimate.tail}f / ${estimate.tailSeconds.toFixed(2)}s · new visible span ${estimate.visible}f / ${estimate.visibleSeconds.toFixed(2)}s · ClipProj ${clipproj}.`;
        root.style.borderColor = match ? "#c8a8f2" : "#8862bd";
    };
    select.onchange = () => {
        const preset = H3_MOTION_CONTEXT_PRESETS.find((item) => item.id === select.value);
        if (preset) apply(preset.label,motionContextPresetValues(preset));
        refresh();
    };
    root.append(title,select,summary,clipNote);
    return {root,refresh};
}
