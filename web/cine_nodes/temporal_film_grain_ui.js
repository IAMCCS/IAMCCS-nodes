// SPDX-License-Identifier: GPL-3.0-or-later

import { app } from "/scripts/app.js";
import { createTemporalGrainHDEditor } from "./temporal_film_grain_hd_editor.js";

const NODE_TYPE = "IAMCCS_CineTemporalFilmGrain4K";
const FLOAT_WIDGETS = ["strength", "grain_size_4k_px", "temporal_persistence", "chroma_amount", "shadow_response", "midtone_response", "highlight_response"];
const PRESETS = {
    "65mm_4k_scan_subtle": { strength: 0.08, grain_size_4k_px: 0.58, temporal_persistence: 0.06, chroma_amount: 0.025, shadow_response: 0.48, midtone_response: 0.82, highlight_response: 0.24 },
    "35mm_vision3_fine": { strength: 0.12, grain_size_4k_px: 0.82, temporal_persistence: 0.09, chroma_amount: 0.05, shadow_response: 0.58, midtone_response: 1.0, highlight_response: 0.32 },
    "35mm_500t_texture": { strength: 0.18, grain_size_4k_px: 1.08, temporal_persistence: 0.12, chroma_amount: 0.09, shadow_response: 0.86, midtone_response: 1.18, highlight_response: 0.38 },
    "16mm_fine_documentary": { strength: 0.24, grain_size_4k_px: 1.62, temporal_persistence: 0.16, chroma_amount: 0.12, shadow_response: 1.02, midtone_response: 1.3, highlight_response: 0.46 },
};
const BLEND_HELP = {
    density_exposure: "Recommended · mean-preserving exposure/density modulation. It adds texture without behaving like a grey overlay.",
    linear_additive: "Adds grain energy in linear light. More neutral, but shadows and black level need closer monitoring.",
    soft_light_luma: "A softer display-referred contrast texture. Use low strength; it is less physically motivated.",
};
// The backend uses unit-variance Gaussian fields. randomAt() is uniform in
// [-1, 1] (variance 1/3), so sqrt(3) gives mini-preview and render output the
// same noise energy instead of making Queue grain look materially stronger.
const BACKEND_NOISE_STD = Math.sqrt(3);

function widget(node, name) { return (node.widgets || []).find((item) => item.name === name); }
function read(node, name, fallback) { const raw = widget(node, name)?.value; return raw == null ? fallback : raw; }
function write(node, name, value) {
    const target = widget(node, name); if (!target) return;
    target.value = value; target.callback?.(value); node.setDirtyCanvas?.(true, true);
}
function hideWidget(target) {
    if (!target || target._iamccsGrainHidden) return;
    target._iamccsGrainType = target.type; target._iamccsGrainCompute = target.computeSize; target._iamccsGrainDraw = target.draw;
    target.serializeValue ||= (() => target.value);
    target.type = "hidden"; target.hidden = true; target.computeSize = () => [0, 0]; target.draw = () => {}; target._iamccsGrainHidden = true;
}
function randomAt(x, y, frame, seed) {
    let n = (x * 374761393 + y * 668265263 + frame * 1442695041 + seed * 69069) | 0;
    n = Math.imul(n ^ (n >>> 13), 1274126177); n ^= n >>> 16;
    return ((n >>> 0) / 4294967295) * 2 - 1;
}
function softLight(base, blend) {
    if (blend <= 0.5) return base - (1 - 2 * blend) * base * (1 - base);
    const d = base <= 0.25 ? ((16 * base - 12) * base + 4) * base : Math.sqrt(base);
    return base + (2 * blend - 1) * (d - base);
}
function srgbToLinear(value) {
    const c = Math.max(0, Math.min(1, Number(value) || 0));
    return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}
function linearToSrgb(value) {
    const c = Math.max(0, Number(value) || 0);
    return c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
}
function formatTime(seconds) {
    const safe = Math.max(0, Number(seconds) || 0); const minutes = Math.floor(safe / 60); const rest = safe - minutes * 60;
    return `${String(minutes).padStart(2, "0")}:${rest.toFixed(2).padStart(5, "0")}`;
}

function mount(node) {
    if (node._iamccsGrainUi) return;
    node._iamccsGrainUi = true;
    ["preset", "blend_method", ...FLOAT_WIDGETS, "seed", "frame_start"].forEach((name) => hideWidget(widget(node, name)));

    const root = document.createElement("div"); root.className = "iamccs-grain-root";
    root.innerHTML = `<style>
      .iamccs-grain-root{width:100%;max-width:100%;height:auto!important;min-height:0!important;padding:8px 8px 12px;box-sizing:border-box;border:1px solid #39434d;border-radius:10px;background:linear-gradient(145deg,#0f1419,#182029);color:#e9edf1;font:10px Inter,Segoe UI,sans-serif;overflow:visible}.iamccs-grain-root *{box-sizing:border-box}.iamccs-grain-head{display:flex;align-items:center;gap:7px;min-width:0;margin-bottom:6px}.iamccs-grain-chip{flex:0 0 auto;padding:3px 6px;border:1px solid #8c7040;border-radius:999px;background:#312818;color:#f0cf8b;font-size:8px;font-weight:900;letter-spacing:.08em}.iamccs-grain-title{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font:700 12px Georgia,serif}.iamccs-grain-live{margin-left:auto;flex:0 0 auto;color:#7ed5a1;font-size:8px;font-weight:900}.iamccs-grain-selects{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:6px}.iamccs-grain-field{display:grid;min-width:0;gap:3px;color:#8f9eab;font-size:8px;font-weight:900;letter-spacing:.05em}.iamccs-grain-field select,.iamccs-grain-field input[type=number]{min-width:0;width:100%;height:27px;border:1px solid #46525e;border-radius:5px;background:#111921;color:#e7edf2;padding:0 7px;font-size:9px}.iamccs-grain-canvas-wrap{position:relative;width:100%;aspect-ratio:16/6.8;border:1px solid #4a545e;border-radius:7px;background:#07090b;overflow:hidden}.iamccs-grain-canvas{display:block;width:100%;height:100%;object-fit:contain}.iamccs-grain-drop{position:absolute;inset:0;display:grid;place-items:center;padding:12px;text-align:center;color:#8f9ba7;background:#0a0e12b8;border:1px dashed #59646f;pointer-events:none;line-height:1.35}.iamccs-grain-drop.hidden{display:none}.iamccs-grain-drop.active{display:grid!important;border:2px solid #d5ad61;background:#14110bd9;color:#ffe2a7}.iamccs-grain-video-tools{display:grid;grid-template-columns:auto auto auto minmax(0,1fr);gap:5px;margin-top:5px}.iamccs-grain-btn{min-width:0;height:27px;border:1px solid #46515d;border-radius:5px;background:#222a33;color:#e6ebef;padding:0 7px;font-size:8px;font-weight:800;cursor:pointer;white-space:nowrap}.iamccs-grain-btn:hover{border-color:#d1a754}.iamccs-grain-btn.gold{background:#826126;border-color:#d7ac55;color:#fff2cd}.iamccs-grain-file{position:absolute;width:1px;height:1px;opacity:0;pointer-events:none}.iamccs-grain-loop{display:grid;grid-template-columns:auto minmax(0,1fr) auto;align-items:center;gap:5px;margin-top:5px;color:#96a3af;font-size:8px}.iamccs-grain-loop input{width:100%;accent-color:#d2a653}.iamccs-grain-loop.disabled{opacity:.45;pointer-events:none}.iamccs-grain-sliders{display:grid;grid-template-columns:1fr 1fr;gap:5px 10px;margin-top:6px;padding:6px;border:1px solid #303b45;border-radius:7px;background:#111820}.iamccs-grain-slider{display:grid;grid-template-columns:1fr auto;gap:2px;min-width:0;color:#9eabb6;font-size:8px;font-weight:800}.iamccs-grain-slider output{color:#f0cf8b;font:800 8px 'Courier New',monospace}.iamccs-grain-slider input{grid-column:1/-1;width:100%;height:14px;accent-color:#d0a457}.iamccs-grain-advanced,.iamccs-grain-notes{margin-top:5px;border:1px solid #303b45;border-radius:7px;background:#111820;overflow:hidden}.iamccs-grain-advanced summary,.iamccs-grain-notes summary{padding:6px 8px;color:#b5c0ca;font-size:8px;font-weight:900;cursor:pointer}.iamccs-grain-advanced .iamccs-grain-sliders{margin:0;border:0;border-top:1px solid #303b45;border-radius:0}.iamccs-grain-method{margin-top:5px;padding:5px 7px;max-height:38px;overflow:auto;border-left:2px solid #d0a457;background:#151b21;color:#b3bec8;white-space:normal;overflow-wrap:anywhere;line-height:1.3}.iamccs-grain-status{margin-top:5px;padding:5px 7px;max-height:38px;overflow:auto;border-left:2px solid #6b8eaa;background:#121a22;color:#a9b6c1;white-space:normal;overflow-wrap:anywhere;line-height:1.3}.iamccs-grain-tip{display:block;width:auto;margin:5px 7px;padding:0;color:#7f8d99;white-space:normal;overflow-wrap:anywhere;line-height:1.35}.iamccs-grain-tip strong{color:#aebac4}.iamccs-grain-root[data-source=video] .iamccs-grain-drop{display:none}
    </style>
    <div class="iamccs-grain-head"><span class="iamccs-grain-chip">CINE FINISH</span><span class="iamccs-grain-title">Temporal Film Grain · 4K Scan</span><span class="iamccs-grain-live">● LIVE</span></div>
    <div class="iamccs-grain-selects"><label class="iamccs-grain-field">PRESET<select class="iamccs-grain-preset"></select></label><label class="iamccs-grain-field">BLEND<select class="iamccs-grain-blend"></select></label></div>
    <div class="iamccs-grain-canvas-wrap"><canvas class="iamccs-grain-canvas" width="442" height="188"></canvas><div class="iamccs-grain-drop">TONAL REFERENCE<br>Open or drop a local video to preview a real 3-second loop.</div></div>
    <div class="iamccs-grain-video-tools"><button class="iamccs-grain-btn gold iamccs-grain-open">OPEN VIDEO</button><button class="iamccs-grain-btn iamccs-grain-hd">OPEN HD</button><button class="iamccs-grain-btn iamccs-grain-clear">CLEAR</button><select class="iamccs-grain-btn iamccs-grain-view"><option value="grain">GRAIN</option><option value="split">SPLIT · ORIGINAL / GRAIN</option><option value="original">ORIGINAL</option></select><input class="iamccs-grain-file" type="file" accept="video/*,.mp4,.mov,.mkv,.webm,.avi"></div>
    <div class="iamccs-grain-loop disabled"><span>3s LOOP START</span><input type="range" min="0" max="0" value="0" step="0.04"><output>00:00.00</output></div>
    <div class="iamccs-grain-sliders iamccs-grain-primary"></div>
    <details class="iamccs-grain-advanced"><summary>ADVANCED TONAL RESPONSE · SEED / CHUNK OFFSET</summary><div class="iamccs-grain-sliders iamccs-grain-tonal"></div><div style="display:grid;grid-template-columns:1fr 1fr;gap:7px;padding:8px"><label class="iamccs-grain-field">SEED<input class="iamccs-grain-seed" type="number" min="0" step="1"></label><label class="iamccs-grain-field">FRAME START<input class="iamccs-grain-frame" type="number" min="0" step="1"></label></div></details>
    <div class="iamccs-grain-method"></div><div class="iamccs-grain-status"></div>
    <details class="iamccs-grain-notes"><summary>INFO · PREVIEW &amp; PIPELINE</summary><div class="iamccs-grain-tip"><strong>Preview only:</strong> the local file stays in this browser session and is never saved in the workflow or sent through Queue. Backend IMAGE frames remain render truth.</div><div class="iamccs-grain-tip"><strong>Pipeline:</strong> Video Editor master → final upscale/detail → this grain node → Exporter.</div></details>`;

    const canvas = root.querySelector("canvas"); const context = canvas.getContext("2d", { alpha: false, willReadFrequently: true });
    const presetSelect = root.querySelector(".iamccs-grain-preset"); const blendSelect = root.querySelector(".iamccs-grain-blend"); const blendHelp = root.querySelector(".iamccs-grain-method");
    const fileInput = root.querySelector(".iamccs-grain-file"); const openButton = root.querySelector(".iamccs-grain-open"); const hdButton = root.querySelector(".iamccs-grain-hd"); const clearButton = root.querySelector(".iamccs-grain-clear"); const viewSelect = root.querySelector(".iamccs-grain-view");
    const loopControl = root.querySelector(".iamccs-grain-loop"); const loopSlider = loopControl.querySelector("input"); const loopOutput = loopControl.querySelector("output"); const drop = root.querySelector(".iamccs-grain-drop"); const status = root.querySelector(".iamccs-grain-status");
    const seedInput = root.querySelector(".iamccs-grain-seed"); const frameInput = root.querySelector(".iamccs-grain-frame");
    const video = document.createElement("video"); video.muted = true; video.playsInline = true; video.preload = "metadata"; video.loop = false;
    let objectUrl = ""; let previewFrame = 0; let raf = 0; let lastDraw = 0; let loopStart = 0; let loopLength = 3;

    Object.keys(PRESETS).concat("custom_box_values").forEach((name) => presetSelect.appendChild(new Option(name.replaceAll("_", " ").toUpperCase(), name)));
    ["density_exposure", "linear_additive", "soft_light_luma"].forEach((name) => blendSelect.appendChild(new Option(name.replaceAll("_", " ").toUpperCase(), name)));
    presetSelect.value = String(read(node, "preset", "65mm_4k_scan_subtle")); blendSelect.value = String(read(node, "blend_method", "density_exposure"));
    seedInput.value = String(read(node, "seed", 1)); frameInput.value = String(read(node, "frame_start", 0));

    const sliderDefinitions = [
        ["strength", "STRENGTH", 0, 1, 0.01, 2], ["grain_size_4k_px", "4K GRAIN SIZE · PX", 0.35, 4, 0.05, 2],
        ["temporal_persistence", "TEMPORAL PERSISTENCE", 0, 0.85, 0.01, 2], ["chroma_amount", "CHROMA", 0, 0.5, 0.01, 2],
        ["shadow_response", "SHADOW RESPONSE", 0, 2, 0.02, 2], ["midtone_response", "MIDTONE RESPONSE", 0, 2, 0.02, 2], ["highlight_response", "HIGHLIGHT RESPONSE", 0, 2, 0.02, 2],
    ];
    const sliderControls = new Map();
    sliderDefinitions.forEach(([name, label, min, max, step, digits], index) => {
        const wrap = document.createElement("label"); wrap.className = "iamccs-grain-slider"; wrap.append(document.createTextNode(label));
        const output = document.createElement("output"); const input = document.createElement("input"); input.type = "range"; input.min = String(min); input.max = String(max); input.step = String(step); input.value = String(read(node, name, min)); output.value = Number(input.value).toFixed(digits);
        input.oninput = () => { output.value = Number(input.value).toFixed(digits); write(node, name, Number(input.value)); status.textContent = `${label} updated live. The visible box value is render truth.`; };
        wrap.append(output, input); (index < 4 ? root.querySelector(".iamccs-grain-primary") : root.querySelector(".iamccs-grain-tonal")).appendChild(wrap); sliderControls.set(name, { input, output, digits });
    });

    function syncControls() {
        const storedPreset = String(read(node, "preset", presetSelect.value));
        if (Array.from(presetSelect.options).some((option) => option.value === storedPreset)) presetSelect.value = storedPreset;
        sliderControls.forEach(({ input, output, digits }, name) => { input.value = String(read(node, name, input.value)); output.value = Number(input.value).toFixed(digits); });
        seedInput.value = String(read(node, "seed", seedInput.value)); frameInput.value = String(read(node, "frame_start", frameInput.value));
        blendSelect.value = String(read(node, "blend_method", blendSelect.value)); blendHelp.textContent = `BLEND · ${blendSelect.value.replaceAll("_", " ").toUpperCase()} — ${BLEND_HELP[blendSelect.value]}`;
    }
    function applyPreset(name) {
        write(node, "preset", name); const preset = PRESETS[name];
        if (!preset) { status.textContent = "CUSTOM BOX VALUES · no value was overwritten."; return; }
        Object.entries(preset).forEach(([key, next]) => write(node, key, next)); syncControls(); status.textContent = `${name.replaceAll("_", " ").toUpperCase()} applied immediately. Every slider remains editable.`;
    }
    presetSelect.onchange = () => applyPreset(presetSelect.value);
    blendSelect.onchange = () => { write(node, "blend_method", blendSelect.value); syncControls(); status.textContent = `Blend changed live to ${blendSelect.value.replaceAll("_", " ").toUpperCase()}.`; };
    seedInput.onchange = () => write(node, "seed", Math.max(0, Math.trunc(Number(seedInput.value) || 0)));
    frameInput.onchange = () => write(node, "frame_start", Math.max(0, Math.trunc(Number(frameInput.value) || 0)));

    function releaseVideo() {
        video.pause(); video.removeAttribute("src"); video.load(); if (objectUrl) URL.revokeObjectURL(objectUrl); objectUrl = ""; root.dataset.source = "tonal"; drop.classList.remove("hidden", "active"); loopControl.classList.add("disabled"); loopSlider.max = "0"; loopSlider.value = "0"; loopOutput.value = "00:00.00"; status.textContent = "Tonal reference active. Open or drop a video for a private three-second loop.";
    }
    function loadVideo(file) {
        if (!file || (!String(file.type).startsWith("video/") && !/\.(mp4|mov|mkv|webm|avi)$/i.test(file.name))) { status.textContent = "Preview rejected: choose a video file."; return; }
        releaseVideo(); objectUrl = URL.createObjectURL(file); video.src = objectUrl; status.textContent = `Reading ${file.name}…`;
        video.onloadedmetadata = async () => {
            if (!Number.isFinite(video.duration) || video.duration <= 0) { status.textContent = "This browser cannot read the selected video's duration."; return; }
            loopLength = Math.min(3, video.duration); loopStart = 0; loopSlider.max = String(Math.max(0, video.duration - loopLength)); loopSlider.value = "0"; loopOutput.value = formatTime(0); loopControl.classList.remove("disabled"); root.dataset.source = "video"; drop.classList.add("hidden");
            status.textContent = `${file.name} · private ${loopLength.toFixed(2)}s loop · ${video.videoWidth}×${video.videoHeight}. Nothing was queued.`;
            video.currentTime = 0; try { await video.play(); } catch { status.textContent += " Press the canvas once if browser autoplay is blocked."; }
        };
        video.onerror = () => { status.textContent = "The browser could not decode this video. Try MP4/H.264 or WebM for preview."; };
    }
    openButton.onclick = () => fileInput.click(); clearButton.onclick = releaseVideo; fileInput.onchange = () => { loadVideo(fileInput.files?.[0]); fileInput.value = ""; };
    loopSlider.oninput = () => { loopStart = Number(loopSlider.value) || 0; loopOutput.value = formatTime(loopStart); if (objectUrl) video.currentTime = loopStart; };
    canvas.onclick = async () => { if (!objectUrl) return; if (video.paused) { try { await video.play(); } catch {} } else video.pause(); };
    ["dragenter", "dragover"].forEach((type) => root.addEventListener(type, (event) => { event.preventDefault(); event.stopPropagation(); drop.classList.add("active"); }));
    ["dragleave", "drop"].forEach((type) => root.addEventListener(type, (event) => { event.preventDefault(); event.stopPropagation(); drop.classList.remove("active"); if (type === "drop") loadVideo(Array.from(event.dataTransfer?.files || [])[0]); }));

    function drawSource() {
        const width = canvas.width, height = canvas.height; context.fillStyle = "#07090b"; context.fillRect(0, 0, width, height);
        if (objectUrl && video.readyState >= 2 && video.videoWidth > 0) {
            const end = Math.min(video.duration, loopStart + loopLength); if (video.currentTime < loopStart || video.currentTime >= end - 0.015) video.currentTime = loopStart;
            const scale = Math.min(width / video.videoWidth, height / video.videoHeight); const drawWidth = video.videoWidth * scale, drawHeight = video.videoHeight * scale;
            context.drawImage(video, (width - drawWidth) / 2, (height - drawHeight) / 2, drawWidth, drawHeight); return;
        }
        const gradient = context.createLinearGradient(0, 0, width, 0); gradient.addColorStop(0, "#07090c"); gradient.addColorStop(0.32, "#4b5360"); gradient.addColorStop(0.66, "#989a92"); gradient.addColorStop(1, "#ece2ca"); context.fillStyle = gradient; context.fillRect(0, 0, width, height);
        const radial = context.createRadialGradient(width * 0.5, height * 0.47, 10, width * 0.5, height * 0.47, width * 0.55); radial.addColorStop(0, "rgba(70,100,118,.26)"); radial.addColorStop(1, "rgba(0,0,0,.24)"); context.fillStyle = radial; context.fillRect(0, 0, width, height);
    }
    function applyPreviewGrain(original) {
        const processed = new ImageData(new Uint8ClampedArray(original.data), original.width, original.height); const data = processed.data;
        const strength = Number(read(node, "strength", 0.1)), chroma = Number(read(node, "chroma_amount", 0.05)), persistence = Number(read(node, "temporal_persistence", 0.08));
        const grainSize = Number(read(node, "grain_size_4k_px", 0.72)), shadow = Number(read(node, "shadow_response", 0.58)), midtone = Number(read(node, "midtone_response", 1)), highlight = Number(read(node, "highlight_response", 0.32));
        const blend = String(read(node, "blend_method", "density_exposure")), seed = Number(read(node, "seed", 1)), frameOffset = Number(read(node, "frame_start", 0));
        const cell = Math.max(0.35, grainSize * Math.max(original.width, original.height) / 4096), fresh = Math.sqrt(Math.max(0, 1 - persistence * persistence)), currentFrame = previewFrame + frameOffset, previousFrame = Math.max(frameOffset, currentFrame - 1);
        for (let y = 0; y < original.height; y++) for (let x = 0; x < original.width; x++) {
            const index = (y * original.width + x) * 4, nx = Math.floor(x / cell), ny = Math.floor(y / cell);
            const common = (randomAt(nx, ny, currentFrame, seed) * fresh + randomAt(nx, ny, previousFrame, seed) * persistence) * BACKEND_NOISE_STD;
            const linearRgb = [
                srgbToLinear(data[index] / 255),
                srgbToLinear(data[index + 1] / 255),
                srgbToLinear(data[index + 2] / 255),
            ];
            const baseLuma = linearRgb[0] * 0.2126 + linearRgb[1] * 0.7152 + linearRgb[2] * 0.0722;
            const shadowW = Math.max(0, Math.min(1, (0.5 - baseLuma) / 0.5)), highlightW = Math.max(0, Math.min(1, (baseLuma - 0.5) / 0.5)), midW = Math.max(0, 1 - shadowW - highlightW), tone = shadowW * shadow + midW * midtone + highlightW * highlight, sigma = strength * 0.19 * Math.max(0, Math.min(2, tone));
            for (let channel = 0; channel < 3; channel++) {
                const base = data[index + channel] / 255, noise = common * (1 - chroma) + randomAt(nx + channel * 139, ny, currentFrame, seed + 17) * BACKEND_NOISE_STD * chroma; let out;
                if (blend === "linear_additive") out = linearToSrgb(linearRgb[channel] + noise * sigma * 0.32);
                else if (blend === "soft_light_luma") out = softLight(base, Math.max(0, Math.min(1, 0.5 + noise * sigma * 1.9)));
                else out = linearToSrgb(linearRgb[channel] * Math.exp(noise * sigma - 0.5 * sigma * sigma));
                data[index + channel] = Math.round(Math.max(0, Math.min(1, out)) * 255);
            }
        }
        return processed;
    }
    function draw(now = 0) {
        if (!root.isConnected) return; if (document.hidden || now - lastDraw < 1000 / 24) { raf = requestAnimationFrame(draw); return; }
        lastDraw = now; previewFrame += 1; drawSource(); const original = context.getImageData(0, 0, canvas.width, canvas.height); const view = viewSelect.value;
        if (view !== "original") { const processed = applyPreviewGrain(original); context.putImageData(processed, 0, 0); if (view === "split") { context.putImageData(original, 0, 0, 0, 0, Math.floor(canvas.width / 2), canvas.height); context.fillStyle = "#f3cf82"; context.fillRect(Math.floor(canvas.width / 2) - 1, 0, 2, canvas.height); } }
        context.fillStyle = "rgba(5,7,9,.72)"; context.fillRect(7, 7, Math.min(300, canvas.width - 14), 20); context.fillStyle = "#f3dfb2"; context.font = "bold 9px Segoe UI"; context.fillText(`${objectUrl ? "LOCAL VIDEO · 3s LOOP" : "TONAL REFERENCE"} · ${view.toUpperCase()} · F${String(previewFrame % 72).padStart(2, "0")}`, 13, 21);
        raf = requestAnimationFrame(draw);
    }

    const readEditorValues = () => ({
        preset: String(read(node, "preset", "65mm_4k_scan_subtle")), blend_method: String(read(node, "blend_method", "density_exposure")), strength: Number(read(node, "strength", 0.08)), grain_size_4k_px: Number(read(node, "grain_size_4k_px", 0.58)), temporal_persistence: Number(read(node, "temporal_persistence", 0.06)), chroma_amount: Number(read(node, "chroma_amount", 0.025)), shadow_response: Number(read(node, "shadow_response", 0.48)), midtone_response: Number(read(node, "midtone_response", 0.82)), highlight_response: Number(read(node, "highlight_response", 0.24)), seed: Number(read(node, "seed", 1)),
    });
    const hdEditor = createTemporalGrainHDEditor({
        video,
        readValues: readEditorValues,
        writeValue: (name, value) => { write(node, name, value); syncControls(); },
        applyPreset,
        presets: PRESETS,
        loopSlider,
        viewSelect,
        setStatus: (message) => { status.textContent = message; },
    });
    hdButton.onclick = hdEditor.show;
    syncControls(); releaseVideo();
    const dom = node.addDOMWidget("Temporal Grain Lab", "iamccs_temporal_grain_lab", root, { serialize: false });
    const advanced = root.querySelector(".iamccs-grain-advanced"); const notes = root.querySelector(".iamccs-grain-notes");
    let measuredHeight = 500;
    const measuredPanelHeight = () => {
        const rootRect = root.getBoundingClientRect();
        const style = getComputedStyle(root);
        const bottomInset = (parseFloat(style.paddingBottom) || 0) + (parseFloat(style.borderBottomWidth) || 0);
        let childBottom = 0;
        Array.from(root.children).forEach((child) => {
            if (child.tagName === "STYLE" || getComputedStyle(child).display === "none") return;
            const childRect = child.getBoundingClientRect();
            const childStyle = getComputedStyle(child);
            const marginBottom = parseFloat(childStyle.marginBottom) || 0;
            childBottom = Math.max(childBottom, childRect.bottom - rootRect.top + marginBottom);
        });
        // Never use rootRect.height/scrollHeight here: the DOM-widget wrapper may
        // stretch them to the current node size, creating a self-growing padding
        // loop. Child geometry is the only content truth.
        if (childBottom > 0) measuredHeight = Math.max(420, Math.ceil(childBottom + bottomInset));
        return measuredHeight;
    };
    dom.computeSize = () => [Math.max(466, Number(node.size?.[0] || 480) - 14), measuredPanelHeight()]; node.resizable = true;
    let fittedHeight = 0; let lastWidgetTop = 72; let applyingFit = false; let fitQueued = false;
    const fit = () => {
        fitQueued = false;
        const reportedTop = Number(dom.last_y);
        if (Number.isFinite(reportedTop) && reportedTop >= 24 && reportedTop <= 240) lastWidgetTop = reportedTop;
        const requiredHeight = Math.ceil(lastWidgetTop + measuredPanelHeight() + 18);
        if (Math.abs(requiredHeight - fittedHeight) < 2 && Math.abs(Number(node.size?.[1] || 0) - requiredHeight) < 2) return;
        fittedHeight = requiredHeight;
        applyingFit = true;
        node.setSize?.([Math.max(480, Number(node.size?.[0] || 480)), requiredHeight]);
        applyingFit = false;
        node.setDirtyCanvas?.(true, true);
    };
    const scheduleFit = () => {
        if (fitQueued) return;
        fitQueued = true;
        requestAnimationFrame(() => requestAnimationFrame(fit));
    };
    scheduleFit(); [advanced, notes].forEach((details) => details.addEventListener("toggle", scheduleFit));
    const resizeObserver = new ResizeObserver(scheduleFit); resizeObserver.observe(root);
    const previousResize = node.onResize; node.onResize = function() { const result = previousResize?.apply?.(this, arguments); if (!applyingFit) scheduleFit(); return result; };
    const previousConfigure = node.onConfigure; node.onConfigure = function() { const result = previousConfigure?.apply?.(this, arguments); syncControls(); scheduleFit(); return result; };
    const previousRemoved = node.onRemoved; node.onRemoved = function() { resizeObserver.disconnect(); cancelAnimationFrame(raf); hdEditor.destroy(); releaseVideo(); return previousRemoved?.apply?.(this, arguments); };
    raf = requestAnimationFrame(draw);
}

app.registerExtension({
    name: "IAMCCS.CineTemporalFilmGrain4K.UI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_TYPE) return;
        const previous = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() { const result = previous?.apply?.(this, arguments); try { mount(this); } catch (error) { console.error("[IAMCCS Grain] UI mount failed", error); } return result; };
    },
});
