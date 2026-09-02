// SPDX-License-Identifier: GPL-3.0-or-later

const VERTEX_SHADER = `
attribute vec2 a_position;
varying vec2 v_uv;
void main() {
  v_uv = a_position * 0.5 + 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}`;

const FRAGMENT_SHADER = `
precision highp float;
varying vec2 v_uv;
uniform sampler2D u_video;
uniform vec2 u_resolution;
uniform float u_frame;
uniform float u_seed;
uniform float u_strength;
uniform float u_grain_size;
uniform float u_persistence;
uniform float u_chroma;
uniform float u_shadow;
uniform float u_midtone;
uniform float u_highlight;
uniform int u_blend;
uniform int u_view;

float hash21(vec2 p, float frame, float seed) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031 + vec3(frame * 0.013, seed * 0.017, frame * 0.019));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z) * 2.0 - 1.0;
}
vec3 srgb_to_linear(vec3 c) {
  vec3 low = c / 12.92;
  vec3 high = pow((c + 0.055) / 1.055, vec3(2.4));
  return mix(high, low, step(c, vec3(0.04045)));
}
vec3 linear_to_srgb(vec3 c) {
  c = max(c, vec3(0.0));
  vec3 low = c * 12.92;
  vec3 high = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
  return mix(high, low, step(c, vec3(0.0031308)));
}
float soft_light(float base, float blend) {
  if (blend <= 0.5) return base - (1.0 - 2.0 * blend) * base * (1.0 - base);
  float d = base <= 0.25 ? ((16.0 * base - 12.0) * base + 4.0) * base : sqrt(base);
  return base + (2.0 * blend - 1.0) * (d - base);
}
void main() {
  vec4 source = texture2D(u_video, v_uv);
  if (u_view == 2 || (u_view == 1 && v_uv.x < 0.5)) {
    gl_FragColor = source;
    return;
  }
  float resolved_size = max(0.35, u_grain_size * max(u_resolution.x, u_resolution.y) / 4096.0);
  vec2 cell = floor(gl_FragCoord.xy / resolved_size);
  float current = hash21(cell, u_frame, u_seed);
  float previous = hash21(cell, max(0.0, u_frame - 1.0), u_seed);
  float fresh = sqrt(max(0.0, 1.0 - u_persistence * u_persistence));
  // hash21 is uniform [-1,1]. Match the unit-variance Gaussian field used by
  // the Python render path so HD preview is an exposure-accurate judgement.
  float common = (current * fresh + previous * u_persistence) * 1.73205080757;
  vec3 independent = vec3(
    hash21(cell + vec2(139.0, 17.0), u_frame, u_seed + 17.0),
    hash21(cell + vec2(278.0, 31.0), u_frame, u_seed + 29.0),
    hash21(cell + vec2(417.0, 47.0), u_frame, u_seed + 43.0)
  ) * 1.73205080757;
  independent.b *= 1.08;
  vec3 noise = mix(vec3(common), independent, u_chroma);
  vec3 linear = srgb_to_linear(source.rgb);
  float luma = dot(linear, vec3(0.2126, 0.7152, 0.0722));
  float shadow_w = clamp((0.5 - luma) / 0.5, 0.0, 1.0);
  float highlight_w = clamp((luma - 0.5) / 0.5, 0.0, 1.0);
  float mid_w = clamp(1.0 - shadow_w - highlight_w, 0.0, 1.0);
  float tone = clamp(shadow_w * u_shadow + mid_w * u_midtone + highlight_w * u_highlight, 0.0, 2.0);
  float sigma = u_strength * 0.19 * tone;
  vec3 result;
  if (u_blend == 1) {
    result = linear_to_srgb(linear + noise * sigma * 0.32);
  } else if (u_blend == 2) {
    vec3 b = clamp(vec3(0.5) + noise * sigma * 1.9, 0.0, 1.0);
    result = vec3(soft_light(source.r, b.r), soft_light(source.g, b.g), soft_light(source.b, b.b));
  } else {
    result = linear_to_srgb(linear * exp(noise * sigma - 0.5 * sigma * sigma));
  }
  gl_FragColor = vec4(clamp(result, 0.0, 1.0), source.a);
}`;

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader) || "Unknown WebGL shader error";
    gl.deleteShader(shader);
    throw new Error(message);
  }
  return shader;
}

function createRenderer(canvas) {
  const gl = canvas.getContext("webgl", { alpha: false, antialias: false, preserveDrawingBuffer: false });
  if (!gl) throw new Error("WebGL is unavailable in this browser");
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) || "WebGL link failed");
  gl.useProgram(program);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
  const position = gl.getAttribLocation(program, "a_position");
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  const locations = {};
  ["video", "resolution", "frame", "seed", "strength", "grain_size", "persistence", "chroma", "shadow", "midtone", "highlight", "blend", "view"].forEach((name) => {
    locations[name] = gl.getUniformLocation(program, `u_${name}`);
  });
  gl.uniform1i(locations.video, 0);
  return {
    draw(video, values, view, frame) {
      if (!video || video.readyState < 2 || !video.videoWidth || !video.videoHeight) return;
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
      gl.uniform2f(locations.resolution, canvas.width, canvas.height);
      gl.uniform1f(locations.frame, frame);
      gl.uniform1f(locations.seed, Number(values.seed || 1));
      gl.uniform1f(locations.strength, Number(values.strength || 0));
      gl.uniform1f(locations.grain_size, Number(values.grain_size_4k_px || 0.58));
      gl.uniform1f(locations.persistence, Number(values.temporal_persistence || 0));
      gl.uniform1f(locations.chroma, Number(values.chroma_amount || 0));
      gl.uniform1f(locations.shadow, Number(values.shadow_response || 0));
      gl.uniform1f(locations.midtone, Number(values.midtone_response || 0));
      gl.uniform1f(locations.highlight, Number(values.highlight_response || 0));
      gl.uniform1i(locations.blend, values.blend_method === "linear_additive" ? 1 : values.blend_method === "soft_light_luma" ? 2 : 0);
      gl.uniform1i(locations.view, view === "split" ? 1 : view === "original" ? 2 : 0);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    },
    destroy() {
      gl.deleteTexture(texture);
      gl.deleteBuffer(buffer);
      gl.deleteProgram(program);
    },
  };
}

export function validateTemporalGrainHDShader() {
  const canvas = document.createElement("canvas");
  canvas.width = 16;
  canvas.height = 16;
  const renderer = createRenderer(canvas);
  renderer.destroy();
  return true;
}

export function createTemporalGrainHDEditor({ video, readValues, writeValue, applyPreset, presets, loopSlider, viewSelect, setStatus }) {
  const overlay = document.createElement("div");
  overlay.className = "iamccs-grain-hd-overlay";
  overlay.innerHTML = `<style>
    .iamccs-grain-hd-overlay{position:fixed;inset:0;z-index:2147483000;display:none;grid-template-rows:auto minmax(0,1fr) auto;background:#050708f2;color:#edf1f4;font:11px Inter,Segoe UI,sans-serif}.iamccs-grain-hd-overlay.open{display:grid}.iamccs-grain-hd-head,.iamccs-grain-hd-foot{display:flex;align-items:center;gap:10px;min-width:0;padding:10px 14px;border-bottom:1px solid #34404a;background:#10161c}.iamccs-grain-hd-foot{border-top:1px solid #34404a;border-bottom:0}.iamccs-grain-hd-title{color:#f1d492;font:700 15px Georgia,serif}.iamccs-grain-hd-meta{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#93a3af}.iamccs-grain-hd-spacer{flex:1}.iamccs-grain-hd-btn,.iamccs-grain-hd-select{height:30px;border:1px solid #52606d;border-radius:4px;background:#1c252d;color:#edf1f4;padding:0 10px;font-size:10px;font-weight:850}.iamccs-grain-hd-btn.gold{border-color:#d5aa53;background:#7c5d25;color:#fff2cb}.iamccs-grain-hd-body{display:grid;grid-template-columns:minmax(0,1fr) 310px;min-height:0}.iamccs-grain-hd-stage{position:relative;display:block;min-height:0;overflow:auto;padding:12px;background:repeating-conic-gradient(#0a0d0f 0 25%,#0d1114 0 50%) 50%/22px 22px;text-align:center}.iamccs-grain-hd-canvas{display:block;margin:auto;max-width:none;max-height:none;box-shadow:0 0 0 1px #3c4650,0 12px 50px #000;image-rendering:auto}.iamccs-grain-hd-controls{min-height:0;overflow:auto;padding:12px;border-left:1px solid #34404a;background:linear-gradient(180deg,#121a21,#0b1015)}.iamccs-grain-hd-controls h3{margin:0 0 4px;color:#f1d492;font:700 14px Georgia,serif}.iamccs-grain-hd-controls p{margin:0 0 10px;color:#8f9faa;line-height:1.4}.iamccs-grain-hd-field{display:grid;gap:4px;margin-bottom:9px;color:#aeb9c2;font-size:9px;font-weight:850;letter-spacing:.035em}.iamccs-grain-hd-field select{width:100%;height:30px;border:1px solid #4d5a66;border-radius:4px;background:#182129;color:#edf1f4;padding:0 8px}.iamccs-grain-hd-slider{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:3px;margin-bottom:9px;color:#aeb9c2;font-size:9px;font-weight:800}.iamccs-grain-hd-slider output{color:#f1d492;font:800 10px Consolas,monospace}.iamccs-grain-hd-slider input{grid-column:1/-1;width:100%;accent-color:#d5aa53}.iamccs-grain-hd-live{margin:10px 0;padding:8px;border-left:3px solid #77d49c;background:#111c18;color:#aeeac5;line-height:1.35}.iamccs-grain-hd-loop{display:grid;grid-template-columns:auto minmax(140px,380px) auto;align-items:center;gap:8px;min-width:240px}.iamccs-grain-hd-loop input{width:100%;accent-color:#d5aa53}.iamccs-grain-hd-note{color:#8c9aa6;margin-left:auto;text-align:right}@media(max-width:900px){.iamccs-grain-hd-body{grid-template-columns:1fr}.iamccs-grain-hd-controls{position:absolute;right:12px;top:58px;bottom:58px;width:min(310px,42vw);z-index:2;border:1px solid #44515c;box-shadow:0 8px 30px #000b}}
  </style><div class="iamccs-grain-hd-head"><span class="iamccs-grain-hd-title">IAMCCS Grain HD Editor</span><span class="iamccs-grain-hd-meta"></span><span class="iamccs-grain-hd-spacer"></span><select class="iamccs-grain-hd-select iamccs-grain-hd-view"><option value="grain">GRAIN</option><option value="split">SPLIT · ORIGINAL / GRAIN</option><option value="original">ORIGINAL</option></select><select class="iamccs-grain-hd-select iamccs-grain-hd-zoom" title="Use 100% to inspect the actual grain pixels"><option value="fit">FIT</option><option value="1" selected>100% · PIXEL</option><option value="2">200%</option></select><button class="iamccs-grain-hd-btn iamccs-grain-hd-play">PAUSE</button><button class="iamccs-grain-hd-btn gold iamccs-grain-hd-close">CLOSE</button></div><div class="iamccs-grain-hd-body"><div class="iamccs-grain-hd-stage"><canvas class="iamccs-grain-hd-canvas"></canvas></div><aside class="iamccs-grain-hd-controls"><h3>Live Grain Controls</h3><p>Every change updates both this full-resolution preview and the node values used by Queue.</p><label class="iamccs-grain-hd-field">PRESET<select data-grain-control="preset"></select></label><label class="iamccs-grain-hd-field">BLEND<select data-grain-control="blend"><option value="density_exposure">DENSITY EXPOSURE</option><option value="linear_additive">LINEAR ADDITIVE</option><option value="soft_light_luma">SOFT LIGHT LUMA</option></select></label><div data-grain-control="sliders"></div><div class="iamccs-grain-hd-live" data-grain-control="status">LIVE · rendered from the current node values</div></aside></div><div class="iamccs-grain-hd-foot"><label class="iamccs-grain-hd-loop"><span>3s LOOP START</span><input type="range" min="0" max="0" step="0.04" value="0"><output>0.00s</output></label><span class="iamccs-grain-hd-note">100% shows real grain pixels · use SPLIT to verify treatment · Queue uses the same values.</span></div>`;
  document.body.appendChild(overlay);
  const canvas = overlay.querySelector("canvas");
  const meta = overlay.querySelector(".iamccs-grain-hd-meta");
  const view = overlay.querySelector(".iamccs-grain-hd-view");
  const zoom = overlay.querySelector(".iamccs-grain-hd-zoom");
  const stage = overlay.querySelector(".iamccs-grain-hd-stage");
  const play = overlay.querySelector(".iamccs-grain-hd-play");
  const close = overlay.querySelector(".iamccs-grain-hd-close");
  const localLoop = overlay.querySelector(".iamccs-grain-hd-loop input");
  const loopOutput = overlay.querySelector(".iamccs-grain-hd-loop output");
  const preset = overlay.querySelector('[data-grain-control="preset"]');
  const blend = overlay.querySelector('[data-grain-control="blend"]');
  const controls = overlay.querySelector('[data-grain-control="sliders"]');
  const controlStatus = overlay.querySelector('[data-grain-control="status"]');
  let renderer = null;
  let raf = 0;
  let frame = 0;
  let open = false;

  const sliderDefinitions = [
    ["strength", "STRENGTH", 0, 1, 0.01], ["grain_size_4k_px", "4K GRAIN SIZE · PX", 0.35, 4, 0.05],
    ["temporal_persistence", "TEMPORAL PERSISTENCE", 0, 0.85, 0.01], ["chroma_amount", "CHROMA", 0, 0.5, 0.01],
    ["shadow_response", "SHADOW RESPONSE", 0, 2, 0.02], ["midtone_response", "MIDTONE RESPONSE", 0, 2, 0.02],
    ["highlight_response", "HIGHLIGHT RESPONSE", 0, 2, 0.02],
  ];
  const sliderInputs = new Map();
  Object.keys(presets || {}).concat("custom_box_values").forEach((name) => preset.appendChild(new Option(name.replaceAll("_", " ").toUpperCase(), name)));
  sliderDefinitions.forEach(([name, label, min, max, step]) => {
    const row = document.createElement("label"); row.className = "iamccs-grain-hd-slider"; row.append(document.createTextNode(label));
    const output = document.createElement("output"); const input = document.createElement("input");
    input.type = "range"; input.min = String(min); input.max = String(max); input.step = String(step);
    input.oninput = () => { const value = Number(input.value); output.value = value.toFixed(2); writeValue?.(name, value); preset.value = "custom_box_values"; controlStatus.textContent = `${label} ${value.toFixed(2)} · LIVE`; };
    row.append(output, input); controls.appendChild(row); sliderInputs.set(name, { input, output });
  });
  const syncControls = () => {
    const values = readValues();
    preset.value = Object.prototype.hasOwnProperty.call(presets || {}, values.preset) ? values.preset : "custom_box_values";
    blend.value = values.blend_method || "density_exposure";
    sliderInputs.forEach(({ input, output }, name) => { input.value = String(Number(values[name] ?? 0)); output.value = Number(input.value).toFixed(2); });
  };
  preset.onchange = () => { applyPreset?.(preset.value); syncControls(); controlStatus.textContent = `${preset.value.replaceAll("_", " ").toUpperCase()} · applied live`; };
  blend.onchange = () => { writeValue?.("blend_method", blend.value); controlStatus.textContent = `${blend.value.replaceAll("_", " ").toUpperCase()} · LIVE`; };

  const syncLoop = () => {
    localLoop.min = loopSlider.min;
    localLoop.max = loopSlider.max;
    localLoop.step = loopSlider.step;
    localLoop.value = loopSlider.value;
    loopOutput.value = `${Number(localLoop.value || 0).toFixed(2)}s`;
  };
  const applyZoom = () => {
    if (!canvas.width || !canvas.height) return;
    if (zoom.value === "fit") {
      canvas.style.width = "auto";
      canvas.style.height = "auto";
      canvas.style.maxWidth = "100%";
      canvas.style.maxHeight = "100%";
      return;
    }
    const scale = Math.max(1, Number(zoom.value) || 1);
    canvas.style.maxWidth = "none";
    canvas.style.maxHeight = "none";
    canvas.style.width = `${canvas.width * scale}px`;
    canvas.style.height = `${canvas.height * scale}px`;
  };
  const draw = () => {
    if (!open) return;
    frame += 1;
    try {
      const values = readValues();
      renderer?.draw(video, values, view.value, frame);
      applyZoom();
      if (frame % 12 === 1) {
        const state = view.value === "original" || Number(values.strength || 0) <= 0 ? "BYPASS" : "LIVE GRAIN";
        const zoomLabel = zoom.value === "fit" ? "FIT" : `${Math.round((Number(zoom.value) || 1) * 100)}%`;
        meta.textContent = `${video.videoWidth}×${video.videoHeight} · ${state} · strength ${Number(values.strength || 0).toFixed(2)} · ${values.blend_method} · ${zoomLabel}`;
      }
    } catch (error) { meta.textContent = `HD preview error: ${error.message}`; }
    raf = requestAnimationFrame(draw);
  };
  const hide = async () => {
    open = false;
    cancelAnimationFrame(raf);
    overlay.classList.remove("open");
    if (document.fullscreenElement === overlay) {
      try { await document.exitFullscreen(); } catch {}
    }
  };
  const show = async () => {
    if (!video?.src || video.readyState < 1 || !video.videoWidth || !video.videoHeight) {
      setStatus("OPEN HD EDITOR requires a local preview video. Use OPEN VIDEO first.");
      return;
    }
    try { renderer ||= createRenderer(canvas); } catch (error) { setStatus(`HD Editor unavailable: ${error.message}`); return; }
    syncLoop();
    syncControls();
    view.value = viewSelect.value === "original" ? "grain" : viewSelect.value;
    meta.textContent = `${video.videoWidth}×${video.videoHeight} · live WebGL grain · private browser session`;
    overlay.classList.add("open");
    open = true;
    play.textContent = video.paused ? "PLAY" : "PAUSE";
    try { await overlay.requestFullscreen(); } catch {}
    raf = requestAnimationFrame(draw);
    requestAnimationFrame(() => {
      stage.scrollLeft = Math.max(0, (stage.scrollWidth - stage.clientWidth) / 2);
      stage.scrollTop = Math.max(0, (stage.scrollHeight - stage.clientHeight) / 2);
    });
  };
  view.onchange = () => { viewSelect.value = view.value; };
  zoom.onchange = () => { applyZoom(); };
  play.onclick = async () => { if (video.paused) { try { await video.play(); } catch {} } else video.pause(); play.textContent = video.paused ? "PLAY" : "PAUSE"; };
  close.onclick = hide;
  localLoop.oninput = () => { loopSlider.value = localLoop.value; loopSlider.oninput?.(); loopOutput.value = `${Number(localLoop.value || 0).toFixed(2)}s`; };
  overlay.addEventListener("dblclick", (event) => { if (event.target === stage) hide(); });
  document.addEventListener("fullscreenchange", () => { if (open && !document.fullscreenElement && overlay.classList.contains("open")) { /* keep windowed overlay open */ } });
  return {
    show,
    hide,
    destroy() { hide(); renderer?.destroy(); overlay.remove(); },
  };
}
