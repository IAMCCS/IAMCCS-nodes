import { app } from "../../../scripts/app.js";

const NODE = "IAMCCS_CineAudioInfo";

function widget(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

function setWidget(node, name, value) {
  const w = widget(node, name);
  if (!w) return;
  w.value = value;
  if (typeof w.callback === "function") w.callback(value);
  node.setDirtyCanvas?.(true, true);
}

function styles() {
  if (document.getElementById("iamccs-cine-audio-info-style")) return;
  const style = document.createElement("style");
  style.id = "iamccs-cine-audio-info-style";
  style.textContent = \`
    .iamccs-cai { box-sizing:border-box; width:100%; min-width:520px; background:#0d1517; color:#e8f3ef; border:1px solid #245052; border-radius:8px; font:12px/1.35 system-ui,Segoe UI,sans-serif; overflow:hidden; }
    .iamccs-cai * { box-sizing:border-box; }
    .iamccs-cai-head { display:flex; justify-content:space-between; gap:10px; align-items:center; padding:10px 12px; background:#142426; border-bottom:1px solid #245052; }
    .iamccs-cai-title { font-weight:800; font-size:14px; color:#fff; }
    .iamccs-cai-sub { color:#a9c4c0; margin-top:2px; font-size:11px; }
    .iamccs-cai-actions { display:flex; gap:6px; flex-wrap:wrap; justify-content:flex-end; }
    .iamccs-cai button { border:1px solid #34787a; background:#0e3034; color:#eafff8; border-radius:6px; padding:6px 9px; font-weight:700; cursor:pointer; }
    .iamccs-cai button.primary { background:#d8a94a; border-color:#f1c66b; color:#17130b; }
    .iamccs-cai-body { padding:10px 12px; display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; }
    .iamccs-cai-card { border:1px solid #213f42; background:#091113; border-radius:6px; padding:8px; min-height:74px; }
    .iamccs-cai-k { color:#f0c96a; font-weight:800; font-size:11px; text-transform:uppercase; margin-bottom:4px; }
    .iamccs-cai-v { color:#d6e6e2; }
    .iamccs-cai-foot { padding:8px 12px; border-top:1px solid #1d3a3d; color:#9fc0bc; background:#0a1012; }
  \`;
  document.head.appendChild(style);
}

function render(node) {
  if (node._iamccsCineAudioInfoReady) return;
  node._iamccsCineAudioInfoReady = true;
  styles();

  const root = document.createElement("div");
  root.className = "iamccs-cai";
    root.innerHTML = \`
    <div class="iamccs-cai-head">
      <div>
        <div class="iamccs-cai-title">IAMCCS CineAudioInfo</div>
        <div class="iamccs-cai-sub">cine_linx audio I/O for TTS, AudioBoard lanes and Shotboard custom audio.</div>
      </div>
      <div class="iamccs-cai-actions">
        <button data-mode="export_tts_srt">Export Master</button>
        <button data-mode="export_speaker_stems">Export A/B</button>
        <button data-mode="inject_generated_audio">Inject Master</button>
        <button data-mode="inject_speaker_stems" class="primary">Inject A/B</button>
        <button data-mode="prepare_audio_board">Prepare Board</button>
        <button data-mode="inspect">Inspect</button>
      </div>
    </div>
    <div class="iamccs-cai-body">
      <div class="iamccs-cai-card">
        <div class="iamccs-cai-k">Input</div>
        <div class="iamccs-cai-v">cine_linx from Dialogue Editor / Bridge. Optional AUDIO from TTS when injecting.</div>
      </div>
      <div class="iamccs-cai-card">
        <div class="iamccs-cai-k">Output</div>
        <div class="iamccs-cai-v">SRT/text for TTS, audio_timeline_json for AudioBoard, cine_linx for Shotboard.</div>
      </div>
      <div class="iamccs-cai-card">
        <div class="iamccs-cai-k">Mode</div>
        <div class="iamccs-cai-v" data-current-mode></div>
      </div>
    </div>
    <div class="iamccs-cai-foot">Audio-first path: export A/B SRT -> generate TTS A/B -> inject speaker stems -> AudioBoard Arranger -> Shotboard V3.</div>
  \`;

  root.querySelectorAll("button[data-mode]").forEach((btn) => {
    btn.addEventListener("click", () => {
      setWidget(node, "mode", btn.dataset.mode);
      refresh();
    });
  });

  function refresh() {
    const mode = widget(node, "mode")?.value || "export_tts_srt";
    const lane = widget(node, "lane_injection_mode")?.value || "slice_master_by_existing_lanes";
    const fps = widget(node, "frame_rate")?.value || 24;
    const target = root.querySelector("[data-current-mode]");
    if (target) target.textContent = \`${mode} | ${lane} | ${fps} fps\`;
  }

  node.addDOMWidget?.("CineAudioInfo", "iamccs_cine_audio_info", root, { serialize: false });
  node.size = [Math.max(node.size?.[0] || 0, 620), Math.max(node.size?.[1] || 0, 260)];
  refresh();
}

app.registerExtension({
  name: "iamccs.cine_audio_info",
  nodeCreated(node) {
    const type = node?.comfyClass || node?.type || node?.constructor?.type || "";
    if (type === NODE || node?.type === NODE) render(node);
  },
});
