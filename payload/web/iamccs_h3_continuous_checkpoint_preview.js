import { app } from '/scripts/app.js';
import { api } from '/scripts/api.js';

function findNode(id) {
  const sid = String(id ?? '');
  return app.graph?._nodes?.find((node) => String(node.id) === sid && String(node.comfyClass || node.type) === 'IAMCCS_MiniMaxH3FL2VAContinuousAV') || null;
}

function ensurePanel(node) {
  if (node._iamccsContinuousCheckpointPanel) return node._iamccsContinuousCheckpointPanel;
  const root = document.createElement('div');
  root.style.cssText = 'box-sizing:border-box;width:100%;background:#10191d;color:#dce8eb;padding:8px;display:flex;flex-direction:column;gap:7px;border:1px solid #29424a;border-radius:6px';
  const title = document.createElement('div');
  title.textContent = 'LONG KEYFRAME · LIVE SEGMENT CHECKPOINT';
  title.style.cssText = 'font-weight:700;letter-spacing:.04em;color:#bfe7ef';
  const status = document.createElement('div');
  status.style.cssText = 'font-size:11px;color:#9eb5bc';
  const select = document.createElement('select');
  select.style.cssText = 'width:100%;background:#1b2c32;color:white;padding:5px;border:1px solid #35515a';
  const video = document.createElement('video');
  video.controls = true;
  video.loop = true;
  video.preload = 'metadata';
  video.style.cssText = 'display:block;width:100%;height:190px;object-fit:contain;background:#000';
  const link = document.createElement('a');
  link.target = '_blank';
  link.textContent = 'Open checkpoint video';
  link.style.cssText = 'color:#8ed9ea;font-size:11px';
  const reset = document.createElement('button');
  reset.type = 'button';
  reset.textContent = 'RESET SEGMENTS';
  reset.title = 'Delete every native video checkpoint and PSkf state from the current True Long run';
  reset.style.cssText = 'width:100%;margin-top:2px;padding:7px;border:1px solid #9b4242;border-radius:4px;background:#4b2023;color:#ffd9d9;font-weight:700;letter-spacing:.05em;cursor:pointer';
  root.append(title, status, select, video, link, reset);

  const records = () => Array.isArray(node.properties?.iamccs_continuous_checkpoints) ? node.properties.iamccs_continuous_checkpoints : [];
  function show() {
    const list = records();
    const item = list[Number(select.value)];
    if (!item) {
      status.textContent = 'No segment checkpoint yet.';
      video.removeAttribute('src');
      link.removeAttribute('href');
      return;
    }
    const url = api.apiURL('/view?' + new URLSearchParams({ filename:item.filename, subfolder:item.subfolder, type:item.type || 'output' }));
    video.src = url;
    link.href = url;
    status.textContent = `${item.label || `SEGMENT ${item.segment}/${item.total}`} · saved before final assembly`;
  }
  function refresh() {
    const list = records();
    select.replaceChildren();
    list.forEach((item, index) => {
      const option = document.createElement('option');
      option.value = String(index);
      option.textContent = item.label || `SEGMENT ${item.segment}/${item.total}`;
      select.append(option);
    });
    select.value = String(Math.max(0, list.length - 1));
    show();
  }
  select.addEventListener('change', show);
  reset.addEventListener('click', async () => {
    const run = String(node.properties?.iamccs_continuous_checkpoint_run || '');
    const count = records().length;
    if (!run) {
      status.textContent = 'No segment checkpoint run to reset.';
      return;
    }
    if (!window.confirm(`Delete all ${count} segment checkpoints and PSkf states from this True Long run?\n\n${run}`)) return;
    reset.disabled = true;
    reset.textContent = 'RESETTING…';
    try {
      const response = await api.fetchApi('/api/iamccs/h3/continuous/reset_segments', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({run}),
      });
      const result = await response.json();
      if (!response.ok || !result.ok) throw new Error(result.error || `HTTP ${response.status}`);
      node.properties.iamccs_continuous_checkpoints = [];
      delete node.properties.iamccs_continuous_checkpoint_run;
      refresh();
      status.textContent = `Reset complete · ${Number(result.removed_files || 0)} files removed.`;
      node.setDirtyCanvas?.(true, true);
      app.graph?.setDirtyCanvas?.(true, true);
    } catch (error) {
      status.textContent = `Reset failed · ${error.message}`;
    } finally {
      reset.disabled = false;
      reset.textContent = 'RESET SEGMENTS';
    }
  });
  node.addDOMWidget('continuous_segment_checkpoints', 'custom', root, { serialize:false, hideOnZoom:false, getMinHeight:()=>335, getMaxHeight:()=>380 });
  node._iamccsContinuousCheckpointPanel = { refresh };
  refresh();
  node.setSize?.(node.computeSize());
  return node._iamccsContinuousCheckpointPanel;
}

app.registerExtension({
  name: 'IAMCCS.H3.ContinuousSegmentCheckpoints',
  nodeCreated(node) {
    if (String(node.comfyClass || node.type) === 'IAMCCS_MiniMaxH3FL2VAContinuousAV') ensurePanel(node);
  },
  loadedGraphNode(node) {
    if (String(node.comfyClass || node.type) === 'IAMCCS_MiniMaxH3FL2VAContinuousAV') ensurePanel(node).refresh();
  },
});

api.addEventListener('iamccs-h3-continuous-checkpoint', (event) => {
  const item = event.detail || {};
  const node = findNode(item.node);
  if (!node) return;
  node.properties ||= {};
  if (node.properties.iamccs_continuous_checkpoint_run !== item.run) {
    node.properties.iamccs_continuous_checkpoint_run = item.run;
    node.properties.iamccs_continuous_checkpoints = [];
  }
  const records = node.properties.iamccs_continuous_checkpoints ||= [];
  const existing = records.findIndex((record) => Number(record.segment) === Number(item.segment));
  if (existing >= 0) records[existing] = item;
  else records.push(item);
  records.sort((a,b) => Number(a.segment || 0) - Number(b.segment || 0));
  ensurePanel(node).refresh();
  node.setDirtyCanvas?.(true, true);
});
