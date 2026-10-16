// SwapChar source duration; no model execution or AI requests.
const value = (node, name, fallback = 0) => node?.widgets?.find(w => w.name === name)?.value ?? fallback;
const type = node => node?.comfyClass || node?.type;

function sourceLoader(inputNode, inputName, pool, seen = new Set()) {
    if (!inputNode || seen.has(inputNode)) return null;
    seen.add(inputNode);
    const graph = inputNode.graph;
    const input = inputNode.inputs?.find(item => item.name === inputName);
    const link = graph?.links?.[input?.link];
    if (link && Number(link.origin_id) < 0) {
        const parent = pool.find(candidate => candidate.subgraph === graph);
        const socket = parent?.inputs?.[link.origin_slot];
        if (socket) return sourceLoader(parent, socket.name, pool, seen);
    }
    const source = graph?.getNodeById?.(link?.origin_id);
    if (/^VHS_LoadVideo/.test(type(source) || '')) return source;
    for (const socket of source?.inputs || []) {
        if (['IMAGE', '*'].includes(socket.type)) {
            const loader = sourceLoader(source, socket.name, pool, seen);
            if (loader) return loader;
        }
    }
    return null;
}

export function swapVideoSource(node, pool) {
    const prompters = pool.filter(n => type(n) === 'IAMCCS_Prompter' && ![2,4].includes(n.mode));
    const videos = prompters.map(n => {
        try { return JSON.parse(String(value(n, 'project_data', '{}'))).edit_video; } catch { return null; }
    }).filter(video => video?.path);
    const policy = value(node, 'h3_edit_video_source', 'auto');
    if (policy !== 'backend' && videos.length > 1) throw Error('More than one connected Prompter supplies a video. Select one source.');
    if (policy !== 'backend' && videos.length === 1) {
        const video = videos[0];
        return {kind:'prompter', path:video.path, start:Number(video.start_seconds || 0), trim:Number(video.duration_seconds || 0)};
    }
    if (policy === 'prompter') throw Error('Choose one source video in the connected Prompter.');
    const inputs = pool.filter(n => type(n) === 'IAMCCS_H3FaceSwapInput' && ![2,4].includes(n.mode));
    if (inputs.length !== 1) throw Error('Connect one H3 Swap Input to read the source duration.');
    let loader = sourceLoader(inputs[0], 'source_video', pool);
    // Native subgraph input proxies may hide the external IMAGE link.
    if (!loader) {
        const candidates = pool.filter(n => /^VHS_LoadVideo/.test(type(n) || '') && ![2,4].includes(n.mode));
        if (candidates.length === 1) loader = candidates[0];
    }
    if (!loader) throw Error('Cannot identify the video loader connected to H3 Swap Input.');
    const path = String(value(loader, 'video', '')).trim();
    if (!path) throw Error('Select the SwapChar source video first.');
    return {kind:'backend', path, force:Number(value(loader,'force_rate')), skip:Number(value(loader,'skip_first_frames')),
        every:Math.max(1, Number(value(loader,'select_every_nth',1))), cap:Number(value(loader,'frame_load_cap')),
        fps:Number(value(inputs[0],'source_fps',24)), start:Number(value(node,'v2v_source_offset_seconds'))};
}

export function swapVideoDuration(source, metadata, outputFps) {
    const rawFrames = Number(metadata.frame_count), rawFps = Number(metadata.fps);
    if (!(rawFrames > 0 && rawFps > 0)) throw Error('Invalid video duration/FPS.');
    let seconds;
    if (source.kind === 'prompter') {
        seconds = rawFrames / rawFps - source.start;
        if (source.trim > 0) seconds = Math.min(seconds, source.trim);
        seconds = Math.round(seconds * 24) / 24;
    } else {
        let frames = source.force > 0 ? Math.floor(rawFrames / rawFps * source.force) : rawFrames;
        frames = Math.ceil(Math.max(0, frames - source.skip) / source.every);
        if (source.cap > 0) frames = Math.min(frames, source.cap);
        seconds = frames / source.fps;
    }
    // Source offset is applied again by the swap backend after video loading.
    seconds -= Number(source.offset || 0);
    if (source.kind === 'backend') seconds -= source.start;
    const frames = Math.floor(seconds * outputFps + 1e-6);
    if (frames < 1) throw Error('The selected video range is empty.');
    return frames / outputFps;
}

export function createSwapDuration(node, {api, mode, connected, setDuration, refresh}) {
    let requestId = 0, pending = null, disposed = false, lastError = '', failedKey = '';
    const state = () => { node.properties ||= {}; return node.properties.iamccs_swap_duration ||= {mode:'source'}; };
    const tick = async (force = false) => {
        if (disposed || mode() !== 'v2va_face_swap_lora') return;
        let source, key;
        try {
            source = swapVideoSource(node, connected());
            source.offset = source.kind === 'prompter' ? Number(value(node,'v2v_source_offset_seconds')) : 0;
            key = JSON.stringify([source, Number(value(node,'frame_rate',24))]);
        } catch (error) {
            if (lastError !== error.message) { requestId++; pending = null; lastError = error.message; refresh(); }
            return;
        }
        const data = state();
        if (!force && key === data.key && data.seconds > 0) {
            lastError = '';
            if (data.mode !== 'custom' && Math.abs(Number(value(node,'duration_seconds')) - data.seconds) > 1e-6) setDuration(data.seconds, 'swap_video');
            return;
        }
        if (pending?.key === key) return pending.promise;
        if (!force && failedKey === key) return;
        if (data.key && data.key !== key) data.mode = 'source';
        data.key = key; data.seconds = 0;
        const id = ++requestId;
        const promise = (async () => {
            try {
                const response = await api.fetchApi('/iamccs/prompter/edit-video/metadata', {
                    method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:source.path})});
                const report = await response.json();
                if (!response.ok || !report.ok) throw Error(report.error || 'Cannot read video duration. Restart ComfyUI after updating.');
                if (disposed || id !== requestId || mode() !== 'v2va_face_swap_lora') return;
                const seconds = swapVideoDuration(source, report.video, Number(value(node,'frame_rate',24)));
                data.key = key; data.seconds = seconds; lastError = '';
                if (data.mode !== 'custom') setDuration(seconds, 'swap_video');
                else refresh();
            } catch (error) {
                if (id === requestId) { failedKey = key; lastError = error.message; refresh(); }
            } finally { if (pending?.id === id) pending = null; }
        })();
        pending = {key, id, promise};
        return promise;
    };
    node._iamccsSwapDurationFromShotboard = seconds => {
        if (mode() !== 'v2va_face_swap_lora') return;
        state().mode = 'custom';
        setDuration(seconds, 'custom');
    };
    node._iamccsSwapDurationReady = async () => {
        if (mode() !== 'v2va_face_swap_lora') return;
        // Also detects a source file replaced in place under the same filename.
        await tick(true);
        if (state().mode !== 'custom' && lastError) throw Error(`SwapChar duration: ${lastError}`);
    };
    return {
        tick,
        dispose() { disposed = true; requestId++; delete node._iamccsSwapDurationFromShotboard; delete node._iamccsSwapDurationReady; },
        render(container) {
            const data = state();
            const box = document.createElement('section'); box.style.cssText='grid-column:1/-1;display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:10px;border:1px solid #a8844b;border-radius:8px;';
            const label = document.createElement('strong'); label.textContent='SWAPCHAR DURATION';
            const select = document.createElement('select');
            select.add(new Option('VIDEO DURATION · automatic', 'source'));
            select.add(new Option('CUSTOM DURATION · synced with Shotboard', 'custom'));
            select.value = data.mode === 'custom' ? 'custom' : 'source';
            select.onchange = () => { data.mode = select.value; if (data.mode === 'source' && data.seconds > 0) setDuration(data.seconds, 'swap_video'); else refresh(); };
            const info = document.createElement('small');
            info.textContent = lastError || (data.seconds > 0 ? `Video range: ${data.seconds.toFixed(3)} s. Editing duration here or in Shotboard selects CUSTOM.` : 'Reading video duration…');
            box.append(label, select, info); container.append(box);
        },
    };
}

export function installSwapDurationQueueGuard(app) {
    if (app._iamccsSwapDurationQueueGuard) return;
    app._iamccsSwapDurationQueueGuard = true;
    const graphToPrompt = app.graphToPrompt;
    app.graphToPrompt = async function (...args) {
        const visit = async graph => {
            for (const node of graph?._nodes || []) {
                if (![2,4].includes(node.mode)) await node._iamccsSwapDurationReady?.();
                if (node.subgraph) await visit(node.subgraph);
            }
        };
        await visit(this.graph);
        return graphToPrompt.apply(this, args);
    };
}
