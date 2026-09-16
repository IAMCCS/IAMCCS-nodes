# SPDX-License-Identifier: GPL-3.0-or-later
"""Manual, duration-preserving post treatment of existing LatentGoAhead masters."""
import asyncio
import json
import re
import subprocess
import uuid
from pathlib import Path


def run_path(run):
    import folder_paths
    if not re.fullmatch(r'[0-9a-f]{32}', str(run)):
        raise ValueError('Invalid LatentGoAhead run ID')
    root = Path(folder_paths.get_output_directory()).resolve() / 'IAMCCS' / 'LatentGoAhead'
    path = (root / run).resolve()
    if path.parent != root.resolve() or not path.is_dir():
        raise ValueError('Run is not a local LatentGoAhead checkpoint directory')
    return path


def inspect_run(path):
    import av
    clips = sorted(path.glob('interval_*_delivered.mp4'))
    if len(clips) < 2 or not (path / 'final_film.mp4').is_file():
        raise ValueError('A completed master with at least two delivered intervals is required')
    counts = []
    fps = None
    for clip in clips:
        with av.open(str(clip)) as source:
            stream = source.streams.video[0]
            rate = float(stream.average_rate)
            if fps is not None and rate != fps:
                raise ValueError('Interval frame rates differ')
            fps = rate
            counts.append(stream.frames or sum(1 for _ in source.decode(video=0)))
    with av.open(str(path / 'final_film.mp4')) as source:
        stream = source.streams.video[0]
        total = stream.frames or sum(1 for _ in source.decode(video=0))
        if total != sum(counts) or float(stream.average_rate) != fps:
            raise ValueError('Master and interval timing differ; refusing ambiguous edits')
    boundaries = [sum(counts[:i]) for i in range(1, len(counts))]
    return dict(fps=fps, frames=total, boundaries=boundaries, counts=counts)


def windows_for(info, edits):
    windows = []
    for index, entry in enumerate(edits):
        seam = int(entry['seam'])
        radius = int(entry.get('radius', 1))
        offset = int(entry.get('offset', 0))
        method = str(entry.get('method', 'smoothstep')).strip().lower()
        if not 0 <= seam < len(info['boundaries']) or not 0 <= radius <= 6 or abs(offset) > 6:
            raise ValueError('Invalid seam controls')
        if method not in ('linear', 'smoothstep', 'cosine'):
            raise ValueError('Invalid seam blend curve')
        if not radius:
            continue
        b = info['boundaries'][seam] + offset
        lo, hi = b-radius-1, b+radius
        if lo < 0 or hi >= info['frames']:
            raise ValueError('Blend exceeds the available frames')
        windows.append(dict(lo=lo, hi=hi, method=method, seam=seam))
    windows.sort(key=lambda item: item['lo'])
    if any(a['hi'] >= b['lo'] for a,b in zip(windows, windows[1:])):
        raise ValueError('Blend windows overlap')
    return windows


def smooth_frames(frames, windows):
    """Replace only interior frames between endpoints, preserving frame count."""
    import numpy as np
    source = iter(frames)
    index = 0
    for window in windows:
        lo, hi = window['lo'], window['hi']
        method = window.get('method', 'smoothstep')
        while index < lo:
            yield next(source)
            index += 1
        left = next(source)
        yield left
        index += 1
        right = None
        while index <= hi:
            right = next(source)
            index += 1
        for j in range(1, hi-lo):
            alpha = j / (hi-lo)
            if method == 'smoothstep':
                alpha = alpha * alpha * (3 - 2 * alpha)
            elif method == 'cosine':
                alpha = (1 - np.cos(np.pi * alpha)) / 2
            yield np.floor(left.astype(np.float32)*(1-alpha)+right.astype(np.float32)*alpha+0.5).clip(0,255).astype(np.uint8)
        yield right
    yield from source


def export_run(path, edits):
    import av
    from .iamccs_minimax_h3_shotboard import _find_ffmpeg
    info = inspect_run(path)
    windows = windows_for(info, edits)
    # Empty windows mean an explicit CUT-only export: preserve all source frames.
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise ValueError('FFmpeg is unavailable')
    stem = 'seam_review_' + uuid.uuid4().hex
    intermediate = path / (stem + '_video.mp4')
    output = path / (stem + '.mp4')
    try:
        with av.open(str(path / 'final_film.mp4')) as source, av.open(str(intermediate), 'w') as dest:
            original = source.streams.video[0]
            stream = dest.add_stream('libx264', rate=original.average_rate)
            stream.width,stream.height = original.width,original.height
            stream.pix_fmt = 'yuv420p'
            stream.options = {'crf':'16','preset':'fast'}
            count = 0
            for pixels in smooth_frames((f.to_ndarray(format='rgb24') for f in source.decode(video=0)), windows):
                frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
                frame.pts = count
                for packet in stream.encode(frame):
                    dest.mux(packet)
                count += 1
            for packet in stream.encode():
                dest.mux(packet)
            if count != info['frames']:
                raise ValueError('Frame count changed; export aborted')
        subprocess.run([ffmpeg,'-nostdin','-v','error','-n','-i',str(intermediate),'-i',str(path/'final_film.mp4'),
                        '-map','0:v:0','-map','1:a:0?','-c','copy',str(output)],check=True,capture_output=True,timeout=300)
        (path / (stem+'.json')).write_text(json.dumps(dict(source='final_film.mp4',edits=edits,windows=windows,
            frames=info['frames'],audio='original packets copied',method='duration-preserving local dissolve'),indent=2),encoding='utf-8')
        source_workflow = path / 'final_film.workflow.json'
        if source_workflow.is_file():
            workflow = json.loads(source_workflow.read_text(encoding='utf-8'))
            workflow.setdefault('extra', {})['iamccs_seam_export'] = dict(edits=edits, windows=windows, video=output.name)
            for node in workflow.get('nodes', []):
                if node.get('type') == 'IAMCCS_MiniMaxH3LatentGoAhead':
                    node.setdefault('properties', {})['iamccs_ahead_seams'] = dict(run=path.name, edits=edits)
            output.with_suffix('.workflow.json').write_text(json.dumps(workflow,ensure_ascii=False,indent=2),encoding='utf-8')
    finally:
        if intermediate.exists():
            intermediate.unlink()
    return dict(filename=output.name,subfolder='IAMCCS/LatentGoAhead/'+path.name,type='output',**info)


def register_routes():
    from server import PromptServer
    from aiohttp import web
    @PromptServer.instance.routes.get('/iamccs/ahead/cuts/{run}')
    async def cuts_metadata(request):
        from .iamccs_ahead_cuts import inspect_cuts
        try:
            result = await asyncio.to_thread(inspect_cuts, run_path(request.match_info['run']))
            return web.json_response(result)
        except (ValueError, OSError, KeyError) as exc:
            return web.json_response({'error':str(exc)}, status=400)
    @PromptServer.instance.routes.get('/iamccs/ahead/seams/{run}')
    async def metadata(request):
        try:
            result = await asyncio.to_thread(inspect_run, run_path(request.match_info['run']))
            return web.json_response(result)
        except (ValueError, OSError) as exc:
            return web.json_response({'error':str(exc)},status=400)
    @PromptServer.instance.routes.post('/iamccs/ahead/seams/export')
    async def export(request):
        try:
            payload = await request.json()
            edits = payload.get('edits',[])
            if not isinstance(edits,list) or len(edits)>100:
                raise ValueError('Invalid edit list')
            result = await asyncio.to_thread(export_run,run_path(payload.get('run','')),edits)
            return web.json_response(result)
        except Exception as exc:
            return web.json_response({'error':str(exc)},status=400)
