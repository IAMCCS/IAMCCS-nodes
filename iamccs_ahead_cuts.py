"""Validated, one-based inclusive delivery boundaries; original latents stay intact."""
import hashlib
import json
import logging


def contract(plan):
    data = {k: plan.get(k) for k in ('width', 'height')}
    data['chunks'] = [{k: c.get(k) for k in ('frame_count', 'first_image', 'last_image', 'prompt', 'creative_prompt')} for c in plan['chunks']]
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate(recipe, plan):
    value = json.loads(recipe) if isinstance(recipe, str) else recipe
    if not value:
        return {}
    if not isinstance(value, dict):
        raise ValueError('Invalid exact cut recipe')
    if value.get('contract') != contract(plan):
        logging.warning('Exact cut ignored: timeline or canvas changed. Freeze-aware trim remains active; choose a new OUT after this generation.')
        return {}
    cuts = value.get('cuts')
    if not isinstance(cuts, dict) or len(cuts) > len(plan['chunks']):
        raise ValueError('Invalid exact cut recipe')
    result = {}
    for key, keep in cuts.items():
        if not isinstance(key, str) or not key.isdecimal():
            raise ValueError('Invalid cut interval')
        index = int(key)
        if not 0 <= index < len(plan['chunks']) or type(keep) is not int or not 2 <= keep <= int(plan['chunks'][index]['frame_count']):
            raise ValueError('Exact cut must retain 2 to frame_count frames')
        result[index] = keep
    return result


def inspect_cuts(path):
    import av
    from safetensors import safe_open
    sidecar = path / 'final_film.workflow.json'
    if sidecar.exists():
        plan = json.loads(sidecar.read_text(encoding='utf8'))['extra']['iamccs_generation_provenance']['resolved_shotplan']
    else:
        metadata = path / 'final_film.metadata.json'
        if not metadata.exists():
            raise ValueError('This run has no queue-time provenance. Complete a new generation to enable safe exact cuts.')
        plan = json.loads(metadata.read_text(encoding='utf8'))['resolved_shotplan']
    intervals = []
    for index, chunk in enumerate(plan['chunks']):
        delivered_filename = f'interval_{index+1:04d}_delivered.mp4'
        raw_av_filename = f'interval_{index+1:04d}_raw_av_review.mp4'
        raw_video_filename = f'interval_{index+1:04d}_video_review.mp4'
        filmstrip_filename = (
            raw_av_filename if (path / raw_av_filename).is_file()
            else raw_video_filename if (path / raw_video_filename).is_file()
            else delivered_filename
        )
        with av.open(str(path / delivered_filename)) as media:
            stream = media.streams.video[0]
            delivered_frames = stream.frames or sum(1 for _ in media.decode(video=0))
            fps = float(stream.average_rate)
        with av.open(str(path / filmstrip_filename)) as media:
            stream = media.streams.video[0]
            raw_frames = stream.frames or sum(1 for _ in media.decode(video=0))
        trim_frames = max(0, int(raw_frames) - int(delivered_frames))
        trim_source = 'unknown'
        checkpoint = path / f'interval_{index+1:04d}.safetensors'
        if checkpoint.is_file():
            try:
                with safe_open(str(checkpoint), framework='pt', device='cpu') as tensors:
                    metadata = tensors.metadata() or {}
                trim_frames = max(0, int(metadata.get('freeze_tail_trim_frames', trim_frames)))
                trim_source = str(metadata.get('freeze_tail_trim_source', 'freeze_aware'))
            except (OSError, ValueError, TypeError):
                pass
        suggested_out = max(2, int(raw_frames) - trim_frames) if trim_frames else int(raw_frames)
        intervals.append(dict(
            filename=raw_av_filename if (path / raw_av_filename).is_file() else delivered_filename,
            delivered_filename=delivered_filename,
            filmstrip_filename=filmstrip_filename,
            frames=int(raw_frames),
            raw_frames=int(raw_frames),
            delivered_frames=int(delivered_frames),
            fps=fps,
            analysis=dict(
                detected=bool(trim_frames),
                trim_frames=trim_frames,
                first_frozen_frame=suggested_out + 1 if trim_frames else None,
                suggested_out=suggested_out,
                source=trim_source,
                explanation=(
                    f'The terminal-motion detector found {trim_frames} repeated/near-static frame(s). '
                    f'Keep frame {suggested_out} as the last included frame.'
                    if trim_frames else
                    'No terminal freeze was detected. Keep the OUT handle at the right edge unless visual review shows a false negative.'
                ),
            ),
        ))
    return dict(contract=contract(plan), intervals=intervals)
