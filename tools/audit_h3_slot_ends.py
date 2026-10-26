"""Inspect authored multi-shot cuts in a saved native H3 video, without loading models."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def audit(video, output, final=None):
    metadata = json.loads(Path(str(video) + '.iamccs.json').read_text(encoding='utf-8'))
    generation = metadata['generation']
    plan = generation['shotplan']
    chunk = plan['chunks'][generation['segment_index']]
    if not chunk.get('multishot_joint_window'):
        raise ValueError('This audit requires a joint multi-shot H3 native checkpoint.')
    start = int(chunk['timeline_start_frame'])
    prefix = int(chunk.get('latent_tail_context_prefix_frames', 0))
    count = int(metadata['frame_count'])
    cuts = sorted({int(g['global_frame']) - start for g in chunk['guides']
                   if g.get('kind') == 'image' and 0 < int(g['global_frame']) - start < count})
    wanted = {f for cut in cuts + [count] for f in range(max(0, cut-4), min(count, cut+1))}
    capture = cv2.VideoCapture(str(video))
    comparison = cv2.VideoCapture(str(final)) if final else None
    if not capture.isOpened() or comparison is not None and not comparison.isOpened():
        raise ValueError('Cannot open the requested video files.')
    frames, detail, mismatches = {}, [], []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            index = len(detail)
            if index in wanted:
                frames[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detail.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            if comparison is not None:
                other_ok, other = comparison.read()
                if not other_ok or other.shape != frame.shape or not np.array_equal(frame, other):
                    mismatches.append(index)
        if len(detail) != count:
            raise ValueError(f'Video has {len(detail)} frames; checkpoint declares {count}.')
        comparison_extra = bool(comparison.read()[0]) if comparison is not None else False
    finally:
        capture.release()
        if comparison is not None:
            comparison.release()

    rows = []
    for cut in cuts:
        phase = (cut + prefix) % 17
        bounds = (0, 1, 5, 9, 13, 17)  # Stock H3 FRAME_PER_TOKEN = (1, 4, 4, 4, 4).
        lo = max(v for v in bounds if v <= phase)
        hi = min(v for v in bounds if v > phase)
        rows.append({
            'local_cut_frame': cut, 'global_cut_frame': start + cut,
            'sample_phase_mod_17': phase, 'cut_inside_video_token': phase != lo,
            'token_visible_frame_range': [cut-phase+lo, cut-phase+hi-1],
            'preceding_frame_detail_proxy': [round(v, 3) for v in detail[max(0, cut-4):cut]],
        })
    result = {
        'native_video': str(video), 'frame_count': count, 'cuts': rows,
        'note': 'Laplacian variance is a detail proxy. Token crossing is structural; neither proves the cause of a visual artifact.',
    }
    if final:
        result['final_comparison'] = {
            'video': str(final), 'pixel_identical': not mismatches and not comparison_extra,
            'different_frame_count': len(mismatches), 'has_extra_frames': comparison_extra,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix('.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    width, height = frames[min(frames)].shape[1], frames[min(frames)].shape[0]
    sheet = Image.new('RGB', (width*5, (height+26)*(len(cuts)+1)), '#121212')
    draw = ImageDraw.Draw(sheet)
    for row, cut in enumerate(cuts + [count]):
        for column, index in enumerate(range(max(0, cut-4), min(count, cut+1))):
            x, y = column*width, row*(height+26)
            sheet.paste(Image.fromarray(frames[index]), (x, y))
            label = f'Frame {index} | detail proxy {detail[index]:.1f}'
            if index == cut:
                label += ' | NEW SHOT'
            draw.text((x+5, y+height+5), label, fill='white')
    sheet.save(output.with_suffix('.jpg'), quality=95)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('native', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--final', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.native, args.output, args.final), indent=2))
