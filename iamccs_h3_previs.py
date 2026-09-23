"""Previs depth producer for the existing R42/R43 H3 control transport."""
import json
import math
import torch


def blocking_prompt(bindings_json):
    data = json.loads(bindings_json)
    if not isinstance(data, list) or len(data) > 32:
        raise ValueError('Bindings must be a JSON list of at most 32 subjects.')
    lines = ['Follow the supplied spatial control for camera perspective, parallax and blocking. '
             'Preserve the designed appearance from the image references. '
             'Use natural articulation inside the supplied coarse trajectories.']
    for item in data:
        if not isinstance(item, dict):
            raise ValueError('Each binding must contain proxy, subject and trajectory.')
        values = [str(item.get(key, '')).strip() for key in ('proxy', 'subject', 'trajectory')]
        if not all(values):
            raise ValueError('Each binding requires proxy, subject and trajectory descriptions.')
        lines.append(f'{values[1]} follows {values[2]}. Proxy label: {values[0]}. '
                     'The proxy defines blocking only; do not reproduce its primitive shape or material.')
    return '\n'.join(lines)


class IAMCCS_H3PrevisControl:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'cine_linx': ('IAMCCS_SUPERNODE_LINX',),
            'enabled': ('BOOLEAN', {'default': False}),
            'representation': (['direct_depth', 'rgb_proxy_to_depth'],),
            'source_fps': ('FLOAT', {'default': 24., 'min': 1., 'max': 240.}),
            'source_offset_seconds': ('FLOAT', {'default': 0., 'min': 0., 'max': 86400.}),
            'depth_polarity': (['near_white', 'near_black'],),
            'preprocess_resolution': ('INT', {'default': 512, 'min': 128, 'max': 2048, 'step': 64}),
            'bindings_json': ('STRING', {'multiline': True, 'default': '[{"proxy":"yellow cylinder","subject":"the protagonist in <Picture 1>","trajectory":"from screen left to centre, then stops on the foreground mark"}]'}),
        }, 'optional': {'previs_video': ('IMAGE', {'lazy': True})}}

    RETURN_TYPES = ('IAMCCS_SUPERNODE_LINX', 'IMAGE', 'STRING', 'STRING')
    RETURN_NAMES = ('cine_linx', 'exact_depth_preview', 'blocking_prompt', 'manifest')
    FUNCTION = 'inject'
    CATEGORY = 'IAMCCS/MiniMax H3/Previs'

    def check_lazy_status(self, cine_linx, enabled=False, previs_video=None, **kwargs):
        return ['previs_video'] if enabled and previs_video is None else []

    def inject(self, cine_linx, enabled, representation, source_fps,
               source_offset_seconds, depth_polarity, preprocess_resolution,
               bindings_json, previs_video=None):
        if not enabled:
            return cine_linx, None, '', json.dumps({'enabled': False})
        if not isinstance(cine_linx, dict):
            raise ValueError('PREVIS requires the Settings/CineH3Input bus before Shotboard.')
        if not torch.is_tensor(previs_video) or previs_video.ndim != 4 or len(previs_video) < 1:
            raise ValueError('Connect decoded previs IMAGE frames; a filename is not an IMAGE batch.')
        if not math.isfinite(source_fps) or source_fps <= 0 or not math.isfinite(source_offset_seconds) or source_offset_seconds < 0:
            raise ValueError('Invalid previs FPS or source offset.')
        prompt = blocking_prompt(bindings_json)
        from .iamccs_cine_h3_bus import IAMCCS_CineH3FunControlInput
        producer = IAMCCS_CineH3FunControlInput
        if representation == 'rgb_proxy_to_depth':
            depth = producer._preprocess(previs_video, 'depth_anything', preprocess_resolution)
        elif representation == 'direct_depth':
            if previs_video.shape[-1] != 3:
                raise ValueError('Direct depth requires an RGB IMAGE batch containing grayscale depth.')
            # Reject ID/color renders: they do not encode geometric distance.
            if float((previs_video[..., 0] - previs_video[..., 1]).abs().max()) > .03 or float((previs_video[..., 1] - previs_video[..., 2]).abs().max()) > .03:
                raise ValueError('Direct depth is not grayscale. Use RGB proxy to depth for colored primitives.')
            depth = previs_video
        else:
            raise ValueError('Unknown previs representation.')
        if not bool(torch.isfinite(depth).all()) or float(depth.min()) < 0 or float(depth.max()) > 1:
            raise ValueError('Depth must contain finite normalized values in [0,1].')
        if representation == 'direct_depth' and depth_polarity == 'near_black':
            depth = 1 - depth
        result = producer().inject(cine_linx, source_fps, control_video=depth)
        out = result['result'][0] if isinstance(result, dict) else result[0]
        manifest = {'schema': 'iamccs.h3.previs', 'version': 1, 'enabled': True,
                    'representation': representation, 'source_fps': source_fps,
                    'source_offset_seconds': source_offset_seconds, 'frames': len(depth),
                    'camera_authority': True, 'geometry_authority': True,
                    'identity_authority': False, 'proxy_rgb_to_model': False,
                    'bindings': json.loads(bindings_json),
                    'binding_contract': 'text direction, not deterministic object tracking',
                    'depth_polarity': 'near_white', 'prompt': prompt}
        out['resources']['iamccs_h3_previs_manifest'] = manifest
        out['outputs']['iamccs_h3_previs_manifest'] = manifest
        out['resources']['iamccs_minimax_h3_control_video_meta']['previs'] = manifest
        return out, depth, prompt, json.dumps(manifest, ensure_ascii=False, indent=2)


NODE_CLASS_MAPPINGS = {'IAMCCS_H3PrevisControl': IAMCCS_H3PrevisControl}
NODE_DISPLAY_NAME_MAPPINGS = {'IAMCCS_H3PrevisControl': 'IAMCCS H3 PREVIS · Camera + Subject Blocking'}
