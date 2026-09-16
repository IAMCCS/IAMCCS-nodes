# SPDX-License-Identifier: GPL-3.0-or-later
"""Audio clock and immutable-prefix utilities for LatentGoAhead.

Independent implementation using seconds and native H3 40Hz audio tokens.
Not enabled in sampling until target-only position shifting is verified.
"""
import math
import copy


def shifted_layout(layout, origin_seconds):
    result=copy.copy(layout)
    result.position_ids=layout.position_ids.clone()
    spans=[(a,b) for a,b,kind in layout.segments if kind=='audio']
    if len(spans)!=1:
        raise RuntimeError('LatentGoAhead requires exactly one target audio segment')
    a,b=spans[0]
    result.position_ids[a:b,0]+=float(origin_seconds)*40
    return result


def model_with_audio_origin(model, origin_seconds):
    """Instance-local sampler wrapper. Never patches ComfyUI global classes."""
    clone=model.clone()
    previous=clone.model_options.get('model_function_wrapper')
    def wrapper(apply_model,args):
        updated=dict(args)
        cond=dict(args['c'])
        payload=cond.get('minimax_payload')
        if not isinstance(payload,dict) or payload.get('layout') is None:
            raise RuntimeError('LatentGoAhead audio-origin payload missing; refusing unaligned sampling')
        payload=dict(payload)
        payload['layout']=shifted_layout(payload['layout'],origin_seconds)
        cond['minimax_payload']=payload
        updated['c']=cond
        if previous is not None:return previous(apply_model,updated)
        return apply_model(updated['input'],updated['timestep'],**cond)
    clone.set_model_unet_function_wrapper(wrapper)
    return clone


def prefix_window(*, delivered_frames, fps, source_tokens, source_origin_seconds,
                  context_seconds):
    if fps <= 0 or delivered_frames < 1 or source_tokens < 1 or context_seconds <= 0:
        raise ValueError('Invalid LatentGoAhead audio clock')
    boundary = (delivered_frames - 1) / fps
    end = delivered_frames / fps
    first = max(0, math.floor((end-context_seconds-source_origin_seconds)*40 + 1e-8))
    last = min(source_tokens, math.ceil((end-source_origin_seconds)*40 - 1e-8))
    if first >= last or source_origin_seconds + last/40 < boundary-1e-6:
        raise ValueError('Audio history does not reach the video boundary')
    origin = source_origin_seconds + first/40 - boundary
    return first, last, origin


def make_prefix_target(source, *, first, last, origin_seconds, output_seconds):
    import torch
    if source.ndim != 4 or source.shape[0] != 1 or not 0 <= first < last <= source.shape[-1]:
        raise ValueError('Invalid original audio latent prefix')
    if origin_seconds >= 0 or output_seconds <= 0:
        raise ValueError('Frozen prefix must start before target video')
    prefix=source[...,first:last].detach().clone()
    total=math.ceil((output_seconds-origin_seconds)*40-1e-8)
    if total <= prefix.shape[-1]: raise ValueError('No future audio target rows')
    target=source.new_zeros((*source.shape[:-1],total))
    target[...,:prefix.shape[-1]]=prefix
    mask=torch.ones_like(target)
    mask[...,:prefix.shape[-1]]=0
    return target,mask,prefix


def assert_prefix_unchanged(sampled, prefix):
    import torch
    actual=sampled[...,:prefix.shape[-1]].detach().cpu()
    expected=prefix.detach().cpu()
    # Native H3 audio scaling involves a multiply/divide round trip. Permit
    # only float32 roundoff, not a generatively modified audio prefix.
    tolerance=torch.finfo(torch.float32).eps
    if not torch.allclose(actual.float(),expected.float(),rtol=tolerance,atol=tolerance,equal_nan=False):
        raise RuntimeError('LatentGoAhead frozen audio prefix changed during denoising')


def trim_decoded_audio(waveform, *, origin_seconds, output_seconds, sample_rate):
    start=round(-origin_seconds*sample_rate)
    count=round(output_seconds*sample_rate)
    if start < 0 or count < 1 or start+count > waveform.shape[-1]:
        raise ValueError('Decoded audio does not cover requested interval')
    return waveform[...,start:start+count]
