"""Delivery-only AV joins; never changes latent history or checkpoints."""


def periodic_flash_guard(clips, mode='auto', sensitivity=2.0, radius=2):
    """Repair isolated MiniMax-H3 17-frame decode flashes without retiming.

    H3's temporal VAE works on a 17-frame cadence.  Some local decoders expose
    a luminance/appearance spike at those internal boundaries.  AUTO measures
    each candidate against neighbouring motion and replaces only a strong
    isolated window.  The input tensors and frame counts are preserved.
    """
    import torch
    mode = str(mode or 'auto').strip().lower()
    if mode not in ('off', 'auto', 'all_periodic'):
        raise ValueError('Unknown H3 flash guard policy')
    sensitivity = max(1.1, min(5.0, float(sensitivity)))
    radius = max(1, min(6, int(radius)))
    if mode == 'off':
        return list(clips), []
    treated=[];audit=[]
    for clip_index, original in enumerate(clips):
        clip=original.clone()
        if len(clip) < 12:
            treated.append(clip);continue
        # Downsample only for detection.  Full-resolution frames are blended.
        step_y=max(1,int(clip.shape[1])//96);step_x=max(1,int(clip.shape[2])//96)
        probe=clip[:,::step_y,::step_x].float()
        changes=(probe[1:]-probe[:-1]).abs().mean(dim=tuple(range(1,probe.ndim)))
        used=[]
        # Observed H3 local VAE boundaries are transitions 5 + 17*n.
        for expected in range(5, len(clip)-1, 17):
            candidates=range(max(0,expected-1),min(len(changes),expected+2))
            transition=max(candidates,key=lambda n:float(changes[n]))
            neighbours=[float(changes[n]) for n in range(max(0,transition-8),min(len(changes),transition+9))
                        if abs(n-transition)>2]
            if not neighbours:continue
            local=float(torch.tensor(neighbours).median())
            peak=float(changes[transition]);ratio=peak/max(local,1e-6)
            selected=mode=='all_periodic' or (peak>=0.025 and ratio>=sensitivity)
            if not selected:continue
            lo=max(0,transition-radius);hi=min(len(clip)-1,transition+radius+1)
            if hi-lo<3 or any(not (hi<u or lo>v) for u,v in used):continue
            left=original[lo];right=original[hi]
            weights=torch.linspace(0,1,hi-lo+1,device=clip.device,dtype=clip.dtype)
            weights=weights.square()*(3-2*weights)
            shape=(-1,)+(1,)*(clip.ndim-1)
            clip[lo:hi+1]=left.unsqueeze(0)*(1-weights.reshape(shape))+right.unsqueeze(0)*weights.reshape(shape)
            used.append((lo,hi));audit.append({
                'interval':clip_index+1,'expected_transition':expected,
                'detected_transition':transition,'window_start':lo,'window_end':hi,
                'peak_change':round(peak,6),'local_change':round(local,6),
                'peak_ratio':round(ratio,3),'method':'smoothstep','duration_preserved':True})
        treated.append(clip)
    return treated,audit


def click_safe_audio(waveforms, sample_rate, milliseconds=40):
    """Concatenate intervals and suppress sample discontinuities in-place in time.

    A short equal-power fade to zero and back is applied around every boundary.
    Unlike an audio crossfade, this keeps the exact sample count and therefore
    cannot move picture sync or shorten the delivery.
    """
    import math
    import torch
    if not waveforms:
        raise ValueError('At least one audio interval is required')
    audio = torch.cat(waveforms, dim=-1).clone()
    fade = max(0, round(float(milliseconds) * int(sample_rate) / 1000.0))
    if fade == 0 or len(waveforms) == 1:
        return audio
    boundary = 0
    for left, right in zip(waveforms[:-1], waveforms[1:]):
        boundary += left.shape[-1]
        width = min(fade, boundary, audio.shape[-1] - boundary, left.shape[-1], right.shape[-1])
        if width < 2:
            continue
        phase = torch.linspace(0, math.pi / 2, width, device=audio.device, dtype=audio.dtype)
        audio[..., boundary-width:boundary] *= torch.cos(phase)
        audio[..., boundary:boundary+width] *= torch.sin(phase)
    return audio


def assemble(clips, waveforms, sample_rate, fps=24, mode='none', overlap=9,
             audio_join='hard_cut', audio_smoothing_ms=20):
    import torch
    if mode not in ('none', 'linear', 'smoothstep'):
        raise ValueError('Unknown AV join blend')
    if len(clips) != len(waveforms) or not clips:
        raise ValueError('AV intervals must match')
    if audio_join not in ('hard_cut', 'click_safe'):
        raise ValueError('Unknown audio join policy')
    if mode == 'none' or len(clips) == 1:
        audio = (click_safe_audio(waveforms, sample_rate, audio_smoothing_ms)
                 if audio_join == 'click_safe' else torch.cat(waveforms, dim=-1))
        return torch.cat(clips), audio
    overlap = int(overlap)
    if overlap < 1 or any(len(c) <= 2*overlap for c in clips):
        raise ValueError('Overlap needs more than twice its frame count in every interval. Reduce blend frames or keep a longer cut.')
    parts=[];audio_parts=[]
    # Integer sample boundaries derive from the same frame clock as video.
    for i, (clip, waveform) in enumerate(zip(clips, waveforms)):
        start = overlap if i else 0
        end = len(clip)-overlap if i+1<len(clips) else len(clip)
        expected = round(len(clip)*sample_rate/fps)
        if waveform.shape[-1] < expected:
            raise ValueError('Decoded audio is shorter than its video interval')
        parts.append(clip[start:end])
        audio_parts.append(waveform[...,round(start*sample_rate/fps):round(end*sample_rate/fps)])
        if i+1 == len(clips):
            continue
        # Interior weights include the successor's first frame immediately.
        # 0..1 endpoints made the first blend frame a duplicate of the previous
        # tail and exposed successor frame 1 first (an apparent +1f offset).
        weights=(torch.arange(overlap,device=clip.device,dtype=clip.dtype)+1)/(overlap+1)
        if mode=='smoothstep':weights=weights.square()*(3-2*weights)
        weights=weights.reshape((-1,)+(1,)*(clip.ndim-1))
        parts.append(clip[-overlap:]*(1-weights)+clips[i+1][:overlap]*weights)
        tail=waveform[...,round((len(clip)-overlap)*sample_rate/fps):expected]
        head=waveforms[i+1][...,:tail.shape[-1]]
        aw=(torch.arange(tail.shape[-1],device=tail.device,dtype=tail.dtype)+1)/(tail.shape[-1]+1)
        if mode=='smoothstep':aw=aw.square()*(3-2*aw)
        audio_parts.append(tail*(1-aw)+head*aw)
    video=torch.cat(parts);audio=torch.cat(audio_parts,dim=-1)
    # Rational frame/sample rounding can differ by a sample across many joins.
    wanted=round(len(video)*sample_rate/fps)
    if audio.shape[-1]<wanted:
        audio=torch.nn.functional.pad(audio,(0,wanted-audio.shape[-1]))
    return video,audio[...,:wanted]
