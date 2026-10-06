"""Guide transport for Fast H3 refinement; no model or VAE dependencies."""

def conditioning_guides(conditioning):
    guides, seen = [], set()
    for _, metadata in conditioning:
        for guide in metadata.get("minimax_keyframes", []):
            key = (int(guide["resolved_frame_index"]), id(guide.get("latent")),
                   id(guide.get("audio_latent")))
            if key not in seen:
                guides.append(guide)
                seen.add(key)
    return guides


def pin_still_guides(video, video_mask, guides):
    """Pin image guides on H3's containing temporal token; leave clip guides soft."""
    import torch
    output = video.clone()
    mask = torch.ones_like(video) if video_mask is None else torch.broadcast_to(video_mask, video.shape).clone()
    occupied = set()
    for guide in guides:
        z = guide.get("latent")
        if z is None or z.shape[2] != 1:
            continue
        frame = int(guide["resolved_frame_index"])
        group, rem = divmod(frame, 17)
        index = group * 5 + (0 if rem == 0 else 1 + (rem - 1) // 4)
        if frame < 0 or index >= video.shape[2]:
            raise ValueError("Stage-2 Masked guide is outside the video")
        if index in occupied or torch.any(mask[:, :, index:index + 1] == 0):
            raise ValueError("Stage-2 Masked guides collide with another guide or protected video token")
        target = output[:, :, index:index + 1]
        if z.shape != target.shape:
            raise ValueError("Stage-2 guide does not match the delivery latent grid")
        target.copy_(z.to(device=video.device, dtype=video.dtype))
        mask[:, :, index:index + 1] = 0
        occupied.add(index)
    return output, mask, len(occupied)
