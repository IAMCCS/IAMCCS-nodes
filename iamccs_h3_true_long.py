# SPDX-License-Identifier: GPL-3.0-or-later
"""True Long delivery boundaries shared by rendered video and native AV state."""

from .iamccs_h3_continuous_av.latent_math import phase_aligned_extended_context_slice


def terminal_boundary(images, handover, video_steps, context_frames, head_frames, final=False):
    total = int(images.shape[0])
    end = total - 1
    hold_start = int(handover.get("freeze_start_frame", -1))
    if handover.get("freeze_detected") and head_frames < hold_start < total:
        end = hold_start
    # Catch short repeated holds below the general eight-frame detector window.
    tail = images[max(head_frames, end - 12):end + 1, ::4, ::4, :3].float()
    if len(tail) >= 3:
        delta = (tail[1:] - tail[:-1]).abs()
        means = delta.mean(dim=(1, 2, 3))
        active = (delta.amax(dim=-1) > 0.01).float().mean(dim=(1, 2))
        still = ((means <= 0.002) & (active <= 0.01)).tolist()
        count = 0
        for value in reversed(still):
            if not value:
                break
            count += 1
        # The final delivery may contain one duplicated terminal frame even
        # when the normal freeze detector correctly stays silent.  Removing
        # that single exact/near-exact duplicate is safe only on the final
        # interval: intermediate boundaries retain the stricter two-frame
        # threshold because their endpoint also authors the next PSkf state.
        minimum_hold = 1 if final else 2
        if count >= minimum_hold:
            end -= count
    visual_end = end
    if not final:
        window = phase_aligned_extended_context_slice(video_steps, int(context_frames), ideal_last_frame=end)
        end = int(window["source_end_frame"]) - 1
    if end < head_frames:
        raise ValueError("True Long: no visible frame remains before the accepted AV boundary")
    result = dict(handover)
    result.update(available=True, frame_count=total, handover_end_frame=end,
                  visible_end_frame=end, landing_tail_frames=total - end - 1,
                  phase_aligned_target_end_frame=end, phase_aware_target_end_frame=end,
                  detector_mode="iamccs_visible_terminal", no_lock_fallback_applied=False,
                  boundary_policy="true_long_shared_av_end", boundary_revision=3,
                  alignment_trim_frames=visual_end - end,
                  short_hold_trim_frames=total - 1 - visual_end)
    return result
