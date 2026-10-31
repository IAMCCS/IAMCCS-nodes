# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""Three editorial AudioBoard tracks, native H3 routing and prompt guidance."""
import json


def prompt_context(config, task):
    if not isinstance(config, dict) or not config.get("enabled"):
        return ""
    transcript = str(config.get("transcript", "") or "").strip()
    if config.get("lipsync") and not transcript:
        raise ValueError("MULTITRACKS lip sync: enter the exact voice transcript before requesting the prompt.")
    reference = str(task) == "ref2va"
    rows = config.get("tracks") if isinstance(config.get("tracks"), list) else []
    count = len(rows) or 3
    if reference and count > 3:
        raise ValueError("H3 supports at most three audio reference buses.")
    labels = "; ".join(f"Track {i} = <Audio {i}>" for i in range(1, count + 1)) if reference else (
        "The supplied editorial tracks are mixed into one locked custom audio guide. "
        "Do not invent <Audio N> reference tokens for FL2VA/T2VA.")
    return (
        "MULTITRACKS AUTHORING CONTRACT. Preserve the user's stated mapping of each editorial track to "
        "footsteps, beat, music, speech or other sound. Do not assume a track's role from its number. "
        f"{labels} Explicit source roles and timing: {json.dumps(rows, ensure_ascii=False)}. "
        "Use only supplied Picture N references; attach each requested character action to its stated sound source. "
        "Describe contact events in action fields and their corresponding impacts in soundscape; describe beat-led "
        "movement chronologically. Put requested score in non_diegetic_music. "
        "For REF2VA use the full six-section grammar, define each connected <Audio N> and distinguish exact-source "
        "reuse intent from timbre/rhythm inspiration. For FL2VA/T2VA use the integrated audiovisual grammar. "
        "Treat synchronization as requested behavior, not a guaranteed measured result. "
        "Any supplied dialogue must be copied verbatim inside <d>[Language] ...</d>, with stable speaker labels "
        "and visible mouth articulation only for on-screen speech. Off-screen narration retains its off-screen role and does not animate a listener mouth. Never invent a transcript, claim to have listened to an unprovided "
        "audio file, or treat filenames as speech. "
        + ("EXACT USER TRANSCRIPT: " + transcript if transcript else "No transcript was provided; author no new dialogue.")
    )


class IAMCCS_H3MultitrackAudioRouter:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"cine_linx": ("IAMCCS_SUPERNODE_LINX",)},
                "optional": {"segment_index": ("INT", {"default": 0, "min": 0}), "reference_track_count": ("INT", {"default": 3, "min": 1, "max": 3})}}

    RETURN_TYPES = ("IAMCCS_SUPERNODE_LINX", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = ("cine_linx", "track_1", "track_2", "track_3", "master_audio", "report")
    FUNCTION = "route"
    CATEGORY = "IAMCCS/MiniMax H3/Audio"

    def route(self, cine_linx, segment_index=0, reference_track_count=3):
        from .iamccs_minimax_h3_atomic_backend import _resolve_shotplan
        from .iamccs_minimax_h3_audio_timeline import _audio_contract, _segment_audible, mix_audio_timeline
        plan = _resolve_shotplan(cine_linx)
        ref = str(plan.get("task_mode", "")).startswith("ref2")
        family = "ref2va" if ref else "fl2va"
        if not ref and plan.get("task_mode") not in {"t2va", "i2va", "fl2va"}:
            raise ValueError("MULTITRACKS: select native T2VA, I2VA, FL2VA or REF2VA in Settings PRO.")
        segments, contract = _audio_contract(cine_linx)
        tracks = []
        count = max(1, min(3, int(reference_track_count))) if ref else 3
        for index in range(count):
            selected = [row for row in segments if int(row.get("track", 0)) == index and not row.get("placeholder")]
            if not selected:
                raise ValueError(f"MULTITRACKS: add real audio to AudioBoard Track {index + 1}.")
            selected = [{**row, "mute": not _segment_audible(row, contract)} for row in selected]
            # Keep tensor resources shared; isolate only the edited metadata and plan.
            branch = dict(cine_linx)
            branch["resources"] = dict(cine_linx.get("resources", {}))
            branch["resources"]["iamccs_minimax_h3_shotplan"] = {**plan, "audio_mode": "h3_custom_audio_drive"}
            old_tracks = branch["resources"].get("cine_audio_tracks", {})
            branch["resources"]["cine_audio_tracks"] = {**old_tracks, "shotboard_segments": selected, "segments": selected}
            _, _, audio, _ = mix_audio_timeline(branch, int(segment_index), [],
                target_sample_rate=32000, mapping_policy="absolute_timeline",
                coverage_policy="allow_silence", headroom="none")
            if ref:
                # Audio references do not need the generated video's alignment padding.
                seconds = float(plan.get("requested_duration_seconds", audio["waveform"].shape[-1] / 32000))
                audio = {**audio, "waveform": audio["waveform"][..., :max(1, round(seconds * 32000))].clone()}
            tracks.append(audio)
        out = dict(cine_linx)
        out["resources"] = dict(cine_linx.get("resources", {}))
        mode = "h3_ref2va_audio" if ref else "h3_custom_audio_drive"
        out["resources"]["iamccs_minimax_h3_shotplan"] = {**plan, "audio_mode": mode}
        # The existing mixer owns gain, pan, fades, trims, placement and master headroom.
        mix_branch = dict(cine_linx)
        mix_branch["resources"] = {**cine_linx.get("resources", {}),
            "iamccs_minimax_h3_shotplan": {**plan, "audio_mode": "h3_custom_audio_drive"}}
        _, master, _, _ = mix_audio_timeline(mix_branch, int(segment_index), [],
            target_sample_rate=32000, mapping_policy="absolute_timeline",
            coverage_policy="allow_silence", headroom="peak_normalize")
        out["resources"]["iamccs_minimax_h3_custom_audio"] = master
        out["resources"]["iamccs_h3_multitracks"] = {"family": family, "tracks": tracks,
            "audio_labels": [f"<Audio {i}>" for i in range(1, count + 1)] if ref else [],
            "video_reference_soundtrack": "disabled; explicit tracks own the sound references"}
        report = json.dumps({"family": family, "tracks": count, "audio_mode": mode,
                             "routing": "three references" if ref else "one locked master from three editorial tracks"})
        return out, *(tracks + [None] * (3 - len(tracks))), master, report
