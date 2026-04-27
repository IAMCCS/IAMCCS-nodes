from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import torch


_log = logging.getLogger("IAMCCS.LTX2.CinematicFLF")


def _json_dumps(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return int(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _fix_ltx_frames(frames: int, mode: str) -> int:
    frames = max(1, int(frames))
    mode = str(mode or "up")
    if mode == "none":
        return frames
    rem = (frames - 1) % 8
    if rem == 0:
        return frames
    down = max(1, frames - rem)
    up = frames + (8 - rem)
    if mode == "down":
        return down
    if mode == "nearest":
        return up if (up - frames) <= (frames - down) else down
    return up


def _audio_duration_seconds(audio: Any) -> Optional[float]:
    if audio is None:
        return None
    current = audio
    for _ in range(6):
        if current is None:
            return None
        if isinstance(current, dict):
            if "waveform" in current and "sample_rate" in current:
                waveform = current.get("waveform")
                sample_rate = _safe_int(current.get("sample_rate"), 0)
                if sample_rate <= 0 or not torch.is_tensor(waveform):
                    return None
                return float(waveform.shape[-1]) / float(sample_rate)
            if "audio" in current:
                current = current.get("audio")
                continue
            if len(current) == 1:
                current = next(iter(current.values()))
                continue
        if isinstance(current, (list, tuple)) and current:
            current = current[0]
            continue
        break
    return None


def _parse_numeric_list(text: str) -> List[float]:
    if not text:
        return []
    values: List[float] = []
    for part in re.split(r"[\n,;|]+", str(text)):
        token = part.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except Exception:
            continue
    return values


def _parse_strengths(text: str, count: int, default: float) -> List[float]:
    raw = [_clamp(v, 0.0, 1.0) for v in _parse_numeric_list(text)]
    if not raw:
        raw = [float(default)]
    while len(raw) < int(count):
        raw.append(raw[-1])
    return raw[: int(count)]


def _split_custom_line(line: str) -> List[str]:
    if "|" in line:
        return [p.strip() for p in line.split("|")]
    if "," in line:
        return [p.strip() for p in line.split(",")]
    return [line.strip()]


def _position_to_frame(value: float, mode: str, total_frames: int, fps: float) -> int:
    total_frames = max(1, int(total_frames))
    fps = max(0.001, float(fps))
    mode = str(mode or "ratio")
    if value < 0:
        return max(0, min(total_frames - 1, total_frames + int(round(value))))
    if mode == "seconds":
        frame = int(round(float(value) * fps))
    elif mode == "frames":
        frame = int(round(float(value)))
    else:
        frame = int(round(_clamp(float(value), 0.0, 1.0) * float(total_frames - 1)))
    return max(0, min(total_frames - 1, frame))


def _frame_to_ratio(frame: int, total_frames: int) -> float:
    total_frames = max(1, int(total_frames))
    if total_frames <= 1:
        return 0.0
    return _clamp(float(frame) / float(total_frames - 1), 0.0, 1.0)


def _preset_positions(preset: str) -> Tuple[List[float], List[int], List[float], List[str]]:
    preset = str(preset or "shot_reverse_shot")
    if preset == "room_coverage":
        return (
            [0.0, 0.25, 0.50, 0.75, -1.0],
            [1, 2, 3, 2, 1],
            [0.92, 0.72, 0.72, 0.70, 0.88],
            ["establish", "character_a", "wide_or_room", "character_b", "resolve"],
        )
    if preset == "two_reference_dialogue":
        return (
            [0.0, 0.22, 0.50, 0.78, -1.0],
            [1, 2, 1, 2, 1],
            [0.94, 0.78, 0.70, 0.78, 0.88],
            ["speaker_a_open", "speaker_b_reply", "speaker_a_react", "speaker_b_close", "speaker_a_tail"],
        )
    if preset == "beat_markers":
        return ([0.0, 0.33, 0.66, -1.0], [1, 2, 3, 4], [0.90, 0.72, 0.72, 0.86], ["beat_1", "beat_2", "beat_3", "beat_4"])
    return (
        [0.0, 0.30, 0.58, -1.0],
        [1, 2, 1, 2],
        [0.95, 0.78, 0.72, 0.90],
        ["field_open", "reverse", "field_reaction", "reverse_tail"],
    )


class IAMCCS_LTX2_CinematicShotPlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "duration_source": (["manual", "audio", "max_manual_audio"], {"default": "manual"}),
                "duration_seconds": ("FLOAT", {"default": 8.0, "min": 0.01, "max": 36000.0, "step": 0.01}),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.001, "max": 240.0, "step": 0.01}),
                "total_frames_override": ("INT", {"default": 0, "min": 0, "max": 10000000, "step": 1}),
                "ltx_round_mode": (["up", "nearest", "down", "none"], {"default": "up"}),
                "preset": (["shot_reverse_shot", "room_coverage", "two_reference_dialogue", "beat_markers", "custom"], {"default": "shot_reverse_shot"}),
                "reference_count": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
                "position_mode": (["ratio", "seconds", "frames"], {"default": "ratio"}),
                "positions": ("STRING", {"default": "0\n0.30\n0.58\n-1", "multiline": True}),
                "strengths": ("STRING", {"default": "0.95\n0.78\n0.72\n0.90", "multiline": True}),
                "min_spacing_frames": ("INT", {"default": 8, "min": 0, "max": 100000, "step": 1}),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = (
        "STRING",
        "INT",
        "FLOAT",
        "INT", "INT", "INT", "INT", "INT", "INT", "INT", "INT",
        "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT",
        "STRING",
    )
    RETURN_NAMES = (
        "plan_json",
        "total_frames",
        "total_seconds",
        "frame_1", "frame_2", "frame_3", "frame_4", "frame_5", "frame_6", "frame_7", "frame_8",
        "strength_1", "strength_2", "strength_3", "strength_4", "strength_5", "strength_6", "strength_7", "strength_8",
        "report",
    )
    FUNCTION = "plan"
    CATEGORY = "IAMCCS/LTX-2/Cinematic"

    def _build_entries_from_custom(
        self,
        positions: str,
        strengths: str,
        position_mode: str,
        total_frames: int,
        fps: float,
        reference_count: int,
    ) -> List[Dict[str, Any]]:
        raw_lines = [line.strip() for line in str(positions or "").splitlines() if line.strip()]
        simple_positions: List[float] = []
        detailed_entries: List[Dict[str, Any]] = []
        for idx, line in enumerate(raw_lines):
            parts = _split_custom_line(line)
            if not parts:
                continue
            pos_token = parts[0].strip().lower()
            if pos_token in {"end", "last", "tail"}:
                value = -1.0
            else:
                try:
                    value = float(pos_token)
                except Exception:
                    continue
            if len(parts) == 1:
                simple_positions.append(value)
                continue
            strength = _safe_float(parts[1], 0.8) if len(parts) > 1 else 0.8
            ref_index = _safe_int(parts[2], (idx % reference_count) + 1) if len(parts) > 2 else (idx % reference_count) + 1
            label = parts[3] if len(parts) > 3 else f"custom_{idx + 1}"
            frame = _position_to_frame(value, position_mode, total_frames, fps)
            detailed_entries.append({
                "frame": int(frame),
                "position": _frame_to_ratio(frame, total_frames),
                "reference_index": max(1, min(int(reference_count), int(ref_index))),
                "strength": _clamp(strength, 0.0, 1.0),
                "label": label,
            })

        if detailed_entries:
            return detailed_entries

        if not simple_positions:
            simple_positions = [0.0, 0.30, 0.58, -1.0]
        parsed_strengths = _parse_strengths(strengths, len(simple_positions), 0.8)
        entries = []
        for idx, pos in enumerate(simple_positions):
            frame = _position_to_frame(pos, position_mode, total_frames, fps)
            entries.append({
                "frame": int(frame),
                "position": _frame_to_ratio(frame, total_frames),
                "reference_index": (idx % max(1, int(reference_count))) + 1,
                "strength": float(parsed_strengths[idx]),
                "label": f"custom_{idx + 1}",
            })
        return entries

    def _filter_spacing(self, entries: List[Dict[str, Any]], min_spacing_frames: int, total_frames: int) -> List[Dict[str, Any]]:
        min_spacing = max(0, int(min_spacing_frames))
        ordered = sorted(entries, key=lambda item: int(item.get("frame", 0)))
        if min_spacing <= 0 or len(ordered) <= 2:
            return ordered
        kept: List[Dict[str, Any]] = []
        last_frame: Optional[int] = None
        for idx, entry in enumerate(ordered):
            frame = int(entry.get("frame", 0))
            is_tail = frame >= int(total_frames) - 1
            if last_frame is None or frame - last_frame >= min_spacing or is_tail:
                kept.append(entry)
                last_frame = frame
        return kept

    def plan(
        self,
        duration_source,
        duration_seconds,
        fps,
        total_frames_override,
        ltx_round_mode,
        preset,
        reference_count,
        position_mode,
        positions,
        strengths,
        min_spacing_frames,
        audio=None,
    ):
        fps = max(0.001, float(fps))
        manual_seconds = max(0.01, float(duration_seconds))
        audio_seconds = _audio_duration_seconds(audio)
        if duration_source == "audio" and audio_seconds is not None:
            total_seconds = float(audio_seconds)
            duration_note = "audio"
        elif duration_source == "max_manual_audio" and audio_seconds is not None:
            total_seconds = max(manual_seconds, float(audio_seconds))
            duration_note = "max_manual_audio"
        else:
            total_seconds = manual_seconds
            duration_note = "manual"

        if int(total_frames_override) > 0:
            total_frames = max(1, int(total_frames_override))
            frames_note = "override"
        else:
            total_frames = _fix_ltx_frames(int(round(total_seconds * fps)), str(ltx_round_mode))
            frames_note = f"{ltx_round_mode}"

        reference_count = max(1, min(8, int(reference_count)))
        if preset == "custom":
            entries = self._build_entries_from_custom(
                positions, strengths, position_mode, total_frames, fps, reference_count
            )
            preset_note = "custom"
        else:
            preset_positions, preset_refs, preset_strengths, preset_labels = _preset_positions(str(preset))
            entries = []
            for idx, pos in enumerate(preset_positions):
                frame = _position_to_frame(pos, "ratio", total_frames, fps)
                ref_index = ((int(preset_refs[idx]) - 1) % reference_count) + 1
                entries.append({
                    "frame": int(frame),
                    "position": _frame_to_ratio(frame, total_frames),
                    "reference_index": int(ref_index),
                    "strength": float(preset_strengths[idx]),
                    "label": str(preset_labels[idx]),
                })
            preset_note = str(preset)

        entries = self._filter_spacing(entries, min_spacing_frames, total_frames)
        for order, entry in enumerate(entries):
            entry["order"] = int(order)
            entry["frame"] = max(0, min(int(total_frames) - 1, int(entry.get("frame", 0))))
            entry["position"] = _frame_to_ratio(int(entry["frame"]), total_frames)
            entry["reference_index"] = max(1, min(reference_count, int(entry.get("reference_index", 1))))
            entry["strength"] = _clamp(float(entry.get("strength", 0.8)), 0.0, 1.0)

        frames = [int(entry.get("frame", 0)) for entry in entries][:8]
        strength_values = [float(entry.get("strength", 0.0)) for entry in entries][:8]
        while len(frames) < 8:
            frames.append(0)
        while len(strength_values) < 8:
            strength_values.append(0.0)

        plan = {
            "schema": "iamccs_ltx2_cinematic_flf_plan_v1",
            "preset": preset_note,
            "duration_source": duration_note,
            "fps": float(fps),
            "total_seconds": float(total_frames) / float(fps),
            "manual_seconds": float(manual_seconds),
            "audio_seconds": audio_seconds,
            "total_frames": int(total_frames),
            "ltx_round_mode": frames_note,
            "reference_count": int(reference_count),
            "entries": entries,
        }
        report = (
            f"cinematic_plan preset={preset_note} refs={reference_count} "
            f"frames={total_frames} @ {fps:.3f}fps entries={len(entries)} duration={duration_note}"
        )
        _log.info("[CinematicShotPlanner] %s", report)
        return (
            _json_dumps(plan),
            int(total_frames),
            float(plan["total_seconds"]),
            *frames,
            *strength_values,
            report,
        )


class IAMCCS_LTX2_CinematicRefLatentControl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE", {}),
                "latent": ("LATENT", {}),
                "plan_json": ("STRING", {"default": "{}", "multiline": True}),
                "default_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "lock_slots": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "soft_transition_slots": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1}),
                "reference_fallback": (["cycle", "clamp", "skip"], {"default": "cycle"}),
            },
            "optional": {
                "reference_1": ("IMAGE",),
                "reference_2": ("IMAGE",),
                "reference_3": ("IMAGE",),
                "reference_4": ("IMAGE",),
                "reference_5": ("IMAGE",),
                "reference_6": ("IMAGE",),
                "reference_7": ("IMAGE",),
                "reference_8": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "STRING")
    RETURN_NAMES = ("latent", "applied_count", "report")
    FUNCTION = "apply"
    CATEGORY = "IAMCCS/LTX-2/Cinematic"

    @staticmethod
    def _common_upscale_nhwc_to(images: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        try:
            import comfy.utils  # type: ignore
        except Exception as e:
            raise ImportError("comfy.utils is required for this node") from e

        if int(images.shape[1]) == int(target_h) and int(images.shape[2]) == int(target_w):
            return images
        return comfy.utils.common_upscale(
            images.movedim(-1, 1),
            int(target_w),
            int(target_h),
            "bilinear",
            "center",
        ).movedim(1, -1)

    @classmethod
    def _encode_image(cls, vae, image: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        if int(image.shape[0]) > 1:
            image = image[:1]
        pixels = cls._common_upscale_nhwc_to(image, int(target_h), int(target_w))
        return vae.encode(pixels[:, :, :, :3])

    @staticmethod
    def _match_latent_batch(base_samples: torch.Tensor, other_samples: torch.Tensor) -> torch.Tensor:
        bb = int(base_samples.shape[0])
        ob = int(other_samples.shape[0])
        if bb == ob:
            return other_samples
        if ob == 1 and bb > 1:
            reps = [bb] + [1] * (other_samples.dim() - 1)
            return other_samples.repeat(*reps)
        return other_samples[:bb]

    @staticmethod
    def _ensure_noise_mask(latent: Dict[str, Any], samples: torch.Tensor) -> torch.Tensor:
        batch, _, latent_frames, _, _ = samples.shape
        if "noise_mask" in latent and latent["noise_mask"] is not None:
            noise_mask = latent["noise_mask"].clone()
            if int(noise_mask.shape[0]) == 1 and int(batch) > 1:
                noise_mask = noise_mask.repeat(int(batch), 1, 1, 1, 1)
            elif int(noise_mask.shape[0]) != int(batch):
                noise_mask = noise_mask[: int(batch)]
            return noise_mask
        return torch.ones((batch, 1, latent_frames, 1, 1), dtype=torch.float32, device=samples.device)

    @staticmethod
    def _load_plan(plan_json: str) -> Dict[str, Any]:
        try:
            plan = json.loads(str(plan_json or "{}"))
            if isinstance(plan, dict):
                return plan
        except Exception:
            pass
        return {"entries": []}

    @staticmethod
    def _select_reference(refs: List[Optional[torch.Tensor]], ref_index: int, fallback: str) -> Optional[torch.Tensor]:
        available = [(idx + 1, image) for idx, image in enumerate(refs) if image is not None]
        if not available:
            return None
        ref_index = int(ref_index)
        if 1 <= ref_index <= len(refs) and refs[ref_index - 1] is not None:
            return refs[ref_index - 1]
        if fallback == "skip":
            return None
        if fallback == "clamp":
            target = max(1, min(ref_index, len(available)))
            return available[target - 1][1]
        return available[(max(1, ref_index) - 1) % len(available)][1]

    @staticmethod
    def _frame_to_latent_index(frame: int, total_pixel_frames: int, time_scale_factor: int, latent_frames: int, slots: int) -> int:
        if int(frame) < 0:
            frame = int(total_pixel_frames) + int(frame)
        frame = max(0, min(int(total_pixel_frames) - 1, int(frame)))
        latent_idx = int(frame) // max(1, int(time_scale_factor))
        if frame >= int(total_pixel_frames) - max(1, int(time_scale_factor)):
            latent_idx = int(latent_frames) - int(slots)
        return max(0, min(int(latent_frames) - int(slots), int(latent_idx)))

    @staticmethod
    def _smoothstep(x: torch.Tensor) -> torch.Tensor:
        return x * x * (3.0 - 2.0 * x)

    def _apply_soft_transition(
        self,
        samples: torch.Tensor,
        noise_mask: torch.Tensor,
        ref_slot: torch.Tensor,
        start_idx: int,
        strength: float,
        soft_transition_slots: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        soft_slots = max(0, min(int(soft_transition_slots), int(start_idx)))
        if soft_slots <= 0:
            return samples, noise_mask, 0
        trans_start = int(start_idx) - soft_slots
        weights = torch.linspace(
            0.0,
            1.0,
            steps=soft_slots + 2,
            device=samples.device,
            dtype=torch.float32,
        )[1:-1]
        for offset in range(soft_slots):
            alpha = float(self._smoothstep(weights[offset : offset + 1]).item())
            t = trans_start + offset
            samples[:, :, t : t + 1] = (
                (1.0 - alpha) * samples[:, :, t : t + 1] + alpha * ref_slot
            )
            target_mask = 1.0 - (float(strength) * alpha * 0.5)
            cur = noise_mask[:, :, t : t + 1]
            noise_mask[:, :, t : t + 1] = torch.minimum(cur, torch.full_like(cur, float(target_mask)))
        return samples, noise_mask, soft_slots

    def apply(
        self,
        vae,
        latent,
        plan_json,
        default_strength,
        lock_slots,
        soft_transition_slots,
        reference_fallback,
        reference_1=None,
        reference_2=None,
        reference_3=None,
        reference_4=None,
        reference_5=None,
        reference_6=None,
        reference_7=None,
        reference_8=None,
    ):
        samples_in = latent.get("samples")
        if samples_in is None:
            raise ValueError("LATENT input is missing 'samples'")

        samples = samples_in.clone()
        batch, _, latent_frames, latent_height, latent_width = samples.shape
        scale_factors = getattr(vae, "downscale_index_formula", (8, 8, 8))
        time_scale_factor = int(scale_factors[0])
        height_scale_factor = int(scale_factors[1])
        width_scale_factor = int(scale_factors[2])
        target_width = int(latent_width) * width_scale_factor
        target_height = int(latent_height) * height_scale_factor
        total_pixel_frames = (int(latent_frames) - 1) * max(1, time_scale_factor) + 1
        lock_slots = max(1, min(8, int(lock_slots), int(latent_frames)))
        default_strength = _clamp(default_strength, 0.0, 1.0)

        plan = self._load_plan(plan_json)
        entries = plan.get("entries", [])
        if not isinstance(entries, list):
            entries = []
        refs = [reference_1, reference_2, reference_3, reference_4, reference_5, reference_6, reference_7, reference_8]
        noise_mask = self._ensure_noise_mask(latent, samples)

        encoded_cache: Dict[int, torch.Tensor] = {}
        applied = 0
        ops: List[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            ref_index = max(1, int(entry.get("reference_index", 1)))
            image = self._select_reference(refs, ref_index, str(reference_fallback))
            if image is None:
                ops.append(f"skip(ref={ref_index})")
                continue
            if ref_index not in encoded_cache:
                ref_latent = self._encode_image(vae, image, target_height, target_width)
                ref_latent = self._match_latent_batch(samples, ref_latent)
                encoded_cache[ref_index] = ref_latent.to(device=samples.device, dtype=samples.dtype)
            ref_latent = encoded_cache[ref_index]
            slots = min(lock_slots, int(ref_latent.shape[2]), int(latent_frames))
            if slots <= 0:
                continue
            frame = int(entry.get("frame", 0))
            strength = _clamp(entry.get("strength", default_strength), 0.0, 1.0)
            if strength <= 0.0:
                continue
            latent_idx = self._frame_to_latent_index(frame, total_pixel_frames, time_scale_factor, latent_frames, slots)
            ref_slice = ref_latent[:, :, :slots]
            if frame >= total_pixel_frames - time_scale_factor:
                ref_slice = ref_latent[:, :, -slots:]

            samples, noise_mask, soft_used = self._apply_soft_transition(
                samples,
                noise_mask,
                ref_slice[:, :, :1],
                latent_idx,
                strength,
                int(soft_transition_slots),
            )
            samples[:, :, latent_idx : latent_idx + slots] = ref_slice
            cur = noise_mask[:, :, latent_idx : latent_idx + slots]
            noise_mask[:, :, latent_idx : latent_idx + slots] = torch.minimum(
                cur,
                torch.full_like(cur, 1.0 - float(strength)),
            )
            applied += 1
            label = str(entry.get("label", f"ref_{ref_index}"))
            ops.append(f"{label}:f{frame}->t{latent_idx}+{slots},ref={ref_index},s={strength:.2f},soft={soft_used}")

        out_latent = dict(latent)
        out_latent["samples"] = samples
        out_latent["noise_mask"] = noise_mask
        report = (
            f"cinematic_ref_latent applied={applied}/{len(entries)} "
            f"latent_frames={latent_frames} pixel_frames={total_pixel_frames} | "
            + ("; ".join(ops) if ops else "no-op")
        )
        _log.info("[CinematicRefLatentControl] %s", report)
        return (out_latent, int(applied), report)


class IAMCCS_LTX2_AudioPromptDirector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_prompt": ("STRING", {"default": "", "multiline": True}),
                "action_prompt": ("STRING", {"default": "", "multiline": True}),
                "dialogue_text": ("STRING", {"default": "", "multiline": True}),
                "voice_direction": ("STRING", {"default": "", "multiline": True}),
                "speaker_label": ("STRING", {"default": "the character"}),
                "audio_role": (["off", "dialogue_only", "dialogue_plus_action", "ambience_only"], {"default": "dialogue_plus_action"}),
                "prompt_style": (["cinematic", "literal_ltx", "compact"], {"default": "cinematic"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.001, "max": 240.0, "step": 0.01}),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("prompt", "dialogue_clause", "audio_seconds", "audio_frames", "report")
    FUNCTION = "build"
    CATEGORY = "IAMCCS/LTX-2/Cinematic"

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip())

    def build(
        self,
        scene_prompt,
        action_prompt,
        dialogue_text,
        voice_direction,
        speaker_label,
        audio_role,
        prompt_style,
        fps,
        audio=None,
    ):
        scene = self._clean(scene_prompt)
        action = self._clean(action_prompt)
        dialogue = self._clean(dialogue_text)
        voice = self._clean(voice_direction)
        speaker = self._clean(speaker_label) or "the character"
        audio_seconds = _audio_duration_seconds(audio)
        audio_frames = int(round(float(audio_seconds or 0.0) * max(0.001, float(fps))))

        dialogue_clause = ""
        if audio_role in {"dialogue_only", "dialogue_plus_action"} and dialogue:
            dialogue_clause = f'{speaker} says "{dialogue}"'
            if voice:
                dialogue_clause += f" with {voice}"
        elif audio_role == "ambience_only":
            dialogue_clause = "the provided audio is used as ambience and timing texture"

        parts: List[str] = []
        if prompt_style == "compact":
            parts = [p for p in [scene, action, dialogue_clause] if p]
        elif prompt_style == "literal_ltx":
            if scene:
                parts.append(f"Scene: {scene}")
            if action:
                parts.append(f"Action and camera direction: {action}")
            if dialogue_clause:
                parts.append(f"Audio/dialogue: {dialogue_clause}")
            if audio_role != "off":
                parts.append("Follow the selected audio for voice timing, while the visual action remains directed by the prompt.")
        else:
            if scene:
                parts.append(scene)
            if action:
                parts.append(action)
            if dialogue_clause:
                parts.append(dialogue_clause)
            if audio_role != "off":
                parts.append("The voice follows the provided audio, but blocking, camera movement, expressions, and cuts follow the written direction.")

        if not parts:
            parts.append("A cinematic shot with coherent motion and natural continuity.")
        prompt = ". ".join(part.rstrip(".") for part in parts if part).strip()
        if prompt and not prompt.endswith("."):
            prompt += "."

        report = (
            f"audio_prompt_director role={audio_role} style={prompt_style} "
            f"audio_seconds={(audio_seconds if audio_seconds is not None else 0.0):.3f} frames={audio_frames}"
        )
        return (prompt, dialogue_clause, float(audio_seconds or 0.0), int(audio_frames), report)


class IAMCCS_LTX2_CinematicPromptRelayAdapter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan_json": ("STRING", {"default": "{}", "multiline": True}),
                "global_prompt": ("STRING", {"default": "", "multiline": True}),
                "local_prompt_lines": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional. One line per PromptRelay segment, or use | separators. Supports {label}, {role}, {ref}, {start}, {end}, {frames}, {global}.",
                }),
                "role_lines": ("STRING", {
                    "default": (
                        "reference 1: field / character A\n"
                        "reference 2: reverse field / character B\n"
                        "reference 3: room / wide shot\n"
                        "reference 4: insert / beat detail"
                    ),
                    "multiline": True,
                }),
                "segment_mode": (["plan_keyframes", "even_by_entries", "single_scene"], {"default": "plan_keyframes"}),
                "include_global_in_local": (["no", "yes"], {"default": "no"}),
                "merge_short_segments": (["yes", "no"], {"default": "yes"}),
                "min_segment_frames": ("INT", {"default": 4, "min": 1, "max": 100000, "step": 1}),
                "default_local_prompt": ("STRING", {"default": "cinematic continuity, natural motion, coherent staging", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("global_prompt", "local_prompts", "segment_lengths", "segment_count", "report")
    FUNCTION = "build"
    CATEGORY = "IAMCCS/LTX-2/Cinematic"

    @staticmethod
    def _load_plan(plan_json: Any) -> Dict[str, Any]:
        if isinstance(plan_json, dict):
            return plan_json
        try:
            data = json.loads(str(plan_json or "{}"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _clean(text: Any) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip())

    @staticmethod
    def _split_prompt_lines(text: str) -> List[str]:
        raw = str(text or "").strip()
        if not raw:
            return []
        if "|" in raw:
            parts = raw.split("|")
        else:
            parts = raw.splitlines()
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _parse_roles(text: str) -> List[str]:
        roles: List[str] = []
        for line in str(text or "").splitlines():
            token = line.strip()
            if not token:
                continue
            if ":" in token:
                token = token.split(":", 1)[1].strip()
            roles.append(token)
        if not roles:
            roles = ["reference 1", "reference 2", "reference 3", "reference 4"]
        return roles

    @staticmethod
    def _valid_entries(plan: Dict[str, Any], total_frames: int) -> List[Dict[str, Any]]:
        entries = plan.get("entries", [])
        if not isinstance(entries, list):
            return []
        out: List[Dict[str, Any]] = []
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            frame = max(0, min(max(0, int(total_frames) - 1), _safe_int(entry.get("frame", 0), 0)))
            out.append({
                "frame": frame,
                "reference_index": max(1, _safe_int(entry.get("reference_index", idx + 1), idx + 1)),
                "label": str(entry.get("label", f"beat_{idx + 1}")),
                "strength": _clamp(entry.get("strength", 0.8), 0.0, 1.0),
            })
        out.sort(key=lambda item: (int(item.get("frame", 0)), int(item.get("reference_index", 1))))
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for entry in out:
            key = int(entry["frame"])
            if key in seen:
                continue
            deduped.append(entry)
            seen.add(key)
        return deduped

    @staticmethod
    def _distributed_lengths(total_frames: int, count: int) -> List[int]:
        total_frames = max(1, int(total_frames))
        count = max(1, min(int(count), int(total_frames)))
        base = total_frames // count
        remainder = total_frames % count
        return [base + (1 if idx < remainder else 0) for idx in range(count)]

    def _segments_from_entries(
        self,
        entries: List[Dict[str, Any]],
        total_frames: int,
        segment_mode: str,
        manual_count: int,
    ) -> List[Dict[str, Any]]:
        total_frames = max(1, int(total_frames))
        if segment_mode == "single_scene":
            entry = entries[0] if entries else {}
            return [{"start": 0, "end": total_frames, "frames": total_frames, "entry": entry, "kind": "single"}]

        if segment_mode == "even_by_entries":
            count = max(len(entries), int(manual_count), 1)
            lengths = self._distributed_lengths(total_frames, count)
            segments: List[Dict[str, Any]] = []
            cursor = 0
            for idx, length in enumerate(lengths):
                entry = entries[min(idx, len(entries) - 1)] if entries else {}
                end = cursor + int(length)
                segments.append({"start": cursor, "end": end, "frames": int(length), "entry": entry, "kind": "even"})
                cursor = end
            return segments

        if not entries:
            return [{"start": 0, "end": total_frames, "frames": total_frames, "entry": {}, "kind": "empty"}]

        segments = []
        cursor = 0
        for idx, entry in enumerate(entries):
            start = int(entry.get("frame", 0))
            if start > cursor:
                segments.append({
                    "start": cursor,
                    "end": start,
                    "frames": start - cursor,
                    "entry": {},
                    "kind": "prelude",
                })
            next_start = int(entries[idx + 1].get("frame", total_frames)) if idx + 1 < len(entries) else total_frames
            end = max(start + 1, min(total_frames, next_start))
            if end > start:
                segments.append({
                    "start": start,
                    "end": end,
                    "frames": end - start,
                    "entry": entry,
                    "kind": "plan",
                })
            cursor = max(cursor, end)
        if cursor < total_frames:
            segments.append({"start": cursor, "end": total_frames, "frames": total_frames - cursor, "entry": entries[-1], "kind": "tail"})
        return [seg for seg in segments if int(seg.get("frames", 0)) > 0]

    @staticmethod
    def _merge_short(segments: List[Dict[str, Any]], min_segment_frames: int, enabled: bool) -> List[Dict[str, Any]]:
        if not enabled:
            return segments
        min_frames = max(1, int(min_segment_frames))
        merged: List[Dict[str, Any]] = []
        for seg in segments:
            seg = dict(seg)
            if int(seg.get("frames", 0)) < min_frames and merged:
                prev = merged[-1]
                prev["end"] = seg.get("end", prev.get("end", 0))
                prev["frames"] = int(prev.get("frames", 0)) + int(seg.get("frames", 0))
                if not prev.get("entry") and seg.get("entry"):
                    prev["entry"] = seg.get("entry")
                prev["kind"] = f"{prev.get('kind', 'segment')}+{seg.get('kind', 'short')}"
            else:
                merged.append(seg)
        return merged

    @staticmethod
    def _format_template(template: str, mapping: Dict[str, Any]) -> str:
        try:
            return str(template).format(**mapping)
        except Exception:
            return str(template)

    def _auto_prompt(
        self,
        segment: Dict[str, Any],
        role: str,
        global_prompt: str,
        include_global: bool,
        default_local_prompt: str,
    ) -> str:
        entry = segment.get("entry") if isinstance(segment.get("entry"), dict) else {}
        label = self._clean(str(entry.get("label", "")).replace("_", " "))
        parts: List[str] = []
        if include_global and global_prompt:
            parts.append(global_prompt)
        if role:
            parts.append(role)
        if label:
            parts.append(label)
        if segment.get("kind") == "prelude":
            parts.append("establish the scene before the next beat")
        if not parts:
            parts.append(self._clean(default_local_prompt) or "cinematic continuity")
        return ". ".join(part.rstrip(".") for part in parts if part).strip()

    def build(
        self,
        plan_json,
        global_prompt,
        local_prompt_lines,
        role_lines,
        segment_mode,
        include_global_in_local,
        merge_short_segments,
        min_segment_frames,
        default_local_prompt,
    ):
        plan = self._load_plan(plan_json)
        total_frames = max(1, _safe_int(plan.get("total_frames", 0), 0))
        if total_frames <= 1:
            entries_for_total = plan.get("entries", [])
            if isinstance(entries_for_total, list) and entries_for_total:
                total_frames = max(1, max(_safe_int(e.get("frame", 0), 0) for e in entries_for_total if isinstance(e, dict)) + 1)

        global_out = self._clean(global_prompt)
        if not global_out:
            global_out = "A cinematic shot with coherent motion and natural continuity."

        manual_lines = self._split_prompt_lines(local_prompt_lines)
        roles = self._parse_roles(role_lines)
        entries = self._valid_entries(plan, total_frames)
        segments = self._segments_from_entries(entries, total_frames, str(segment_mode), len(manual_lines))
        segments = self._merge_short(segments, int(min_segment_frames), str(merge_short_segments) == "yes")
        if not segments:
            segments = [{"start": 0, "end": total_frames, "frames": total_frames, "entry": {}, "kind": "empty"}]

        include_global = str(include_global_in_local) == "yes"
        prompts: List[str] = []
        lengths: List[int] = []
        ops: List[str] = []
        for idx, segment in enumerate(segments):
            entry = segment.get("entry") if isinstance(segment.get("entry"), dict) else {}
            ref_index = max(1, _safe_int(entry.get("reference_index", idx + 1), idx + 1))
            role = roles[min(ref_index - 1, len(roles) - 1)] if roles else f"reference {ref_index}"
            mapping = {
                "label": self._clean(str(entry.get("label", f"segment_{idx + 1}")).replace("_", " ")),
                "role": role,
                "ref": ref_index,
                "start": int(segment.get("start", 0)),
                "end": int(segment.get("end", 0)),
                "frames": int(segment.get("frames", 0)),
                "global": global_out,
            }
            if manual_lines:
                template = manual_lines[min(idx, len(manual_lines) - 1)]
                prompt = self._clean(self._format_template(template, mapping))
                if include_global and global_out and global_out not in prompt:
                    prompt = f"{global_out}. {prompt}"
            else:
                prompt = self._auto_prompt(segment, role, global_out, include_global, default_local_prompt)
            prompts.append(prompt)
            lengths.append(max(1, int(segment.get("frames", 1))))
            ops.append(f"{idx + 1}:{segment.get('kind')} f{mapping['start']}-{mapping['end']} len={mapping['frames']} ref={ref_index}")

        segment_lengths = ",".join(str(length) for length in lengths)
        local_prompts = "\n|\n".join(prompts)
        report = (
            f"prompt_relay_adapter segments={len(prompts)} total_frames={total_frames} "
            f"length_sum={sum(lengths)} mode={segment_mode} | "
            + "; ".join(ops)
        )
        return (global_out, local_prompts, segment_lengths, int(len(prompts)), report)


def _cin_clean(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _cin_slug(text: Any, default: str) -> str:
    raw = re.sub(r"[^A-Za-z0-9_-]+", "_", str(text or "").strip()).strip("_")
    return raw[:64] or default


def _cin_join_prompt_parts(*parts: Any) -> str:
    cleaned: List[str] = []
    for part in parts:
        value = _cin_clean(part)
        if not value or value.lower() in {"none", "off", "no", "custom"}:
            continue
        cleaned.append(value.rstrip("."))
    return ". ".join(cleaned).strip()


class IAMCCS_LTX2_CinematicPromptComposer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "framing": ([
                    "custom",
                    "wide establishing shot",
                    "medium shot",
                    "medium close-up",
                    "close-up",
                    "extreme close-up",
                    "over-the-shoulder",
                    "reverse angle",
                    "insert detail",
                    "two-shot",
                    "tracking shot",
                ], {"default": "medium close-up"}),
                "camera_movement": ([
                    "custom",
                    "locked-off camera",
                    "slow push-in",
                    "slow pull-back",
                    "subtle handheld",
                    "gentle pan",
                    "tracking left to right",
                    "tracking forward",
                    "static observational",
                    "match cut motion",
                ], {"default": "subtle handheld"}),
                "lens_style": ([
                    "custom",
                    "natural 35mm lens",
                    "intimate 50mm lens",
                    "compressed telephoto portrait",
                    "wide lens with room geography",
                    "shallow depth of field",
                    "deep focus",
                    "anamorphic cinematic look",
                ], {"default": "natural 35mm lens"}),
                "lighting": ([
                    "custom",
                    "soft window light",
                    "warm practical lamps",
                    "noir desk lamp",
                    "moonlight through window",
                    "rainy night reflections",
                    "overcast natural light",
                    "low-key cinematic lighting",
                    "high contrast silhouette",
                ], {"default": "low-key cinematic lighting"}),
                "subject": ("STRING", {"default": "Mara", "multiline": False}),
                "action_prompt": ("STRING", {
                    "default": "she listens, then speaks softly while holding eye contact",
                    "multiline": True,
                }),
                "emotion": ([
                    "custom",
                    "controlled hurt",
                    "calm suspicion",
                    "quiet regret",
                    "restrained fear",
                    "tender hesitation",
                    "cold determination",
                    "nervous confession",
                    "silent reaction",
                    "neutral documentary realism",
                ], {"default": "controlled hurt"}),
                "location_context": ("STRING", {
                    "default": "inside a small kitchen at night, rain on the window",
                    "multiline": True,
                }),
                "continuity_note": ("STRING", {
                    "default": "keep coherent eyelines, natural acting, stable identity, cinematic room geography",
                    "multiline": True,
                }),
            },
            "optional": {
                "extra_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("shot_prompt", "report")
    FUNCTION = "compose"
    CATEGORY = "IAMCCS/LTX-2/Cinematic/User Friendly"

    def compose(
        self,
        framing,
        camera_movement,
        lens_style,
        lighting,
        subject,
        action_prompt,
        emotion,
        location_context,
        continuity_note,
        extra_prompt="",
    ):
        subject_text = _cin_clean(subject) or "the subject"
        action = _cin_clean(action_prompt)
        if action:
            action = f"{subject_text} {action}" if subject_text.lower() not in action.lower() else action
        prompt = _cin_join_prompt_parts(
            framing,
            action,
            emotion,
            camera_movement,
            lens_style,
            lighting,
            location_context,
            continuity_note,
            extra_prompt,
        )
        if not prompt:
            prompt = f"medium close-up on {subject_text}, natural cinematic acting"
        report = f"prompt_composer subject={subject_text} framing={framing} movement={camera_movement}"
        return (prompt, report)


class IAMCCS_LTX2_CinematicShotLineBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "duration_seconds": ("FLOAT", {"default": 4.0, "min": 0.01, "max": 3600.0, "step": 0.01}),
                "cut_mode": (["hard_cut", "continuity_cut", "soft_cut", "match_cut"], {"default": "hard_cut"}),
                "primary_reference_index": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "reference_mode": (["single", "start_to_end", "dual"], {"default": "single"}),
                "secondary_reference_index": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1}),
                "audio_index": ("INT", {"default": 1, "min": 0, "max": 8, "step": 1}),
                "label": ("STRING", {"default": "campo_mara", "multiline": False}),
                "framing": ([
                    "custom",
                    "wide establishing shot",
                    "medium shot",
                    "medium close-up",
                    "close-up",
                    "extreme close-up",
                    "over-the-shoulder",
                    "reverse angle",
                    "insert detail",
                    "two-shot",
                ], {"default": "medium close-up"}),
                "subject": ("STRING", {"default": "Mara", "multiline": False}),
                "action_prompt": ("STRING", {
                    "default": "near the kitchen window, she speaks softly but firmly",
                    "multiline": True,
                }),
                "dialogue_text": ("STRING", {
                    "default": "You left the city without saying goodbye.",
                    "multiline": True,
                }),
                "voice_direction": ("STRING", {
                    "default": "controlled hurt voice",
                    "multiline": True,
                }),
                "camera_movement": ([
                    "custom",
                    "locked-off camera",
                    "slow push-in",
                    "slow pull-back",
                    "subtle handheld",
                    "gentle pan",
                    "tracking shot",
                ], {"default": "subtle handheld"}),
                "lighting": ([
                    "custom",
                    "soft window light",
                    "warm practical lamps",
                    "noir desk lamp",
                    "moonlight through window",
                    "rainy night reflections",
                    "low-key cinematic lighting",
                ], {"default": "rainy night reflections"}),
                "continuity_note": ("STRING", {
                    "default": "keep coherent eyelines and stable character identity",
                    "multiline": True,
                }),
            },
            "optional": {
                "prompt_override": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "If connected or filled, this replaces the dropdown-composed shot prompt.",
                }),
                "extra_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("shot_line", "shot_prompt", "reference_token", "report")
    FUNCTION = "build"
    CATEGORY = "IAMCCS/LTX-2/Cinematic/User Friendly"

    @staticmethod
    def _reference_token(primary: int, mode: str, secondary: int) -> str:
        primary = max(1, min(8, int(primary)))
        secondary = max(0, min(8, int(secondary)))
        if str(mode) == "start_to_end" and secondary > 0:
            return f"{primary}>{secondary}"
        if str(mode) == "dual" and secondary > 0:
            return f"{primary}+{secondary}"
        return str(primary)

    def build(
        self,
        duration_seconds,
        cut_mode,
        primary_reference_index,
        reference_mode,
        secondary_reference_index,
        audio_index,
        label,
        framing,
        subject,
        action_prompt,
        dialogue_text,
        voice_direction,
        camera_movement,
        lighting,
        continuity_note,
        prompt_override="",
        extra_prompt="",
    ):
        ref_token = self._reference_token(primary_reference_index, reference_mode, secondary_reference_index)
        safe_label = _cin_slug(label, f"shot_ref_{ref_token}")
        prompt = _cin_clean(prompt_override)
        if not prompt:
            subject_text = _cin_clean(subject) or "the subject"
            action = _cin_clean(action_prompt)
            if action and subject_text.lower() not in action.lower():
                action = f"{subject_text} {action}"
            prompt = _cin_join_prompt_parts(
                framing,
                action,
                camera_movement,
                lighting,
                continuity_note,
                extra_prompt,
            )
        line = " | ".join([
            f"{max(0.01, _safe_float(duration_seconds, 4.0)):.2f}".rstrip("0").rstrip("."),
            str(cut_mode),
            ref_token,
            str(max(0, min(8, _safe_int(audio_index, 0)))),
            safe_label,
            prompt,
            _cin_clean(dialogue_text),
            _cin_clean(voice_direction),
        ])
        report = f"shot_line label={safe_label} ref={ref_token} audio={audio_index} cut={cut_mode}"
        return (line, prompt, ref_token, report)


class IAMCCS_LTX2_CinematicV2VTimelineLineBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "duration_seconds": ("FLOAT", {"default": 4.0, "min": 0.01, "max": 3600.0, "step": 0.01}),
                "cut_mode": (["hard_cut", "continuity_cut", "soft_cut", "match_cut"], {"default": "hard_cut"}),
                "source_video_index": ("INT", {"default": 1, "min": 0, "max": 8, "step": 1}),
                "source_range_mode": (["all", "start_plus_count", "frame_range", "tail", "custom"], {"default": "start_plus_count"}),
                "source_start_frame": ("INT", {"default": 0, "min": 0, "max": 10000000, "step": 1}),
                "source_frame_count": ("INT", {"default": 96, "min": 1, "max": 10000000, "step": 1}),
                "source_end_frame": ("INT", {"default": 96, "min": 1, "max": 10000000, "step": 1}),
                "custom_source_range": ("STRING", {"default": "0+96", "multiline": False}),
                "primary_reference_index": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "reference_mode": (["single", "start_to_end", "dual"], {"default": "single"}),
                "secondary_reference_index": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1}),
                "audio_index": ("INT", {"default": 1, "min": 0, "max": 8, "step": 1}),
                "label": ("STRING", {"default": "camera_A_mara", "multiline": False}),
                "framing": ([
                    "custom",
                    "V2V close-up",
                    "V2V medium close-up",
                    "V2V reverse angle",
                    "V2V over-the-shoulder",
                    "V2V wide shot",
                    "V2V insert detail",
                    "V2V tracking shot",
                ], {"default": "V2V close-up"}),
                "subject": ("STRING", {"default": "Mara", "multiline": False}),
                "action_prompt": ("STRING", {
                    "default": "speaking, preserve the original head movement and natural eye line",
                    "multiline": True,
                }),
                "dialogue_text": ("STRING", {
                    "default": "You left the city without saying goodbye.",
                    "multiline": True,
                }),
                "voice_direction": ("STRING", {
                    "default": "controlled hurt voice",
                    "multiline": True,
                }),
                "v2v_mode": ([
                    "v2v_source_plus_reference",
                    "v2v_source_context",
                    "i2v_from_reference",
                    "two_segments_if_long",
                    "loop_if_long",
                ], {"default": "v2v_source_plus_reference"}),
                "continuity_note": ("STRING", {
                    "default": "keep source performance, improve cinematic lighting, stable identity",
                    "multiline": True,
                }),
            },
            "optional": {
                "prompt_override": ("STRING", {"default": "", "multiline": True}),
                "extra_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("timeline_line", "shot_prompt", "source_range", "reference_token", "report")
    FUNCTION = "build"
    CATEGORY = "IAMCCS/LTX-2/Cinematic/User Friendly"

    @staticmethod
    def _source_range(mode: str, start: int, count: int, end: int, custom: str) -> str:
        mode = str(mode or "all")
        start = max(0, int(start))
        count = max(1, int(count))
        end = max(start + 1, int(end))
        if mode == "start_plus_count":
            return f"{start}+{count}"
        if mode == "frame_range":
            return f"{start}-{end}"
        if mode == "tail":
            return f"tail{count}"
        if mode == "custom":
            return _cin_clean(custom) or f"{start}+{count}"
        return "all"

    def build(
        self,
        duration_seconds,
        cut_mode,
        source_video_index,
        source_range_mode,
        source_start_frame,
        source_frame_count,
        source_end_frame,
        custom_source_range,
        primary_reference_index,
        reference_mode,
        secondary_reference_index,
        audio_index,
        label,
        framing,
        subject,
        action_prompt,
        dialogue_text,
        voice_direction,
        v2v_mode,
        continuity_note,
        prompt_override="",
        extra_prompt="",
    ):
        ref_token = IAMCCS_LTX2_CinematicShotLineBuilder._reference_token(
            primary_reference_index,
            reference_mode,
            secondary_reference_index,
        )
        source_range = self._source_range(
            source_range_mode,
            _safe_int(source_start_frame, 0),
            _safe_int(source_frame_count, 96),
            _safe_int(source_end_frame, 96),
            custom_source_range,
        )
        safe_label = _cin_slug(label, f"v2v_source_{source_video_index}")
        prompt = _cin_clean(prompt_override)
        if not prompt:
            subject_text = _cin_clean(subject) or "the subject"
            action = _cin_clean(action_prompt)
            if action and subject_text.lower() not in action.lower():
                action = f"{subject_text} {action}"
            prompt = _cin_join_prompt_parts(framing, action, continuity_note, extra_prompt)
        line = " | ".join([
            f"{max(0.01, _safe_float(duration_seconds, 4.0)):.2f}".rstrip("0").rstrip("."),
            str(cut_mode),
            str(max(0, min(8, _safe_int(source_video_index, 1)))),
            source_range,
            ref_token,
            str(max(0, min(8, _safe_int(audio_index, 0)))),
            safe_label,
            prompt,
            _cin_clean(dialogue_text),
            _cin_clean(voice_direction),
            str(v2v_mode),
        ])
        report = (
            f"v2v_line label={safe_label} source={source_video_index} "
            f"range={source_range} ref={ref_token} audio={audio_index}"
        )
        return (line, prompt, source_range, ref_token, report)


class IAMCCS_LTX2_CinematicLineStacker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["multigen_shot_lines", "v2v_timeline_lines", "generic_lines"], {"default": "multigen_shot_lines"}),
                "skip_empty": (["yes", "no"], {"default": "yes"}),
                "comment_header": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "line_1": ("STRING", {"default": "", "multiline": True}),
                "line_2": ("STRING", {"default": "", "multiline": True}),
                "line_3": ("STRING", {"default": "", "multiline": True}),
                "line_4": ("STRING", {"default": "", "multiline": True}),
                "line_5": ("STRING", {"default": "", "multiline": True}),
                "line_6": ("STRING", {"default": "", "multiline": True}),
                "line_7": ("STRING", {"default": "", "multiline": True}),
                "line_8": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("lines", "line_count", "report")
    FUNCTION = "stack"
    CATEGORY = "IAMCCS/LTX-2/Cinematic/User Friendly"

    def stack(
        self,
        mode,
        skip_empty,
        comment_header,
        line_1="",
        line_2="",
        line_3="",
        line_4="",
        line_5="",
        line_6="",
        line_7="",
        line_8="",
    ):
        lines: List[str] = []
        header = str(comment_header or "").strip()
        if header:
            for raw in header.splitlines():
                raw = raw.strip()
                if raw:
                    lines.append(raw if raw.startswith("#") else f"# {raw}")
        for value in (line_1, line_2, line_3, line_4, line_5, line_6, line_7, line_8):
            for raw in str(value or "").splitlines():
                text = raw.strip()
                if not text and str(skip_empty) == "yes":
                    continue
                if text:
                    lines.append(text)
        output = "\n".join(lines)
        real_count = sum(1 for line in lines if line and not line.lstrip().startswith("#"))
        report = f"line_stacker mode={mode} line_count={real_count}"
        return (output, int(real_count), report)


class IAMCCS_LTX2_CinematicMultiGenPlanner:
    DEFAULT_SHOTS = (
        "4.0 | hard_cut | 1 | 1 | field_A | Medium close-up on character A, he speaks with restrained emotion | I knew you would come back. | tired intimate low voice\n"
        "3.5 | hard_cut | 2 | 2 | reverse_B | Reverse angle on character B, she listens then answers | I never really left. | calm low voice\n"
        "2.5 | hard_cut | 3 | 0 | insert_photo | Close-up insert of the old photograph on the table | | \n"
        "4.0 | continuity_cut | 1>2 | 1 | return_A_pushin | Return to character A, slow push-in, visible hesitation | Then why does it feel different? | quiet broken voice"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_index": ("INT", {"default": 1, "min": 1, "max": 100000, "step": 1}),
                "render_id": ("STRING", {"default": "scene01"}),
                "scene_prompt": ("STRING", {
                    "default": "A cinematic dialogue scene with coherent eyelines, natural acting, film lighting.",
                    "multiline": True,
                }),
                "shot_lines": ("STRING", {
                    "default": cls.DEFAULT_SHOTS,
                    "multiline": True,
                    "tooltip": "One shot per line: seconds | cut_mode | ref or ref>ref | audio_index | label | shot prompt | dialogue | voice direction",
                }),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.001, "max": 240.0, "step": 0.01}),
                "ltx_round_mode": (["up", "nearest", "down", "none"], {"default": "up"}),
                "duration_mode": (["line_seconds", "audio_when_available", "max_line_audio"], {"default": "max_line_audio"}),
                "audio_padding_seconds": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 30.0, "step": 0.01}),
                "default_duration_seconds": ("FLOAT", {"default": 4.0, "min": 0.01, "max": 3600.0, "step": 0.01}),
                "default_cut_mode": (["hard_cut", "continuity_cut", "soft_cut", "match_cut"], {"default": "hard_cut"}),
                "default_audio_assignment": (["shot_number", "none"], {"default": "shot_number"}),
                "prompt_mode": (["scene_plus_shot", "shot_only"], {"default": "scene_plus_shot"}),
                "default_reference_strength": ("FLOAT", {"default": 0.92, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tail_reference_strength": ("FLOAT", {"default": 0.82, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tail_anchor_mode": (["auto", "same_reference", "secondary_only", "none"], {"default": "auto"}),
                "hard_cut_overlap_frames": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "continuity_overlap_frames": ("INT", {"default": 9, "min": 0, "max": 4096, "step": 1}),
            },
            "optional": {
                "audio_1": ("AUDIO",),
                "audio_2": ("AUDIO",),
                "audio_3": ("AUDIO",),
                "audio_4": ("AUDIO",),
                "audio_5": ("AUDIO",),
                "audio_6": ("AUDIO",),
                "audio_7": ("AUDIO",),
                "audio_8": ("AUDIO",),
            },
        }

    RETURN_TYPES = (
        "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING",
        "INT", "INT", "FLOAT", "FLOAT", "FLOAT", "INT", "INT", "INT", "INT",
        "FLOAT", "FLOAT", "STRING", "BOOLEAN", "INT", "STRING",
    )
    RETURN_NAMES = (
        "multigen_plan_json",
        "shot_plan_json",
        "shot_prompt",
        "dialogue_text",
        "voice_direction",
        "shot_label",
        "shot_render_id",
        "shot_number",
        "shot_count",
        "total_scene_seconds",
        "shot_start_seconds",
        "shot_duration_seconds",
        "shot_total_frames",
        "primary_reference_index",
        "secondary_reference_index",
        "audio_index",
        "audio_start_seconds",
        "audio_duration_seconds",
        "cut_mode",
        "use_context_bridge",
        "recommended_overlap_frames",
        "report",
    )
    FUNCTION = "plan"
    CATEGORY = "IAMCCS/LTX-2/Cinematic"

    @staticmethod
    def _clean(text: Any) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip())

    @staticmethod
    def _safe_label(text: str, default: str) -> str:
        raw = re.sub(r"[^A-Za-z0-9_-]+", "_", str(text or "").strip()).strip("_")
        return raw[:64] or default

    @staticmethod
    def _normalize_cut_mode(value: Any, default: str) -> str:
        token = str(value or default or "hard_cut").strip().lower().replace("-", "_").replace(" ", "_")
        if token in {"hard", "cut", "hardcut", "hard_cut", "none", "new_shot"}:
            return "hard_cut"
        if token in {"continuity", "cont", "bridge", "context", "continuity_cut", "continue"}:
            return "continuity_cut"
        if token in {"soft", "soft_cut", "softcut", "dissolve"}:
            return "soft_cut"
        if token in {"match", "match_cut", "matchcut", "graphic_match"}:
            return "match_cut"
        return str(default or "hard_cut")

    @staticmethod
    def _parse_reference_token(token: Any, default_ref: int) -> Tuple[int, int, str]:
        raw = str(token or "").strip().lower()
        values = [max(1, _safe_int(v, 1)) for v in re.findall(r"\d+", raw)]
        if not values:
            return max(1, int(default_ref)), 0, "single"
        primary = values[0]
        secondary = values[1] if len(values) > 1 else 0
        if ">" in raw or "to" in raw:
            mode = "start_to_end"
        elif "+" in raw or "," in raw or "/" in raw:
            mode = "dual"
        else:
            mode = "single"
        return primary, secondary, mode

    @staticmethod
    def _parse_audio_index(token: Any, fallback: int) -> int:
        raw = str(token or "").strip().lower()
        if raw in {"", "-", "none", "off", "no", "0"}:
            return 0
        found = re.findall(r"\d+", raw)
        if found:
            return max(0, min(8, _safe_int(found[0], fallback)))
        return max(0, min(8, int(fallback)))

    @staticmethod
    def _audio_map(**kwargs) -> Dict[int, Any]:
        return {idx: kwargs.get(f"audio_{idx}") for idx in range(1, 9)}

    def _parse_shot_lines(
        self,
        shot_lines: str,
        default_duration_seconds: float,
        default_cut_mode: str,
        default_audio_assignment: str,
    ) -> List[Dict[str, Any]]:
        lines = [line.strip() for line in str(shot_lines or "").splitlines() if line.strip() and not line.strip().startswith("#")]
        if not lines:
            lines = [line.strip() for line in self.DEFAULT_SHOTS.splitlines() if line.strip()]
        shots: List[Dict[str, Any]] = []
        for idx, line in enumerate(lines):
            parts = [part.strip() for part in line.split("|")]
            while len(parts) < 8:
                parts.append("")
            number = idx + 1
            duration = max(0.01, _safe_float(parts[0], default_duration_seconds))
            cut_mode = self._normalize_cut_mode(parts[1], default_cut_mode)
            primary_ref, secondary_ref, ref_mode = self._parse_reference_token(parts[2], number)
            fallback_audio = number if str(default_audio_assignment) == "shot_number" else 0
            audio_index = self._parse_audio_index(parts[3], fallback_audio)
            label = self._clean(parts[4]) or f"shot_{number}"
            shot_prompt = self._clean(parts[5]) or label
            dialogue = self._clean(parts[6])
            voice = self._clean(parts[7])
            shots.append({
                "shot_number": number,
                "line": line,
                "requested_seconds": float(duration),
                "cut_mode": cut_mode,
                "primary_reference_index": int(primary_ref),
                "secondary_reference_index": int(secondary_ref),
                "reference_mode": ref_mode,
                "audio_index": int(audio_index),
                "label": label,
                "shot_prompt": shot_prompt,
                "dialogue_text": dialogue,
                "voice_direction": voice,
            })
        return shots

    def _resolve_durations(
        self,
        shots: List[Dict[str, Any]],
        audio_inputs: Dict[int, Any],
        fps: float,
        ltx_round_mode: str,
        duration_mode: str,
        audio_padding_seconds: float,
    ) -> List[Dict[str, Any]]:
        cursor_frame = 0
        fps = max(0.001, float(fps))
        for shot in shots:
            audio_idx = int(shot.get("audio_index", 0))
            audio_seconds_raw = _audio_duration_seconds(audio_inputs.get(audio_idx)) if audio_idx > 0 else None
            audio_seconds = float(audio_seconds_raw or 0.0)
            requested = max(0.01, float(shot.get("requested_seconds", 0.01)))
            if str(duration_mode) == "audio_when_available" and audio_seconds_raw is not None:
                target_seconds = max(0.01, audio_seconds + float(audio_padding_seconds))
            elif str(duration_mode) == "max_line_audio" and audio_seconds_raw is not None:
                target_seconds = max(requested, audio_seconds + float(audio_padding_seconds))
            else:
                target_seconds = requested

            raw_frames = max(1, int(round(target_seconds * fps)))
            total_frames = _fix_ltx_frames(raw_frames, str(ltx_round_mode))
            duration_seconds = float(total_frames) / fps
            shot["audio_seconds"] = audio_seconds
            shot["duration_seconds"] = duration_seconds
            shot["total_frames"] = int(total_frames)
            shot["start_frame"] = int(cursor_frame)
            shot["end_frame"] = int(cursor_frame + total_frames)
            shot["start_seconds"] = float(cursor_frame) / fps
            shot["end_seconds"] = float(cursor_frame + total_frames) / fps
            shot["audio_start_seconds"] = shot["start_seconds"]
            shot["audio_duration_seconds"] = audio_seconds if audio_seconds_raw is not None else duration_seconds
            cursor_frame += int(total_frames)
        return shots

    @staticmethod
    def _recommended_overlap(cut_mode: str, hard_cut_overlap_frames: int, continuity_overlap_frames: int) -> Tuple[bool, int]:
        cut_mode = str(cut_mode or "hard_cut")
        if cut_mode == "hard_cut":
            overlap = max(0, int(hard_cut_overlap_frames))
            return (overlap > 0, overlap)
        overlap = max(0, int(continuity_overlap_frames))
        return (overlap > 0, overlap)

    def _build_shot_plan(
        self,
        shot: Dict[str, Any],
        fps: float,
        default_reference_strength: float,
        tail_reference_strength: float,
        tail_anchor_mode: str,
    ) -> Dict[str, Any]:
        total_frames = max(1, int(shot.get("total_frames", 1)))
        label = str(shot.get("label", "shot"))
        cut_mode = str(shot.get("cut_mode", "hard_cut"))
        primary = max(1, int(shot.get("primary_reference_index", 1)))
        secondary = max(0, int(shot.get("secondary_reference_index", 0)))
        entries: List[Dict[str, Any]] = []
        entries.append({
            "frame": 0,
            "position": 0.0,
            "reference_index": primary,
            "strength": _clamp(default_reference_strength, 0.0, 1.0),
            "label": f"{label}_start",
            "cut_mode": cut_mode,
        })

        should_tail = False
        tail_ref = 0
        mode = str(tail_anchor_mode or "auto")
        if mode == "same_reference":
            should_tail = True
            tail_ref = primary
        elif mode == "secondary_only":
            should_tail = secondary > 0
            tail_ref = secondary
        elif mode == "auto":
            if secondary > 0:
                should_tail = True
                tail_ref = secondary
            elif cut_mode in {"continuity_cut", "soft_cut", "match_cut"}:
                should_tail = True
                tail_ref = primary

        if should_tail and tail_ref > 0 and total_frames > 1:
            entries.append({
                "frame": total_frames - 1,
                "position": 1.0,
                "reference_index": int(tail_ref),
                "strength": _clamp(tail_reference_strength, 0.0, 1.0),
                "label": f"{label}_tail",
                "cut_mode": cut_mode,
            })

        return {
            "kind": "iamccs_ltx2_cinematic_shot_plan",
            "version": 1,
            "source": "IAMCCS_LTX2_CinematicMultiGenPlanner",
            "shot_number": int(shot.get("shot_number", 1)),
            "label": label,
            "cut_mode": cut_mode,
            "fps": float(fps),
            "total_frames": int(total_frames),
            "total_seconds": float(shot.get("duration_seconds", total_frames / max(0.001, float(fps)))),
            "reference_mode": str(shot.get("reference_mode", "single")),
            "entries": entries,
        }

    def _build_prompt(self, scene_prompt: str, shot_prompt: str, prompt_mode: str) -> str:
        scene = self._clean(scene_prompt)
        shot = self._clean(shot_prompt)
        if str(prompt_mode) == "shot_only":
            return shot or scene or "A cinematic shot."
        parts = [part for part in [scene, shot] if part]
        return ". ".join(part.rstrip(".") for part in parts).strip() or "A cinematic shot."

    def plan(
        self,
        shot_index,
        render_id,
        scene_prompt,
        shot_lines,
        fps,
        ltx_round_mode,
        duration_mode,
        audio_padding_seconds,
        default_duration_seconds,
        default_cut_mode,
        default_audio_assignment,
        prompt_mode,
        default_reference_strength,
        tail_reference_strength,
        tail_anchor_mode,
        hard_cut_overlap_frames,
        continuity_overlap_frames,
        audio_1=None,
        audio_2=None,
        audio_3=None,
        audio_4=None,
        audio_5=None,
        audio_6=None,
        audio_7=None,
        audio_8=None,
    ):
        audio_inputs = self._audio_map(
            audio_1=audio_1,
            audio_2=audio_2,
            audio_3=audio_3,
            audio_4=audio_4,
            audio_5=audio_5,
            audio_6=audio_6,
            audio_7=audio_7,
            audio_8=audio_8,
        )
        shots = self._parse_shot_lines(
            shot_lines,
            float(default_duration_seconds),
            str(default_cut_mode),
            str(default_audio_assignment),
        )
        shots = self._resolve_durations(
            shots,
            audio_inputs,
            float(fps),
            str(ltx_round_mode),
            str(duration_mode),
            float(audio_padding_seconds),
        )
        shot_count = max(1, len(shots))
        selected_idx = max(0, min(int(shot_index) - 1, shot_count - 1))
        selected = shots[selected_idx]
        use_context, overlap = self._recommended_overlap(
            str(selected.get("cut_mode", "hard_cut")),
            int(hard_cut_overlap_frames),
            int(continuity_overlap_frames),
        )
        shot_plan = self._build_shot_plan(
            selected,
            float(fps),
            float(default_reference_strength),
            float(tail_reference_strength),
            str(tail_anchor_mode),
        )
        prompt = self._build_prompt(scene_prompt, str(selected.get("shot_prompt", "")), str(prompt_mode))
        total_scene_frames = int(max(shot.get("end_frame", 0) for shot in shots)) if shots else 0
        total_scene_seconds = float(total_scene_frames) / max(0.001, float(fps))
        safe_render = self._safe_label(render_id, "scene")
        safe_label = self._safe_label(str(selected.get("label", "")), f"shot_{selected_idx + 1}")
        shot_render_id = f"{safe_render}_shot_{selected_idx + 1:03d}_{safe_label}"

        plan = {
            "kind": "iamccs_ltx2_cinematic_multigen_plan",
            "version": 1,
            "fps": float(fps),
            "ltx_round_mode": str(ltx_round_mode),
            "duration_mode": str(duration_mode),
            "audio_padding_seconds": float(audio_padding_seconds),
            "render_id": str(render_id or ""),
            "shot_count": int(shot_count),
            "selected_shot_index": int(selected_idx),
            "total_scene_frames": int(total_scene_frames),
            "total_scene_seconds": float(total_scene_seconds),
            "shots": shots,
        }

        report_lines = [
            f"multigen shots={shot_count} selected={selected_idx + 1}/{shot_count}",
            f"cut={selected.get('cut_mode')} context={bool(use_context)} overlap={overlap}",
            f"frames={selected.get('total_frames')} duration={float(selected.get('duration_seconds', 0.0)):.3f}s",
            f"audio_index={selected.get('audio_index')} audio={float(selected.get('audio_seconds', 0.0)):.3f}s",
        ]
        return (
            _json_dumps(plan),
            _json_dumps(shot_plan),
            prompt,
            str(selected.get("dialogue_text", "")),
            str(selected.get("voice_direction", "")),
            str(selected.get("label", "")),
            shot_render_id,
            int(selected_idx + 1),
            int(shot_count),
            float(total_scene_seconds),
            float(selected.get("start_seconds", 0.0)),
            float(selected.get("duration_seconds", 0.0)),
            int(selected.get("total_frames", 1)),
            int(selected.get("primary_reference_index", 1)),
            int(selected.get("secondary_reference_index", 0)),
            int(selected.get("audio_index", 0)),
            float(selected.get("audio_start_seconds", 0.0)),
            float(selected.get("audio_duration_seconds", 0.0)),
            str(selected.get("cut_mode", "hard_cut")),
            bool(use_context),
            int(overlap),
            " | ".join(report_lines),
        )


class IAMCCS_LTX2_CinematicShotAudioSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_index": ("INT", {"default": 1, "min": 0, "max": 8, "step": 1}),
                "target_duration_seconds": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 36000.0, "step": 0.01}),
                "fallback": (["silent", "first_connected", "error"], {"default": "silent"}),
                "trim_or_pad": (["pad_or_trim", "as_is"], {"default": "pad_or_trim"}),
                "silent_sample_rate": ("INT", {"default": 44100, "min": 8000, "max": 192000, "step": 1}),
            },
            "optional": {
                "audio_1": ("AUDIO",),
                "audio_2": ("AUDIO",),
                "audio_3": ("AUDIO",),
                "audio_4": ("AUDIO",),
                "audio_5": ("AUDIO",),
                "audio_6": ("AUDIO",),
                "audio_7": ("AUDIO",),
                "audio_8": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("AUDIO", "FLOAT", "STRING")
    RETURN_NAMES = ("audio", "audio_seconds", "report")
    FUNCTION = "select"
    CATEGORY = "IAMCCS/LTX-2/Cinematic"

    @staticmethod
    def _unwrap_audio(audio: Any) -> Optional[Dict[str, Any]]:
        current = audio
        for _ in range(6):
            if current is None:
                return None
            if isinstance(current, dict):
                if "waveform" in current and "sample_rate" in current:
                    return current
                if "audio" in current:
                    current = current.get("audio")
                    continue
                if len(current) == 1:
                    current = next(iter(current.values()))
                    continue
            if isinstance(current, (list, tuple)) and current:
                current = current[0]
                continue
            break
        return None

    @staticmethod
    def _normalize_waveform(waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 3:
            return waveform
        if waveform.ndim == 2:
            return waveform.unsqueeze(0)
        if waveform.ndim == 1:
            return waveform.view(1, 1, -1)
        raise ValueError(f"Unsupported AUDIO waveform rank: {waveform.ndim}")

    def _prepare_audio(self, audio: Dict[str, Any], target_duration_seconds: float, trim_or_pad: str) -> Dict[str, Any]:
        waveform = audio.get("waveform")
        sample_rate = _safe_int(audio.get("sample_rate"), 0)
        if sample_rate <= 0 or not torch.is_tensor(waveform):
            raise ValueError("Invalid AUDIO: expected waveform tensor and sample_rate")
        waveform = self._normalize_waveform(waveform).detach().clone()
        out = dict(audio)
        if str(trim_or_pad) == "pad_or_trim":
            target_samples = max(0, int(round(float(target_duration_seconds) * float(sample_rate))))
            current_samples = int(waveform.shape[-1])
            if target_samples <= 0:
                target_samples = current_samples
            if current_samples > target_samples:
                waveform = waveform[:, :, :target_samples]
            elif current_samples < target_samples:
                pad = torch.zeros(
                    (waveform.shape[0], waveform.shape[1], target_samples - current_samples),
                    dtype=waveform.dtype,
                    device=waveform.device,
                )
                waveform = torch.cat([waveform, pad], dim=-1)
        out["waveform"] = waveform
        out["sample_rate"] = int(sample_rate)
        return out

    @staticmethod
    def _silent_audio(target_duration_seconds: float, sample_rate: int) -> Dict[str, Any]:
        samples = max(1, int(round(max(0.01, float(target_duration_seconds)) * float(sample_rate))))
        return {
            "waveform": torch.zeros((1, 1, samples), dtype=torch.float32),
            "sample_rate": int(sample_rate),
        }

    def select(
        self,
        audio_index,
        target_duration_seconds,
        fallback,
        trim_or_pad,
        silent_sample_rate,
        audio_1=None,
        audio_2=None,
        audio_3=None,
        audio_4=None,
        audio_5=None,
        audio_6=None,
        audio_7=None,
        audio_8=None,
    ):
        audios = {
            1: audio_1,
            2: audio_2,
            3: audio_3,
            4: audio_4,
            5: audio_5,
            6: audio_6,
            7: audio_7,
            8: audio_8,
        }
        selected_index = max(0, min(8, int(audio_index)))
        selected = self._unwrap_audio(audios.get(selected_index)) if selected_index > 0 else None
        source = f"audio_{selected_index}" if selected is not None else "none"
        if selected is None and str(fallback) == "first_connected":
            for idx in range(1, 9):
                selected = self._unwrap_audio(audios.get(idx))
                if selected is not None:
                    selected_index = idx
                    source = f"audio_{idx}"
                    break
        if selected is None:
            if str(fallback) == "error":
                raise ValueError(f"No audio connected for audio_index={selected_index}")
            selected = self._silent_audio(float(target_duration_seconds), int(silent_sample_rate))
            source = "silent"

        out = self._prepare_audio(selected, float(target_duration_seconds), str(trim_or_pad))
        seconds = _audio_duration_seconds(out) or 0.0
        report = f"cinematic_audio_selector source={source} index={selected_index} seconds={seconds:.3f} mode={trim_or_pad}"
        return (out, float(seconds), report)


class IAMCCS_LTX2_CinematicV2VTimelinePlanner:
    DEFAULT_TIMELINE = (
        "4.0 | hard_cut | 1 | all | 1 | 1 | field_A | V2V close-up on character A speaking, keep source motion | I knew you would come back. | tired low voice | v2v_source_plus_reference\n"
        "3.5 | hard_cut | 2 | all | 2 | 2 | reverse_B | V2V reverse angle on character B answering | I never really left. | calm low voice | v2v_source_plus_reference\n"
        "6.0 | continuity_cut | 1 | 48+145 | 1>2 | 0 | return_A | Continue from source video 1, slow push-in and reaction | | | two_segments_if_long"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "render_id": ("STRING", {"default": "v2v_scene01"}),
                "scene_prompt": ("STRING", {
                    "default": "A cinematic V2V sequence with coherent continuity, controlled cuts, natural motion.",
                    "multiline": True,
                }),
                "timeline_lines": ("STRING", {
                    "default": cls.DEFAULT_TIMELINE,
                    "multiline": True,
                    "tooltip": "seconds | cut_mode | source_video | source_range | reference | audio | label | prompt | dialogue | voice | v2v_mode",
                }),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.001, "max": 240.0, "step": 0.01}),
                "ltx_round_mode": (["up", "nearest", "down", "none"], {"default": "up"}),
                "duration_mode": (["line_seconds", "source_range_when_known", "audio_when_available", "max_line_audio_source"], {"default": "max_line_audio_source"}),
                "audio_padding_seconds": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 30.0, "step": 0.01}),
                "default_duration_seconds": ("FLOAT", {"default": 4.0, "min": 0.01, "max": 3600.0, "step": 0.01}),
                "default_cut_mode": (["hard_cut", "continuity_cut", "soft_cut", "match_cut"], {"default": "hard_cut"}),
                "default_v2v_mode": (["v2v_source_plus_reference", "v2v_source_context", "i2v_from_reference", "two_segments_if_long", "loop_if_long"], {"default": "v2v_source_plus_reference"}),
                "prompt_mode": (["scene_plus_shot", "shot_only"], {"default": "scene_plus_shot"}),
                "default_reference_strength": ("FLOAT", {"default": 0.88, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tail_reference_strength": ("FLOAT", {"default": 0.72, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tail_anchor_mode": (["auto", "same_reference", "secondary_only", "none"], {"default": "auto"}),
                "hard_cut_overlap_frames": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "continuity_overlap_frames": ("INT", {"default": 9, "min": 0, "max": 4096, "step": 1}),
                "max_single_v2v_seconds": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 3600.0, "step": 0.1}),
            },
            "optional": {
                "source_video_1": ("IMAGE",),
                "source_video_2": ("IMAGE",),
                "source_video_3": ("IMAGE",),
                "source_video_4": ("IMAGE",),
                "source_video_5": ("IMAGE",),
                "source_video_6": ("IMAGE",),
                "source_video_7": ("IMAGE",),
                "source_video_8": ("IMAGE",),
                "audio_1": ("AUDIO",),
                "audio_2": ("AUDIO",),
                "audio_3": ("AUDIO",),
                "audio_4": ("AUDIO",),
                "audio_5": ("AUDIO",),
                "audio_6": ("AUDIO",),
                "audio_7": ("AUDIO",),
                "audio_8": ("AUDIO",),
            },
        }

    RETURN_TYPES = (
        "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING",
        "INT", "INT", "INT",
        "FLOAT", "FLOAT", "FLOAT",
        "INT", "INT", "INT", "INT", "INT", "INT", "INT", "INT",
        "FLOAT", "STRING", "STRING", "BOOLEAN", "INT", "STRING",
    )
    RETURN_NAMES = (
        "timeline_plan_json",
        "shot_plan_json",
        "shot_prompt",
        "dialogue_text",
        "voice_direction",
        "shot_label",
        "shot_render_id",
        "current_segment",
        "total_segments",
        "shot_number",
        "total_scene_seconds",
        "shot_start_seconds",
        "shot_duration_seconds",
        "shot_total_frames",
        "source_video_index",
        "source_start_frame",
        "source_end_frame",
        "source_frame_count",
        "primary_reference_index",
        "secondary_reference_index",
        "audio_index",
        "audio_duration_seconds",
        "cut_mode",
        "recommended_generation_mode",
        "use_context_bridge",
        "recommended_overlap_frames",
        "report",
    )
    FUNCTION = "plan"
    CATEGORY = "IAMCCS/LTX-2/Cinematic"

    @staticmethod
    def _clean(text: Any) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip())

    @staticmethod
    def _safe_label(text: str, default: str) -> str:
        raw = re.sub(r"[^A-Za-z0-9_-]+", "_", str(text or "").strip()).strip("_")
        return raw[:64] or default

    @staticmethod
    def _image_count(images: Any) -> Optional[int]:
        if torch.is_tensor(images) and images.ndim >= 4:
            return int(images.shape[0])
        return None

    @staticmethod
    def _normalize_v2v_mode(value: Any, default: str) -> str:
        token = str(value or default or "v2v_source_plus_reference").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "v2v": "v2v_source_plus_reference",
            "source": "v2v_source_context",
            "context": "v2v_source_context",
            "i2v": "i2v_from_reference",
            "two": "two_segments_if_long",
            "2seg": "two_segments_if_long",
            "two_segments": "two_segments_if_long",
            "loop": "loop_if_long",
            "extend": "loop_if_long",
        }
        token = aliases.get(token, token)
        valid = {"v2v_source_plus_reference", "v2v_source_context", "i2v_from_reference", "two_segments_if_long", "loop_if_long"}
        return token if token in valid else str(default or "v2v_source_plus_reference")

    @staticmethod
    def _parse_video_index(token: Any, default: int) -> int:
        raw = str(token or "").strip().lower()
        if raw in {"", "-", "none", "off", "0"}:
            return 0
        found = re.findall(r"\d+", raw)
        if found:
            return max(0, min(8, _safe_int(found[0], default)))
        return max(0, min(8, int(default)))

    @staticmethod
    def _parse_range(range_text: Any, source_total: Optional[int], fallback_frames: int) -> Tuple[int, int, int, str]:
        raw = str(range_text or "").strip().lower()
        total = int(source_total) if source_total is not None and int(source_total) > 0 else None
        fallback = max(1, int(fallback_frames))

        if raw in {"", "auto", "all", "*"}:
            if total is not None:
                return 0, total, total, "all"
            return 0, fallback, fallback, "duration"

        if raw.startswith("tail") or raw.startswith("last"):
            nums = re.findall(r"\d+", raw)
            count = max(1, _safe_int(nums[0], fallback) if nums else fallback)
            if total is None:
                return 0, count, count, "tail_no_total"
            end = total
            start = max(0, end - count)
            return start, end, max(1, end - start), "tail"

        plus_match = re.match(r"^\s*(\d+)\s*\+\s*(\d+)\s*$", raw)
        if plus_match:
            start = max(0, _safe_int(plus_match.group(1), 0))
            count = max(1, _safe_int(plus_match.group(2), fallback))
            end = start + count
            if total is not None:
                end = min(total, end)
            return start, max(start + 1, end), max(1, end - start), "start_plus_count"

        range_match = re.match(r"^\s*(\d+)\s*[:-]\s*(\d+)\s*$", raw)
        if range_match:
            start = max(0, _safe_int(range_match.group(1), 0))
            end = max(start + 1, _safe_int(range_match.group(2), start + fallback))
            if total is not None:
                end = min(total, end)
            return start, max(start + 1, end), max(1, end - start), "range"

        nums = re.findall(r"\d+", raw)
        if len(nums) == 1:
            count = max(1, _safe_int(nums[0], fallback))
            end = count if total is None else min(total, count)
            return 0, max(1, end), max(1, end), "count"

        return 0, fallback, fallback, "fallback"

    def _parse_lines(self, timeline_lines: str, default_duration_seconds: float, default_cut_mode: str, default_v2v_mode: str) -> List[Dict[str, Any]]:
        lines = [line.strip() for line in str(timeline_lines or "").splitlines() if line.strip() and not line.strip().startswith("#")]
        if not lines:
            lines = [line.strip() for line in self.DEFAULT_TIMELINE.splitlines() if line.strip()]
        shots: List[Dict[str, Any]] = []
        for idx, line in enumerate(lines):
            parts = [part.strip() for part in line.split("|")]
            while len(parts) < 11:
                parts.append("")
            number = idx + 1
            duration = max(0.01, _safe_float(parts[0], default_duration_seconds))
            cut_mode = IAMCCS_LTX2_CinematicMultiGenPlanner._normalize_cut_mode(parts[1], default_cut_mode)
            source_video_index = self._parse_video_index(parts[2], number)
            range_text = parts[3]
            primary, secondary, ref_mode = IAMCCS_LTX2_CinematicMultiGenPlanner._parse_reference_token(parts[4], number)
            audio_index = IAMCCS_LTX2_CinematicMultiGenPlanner._parse_audio_index(parts[5], number)
            label = self._clean(parts[6]) or f"shot_{number}"
            shot_prompt = self._clean(parts[7]) or label
            dialogue = self._clean(parts[8])
            voice = self._clean(parts[9])
            v2v_mode = self._normalize_v2v_mode(parts[10], default_v2v_mode)
            shots.append({
                "shot_number": number,
                "line": line,
                "requested_seconds": float(duration),
                "cut_mode": cut_mode,
                "source_video_index": int(source_video_index),
                "source_range_text": range_text,
                "primary_reference_index": int(primary),
                "secondary_reference_index": int(secondary),
                "reference_mode": ref_mode,
                "audio_index": int(audio_index),
                "label": label,
                "shot_prompt": shot_prompt,
                "dialogue_text": dialogue,
                "voice_direction": voice,
                "v2v_mode": v2v_mode,
            })
        return shots

    def _resolve_shots(
        self,
        shots: List[Dict[str, Any]],
        source_counts: Dict[int, Optional[int]],
        audio_inputs: Dict[int, Any],
        fps: float,
        ltx_round_mode: str,
        duration_mode: str,
        audio_padding_seconds: float,
        max_single_v2v_seconds: float,
    ) -> List[Dict[str, Any]]:
        cursor_frame = 0
        fps = max(0.001, float(fps))
        for shot in shots:
            source_idx = int(shot.get("source_video_index", 0))
            source_total = source_counts.get(source_idx)
            requested = max(0.01, float(shot.get("requested_seconds", 0.01)))
            requested_frames = max(1, int(round(requested * fps)))
            start, end, source_count, range_mode = self._parse_range(
                shot.get("source_range_text", ""),
                source_total,
                requested_frames,
            )

            audio_idx = int(shot.get("audio_index", 0))
            audio_seconds_raw = _audio_duration_seconds(audio_inputs.get(audio_idx)) if audio_idx > 0 else None
            audio_seconds = float(audio_seconds_raw or 0.0)
            source_seconds = float(source_count) / fps

            if str(duration_mode) == "source_range_when_known" and source_total is not None:
                target_seconds = max(0.01, source_seconds)
            elif str(duration_mode) == "audio_when_available" and audio_seconds_raw is not None:
                target_seconds = max(0.01, audio_seconds + float(audio_padding_seconds))
            elif str(duration_mode) == "max_line_audio_source":
                candidates = [requested]
                if audio_seconds_raw is not None:
                    candidates.append(audio_seconds + float(audio_padding_seconds))
                if source_total is not None:
                    candidates.append(source_seconds)
                target_seconds = max(candidates)
            else:
                target_seconds = requested

            total_frames = _fix_ltx_frames(max(1, int(round(target_seconds * fps))), str(ltx_round_mode))
            shot["source_start_frame"] = int(start)
            shot["source_end_frame"] = int(end)
            shot["source_frame_count"] = int(max(1, source_count))
            shot["source_range_mode"] = range_mode
            shot["source_total_frames"] = int(source_total or 0)
            shot["audio_seconds"] = audio_seconds
            shot["duration_seconds"] = float(total_frames) / fps
            shot["total_frames"] = int(total_frames)
            shot["start_frame"] = int(cursor_frame)
            shot["end_frame"] = int(cursor_frame + total_frames)
            shot["start_seconds"] = float(cursor_frame) / fps
            shot["end_seconds"] = float(cursor_frame + total_frames) / fps
            base_mode = str(shot.get("v2v_mode", "v2v_source_plus_reference"))
            if float(shot["duration_seconds"]) > float(max_single_v2v_seconds):
                if base_mode == "loop_if_long":
                    rec_mode = "loop_extend"
                elif base_mode in {"two_segments_if_long", "v2v_source_plus_reference", "v2v_source_context"}:
                    rec_mode = "two_segments"
                else:
                    rec_mode = "single_v2v"
            else:
                rec_mode = "single_v2v" if base_mode != "i2v_from_reference" else "single_i2v"
            shot["recommended_generation_mode"] = rec_mode
            cursor_frame += int(total_frames)
        return shots

    def _build_shot_plan(self, shot: Dict[str, Any], fps: float, default_reference_strength: float, tail_reference_strength: float, tail_anchor_mode: str) -> Dict[str, Any]:
        return IAMCCS_LTX2_CinematicMultiGenPlanner()._build_shot_plan(
            shot,
            fps,
            default_reference_strength,
            tail_reference_strength,
            tail_anchor_mode,
        )

    def _build_prompt(self, scene_prompt: str, shot_prompt: str, prompt_mode: str) -> str:
        return IAMCCS_LTX2_CinematicMultiGenPlanner()._build_prompt(scene_prompt, shot_prompt, prompt_mode)

    @staticmethod
    def _recommended_overlap(cut_mode: str, hard_cut_overlap_frames: int, continuity_overlap_frames: int) -> Tuple[bool, int]:
        return IAMCCS_LTX2_CinematicMultiGenPlanner._recommended_overlap(cut_mode, hard_cut_overlap_frames, continuity_overlap_frames)

    def plan(
        self,
        segment_index,
        render_id,
        scene_prompt,
        timeline_lines,
        fps,
        ltx_round_mode,
        duration_mode,
        audio_padding_seconds,
        default_duration_seconds,
        default_cut_mode,
        default_v2v_mode,
        prompt_mode,
        default_reference_strength,
        tail_reference_strength,
        tail_anchor_mode,
        hard_cut_overlap_frames,
        continuity_overlap_frames,
        max_single_v2v_seconds,
        source_video_1=None,
        source_video_2=None,
        source_video_3=None,
        source_video_4=None,
        source_video_5=None,
        source_video_6=None,
        source_video_7=None,
        source_video_8=None,
        audio_1=None,
        audio_2=None,
        audio_3=None,
        audio_4=None,
        audio_5=None,
        audio_6=None,
        audio_7=None,
        audio_8=None,
    ):
        source_counts = {
            1: self._image_count(source_video_1),
            2: self._image_count(source_video_2),
            3: self._image_count(source_video_3),
            4: self._image_count(source_video_4),
            5: self._image_count(source_video_5),
            6: self._image_count(source_video_6),
            7: self._image_count(source_video_7),
            8: self._image_count(source_video_8),
        }
        audio_inputs = {
            1: audio_1,
            2: audio_2,
            3: audio_3,
            4: audio_4,
            5: audio_5,
            6: audio_6,
            7: audio_7,
            8: audio_8,
        }
        shots = self._parse_lines(timeline_lines, float(default_duration_seconds), str(default_cut_mode), str(default_v2v_mode))
        shots = self._resolve_shots(
            shots,
            source_counts,
            audio_inputs,
            float(fps),
            str(ltx_round_mode),
            str(duration_mode),
            float(audio_padding_seconds),
            float(max_single_v2v_seconds),
        )
        total_segments = max(1, len(shots))
        current_segment = max(0, min(int(segment_index), total_segments - 1))
        selected = shots[current_segment]
        use_context, overlap = self._recommended_overlap(str(selected.get("cut_mode", "hard_cut")), int(hard_cut_overlap_frames), int(continuity_overlap_frames))
        shot_plan = self._build_shot_plan(selected, float(fps), float(default_reference_strength), float(tail_reference_strength), str(tail_anchor_mode))
        prompt = self._build_prompt(scene_prompt, str(selected.get("shot_prompt", "")), str(prompt_mode))
        total_scene_frames = int(max(shot.get("end_frame", 0) for shot in shots)) if shots else 0
        total_scene_seconds = float(total_scene_frames) / max(0.001, float(fps))
        safe_render = self._safe_label(render_id, "v2v_scene")
        safe_label = self._safe_label(str(selected.get("label", "")), f"shot_{current_segment + 1}")
        shot_render_id = f"{safe_render}_seg_{current_segment + 1:03d}_{safe_label}"

        plan = {
            "kind": "iamccs_ltx2_cinematic_v2v_timeline_plan",
            "version": 1,
            "fps": float(fps),
            "render_id": str(render_id or ""),
            "duration_mode": str(duration_mode),
            "ltx_round_mode": str(ltx_round_mode),
            "segment_index": int(current_segment),
            "total_segments": int(total_segments),
            "total_scene_frames": int(total_scene_frames),
            "total_scene_seconds": float(total_scene_seconds),
            "shots": shots,
        }
        report = (
            f"v2v_timeline segment={current_segment + 1}/{total_segments} "
            f"src={selected.get('source_video_index')} range={selected.get('source_start_frame')}-{selected.get('source_end_frame')} "
            f"ref={selected.get('primary_reference_index')} audio={selected.get('audio_index')} "
            f"cut={selected.get('cut_mode')} mode={selected.get('recommended_generation_mode')} overlap={overlap}"
        )
        return (
            _json_dumps(plan),
            _json_dumps(shot_plan),
            prompt,
            str(selected.get("dialogue_text", "")),
            str(selected.get("voice_direction", "")),
            str(selected.get("label", "")),
            shot_render_id,
            int(current_segment),
            int(total_segments),
            int(current_segment + 1),
            float(total_scene_seconds),
            float(selected.get("start_seconds", 0.0)),
            float(selected.get("duration_seconds", 0.0)),
            int(selected.get("total_frames", 1)),
            int(selected.get("source_video_index", 0)),
            int(selected.get("source_start_frame", 0)),
            int(selected.get("source_end_frame", 1)),
            int(selected.get("source_frame_count", 1)),
            int(selected.get("primary_reference_index", 1)),
            int(selected.get("secondary_reference_index", 0)),
            int(selected.get("audio_index", 0)),
            float(selected.get("audio_seconds", 0.0)),
            str(selected.get("cut_mode", "hard_cut")),
            str(selected.get("recommended_generation_mode", "single_v2v")),
            bool(use_context),
            int(overlap),
            report,
        )


class IAMCCS_LTX2_CinematicV2VAssetSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_video_index": ("INT", {"default": 1, "min": 0, "max": 8, "step": 1}),
                "primary_reference_index": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "secondary_reference_index": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1}),
                "fallback": (["first_connected", "primary_reference_as_video", "error"], {"default": "first_connected"}),
            },
            "optional": {
                "source_video_1": ("IMAGE",),
                "source_video_2": ("IMAGE",),
                "source_video_3": ("IMAGE",),
                "source_video_4": ("IMAGE",),
                "source_video_5": ("IMAGE",),
                "source_video_6": ("IMAGE",),
                "source_video_7": ("IMAGE",),
                "source_video_8": ("IMAGE",),
                "reference_1": ("IMAGE",),
                "reference_2": ("IMAGE",),
                "reference_3": ("IMAGE",),
                "reference_4": ("IMAGE",),
                "reference_5": ("IMAGE",),
                "reference_6": ("IMAGE",),
                "reference_7": ("IMAGE",),
                "reference_8": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "INT", "STRING")
    RETURN_NAMES = ("source_video", "primary_reference", "secondary_reference", "source_video_frames", "report")
    FUNCTION = "select"
    CATEGORY = "IAMCCS/LTX-2/Cinematic"

    @staticmethod
    def _first_connected(items: Dict[int, Any]) -> Tuple[int, Any]:
        for idx in range(1, 9):
            value = items.get(idx)
            if value is not None:
                return idx, value
        return 0, None

    @staticmethod
    def _count(images: Any) -> int:
        if torch.is_tensor(images) and images.ndim >= 4:
            return int(images.shape[0])
        return 0

    def select(
        self,
        source_video_index,
        primary_reference_index,
        secondary_reference_index,
        fallback,
        source_video_1=None,
        source_video_2=None,
        source_video_3=None,
        source_video_4=None,
        source_video_5=None,
        source_video_6=None,
        source_video_7=None,
        source_video_8=None,
        reference_1=None,
        reference_2=None,
        reference_3=None,
        reference_4=None,
        reference_5=None,
        reference_6=None,
        reference_7=None,
        reference_8=None,
    ):
        videos = {
            1: source_video_1,
            2: source_video_2,
            3: source_video_3,
            4: source_video_4,
            5: source_video_5,
            6: source_video_6,
            7: source_video_7,
            8: source_video_8,
        }
        refs = {
            1: reference_1,
            2: reference_2,
            3: reference_3,
            4: reference_4,
            5: reference_5,
            6: reference_6,
            7: reference_7,
            8: reference_8,
        }
        src_idx = max(0, min(8, int(source_video_index)))
        primary_idx = max(1, min(8, int(primary_reference_index)))
        secondary_idx = max(0, min(8, int(secondary_reference_index)))
        source = videos.get(src_idx) if src_idx > 0 else None
        primary = refs.get(primary_idx)
        secondary = refs.get(secondary_idx) if secondary_idx > 0 else None

        source_label = f"source_video_{src_idx}" if source is not None else "none"
        if primary is None:
            fallback_ref_idx, primary = self._first_connected(refs)
            primary_idx = fallback_ref_idx
        if primary is None:
            raise ValueError("CinematicV2VAssetSelector: no primary reference connected")
        if secondary is None:
            secondary = primary
            secondary_idx = primary_idx

        if source is None and str(fallback) == "first_connected":
            fallback_src_idx, source = self._first_connected(videos)
            src_idx = fallback_src_idx
            source_label = f"source_video_{src_idx}" if source is not None else "none"
        if source is None and str(fallback) == "primary_reference_as_video":
            source = primary
            source_label = "primary_reference_as_video"
        if source is None:
            raise ValueError(f"CinematicV2VAssetSelector: no source video connected for index={src_idx}")

        count = self._count(source)
        report = f"v2v_asset_selector source={source_label} frames={count} primary_ref={primary_idx} secondary_ref={secondary_idx}"
        return (source, primary, secondary, int(count), report)


NODE_CLASS_MAPPINGS = {
    "IAMCCS_LTX2_CinematicShotPlanner": IAMCCS_LTX2_CinematicShotPlanner,
    "IAMCCS_LTX2_CinematicRefLatentControl": IAMCCS_LTX2_CinematicRefLatentControl,
    "IAMCCS_LTX2_AudioPromptDirector": IAMCCS_LTX2_AudioPromptDirector,
    "IAMCCS_LTX2_CinematicPromptRelayAdapter": IAMCCS_LTX2_CinematicPromptRelayAdapter,
    "IAMCCS_LTX2_CinematicPromptComposer": IAMCCS_LTX2_CinematicPromptComposer,
    "IAMCCS_LTX2_CinematicShotLineBuilder": IAMCCS_LTX2_CinematicShotLineBuilder,
    "IAMCCS_LTX2_CinematicV2VTimelineLineBuilder": IAMCCS_LTX2_CinematicV2VTimelineLineBuilder,
    "IAMCCS_LTX2_CinematicLineStacker": IAMCCS_LTX2_CinematicLineStacker,
    "IAMCCS_LTX2_CinematicMultiGenPlanner": IAMCCS_LTX2_CinematicMultiGenPlanner,
    "IAMCCS_LTX2_CinematicShotAudioSelector": IAMCCS_LTX2_CinematicShotAudioSelector,
    "IAMCCS_LTX2_CinematicV2VTimelinePlanner": IAMCCS_LTX2_CinematicV2VTimelinePlanner,
    "IAMCCS_LTX2_CinematicV2VAssetSelector": IAMCCS_LTX2_CinematicV2VAssetSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_LTX2_CinematicShotPlanner": "LTX-2 Cinematic Shot Planner",
    "IAMCCS_LTX2_CinematicRefLatentControl": "LTX-2 Cinematic Ref -> Latent",
    "IAMCCS_LTX2_AudioPromptDirector": "LTX-2 Audio Prompt Director",
    "IAMCCS_LTX2_CinematicPromptRelayAdapter": "LTX-2 Cinematic PromptRelay Adapter",
    "IAMCCS_LTX2_CinematicPromptComposer": "LTX-2 Cinematic Prompt Composer",
    "IAMCCS_LTX2_CinematicShotLineBuilder": "LTX-2 Cinematic Shot Line Builder",
    "IAMCCS_LTX2_CinematicV2VTimelineLineBuilder": "LTX-2 Cinematic V2V Line Builder",
    "IAMCCS_LTX2_CinematicLineStacker": "LTX-2 Cinematic Line Stacker",
    "IAMCCS_LTX2_CinematicMultiGenPlanner": "LTX-2 Cinematic MultiGen Planner",
    "IAMCCS_LTX2_CinematicShotAudioSelector": "LTX-2 Cinematic Shot Audio Selector",
    "IAMCCS_LTX2_CinematicV2VTimelinePlanner": "LTX-2 Cinematic V2V Timeline Planner",
    "IAMCCS_LTX2_CinematicV2VAssetSelector": "LTX-2 Cinematic V2V Asset Selector",
}
