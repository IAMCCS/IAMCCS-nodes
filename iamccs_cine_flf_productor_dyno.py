from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict

import torch
from comfy_extras.nodes_lt import LTXVAddGuide

try:
    from comfy_extras.nodes_lt import _append_guide_attention_entry
except Exception:  # pragma: no cover - depends on ComfyUI/KJ/LTX version
    _append_guide_attention_entry = None

from .iamccs_cine_nodes import (
    IAMCCS_CineFLFProductor,
    _clamp,
    _json_report,
    _safe_bool,
    _safe_float,
    _safe_int,
)


class IAMCCS_CineFLFProductorDyno(IAMCCS_CineFLFProductor):
    """Dynamic guide Productor variant.

    The normal Productor keeps the Filmmaker V3 clean guide path for maximum
    image lock. Dyno keeps the same inputs/outputs and guide plan contract, but
    also appends the legacy guide attention entry used by the older dynamic
    FLF path. This makes it safe to A/B test motion-heavy shots without
    changing existing workflows.
    """

    CATEGORY = "IAMCCS/Cine/02 Single Generation"

    @classmethod
    def INPUT_TYPES(cls):
        schema = deepcopy(IAMCCS_CineFLFProductor.INPUT_TYPES())
        schema.setdefault("optional", {})
        schema["optional"]["guide_attention_mode"] = (
            ["skip_step_transitions", "all_guides", "off"],
            {
                "default": "skip_step_transitions",
                "tooltip": (
                    "Controls Dyno guide_attention_entries. "
                    "skip_step_transitions keeps step-transition pairs on the clean Production guide path."
                ),
            },
        )
        return schema

    @staticmethod
    def _normalise_attention_mode(value: Any) -> str:
        mode = str(value or "skip_step_transitions").strip().lower()
        if mode in {"all", "all_guides", "on", "true"}:
            return "all_guides"
        if mode in {"off", "none", "false", "0"}:
            return "off"
        return "skip_step_transitions"

    @staticmethod
    def _apply_attention_entry(positive, negative, encoded, strength: float):
        if _append_guide_attention_entry is None:
            return positive, negative, False, "guide attention helper unavailable"
        pre_filter_count = int(encoded.shape[2] * encoded.shape[3] * encoded.shape[4])
        guide_latent_shape = [int(v) for v in encoded.shape[2:]]
        positive, negative = _append_guide_attention_entry(
            positive,
            negative,
            pre_filter_count,
            guide_latent_shape,
            strength=float(strength),
        )
        return positive, negative, True, ""

    @staticmethod
    def _execute_guide_data(
        positive,
        negative,
        vae,
        latent,
        guide_data: Dict[str, Any],
        strength_scale: float,
        tail_safety_frames: int,
        guide_attention_mode: str = "skip_step_transitions",
    ):
        latent_samples = latent.get("samples") if isinstance(latent, dict) else None
        if not torch.is_tensor(latent_samples) or latent_samples.ndim != 5:
            return positive, negative, latent, {
                "applied_guides": [],
                "skipped_guides": [{"reason": "invalid latent samples"}],
                "latent_pixel_frames": 0,
                "guide_attention_available": _append_guide_attention_entry is not None,
                "guide_attention_entries": False,
                "guide_attention_entry_count": 0,
            }

        scale_factors = vae.downscale_index_formula
        latent_image = latent_samples.clone()
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"].clone()
        else:
            batch, _, latent_frames, _, _ = latent_image.shape
            noise_mask = torch.ones((batch, 1, latent_frames, 1, 1), dtype=torch.float32, device=latent_image.device)

        _, _, latent_length, latent_height, latent_width = latent_image.shape
        time_scale_factor = int(scale_factors[0]) if scale_factors else 8
        latent_pixel_frames = max(1, (int(latent_length) - 1) * max(1, time_scale_factor) + 1)
        max_frame = max(0, int(latent_pixel_frames) - 1 - max(0, int(tail_safety_frames)))
        images = guide_data.get("images", []) if isinstance(guide_data, dict) else []
        insert_frames = guide_data.get("insert_frames", []) if isinstance(guide_data, dict) else []
        strengths = guide_data.get("strengths", []) if isinstance(guide_data, dict) else []
        labels = guide_data.get("labels", []) if isinstance(guide_data, dict) else []
        refs = guide_data.get("reference_indices", []) if isinstance(guide_data, dict) else []
        step_sources = guide_data.get("step_transition_sources", []) if isinstance(guide_data, dict) else []
        step_targets = guide_data.get("step_transition_targets", []) if isinstance(guide_data, dict) else []
        step_protected = guide_data.get("step_transition_protected", []) if isinstance(guide_data, dict) else []
        attention_mode = IAMCCS_CineFLFProductorDyno._normalise_attention_mode(guide_attention_mode)

        applied = []
        skipped = []
        attention_entries = 0
        attention_skipped = []
        for idx, img in enumerate(images):
            label = str(labels[idx]) if idx < len(labels) else f"guide_{idx + 1}"
            ref = _safe_int(refs[idx], idx + 1) if idx < len(refs) else idx + 1
            if not torch.is_tensor(img) or img.ndim != 4 or int(img.shape[0]) <= 0:
                skipped.append({"label": label, "reference_index": int(ref), "reason": "empty guide image tensor"})
                continue
            requested_frame = _safe_int(insert_frames[idx], 0) if idx < len(insert_frames) else 0
            frame_idx = requested_frame if requested_frame < 0 else min(max(0, requested_frame), max_frame)
            strength = _clamp((_safe_float(strengths[idx], 1.0) if idx < len(strengths) else 1.0) * float(strength_scale), 0.0, 1.0, 1.0)
            if strength <= 0:
                skipped.append({"label": label, "reference_index": int(ref), "frame": int(frame_idx), "reason": "zero strength"})
                continue
            try:
                image_1, encoded = LTXVAddGuide.encode(vae, latent_width, latent_height, img, scale_factors)
                conditioning_frame, latent_idx = LTXVAddGuide.get_latent_index(positive, latent_length, len(image_1), frame_idx, scale_factors)
                if latent_idx + encoded.shape[2] > latent_length:
                    skipped.append({
                        "label": label,
                        "reference_index": int(ref),
                        "frame": int(frame_idx),
                        "reason": "guide exceeds latent length",
                    })
                    continue
                positive, negative, latent_image, noise_mask = LTXVAddGuide.append_keyframe(
                    positive,
                    negative,
                    conditioning_frame,
                    latent_image,
                    noise_mask,
                    encoded,
                    float(strength),
                    scale_factors,
                )
                is_step_source = _safe_bool(step_sources[idx], False) if idx < len(step_sources) else False
                is_step_target = _safe_bool(step_targets[idx], False) if idx < len(step_targets) else False
                if idx > 0 and idx - 1 < len(step_sources):
                    is_step_target = bool(is_step_target or _safe_bool(step_sources[idx - 1], False))
                is_step_protected = _safe_bool(step_protected[idx], False) if idx < len(step_protected) else False
                is_step_protected = bool(is_step_protected or is_step_source or is_step_target)
                if attention_mode == "off":
                    did_attention = False
                    attention_reason = "guide_attention_mode=off"
                elif attention_mode == "skip_step_transitions" and is_step_protected:
                    positive, negative, did_attention, attention_reason = IAMCCS_CineFLFProductorDyno._apply_attention_entry(
                        positive,
                        negative,
                        encoded,
                        0.0,
                    )
                    if did_attention:
                        attention_reason = "step transition protected; guide_attention_entry kept with zero strength for LTX mask alignment"
                else:
                    positive, negative, did_attention, attention_reason = IAMCCS_CineFLFProductorDyno._apply_attention_entry(
                        positive,
                        negative,
                        encoded,
                        float(strength),
                    )
                if did_attention:
                    attention_entries += 1
                elif attention_reason:
                    attention_skipped.append({
                        "label": label,
                        "reference_index": int(ref),
                        "reason": attention_reason,
                        "step_transition_protected": bool(is_step_protected),
                    })
            except Exception as exc:
                skipped.append({"label": label, "reference_index": int(ref), "frame": int(frame_idx), "reason": f"dyno guide apply failed: {exc}"})
                continue
            applied.append({
                "label": label,
                "reference_index": int(ref),
                "requested_frame": int(requested_frame),
                "frame": int(frame_idx),
                "strength": float(strength),
                "guide_attention_entry": did_attention,
                "guide_attention_strength": 0.0 if (attention_mode == "skip_step_transitions" and is_step_protected) else float(strength) if did_attention else None,
                "step_transition_protected": bool(is_step_protected),
            })

        return positive, negative, {"samples": latent_image, "noise_mask": noise_mask}, {
            "applied_guides": applied,
            "skipped_guides": skipped,
            "attention_skipped": attention_skipped,
            "latent_pixel_frames": int(latent_pixel_frames),
            "guide_attention_available": _append_guide_attention_entry is not None,
            "guide_attention_entries": attention_entries > 0,
            "guide_attention_entry_count": int(attention_entries),
            "guide_attention_mode": attention_mode,
            "guide_strength_semantics": "append_keyframe_plus_legacy_guide_attention_entry",
            "compatibility_mode": "dyno_motion_conditioning_guide",
        }

    def execute(
        self,
        positive,
        negative,
        vae,
        latent,
        multi_input,
        guide_plan_json,
        strength_scale,
        tail_safety_frames,
        timeline_data="",
        duration_seconds=20,
        frame_rate=24,
        guide_data=None,
        guide_attention_mode="skip_step_transitions",
    ):
        attention_mode = self._normalise_attention_mode(guide_attention_mode)
        if isinstance(guide_data, dict) and guide_data.get("images"):
            positive, negative, current_latent, guide_data_report = self._execute_guide_data(
                positive,
                negative,
                vae,
                latent,
                guide_data,
                float(strength_scale),
                int(tail_safety_frames),
                attention_mode,
            )
            report = _json_report({
                "node": "IAMCCS_CineFLFProductorDyno",
                "mode": "cine_guide_data_compatible_dyno",
                "guide_count": len(guide_data.get("images", [])),
                "applied_count": len(guide_data_report.get("applied_guides", [])),
                "skipped_count": len(guide_data_report.get("skipped_guides", [])),
                **guide_data_report,
                "truth": (
                    "Dyno consumed GUIDE_DATA and appended guide_attention_entries according to guide_attention_mode. "
                    "Step-transition-protected guides stay on the clean Production guide path by default."
                ),
            })
            return positive, negative, current_latent, report

        plan = self._parse_plan(guide_plan_json, timeline_data, duration_seconds, frame_rate)
        clean_guide_data = self._guide_data_from_plan_and_images(plan, multi_input, float(strength_scale))
        positive, negative, current_latent, guide_data_report = self._execute_guide_data(
            positive,
            negative,
            vae,
            latent,
            clean_guide_data,
            1.0,
            int(tail_safety_frames),
            attention_mode,
        )
        report = _json_report({
            "node": "IAMCCS_CineFLFProductorDyno",
            "mode": "explicit_dyno_guide_productor",
            "plan_source": plan.get("source", "") if isinstance(plan, dict) else "",
            "reference_count": int(multi_input.shape[0]) if torch.is_tensor(multi_input) and multi_input.ndim == 4 else 0,
            "guide_count": len(clean_guide_data.get("images", [])),
            "applied_count": len(guide_data_report.get("applied_guides", [])),
            "skipped_count": len(guide_data_report.get("skipped_guides", [])),
            **guide_data_report,
            "truth": (
                "Dyno uses the same guide plan and image slots as CineFLFProductor, then conditionally appends "
                "legacy guide_attention_entries. Default skip_step_transitions protects dolly/bridge pairs."
            ),
        })
        return positive, negative, current_latent, report
