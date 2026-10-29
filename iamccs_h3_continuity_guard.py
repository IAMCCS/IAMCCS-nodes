# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only continuation diagnostics. Reports are evidence, not identity scores."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import shutil
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def settings_schema():
    # Retired fields keep their positions so existing workflows still load.
    retired = "Retired compatibility field. Ignored; no repair or reference injection runs."
    return {
        "h3_guard_mode": (["off", "monitor", "repair"], {"default": "off", "display_name": "GUARD - MODE", "tooltip": "Read-only Extended AV diagnostics. Legacy repair values resolve to Off with a warning."}),
        "h3_guard_reference_memory": ("BOOLEAN", {"default": True, "display_name": "RETIRED - REFERENCE MEMORY", "tooltip": retired}),
        "h3_guard_repair_denoise": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.35, "step": 0.01, "display_name": "RETIRED - REPAIR DENOISE", "tooltip": retired}),
        "h3_guard_repair_steps": ("INT", {"default": 6, "min": 1, "max": 30, "display_name": "RETIRED - REPAIR STEPS", "tooltip": retired}),
        "h3_guard_detail_ratio": ("FLOAT", {"default": 0.65, "min": 0.1, "max": 1.0, "step": 0.01, "display_name": "GUARD - DETAIL LOSS THRESHOLD", "tooltip": "Diagnostic heuristic only. Camera, focus and motion changes can cause false alarms."}),
        "h3_guard_motion_tolerance": ("FLOAT", {"default": 0.20, "min": 0.01, "max": 1.0, "step": 0.01, "display_name": "RETIRED - MOTION CHANGE LIMIT", "tooltip": retired}),
    }



def effective_mode(config):
    mode = config.get("mode", "off")
    if mode == "repair":
        logging.getLogger(__name__).warning(
            "Continuity Guard Repair has been retired after unsuccessful validation; using Off. "
            "Select Monitor for read-only diagnostics.")
        return "off"
    if mode not in {"off", "monitor"}:
        raise ValueError("Unknown Continuity Guard mode")
    return mode


def configuration(settings):
    mode = effective_mode({"mode": settings.get("h3_guard_mode", "off")})
    threshold = float(settings.get("h3_guard_detail_ratio", 0.65))
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Invalid Continuity Guard detail ratio")
    return {"mode": mode, "detail_ratio": threshold}


def storage(output_directory, render_id):
    if not render_id:
        raise ValueError("Continuity Guard requires the R49 render identity")
    key = hashlib.sha256(str(render_id).encode()).hexdigest()[:24]
    path = Path(output_directory) / "IAMCCS" / "continuity_guard" / key
    path.mkdir(parents=True, exist_ok=True)
    return path


def conditioning_identity(render_id, index):
    # R49 assigns the definitive, numbered name at the first native checkpoint.
    if int(index) == 0:
        return uuid.uuid4().hex
    if not str(render_id or "").strip():
        raise ValueError("Guard continuation requires its predecessor's checkpoint render identity")
    return str(render_id)


def bind_checkpoint(output_directory, report, render_id):
    """Bind the first segment's provisional measurements to R49's final name."""
    key = report["storage_key"]
    if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{24}", key):
        raise ValueError("Invalid Continuity Guard storage key")
    source = Path(output_directory) / "IAMCCS" / "continuity_guard" / key
    destination = storage(output_directory, render_id)
    if source == destination:
        return
    if int(report["segment_index"]) != 0:
        raise ValueError("Guard continuation measurements do not match the checkpoint render identity")
    for name in ("baseline.json", "segment_0000.json"):
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)
    if not (destination / "baseline.json").is_file():
        raise ValueError("Guard first-segment baseline is missing before checkpoint binding")
    write_report(destination / "checkpoint_identity.json", {"render_id": str(render_id), "provisional_storage_key": key})


def write_report(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def analysis_frames(images):
    # One analysis size for all segments; at most 48 frames, with uniform timing.
    ids = torch.linspace(0, images.shape[0] - 1, min(48, images.shape[0]), device=images.device).round().long()
    x = images[ids, ..., :3].movedim(-1, 1).float()
    x = F.interpolate(x, size=(144, 256), mode="area")
    return (x[:, 0] * 0.2126 + x[:, 1] * 0.7152 + x[:, 2] * 0.0722).clamp(0, 1).cpu().numpy()


def measure(images):
    import cv2
    gray = analysis_frames(images)
    detail = [float(np.mean(np.abs(cv2.Laplacian(f, cv2.CV_32F)))) for f in gray]
    light = gray.mean(axis=(1, 2))
    intervals = max(1, images.shape[0] - 1) / max(1, len(gray) - 1)
    flow = []
    for a, b in zip(gray[:-1], gray[1:]):
        field = cv2.calcOpticalFlowFarneback((a * 255).astype('uint8'), (b * 255).astype('uint8'), None, .5, 3, 15, 3, 5, 1.2, 0)
        flow.append(float(np.median(np.linalg.norm(field, axis=-1))) / intervals)
    # Native-resolution tiles prevent large textured regions from hiding local blur.
    ids = torch.linspace(0, images.shape[0]-1, min(24, images.shape[0]), device=images.device).long()
    native = images[ids, ..., :3].float().mean(-1).cpu().numpy()
    tiles = []
    for row in range(4):
        for col in range(4):
            patch = native[:, row*native.shape[1]//4:(row+1)*native.shape[1]//4,
                           col*native.shape[2]//4:(col+1)*native.shape[2]//4]
            values = [float(np.mean(abs(cv2.Laplacian(f, cv2.CV_32F)))) for f in patch]
            tiles.append(float(np.median(values)))
    return {"local_detail": tiles, "detail": float(np.median(detail)), "luminance": float(np.median(light)),
            "flicker": float(np.median(np.abs(np.diff(light, n=2)))) if len(light) > 2 else 0.0,
            "motion": float(np.median(flow)) if flow else 0.0,
            "frames": int(images.shape[0]), "analysis": "luma_texture_optical_flow_v1",
            "limitations": "No semantic identity, intent or lip-sync certification."}


def assess(current, baseline, config):
    ratio = current["detail"] / max(baseline["detail"], 1e-6)
    exposure = abs(current["luminance"] - baseline["luminance"])
    moving = current["motion"] / max(baseline["motion"], 0.02)
    uncertain = exposure > 0.15 or moving < 0.4 or moving > 2.5
    local_before = baseline.get("local_detail", [])
    local_now = current.get("local_detail", [])
    local_ratios = [b/max(a, 1e-6) for a,b in zip(local_before, local_now)]
    # 10% local texture loss is a candidate trigger, never proof of degradation.
    affected = [i for i,r in enumerate(local_ratios) if r < .90 and local_before[i] > .005]
    loss = ratio < config["detail_ratio"] or bool(affected)
    return {"detail_ratio": ratio, "local_detail_ratios": local_ratios, "affected_tiles": affected, "exposure_change": exposure, "motion_ratio": moving,
            "texture_loss_detected": bool(loss and not uncertain),
            "review_recommended": bool(uncertain or loss)}


def process(config, sampled, raw_frames, *, directory, index, head_frames, visible_frames):
    """Measure decoded frames without modifying video, audio or conditioning."""
    if effective_mode(config) != "monitor":
        return sampled, raw_frames, {"status": "off", "repair_attempts": 0}

    visible = raw_frames[head_frames:head_frames + visible_frames]
    current = measure(visible)
    baseline_path = directory / "baseline.json"
    report = {"schema": "iamccs.guard.report.v1", "segment_index": index,
              "storage_key": directory.name,
              "mode": config["mode"], "before": current, "repair_attempts": 0,
              "audio_policy": "Original sampled audio preserved exactly", "status": "monitored"}
    if index == 0:
        write_report(baseline_path, current)
        report["status"] = "baseline_recorded"
    elif not baseline_path.is_file():
        raise ValueError("Guard baseline missing. Start this render from its first segment.")
    else:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        verdict = assess(current, baseline, config)
        report["assessment"] = verdict
        if verdict["review_recommended"]:
            report["status"] = "review_recommended"
    write_report(directory / f"segment_{index:04d}.json", report)
    sampled = {**sampled, "_iamccs_continuity_guard": report}
    return sampled, raw_frames, report
