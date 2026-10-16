"""Local video sources and sampled visual evidence for H3 editing."""
from __future__ import annotations

import base64
import os
from pathlib import Path

import cv2
import folder_paths


def local_video_path(value):
    path = str(value or "").strip().strip('"')
    candidates = [Path(path)] if os.path.isabs(path) else [
        Path(folder_paths.get_input_directory()) / path,
        Path(folder_paths.get_output_directory()) / path,
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
            return str(candidate.resolve())
    raise ValueError("H3 edit video is unavailable: " + path)


def select_video(prompter_video, backend_video, policy="auto"):
    if policy not in {"auto", "prompter", "backend"}:
        raise ValueError("Unknown H3 edit video source: " + str(policy))
    selected = prompter_video if policy == "prompter" or (policy == "auto" and prompter_video) else backend_video
    if not selected:
        raise ValueError("Add a source video in Prompter or the H3 VFX backend input.")
    selected = dict(selected) if isinstance(selected, dict) else {"path": selected}
    selected["path"] = local_video_path(selected.get("path"))
    if selected.get("skip_first_frames"):
        capture = cv2.VideoCapture(selected["path"])
        try:
            source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        finally:
            capture.release()
        selected["start_seconds"] = float(selected.get("start_seconds", 0) or 0) + float(selected["skip_first_frames"]) / float(selected.get("force_rate") or source_fps)
        selected.pop("skip_first_frames", None)
    selected["start_seconds"] = max(0.0, float(selected.get("start_seconds", 0) or 0))
    selected["duration_seconds"] = max(0.0, float(selected.get("duration_seconds", 0) or 0))
    return selected


def video_metadata(path):
    capture = cv2.VideoCapture(local_video_path(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frames < 1:
            raise ValueError("Unable to read video duration/FPS")
        return {"fps": fps, "frame_count": frames, "duration_seconds": frames / fps}
    finally:
        capture.release()


def video_evidence(descriptor, count=6):
    selected = select_video(descriptor, None, "prompter")
    capture = cv2.VideoCapture(selected["path"])
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count < 1:
            raise ValueError("Unable to read video duration/FPS")
        start = selected["start_seconds"]
        available = frame_count / fps - start
        duration = min(available, selected["duration_seconds"] or available)
        if duration <= 0:
            raise ValueError("The selected video trim is empty")
        images = []
        for index in range(max(2, min(8, int(count)))):
            seconds = start + max(0, duration - 1 / fps) * index / (max(2, min(8, int(count))) - 1)
            capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Could not decode the source video at {seconds:.3f}s")
            height, width = frame.shape[:2]
            scale = min(1.0, 768 / max(width, height))
            frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise ValueError("Could not prepare video evidence")
            images.append({"slot": f"Video 1 frame {index + 1}", "name": f"Video 1 at {seconds - start:.3f}s",
                           "role": "source_video_frame", "mime_type": "image/jpeg",
                           "data": base64.b64encode(encoded).decode("ascii")})
        return images, {"path": selected["path"], "start_seconds": start, "duration_seconds": duration,
                        "fps": fps, "sampled_frames": len(images), "temporal_precision": "sampled visual evidence; audio not transcribed"}
    finally:
        capture.release()


def load_edit_video(descriptor, width=0, height=0, duration=0):
    import nodes

    selected = select_video(descriptor, None, "prompter")
    seconds = selected["duration_seconds"] or float(duration or 0)
    capture = cv2.VideoCapture(selected["path"])
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if fps <= 0 or selected["start_seconds"] >= total / fps:
        raise ValueError("H3 edit source has an invalid duration or trim")
    available = total / fps - selected["start_seconds"]
    seconds = min(seconds or available, available)
    if width and height and source_width and source_height:
        scale = min(width / source_width, height / source_height, 1.0)
        width, height = max(8, round(source_width * scale / 8) * 8), max(8, round(source_height * scale / 8) * 8)
    loaded = nodes.NODE_CLASS_MAPPINGS["VHS_LoadVideoPath"]().load_video(
        video=selected["path"], force_rate=24, custom_width=int(width), custom_height=int(height),
        frame_load_cap=max(1, round(seconds * 24)), skip_first_frames=round(selected["start_seconds"] * 24),
        select_every_nth=1, format="None")
    if isinstance(loaded, dict):
        loaded = loaded["result"]
    frames, _, audio, _ = loaded
    from .iamccs_h3_swap_video import optional_video_audio
    audio = optional_video_audio(audio)
    return {"video": frames, "audio": audio, "fps": 24.0, "descriptor": selected}
