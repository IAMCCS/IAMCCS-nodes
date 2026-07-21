from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
import time
from fractions import Fraction
from typing import Any, Dict, List, Tuple

import torch
import torchaudio
from comfy_api.latest import InputImpl, Types

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional preview helper.
    Image = None

try:
    import folder_paths  # type: ignore
except Exception:  # pragma: no cover - ComfyUI provides this at runtime.
    folder_paths = None


SUPERNODE_LINX_TYPE = "IAMCCS_SUPERNODE_LINX"
MAX_TRACK_OUTS = 5
_VIDEO_TAKE_REGISTRY: Dict[str, Dict[int, Dict[str, Any]]] = {}
_VIDEO_EDITOR_MANIFEST_REGISTRY: Dict[str, Dict[str, Any]] = {}


def _safe_slug(value: Any, fallback: str = "shotboard_video_editor_live") -> str:
    text = str(value or "").strip() or fallback
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._-") or fallback


def _ltx_rounded_pixel_frames(frames: Any) -> int:
    value = max(1, _safe_int(frames, 1))
    return max(1, int(math.ceil(max(0, value - 1) / 8.0) * 8 + 1))


def _parking_root(session_key: Any) -> str:
    session = _safe_slug(session_key)
    roots: List[str] = []
    if folder_paths is not None:
        for getter_name in ("get_output_directory", "get_temp_directory"):
            getter = getattr(folder_paths, getter_name, None)
            if callable(getter):
                try:
                    roots.append(str(getter()))
                except Exception:
                    pass
    root = roots[0] if roots else os.path.join(os.getcwd(), "output")
    path = os.path.abspath(os.path.join(root, "IAMCCS_video_editor_parking", session))
    os.makedirs(path, exist_ok=True)
    return path


def _tensor_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().contiguous()
    return value


def _audio_to_cpu(audio: Any) -> Any:
    if not isinstance(audio, dict):
        return None
    waveform = audio.get("waveform")
    if waveform is None:
        return None
    return {
        "waveform": _tensor_to_cpu(waveform),
        "sample_rate": int(audio.get("sample_rate") or 44100),
    }


def _images_to_parking_uint8(images: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(images):
        raise ValueError("IAMCCS Shotboard Video Editor: cannot park video without an image tensor.")
    if images.dtype == torch.uint8:
        return images.detach().cpu().contiguous()
    return torch.clamp(images.detach().float().cpu(), 0.0, 1.0).mul(255.0).round().to(torch.uint8).contiguous()


def _images_from_parking_tensor(images: torch.Tensor, storage: Any = "") -> torch.Tensor:
    if not torch.is_tensor(images):
        raise ValueError("IAMCCS Shotboard Video Editor: parked video images are not a tensor.")
    if images.dtype == torch.uint8 or str(storage) == "uint8_0_255":
        return images.float().div_(255.0).contiguous()
    return images.float().contiguous()


def _ensure_disk_space_for_file(path: str, expected_bytes: int, label: str) -> None:
    try:
        usage = shutil.disk_usage(os.path.dirname(os.path.abspath(path)))
        # Keep a small reserve for zip metadata, previews, and ComfyUI bookkeeping.
        required = int(expected_bytes * 1.15) + 128 * 1024 * 1024
        if usage.free < required:
            free_gb = usage.free / (1024 ** 3)
            need_gb = required / (1024 ** 3)
            raise RuntimeError(
                f"IAMCCS Shotboard Video Editor: not enough free disk space to park {label}. "
                f"Need about {need_gb:.2f} GB, available {free_gb:.2f} GB. "
                "Clear ComfyUI output/temp or choose a drive with more space."
            )
    except RuntimeError:
        raise
    except Exception:
        pass


def _output_dir() -> str:
    if folder_paths is not None:
        getter = getattr(folder_paths, "get_output_directory", None)
        if callable(getter):
            try:
                return os.path.abspath(str(getter()))
            except Exception:
                pass
    return os.path.abspath(os.path.join(os.getcwd(), "output"))


def _preview_video_metadata(path: str, root: str, fps: float) -> Dict[str, Any]:
    out_dir = _output_dir()
    rel_dir = os.path.relpath(root, out_dir)
    if rel_dir == ".":
        rel_dir = ""
    name = os.path.basename(path)
    return {
        "preview_video": {
            "filename": name,
            "subfolder": rel_dir.replace("\\", "/"),
            "type": "output",
            "fps": float(fps),
            "schema": 2,
            "codec": "h264_yuv420p",
        },
        "preview_video_file": name,
        "preview_video_path": path,
        "preview_video_subfolder": rel_dir.replace("\\", "/"),
        "preview_video_type": "output",
        "preview_video_fps": float(fps),
        "preview_video_schema": 2,
        "preview_video_codec": "h264_yuv420p",
    }


def _write_video_take_preview(images: Any, path: str, fps: float) -> Dict[str, Any]:
    """Write a lightweight H.264 monitor proxy without changing the parked take."""
    if not torch.is_tensor(images) or images.ndim < 4 or int(images.shape[0]) <= 0:
        return {"preview_video_error": "No video frames available for preview."}
    try:
        if os.path.isfile(path) and os.path.getsize(path) > 1024:
            return {"preview_video_path": path}
        import imageio_ffmpeg  # type: ignore

        first = images[0].detach()
        if first.ndim == 3 and first.shape[0] in (1, 3, 4) and first.shape[-1] not in (1, 3, 4):
            first = first.permute(1, 2, 0)
        if first.ndim != 3:
            raise ValueError("Preview frame has no RGB dimensions.")
        height = int(first.shape[0])
        width = int(first.shape[1])
        if width <= 0 or height <= 0:
            raise ValueError("Preview frame dimensions are invalid.")
        preview_fps = max(1.0, min(30.0, float(fps or 24.0)))
        # Chromium/Electron does not reliably decode the RGB/4:4:4 output
        # ffmpeg may infer from tensor frames. Force the browser-safe H.264
        # 4:2:0 profile used only for the monitor proxy.
        output_params = ["-movflags", "+faststart", "-preset", "veryfast", "-pix_fmt", "yuv420p"]
        if width > 960:
            output_params.extend(["-vf", "scale=960:-2"])
        writer = imageio_ffmpeg.write_frames(
            path,
            (width, height),
            fps=preview_fps,
            codec="libx264",
            quality=7,
            macro_block_size=2,
            ffmpeg_log_level="error",
            output_params=output_params,
        )
        writer.send(None)
        try:
            for index in range(int(images.shape[0])):
                frame = images[index].detach()
                if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
                    frame = frame.permute(1, 2, 0)
                if frame.shape[-1] == 1:
                    frame = frame.repeat(1, 1, 3)
                elif frame.shape[-1] > 3:
                    frame = frame[..., :3]
                if frame.dtype != torch.uint8:
                    frame = torch.clamp(frame.float(), 0.0, 1.0).mul(255.0).round().to(torch.uint8)
                writer.send(frame.contiguous().cpu().numpy().tobytes())
        finally:
            writer.close()
        if not os.path.isfile(path) or os.path.getsize(path) <= 1024:
            raise RuntimeError("FFmpeg did not create a readable preview video.")
        return {"preview_video_path": path, "preview_video_fps": preview_fps}
    except Exception as exc:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
        return {"preview_video_error": str(exc)}


def ensure_parked_take_preview_video(parking_path: Any) -> Dict[str, Any]:
    """Materialize an H.264 monitor proxy for a previously parked take on demand."""
    path = os.path.abspath(str(parking_path or ""))
    if not path or not os.path.isfile(path):
        raise FileNotFoundError("Parked take file was not found.")
    if not path.lower().endswith(".iamccs_take.pt"):
        raise ValueError("Preview source must be an IAMCCS parked take.")
    root = os.path.dirname(path)
    preview_path = os.path.splitext(path)[0] + "_preview_v2.mp4"
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict) or not torch.is_tensor(data.get("images")):
        raise ValueError("Parked take contains no readable video frames.")
    fps = max(1.0, float(data.get("fps") or 24.0))
    result = _write_video_take_preview(
        _images_from_parking_tensor(data["images"], data.get("image_storage")),
        preview_path,
        fps,
    )
    if result.get("preview_video_error"):
        raise RuntimeError(str(result["preview_video_error"]))
    return _preview_video_metadata(preview_path, root, float(result.get("preview_video_fps") or fps))


def _save_video_take_preview(session_key: Any, slot: int, images: Any, root: str, fps: float = 24.0, preview_stem: str = "") -> Dict[str, Any]:
    if not torch.is_tensor(images) or images.ndim < 4 or int(images.shape[0]) <= 0:
        return {}
    total_frames = int(images.shape[0])
    stamp = int(time.time() * 1000)
    stem = _safe_slug(preview_stem or f"T{int(slot):02d}_A{int(slot):02d}_{stamp}")
    preview_path = os.path.join(root, f"{stem}_preview_v2.mp4")
    preview = _write_video_take_preview(images, preview_path, fps)
    output = _preview_video_metadata(preview_path, root, float(preview.get("preview_video_fps") or fps)) if preview.get("preview_video_path") else {}
    if preview.get("preview_video_error"):
        output["preview_video_error"] = preview["preview_video_error"]
    if Image is None:
        return output
    try:
        preview_items: List[Dict[str, Any]] = []
        out_dir = _output_dir()
        rel_dir = os.path.relpath(root, out_dir)
        if rel_dir == ".":
            rel_dir = ""
        sample_count = min(16, max(1, total_frames))
        sample_indexes = sorted(set(int(round(i * (total_frames - 1) / max(1, sample_count - 1))) for i in range(sample_count)))
        first_name = ""
        for frame_index in sample_indexes:
            frame = images[frame_index].detach().cpu().float()
            if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
                frame = frame.permute(1, 2, 0)
            if frame.shape[-1] > 3:
                frame = frame[..., :3]
            frame = frame.clamp(0, 1)
            array = (frame.numpy() * 255.0).round().astype("uint8")
            preview_name = f"T{int(slot):02d}_A{int(slot):02d}_{stamp}_preview_{int(frame_index):05d}.png"
            preview_image_path = os.path.join(root, preview_name)
            Image.fromarray(array).save(preview_image_path)
            if not first_name:
                first_name = preview_name
            preview_items.append({
                "filename": preview_name,
                "frame": int(frame_index),
                "subfolder": rel_dir.replace("\\", "/"),
                "type": "output",
            })
        output.update({
            "preview_image": first_name,
            "preview_image_file": first_name,
            "preview_image_path": os.path.join(root, first_name) if first_name else "",
            "preview_subfolder": rel_dir.replace("\\", "/"),
            "preview_type": "output",
            "preview_strip": preview_items,
        })
    except Exception as exc:
        output["preview_error"] = str(exc)
    return output


def _safe_json_loads(value: Any, fallback: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        text = str(value or "").strip()
        if not text:
            return fallback
        return json.loads(text)
    except Exception:
        return fallback


def _json_dump(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def _clone_linx(cine_linx: Any, mode: str = "iamccs_multigeneration") -> Dict[str, Any]:
    if isinstance(cine_linx, dict):
        return copy.deepcopy(cine_linx)
    return {
        "type": SUPERNODE_LINX_TYPE,
        "mode": mode,
        "resources": {},
        "outputs": {},
        "chain": [],
        "stages": [],
    }


def _resources(cine_linx: Dict[str, Any]) -> Dict[str, Any]:
    resources = cine_linx.setdefault("resources", {})
    if not isinstance(resources, dict):
        resources = {}
        cine_linx["resources"] = resources
    return resources


def _outputs(cine_linx: Dict[str, Any]) -> Dict[str, Any]:
    outputs = cine_linx.setdefault("outputs", {})
    if not isinstance(outputs, dict):
        outputs = {}
        cine_linx["outputs"] = outputs
    return outputs


def _payload(cine_linx: Dict[str, Any]) -> Dict[str, Any]:
    resources = _resources(cine_linx)
    payload = resources.get("cine_payload")
    if not isinstance(payload, dict):
        payload = {}
        resources["cine_payload"] = payload
    return payload


def _refresh_linx_index(cine_linx: Dict[str, Any]) -> None:
    resources = _resources(cine_linx)
    cine_linx["resource_keys"] = sorted(resources.keys())
    cine_linx["resource_types"] = {key: type(value).__name__ for key, value in resources.items()}


def _safe_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return int(fallback)


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(fallback)


def _seconds_for_template(template: str, custom_chunk_seconds: Any) -> float:
    normalized = str(template or "20s").strip().lower()
    if normalized == "custom":
        return max(1.0, _safe_float(custom_chunk_seconds, 20.0))
    if normalized.endswith("s"):
        normalized = normalized[:-1]
    return max(1.0, _safe_float(normalized, 20.0))


def _segments(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        seg = copy.deepcopy(item)
        seg["id"] = str(seg.get("id") or f"multi_src_{index + 1:03d}")
        seg["type"] = "audio"
        seg["start"] = max(0, _safe_int(seg.get("start", 0), 0))
        seg["length"] = max(1, _safe_int(seg.get("length", seg.get("audioDurationFrames", 1)), 1))
        seg["track"] = max(0, _safe_int(seg.get("track", 0), 0))
        seg["trimStart"] = max(0, _safe_int(seg.get("trimStart", 0), 0))
        seg["audioDurationFrames"] = max(seg["trimStart"] + seg["length"], _safe_int(seg.get("audioDurationFrames", seg["length"]), seg["length"]))
        out.append(seg)
    return sorted(out, key=lambda seg: (int(seg.get("start", 0)), int(seg.get("track", 0))))


def _max_end(segments: List[Dict[str, Any]]) -> int:
    return max([_safe_int(seg.get("start", 0), 0) + _safe_int(seg.get("length", 1), 1) for seg in segments] or [0])


def _safe_bool(value: Any, fallback: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(fallback)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled"}:
        return False
    return bool(fallback)


def _roll_contract_from_linx(cine_linx: Any, fps: float = 24.0) -> Dict[str, Any]:
    linx = cine_linx if isinstance(cine_linx, dict) else {}
    resources = linx.get("resources", {}) if isinstance(linx.get("resources", {}), dict) else {}
    payload = resources.get("cine_payload") if isinstance(resources.get("cine_payload"), dict) else {}
    candidates = [
        resources.get("cine_roll_contract"),
        resources.get("cine_audio_tracks", {}).get("roll_contract") if isinstance(resources.get("cine_audio_tracks"), dict) else None,
        payload.get("roll_contract"),
        payload.get("roll"),
    ]
    raw = next((item for item in candidates if isinstance(item, dict)), {})
    safe_fps = max(1.0, _safe_float(raw.get("frame_rate", fps), fps))
    enabled = _safe_bool(raw.get("enabled", False), False)
    seconds = max(0.0, min(30.0, _safe_float(raw.get("seconds", 1.0), 1.0)))
    frames = max(0, _safe_int(raw.get("frames", seconds * safe_fps), 0)) if enabled else 0
    if enabled and frames <= 0:
        enabled = False
    return {
        "schema": "iamccs.audio.roll_contract",
        "schema_version": 1,
        "enabled": bool(enabled),
        "seconds": float(frames / safe_fps if frames > 0 else seconds),
        "frames": int(frames),
        "frame_rate": float(safe_fps),
        "mode": "generation_duration_extension",
        "first_take_pre_frames": 0,
        "subsequent_take_pre_frames": int(frames),
        "post_frames": int(frames),
    }


def _master_source_segment_from_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    master = manifest.get("master") if isinstance(manifest.get("master"), dict) else {}
    source = master.get("source_master_segment") if isinstance(master.get("source_master_segment"), dict) else {}
    if not source:
        source = manifest.get("source_master_segment") if isinstance(manifest.get("source_master_segment"), dict) else {}
    def has_media(item: Any) -> bool:
        return isinstance(item, dict) and bool(
            str(item.get("audioFile") or item.get("audioB64") or "").strip()
            or str(item.get("sourceAudioFile") or item.get("sourceAudioB64") or "").strip()
        )

    def materialize_source(item: Dict[str, Any]) -> Dict[str, Any]:
        out = copy.deepcopy(item)
        source_file = str(out.get("sourceAudioFile") or "").strip()
        source_b64 = str(out.get("sourceAudioB64") or "").strip()
        if source_file and not str(out.get("audioFile") or "").strip():
            out["audioFile"] = source_file
        if source_b64 and not str(out.get("audioB64") or "").strip() and not str(out.get("audioFile") or "").strip():
            out["audioB64"] = source_b64
        if out.get("sourceAudioUploadType") and not out.get("audioUploadType"):
            out["audioUploadType"] = out.get("sourceAudioUploadType")
        if out.get("sourceAudioDurationFrames"):
            out["audioDurationFrames"] = max(
                _safe_int(out.get("audioDurationFrames"), 0),
                _safe_int(out.get("sourceAudioDurationFrames"), 0),
            )
        if out.get("sourceTrimStart") is not None:
            out["trimStart"] = _safe_int(out.get("sourceTrimStart"), _safe_int(out.get("trimStart"), 0))
        return out

    if isinstance(source, dict) and has_media(source):
        return materialize_source(source)

    # A previous publish may have stripped sourceSegment.audioFile while the
    # physical T-lane still carries sourceAudioFile metadata. Reconstruct the
    # original source here so post-roll is real audio, not padded silence.
    candidates: List[Dict[str, Any]] = []
    candidates.extend(_segments(master.get("segments")))
    for track in manifest.get("tracks") if isinstance(manifest.get("tracks"), list) else []:
        if isinstance(track, dict):
            candidates.extend(_segments(track.get("segments")))
    for item in candidates:
        source_file = str(item.get("sourceAudioFile") or "").strip()
        source_b64 = str(item.get("sourceAudioB64") or "").strip()
        # Never mistake the 15 s physical T-lane WAV for the original source.
        # It is exactly the file that needs roll headroom.
        if not source_file and not source_b64:
            is_physical_chunk = bool(item.get("physicalChunk")) or str(item.get("audioPublishSchema") or "") == "iamccs.audio.publish.v2" or str(item.get("timelineId") or "").startswith("T")
            if is_physical_chunk:
                continue
            source_file = str(item.get("audioFile") or "").strip()
            source_b64 = str(item.get("audioB64") or "").strip()
        if not source_file and not source_b64:
            continue
        return materialize_source({
            "id": str(item.get("sourceSegmentId") or item.get("id") or "master_audio"),
            "name": str(item.get("sourceName") or item.get("name") or item.get("fileName") or "master_audio"),
            "audioFile": source_file,
            "audioB64": source_b64,
            "audioUploadType": item.get("sourceAudioUploadType") or item.get("audioUploadType") or "input",
            "trimStart": _safe_int(item.get("sourceTrimStart"), 0),
            "sourceAudioDurationFrames": _safe_int(item.get("sourceAudioDurationFrames"), 0),
            "audioDurationFrames": _safe_int(item.get("sourceAudioDurationFrames"), 0),
        })
    return {}


def _extend_visual_timeline_for_roll(visual: Dict[str, Any], target_frames: int, fps: float) -> Dict[str, Any]:
    out = copy.deepcopy(visual) if isinstance(visual, dict) else {}
    target = max(1, _safe_int(target_frames, 1))
    raw_segments = out.get("segments") if isinstance(out.get("segments"), list) else []
    segments = [copy.deepcopy(item) for item in raw_segments if isinstance(item, dict)]
    visual_indexes = [
        index for index, item in enumerate(segments)
        if str(item.get("type", "image") or "image").lower() not in {"audio", "text"}
        and not _safe_bool(item.get("placeholder", False), False)
    ]
    if visual_indexes:
        last_index = max(visual_indexes, key=lambda index: _safe_int(segments[index].get("start", segments[index].get("frame", 0)), 0))
        last = segments[last_index]
        start = max(0, _safe_int(last.get("start", last.get("frame", 0)), 0))
        current_end = start + max(1, _safe_int(last.get("length", last.get("len", 1)), 1))
        if current_end < target:
            last["length"] = int(target - start)
            last["len"] = int(target - start)
        segments[last_index] = last
    out["segments"] = segments
    for key in ("shotClips", "shots"):
        if isinstance(out.get(key), list):
            out[key] = copy.deepcopy(segments)
    out["duration_frames"] = int(target)
    out["duration_seconds"] = float(target) / max(1.0, float(fps))
    out["roll_applied"] = True
    out["roll_mode"] = "generation_duration_extension"
    return out


def _roll_audio_segment(
    source: Dict[str, Any],
    take: Dict[str, Any],
    generation_start: int,
    generation_duration: int,
    track_layout: str,
) -> Dict[str, Any]:
    out = copy.deepcopy(source)
    take_index = max(1, _safe_int(take.get("take_index"), 1))
    source_start = _safe_int(source.get("start", 0), 0)
    relative_start = max(0, int(generation_start) - source_start)
    out["id"] = f"{source.get('id', 'master_audio')}_{_take_timeline_id(take_index)}_roll"
    out["start"] = 0
    out["length"] = max(1, int(generation_duration))
    out["trimStart"] = max(0, _safe_int(source.get("trimStart", 0), 0) + relative_start)
    out["audioDurationFrames"] = max(
        out["length"],
        _safe_int(source.get("audioDurationFrames", source.get("length", out["length"])), out["length"]),
    )
    out["timelineId"] = _take_timeline_id(take_index)
    out["multiTakeIndex"] = int(take_index)
    out["multiGenerationClip"] = True
    out["rollAudioSource"] = True
    out["sourceSegmentId"] = str(source.get("id", ""))
    out["sourceGlobalStart"] = int(generation_start)
    out["sourceGlobalEnd"] = int(generation_start + generation_duration)
    out["track"] = 0 if str(track_layout) == "collapse_to_lane_1" else max(0, _safe_int(source.get("track", 0), 0))
    out["sourceTrack"] = max(0, _safe_int(source.get("track", 0), 0))
    return out


def _apply_roll_to_take(take: Dict[str, Any], roll: Dict[str, Any], fps: float, source_master: Dict[str, Any], track_layout: str) -> None:
    take_index = max(1, _safe_int(take.get("take_index"), 1))
    nominal_start = max(0, _safe_int(take.get("global_start_frames", take.get("source_global_start_frames", 0)), 0))
    nominal_duration = max(1, _safe_int(take.get("nominal_duration_frames", take.get("duration_frames", 1)), 1))
    nominal_end = nominal_start + nominal_duration
    requested = max(0, _safe_int(roll.get("frames", 0), 0)) if _safe_bool(roll.get("enabled"), False) else 0
    # Roll is an extension of the real source window, never a request for
    # synthetic silence.  A final split may have less material than the
    # requested post-roll, while a trimmed source can also limit pre-roll.
    source_start = max(0, _safe_int(source_master.get("start", 0), 0)) if isinstance(source_master, dict) else 0
    source_trim_start = max(0, _safe_int(source_master.get("trimStart", 0), 0)) if isinstance(source_master, dict) else 0
    source_total_frames = max(0, _safe_int(
        source_master.get("audioDurationFrames", source_master.get("length", 0)) if isinstance(source_master, dict) else 0,
        0,
    ))
    source_end = source_start + max(0, source_total_frames - source_trim_start)
    pre_available = max(0, nominal_start - source_start)
    post_available = max(0, source_end - nominal_end) if source_total_frames > 0 else requested
    pre_frames = min(requested, pre_available) if requested > 0 else 0
    post_frames = min(requested, post_available) if requested > 0 else 0
    generation_start = max(source_start, nominal_start - pre_frames)
    generation_end = nominal_end + post_frames
    generation_duration = max(1, generation_end - generation_start)
    take["nominal_global_start_frames"] = int(nominal_start)
    take["nominal_global_end_frames"] = int(nominal_end)
    take["nominal_duration_frames"] = int(nominal_duration)
    take["generation_start_frames"] = int(generation_start)
    take["generation_end_frames"] = int(generation_end)
    take["generation_duration_frames"] = int(generation_duration)
    take["pre_roll_frames"] = int(pre_frames)
    take["post_roll_frames"] = int(post_frames)
    take["pre_roll_seconds"] = float(pre_frames) / max(1.0, float(fps))
    take["post_roll_seconds"] = float(post_frames) / max(1.0, float(fps))
    take["roll_contract"] = copy.deepcopy(roll)
    take["roll_contract"]["requested_frames"] = int(requested)
    take["roll_contract"]["effective_pre_frames"] = int(pre_frames)
    take["roll_contract"]["effective_post_frames"] = int(post_frames)
    take["roll_contract"]["source_end_frames"] = int(source_end)
    take["roll_contract"]["clamped_to_source"] = bool(pre_frames < requested or post_frames < requested)
    take["duration_frames"] = int(generation_duration)
    take["duration_seconds"] = float(generation_duration) / max(1.0, float(fps))
    if requested > 0 and (pre_frames < requested or post_frames < requested):
        print(
            "[IAMCCS MultiTimelineBridge] ROLL_AUDIO_CLAMP "
            f"take=T{take_index:02d} requested={requested}f "
            f"effective_pre={pre_frames}f effective_post={post_frames}f "
            f"source_window={source_start}:{source_end} nominal={nominal_start}:{nominal_end}"
        )
    if source_master and (str(source_master.get("audioFile", "") or "").strip() or str(source_master.get("audioB64", "") or "").strip()):
        take["nominal_audioSegments"] = copy.deepcopy(take.get("audioSegments", []))
        take["audioSegments"] = [_roll_audio_segment(source_master, take, generation_start, generation_duration, track_layout)]
        print(
            "[IAMCCS MultiTimelineBridge] ROLL_AUDIO_SOURCE "
            f"take=T{take_index:02d} file={source_master.get('audioFile') or '<b64>'} "
            f"trim_start={_safe_int(source_master.get('trimStart'), 0)} "
            f"available_frames={_safe_int(source_master.get('audioDurationFrames'), 0)} "
            f"window={generation_start}:{generation_end}"
        )
    elif requested > 0:
        print(
            "[IAMCCS MultiTimelineBridge] ROLL_AUDIO_SOURCE_MISSING "
            f"take=T{take_index:02d} window={generation_start}:{generation_end}. "
            "The physical chunk has no readable reference to the original source."
        )
    visual = take.get("visual_timeline") if isinstance(take.get("visual_timeline"), dict) else {}
    if visual:
        take["visual_timeline"] = _extend_visual_timeline_for_roll(visual, generation_duration, fps)
        take["visual_timeline"]["nominal_duration_frames"] = int(nominal_duration)
        take["visual_timeline"]["pre_roll_frames"] = int(pre_frames)
        take["visual_timeline"]["post_roll_frames"] = int(post_frames)


def _parse_track_jsons(track_jsons: Tuple[Any, ...]) -> List[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []
    for index, value in enumerate(track_jsons):
        data = _safe_json_loads(value, {})
        if isinstance(data, dict) and data:
            data = copy.deepcopy(data)
            data.setdefault("track_index", index)
            data.setdefault("track_name", f"A{index + 1}")
            data["segments"] = _segments(data.get("segments"))
            parsed.append(data)
    return parsed


def _bus_manifest(
    cine_linx: Any,
    bus_manifest_json: Any,
    master_out_json: Any,
    track_jsons: Tuple[Any, ...],
) -> Dict[str, Any]:
    linx = cine_linx if isinstance(cine_linx, dict) else {}
    resources = linx.get("resources", {}) if isinstance(linx.get("resources", {}), dict) else {}
    manifest = _safe_json_loads(bus_manifest_json, {})
    if not isinstance(manifest, dict) or not manifest:
        manifest = resources.get("cine_audio_bus_out") if isinstance(resources.get("cine_audio_bus_out"), dict) else {}
    if not isinstance(manifest, dict):
        manifest = {}

    master = _safe_json_loads(master_out_json, {})
    if not isinstance(master, dict) or not master:
        master = manifest.get("master") if isinstance(manifest.get("master"), dict) else {}
    if not isinstance(master, dict) or not master:
        audio_tracks = resources.get("cine_audio_tracks") if isinstance(resources.get("cine_audio_tracks"), dict) else {}
        master = {
            "schema": "iamccs.audio_bus_out.master",
            "source": "IAMCCS_MultiTimelineBridge",
            "segments": _segments(audio_tracks.get("all_segments") or audio_tracks.get("segments")),
            "masterBus": audio_tracks.get("master_bus") if isinstance(audio_tracks.get("master_bus"), dict) else {},
            "duration_frames": _safe_int(audio_tracks.get("source_end_frames", audio_tracks.get("duration_frames", 0)), 0),
        }
    master = copy.deepcopy(master)
    master["segments"] = _segments(master.get("segments"))
    master["duration_frames"] = max(_safe_int(master.get("duration_frames", 0), 0), _max_end(master["segments"]))
    audio_tracks = resources.get("cine_audio_tracks") if isinstance(resources.get("cine_audio_tracks"), dict) else {}
    source_master_segment = audio_tracks.get("source_master_segment") if isinstance(audio_tracks.get("source_master_segment"), dict) else {}
    if source_master_segment and not master.get("source_master_segment"):
        master["source_master_segment"] = copy.deepcopy(source_master_segment)

    tracks = _parse_track_jsons(track_jsons)
    if not tracks and isinstance(manifest.get("tracks"), list):
        for index, item in enumerate(manifest.get("tracks") or []):
            if not isinstance(item, dict):
                continue
            track = copy.deepcopy(item)
            track.setdefault("track_index", index)
            track.setdefault("track_name", f"A{index + 1}")
            track["segments"] = _segments(track.get("segments"))
            track["duration_frames"] = max(_safe_int(track.get("duration_frames", 0), 0), _max_end(track["segments"]))
            tracks.append(track)
    if not tracks:
        for track_index in range(MAX_TRACK_OUTS):
            track_segments = [seg for seg in master["segments"] if _safe_int(seg.get("track", 0), 0) == track_index]
            tracks.append({
                "schema": "iamccs.audio_bus_out.track",
                "source": "IAMCCS_MultiTimelineBridge",
                "track_index": track_index,
                "track_name": f"A{track_index + 1}",
                "segments": track_segments,
                "duration_frames": _max_end(track_segments),
            })

    generation_index = manifest.get("generation_index") if isinstance(manifest.get("generation_index"), dict) else {}
    if not generation_index:
        generation_index = resources.get("cine_audio_generation_index") if isinstance(resources.get("cine_audio_generation_index"), dict) else {}
    if not generation_index:
        generation_index = _safe_json_loads(resources.get("cine_audio_generation_index_json"), {})
    if not isinstance(generation_index, dict):
        generation_index = {}

    return {
        "schema": "iamccs.audio_bus_out.manifest",
        "schema_version": 1,
        "source": "IAMCCS_MultiTimelineBridge",
        "master": master,
        "tracks": tracks[:MAX_TRACK_OUTS],
        "generation_index": generation_index,
        "source_master_segment": copy.deepcopy(master.get("source_master_segment", {})),
    }


def _source_from_manifest(manifest: Dict[str, Any], source_bus: str) -> Dict[str, Any]:
    source = str(source_bus or "master_out")
    if source == "master_out":
        return copy.deepcopy(manifest.get("master") if isinstance(manifest.get("master"), dict) else {})
    if source.startswith("track_"):
        track_number = max(1, _safe_int(source.split("_", 1)[1], 1))
        tracks = manifest.get("tracks") if isinstance(manifest.get("tracks"), list) else []
        index = track_number - 1
        for track in tracks:
            if not isinstance(track, dict):
                continue
            track_index = _safe_int(track.get("track_index"), -1)
            track_name = str(track.get("track_name") or "").strip().lower()
            if track_index == index or track_name == f"a{track_number}":
                return copy.deepcopy(track)
        if 0 <= index < len(tracks) and isinstance(tracks[index], dict):
            return copy.deepcopy(tracks[index])
    return {}


def _slice_segments(
    segments: List[Dict[str, Any]],
    window_start: int,
    window_length: int,
    track_layout: str,
) -> List[Dict[str, Any]]:
    window_end = window_start + window_length
    sliced: List[Dict[str, Any]] = []
    for index, seg in enumerate(segments):
        seg_start = _safe_int(seg.get("start", 0), 0)
        seg_length = max(1, _safe_int(seg.get("length", 1), 1))
        seg_end = seg_start + seg_length
        overlap_start = max(seg_start, window_start)
        overlap_end = min(seg_end, window_end)
        if overlap_end <= overlap_start:
            continue
        offset = overlap_start - seg_start
        out = copy.deepcopy(seg)
        out["id"] = f"{seg.get('id', 'aud')}_take_{window_start}_{index}"
        out["start"] = overlap_start - window_start
        out["length"] = overlap_end - overlap_start
        out["trimStart"] = max(0, _safe_int(seg.get("trimStart", 0), 0) + offset)
        out["audioDurationFrames"] = max(out["trimStart"] + out["length"], _safe_int(seg.get("audioDurationFrames", seg_length), seg_length))
        out["sourceSegmentId"] = str(seg.get("id", ""))
        out["sourceGlobalStart"] = overlap_start
        out["sourceGlobalEnd"] = overlap_end
        out["sourceTrack"] = _safe_int(seg.get("track", 0), 0)
        out["multiGenerationClip"] = True
        if str(track_layout) == "collapse_to_lane_1":
            out["track"] = 0
        sliced.append(out)
    return sorted(sliced, key=lambda seg: (_safe_int(seg.get("start", 0), 0), _safe_int(seg.get("track", 0), 0)))


def _timeline_for_take(take: Dict[str, Any], fps: float, track_layout: str) -> Dict[str, Any]:
    segments = _segments(take.get("audioSegments"))
    track_count = 1
    if str(track_layout) == "preserve_bus_tracks":
        track_count = max([_safe_int(seg.get("track", 0), 0) + 1 for seg in segments] or [1])
    return {
        "schema": "iamccs.multigeneration.take_audio_timeline",
        "schema_version": 1,
        "timeline_id": str(take.get("timeline_id", "")),
        "take_index": _safe_int(take.get("take_index", 1), 1),
        "frame_rate": float(fps),
        "duration_frames": _safe_int(take.get("duration_frames", 0), 0),
        "duration_seconds": _safe_int(take.get("duration_frames", 0), 0) / max(1.0, float(fps)),
        "nominal_duration_frames": _safe_int(take.get("nominal_duration_frames", take.get("duration_frames", 0)), 0),
        "generation_start_frames": _safe_int(take.get("generation_start_frames", take.get("global_start_frames", 0)), 0),
        "generation_end_frames": _safe_int(take.get("generation_end_frames", take.get("global_end_frames", 0)), 0),
        "pre_roll_frames": _safe_int(take.get("pre_roll_frames", 0), 0),
        "post_roll_frames": _safe_int(take.get("post_roll_frames", 0), 0),
        "audioSegments": segments,
        "audioTrackCount": track_count,
        "audioBusMode": "all_tracks",
        "onlyFirstTrack": False,
        "use_custom_audio": any(str(seg.get("audioFile", "")).strip() or str(seg.get("audioB64", "")).strip() for seg in segments),
    }


def _take_package_for_active_take(
    generation_index: Dict[str, Any],
    active_take: Dict[str, Any],
    take_timeline: Dict[str, Any],
    fps: float,
) -> Dict[str, Any]:
    take_index = max(1, _safe_int(active_take.get("take_index", generation_index.get("active_take", 1)), 1))
    timeline_id = str(active_take.get("timeline_id") or _take_timeline_id(take_index))
    audio_lane = str(active_take.get("audio_lane") or _take_audio_lane_name(take_index))
    visual_timeline = active_take.get("visual_timeline") if isinstance(active_take.get("visual_timeline"), dict) else {}
    visual_segments = visual_timeline.get("segments") if isinstance(visual_timeline.get("segments"), list) else []
    visual_rows = visual_timeline.get("rows") if isinstance(visual_timeline.get("rows"), list) else []
    global_prompt = str(visual_timeline.get("global_prompt", visual_timeline.get("prompt", "")) or "")
    duration_frames = max(1, _safe_int(active_take.get("duration_frames", take_timeline.get("duration_frames", 0)), 0))
    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    # tail_trim_frames defaults to 0 because IAMCCS_LTXVideoDurationCrop handles the LTX 8n+1 extra frame trim.
    # The old default of 1 caused double-trimming: crop (241→240) + VideoEditor (240→239) = 9.958s.
    # With tail_trim_frames=0 and the crop node in the pipeline:
    #   LTX generates 241 → crop → 240 → VideoEditor parks 240 → SaveVideo = 10.000s exactly.
    tail_trim_frames = max(
        0,
        _safe_int(
            active_take.get(
                "tail_trim_frames",
                generation_index.get("tail_trim_frames", generation_index.get("shotboard_tail_trim_frames", 0)),
            ),
            0,
        ),
    )
    return {
        "schema": "iamccs.multigeneration.take_package",
        "schema_version": 1,
        "take_id": timeline_id,
        "take_index": int(take_index),
        "timeline_id": timeline_id,
        "audio_lane": audio_lane,
        "audio_track_index": max(0, take_index - 1),
        "duration_frames": int(duration_frames),
        "duration_seconds": duration_frames / max(1.0, float(fps)),
        "nominal_duration_frames": int(_safe_int(active_take.get("nominal_duration_frames", duration_frames), duration_frames)),
        "generation_start_frames": int(_safe_int(active_take.get("generation_start_frames", active_take.get("global_start_frames", 0)), 0)),
        "generation_end_frames": int(_safe_int(active_take.get("generation_end_frames", active_take.get("global_end_frames", duration_frames)), duration_frames)),
        "pre_roll_frames": int(_safe_int(active_take.get("pre_roll_frames", 0), 0)),
        "post_roll_frames": int(_safe_int(active_take.get("post_roll_frames", 0), 0)),
        "roll_contract": copy.deepcopy(active_take.get("roll_contract", {})) if isinstance(active_take.get("roll_contract"), dict) else {},
        "frame_rate": float(fps),
        "tail_trim_frames": int(tail_trim_frames),
        "global_prompt": global_prompt,
        "visual_segments": copy.deepcopy(visual_segments),
        "visual_rows": copy.deepcopy(visual_rows),
        "audio_segments": copy.deepcopy(take_timeline.get("audioSegments", [])),
        "audio_track_count": int(take_timeline.get("audioTrackCount", 1)),
        "source": "IAMCCS_MultiTimelineBridge",
        "truth": "Immutable rendered-take identity. Downstream collectors must park this take by take_index/timeline_id and must hard-fail on mismatches.",
    }


def _take_sort_key(seg: Dict[str, Any], fallback: int) -> Tuple[int, int, int]:
    take_index = _safe_int(seg.get("multiTakeIndex", seg.get("take_index", fallback + 1)), fallback + 1)
    global_start = _safe_int(seg.get("sourceGlobalStart", seg.get("globalStart", seg.get("start", 0))), 0)
    track = _safe_int(seg.get("track", 0), 0)
    return take_index, global_start, track


def _segments_from_generation_index(generation_index: Any) -> List[Dict[str, Any]]:
    if not isinstance(generation_index, dict):
        return []
    out: List[Dict[str, Any]] = []
    for take in generation_index.get("takes") if isinstance(generation_index.get("takes"), list) else []:
        if isinstance(take, dict):
            out.extend(_segments(take.get("segments")))
    return out


def _collect_prechunked_segments(manifest: Dict[str, Any], source_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = list(source_segments)
    master = manifest.get("master") if isinstance(manifest.get("master"), dict) else {}
    candidates.extend(_segments(master.get("segments")))
    for track in manifest.get("tracks") if isinstance(manifest.get("tracks"), list) else []:
        if isinstance(track, dict):
            candidates.extend(_segments(track.get("segments")))
    candidates.extend(_segments_from_generation_index(manifest.get("generation_index")))

    out: List[Dict[str, Any]] = []
    seen = set()
    for seg in candidates:
        if not (bool(seg.get("multiGenerationClip")) or str(seg.get("timelineId", "") or "").startswith("T")):
            continue
        key = (
            str(seg.get("id", "")),
            str(seg.get("timelineId", "")),
            _safe_int(seg.get("multiTakeIndex", seg.get("take_index", 0)), 0),
            _safe_int(seg.get("track", 0), 0),
            _safe_int(seg.get("sourceGlobalStart", seg.get("globalStart", seg.get("start", 0))), 0),
            _safe_int(seg.get("length", 1), 1),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(copy.deepcopy(seg))
    if any(bool(seg.get("physicalChunk")) or str(seg.get("audioPublishSchema", "")) == "iamccs.audio.publish.v2" for seg in out):
        def _derived_pluri(item: Dict[str, Any]) -> bool:
            return bool(item.get("shotboardPluriPublish")) or bool(re.search(r"_pluri_t\d+", str(item.get("id", "")), re.IGNORECASE))

        def _timeline_key(item: Dict[str, Any], fallback: int) -> int:
            raw = str(item.get("timelineId") or "")
            return max(1, _safe_int(item.get("multiTakeIndex", item.get("take_index", 0)), 0) or _take_index_from_timeline_id(raw) or fallback)

        def _score(item: Dict[str, Any]) -> int:
            value = 0
            if bool(item.get("physicalChunk")) or str(item.get("audioPublishSchema", "")) == "iamccs.audio.publish.v2":
                value += 100
            if not _derived_pluri(item):
                value += 50
            if str(item.get("audioFile", "")).strip() or str(item.get("audioB64", "")).strip():
                value += 20
            take = _timeline_key(item, 1)
            if _safe_int(item.get("track", -999), -999) == take - 1:
                value += 5
            return value

        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for index, seg in enumerate(out, start=1):
            grouped.setdefault(_timeline_key(seg, index), []).append(seg)
        out = [
            sorted(group, key=lambda item: (_score(item), -_safe_int(item.get("start", 0), 0)), reverse=True)[0]
            for _, group in sorted(grouped.items(), key=lambda pair: pair[0])
        ]
    return sorted(out, key=lambda item: _take_sort_key(item, 0))


def _takes_from_prechunked_segments(
    segments: List[Dict[str, Any]],
    fps: float,
    chunk_frames: int,
    chunk_seconds: float,
    source_bus: str,
    track_layout: str,
    visual_timelines: Any,
) -> List[Dict[str, Any]]:
    multi_segments = [
        copy.deepcopy(seg)
        for seg in segments
        if bool(seg.get("multiGenerationClip")) or str(seg.get("timelineId", "") or "").startswith("T")
    ]
    if not multi_segments:
        return []

    groups: Dict[int, List[Dict[str, Any]]] = {}
    for fallback, seg in enumerate(sorted(multi_segments, key=lambda item: _take_sort_key(item, fallback=0))):
        take_index = _safe_int(seg.get("multiTakeIndex", seg.get("take_index", fallback + 1)), fallback + 1)
        groups.setdefault(max(1, take_index), []).append(seg)

    takes: List[Dict[str, Any]] = []
    for order, take_index in enumerate(sorted(groups), start=1):
        group = groups[take_index]
        global_start = min([
            _safe_int(seg.get("sourceGlobalStart", seg.get("globalStart", seg.get("start", 0))), 0)
            for seg in group
        ] or [0])
        global_end = max([
            _safe_int(seg.get("sourceGlobalEnd", seg.get("globalEnd", _safe_int(seg.get("start", 0), 0) + _safe_int(seg.get("length", 1), 1))), 0)
            for seg in group
        ] or [global_start + chunk_frames])
        duration = max(1, global_end - global_start)
        audio_segments: List[Dict[str, Any]] = []
        for seg_index, seg in enumerate(group):
            out = copy.deepcopy(seg)
            source_global_start = _safe_int(out.get("sourceGlobalStart", out.get("globalStart", out.get("start", 0))), 0)
            out["id"] = str(out.get("id") or f"multi_take_{take_index:02d}_{seg_index + 1:02d}")
            out["start"] = max(0, source_global_start - global_start)
            out["length"] = max(1, _safe_int(out.get("length", 1), 1))
            out["sourceTrack"] = _safe_int(out.get("sourceTrack", out.get("track", 0)), 0)
            out["timelineId"] = str(out.get("timelineId") or f"T{take_index:02d}")
            out["multiTakeIndex"] = int(take_index)
            if str(track_layout) == "collapse_to_lane_1":
                out["track"] = 0
            audio_segments.append(out)
        timeline_id = str(group[0].get("timelineId") or f"T{take_index:02d}")
        take = {
            "schema": "iamccs.multigeneration.take",
            "schema_version": 1,
            "take_index": int(order),
            "source_take_index": int(take_index),
            "timeline_id": timeline_id,
            "source_bus": str(source_bus),
            "global_start_frames": int(global_start),
            "global_end_frames": int(global_start + duration),
            "local_start_frames": 0,
            "duration_frames": int(duration),
            "duration_seconds": duration / fps,
            "chunk_frames": int(chunk_frames),
            "chunk_seconds": float(chunk_seconds),
            "audioSegments": sorted(audio_segments, key=lambda item: (_safe_int(item.get("start", 0), 0), _safe_int(item.get("track", 0), 0))),
            "audioTrackCount": max([_safe_int(seg.get("track", 0), 0) + 1 for seg in audio_segments] or [1]),
            "visual_timeline_key": timeline_id,
            "visual_timeline": visual_timelines.get(timeline_id) if isinstance(visual_timelines, dict) else None,
            "prechunked": True,
        }
        take["take_audio_timeline"] = _timeline_for_take(take, fps, str(track_layout))
        takes.append(take)
    return takes


def _apply_active_take(
    cine_linx: Dict[str, Any],
    generation_index: Dict[str, Any],
    active_take: Dict[str, Any],
    track_layout: str,
) -> None:
    fps = _safe_float(generation_index.get("frame_rate", 24.0), 24.0)
    # The planner owns the visual timeline.  A bridge input may omit the
    # optional visual_timelines_json while cine_board_timeline_data still
    # contains the selected T01/T02 contract.  Recover it before packaging
    # the take so a roll cannot extend audio alone.
    active_visual = active_take.get("visual_timeline") if isinstance(active_take.get("visual_timeline"), dict) else {}
    active_visual_segments = active_visual.get("segments") if isinstance(active_visual.get("segments"), list) else []
    visual_recovered = False
    if not active_visual_segments:
        board_timeline = _safe_json_loads(_resources(cine_linx).get("cine_board_timeline_data"), {})
        multi = board_timeline.get("multiGeneration") if isinstance(board_timeline, dict) and isinstance(board_timeline.get("multiGeneration"), dict) else {}
        visuals = multi.get("visualTimelines") if isinstance(multi.get("visualTimelines"), dict) else {}
        timeline_id = str(active_take.get("timeline_id") or _take_timeline_id(active_take.get("take_index", 1)))
        candidates = [timeline_id]
        take_number = _safe_int(active_take.get("take_index", 1), 1)
        if take_number > 0:
            candidates.extend([_take_timeline_id(take_number), f"T{take_number}"])
        for candidate_key in candidates:
            candidate = visuals.get(candidate_key)
            if isinstance(candidate, dict) and isinstance(candidate.get("segments"), list) and candidate.get("segments"):
                active_visual = copy.deepcopy(candidate)
                active_visual_segments = active_visual.get("segments", [])
                visual_recovered = True
                break

    target_visual_frames = max(0, _safe_int(active_take.get("duration_frames", 0), 0))
    if active_visual_segments and target_visual_frames > 0:
        current_visual_end = _max_end([
            item for item in active_visual_segments
            if isinstance(item, dict)
            and str(item.get("type", "image") or "image").lower() not in {"audio", "text"}
            and not _safe_bool(item.get("placeholder", False), False)
        ])
        if current_visual_end < target_visual_frames:
            active_visual = _extend_visual_timeline_for_roll(active_visual, target_visual_frames, fps)
            print(
                "[IAMCCS MultiTimelineBridge] VISUAL_ROLL_CONTRACT "
                f"timeline={active_take.get('timeline_id', '')} "
                f"visual_end_before={int(current_visual_end)} "
                f"visual_end_after={int(target_visual_frames)} "
                f"extended_frames={int(target_visual_frames - current_visual_end)}"
            )
        else:
            active_visual = copy.deepcopy(active_visual)
            active_visual["duration_frames"] = int(target_visual_frames)
            active_visual["duration_seconds"] = float(target_visual_frames) / max(1.0, float(fps))
        active_take["visual_timeline"] = active_visual
        active_visual_segments = active_visual.get("segments") if isinstance(active_visual.get("segments"), list) else []
        if visual_recovered:
            print(
                "[IAMCCS MultiTimelineBridge] VISUAL_TIMELINE_RECOVERED "
                f"timeline={active_take.get('timeline_id', '')} segments={len(active_visual_segments)} "
                f"duration_frames={int(target_visual_frames)} source=cine_board_timeline_data"
            )

    take_timeline = _timeline_for_take(active_take, fps, track_layout)
    take_package = _take_package_for_active_take(generation_index, active_take, take_timeline, fps)
    duration_frames = _safe_int(take_timeline.get("duration_frames", 0), 0)
    duration_seconds = duration_frames / max(1.0, fps)
    resources = _resources(cine_linx)
    outputs = _outputs(cine_linx)
    payload = _payload(cine_linx)

    resources["cine_multigeneration_index"] = generation_index
    resources["cine_multigeneration_index_json"] = _json_dump(generation_index)
    resources["cine_multigeneration_active_take"] = active_take
    resources["cine_multigeneration_active_take_json"] = _json_dump(active_take)
    resources["cine_multigeneration_take_package"] = take_package
    resources["cine_multigeneration_take_package_json"] = _json_dump(take_package)
    take_timeline_json = _json_dump(take_timeline)
    take_segments = copy.deepcopy(take_timeline.get("audioSegments", []))
    resources["cine_multigeneration_take_audio_timeline"] = take_timeline
    resources["cine_multigeneration_take_audio_timeline_json"] = take_timeline_json
    resources["cine_audio_timeline"] = copy.deepcopy(take_timeline)
    resources["cine_audio_timeline_json"] = take_timeline_json
    resources["cine_roll_contract"] = copy.deepcopy(active_take.get("roll_contract", {})) if isinstance(active_take.get("roll_contract"), dict) else {}
    resources["cine_visual_segments_json"] = _json_dump(active_visual_segments)
    payload["visual_segments"] = copy.deepcopy(active_visual_segments)
    resources["cine_audio_tracks"] = {
        "source": "IAMCCS_MultiTimelineBridge_active_take",
        "shotboard_segments": copy.deepcopy(take_segments),
        "segments": copy.deepcopy(take_segments),
        "all_segments": copy.deepcopy(take_segments),
        "audioTrackCount": int(take_timeline.get("audioTrackCount", 1)),
        "duration_frames": int(duration_frames),
        "source_end_frames": int(duration_frames),
        "timeline_id": str(active_take.get("timeline_id", "")),
        "active_take": int(active_take.get("take_index", generation_index.get("active_take", 1)) or 1),
        "source_master_segment": copy.deepcopy(resources.get("cine_audio_source_master_segment", {})) if isinstance(resources.get("cine_audio_source_master_segment"), dict) else {},
        "roll_contract": copy.deepcopy(active_take.get("roll_contract", {})) if isinstance(active_take.get("roll_contract"), dict) else {},
    }
    resources["cine_use_custom_audio"] = bool(take_timeline.get("use_custom_audio", False))
    resources["cine_duration_seconds"] = float(duration_seconds)
    # Keep the active take contract on the visible/narrative duration.
    # Technical LTX padding must never be injected here: the Shotboard planner,
    # PromptRelay segment lengths and guide insert frames all consume this
    # cine_linx contract. Padding here can make a later slot inherit pixels or
    # prompts from an earlier slot.
    resources["cine_max_frames"] = int(duration_frames)

    payload["multi_generation"] = generation_index
    payload["multi_generation_active_take"] = active_take
    payload["take_package"] = take_package
    payload["timeline_id"] = str(active_take.get("timeline_id", ""))
    payload["duration_seconds"] = float(duration_seconds)
    payload["duration_frames"] = int(duration_frames)
    payload["max_frames"] = int(duration_frames)
    payload.pop("ltxv_length", None)
    payload["audioSegments"] = take_timeline["audioSegments"]
    payload["audioTrackCount"] = take_timeline["audioTrackCount"]
    payload["use_custom_audio"] = bool(take_timeline["use_custom_audio"])
    payload["audioSyncMode"] = "timeline_audio"

    # Keep the routed visual contract in the linx itself. TakeRouter consumes
    # this resource when its optional timeline_data widget is empty, which is
    # the normal workflow path.
    board_timeline = _safe_json_loads(resources.get("cine_board_timeline_data"), {})
    if isinstance(board_timeline, dict) and active_visual:
        timeline_id = str(active_take.get("timeline_id") or _take_timeline_id(active_take.get("take_index", 1)))
        multi = board_timeline.get("multiGeneration") if isinstance(board_timeline.get("multiGeneration"), dict) else {}
        visuals = multi.get("visualTimelines") if isinstance(multi.get("visualTimelines"), dict) else {}
        visuals[timeline_id] = copy.deepcopy(active_visual)
        multi.update({
            "enabled": True,
            "activeTake": int(active_take.get("take_index", 1) or 1),
            "activeTimelineId": timeline_id,
            "visualTimelines": visuals,
        })
        board_timeline["multiGeneration"] = multi
        board_timeline["visualTimelines"] = visuals
        board_timeline["activeTake"] = int(active_take.get("take_index", 1) or 1)
        board_timeline["activeTimelineId"] = timeline_id
        board_timeline["segments"] = copy.deepcopy(active_visual_segments)
        board_timeline["rows"] = copy.deepcopy(active_visual.get("rows", [])) if isinstance(active_visual.get("rows"), list) else []
        board_timeline["duration_seconds"] = float(active_visual.get("duration_seconds", duration_seconds) or duration_seconds)
        board_timeline["frame_rate"] = float(active_visual.get("frame_rate", fps) or fps)
        board_timeline["roll_contract"] = copy.deepcopy(active_take.get("roll_contract", {}))
        board_timeline_json = _json_dump(board_timeline)
        resources["cine_board_timeline_data"] = board_timeline_json
        outputs["timeline_data"] = board_timeline_json
        payload["timeline_data"] = board_timeline_json

    outputs["generation_index_json"] = _json_dump(generation_index)
    outputs["active_take_json"] = _json_dump(active_take)
    outputs["take_package_json"] = _json_dump(take_package)
    outputs["take_audio_timeline_json"] = take_timeline_json
    outputs["audio_timeline_json"] = take_timeline_json
    outputs["audio_segments_json"] = _json_dump(take_segments)
    outputs["duration_seconds"] = float(duration_seconds)
    outputs["max_frames"] = int(duration_frames)


def _take_audio_lane_name(take_index: Any) -> str:
    take = max(1, _safe_int(take_index, 1))
    return f"A{take}"


def _take_timeline_id(take_index: Any) -> str:
    take = max(1, _safe_int(take_index, 1))
    return f"T{take:02d}"


def _take_index_from_timeline_id(value: Any) -> int:
    raw = str(value or "").strip()
    if not raw:
        return 0
    match = re.search(r"(\d+)", raw)
    return max(0, _safe_int(match.group(1), 0)) if match else 0


def _v4_visual_source(value: Any) -> Dict[str, Any]:
    data = _safe_json_loads(value, {}) if isinstance(value, str) else value
    return copy.deepcopy(data) if isinstance(data, dict) else {}


def _v4_visual_segments(visual: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("segments", "shotClips", "shots", "visualClips", "visual_clips", "clips"):
        value = visual.get(key)
        if isinstance(value, list):
            return [copy.deepcopy(item) for item in value if isinstance(item, dict)]
    return []


def _normalise_v4_visual_timeline(visual: Dict[str, Any], base: Dict[str, Any], timeline_id: str, take_index: int) -> Dict[str, Any]:
    out = copy.deepcopy(visual) if isinstance(visual, dict) else {}
    fps = _safe_float(out.get("frame_rate", out.get("fps", base.get("frame_rate", base.get("fps", 24)))), 24.0)
    duration = _safe_float(
        out.get("duration_seconds", out.get("durationSeconds", out.get("duration", base.get("duration_seconds", 0)))),
        _safe_float(base.get("duration_seconds"), 0.0),
    )
    segments = _v4_visual_segments(out)
    rows = out.get("rows") if isinstance(out.get("rows"), list) else []
    motion_segments = out.get("motionSegments", out.get("motionClips", out.get("motion_parts", out.get("motionParts", []))))
    if not isinstance(motion_segments, list):
        motion_segments = []
    video_segments = out.get("videoSegments", out.get("videoClips", out.get("sourceVideoSegments", [])))
    if not isinstance(video_segments, list):
        video_segments = []
    audio_segments = out.get("audioSegments", out.get("audioClips", []))
    if not isinstance(audio_segments, list):
        audio_segments = []
    camera_segments = out.get("cameraSegments", out.get("cameraClips", []))
    if not isinstance(camera_segments, list):
        camera_segments = []
    reference_sheets = out.get("referenceSheets", out.get("reference_sheets", []))
    if not isinstance(reference_sheets, list):
        reference_sheets = []

    out.update({
        "schema": str(out.get("schema") or "iamccs.shotboard_v4.visual_timeline"),
        "timeline_id": timeline_id,
        "take_index": int(take_index),
        "frame_rate": float(fps),
        "duration_seconds": float(duration if duration > 0 else 0.01),
        "duration_frames": max(1, int(round(float(duration if duration > 0 else 0.01) * max(1.0, fps)))),
        "global_prompt": str(out.get("global_prompt", out.get("prompt", base.get("global_prompt", ""))) or ""),
        "prompt": str(out.get("prompt", out.get("global_prompt", base.get("global_prompt", ""))) or ""),
        "segments": segments,
        "rows": [copy.deepcopy(item) for item in rows if isinstance(item, dict)],
        "shotClips": segments,
        "shots": segments,
        "motionSegments": [copy.deepcopy(item) for item in motion_segments if isinstance(item, dict)],
        "motionClips": [copy.deepcopy(item) for item in motion_segments if isinstance(item, dict)],
        "motionParts": [copy.deepcopy(item) for item in motion_segments if isinstance(item, dict)],
        "videoSegments": [copy.deepcopy(item) for item in video_segments if isinstance(item, dict)],
        "videoClips": [copy.deepcopy(item) for item in video_segments if isinstance(item, dict)],
        "audioSegments": [copy.deepcopy(item) for item in audio_segments if isinstance(item, dict)],
        "audioClips": [copy.deepcopy(item) for item in audio_segments if isinstance(item, dict)],
        "cameraSegments": [copy.deepcopy(item) for item in camera_segments if isinstance(item, dict)],
        "cameraClips": [copy.deepcopy(item) for item in camera_segments if isinstance(item, dict)],
        "referenceSheets": [copy.deepcopy(item) for item in reference_sheets if isinstance(item, dict)],
        "shotboard_version": 4,
        "adapter_contract": "iamccs_multigeneration_v4_visual_timeline",
    })
    return out


def _normalise_shotboard_v4_for_multigen(timeline_data: Any, active_take: int = 1) -> Dict[str, Any]:
    base = _v4_visual_source(timeline_data)
    fps = _safe_float(base.get("frame_rate", base.get("fps", 24)), 24.0)
    duration = _safe_float(base.get("duration_seconds", base.get("duration", 0)), 0.0)
    active_take = max(1, _safe_int(active_take, 1))
    multi = base.get("multiGeneration") if isinstance(base.get("multiGeneration"), dict) else {}
    if not multi:
        multi = base.get("multi_generation") if isinstance(base.get("multi_generation"), dict) else {}
    active_timeline = str(
        multi.get("activeTimelineId")
        or multi.get("active_timeline_id")
        or base.get("activeTimelineId")
        or _take_timeline_id(active_take)
    )
    active_take = max(1, _safe_int(multi.get("activeTake", multi.get("active_take", active_take)), active_take))
    visual_timelines = multi.get("visualTimelines") if isinstance(multi.get("visualTimelines"), dict) else {}
    if not visual_timelines:
        visual_timelines = {active_timeline: base}

    normalised_visuals: Dict[str, Dict[str, Any]] = {}
    for fallback_index, (timeline_id, visual) in enumerate(visual_timelines.items(), start=1):
        tid = str(timeline_id or _take_timeline_id(fallback_index))
        take_index = _take_index_from_timeline_id(tid) or fallback_index
        normalised_visuals[tid] = _normalise_v4_visual_timeline(
            visual if isinstance(visual, dict) else {},
            base,
            tid,
            take_index,
        )
    if active_timeline not in normalised_visuals:
        active_timeline = sorted(normalised_visuals.keys())[0] if normalised_visuals else _take_timeline_id(active_take)

    out = copy.deepcopy(base)
    out.update({
        "schema": str(base.get("schema") or "iamccs.shotboard_v4.multigeneration_adapter"),
        "schema_version": max(1, _safe_int(base.get("schema_version"), 1)),
        "shotboard_version": 4,
        "frame_rate": float(fps),
        "duration_seconds": float(duration if duration > 0 else normalised_visuals.get(active_timeline, {}).get("duration_seconds", 0.01)),
        "multiGeneration": {
            **(copy.deepcopy(multi) if isinstance(multi, dict) else {}),
            "enabled": True,
            "activeTake": int(active_take),
            "activeTimelineId": active_timeline,
            "visualTimelines": normalised_visuals,
            "adapter": "IAMCCS_ShotboardV4MultigenAdapter",
        },
        "visualTimelines": normalised_visuals,
        "activeTimelineId": active_timeline,
        "activeTake": int(active_take),
        "adapter_contract": "shotboard_v4_to_multigeneration_v3_backend",
    })
    active_visual = normalised_visuals.get(active_timeline, {})
    if active_visual:
        out["segments"] = copy.deepcopy(active_visual.get("segments", []))
        out["rows"] = copy.deepcopy(active_visual.get("rows", []))
        out["motionSegments"] = copy.deepcopy(active_visual.get("motionSegments", []))
        out["motionClips"] = copy.deepcopy(active_visual.get("motionClips", []))
        out["motionParts"] = copy.deepcopy(active_visual.get("motionParts", []))
        out["videoSegments"] = copy.deepcopy(active_visual.get("videoSegments", []))
        out["videoClips"] = copy.deepcopy(active_visual.get("videoClips", []))
        out["audioSegments"] = copy.deepcopy(active_visual.get("audioSegments", []))
        out["audioClips"] = copy.deepcopy(active_visual.get("audioClips", []))
    return out


def _take_identity_from_payload(value: Any) -> Dict[str, Any]:
    data = _safe_json_loads(value, {}) if isinstance(value, str) else value
    if not isinstance(data, dict):
        return {}

    def read(owner: Any, source: str) -> Dict[str, Any]:
        if not isinstance(owner, dict):
            return {}
        timeline_id = str(
            owner.get("activeTimelineId")
            or owner.get("active_timeline_id")
            or owner.get("timeline_id")
            or owner.get("timelineId")
            or owner.get("take_id")
            or ""
        ).strip()
        take_index = _safe_int(
            owner.get("activeTake")
            or owner.get("active_take")
            or owner.get("take_index")
            or owner.get("source_take_index"),
            0,
        )
        if take_index <= 0 and timeline_id:
            take_index = _take_index_from_timeline_id(timeline_id)
        if take_index <= 0:
            return {}
        return {
            "take_index": int(take_index),
            "timeline_id": timeline_id or _take_timeline_id(take_index),
            "source": source,
        }

    multi = data.get("multiGeneration") if isinstance(data.get("multiGeneration"), dict) else {}
    if not multi:
        multi = data.get("multi_generation") if isinstance(data.get("multi_generation"), dict) else {}
    for owner, source in (
        (multi, "multiGeneration"),
        (data.get("takePackageActive"), "takePackageActive"),
        (data.get("active_take_package"), "active_take_package"),
        (data.get("take_package"), "take_package"),
        (data.get("multi_generation_active_take"), "multi_generation_active_take"),
        (data, "payload"),
    ):
        identity = read(owner, source)
        if identity:
            return identity
    return {}


def _active_identity_from_linx(cine_linx: Any) -> Dict[str, Any]:
    linx = cine_linx if isinstance(cine_linx, dict) else {}
    resources = linx.get("resources", {}) if isinstance(linx.get("resources", {}), dict) else {}
    outputs = linx.get("outputs", {}) if isinstance(linx.get("outputs", {}), dict) else {}
    top_payload = linx.get("payload", {}) if isinstance(linx.get("payload", {}), dict) else {}
    resource_payload = resources.get("cine_payload") if isinstance(resources.get("cine_payload"), dict) else {}

    for value in (
        resources.get("cine_take_router_timeline_data"),
        outputs.get("timeline_data"),
        resource_payload.get("timeline_data"),
        top_payload.get("timeline_data"),
        resources.get("cine_board_timeline_data"),
        resources.get("cine_dialogue_shotboard_timeline_json"),
        resource_payload,
        top_payload,
        resources.get("cine_multigeneration_active_take"),
        _safe_json_loads(resources.get("cine_multigeneration_active_take_json"), {}),
        resources.get("cine_multigeneration_index"),
        _safe_json_loads(resources.get("cine_multigeneration_index_json"), {}),
    ):
        identity = _take_identity_from_payload(value)
        if identity:
            return identity

    for item in reversed(linx.get("chain", []) if isinstance(linx.get("chain", []), list) else []):
        identity = _take_identity_from_payload(item)
        if identity:
            identity["source"] = "chain"
            return identity
    return {}


def _shotboard_timeline_identity_from_linx(cine_linx: Any) -> Dict[str, Any]:
    """Read only the active timeline compiled by Shotboard, ignoring stale packages."""
    linx = cine_linx if isinstance(cine_linx, dict) else {}
    resources = linx.get("resources", {}) if isinstance(linx.get("resources", {}), dict) else {}
    outputs = linx.get("outputs", {}) if isinstance(linx.get("outputs", {}), dict) else {}
    top_payload = linx.get("payload", {}) if isinstance(linx.get("payload", {}), dict) else {}
    resource_payload = resources.get("cine_payload") if isinstance(resources.get("cine_payload"), dict) else {}

    def read_timeline(value: Any, source: str) -> Dict[str, Any]:
        data = _safe_json_loads(value, {}) if isinstance(value, str) else value
        if not isinstance(data, dict):
            return {}
        multi = data.get("multiGeneration") if isinstance(data.get("multiGeneration"), dict) else {}
        if not multi:
            multi = data.get("multi_generation") if isinstance(data.get("multi_generation"), dict) else {}
        timeline_id = str(
            multi.get("activeTimelineId")
            or multi.get("active_timeline_id")
            or data.get("activeTimelineId")
            or data.get("active_timeline_id")
            or data.get("timeline_id")
            or ""
        ).strip()
        take_index = _safe_int(
            multi.get("activeTake")
            or multi.get("active_take")
            or data.get("activeTake")
            or data.get("active_take"),
            0,
        )
        if take_index <= 0 and timeline_id:
            take_index = _take_index_from_timeline_id(timeline_id)
        if take_index <= 0:
            return {}
        return {
            "take_index": int(take_index),
            "timeline_id": timeline_id or _take_timeline_id(take_index),
            "source": source,
        }

    for value, source in (
        (resources.get("cine_board_timeline_data"), "resources.cine_board_timeline_data"),
        (resources.get("cine_dialogue_shotboard_timeline_json"), "resources.cine_dialogue_shotboard_timeline_json"),
        (top_payload.get("timeline_data"), "payload.timeline_data"),
        (resource_payload.get("timeline_data"), "resources.cine_payload.timeline_data"),
        (outputs.get("timeline_data"), "outputs.timeline_data"),
    ):
        identity = read_timeline(value, source)
        if identity:
            return identity
    return {}


def _take_from_generation_index(generation_index: Any, take_index: int) -> Dict[str, Any]:
    if not isinstance(generation_index, dict):
        return {}
    timeline_id = _take_timeline_id(take_index)
    for take in generation_index.get("takes") if isinstance(generation_index.get("takes"), list) else []:
        if not isinstance(take, dict):
            continue
        candidate_index = _safe_int(take.get("take_index"), 0)
        candidate_timeline = str(take.get("timeline_id") or "").strip()
        if candidate_index == take_index or candidate_timeline == timeline_id:
            out = copy.deepcopy(take)
            out["take_index"] = take_index
            out["timeline_id"] = timeline_id
            out["audio_lane"] = _take_audio_lane_name(take_index)
            return out
    return {}


def _make_take_audio_contract(takes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    contract: List[Dict[str, Any]] = []
    for idx, take in enumerate(takes):
        take_index = max(1, _safe_int(take.get("take_index", idx + 1), idx + 1))
        timeline_id = str(take.get("timeline_id") or _take_timeline_id(take_index))
        audio_lane = str(take.get("audio_lane") or _take_audio_lane_name(take_index))
        audio_track_index = max(0, _safe_int(take.get("audio_track_index", take_index - 1), take_index - 1))
        contract.append({
            "take_index": take_index,
            "timeline_id": timeline_id,
            "audio_lane": audio_lane,
            "audio_track_index": audio_track_index,
            "mapping": f"{timeline_id}->{audio_lane}",
            "rule": "one_indexed_audio_lane_per_timeline",
        })
    return contract


def _make_sequence_plan(index: Dict[str, Any]) -> Dict[str, Any]:
    takes = index.get("takes") if isinstance(index.get("takes"), list) else []
    contract = _make_take_audio_contract(takes)
    return {
        "schema": "iamccs.multigeneration.sequence_plan",
        "schema_version": 1,
        "source": "IAMCCS_MultiTimelineBridge",
        "mode": "manual_or_sequential",
        "manual_active_take": _safe_int(index.get("active_take", 1), 1),
        "queue_policy": "manual_take_picker_branches_or_external_sequential_queue",
        "contract": contract,
        "steps": [{
            "step_index": idx + 1,
            "take_index": item["take_index"],
            "timeline_id": item["timeline_id"],
            "audio_lane": item["audio_lane"],
            "bridge_action": f"prepare_{item['timeline_id']}_{item['audio_lane']}",
            "expected_video_slot": f"video_take_{item['take_index']:02d}",
        } for idx, item in enumerate(contract)],
        "truth": "T1 uses A1, T2 uses A2, T3 uses A3. Manual mode prepares one take; sequential mode queues each step in order and sends generated clips to the Shotboard Video Editor.",
    }


class IAMCCS_ShotboardV4MultigenAdapter:
    """Normalize Shotboard V4 cine_linx/timeline_data for the V3 multigeneration backend contract."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "strict_mode": (["hard_fail", "warn_only"], {"default": "hard_fail"}),
                "active_take_hint": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
            },
            "optional": {
                "timeline_data": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE, "STRING", "STRING")
    RETURN_NAMES = ("cine_linx", "timeline_data", "report")
    FUNCTION = "adapt"
    CATEGORY = "IAMCCS/Cine/Multigeneration"

    def adapt(self, cine_linx, strict_mode, active_take_hint, timeline_data=""):
        out_linx = _clone_linx(cine_linx, "iamccs_shotboard_v4_multigen_adapter")
        resources = _resources(out_linx)
        outputs = _outputs(out_linx)
        payload = _payload(out_linx)
        source_timeline = str(timeline_data or "").strip() or _timeline_json_from_linx(out_linx)
        if not source_timeline.strip():
            if str(strict_mode) == "hard_fail":
                raise ValueError("IAMCCS ShotboardV4MultigenAdapter hard-fail: no timeline_data found in input cine_linx.")
            source_timeline = "{}"
        normalised = _normalise_shotboard_v4_for_multigen(source_timeline, _safe_int(active_take_hint, 1))
        multi = normalised.get("multiGeneration") if isinstance(normalised.get("multiGeneration"), dict) else {}
        visual_timelines = multi.get("visualTimelines") if isinstance(multi.get("visualTimelines"), dict) else {}
        active_timeline = str(multi.get("activeTimelineId") or normalised.get("activeTimelineId") or "").strip()
        active_visual = visual_timelines.get(active_timeline) if isinstance(visual_timelines.get(active_timeline), dict) else {}
        active_segments = active_visual.get("segments") if isinstance(active_visual.get("segments"), list) else []
        active_rows = active_visual.get("rows") if isinstance(active_visual.get("rows"), list) else []
        if str(strict_mode) == "hard_fail" and (not active_timeline or (not active_segments and not active_rows)):
            raise ValueError(
                "IAMCCS ShotboardV4MultigenAdapter hard-fail: active V4 timeline has no visual segments/rows. "
                f"activeTimelineId={active_timeline!r}."
            )

        normalised_json = _json_dump(normalised)
        resources["cine_board_timeline_data"] = normalised_json
        resources["cine_dialogue_shotboard_timeline_json"] = normalised_json
        resources["cine_shotboard_v4_multigen_timeline_data"] = normalised_json
        resources["cine_shotboard_version"] = 4
        resources["cine_shotboard_v4_visual_timelines"] = copy.deepcopy(visual_timelines)
        resources["cine_shotboard_v4_motion_segments"] = copy.deepcopy(normalised.get("motionSegments", []))
        resources["cine_shotboard_v4_video_segments"] = copy.deepcopy(normalised.get("videoSegments", []))
        resources["cine_shotboard_v4_audio_segments"] = copy.deepcopy(normalised.get("audioSegments", []))
        payload.update({
            "source": "IAMCCS_ShotboardV4MultigenAdapter",
            "shotboard_version": 4,
            "timeline_data": normalised_json,
            "v4_timeline_data": normalised_json,
            "multiGeneration": copy.deepcopy(multi),
            "activeTimelineId": active_timeline,
            "activeTake": _take_index_from_timeline_id(active_timeline) or _safe_int(active_take_hint, 1),
            "visualTimelines": copy.deepcopy(visual_timelines),
            "motionSegments": copy.deepcopy(normalised.get("motionSegments", [])),
            "videoSegments": copy.deepcopy(normalised.get("videoSegments", [])),
            "audioSegments": copy.deepcopy(normalised.get("audioSegments", [])),
        })
        outputs["timeline_data"] = normalised_json
        outputs["shotboard_v4_timeline_data"] = normalised_json
        out_linx["mode"] = "iamccs_shotboard_v4_multigen_adapter"
        out_linx["pipeline_kind"] = "shotboard_v4_to_multigeneration"
        _refresh_linx_index(out_linx)
        report_payload = {
            "node": "IAMCCS_ShotboardV4MultigenAdapter",
            "activeTimelineId": active_timeline,
            "activeTake": payload.get("activeTake"),
            "visualTimelines": len(visual_timelines),
            "active_segments": len(active_segments),
            "active_rows": len(active_rows),
            "motionSegments": len(normalised.get("motionSegments", [])) if isinstance(normalised.get("motionSegments"), list) else 0,
            "videoSegments": len(normalised.get("videoSegments", [])) if isinstance(normalised.get("videoSegments"), list) else 0,
            "audioSegments": len(normalised.get("audioSegments", [])) if isinstance(normalised.get("audioSegments"), list) else 0,
            "truth": "Shotboard V4 is normalized to the existing multigeneration V3 backend contract. No previous timeline fallback is allowed.",
        }
        report = _json_dump(report_payload)
        resources["cine_shotboard_v4_multigen_adapter_report"] = report
        return out_linx, normalised_json, report


def _make_concat_plan(index: Dict[str, Any], source_bus: str) -> Dict[str, Any]:
    takes = index.get("takes") if isinstance(index.get("takes"), list) else []
    contract = _make_take_audio_contract(takes)
    return {
        "schema": "iamccs.multigeneration.concat_plan",
        "schema_version": 1,
        "source": "IAMCCS_MultiTimelineBridge",
        "source_bus": str(source_bus),
        "final_audio_policy": "restore_original_master_or_selected_bus_after_video_concat",
        "video_concat_policy": "hard_cut_in_take_order",
        "take_audio_contract": contract,
        "takes": [{
            "take_index": _safe_int(take.get("take_index", idx + 1), idx + 1),
            "timeline_id": str(take.get("timeline_id", f"T{idx + 1:02d}")),
            "audio_lane": contract[idx]["audio_lane"] if idx < len(contract) else _take_audio_lane_name(idx + 1),
            "global_start_frames": _safe_int(take.get("global_start_frames", 0), 0),
            "duration_frames": _safe_int(take.get("duration_frames", 0), 0),
            "expected_video_slot": f"video_take_{idx + 1:02d}",
        } for idx, take in enumerate(takes)],
    }


class IAMCCS_MultiTimelineBridge:
    """Build a sequenced take index from BusOut audio for chunked video-driven generation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chunk_template": (["10s", "15s", "20s", "25s", "custom"], {"default": "20s"}),
                "custom_chunk_seconds": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 300.0, "step": 0.25}),
                "source_bus": (["master_out", "track_1", "track_2", "track_3", "track_4", "track_5"], {"default": "master_out"}),
                "take_source_mode": (["auto_detect_multi_lanes", "chunk_source_bus"], {"default": "auto_detect_multi_lanes"}),
                "take_count_mode": (["auto_from_audio", "fixed_take_count"], {"default": "auto_from_audio"}),
                "fixed_take_count": ("INT", {"default": 3, "min": 1, "max": 64, "step": 1}),
                "max_takes": ("INT", {"default": 12, "min": 1, "max": 64, "step": 1}),
                "active_take": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "take_track_layout": (["collapse_to_lane_1", "preserve_bus_tracks"], {"default": "collapse_to_lane_1"}),
            },
            "optional": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "bus_manifest_json": ("STRING", {"default": "", "multiline": True}),
                "master_out_json": ("STRING", {"default": "", "multiline": True}),
                "track_1_json": ("STRING", {"default": "", "multiline": True}),
                "track_2_json": ("STRING", {"default": "", "multiline": True}),
                "track_3_json": ("STRING", {"default": "", "multiline": True}),
                "track_4_json": ("STRING", {"default": "", "multiline": True}),
                "track_5_json": ("STRING", {"default": "", "multiline": True}),
                "visual_timelines_json": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE, "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("cine_linx", "generation_index_json", "active_take_json", "concat_plan_json", "report")
    FUNCTION = "build"
    CATEGORY = "IAMCCS/Cine/Multigeneration"

    def build(
        self,
        chunk_template,
        custom_chunk_seconds,
        source_bus,
        take_source_mode,
        take_count_mode,
        fixed_take_count,
        max_takes,
        active_take,
        frame_rate,
        take_track_layout,
        cine_linx=None,
        bus_manifest_json="",
        master_out_json="",
        track_1_json="",
        track_2_json="",
        track_3_json="",
        track_4_json="",
        track_5_json="",
        visual_timelines_json="",
    ):
        fps = max(1.0, float(frame_rate))
        chunk_seconds = _seconds_for_template(chunk_template, custom_chunk_seconds)
        chunk_frames = max(1, int(round(chunk_seconds * fps)))
        max_take_count = max(1, _safe_int(max_takes, 12))
        manifest = _bus_manifest(
            cine_linx,
            bus_manifest_json,
            master_out_json,
            (track_1_json, track_2_json, track_3_json, track_4_json, track_5_json),
        )
        source = _source_from_manifest(manifest, source_bus)
        source_segments = _segments(source.get("segments"))
        source_duration = max(_safe_int(source.get("duration_frames", 0), 0), _max_end(source_segments))
        if source_duration <= 0:
            source_duration = chunk_frames * max(1, _safe_int(fixed_take_count, 3))

        if str(take_count_mode) == "fixed_take_count":
            take_count = max(1, min(max_take_count, _safe_int(fixed_take_count, 3)))
        else:
            take_count = max(1, min(max_take_count, int(math.ceil(source_duration / max(1, chunk_frames)))))

        visual_timelines = _safe_json_loads(visual_timelines_json, {})
        if not isinstance(visual_timelines, (dict, list)):
            visual_timelines = {}

        takes = []
        if str(take_source_mode) == "auto_detect_multi_lanes":
            prechunked_segments = _collect_prechunked_segments(manifest, source_segments)
            if prechunked_segments:
                prechunked_end = max([
                    _safe_int(seg.get("sourceGlobalEnd", seg.get("globalEnd", _safe_int(seg.get("start", 0), 0) + _safe_int(seg.get("length", 1), 1))), 0)
                    for seg in prechunked_segments
                ] or [0])
                source_duration = max(source_duration, prechunked_end)
            takes = _takes_from_prechunked_segments(
                prechunked_segments,
                fps,
                chunk_frames,
                chunk_seconds,
                str(source_bus),
                str(take_track_layout),
                visual_timelines,
            )

        if not takes:
            for index in range(take_count):
                global_start = index * chunk_frames
                remaining = max(0, source_duration - global_start)
                duration = chunk_frames if str(take_count_mode) == "fixed_take_count" else min(chunk_frames, remaining or chunk_frames)
                duration = max(1, int(duration))
                timeline_id = f"T{index + 1:02d}"
                audio_segments = _slice_segments(source_segments, global_start, duration, str(take_track_layout))
                take = {
                    "schema": "iamccs.multigeneration.take",
                    "schema_version": 1,
                    "take_index": index + 1,
                    "timeline_id": timeline_id,
                    "source_bus": str(source_bus),
                    "global_start_frames": int(global_start),
                    "global_end_frames": int(global_start + duration),
                    "local_start_frames": 0,
                    "duration_frames": int(duration),
                    "duration_seconds": duration / fps,
                    "chunk_frames": int(chunk_frames),
                    "chunk_seconds": float(chunk_seconds),
                    "audioSegments": audio_segments,
                    "audioTrackCount": max([_safe_int(seg.get("track", 0), 0) + 1 for seg in audio_segments] or [1]),
                    "visual_timeline_key": timeline_id,
                    "visual_timeline": visual_timelines.get(timeline_id) if isinstance(visual_timelines, dict) else None,
                    "prechunked": False,
                }
                take["take_audio_timeline"] = _timeline_for_take(take, fps, str(take_track_layout))
                takes.append(take)

        roll_contract = _roll_contract_from_linx(cine_linx, fps)
        if roll_contract.get("enabled") and roll_contract.get("frames", 0) > 0:
            source_master_segment = _master_source_segment_from_manifest(manifest)
            for take in takes:
                _apply_roll_to_take(take, roll_contract, fps, source_master_segment, str(take_track_layout))
                take["take_audio_timeline"] = _timeline_for_take(take, fps, str(take_track_layout))

        shotboard_identity = _shotboard_timeline_identity_from_linx(cine_linx)
        upstream_identity = shotboard_identity or _active_identity_from_linx(cine_linx)
        upstream_take = _safe_int(upstream_identity.get("take_index"), 0) if upstream_identity else 0

        if shotboard_identity and upstream_take > len(takes) and len(takes) == 1:
            local_take = takes[0]
            old_take = max(1, _safe_int(local_take.get("take_index"), 1))
            local_take["take_index"] = upstream_take
            local_take["timeline_id"] = str(upstream_identity.get("timeline_id") or _take_timeline_id(upstream_take))
            local_take["audio_lane"] = _take_audio_lane_name(upstream_take)
            local_take["audio_track_index"] = upstream_take - 1
            local_take["timeline_audio_contract"] = f"{local_take['timeline_id']}->{local_take['audio_lane']}"
            local_take["visual_timeline_key"] = local_take["timeline_id"]
            if isinstance(visual_timelines, dict):
                local_take["visual_timeline"] = visual_timelines.get(local_take["timeline_id"]) or local_take.get("visual_timeline")
            # A published single chunk is local to its own file and therefore
            # often starts at frame 0 even when Shotboard selected T02/T03.
            # Restore its nominal global position before applying roll again;
            # otherwise later takes receive post-roll but never pre-roll.
            if roll_contract.get("enabled") and roll_contract.get("frames", 0) > 0:
                nominal_duration = max(
                    1,
                    _safe_int(
                        local_take.get("nominal_duration_frames", local_take.get("duration_frames", chunk_frames)),
                        chunk_frames,
                    ),
                )
                nominal_start = max(0, (upstream_take - 1) * chunk_frames)
                local_take["global_start_frames"] = int(nominal_start)
                local_take["global_end_frames"] = int(nominal_start + nominal_duration)
                local_take["source_global_start_frames"] = int(nominal_start)
                local_take["source_global_end_frames"] = int(nominal_start + nominal_duration)
                _apply_roll_to_take(
                    local_take,
                    roll_contract,
                    fps,
                    _master_source_segment_from_manifest(manifest),
                    str(take_track_layout),
                )
                local_take["take_audio_timeline"] = _timeline_for_take(local_take, fps, str(take_track_layout))
            local_take["take_audio_timeline"] = _timeline_for_take(local_take, fps, str(take_track_layout))
            print(
                "[IAMCCS MultiTimelineBridge] LOCAL_ACTIVE_TAKE_REMAP "
                f"from=T{old_take:02d} "
                f"to=T{upstream_take:02d} "
                f"timeline={local_take.get('timeline_id')} "
                f"reason=single_local_audio_chunk "
                f"nominal_start_frames={local_take.get('nominal_global_start_frames', local_take.get('global_start_frames', 0))} "
                f"generation_duration_frames={local_take.get('generation_duration_frames', local_take.get('duration_frames', 0))}"
            )

        requested_active_take = max(1, _safe_int(active_take, 1))
        active_index = 0
        for idx, take in enumerate(takes):
            candidate_take = max(1, _safe_int(take.get("take_index", idx + 1), idx + 1))
            if candidate_take == requested_active_take:
                active_index = idx
                break

        if shotboard_identity and upstream_take <= 0:
            raise ValueError(
                "IAMCCS MultiTimelineBridge hard-fail: Shotboard active timeline is missing a valid T/A identity."
            )
        if shotboard_identity and not any(max(1, _safe_int(take.get("take_index", idx + 1), idx + 1)) == upstream_take for idx, take in enumerate(takes)):
            available = ",".join([f"T{max(1, _safe_int(take.get('take_index', idx + 1), idx + 1)):02d}" for idx, take in enumerate(takes)])
            raise ValueError(
                "IAMCCS MultiTimelineBridge hard-fail: Shotboard active timeline is outside the generated take index. "
                f"shotboard_take=T{upstream_take:02d}, available_takes={available or 'none'}."
            )
        if upstream_take > 0 and any(max(1, _safe_int(take.get("take_index", idx + 1), idx + 1)) == upstream_take for idx, take in enumerate(takes)):
            active_index = upstream_take - 1
            for idx, take in enumerate(takes):
                candidate_take = max(1, _safe_int(take.get("take_index", idx + 1), idx + 1))
                if candidate_take == upstream_take:
                    active_index = idx
                    break
            sync_source = "SHOTBOARD_TIMELINE" if shotboard_identity else "cine_linx_identity"
            if upstream_take != requested_active_take:
                print(
                    "[IAMCCS MultiTimelineBridge] ACTIVE_TAKE_SYNC "
                    f"mode={sync_source} "
                    f"source={upstream_identity.get('source', 'unknown')} "
                    f"widget_take=T{requested_active_take:02d} "
                    f"timeline_take=T{upstream_take:02d}"
                )
        for idx, take in enumerate(takes):
            take_index = max(1, _safe_int(take.get("take_index", idx + 1), idx + 1))
            take["take_index"] = take_index
            take["timeline_id"] = str(take.get("timeline_id") or _take_timeline_id(take_index))
            take["audio_lane"] = _take_audio_lane_name(take_index)
            take["audio_track_index"] = take_index - 1
            take["timeline_audio_contract"] = f"{take['timeline_id']}->{take['audio_lane']}"

        take_audio_contract = _make_take_audio_contract(takes)
        generation_index = {
            "schema": "iamccs.multigeneration.index",
            "schema_version": 2,
            "source": "IAMCCS_MultiTimelineBridge",
            "frame_rate": float(fps),
            "chunk_template": str(chunk_template),
            "chunk_seconds": float(chunk_seconds),
            "chunk_frames": int(chunk_frames),
            "source_bus": str(source_bus),
            "take_source_mode": str(take_source_mode),
            "source_duration_frames": int(source_duration),
            "source_duration_seconds": source_duration / fps,
            "take_count": int(len(takes)),
            "active_take": int(max(1, _safe_int(takes[active_index].get("take_index", active_index + 1), active_index + 1))),
            "active_timeline_id": str(takes[active_index].get("timeline_id") or _take_timeline_id(takes[active_index].get("take_index", active_index + 1))),
            "active_audio_lane": _take_audio_lane_name(takes[active_index].get("take_index", active_index + 1)),
            "take_track_layout": str(take_track_layout),
            "takes": takes,
            "take_audio_contract": take_audio_contract,
            "roll_contract": copy.deepcopy(roll_contract),
            "roll_enabled": bool(roll_contract.get("enabled")),
            "bus_generation_index": manifest.get("generation_index") if isinstance(manifest.get("generation_index"), dict) else {},
            # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
            # tail_trim_frames=0: trimming is handled by IAMCCS_LTXVideoDurationCrop before VideoEditor parks.
            "tail_trim_frames": 0,
            "truth": "T1=A1, T2=A2, T3=A3. Audio remains BusOut/AudioBoard custom-audio metadata. Each take receives a local audio window for sequential video-driven generation, then video takes are hard-concatenated.",
        }
        active_take = takes[active_index]
        print(
            "[IAMCCS MultiTimelineBridge] ACTIVE_TAKE_COMMIT "
            f"take={max(1, _safe_int(active_take.get('take_index', active_index + 1), active_index + 1))} "
            f"timeline={active_take.get('timeline_id')} "
            f"audio_lane={_take_audio_lane_name(active_take.get('take_index', active_index + 1))} "
            f"duration_frames={_safe_int(active_take.get('duration_frames'), 0)} "
            f"visual_segments={len(active_take.get('visual_timeline', {}).get('segments', []) if isinstance(active_take.get('visual_timeline'), dict) else [])}"
        )
        sequence_plan = _make_sequence_plan(generation_index)
        concat_plan = _make_concat_plan(generation_index, str(source_bus))

        out_linx = _clone_linx(cine_linx)
        resources = _resources(out_linx)
        outputs = _outputs(out_linx)
        resources["cine_multigeneration_concat_plan"] = concat_plan
        resources["cine_multigeneration_concat_plan_json"] = _json_dump(concat_plan)
        resources["cine_multigeneration_sequence_plan"] = sequence_plan
        resources["cine_multigeneration_sequence_plan_json"] = _json_dump(sequence_plan)
        resources["cine_multigeneration_take_audio_contract"] = take_audio_contract
        resources["cine_multigeneration_bus_manifest"] = manifest
        _apply_active_take(out_linx, generation_index, active_take, str(take_track_layout))
        active_take_number = max(1, _safe_int(active_take.get("take_index", active_index + 1), active_index + 1))
        active_timeline_id = str(active_take.get("timeline_id") or _take_timeline_id(active_take_number))
        active_audio_lane = _take_audio_lane_name(active_take_number)
        outputs["concat_plan_json"] = _json_dump(concat_plan)
        out_linx["type"] = SUPERNODE_LINX_TYPE
        out_linx["mode"] = "iamccs_multigeneration"
        out_linx.setdefault("chain", []).append({
            "role": "multigeneration_bridge",
            "name": "IAMCCS_MultiTimelineBridge",
            "active_take": active_take_number,
            "timeline_id": active_timeline_id,
            "audio_lane": active_audio_lane,
        })
        _refresh_linx_index(out_linx)

        report = _json_dump({
            "node": "IAMCCS_MultiTimelineBridge",
            "source_bus": str(source_bus),
            "chunk_seconds": float(chunk_seconds),
            "chunk_frames": int(chunk_frames),
            "take_count": len(takes),
            "active_take": active_take_number,
            "source_segments": len(source_segments),
            "active_segments": len(active_take.get("audioSegments", [])),
            "roll_contract": roll_contract,
            "roll_audio_source": bool(_master_source_segment_from_manifest(manifest)),
            "prechunked": bool(active_take.get("prechunked", False)),
            "active_timeline_id": active_timeline_id,
            "active_audio_lane": active_audio_lane,
            "contract": [item.get("mapping") for item in take_audio_contract],
            "concat_policy": concat_plan["video_concat_policy"],
            "sequence_steps": len(sequence_plan.get("steps", [])),
        })
        outputs["report"] = report
        return out_linx, _json_dump(generation_index), _json_dump(active_take), _json_dump(concat_plan), report



class IAMCCS_MultiTimelineSequentialPicker:
    """Expose T1-A1..T5-A5 as parallel cine_linx outputs for one-queue staged generation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generation_index_json": ("STRING", {"default": "", "multiline": True}),
                "take_track_layout": (["collapse_to_lane_1", "preserve_bus_tracks"], {"default": "collapse_to_lane_1"}),
                "enabled_takes": ("INT", {"default": 3, "min": 1, "max": 5, "step": 1}),
            },
            "optional": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
            },
        }

    RETURN_TYPES = (
        SUPERNODE_LINX_TYPE,
        SUPERNODE_LINX_TYPE,
        SUPERNODE_LINX_TYPE,
        SUPERNODE_LINX_TYPE,
        SUPERNODE_LINX_TYPE,
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "cine_linx_T1_A1",
        "cine_linx_T2_A2",
        "cine_linx_T3_A3",
        "cine_linx_T4_A4",
        "cine_linx_T5_A5",
        "sequence_plan_json",
        "report",
    )
    FUNCTION = "pick_sequence"
    CATEGORY = "IAMCCS/Cine/Multigeneration"

    def pick_sequence(self, generation_index_json, take_track_layout, enabled_takes, cine_linx=None):
        generation_index = _safe_json_loads(generation_index_json, {})
        if not isinstance(generation_index, dict):
            generation_index = {}
        takes = generation_index.get("takes") if isinstance(generation_index.get("takes"), list) else []
        if not takes:
            generation_index.setdefault("frame_rate", 24.0)
            takes = []
            for index in range(max(1, min(5, _safe_int(enabled_takes, 3)))):
                take = {
                    "schema": "iamccs.multigeneration.take",
                    "schema_version": 1,
                    "take_index": index + 1,
                    "timeline_id": _take_timeline_id(index + 1),
                    "audio_lane": _take_audio_lane_name(index + 1),
                    "duration_frames": 1,
                    "audioSegments": [],
                }
                takes.append(take)
            generation_index["takes"] = takes
        max_enabled = max(1, min(5, _safe_int(enabled_takes, 3)))
        outputs: List[Dict[str, Any]] = []
        steps: List[Dict[str, Any]] = []
        for index in range(5):
            source_take = copy.deepcopy(takes[index]) if index < len(takes) else {
                "schema": "iamccs.multigeneration.take",
                "schema_version": 1,
                "take_index": index + 1,
                "timeline_id": _take_timeline_id(index + 1),
                "audio_lane": _take_audio_lane_name(index + 1),
                "duration_frames": 1,
                "audioSegments": [],
                "disabled": True,
            }
            take_index = index + 1
            source_take["take_index"] = take_index
            source_take["timeline_id"] = str(source_take.get("timeline_id") or _take_timeline_id(take_index))
            source_take["audio_lane"] = _take_audio_lane_name(take_index)
            source_take["audio_track_index"] = take_index - 1
            source_take["sequence_enabled"] = take_index <= max_enabled and index < len(takes)
            out_linx = _clone_linx(cine_linx)
            local_index = copy.deepcopy(generation_index)
            local_index["active_take"] = take_index
            local_index["active_timeline_id"] = source_take["timeline_id"]
            local_index["active_audio_lane"] = source_take["audio_lane"]
            _apply_active_take(out_linx, local_index, source_take, str(take_track_layout))
            out_linx["type"] = SUPERNODE_LINX_TYPE
            out_linx["mode"] = "iamccs_multigeneration_sequence_take"
            out_linx.setdefault("chain", []).append({
                "role": "multigeneration_sequence_picker",
                "name": "IAMCCS_MultiTimelineSequentialPicker",
                "take_index": take_index,
                "timeline_id": source_take["timeline_id"],
                "audio_lane": source_take["audio_lane"],
                "enabled": bool(source_take["sequence_enabled"]),
            })
            _outputs(out_linx)["report"] = _json_dump({
                "node": "IAMCCS_MultiTimelineSequentialPicker",
                "take_index": take_index,
                "timeline_id": source_take["timeline_id"],
                "audio_lane": source_take["audio_lane"],
                "enabled": bool(source_take["sequence_enabled"]),
            })
            _refresh_linx_index(out_linx)
            outputs.append(out_linx)
            steps.append({
                "step_index": take_index,
                "take_index": take_index,
                "timeline_id": source_take["timeline_id"],
                "audio_lane": source_take["audio_lane"],
                "enabled": bool(source_take["sequence_enabled"]),
                "output": f"cine_linx_T{take_index}_A{take_index}",
                "expected_video_slot": f"video_take_{take_index:02d}",
            })
        sequence_plan = {
            "schema": "iamccs.multigeneration.sequence_plan",
            "schema_version": 2,
            "source": "IAMCCS_MultiTimelineSequentialPicker",
            "enabled_takes": max_enabled,
            "steps": steps,
            "truth": "Connect each enabled cine_linx_Tn_An output to its own generation branch, then connect generated videos to IAMCCS Shotboard Video Editor in the same order.",
        }
        report = _json_dump({
            "node": "IAMCCS_MultiTimelineSequentialPicker",
            "enabled_takes": max_enabled,
            "outputs": [f"T{idx + 1}/A{idx + 1}" for idx in range(5)],
            "queue_mode": "parallel_branches_in_one_comfy_queue",
        })
        return (*outputs, _json_dump(sequence_plan), report)


class IAMCCS_MultiTimelineTakePicker:
    """Pick one take from a MultiTimelineBridge index and expose it as active cine_linx audio."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generation_index_json": ("STRING", {"default": "", "multiline": True}),
                "take_index": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "take_track_layout": (["collapse_to_lane_1", "preserve_bus_tracks"], {"default": "collapse_to_lane_1"}),
            },
            "optional": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
            },
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("cine_linx", "active_take_json", "take_audio_timeline_json", "report")
    FUNCTION = "pick"
    CATEGORY = "IAMCCS/Cine/Multigeneration"

    def pick(self, generation_index_json, take_index, take_track_layout, cine_linx=None):
        generation_index = _safe_json_loads(generation_index_json, {})
        if not isinstance(generation_index, dict):
            generation_index = {}
        takes = generation_index.get("takes") if isinstance(generation_index.get("takes"), list) else []
        if not takes:
            empty = {
                "schema": "iamccs.multigeneration.take",
                "schema_version": 1,
                "take_index": 1,
                "timeline_id": "T01",
                "duration_frames": 1,
                "audioSegments": [],
            }
            takes = [empty]
            generation_index["takes"] = takes
            generation_index.setdefault("frame_rate", 24.0)
        active_index = max(1, min(len(takes), _safe_int(take_index, 1))) - 1
        active_take = copy.deepcopy(takes[active_index])
        generation_index["active_take"] = active_index + 1
        out_linx = _clone_linx(cine_linx)
        _apply_active_take(out_linx, generation_index, active_take, str(take_track_layout))
        out_linx["type"] = SUPERNODE_LINX_TYPE
        out_linx["mode"] = "iamccs_multigeneration_take"
        out_linx.setdefault("chain", []).append({
            "role": "multigeneration_take_picker",
            "name": "IAMCCS_MultiTimelineTakePicker",
            "active_take": active_index + 1,
        })
        _refresh_linx_index(out_linx)
        take_timeline = _timeline_for_take(active_take, _safe_float(generation_index.get("frame_rate", 24.0), 24.0), str(take_track_layout))
        report = _json_dump({
            "node": "IAMCCS_MultiTimelineTakePicker",
            "active_take": active_index + 1,
            "timeline_id": str(active_take.get("timeline_id", "")),
            "duration_frames": _safe_int(active_take.get("duration_frames", 0), 0),
            "audio_segments": len(active_take.get("audioSegments", []) if isinstance(active_take.get("audioSegments"), list) else []),
        })
        _outputs(out_linx)["report"] = report
        return out_linx, _json_dump(active_take), _json_dump(take_timeline), report


def _video_components(video: Any):
    if video is None:
        return None
    if not hasattr(video, "get_components"):
        raise ValueError("IAMCCS Video Hard Concat: input is not a Comfy VIDEO object.")
    return video.get_components()


def _input_media_path(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        return ""
    if os.path.isabs(text) and os.path.exists(text):
        return text
    clean = text.lstrip("/").replace("..", "")
    roots = []
    if folder_paths is not None:
        for getter_name in ("get_input_directory", "get_output_directory", "get_temp_directory"):
            getter = getattr(folder_paths, getter_name, None)
            if callable(getter):
                try:
                    roots.append(getter())
                except Exception:
                    pass
    for root in roots:
        candidate = os.path.abspath(os.path.join(root, clean.replace("/", os.sep)))
        try:
            if os.path.commonpath((os.path.abspath(root), candidate)) == os.path.abspath(root) and os.path.exists(candidate):
                return candidate
        except Exception:
            if os.path.exists(candidate):
                return candidate
    return ""


def _manual_editor_clips(edits: Any, media_type: str) -> List[Tuple[str, Dict[str, Any]]]:
    if not isinstance(edits, dict):
        return []
    ui_state = edits.get("ui_state") if isinstance(edits.get("ui_state"), dict) else {}
    clips = ui_state.get("clips") if isinstance(ui_state.get("clips"), dict) else {}
    out: List[Tuple[str, Dict[str, Any]]] = []
    for key, item in clips.items():
        if not isinstance(item, dict):
            continue
        if not bool(item.get("manual")):
            continue
        if str(item.get("type") or "").lower() != media_type:
            continue
        media_path = str(item.get("mediaPath") or item.get("media_path") or item.get("path") or "").strip()
        if not media_path:
            continue
        out.append((str(key), copy.deepcopy(item)))
    return sorted(out, key=lambda pair: (_safe_float(pair[1].get("start", 0), 0.0), pair[0]))


def _load_manual_audio_clip(item: Dict[str, Any]) -> Dict[str, Any] | None:
    path = _input_media_path(item.get("mediaPath") or item.get("media_path") or item.get("path"))
    if not path:
        return None
    waveform, sample_rate = _load_audio_waveform_file(path)
    return {"waveform": waveform, "sample_rate": int(sample_rate or 44100)}


def _load_audio_waveform_file(path: str) -> Tuple[torch.Tensor, int]:
    wav_exc = None
    sf_exc = None
    torch_exc = None
    ext = os.path.splitext(str(path or ""))[1].lower()
    if ext == ".wav":
        try:
            # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
            import numpy as np  # type: ignore
            from scipy.io import wavfile as scipy_wavfile  # type: ignore

            sample_rate, data = scipy_wavfile.read(path)
            if getattr(data, "ndim", 1) == 1:
                data = data.reshape(-1, 1)
            raw_dtype = getattr(data, "dtype", None)
            if raw_dtype is not None and raw_dtype.kind in {"i", "u"}:
                info = np.iinfo(raw_dtype)
                scale = float(max(abs(int(info.min)), abs(int(info.max))) or 1.0)
                data = data.astype(np.float32, copy=False) / scale
            else:
                data = data.astype(np.float32, copy=False)
            waveform = torch.from_numpy(data.T).unsqueeze(0).contiguous()
            return waveform, int(sample_rate or 44100)
        except Exception as exc:
            wav_exc = exc
    try:
        # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        import soundfile as sf  # type: ignore

        data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
        waveform = torch.from_numpy(data.T).unsqueeze(0).contiguous()
        return waveform, int(sample_rate or 44100)
    except Exception as exc:
        sf_exc = exc
    try:
        # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        waveform, sample_rate = torchaudio.load(path)
        if waveform.dim() == 1:
            waveform = waveform.reshape(1, -1)
        return waveform.unsqueeze(0).contiguous(), int(sample_rate or 44100)
    except Exception as exc:
        torch_exc = exc
    raise ValueError(
        f"IAMCCS Shotboard Video Editor: could not load audio file {path!r}; "
        f"wav={wav_exc!r}; soundfile={sf_exc!r}; torchaudio={torch_exc!r}"
    )


def _audio_manifest_path(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    direct = str(
        item.get("path")
        or item.get("mediaPath")
        or item.get("media_path")
        or item.get("package_path")
        or ""
    ).strip()
    if direct:
        found = _input_media_path(direct)
        if found:
            return found
    filename = str(item.get("audio_preview_file") or item.get("filename") or "").strip()
    if not filename:
        return ""
    subfolder = str(item.get("audio_preview_subfolder") or item.get("subfolder") or "").strip().replace("\\", "/")
    audio_type = str(item.get("audio_preview_type") or item.get("type") or "output").strip().lower()
    roots = []
    if folder_paths is not None:
        getter_names = {
            "input": ("get_input_directory",),
            "output": ("get_output_directory",),
            "temp": ("get_temp_directory",),
        }.get(audio_type, ("get_output_directory", "get_input_directory", "get_temp_directory"))
        for getter_name in getter_names:
            getter = getattr(folder_paths, getter_name, None)
            if callable(getter):
                try:
                    roots.append(str(getter()))
                except Exception:
                    pass
    for root in roots:
        candidate = os.path.abspath(os.path.join(root, subfolder.replace("/", os.sep), filename))
        try:
            if os.path.commonpath((os.path.abspath(root), candidate)) == os.path.abspath(root) and os.path.exists(candidate):
                return candidate
        except Exception:
            if os.path.exists(candidate):
                return candidate
    return ""


def _load_audio_manifest_entry(item: Dict[str, Any]) -> Dict[str, Any] | None:
    path = _audio_manifest_path(item)
    if not path:
        return None
    waveform, sample_rate = _load_audio_waveform_file(path)
    trim_start = _safe_float(
        item.get("_render_trim_start_seconds")
        or item.get("renderTrimStartSeconds")
        or item.get("render_trim_start_seconds"),
        0.0,
    )
    trim_end = _safe_float(
        item.get("_render_trim_end_seconds")
        or item.get("renderTrimEndSeconds")
        or item.get("render_trim_end_seconds"),
        0.0,
    )
    if trim_end <= trim_start:
        pre_roll_frames = _safe_int(item.get("preRollFrames") or item.get("pre_roll_frames"), 0)
        nominal_duration = _safe_float(item.get("nominalDurationSeconds") or item.get("nominal_duration_seconds"), 0.0)
        if pre_roll_frames > 0:
            trim_start = pre_roll_frames / max(1.0, _safe_float(item.get("fps"), 24.0))
        if nominal_duration > 0:
            trim_end = trim_start + nominal_duration
    if trim_end > trim_start:
        sample_count = max(1, int(waveform.shape[-1]))
        start_sample = max(0, min(sample_count - 1, int(round(trim_start * int(sample_rate or 44100)))))
        end_sample = max(start_sample + 1, min(sample_count, int(round(trim_end * int(sample_rate or 44100)))))
        waveform = waveform[..., start_sample:end_sample].contiguous()
    return {"waveform": waveform, "sample_rate": int(sample_rate or 44100)}


def _manifest_master_audio_item(manifest: Dict[str, Any], assets: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(manifest, dict):
        return {}
    for key in ("master_excerpt", "master_audio"):
        item = manifest.get(key)
        if isinstance(item, dict):
            return item
    if not isinstance(assets, dict):
        assets = {}
    for key in ("master_excerpt", "master_audio"):
        item = assets.get(key)
        if isinstance(item, dict):
            return item
    clips = manifest.get("clips") if isinstance(manifest.get("clips"), list) else []
    for clip in clips:
        if not isinstance(clip, dict):
            continue
        track_id = str(clip.get("trackId") or "").strip().upper()
        role = str(clip.get("role") or "").strip().lower()
        lane = str(clip.get("audioLane") or "").strip().upper()
        if track_id not in {"AM", "MASTER"} and lane != "MASTER" and role not in {"master_audio", "master_excerpt"}:
            continue
        asset = assets.get(str(clip.get("assetId") or ""))
        if isinstance(asset, dict):
            return asset
    for key, item in assets.items():
        if not isinstance(item, dict):
            continue
        text = " ".join([
            str(key),
            str(item.get("id") or ""),
            str(item.get("role") or ""),
            str(item.get("trackId") or ""),
            str(item.get("audioLane") or ""),
            str(item.get("timelineId") or ""),
        ]).lower()
        if "master" in text:
            return item
    return {}


def _master_audio_asset_from_linx(linx: Any) -> Dict[str, Any]:
    if not isinstance(linx, dict):
        return {}
    resources = _resources(linx)
    outputs = _outputs(linx)
    payload = _payload(linx)
    containers = (linx, resources, outputs, payload)
    candidates: List[Dict[str, Any]] = []
    for container in containers:
        for key in ("cine_audio_master_audio_asset", "master_audio_asset", "masterAudioAsset", "master_audio", "master_excerpt"):
            item = container.get(key)
            if isinstance(item, dict):
                candidates.append(item)
        multi = container.get("multiGeneration")
        if isinstance(multi, dict):
            for key in ("master_audio_asset", "masterAudioAsset", "master_audio", "masterExcerpt"):
                item = multi.get(key)
                if isinstance(item, dict):
                    candidates.append(item)
    for item in candidates:
        asset = copy.deepcopy(item)
        if not str(asset.get("path") or asset.get("audioFile") or asset.get("filename") or asset.get("fileName") or "").strip():
            continue
        asset.setdefault("id", "master_audio")
        asset.setdefault("role", "master_audio")
        asset.setdefault("audioLane", "MASTER")
        asset.setdefault("timelineId", "MASTER")
        return asset
    return {}


def _master_audio_fingerprint(item: Any) -> Tuple[str, ...]:
    if not isinstance(item, dict):
        return ()
    return tuple(str(item.get(key) or "").strip() for key in (
        "masterRangeSignature",
        "sourceSegmentId",
        "sourceAudioFile",
        "audioFile",
        "path",
        "filename",
        "fileName",
        "physicalStartFrame",
        "physicalDurationFrames",
        "duration_seconds",
        "duration",
    ))


def _replace_manifest_master_audio(manifest: Dict[str, Any], master_asset: Dict[str, Any], fps: float) -> bool:
    if not isinstance(manifest, dict) or not isinstance(master_asset, dict):
        return False
    asset = copy.deepcopy(master_asset)
    if not str(asset.get("path") or asset.get("audioFile") or asset.get("filename") or asset.get("fileName") or "").strip():
        return False
    asset.update({
        "id": "master_audio",
        "role": "master_audio",
        "type": "audio",
        "takeIndex": 0,
        "timelineId": "MASTER",
        "audioLane": "MASTER",
    })
    duration = _safe_float(
        asset.get("duration_seconds") or asset.get("duration") or asset.get("physicalDurationFrames", 0) / max(1.0, fps),
        0.0,
    )
    assets = manifest.setdefault("assets", {})
    if not isinstance(assets, dict):
        assets = {}
        manifest["assets"] = assets
    # There can be only one active master. Leaving master_excerpt behind makes
    # legacy lookup pick an older song before the newly published master_audio.
    manifest.pop("master_excerpt", None)
    manifest.pop("master_audio", None)
    assets.pop("master_excerpt", None)
    assets.pop("master_audio", None)
    manifest["master_audio"] = copy.deepcopy(asset)
    assets["master_audio"] = copy.deepcopy(asset)
    clips = manifest.setdefault("clips", [])
    if not isinstance(clips, list):
        clips = []
        manifest["clips"] = clips
    clips[:] = [
        clip for clip in clips
        if not (
            isinstance(clip, dict)
            and (
                str(clip.get("id") or "") == "clip_MASTER_AUDIO"
                or str(clip.get("trackId") or "").strip().upper() in {"AM", "MASTER"}
                or str(clip.get("audioLane") or "").strip().upper() == "MASTER"
                or str(clip.get("role") or "").strip().lower() in {"master_audio", "master_excerpt"}
            )
        )
    ]
    if duration > 0:
        clips.append({
            "id": "clip_MASTER_AUDIO",
            "assetId": "master_audio",
            "type": "audio",
            "takeIndex": 0,
            "timelineId": "MASTER",
            "audioLane": "MASTER",
            "startTime": 0.0,
            "duration": float(duration),
            "trimStart": 0.0,
            "trimEnd": float(duration),
            "trackId": "AM",
            "trackIndex": 10,
            "muted": False,
            "volume": 1.0,
            "linkedClipIds": [],
            "role": "master_audio",
        })
    _update_manifest_duration(manifest)
    return True


def _manifest_master_audio_clip(manifest: Dict[str, Any]) -> Dict[str, Any]:
    clips = manifest.get("clips") if isinstance(manifest.get("clips"), list) else []
    for clip in clips:
        if not isinstance(clip, dict):
            continue
        track_id = str(clip.get("trackId") or "").strip().upper()
        role = str(clip.get("role") or "").strip().lower()
        lane = str(clip.get("audioLane") or "").strip().upper()
        if track_id in {"AM", "MASTER"} or lane == "MASTER" or role in {"master_audio", "master_excerpt"}:
            return clip
    return {}


def _manifest_has_active_master_audio_clip(manifest: Dict[str, Any]) -> bool:
    clips = manifest.get("clips") if isinstance(manifest.get("clips"), list) else []
    for clip in clips:
        if not isinstance(clip, dict) or bool(clip.get("muted")):
            continue
        track_id = str(clip.get("trackId") or "").strip().upper()
        role = str(clip.get("role") or "").strip().lower()
        lane = str(clip.get("audioLane") or "").strip().upper()
        if track_id in {"AM", "MASTER"} or lane == "MASTER" or role in {"master_audio", "master_excerpt"}:
            return True
    return False


def _load_manual_video_clip(item: Dict[str, Any]) -> Any:
    path = _input_media_path(item.get("mediaPath") or item.get("media_path") or item.get("path"))
    if not path:
        raise ValueError(f"IAMCCS Shotboard Video Editor: manual video file not found: {item.get('mediaPath') or item.get('path')}")
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise ValueError("IAMCCS Shotboard Video Editor: OpenCV is required to load manual video files.") from exc
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"IAMCCS Shotboard Video Editor: could not open manual video file: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    frames = []
    max_frames = max(1, _safe_int(item.get("maxFrames", 0), 0) or 3600)
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(frame).float().div_(255.0))
    cap.release()
    if not frames:
        raise ValueError(f"IAMCCS Shotboard Video Editor: manual video has no readable frames: {path}")
    images = torch.stack(frames, dim=0).contiguous()
    frame_rate = Fraction(round(max(1.0, fps) * 1000), 1000)
    audio = None
    return Types.VideoComponents(images=images, audio=audio, frame_rate=frame_rate)


def _crop_audio_to_video_frames(audio: Any, frame_count: int, fps: float) -> Any:
    if not isinstance(audio, dict) or audio.get("waveform") is None:
        return audio
    waveform = audio.get("waveform")
    if not torch.is_tensor(waveform):
        return audio
    sample_rate = int(audio.get("sample_rate") or 44100)
    target_samples = max(1, int(round((max(1, int(frame_count)) / max(1.0, float(fps))) * sample_rate)))
    if int(waveform.shape[-1]) <= target_samples:
        return audio
    trimmed = dict(audio)
    trimmed["waveform"] = waveform[..., :target_samples].contiguous()
    return trimmed


def _persist_video_take(
    session_key: Any,
    slot: int,
    video: Any,
    audio: Any = None,
    target_duration_seconds: float | None = None,
    target_duration_frames: int | None = None,
    tail_trim_frames: int | None = None,
) -> Dict[str, Any]:
    comp = _video_components(video)
    fps = float(comp.frame_rate or 24.0)
    root = _parking_root(session_key)
    filename = f"T{int(slot):02d}_A{int(slot):02d}_{int(time.time() * 1000)}.iamccs_take.pt"
    path = os.path.join(root, filename)
    # The AUDIO socket carries the original timeline waveform assembled by the
    # backend.  VIDEO.audio is the LTX audio-latent reconstruction and cannot
    # be used as an editorial/master reference: it is lossy and can drift from
    # the source that drove lip sync.  Keep it only as a legacy fallback.
    embedded_audio = audio if audio is not None else comp.audio
    images = comp.images
    original_frames = int(images.shape[0])
    exact_target_frames = max(0, _safe_int(target_duration_frames, 0))
    target_seconds = _safe_float(target_duration_seconds, 0.0)
    if exact_target_frames > 0:
        target_frames = max(1, exact_target_frames)
    elif target_seconds > 0:
        target_frames = max(1, int(round(target_seconds * max(1.0, fps))))
    else:
        target_frames = 0
    duration_crop_frames = 0
    if target_frames > 0:
        if target_frames < int(images.shape[0]):
            images = images[:target_frames].contiguous()
            embedded_audio = _crop_audio_to_video_frames(embedded_audio, target_frames, fps)
            duration_crop_frames = original_frames - int(images.shape[0])
    requested_tail_trim = max(0, _safe_int(tail_trim_frames, 0))
    effective_tail_trim = min(requested_tail_trim, max(0, int(images.shape[0]) - 1))
    if effective_tail_trim > 0:
        keep_frames = max(1, int(images.shape[0]) - int(effective_tail_trim))
        images = images[:keep_frames].contiguous()
        embedded_audio = _crop_audio_to_video_frames(embedded_audio, keep_frames, fps)
    parked_images = _images_to_parking_uint8(images)
    audio_cpu = _audio_to_cpu(embedded_audio)
    expected_bytes = parked_images.numel() * parked_images.element_size()
    if isinstance(audio_cpu, dict) and torch.is_tensor(audio_cpu.get("waveform")):
        expected_bytes += audio_cpu["waveform"].numel() * audio_cpu["waveform"].element_size()
    _ensure_disk_space_for_file(path, expected_bytes, f"T{int(slot):02d}/A{int(slot):02d}")
    payload = {
        "schema": "iamccs.video_editor.parked_take",
        "schema_version": 1,
        "slot": int(slot),
        "timeline_id": _take_timeline_id(slot),
        "audio_lane": _take_audio_lane_name(slot),
        "fps": fps,
        "frames": int(images.shape[0]),
        "duration_seconds": int(images.shape[0]) / max(1.0, fps),
        "original_frames": int(original_frames),
        "target_duration_frames": int(target_frames or 0),
        "duration_crop_frames": int(duration_crop_frames),
        "tail_trim_frames": int(effective_tail_trim),
        "images": parked_images,
        "image_storage": "uint8_0_255",
        "audio": audio_cpu,
        "created_at": time.time(),
    }
    try:
        torch.save(payload, path)
    except Exception as exc:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
        raise RuntimeError(
            "IAMCCS Shotboard Video Editor: failed to park rendered take. "
            "This usually means the ComfyUI output drive is full or the file is locked. "
            f"Target: {path}"
        ) from exc
    preview = _save_video_take_preview(
        session_key,
        slot,
        images,
        root,
        fps=fps,
        preview_stem=os.path.splitext(filename)[0],
    )
    return {
        "parking_tensor_path": path,
        "parking_tensor_file": filename,
        "parking_session": _safe_slug(session_key),
        "parking_format": "iamccs_take_tensor",
        "parking_storage": "uint8_0_255",
        "parking_frames": int(images.shape[0]),
        "parking_duration_seconds": int(images.shape[0]) / max(1.0, fps),
        "parking_original_frames": int(original_frames),
        "parking_target_frames": int(target_frames or 0),
        "parking_duration_crop_frames": int(duration_crop_frames),
        "parking_tail_trim_frames": int(effective_tail_trim),
        "parking_bytes": int(expected_bytes),
        **preview,
    }


def _load_parked_video_clip(item: Dict[str, Any]) -> Any | None:
    path = str(
        item.get("parking_tensor_path")
        or item.get("parked_tensor_path")
        or item.get("video_tensor_path")
        or ""
    ).strip()
    if not path:
        return None
    if not os.path.isabs(path):
        path = _input_media_path(path)
    if not path or not os.path.exists(path):
        return None
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict) or data.get("images") is None:
        return None
    images = data.get("images")
    if not torch.is_tensor(images):
        return None
    images = _images_from_parking_tensor(images, data.get("image_storage") or item.get("parking_storage"))
    audio = data.get("audio") if isinstance(data.get("audio"), dict) else None
    if isinstance(audio, dict) and audio.get("waveform") is not None:
        audio = {
            "waveform": audio.get("waveform").contiguous(),
            "sample_rate": int(audio.get("sample_rate") or 44100),
        }
    else:
        audio = None
    fps = float(data.get("fps") or item.get("fps") or 24.0)
    frame_rate = Fraction(round(max(1.0, fps) * 1000), 1000)
    return Types.VideoComponents(images=images, audio=audio, frame_rate=frame_rate)


def _normalize_audio_channels(waveform: torch.Tensor, channels: int) -> torch.Tensor:
    if waveform.ndim != 3:
        raise ValueError("IAMCCS Video Hard Concat: AUDIO waveform must be [batch, channels, samples].")
    if waveform.shape[1] == channels:
        return waveform
    if waveform.shape[1] == 1 and channels == 2:
        return waveform.repeat(1, 2, 1)
    if waveform.shape[1] < channels:
        pad = torch.zeros(
            waveform.shape[0],
            channels - waveform.shape[1],
            waveform.shape[2],
            dtype=waveform.dtype,
            device=waveform.device,
        )
        return torch.cat((waveform, pad), dim=1)
    return waveform[:, :channels, :]


def _concat_audio(audio_items: List[Tuple[Any, int, float]]) -> Dict[str, Any] | None:
    usable = [audio for audio, _, _ in audio_items if isinstance(audio, dict) and audio.get("waveform") is not None]
    if not usable:
        return None
    target_rate = int(usable[0].get("sample_rate") or 44100)
    max_channels = max(int(audio["waveform"].shape[1]) for audio in usable)
    pieces = []
    for audio, frame_count, fps in audio_items:
        expected_samples = max(1, int(math.ceil((max(0, frame_count) / max(1.0, fps)) * target_rate)))
        if isinstance(audio, dict) and audio.get("waveform") is not None:
            waveform = audio["waveform"]
            sample_rate = int(audio.get("sample_rate") or target_rate)
            if sample_rate != target_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, target_rate)
            waveform = _normalize_audio_channels(waveform, max_channels)
            if waveform.shape[-1] < expected_samples:
                pad = torch.zeros(
                    waveform.shape[0],
                    waveform.shape[1],
                    expected_samples - waveform.shape[-1],
                    dtype=waveform.dtype,
                    device=waveform.device,
                )
                waveform = torch.cat((waveform, pad), dim=2)
            else:
                waveform = waveform[..., :expected_samples]
        else:
            waveform = torch.zeros(1, max_channels, expected_samples)
        pieces.append(waveform)
    return {"waveform": torch.cat(pieces, dim=2), "sample_rate": target_rate}



def _parse_take_order(value: Any, max_count: int) -> List[int]:
    text = str(value or "").strip()
    if not text:
        return list(range(1, max_count + 1))
    out: List[int] = []
    for part in text.replace(";", ",").split(","):
        number = _safe_int(part.strip(), 0)
        if 1 <= number <= max_count and number not in out:
            out.append(number)
    return out or list(range(1, max_count + 1))


def _clip_edit_for_take(edits: Any, take_index: int) -> Dict[str, Any]:
    if not isinstance(edits, dict):
        return {}
    clips = edits.get("clips") if isinstance(edits.get("clips"), dict) else edits
    for key in (str(take_index), f"T{take_index:02d}", f"video_take_{take_index:02d}"):
        item = clips.get(key) if isinstance(clips, dict) else None
        if isinstance(item, dict):
            return item
    return {}


def _trim_component(comp: Any, fps: float, trim_in_seconds: float, trim_out_seconds: float):
    frame_count = int(comp.images.shape[0])
    start = max(0, min(frame_count - 1, int(round(max(0.0, trim_in_seconds) * max(1.0, fps)))))
    if trim_out_seconds > 0:
        end = max(start + 1, min(frame_count, int(round(trim_out_seconds * max(1.0, fps)))))
    else:
        end = frame_count
    images = comp.images[start:end]
    audio = comp.audio
    if isinstance(audio, dict) and audio.get("waveform") is not None:
        sample_rate = int(audio.get("sample_rate") or 44100)
        sample_start = max(0, int(round((start / max(1.0, fps)) * sample_rate)))
        sample_end = max(sample_start + 1, int(round((end / max(1.0, fps)) * sample_rate)))
        waveform = audio["waveform"][..., sample_start:sample_end]
        audio = {"waveform": waveform, "sample_rate": sample_rate}
    return images, audio, start, end


def _audio_waveform(audio: Any):
    if not isinstance(audio, dict) or audio.get("waveform") is None:
        return None, 44100
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate") or 44100)
    if waveform is None:
        return None, sample_rate
    if waveform.dim() == 1:
        waveform = waveform.reshape(1, 1, -1)
    elif waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    return waveform, sample_rate


def _audio_duration_seconds(audio: Any) -> float:
    waveform, sample_rate = _audio_waveform(audio)
    if waveform is None:
        return 0.0
    return float(waveform.shape[-1]) / max(1.0, float(sample_rate))


def _mix_editor_audio_tracks(audio_tracks: List[Any]) -> Any:
    prepared = []
    target_rate = 0
    target_channels = 1
    target_device = None
    target_dtype = None
    for audio in audio_tracks:
        waveform, sample_rate = _audio_waveform(audio)
        if waveform is None:
            continue
        if not target_rate:
            target_rate = sample_rate
            target_device = waveform.device
            target_dtype = waveform.dtype
        if sample_rate != target_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_rate)
        if target_device is not None and waveform.device != target_device:
            waveform = waveform.to(target_device)
        if target_dtype is not None and waveform.dtype != target_dtype:
            waveform = waveform.to(target_dtype)
        target_channels = max(target_channels, int(waveform.shape[-2]))
        prepared.append(waveform)
    if not prepared:
        return None
    max_samples = max(int(w.shape[-1]) for w in prepared)
    padded = []
    for waveform in prepared:
        if int(waveform.shape[-2]) < target_channels:
            waveform = waveform.repeat_interleave(target_channels, dim=-2)[..., :target_channels, :]
        if int(waveform.shape[-1]) < max_samples:
            waveform = torch.nn.functional.pad(waveform, (0, max_samples - int(waveform.shape[-1])))
        padded.append(waveform)
    mixed = torch.stack(padded, dim=0).sum(dim=0).clamp(-1.0, 1.0)
    return {"waveform": mixed, "sample_rate": int(target_rate or 44100)}


def _mix_manual_audio_into_timeline(
    base_audio: Any,
    manual_items: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    frame_count: int,
    fps: float,
) -> Any:
    """Place manual AudioBoard editor clips at their manifest timeline positions."""
    loaded: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for clip, asset in manual_items:
        if not isinstance(clip, dict) or not isinstance(asset, dict):
            continue
        audio = _load_manual_audio_clip(asset)
        if isinstance(audio, dict) and audio.get("waveform") is not None:
            loaded.append((clip, audio))
    base_waveform, base_rate = _audio_waveform(base_audio)
    if base_waveform is None and not loaded:
        return None
    target_rate = int(base_rate or 0)
    if target_rate <= 0 and loaded:
        target_rate = int(loaded[0][1].get("sample_rate") or 44100)
    target_rate = max(1, target_rate or 44100)
    target_channels = int(base_waveform.shape[-2]) if base_waveform is not None else 1
    for _, audio in loaded:
        waveform = audio.get("waveform")
        if torch.is_tensor(waveform):
            target_channels = max(target_channels, int(waveform.shape[-2]))
    target_samples = max(1, int(round(max(1, int(frame_count)) / max(1.0, float(fps)) * target_rate)))
    mixed = torch.zeros((1, target_channels, target_samples), dtype=torch.float32)

    def add_audio(waveform: torch.Tensor, sample_rate: int, destination: int, volume: float = 1.0) -> None:
        nonlocal mixed
        if sample_rate != target_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_rate)
        waveform = _normalize_audio_channels(waveform, target_channels).to(dtype=torch.float32, device=mixed.device)
        destination = max(0, int(destination))
        if destination >= target_samples:
            return
        count = min(target_samples - destination, int(waveform.shape[-1]))
        if count <= 0:
            return
        mixed[..., destination:destination + count] += waveform[..., :count] * float(volume)

    if base_waveform is not None:
        add_audio(base_waveform, int(base_rate or target_rate), 0)
    for clip, audio in loaded:
        waveform = audio.get("waveform")
        if not torch.is_tensor(waveform):
            continue
        sample_rate = int(audio.get("sample_rate") or target_rate)
        if sample_rate != target_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_rate)
            sample_rate = target_rate
        source_start = max(0, int(round(max(0.0, _safe_float(clip.get("trimStart"), 0.0)) * sample_rate)))
        source_end = int(waveform.shape[-1])
        clip_duration = max(0.001, _safe_float(clip.get("duration"), 0.0))
        requested = max(1, int(round(clip_duration * sample_rate)))
        source_end = min(source_end, source_start + requested)
        if source_end <= source_start:
            continue
        add_audio(
            waveform[..., source_start:source_end],
            sample_rate,
            int(round(max(0.0, _safe_float(clip.get("startTime"), 0.0)) * target_rate)),
            _safe_float(clip.get("volume", 1.0), 1.0),
        )
    return {"waveform": mixed.clamp(-1.0, 1.0), "sample_rate": target_rate}


def _active_take_from_linx(cine_linx: Any, fallback: int = 1) -> int:
    linx = cine_linx if isinstance(cine_linx, dict) else {}
    resources = linx.get("resources", {}) if isinstance(linx.get("resources", {}), dict) else {}
    payload = linx.get("payload", {}) if isinstance(linx.get("payload", {}), dict) else {}
    identity = _active_identity_from_linx(cine_linx)
    if identity:
        value = _safe_int(identity.get("take_index"), 0)
        if value > 0:
            return value
    take_package = _take_package_from_linx(cine_linx)
    if take_package:
        value = _safe_int(take_package.get("take_index"), 0)
        if value > 0:
            return value

    def take_from_segments(items: Any) -> int:
        source = items if isinstance(items, list) else []
        candidates: List[int] = []
        for seg in source:
            if not isinstance(seg, dict):
                continue
            value = _safe_int(seg.get("multiTakeIndex"), 0)
            if value <= 0:
                raw = str(seg.get("timelineId") or seg.get("timeline_id") or "")
                digits = "".join(ch for ch in raw if ch.isdigit())
                value = _safe_int(digits, 0)
            if value > 0:
                candidates.append(value)
        if not candidates:
            return 0
        counts: Dict[int, int] = {}
        for value in candidates:
            counts[value] = counts.get(value, 0) + 1
        return sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)[0][0]

    def take_from_audio_payload() -> int:
        pools: List[Any] = []
        tracks = resources.get("cine_audio_tracks") if isinstance(resources.get("cine_audio_tracks"), dict) else {}
        for key in ("shotboard_segments", "segments", "all_segments"):
            pools.append(tracks.get(key))
        timeline_json = _safe_json_loads(resources.get("cine_audio_timeline_json", ""), {})
        if isinstance(timeline_json, dict):
            pools.append(timeline_json.get("audioSegments"))
            pools.append(timeline_json.get("segments"))
        resource_payload = resources.get("cine_payload") if isinstance(resources.get("cine_payload"), dict) else {}
        for owner in (payload, resource_payload):
            pools.append(owner.get("audioSegments") if isinstance(owner, dict) else None)
            audio_data = owner.get("audio_data") if isinstance(owner, dict) else None
            parsed_audio = _safe_json_loads(audio_data, {}) if isinstance(audio_data, str) else audio_data
            if isinstance(parsed_audio, dict):
                pools.append(parsed_audio.get("audioSegments"))
                pools.append(parsed_audio.get("segments"))
        for items in pools:
            value = take_from_segments(items)
            if value > 0:
                return value
        return 0

    active_take = resources.get("cine_multigeneration_active_take")
    if isinstance(active_take, dict):
        value = active_take.get("take_index") or active_take.get("source_take_index")
        if _safe_int(value, 0) > 0:
            return _safe_int(value, fallback)
    active_take_json = _safe_json_loads(resources.get("cine_multigeneration_active_take_json"), {})
    if isinstance(active_take_json, dict):
        value = active_take_json.get("take_index") or active_take_json.get("source_take_index")
        if _safe_int(value, 0) > 0:
            return _safe_int(value, fallback)
    payload_take = payload.get("multi_generation_active_take")
    if isinstance(payload_take, dict):
        value = payload_take.get("take_index") or payload_take.get("source_take_index")
        if _safe_int(value, 0) > 0:
            return _safe_int(value, fallback)
    index = resources.get("cine_multigeneration_index")
    if isinstance(index, dict) and _safe_int(index.get("active_take"), 0) > 0:
        return _safe_int(index.get("active_take"), fallback)
    index_json = _safe_json_loads(resources.get("cine_multigeneration_index_json"), {})
    if isinstance(index_json, dict) and _safe_int(index_json.get("active_take"), 0) > 0:
        return _safe_int(index_json.get("active_take"), fallback)
    resource_payload = resources.get("cine_payload") if isinstance(resources.get("cine_payload"), dict) else {}
    for key in ("timeline_id", "active_timeline_id"):
        raw = str(payload.get(key) or resource_payload.get(key) or resources.get(key) or "")
        digits = "".join(ch for ch in raw if ch.isdigit())
        if digits:
            return max(1, _safe_int(digits, fallback))
    audio_take = take_from_audio_payload()
    if audio_take > 0:
        return audio_take
    for item in reversed(linx.get("chain", []) if isinstance(linx.get("chain", []), list) else []):
        if not isinstance(item, dict):
            continue
        value = item.get("take_index") or item.get("active_take") or item.get("active_take_collected")
        if _safe_int(value, 0) > 0:
            return _safe_int(value, fallback)
    return max(1, _safe_int(fallback, 1))


def _take_package_from_linx(cine_linx: Any) -> Dict[str, Any]:
    linx = cine_linx if isinstance(cine_linx, dict) else {}
    resources = linx.get("resources", {}) if isinstance(linx.get("resources", {}), dict) else {}
    payload = linx.get("payload", {}) if isinstance(linx.get("payload", {}), dict) else {}
    candidates = [
        resources.get("cine_take_router_package"),
        resources.get("cine_multigeneration_take_package"),
        resources.get("cine_take_package"),
        payload.get("take_package"),
        _safe_json_loads(resources.get("cine_take_router_package_json"), {}),
        _safe_json_loads(resources.get("cine_multigeneration_take_package_json"), {}),
        _safe_json_loads(resources.get("cine_take_package_json"), {}),
        _safe_json_loads(payload.get("take_package_json"), {}),
    ]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        take_index = _safe_int(candidate.get("take_index"), 0)
        timeline_id = str(candidate.get("timeline_id") or candidate.get("take_id") or "").strip()
        if take_index > 0 and timeline_id:
            normalized_timeline = _take_timeline_id(take_index)
            if timeline_id != normalized_timeline:
                raise ValueError(
                    "IAMCCS TakePackage hard-fail: "
                    f"take_index={take_index} but timeline_id={timeline_id!r}; expected {normalized_timeline!r}."
                )
            audio_lane = str(candidate.get("audio_lane") or _take_audio_lane_name(take_index))
            expected_lane = _take_audio_lane_name(take_index)
            if audio_lane != expected_lane:
                raise ValueError(
                    "IAMCCS TakePackage hard-fail: "
                    f"take_index={take_index} but audio_lane={audio_lane!r}; expected {expected_lane!r}."
                )
            return candidate
    return {}


def _has_multigeneration_contract(cine_linx: Any) -> bool:
    linx = cine_linx if isinstance(cine_linx, dict) else {}
    resources = linx.get("resources", {}) if isinstance(linx.get("resources", {}), dict) else {}
    payload = linx.get("payload", {}) if isinstance(linx.get("payload", {}), dict) else {}
    return bool(
        resources.get("cine_multigeneration_index")
        or resources.get("cine_multigeneration_index_json")
        or payload.get("multi_generation")
        or str(linx.get("mode", "")) == "iamccs_multigeneration"
    )


def _timeline_json_from_linx(cine_linx: Any) -> str:
    linx = cine_linx if isinstance(cine_linx, dict) else {}
    resources = linx.get("resources", {}) if isinstance(linx.get("resources", {}), dict) else {}
    payload = linx.get("payload", {}) if isinstance(linx.get("payload", {}), dict) else {}
    outputs = linx.get("outputs", {}) if isinstance(linx.get("outputs", {}), dict) else {}
    for value in (
        resources.get("cine_take_router_timeline_data"),
        resources.get("cine_board_timeline_data"),
        resources.get("cine_dialogue_shotboard_timeline_json"),
        payload.get("timeline_data"),
        outputs.get("timeline_data"),
    ):
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict):
            return _json_dump(value)
    return ""


def _active_visual_from_timeline_data(timeline_data: Any, take_package: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    data = _safe_json_loads(timeline_data, {})
    if not isinstance(data, dict):
        data = {}
    take_index = max(1, _safe_int(take_package.get("take_index"), 1))
    timeline_id = str(take_package.get("timeline_id") or _take_timeline_id(take_index))
    multi = data.get("multiGeneration") if isinstance(data.get("multiGeneration"), dict) else {}
    visual_timelines = multi.get("visualTimelines") if isinstance(multi.get("visualTimelines"), dict) else {}
    active_visual = visual_timelines.get(timeline_id) if isinstance(visual_timelines.get(timeline_id), dict) else {}
    return data, active_visual


def _build_routed_timeline_data(
    base_timeline_data: Any,
    take_package: Dict[str, Any],
    duration_policy: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    base, active_visual = _active_visual_from_timeline_data(base_timeline_data, take_package)
    take_index = max(1, _safe_int(take_package.get("take_index"), 1))
    timeline_id = str(take_package.get("timeline_id") or _take_timeline_id(take_index))
    if not active_visual:
        raise ValueError(
            "IAMCCS TakeRouter hard-fail: missing visual timeline for "
            f"{timeline_id}. Each generation must own its timeline_data."
        )
    visual_segments = active_visual.get("segments") if isinstance(active_visual.get("segments"), list) else []
    visual_rows = active_visual.get("rows") if isinstance(active_visual.get("rows"), list) else []
    if not visual_segments and not visual_rows:
        raise ValueError(
            "IAMCCS TakeRouter hard-fail: visual timeline "
            f"{timeline_id} has no segments/rows. Refusing to reuse another timeline."
        )
    fps = _safe_float(active_visual.get("frame_rate", base.get("frame_rate", take_package.get("frame_rate", 24.0))), 24.0)
    visual_duration = _safe_float(
        active_visual.get("duration_seconds", active_visual.get("durationSeconds", active_visual.get("duration", 0))),
        0.0,
    )
    package_duration = _safe_float(take_package.get("duration_seconds"), 0.0)
    package_frames = max(0, _safe_int(take_package.get("duration_frames"), 0))
    roll_contract = take_package.get("roll_contract") if isinstance(take_package.get("roll_contract"), dict) else {}
    roll_enabled = _safe_bool(roll_contract.get("enabled"), False) and _safe_int(roll_contract.get("frames"), 0) > 0
    if visual_duration <= 0:
        visual_duration = package_duration if package_duration > 0 else _safe_float(base.get("duration_seconds"), 0.0)
    visual_frames = max(1, int(round(visual_duration * fps)))
    # A roll is a real generation-duration extension, not only an editor handle.
    # Older Shotboard timeline data can still contain the nominal visual length;
    # promote that visual contract to the immutable TakePackage duration before
    # the backend samples, otherwise audio may be 17s while video remains 15s.
    if roll_enabled and package_frames > visual_frames:
        active_visual = _extend_visual_timeline_for_roll(active_visual, package_frames, fps)
        visual_segments = active_visual.get("segments") if isinstance(active_visual.get("segments"), list) else []
        visual_rows = active_visual.get("rows") if isinstance(active_visual.get("rows"), list) else []
        visual_duration = package_duration if package_duration > 0 else float(package_frames) / max(1.0, fps)
        visual_frames = int(package_frames)
    mismatch_frames = abs(visual_frames - package_frames) if package_frames > 0 else 0
    if str(duration_policy) == "hard_fail_mismatch" and package_frames > 0 and mismatch_frames > 1:
        raise ValueError(
            "IAMCCS TakeRouter hard-fail: visual/audio duration mismatch for "
            f"{timeline_id}. visual={visual_frames}f package={package_frames}f. "
            "Fix Shotboard timeline duration or AudioBoard chunk duration before queue."
        )
    if str(duration_policy) == "audio_package_truth" and package_duration > 0:
        visual_duration = package_duration
        visual_frames = package_frames
    global_prompt = str(active_visual.get("global_prompt", active_visual.get("prompt", base.get("global_prompt", ""))) or "")
    routed = copy.deepcopy(base)
    routed["schema"] = "iamccs.cine.filmmaker_timeline"
    routed["schema_version"] = max(2, _safe_int(routed.get("schema_version"), 2))
    routed["global_prompt"] = global_prompt
    routed["prompt"] = global_prompt
    routed["duration_seconds"] = float(visual_duration)
    routed["frame_rate"] = float(fps)
    routed["segments"] = copy.deepcopy(visual_segments)
    routed["rows"] = copy.deepcopy(visual_rows)
    for key in (
        "shotClips",
        "shots",
        "motionSegments",
        "motionClips",
        "motionParts",
        "videoSegments",
        "videoClips",
        "cameraSegments",
        "cameraClips",
        "referenceSheets",
    ):
        if isinstance(active_visual.get(key), list):
            routed[key] = copy.deepcopy(active_visual.get(key))
    routed["audioSegments"] = copy.deepcopy(take_package.get("audio_segments", []))
    if isinstance(active_visual.get("audioClips"), list) and not routed["audioSegments"]:
        routed["audioSegments"] = copy.deepcopy(active_visual.get("audioClips"))
    routed["audioClips"] = copy.deepcopy(routed["audioSegments"])
    routed["audioTrackCount"] = max(1, _safe_int(take_package.get("audio_track_count"), 1))
    routed["audioSyncMode"] = "timeline_audio"
    routed["use_custom_audio"] = bool(routed["audioSegments"])
    routed["shotboard_version"] = max(_safe_int(base.get("shotboard_version"), 3), _safe_int(active_visual.get("shotboard_version"), 3))
    routed["adapter_contract"] = str(active_visual.get("adapter_contract") or base.get("adapter_contract") or "take_router_visual_timeline")
    routed["active_take_package"] = copy.deepcopy(take_package)
    routed["multiGeneration"] = {
        **(copy.deepcopy(routed.get("multiGeneration")) if isinstance(routed.get("multiGeneration"), dict) else {}),
        "enabled": True,
        "activeTake": int(take_index),
        "activeTimelineId": timeline_id,
        "takePackageActive": copy.deepcopy(take_package),
        "timelineDataTruth": f"take_router:{timeline_id}",
    }
    report = {
        "timeline_id": timeline_id,
        "take_index": int(take_index),
        "audio_lane": str(take_package.get("audio_lane") or _take_audio_lane_name(take_index)),
        "visual_segments": len(visual_segments),
        "visual_rows": len(visual_rows),
        "duration_seconds": float(visual_duration),
        "duration_frames": int(visual_frames),
        "package_duration_frames": int(package_frames),
        "duration_mismatch_frames": int(mismatch_frames),
        "global_prompt_chars": len(global_prompt),
        "motion_segments": len(active_visual.get("motionSegments", [])) if isinstance(active_visual.get("motionSegments"), list) else 0,
        "video_segments": len(active_visual.get("videoSegments", [])) if isinstance(active_visual.get("videoSegments"), list) else 0,
        "shotboard_version": int(routed.get("shotboard_version", 3) or 3),
    }
    return routed, report


class IAMCCS_TakePackage:
    """Expose and validate the immutable T/A package before routing timeline_data."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "strict_mode": (["hard_fail"], {"default": "hard_fail"}),
            },
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE, "STRING", "STRING")
    RETURN_NAMES = ("cine_linx", "take_package_json", "report")
    FUNCTION = "package"
    CATEGORY = "IAMCCS/Cine/Multigeneration"

    def package(self, cine_linx, strict_mode):
        out_linx = _clone_linx(cine_linx, "iamccs_take_package")
        resources = _resources(out_linx)
        outputs = _outputs(out_linx)
        payload = _payload(out_linx)
        take_package = _take_package_from_linx(out_linx)
        built = False
        if not take_package:
            raise ValueError(
                "IAMCCS TakePackage hard-fail: no valid TakePackage found. "
                "Bridge/TakePicker must provide immutable T/A identity before generation."
            )
        take_index = max(1, _safe_int(take_package.get("take_index"), 1))
        expected_timeline = _take_timeline_id(take_index)
        expected_lane = _take_audio_lane_name(take_index)
        timeline_id = str(take_package.get("timeline_id") or take_package.get("take_id") or "")
        audio_lane = str(take_package.get("audio_lane") or "")
        if timeline_id != expected_timeline or audio_lane != expected_lane:
            raise ValueError(
                "IAMCCS TakePackage hard-fail: invalid package identity. "
                f"take={take_index} timeline={timeline_id!r} audio={audio_lane!r}; "
                f"expected timeline={expected_timeline!r} audio={expected_lane!r}."
            )
        package_json = _json_dump(take_package)
        resources["cine_multigeneration_take_package"] = copy.deepcopy(take_package)
        resources["cine_multigeneration_take_package_json"] = package_json
        resources["cine_take_package"] = copy.deepcopy(take_package)
        resources["cine_take_package_json"] = package_json
        payload["take_package"] = copy.deepcopy(take_package)
        outputs["take_package_json"] = package_json
        report = {
            "node": "IAMCCS_TakePackage",
            "status": "valid",
            "built_from_active_take": bool(built),
            "take_index": int(take_index),
            "timeline_id": timeline_id,
            "audio_lane": audio_lane,
            "duration_frames": _safe_int(take_package.get("duration_frames"), 0),
            "duration_seconds": _safe_float(take_package.get("duration_seconds"), 0.0),
            "visual_segments": len(take_package.get("visual_segments") if isinstance(take_package.get("visual_segments"), list) else []),
            "audio_segments": len(take_package.get("audio_segments") if isinstance(take_package.get("audio_segments"), list) else []),
            "global_prompt_chars": len(str(take_package.get("global_prompt", "") or "")),
            "truth": "This package is the immutable identity for one rendered generation. Downstream nodes must not infer or remap T/A slots.",
        }
        resources["cine_take_package_report"] = report
        outputs["report"] = _json_dump(report)
        out_linx.setdefault("chain", []).append({
            "role": "take_package",
            "name": "IAMCCS_TakePackage",
            "take_index": int(take_index),
            "timeline_id": timeline_id,
            "audio_lane": audio_lane,
        })
        _refresh_linx_index(out_linx)
        return out_linx, package_json, _json_dump(report)


class IAMCCS_TakeRouter:
    """Materialize a single T/A TakePackage into routed Shotboard timeline_data before generation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "duration_policy": (["visual_timeline_truth", "hard_fail_mismatch", "audio_package_truth"], {"default": "visual_timeline_truth"}),
                "strict_take_package": (["required"], {"default": "required"}),
            },
            "optional": {
                "timeline_data": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("cine_linx", "take_package_json", "timeline_data", "report")
    FUNCTION = "route"
    CATEGORY = "IAMCCS/Cine/Multigeneration"

    def route(self, cine_linx, duration_policy, strict_take_package, timeline_data=""):
        out_linx = _clone_linx(cine_linx, "iamccs_take_router")
        resources = _resources(out_linx)
        outputs = _outputs(out_linx)
        payload = _payload(out_linx)
        take_package = _take_package_from_linx(out_linx)
        if not take_package:
            raise ValueError(
                "IAMCCS TakeRouter hard-fail: missing TakePackage. "
                "Place IAMCCS_TakeRouter after MultiTimelineBridge/TakePicker and before the backend."
            )
        index = resources.get("cine_multigeneration_index") if isinstance(resources.get("cine_multigeneration_index"), dict) else {}
        active = resources.get("cine_multigeneration_active_take") if isinstance(resources.get("cine_multigeneration_active_take"), dict) else {}
        if not isinstance(index.get("takes"), list) or not index.get("takes"):
            raise ValueError(
                "IAMCCS TakeRouter hard-fail: missing MultiTimelineBridge generation index. "
                "LTX-Desktop-style routing requires a concrete indexed take list before generation."
            )
        if not active:
            raise ValueError(
                "IAMCCS TakeRouter hard-fail: missing active take identity. "
                "Select/prepare one T/A take before queue."
            )
        base_timeline = str(timeline_data or "").strip() or _timeline_json_from_linx(out_linx)
        if not base_timeline:
            raise ValueError(
                "IAMCCS TakeRouter hard-fail: no Shotboard timeline_data available. "
                "The backend will not generate from a reconstructed or stale take package."
            )
        visual_identity = _take_identity_from_payload(base_timeline)
        if not visual_identity:
            raise ValueError(
                "IAMCCS TakeRouter hard-fail: Shotboard timeline_data has no active T/A identity. "
                "The selected timeline must declare activeTake and activeTimelineId."
            )
        visual_take = _safe_int(visual_identity.get("take_index"), 0) if visual_identity else 0
        package_take_initial = _safe_int(take_package.get("take_index"), 0)
        package_timeline_initial = str(take_package.get("timeline_id") or "").strip()
        visual_timeline = str(visual_identity.get("timeline_id") or "").strip()
        if visual_take <= 0 or package_take_initial <= 0 or not visual_timeline or not package_timeline_initial:
            raise ValueError(
                "IAMCCS TakeRouter hard-fail: incomplete T/A identity. "
                f"shotboard_take={visual_take}, shotboard_timeline={visual_timeline!r}, "
                f"package_take={package_take_initial}, package_timeline={package_timeline_initial!r}."
            )
        if visual_take != package_take_initial or visual_timeline != package_timeline_initial:
            raise ValueError(
                "IAMCCS TakeRouter hard-fail: Shotboard active timeline does not match TakePackage. "
                f"shotboard=T{visual_take:02d}/{visual_timeline}, "
                f"package=T{package_take_initial:02d}/{package_timeline_initial}. "
                "Prepare the selected take again; no automatic remap is allowed."
            )
        expected_take = _safe_int(index.get("active_take") or active.get("take_index"), 0)
        expected_timeline = str(index.get("active_timeline_id") or active.get("timeline_id") or "").strip()
        package_take = _safe_int(take_package.get("take_index"), 0)
        package_timeline = str(take_package.get("timeline_id") or "").strip()
        if expected_take <= 0 or not expected_timeline:
            raise ValueError(
                "IAMCCS TakeRouter hard-fail: generation index does not expose a selected active take. "
                "MultiTimelineBridge must publish active_take and active_timeline_id."
            )
        if expected_take and package_take and expected_take != package_take:
            raise ValueError(
                "IAMCCS TakeRouter hard-fail: active take mismatch before backend. "
                f"bridge_take=T{expected_take:02d}, package_take=T{package_take:02d}. "
                "Refresh/prepare the selected take before queue."
            )
        if expected_timeline and package_timeline and expected_timeline != package_timeline:
            raise ValueError(
                "IAMCCS TakeRouter hard-fail: active timeline mismatch before backend. "
                f"bridge_timeline={expected_timeline}, package_timeline={package_timeline}. "
                "Refresh/prepare the selected take before queue."
            )
        print(
            "[IAMCCS TakeRouter] ROUTE_COMMIT "
            f"take={package_take or expected_take} "
            f"timeline={package_timeline or expected_timeline} "
            f"audio_lane={take_package.get('audio_lane') or active.get('audio_lane')}"
        )
        routed_timeline, route_report = _build_routed_timeline_data(base_timeline, take_package, str(duration_policy))
        routed_json = json.dumps(routed_timeline, ensure_ascii=False)
        take_json = _json_dump(take_package)
        resources["cine_take_router_timeline_data"] = routed_json
        resources["cine_board_timeline_data"] = routed_json
        resources["cine_take_router_package"] = copy.deepcopy(take_package)
        resources["cine_take_router_package_json"] = take_json
        resources["cine_global_prompt"] = str(routed_timeline.get("global_prompt", ""))
        resources["cine_duration_seconds"] = float(route_report["duration_seconds"])
        resources["cine_frame_rate"] = float(routed_timeline.get("frame_rate", 24.0))
        # Route only the user-visible timeline duration. LTX latent rounding is
        # an implementation detail for downstream nodes; it must not alter the
        # Shotboard timeline, local prompt lengths, or guide-frame map.
        resources["cine_max_frames"] = int(route_report["duration_frames"])
        resources.pop("cine_ltxv_length", None)
        payload["timeline_data"] = routed_json
        payload["global_prompt"] = str(routed_timeline.get("global_prompt", ""))
        payload["duration_seconds"] = float(route_report["duration_seconds"])
        payload["frame_rate"] = float(routed_timeline.get("frame_rate", 24.0))
        payload["duration_frames"] = int(route_report["duration_frames"])
        payload["max_frames"] = int(route_report["duration_frames"])
        payload.pop("ltxv_length", None)
        payload["take_package"] = copy.deepcopy(take_package)
        payload["timeline_id"] = str(take_package.get("timeline_id") or take_package.get("take_id") or "")
        outputs["timeline_data"] = routed_json
        outputs["take_package_json"] = take_json
        outputs["max_frames"] = int(route_report["duration_frames"])
        report = {
            "node": "IAMCCS_TakeRouter",
            "status": "routed",
            **route_report,
            "duration_policy": str(duration_policy),
            "truth": "Backend must consume this routed timeline_data for exactly one T/A generation. No fallback to previous timeline is allowed.",
        }
        resources["cine_take_router_report"] = report
        outputs["report"] = _json_dump(report)
        out_linx.setdefault("chain", []).append({
            "role": "take_router",
            "name": "IAMCCS_TakeRouter",
            "timeline_id": report["timeline_id"],
            "take_index": report["take_index"],
            "audio_lane": report["audio_lane"],
        })
        _refresh_linx_index(out_linx)
        return out_linx, take_json, routed_json, _json_dump(report)


def _video_manifest_entry(slot: int, video: Any) -> Dict[str, Any]:
    comp = _video_components(video)
    fps = float(comp.frame_rate or 24.0)
    frames = int(comp.images.shape[0])
    return {
        "slot": int(slot),
        "timeline_id": _take_timeline_id(slot),
        "audio_lane": _take_audio_lane_name(slot),
        "frames": frames,
        "fps": fps,
        "duration_seconds": frames / max(1.0, fps),
        "height": int(comp.images.shape[1]),
        "width": int(comp.images.shape[2]),
        "has_embedded_audio": comp.audio is not None,
        "collected_runtime": True,
    }


def _audio_preview_peaks(waveform: torch.Tensor, bins: int = 1200) -> List[Dict[str, float]]:
    if waveform is None or not torch.is_tensor(waveform) or waveform.numel() <= 0:
        return []
    if waveform.dim() == 1:
        mono = waveform.float().abs()
    elif waveform.dim() == 2:
        mono = waveform.float().mean(dim=0).abs()
    else:
        mono = waveform.float()[0].mean(dim=0).abs()
    total = int(mono.shape[-1])
    if total <= 0:
        return []
    count = max(64, min(int(bins), total))
    step = max(1, int(math.ceil(total / count)))
    peaks: List[Dict[str, float]] = []
    for start in range(0, total, step):
        chunk = mono[start:min(total, start + step)]
        if chunk.numel() <= 0:
            continue
        value = float(torch.clamp(chunk.max(), 0.0, 1.0).item())
        peaks.append({"min": -value, "max": value, "rms": float(torch.sqrt(torch.mean(chunk * chunk)).item())})
    return peaks


def _persist_audio_preview(session_key: Any, slot: int, audio: Any, root: str) -> Dict[str, Any]:
    waveform, sample_rate = _audio_waveform(audio)
    if waveform is None or not torch.is_tensor(waveform) or waveform.numel() <= 0:
        return {}
    try:
        output_root = _output_dir()
        rel_dir = os.path.relpath(root, output_root).replace("\\", "/")
        stamp = int(time.time() * 1000)
        filename = f"T{int(slot):02d}_A{int(slot):02d}_audio_{stamp}.wav"
        path = os.path.join(root, filename)
        wav = waveform[0].detach().float().cpu().contiguous()
        wav = torch.clamp(wav, -1.0, 1.0)
        torchaudio.save(path, wav, int(sample_rate))
        return {
            "audio_preview_file": filename,
            "audio_preview_subfolder": rel_dir,
            "audio_preview_type": "output",
        }
    except Exception as exc:
        return {"audio_preview_error": str(exc)}


def _audio_manifest_entry(slot: int, audio: Any, session_key: Any = None, root: str | None = None) -> Dict[str, Any] | None:
    waveform, sample_rate = _audio_waveform(audio)
    if waveform is None:
        return None
    manifest = {
        "slot": int(slot),
        "audio_lane": _take_audio_lane_name(slot),
        "sample_rate": int(sample_rate),
        "samples": int(waveform.shape[-1]),
        "channels": int(waveform.shape[-2]),
        "duration_seconds": int(waveform.shape[-1]) / max(1.0, float(sample_rate)),
        "waveform_peaks": _audio_preview_peaks(waveform, 1400),
        "collected_runtime": True,
    }
    if root is not None:
        manifest.update(_persist_audio_preview(session_key, slot, audio, root))
    return manifest


def _editor_manifest_default(session_key: Any, fps: float = 24.0) -> Dict[str, Any]:
    tracks = []
    for index in range(1, 6):
        tracks.append({"id": f"V{index}", "name": f"V{index}", "kind": "video", "muted": False, "locked": False})
    for index in range(1, 6):
        tracks.append({"id": f"A{index}", "name": f"A{index}", "kind": "audio", "muted": False, "locked": False})
    tracks.append({"id": "AM", "name": "MASTER AUDIO", "kind": "master_audio", "muted": False, "locked": False})
    return {
        "schema": "iamccs.shotboard_video_editor.v1",
        "schema_version": 1,
        "session_key": _safe_slug(session_key),
        "created_at": time.time(),
        "updated_at": time.time(),
        "fps": float(fps or 24.0),
        "assets": {},
        "clips": [],
        "tracks": tracks,
        "duration_seconds": 0.0,
        "assembly_order": [],
        "ui_state": {"playhead": 0.0, "zoom_seconds": 20.0, "selected_clip_id": ""},
    }


def _normalize_editor_manifest(value: Any, session_key: Any, fps: float = 24.0) -> Dict[str, Any]:
    data = _safe_json_loads(value, {})
    if not isinstance(data, dict) or data.get("schema") != "iamccs.shotboard_video_editor.v1":
        data = _editor_manifest_default(session_key, fps)
    data.setdefault("schema", "iamccs.shotboard_video_editor.v1")
    data.setdefault("schema_version", 1)
    data["session_key"] = _safe_slug(data.get("session_key") or session_key)
    data["fps"] = float(_safe_float(data.get("fps"), fps or 24.0))
    data.setdefault("assets", {})
    if not isinstance(data["assets"], dict):
        data["assets"] = {}
    data.setdefault("clips", [])
    if not isinstance(data["clips"], list):
        data["clips"] = []
    data.setdefault("tracks", _editor_manifest_default(session_key, fps)["tracks"])
    if not isinstance(data["tracks"], list):
        data["tracks"] = _editor_manifest_default(session_key, fps)["tracks"]
    default_tracks = _editor_manifest_default(session_key, fps)["tracks"]
    existing_ids = {
        str(track.get("id"))
        for track in data["tracks"]
        if isinstance(track, dict) and track.get("id") is not None
    }
    for track in default_tracks:
        if str(track.get("id")) not in existing_ids:
            data["tracks"].append(copy.deepcopy(track))
            existing_ids.add(str(track.get("id")))
    data.setdefault("assembly_order", [])
    if not isinstance(data["assembly_order"], list):
        data["assembly_order"] = []
    data.setdefault("ui_state", {})
    if not isinstance(data["ui_state"], dict):
        data["ui_state"] = {}
    return data


def _editor_track_for_take(take_index: int, kind: str) -> Tuple[str, int]:
    take_index = int(take_index)
    if take_index < 1 or take_index > 5:
        raise ValueError(
            "IAMCCS ShotboardVideoEditorV1 hard-fail: the editor supports T01/A1 through T05/A5. "
            f"Received take T{take_index:02d}."
        )
    if str(kind) == "video":
        return f"V{take_index}", take_index - 1
    return f"A{take_index}", 5 + take_index - 1


def _editor_manifest_for_session(value: Any, session_key: Any, fps: float = 24.0) -> Dict[str, Any]:
    session = _safe_slug(session_key)
    incoming_raw = _safe_json_loads(value, {})
    incoming_explicit = isinstance(incoming_raw, dict) and incoming_raw.get("schema") == "iamccs.shotboard_video_editor.v1"
    incoming = _normalize_editor_manifest(value, session, fps)
    stored = copy.deepcopy(_VIDEO_EDITOR_MANIFEST_REGISTRY.get(session, {}))
    if not isinstance(stored, dict) or stored.get("schema") != "iamccs.shotboard_video_editor.v1":
        return incoming
    incoming_clips = len(incoming.get("clips") if isinstance(incoming.get("clips"), list) else [])
    incoming_assets = len(incoming.get("assets") if isinstance(incoming.get("assets"), dict) else {})
    stored_clips = len(stored.get("clips") if isinstance(stored.get("clips"), list) else [])
    incoming_updated = _safe_float(incoming.get("updated_at"), 0.0)
    stored_updated = _safe_float(stored.get("updated_at"), 0.0)
    incoming_cleared = _safe_float(incoming.get("cleared_at"), 0.0)
    if incoming_explicit and incoming_clips == 0 and incoming_assets == 0 and incoming_cleared > 0:
        _VIDEO_EDITOR_MANIFEST_REGISTRY[session] = copy.deepcopy(incoming)
        return incoming
    # LTX Desktop keeps project/timeline state outside transient UI widgets.
    # If Comfy starts a second queue with an empty or older widget, keep the runtime manifest.
    if stored_clips > incoming_clips and (not incoming_explicit or incoming_updated <= stored_updated):
        stored["fps"] = float(_safe_float(stored.get("fps"), fps or 24.0))
        return _normalize_editor_manifest(stored, session, stored["fps"])
    return incoming


def _take_insert_start(manifest: Dict[str, Any], append_mode: str) -> float:
    if str(append_mode) == "timeline_origin":
        return 0.0
    clips = manifest.get("clips") if isinstance(manifest.get("clips"), list) else []
    ends = [
        _safe_float(clip.get("startTime"), 0.0) + _safe_float(clip.get("duration"), 0.0)
        for clip in clips
        if isinstance(clip, dict) and str(clip.get("type")) == "video"
    ]
    return max(ends or [0.0])


def _update_manifest_duration(manifest: Dict[str, Any]) -> None:
    clips = manifest.get("clips") if isinstance(manifest.get("clips"), list) else []
    duration = 0.0
    for clip in clips:
        if not isinstance(clip, dict):
            continue
        duration = max(duration, _safe_float(clip.get("startTime"), 0.0) + _safe_float(clip.get("duration"), 0.0))
    manifest["duration_seconds"] = float(duration)
    manifest["updated_at"] = time.time()


def _remove_take_clips(manifest: Dict[str, Any], take_index: int) -> None:
    clips = manifest.get("clips") if isinstance(manifest.get("clips"), list) else []
    manifest["clips"] = [
        clip for clip in clips
        if not (isinstance(clip, dict) and _safe_int(clip.get("takeIndex"), 0) == int(take_index))
    ]


def _append_video_take_to_manifest(
    manifest: Dict[str, Any],
    session_key: Any,
    take_index: int,
    video: Any,
    clip_audio: Any = None,
    append_mode: str = "append_sequence",
    replace_existing_take: bool = False,
    target_duration_seconds: float | None = None,
    target_duration_frames: int | None = None,
    tail_trim_frames: int | None = None,
    nominal_duration_seconds: float | None = None,
    nominal_duration_frames: int | None = None,
    pre_roll_frames: int | None = None,
    post_roll_frames: int | None = None,
    roll_contract: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    take_index = max(1, int(take_index))
    comp = _video_components(video)
    fps = float(comp.frame_rate or manifest.get("fps") or 24.0)
    source_duration = int(comp.images.shape[0]) / max(1.0, fps)
    requested_pre_roll_frames = max(0, _safe_int(pre_roll_frames, 0))
    requested_post_roll_frames = max(0, _safe_int(post_roll_frames, 0))
    requested_nominal_frames = max(0, _safe_int(nominal_duration_frames, 0))
    requested_nominal_seconds = _safe_float(nominal_duration_seconds, 0.0)
    if requested_nominal_frames <= 0 and requested_nominal_seconds > 0:
        requested_nominal_frames = max(1, int(round(requested_nominal_seconds * max(1.0, fps))))
    requested_roll_frames = requested_nominal_frames + requested_pre_roll_frames + requested_post_roll_frames
    requested_target_frames = max(0, _safe_int(target_duration_frames, 0))
    requested_target_seconds = _safe_float(target_duration_seconds, 0.0)
    parking_target_frames = max(requested_target_frames, requested_roll_frames)
    parking_target_seconds = max(
        requested_target_seconds,
        float(requested_roll_frames) / max(1.0, fps) if requested_roll_frames > 0 else 0.0,
    )
    duration = min(source_duration, parking_target_seconds) if parking_target_seconds > 0 else source_duration
    duration = max(1.0 / max(1.0, fps), float(duration))
    video_track_id, video_track_index = _editor_track_for_take(take_index, "video")
    audio_track_id, audio_track_index = _editor_track_for_take(take_index, "audio")
    if replace_existing_take:
        _remove_take_clips(manifest, take_index)
    start = _take_insert_start(manifest, append_mode)
    persisted = _persist_video_take(
        session_key,
        take_index,
        video,
        clip_audio,
        target_duration_seconds=duration,
        target_duration_frames=parking_target_frames,
        tail_trim_frames=tail_trim_frames,
    )
    parked_frames = max(1, _safe_int(persisted.get("parking_frames"), int(comp.images.shape[0])))
    parked_duration = _safe_float(persisted.get("parking_duration_seconds"), duration)
    requested_pre_roll_frames = min(requested_pre_roll_frames, max(0, parked_frames - 1))
    nominal_frames = requested_nominal_frames
    if nominal_frames <= 0:
        nominal_frames = max(1, parked_frames - requested_pre_roll_frames - requested_post_roll_frames)
    nominal_frames = min(nominal_frames, max(1, parked_frames - requested_pre_roll_frames))
    post_roll_frames_actual = min(
        requested_post_roll_frames,
        max(0, parked_frames - requested_pre_roll_frames - nominal_frames),
    )
    nominal_start_frames = requested_pre_roll_frames
    nominal_end_frames = min(parked_frames, nominal_start_frames + nominal_frames)
    nominal_frames = max(1, nominal_end_frames - nominal_start_frames)
    duration = float(nominal_frames) / max(1.0, fps)
    nominal_start_seconds = float(nominal_start_frames) / max(1.0, fps)
    nominal_end_seconds = float(nominal_end_frames) / max(1.0, fps)
    roll_contract_data = copy.deepcopy(roll_contract) if isinstance(roll_contract, dict) else {}
    roll_contract_data.update({
        "enabled": bool(nominal_start_frames or post_roll_frames_actual),
        "pre_roll_frames": int(nominal_start_frames),
        "post_roll_frames": int(post_roll_frames_actual),
        "nominal_duration_frames": int(nominal_frames),
        "generation_duration_frames": int(parked_frames),
        "nominal_source_start_frames": int(nominal_start_frames),
        "nominal_source_end_frames": int(nominal_end_frames),
    })
    asset_id = f"take_T{take_index:02d}_video"
    asset = {
        "id": asset_id,
        "type": "video",
        "takeIndex": int(take_index),
        "timelineId": _take_timeline_id(take_index),
        "audioLane": _take_audio_lane_name(take_index),
        "path": persisted.get("parking_tensor_path", ""),
        "duration": float(parked_duration),
        "source_duration": float(source_duration),
        "timeline_duration": float(parked_duration),
        "target_duration_frames": int(parking_target_frames),
        "tail_trim_frames": max(0, _safe_int(tail_trim_frames, 0)),
        "nominal_duration": float(duration),
        "nominal_duration_frames": int(nominal_frames),
        "pre_roll_frames": int(nominal_start_frames),
        "post_roll_frames": int(post_roll_frames_actual),
        "roll_contract": copy.deepcopy(roll_contract_data),
        "fps": float(fps),
        "frames": int(parked_frames),
        "width": int(comp.images.shape[2]),
        "height": int(comp.images.shape[1]),
        "takes": [{
            "path": persisted.get("parking_tensor_path", ""),
            "createdAt": time.time(),
            "width": int(comp.images.shape[2]),
            "height": int(comp.images.shape[1]),
        }],
        **persisted,
    }
    manifest.setdefault("assets", {})[asset_id] = asset
    video_clip_id = f"clip_T{take_index:02d}_V"
    manifest.setdefault("clips", []).append({
        "id": video_clip_id,
        "assetId": asset_id,
        "type": "video",
        "takeIndex": int(take_index),
        "timelineId": _take_timeline_id(take_index),
        "audioLane": _take_audio_lane_name(take_index),
        "startTime": float(start),
        "duration": float(duration),
        "sourceDuration": float(parked_duration),
        "rawSourceDuration": float(source_duration),
        "nominalDuration": float(duration),
        "nominalDurationFrames": int(nominal_frames),
        "generationDuration": float(parked_duration),
        "generationDurationFrames": int(parked_frames),
        "preRoll": float(nominal_start_seconds),
        "postRoll": float(post_roll_frames_actual / max(1.0, fps)),
        "preRollFrames": int(nominal_start_frames),
        "postRollFrames": int(post_roll_frames_actual),
        "rollContract": copy.deepcopy(roll_contract_data),
        "sourceDurationLimit": float(parked_duration),
        "trimStart": float(nominal_start_seconds),
        "trimEnd": float(nominal_end_seconds),
        "trackId": video_track_id,
        "trackIndex": video_track_index,
        "muted": False,
        "volume": 1.0,
        "linkedClipIds": [],
    })
    # Preserve the same source-of-truth rule in the editable lane manifest.
    # This keeps T02's visible waveform and nominal trim aligned to the master
    # rather than to LTX's reconstructed audio component.
    audio_source = clip_audio if clip_audio is not None else comp.audio
    audio_entry = _audio_manifest_entry(take_index, audio_source, session_key=session_key, root=_parking_root(session_key)) if audio_source is not None else None
    if audio_entry:
        audio_asset_id = f"take_T{take_index:02d}_audio"
        manifest["assets"][audio_asset_id] = {
            "id": audio_asset_id,
            "type": "audio",
            "takeIndex": int(take_index),
            "timelineId": _take_timeline_id(take_index),
            "audioLane": _take_audio_lane_name(take_index),
            "duration": float(audio_entry.get("duration_seconds") or duration),
            "timeline_duration": float(audio_entry.get("duration_seconds") or parked_duration),
            "nominal_duration": float(duration),
            "nominal_duration_frames": int(nominal_frames),
            "pre_roll_frames": int(nominal_start_frames),
            "post_roll_frames": int(post_roll_frames_actual),
            "roll_contract": copy.deepcopy(roll_contract_data),
            **audio_entry,
        }
        audio_clip_id = f"clip_T{take_index:02d}_A"
        audio_source_duration = max(
            1.0 / max(1.0, fps),
            _safe_float(audio_entry.get("duration_seconds"), parked_duration),
        )
        audio_trim_start = min(nominal_start_seconds, max(0.0, audio_source_duration - (1.0 / max(1.0, fps))))
        audio_trim_end = min(audio_source_duration, audio_trim_start + duration)
        audio_clip_duration = max(1.0 / max(1.0, fps), audio_trim_end - audio_trim_start)
        manifest["clips"].append({
            "id": audio_clip_id,
            "assetId": audio_asset_id,
            "type": "audio",
            "takeIndex": int(take_index),
            "timelineId": _take_timeline_id(take_index),
            "audioLane": _take_audio_lane_name(take_index),
            "startTime": float(start),
            # The editor displays the nominal window while the complete audio
            # source remains available for non-destructive roll reveals.
            "duration": float(audio_clip_duration),
            "sourceDuration": float(audio_source_duration),
            "nominalDuration": float(audio_clip_duration),
            "nominalDurationFrames": int(round(audio_clip_duration * max(1.0, fps))),
            "generationDuration": float(audio_source_duration),
            "generationDurationFrames": int(round(audio_source_duration * max(1.0, fps))),
            "preRoll": float(audio_trim_start),
            "postRoll": float(max(0.0, audio_source_duration - audio_trim_end)),
            "preRollFrames": int(round(audio_trim_start * max(1.0, fps))),
            "postRollFrames": int(round(max(0.0, audio_source_duration - audio_trim_end) * max(1.0, fps))),
            "rollContract": copy.deepcopy(roll_contract_data),
            "sourceDurationLimit": float(audio_source_duration),
            "trimStart": float(audio_trim_start),
            "trimEnd": float(audio_trim_end),
            "trackId": audio_track_id,
            "trackIndex": audio_track_index,
            "muted": False,
            "volume": 1.0,
            "linkedClipIds": [video_clip_id],
        })
    order = manifest.setdefault("assembly_order", [])
    label = f"T{take_index:02d}/A{take_index}"
    if label not in order:
        order.append(label)
    _update_manifest_duration(manifest)
    return manifest


class IAMCCS_ShotboardVideoEditorV1:
    """Manifest-based take collector/editor for rendered Shotboard generations."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "take_package_json": ("STRING", {"default": "", "forceInput": True}),
                "session_key": ("STRING", {"default": "shotboard_video_editor_v1"}),
                # Hidden in the custom UI. Keep these as strings so stale frontend
                # widget order cannot fail Comfy validation before collect() repairs it.
                "collect_policy": ("STRING", {"default": "append_sequence"}),
                "append_mode": ("STRING", {"default": "append_sequence"}),
                "fps": ("STRING", {"default": "24"}),
            },
            "optional": {
                "cine_editor_linx": (SUPERNODE_LINX_TYPE,),
                "video_1": ("VIDEO",),
                "video_2": ("VIDEO",),
                "video_3": ("VIDEO",),
                "video_4": ("VIDEO",),
                "video_5": ("VIDEO",),
                "audio_1": ("AUDIO",),
                "audio_2": ("AUDIO",),
                "audio_3": ("AUDIO",),
                "audio_4": ("AUDIO",),
                "audio_5": ("AUDIO",),
                "master_audio": ("AUDIO",),
                "editor_manifest_json": ("STRING", {"default": "", "multiline": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE, "STRING", "STRING")
    RETURN_NAMES = ("cine_linx", "editor_manifest_json", "report")
    FUNCTION = "collect"
    CATEGORY = "IAMCCS/Cine/Multigeneration"
    OUTPUT_NODE = True

    def collect(
        self,
        cine_linx,
        session_key,
        collect_policy,
        append_mode,
        fps,
        cine_editor_linx=None,
        take_package_json="",
        video_1=None,
        video_2=None,
        video_3=None,
        video_4=None,
        video_5=None,
        audio_1=None,
        audio_2=None,
        audio_3=None,
        audio_4=None,
        audio_5=None,
        master_audio=None,
        editor_manifest_json="",
        unique_id=None,
        extra_pnginfo=None,
    ):
        valid_collect_policies = {"append_sequence", "replace_same_take", "append_always"}
        valid_append_modes = {"append_sequence", "timeline_origin"}
        raw_fps = fps
        if (
            isinstance(raw_fps, str)
            and raw_fps.strip().startswith("{")
            and "iamccs.shotboard_video_editor.v1" in raw_fps
        ):
            if not editor_manifest_json:
                editor_manifest_json = raw_fps
            fps = 24.0
        fps = max(1.0, _safe_float(fps, 24.0))
        collect_policy = str(collect_policy or "").strip()
        append_mode = str(append_mode or "").strip()
        if collect_policy not in valid_collect_policies:
            collect_policy = "append_sequence"
        if append_mode not in valid_append_modes:
            append_mode = "append_sequence"
        session_key = str(session_key or "").strip()
        if session_key.startswith("{") and "iamccs.shotboard_video_editor.v1" in session_key:
            session_key = "shotboard_video_editor_v1"
        session_key = session_key or "shotboard_video_editor_v1"

        out_linx = _clone_linx(cine_linx, "iamccs_shotboard_video_editor_v1")
        resources = _resources(out_linx)
        outputs = _outputs(out_linx)
        editor_input_resources = _resources(cine_editor_linx) if isinstance(cine_editor_linx, dict) else {}
        editor_input_manifest = (
            editor_manifest_json
            or editor_input_resources.get("cine_video_editor_manifest_json")
            or editor_input_resources.get("editor_manifest_json")
            or ""
        )
        manifest = _editor_manifest_for_session(editor_input_manifest, session_key, fps)
        take_package = _safe_json_loads(take_package_json, {})
        if not isinstance(take_package, dict) or not take_package:
            take_package = _take_package_from_linx(out_linx)
        active_identity = _active_identity_from_linx(out_linx)
        videos = [video_1, video_2, video_3, video_4, video_5]
        audios = [audio_1, audio_2, audio_3, audio_4, audio_5]
        collected: List[Dict[str, Any]] = []
        input_count = sum(1 for item in videos if item is not None)
        if input_count and (not isinstance(take_package, dict) or not take_package):
            raise ValueError(
                "IAMCCS ShotboardVideoEditorV1 hard-fail: rendered video arrived without TakePackage. "
                "The editor will not infer T/A identity from input slot order."
            )
        package_take = _safe_int(take_package.get("take_index"), 0) if isinstance(take_package, dict) else 0
        package_timeline = str(take_package.get("timeline_id") or "").strip() if isinstance(take_package, dict) else ""
        package_audio_lane = str(take_package.get("audio_lane") or "").strip() if isinstance(take_package, dict) else ""
        identity_take = _safe_int(active_identity.get("take_index"), 0) if isinstance(active_identity, dict) else 0
        identity_timeline = str(active_identity.get("timeline_id") or "").strip() if isinstance(active_identity, dict) else ""
        for index, video in enumerate(videos, start=1):
            if video is None:
                continue
            if input_count == 1:
                if package_take <= 0 or not package_timeline or not package_audio_lane:
                    raise ValueError(
                        "IAMCCS ShotboardVideoEditorV1 hard-fail: incomplete TakePackage. "
                        f"take={package_take}, timeline={package_timeline!r}, audio_lane={package_audio_lane!r}."
                    )
                if identity_take <= 0 or not identity_timeline:
                    raise ValueError(
                        "IAMCCS ShotboardVideoEditorV1 hard-fail: cine_linx has no active T/A identity. "
                        "Generated media cannot be parked without a concrete take identity."
                    )
                if identity_take != package_take or identity_timeline != package_timeline:
                    raise ValueError(
                        "IAMCCS ShotboardVideoEditorV1 hard-fail: active identity does not match TakePackage. "
                        f"active=T{identity_take:02d}/{identity_timeline}, "
                        f"package=T{package_take:02d}/{package_timeline}. "
                        "No automatic correction is allowed."
                    )
                take_index = package_take
            else:
                take_index = index
            manifest_clips = manifest.get("clips") if isinstance(manifest.get("clips"), list) else []
            existing_same_take = any(
                isinstance(clip, dict)
                and str(clip.get("type")) == "video"
                and _safe_int(clip.get("takeIndex"), 0) == int(take_index)
                for clip in manifest_clips
            )
            if existing_same_take and str(collect_policy) == "append_sequence":
                raise ValueError(
                    "IAMCCS ShotboardVideoEditorV1 hard-fail: take already parked. "
                    f"T{take_index:02d} already exists in the editor manifest. "
                    "Use replace_same_take explicitly for retakes, or append_always for a new manual take."
                )
            if str(collect_policy) == "append_always":
                existing = [
                    _safe_int(clip.get("takeIndex"), 0)
                    for clip in manifest.get("clips", [])
                    if isinstance(clip, dict)
                ]
                if take_index in existing:
                    take_index = max(existing or [0]) + 1
            manifest = _append_video_take_to_manifest(
                manifest,
                session_key,
                take_index,
                video,
                audios[index - 1] if index - 1 < len(audios) else None,
                append_mode=str(append_mode),
                replace_existing_take=str(collect_policy) == "replace_same_take",
                target_duration_seconds=_safe_float(take_package.get("duration_seconds"), 0.0) if isinstance(take_package, dict) else 0.0,
                target_duration_frames=_safe_int(take_package.get("duration_frames"), 0) if isinstance(take_package, dict) else 0,
                tail_trim_frames=_safe_int(take_package.get("tail_trim_frames"), 0) if isinstance(take_package, dict) else 0,
                nominal_duration_seconds=(
                    _safe_float(take_package.get("nominal_duration_seconds"), 0.0)
                    or _safe_float(take_package.get("nominal_duration_frames"), 0.0) / max(1.0, fps)
                    if isinstance(take_package, dict) else 0.0
                ),
                nominal_duration_frames=_safe_int(take_package.get("nominal_duration_frames"), 0) if isinstance(take_package, dict) else 0,
                pre_roll_frames=_safe_int(take_package.get("pre_roll_frames"), 0) if isinstance(take_package, dict) else 0,
                post_roll_frames=_safe_int(take_package.get("post_roll_frames"), 0) if isinstance(take_package, dict) else 0,
                roll_contract=take_package.get("roll_contract") if isinstance(take_package, dict) and isinstance(take_package.get("roll_contract"), dict) else {},
            )
            collected.append({
                "take_index": int(take_index),
                "timeline_id": _take_timeline_id(take_index),
                "audio_lane": _take_audio_lane_name(take_index),
            })
        if master_audio is not None:
            master = _audio_manifest_entry(0, master_audio, session_key=session_key, root=_parking_root(session_key))
            if master:
                _replace_manifest_master_audio(manifest, master, fps)
        else:
            published_master = _master_audio_asset_from_linx(cine_linx) or _master_audio_asset_from_linx(cine_editor_linx)
            existing_master = _manifest_master_audio_item(
                manifest,
                manifest.get("assets") if isinstance(manifest.get("assets"), dict) else {},
            )
            if published_master and _master_audio_fingerprint(published_master) != _master_audio_fingerprint(existing_master):
                _replace_manifest_master_audio(manifest, published_master, fps)
        manifest_json = _json_dump(manifest)
        _VIDEO_EDITOR_MANIFEST_REGISTRY[_safe_slug(session_key)] = copy.deepcopy(manifest)
        editor_linx_out = _clone_linx(
            cine_editor_linx if isinstance(cine_editor_linx, dict) else {},
            "iamccs_video_editor_inputs",
        )
        editor_resources = _resources(editor_linx_out)
        editor_outputs = _outputs(editor_linx_out)
        editor_inputs = {
            "video_slots_present": [index for index, item in enumerate(videos, start=1) if item is not None],
            "audio_slots_present": [index for index, item in enumerate(audios, start=1) if item is not None],
            "master_audio_present": bool(master_audio is not None or _manifest_master_audio_item(manifest, manifest.get("assets") if isinstance(manifest.get("assets"), dict) else {})),
            "master_audio_source": "AUDIO socket" if master_audio is not None else ("cine_linx master_audio_asset" if _manifest_master_audio_item(manifest, manifest.get("assets") if isinstance(manifest.get("assets"), dict) else {}) else ""),
            "take_package_present": bool(isinstance(take_package, dict) and bool(take_package)),
            "session_key": str(session_key),
            "collect_policy": str(collect_policy),
            "append_mode": str(append_mode),
        }
        editor_resources["cine_video_editor_inputs"] = editor_inputs
        editor_resources["cine_video_editor_manifest"] = copy.deepcopy(manifest)
        editor_resources["cine_video_editor_manifest_json"] = manifest_json
        editor_outputs["editor_manifest_json"] = manifest_json
        editor_outputs["cine_video_editor_inputs_json"] = _json_dump(editor_inputs)
        resources["cine_video_editor_manifest"] = copy.deepcopy(manifest)
        resources["cine_video_editor_manifest_json"] = manifest_json
        resources["cine_editor_linx"] = copy.deepcopy(editor_linx_out)
        resources["cine_editor_linx_json"] = _json_dump(editor_linx_out)
        outputs["editor_manifest_json"] = manifest_json
        outputs["cine_editor_linx_json"] = resources["cine_editor_linx_json"]
        report_obj = {
            "node": "IAMCCS_ShotboardVideoEditorV1",
            "collected": collected,
            "asset_count": len(manifest.get("assets", {})),
            "clip_count": len(manifest.get("clips", [])),
            "duration_seconds": _safe_float(manifest.get("duration_seconds"), 0.0),
            "policy": str(collect_policy),
            "cine_editor_linx_inputs": editor_inputs,
            "truth": "Generated takes are persisted as manifest assets and timeline clips. T/A identity must match cine_linx and TakePackage; no inferred remap is allowed.",
        }
        report = _json_dump(report_obj)
        resources["cine_video_editor_report"] = report_obj
        outputs["report"] = report
        out_linx.setdefault("chain", []).append({
            "role": "shotboard_video_editor_v1",
            "name": "IAMCCS_ShotboardVideoEditorV1",
            "collected": collected,
        })
        _refresh_linx_index(out_linx)
        try:
            workflow = None
            if isinstance(extra_pnginfo, list) and extra_pnginfo and isinstance(extra_pnginfo[0], dict):
                workflow = extra_pnginfo[0].get("workflow")
            elif isinstance(extra_pnginfo, dict):
                workflow = extra_pnginfo.get("workflow")
            uid = unique_id[0] if isinstance(unique_id, list) and unique_id else unique_id
            if workflow and uid is not None:
                for item in workflow.get("nodes", []):
                    if str(item.get("id")) == str(uid):
                        widgets = list(item.get("widgets_values") or [])
                        while len(widgets) < 5:
                            widgets.append("")
                        widgets[0] = str(session_key)
                        widgets[1] = str(collect_policy)
                        widgets[2] = str(append_mode)
                        widgets[3] = float(fps)
                        widgets[4] = manifest_json
                        item["widgets_values"] = widgets
                        break
        except Exception as exc:
            print(f"[IAMCCS ShotboardVideoEditorV1] UI widget sync skipped: {exc}")
        return {
            "ui": {
                "text": [report],
                "iamccs_video_editor_manifest": [manifest_json],
            },
            "result": (out_linx, manifest_json, report),
        }


class IAMCCS_ShotboardVideoEditorRenderV1:
    """Render a manifest-based editor assembly back to a Comfy VIDEO."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "editor_manifest_json": ("STRING", {"default": "", "multiline": True}),
                # Kept as strings for legacy workflow safety. render() normalizes
                # enum-like values before using them, so old widget states cannot
                # fail Comfy validation before migration.
                "audio_policy": ("STRING", {"default": "concat_clip_audio"}),
                "fps_mode": ("STRING", {"default": "from_manifest"}),
                "override_fps": ("STRING", {"default": "24"}),
                "tail_trim_frames_per_clip": ("INT", {"default": 0, "min": 0, "max": 12, "step": 1}),
            },
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "STRING")
    RETURN_NAMES = ("video", "frames", "report")
    FUNCTION = "render"
    CATEGORY = "IAMCCS/Cine/Multigeneration"
    OUTPUT_NODE = True

    def render(self, editor_manifest_json, audio_policy, fps_mode, override_fps, tail_trim_frames_per_clip=0):
        legacy_widget_migration = {}
        raw_audio_policy = audio_policy
        raw_fps_mode = fps_mode
        if str(audio_policy) not in {"concat_clip_audio", "use_master_audio", "first_video_audio", "silent"}:
            audio_policy = "concat_clip_audio"
            legacy_widget_migration["audio_policy"] = {
                "from": str(raw_audio_policy),
                "to": str(audio_policy),
            }
        numeric_fps_mode = isinstance(fps_mode, (int, float))
        if not numeric_fps_mode:
            try:
                numeric_fps_mode = str(fps_mode).strip().replace(".", "", 1).isdigit()
            except Exception:
                numeric_fps_mode = False
        if numeric_fps_mode:
            override_fps = _safe_float(fps_mode, _safe_float(override_fps, 24.0))
            fps_mode = "override_fps"
            legacy_widget_migration["fps_mode"] = {
                "from": str(raw_fps_mode),
                "to": str(fps_mode),
                "override_fps": float(override_fps),
            }
        elif str(fps_mode) not in {"from_manifest", "override_fps"}:
            fps_mode = "from_manifest"
            legacy_widget_migration["fps_mode"] = {
                "from": str(raw_fps_mode),
                "to": str(fps_mode),
            }
        manifest = _normalize_editor_manifest(editor_manifest_json, "shotboard_video_editor_v1", override_fps)
        manifest_tail_trim = _safe_int(manifest.get("render_tail_trim_frames"), 0)
        effective_tail_trim = max(0, _safe_int(tail_trim_frames_per_clip, 0))
        if effective_tail_trim <= 0 and manifest_tail_trim > 0:
            effective_tail_trim = manifest_tail_trim
        clips = [
            clip for clip in manifest.get("clips", [])
            if isinstance(clip, dict) and str(clip.get("type")) == "video"
        ]
        clips.sort(key=lambda clip: (_safe_float(clip.get("startTime"), 0.0), _safe_int(clip.get("takeIndex"), 0)))
        if not clips:
            raise ValueError("IAMCCS ShotboardVideoEditorRenderV1: no video clips found in editor manifest.")
        assets = manifest.get("assets") if isinstance(manifest.get("assets"), dict) else {}
        comps = []
        for clip in clips:
            asset = assets.get(str(clip.get("assetId"))) if isinstance(assets.get(str(clip.get("assetId"))), dict) else {}
            parked = _load_parked_video_clip(asset)
            if parked is None and bool(asset.get("manual")):
                parked = _load_manual_video_clip(asset)
            if parked is None:
                raise ValueError(f"IAMCCS ShotboardVideoEditorRenderV1: missing parked video asset for clip {clip.get('id')}.")
            comp = parked
            fps = float(comp.frame_rate or manifest.get("fps") or 24.0)
            trim_start = _safe_float(clip.get("trimStart"), 0.0)
            trim_end = _safe_float(clip.get("trimEnd"), 0.0)
            asset_limit = _safe_float(asset.get("timeline_duration"), 0.0)
            if asset_limit > 0:
                trim_end = min(trim_end if trim_end > 0 else asset_limit, asset_limit)
            tail_trim = effective_tail_trim
            if tail_trim > 0:
                source_end = trim_end if trim_end > 0 else int(comp.images.shape[0]) / max(1.0, fps)
                trim_end = max(trim_start + (1.0 / max(1.0, fps)), source_end - (tail_trim / max(1.0, fps)))
            images, audio, _, _ = _trim_component(
                comp,
                fps,
                trim_start,
                trim_end,
            )
            comps.append(Types.VideoComponents(images=images, audio=audio, frame_rate=comp.frame_rate))
        first = comps[0]
        first_shape = tuple(first.images.shape[1:3])
        frame_batches = []
        audio_items = []
        for comp in comps:
            if tuple(comp.images.shape[1:3]) != first_shape:
                raise ValueError("IAMCCS ShotboardVideoEditorRenderV1: all parked clips must share width/height for V1 assembly.")
            frame_batches.append(comp.images.to(first.images.device))
            audio_items.append((comp.audio, int(comp.images.shape[0]), float(comp.frame_rate or manifest.get("fps") or 24.0)))
        frames = torch.cat(frame_batches, dim=0)
        fps = float(override_fps) if str(fps_mode) == "override_fps" else _safe_float(manifest.get("fps"), float(first.frame_rate or 24.0))
        audio = None
        master_audio_source = ""
        effective_audio_policy = str(audio_policy)
        manifest_audio_policy = str(manifest.get("render_audio_policy") or manifest.get("audio_policy") or "").strip()
        if effective_audio_policy not in {"concat_clip_audio", "use_master_audio", "first_video_audio", "silent"}:
            effective_audio_policy = manifest_audio_policy if manifest_audio_policy in {"concat_clip_audio", "use_master_audio", "first_video_audio", "silent"} else "concat_clip_audio"
        if effective_audio_policy == "use_master_audio":
            # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
            master_item = _manifest_master_audio_item(manifest, assets)
            master_clip = _manifest_master_audio_clip(manifest)
            if isinstance(master_item, dict) and isinstance(master_clip, dict):
                master_item = copy.deepcopy(master_item)
                if master_clip.get("renderTrimStartSeconds") is not None:
                    master_item["_render_trim_start_seconds"] = master_clip.get("renderTrimStartSeconds")
                if master_clip.get("renderTrimEndSeconds") is not None:
                    master_item["_render_trim_end_seconds"] = master_clip.get("renderTrimEndSeconds")
                if master_clip.get("preRollFrames") is not None:
                    master_item["preRollFrames"] = master_clip.get("preRollFrames")
                if master_clip.get("nominalDurationSeconds") is not None:
                    master_item["nominalDurationSeconds"] = master_clip.get("nominalDurationSeconds")
            audio = _load_audio_manifest_entry(master_item)
            master_audio_source = str(master_item.get("role") or master_item.get("id") or "master_audio") if isinstance(master_item, dict) else ""
            if audio is None:
                asset_keys = sorted(str(key) for key in assets.keys()) if isinstance(assets, dict) else []
                raise ValueError(
                    "IAMCCS ShotboardVideoEditorRenderV1: audio_policy=use_master_audio but no readable "
                    "master_excerpt/master_audio asset was found in the editor manifest. "
                    f"asset_keys={asset_keys}; master_item={master_item!r}"
                )
        elif effective_audio_policy == "first_video_audio":
            audio = first.audio
        elif effective_audio_policy == "concat_clip_audio":
            audio = _concat_audio(audio_items)
            manual_audio_items = []
            for clip in manifest.get("clips", []):
                if not isinstance(clip, dict) or str(clip.get("type")) != "audio":
                    continue
                track_id = str(clip.get("trackId") or "").strip().upper()
                audio_lane = str(clip.get("audioLane") or "").strip().upper()
                role = str(clip.get("role") or "").strip().lower()
                if track_id in {"AM", "MASTER"} or audio_lane == "MASTER" or role in {"master_audio", "master_excerpt"}:
                    continue
                asset = assets.get(str(clip.get("assetId"))) if isinstance(assets.get(str(clip.get("assetId"))), dict) else {}
                if bool(asset.get("manual")):
                    manual_audio_items.append((clip, asset))
            if manual_audio_items:
                audio = _mix_manual_audio_into_timeline(audio, manual_audio_items, int(frames.shape[0]), fps)
        video = InputImpl.VideoFromComponents(Types.VideoComponents(
            images=frames,
            audio=audio,
            frame_rate=Fraction(round(max(1.0, fps) * 1000), 1000),
        ))
        report = _json_dump({
            "node": "IAMCCS_ShotboardVideoEditorRenderV1",
            "clip_count": len(comps),
            "frames": int(frames.shape[0]),
            "fps": float(fps),
            "duration_seconds": int(frames.shape[0]) / max(1.0, fps),
            "audio_policy": effective_audio_policy,
            "requested_audio_policy": str(audio_policy),
            "master_audio_source": master_audio_source,
            "tail_trim_frames_per_clip": int(effective_tail_trim),
            "tail_trim_source": "render_widget" if max(0, _safe_int(tail_trim_frames_per_clip, 0)) > 0 else "editor_manifest",
            "has_audio": audio is not None,
            "legacy_widget_migration": legacy_widget_migration,
        })
        return video, frames, report


class IAMCCS_VideoHardConcat:
    """Hard-concatenate generated take videos into a final VIDEO object."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_1": ("VIDEO",),
                "audio_policy": (["concat_clip_audio", "use_master_audio", "first_video_audio", "silent"], {"default": "concat_clip_audio"}),
                "fps_mode": (["from_first_video", "override_fps"], {"default": "from_first_video"}),
                "override_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
            },
            "optional": {
                "video_2": ("VIDEO",),
                "video_3": ("VIDEO",),
                "video_4": ("VIDEO",),
                "video_5": ("VIDEO",),
                "master_audio": ("AUDIO",),
                "concat_plan_json": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "STRING")
    RETURN_NAMES = ("video", "frames", "report")
    FUNCTION = "concat"
    CATEGORY = "IAMCCS/Cine/Multigeneration"

    def concat(
        self,
        video_1,
        audio_policy,
        fps_mode,
        override_fps,
        video_2=None,
        video_3=None,
        video_4=None,
        video_5=None,
        master_audio=None,
        concat_plan_json="",
    ):
        videos = [video for video in (video_1, video_2, video_3, video_4, video_5) if video is not None]
        if not videos:
            raise ValueError("IAMCCS Video Hard Concat: at least video_1 is required.")

        components = [_video_components(video) for video in videos]
        first = components[0]
        first_shape = tuple(first.images.shape[1:3])
        first_device = first.images.device
        frame_batches = []
        frame_counts = []
        for index, comp in enumerate(components):
            if tuple(comp.images.shape[1:3]) != first_shape:
                raise ValueError(
                    "IAMCCS Video Hard Concat: all takes must share the same height and width. "
                    f"video_1={first_shape}, video_{index + 1}={tuple(comp.images.shape[1:3])}"
                )
            images = comp.images
            if images.device != first_device:
                images = images.to(first_device)
            frame_batches.append(images)
            frame_counts.append(int(images.shape[0]))

        frames = torch.cat(frame_batches, dim=0)
        if str(fps_mode) == "override_fps":
            fps = float(override_fps)
        else:
            fps = float(first.frame_rate)
        frame_rate = Fraction(round(max(1.0, fps) * 1000), 1000)

        audio = None
        if str(audio_policy) == "use_master_audio":
            audio = master_audio
        elif str(audio_policy) == "first_video_audio":
            audio = first.audio
        elif str(audio_policy) == "concat_clip_audio":
            audio = _concat_audio([
                (comp.audio, int(comp.images.shape[0]), float(comp.frame_rate))
                for comp in components
            ])

        video = InputImpl.VideoFromComponents(Types.VideoComponents(images=frames, audio=audio, frame_rate=frame_rate))
        concat_plan = _safe_json_loads(concat_plan_json, {})
        report = _json_dump({
            "node": "IAMCCS_VideoHardConcat",
            "policy": "hard_cut_tensor_concat",
            "take_count": len(videos),
            "frames_per_take": frame_counts,
            "total_frames": int(frames.shape[0]),
            "fps": float(frame_rate),
            "duration_seconds": int(frames.shape[0]) / max(1.0, float(frame_rate)),
            "audio_policy": str(audio_policy),
            "has_audio": audio is not None,
            "concat_plan_takes": len(concat_plan.get("takes", [])) if isinstance(concat_plan, dict) else 0,
        })
        return video, frames, report


class IAMCCS_VideoColorCorrectionControl:
    """Attach modular color-correction metadata to cine_linx for editor/render use."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "enabled": ("BOOLEAN", {"default": True}),
                "exposure": ("FLOAT", {"default": 0.0, "min": -4.0, "max": 4.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "tint": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "vignette": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "look_name": ("STRING", {"default": "neutral"}),
            },
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE, "STRING", "STRING")
    RETURN_NAMES = ("cine_linx", "color_correction_json", "report")
    FUNCTION = "control"
    CATEGORY = "IAMCCS/Cine/Video Editor"

    def control(
        self,
        cine_linx,
        enabled,
        exposure,
        contrast,
        saturation,
        gamma,
        temperature,
        tint,
        vignette,
        look_name,
    ):
        if isinstance(cine_linx, dict):
            # Keep this node lightweight: color controls must not deep-copy
            # large editor manifests, parked assets, or future media handles.
            out_linx = dict(cine_linx)
            out_linx["resources"] = dict(cine_linx.get("resources", {}) if isinstance(cine_linx.get("resources", {}), dict) else {})
            out_linx["outputs"] = dict(cine_linx.get("outputs", {}) if isinstance(cine_linx.get("outputs", {}), dict) else {})
            out_linx["chain"] = list(cine_linx.get("chain", []) if isinstance(cine_linx.get("chain", []), list) else [])
            out_linx["stages"] = list(cine_linx.get("stages", []) if isinstance(cine_linx.get("stages", []), list) else [])
            out_linx["mode"] = "iamccs_video_color_correction"
        else:
            out_linx = _clone_linx(cine_linx, "iamccs_video_color_correction")
        payload = {
            "schema": "iamccs.video_color_correction.v1",
            "enabled": bool(enabled),
            "look_name": str(look_name or "neutral"),
            "exposure": float(exposure),
            "contrast": float(contrast),
            "saturation": float(saturation),
            "gamma": float(gamma),
            "temperature": float(temperature),
            "tint": float(tint),
            "vignette": float(vignette),
            "truth": "Metadata-only color correction control. Downstream editor/render nodes can apply this intent without changing Shotboard generation.",
        }
        resources = _resources(out_linx)
        outputs = _outputs(out_linx)
        resources["cine_video_color_correction"] = payload
        resources["cine_video_color_correction_json"] = _json_dump(payload)
        outputs["color_correction_json"] = _json_dump(payload)
        _refresh_linx_index(out_linx)
        report = _json_dump({
            "node": "IAMCCS_VideoColorCorrectionControl",
            "enabled": bool(enabled),
            "look_name": payload["look_name"],
            "exposure": float(exposure),
            "contrast": float(contrast),
            "saturation": float(saturation),
        })
        return out_linx, _json_dump(payload), report


NODE_CLASS_MAPPINGS = {
    "IAMCCS_ShotboardV4MultigenAdapter": IAMCCS_ShotboardV4MultigenAdapter,
    "IAMCCS_MultiTimelineBridge": IAMCCS_MultiTimelineBridge,
    "IAMCCS_TakePackage": IAMCCS_TakePackage,
    "IAMCCS_TakeRouter": IAMCCS_TakeRouter,
    "IAMCCS_MultiTimelineTakePicker": IAMCCS_MultiTimelineTakePicker,
    "IAMCCS_ShotboardVideoEditorV1": IAMCCS_ShotboardVideoEditorV1,
    "IAMCCS_ShotboardVideoEditorRenderV1": IAMCCS_ShotboardVideoEditorRenderV1,
    "IAMCCS_VideoHardConcat": IAMCCS_VideoHardConcat,
    "IAMCCS_VideoColorCorrectionControl": IAMCCS_VideoColorCorrectionControl,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_ShotboardV4MultigenAdapter": "IAMCCS Shotboard V4 Multigen Adapter",
    "IAMCCS_MultiTimelineBridge": "IAMCCS MultiTimeline Bridge",
    "IAMCCS_TakePackage": "IAMCCS TakePackage",
    "IAMCCS_TakeRouter": "IAMCCS Take Router",
    "IAMCCS_MultiTimelineTakePicker": "IAMCCS MultiTimeline Take Picker",
    "IAMCCS_ShotboardVideoEditorV1": "IAMCCS Shotboard Video Editor V1",
    "IAMCCS_ShotboardVideoEditorRenderV1": "IAMCCS Shotboard Video Editor Render V1",
    "IAMCCS_VideoHardConcat": "IAMCCS Video Hard Concat",
    "IAMCCS_VideoColorCorrectionControl": "IAMCCS Video Color Correction Control",
}



