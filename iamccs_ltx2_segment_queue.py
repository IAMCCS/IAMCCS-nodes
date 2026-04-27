from __future__ import annotations

import copy
import glob
import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
import wave
from pathlib import Path

import numpy as np
import server
import torch
from comfy.cli_args import args
from comfy_api.latest import Types

try:
    from safetensors.torch import load_file as _safetensors_load_file  # type: ignore[import]
    from safetensors.torch import save_file as _safetensors_save_file  # type: ignore[import]
except Exception:
    _safetensors_load_file = None
    _safetensors_save_file = None


log = logging.getLogger("IAMCCS.LTX2.SegmentQueue")

# Carmine Cristallo Scalzi AI reasearch (IAMCCS) - patreon.com/IAMCCS

def _bridge_dir() -> Path:
    try:
        import folder_paths  # type: ignore[import]

        base = Path(folder_paths.get_output_directory())
    except Exception:
        base = Path("output")
    path = base / "iamccs_ltx2_bridges"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _bridge_path(bridge_name: str, render_id: str) -> Path:
    safe_bridge = str(bridge_name or "ltx2_bridge").strip() or "ltx2_bridge"
    safe_render = str(render_id or "").strip() or "default"
    return _bridge_dir() / f"{safe_bridge}_{safe_render}.png"


def _latent_bridge_path(render_id: str) -> Path:
    safe_render = str(render_id or "").strip() or "default"
    return _bridge_dir() / f"ltx2_latent_bridge_{safe_render}.safetensors"


def _latent_bridge_manifest_path(render_id: str) -> Path:
    safe_render = str(render_id or "").strip() or "default"
    return _bridge_dir() / f"ltx2_latent_bridge_{safe_render}.json"


def _resolve_latent_bridge_payload_path(render_id: str) -> Path:
    manifest_path = _latent_bridge_manifest_path(render_id)
    legacy_path = _latent_bridge_path(render_id)
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload_name = data.get("current_file")
            if payload_name:
                payload_path = _bridge_dir() / str(payload_name)
                if payload_path.exists():
                    return payload_path
        except Exception:
            pass
    return legacy_path


def _write_latent_bridge_manifest(render_id: str, payload_path: Path) -> None:
    manifest_path = _latent_bridge_manifest_path(render_id)
    manifest_path.write_text(
        json.dumps({"current_file": payload_path.name}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def _new_latent_bridge_payload_path(render_id: str) -> Path:
    safe_render = str(render_id or "").strip() or "default"
    return _bridge_dir() / f"ltx2_latent_bridge_{safe_render}_{uuid.uuid4().hex[:10]}.safetensors"


def _save_latent_bridge(
    render_id: str,
    latent_tail: torch.Tensor,
    latent_full: torch.Tensor | None = None,
    latent_reference: torch.Tensor | None = None,
    seed_offset_latent_frames: int | None = None,
    overlap_latent_frames: int | None = None,
) -> None:
    path = _new_latent_bridge_payload_path(render_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tail_cpu = latent_tail.detach().cpu().contiguous()
    full_cpu = (
        latent_full.detach().cpu().contiguous()
        if torch.is_tensor(latent_full)
        else None
    )
    reference_cpu = (
        latent_reference.detach().cpu().contiguous()
        if torch.is_tensor(latent_reference)
        else None
    )
    if full_cpu is not None and reference_cpu is not None:
        same_storage = False
        try:
            same_storage = full_cpu.untyped_storage().data_ptr() == reference_cpu.untyped_storage().data_ptr()
        except Exception:
            same_storage = full_cpu.data_ptr() == reference_cpu.data_ptr()
        if same_storage:
            reference_cpu = reference_cpu.clone()
    seed_offset_cpu = torch.tensor([int(seed_offset_latent_frames or 0)], dtype=torch.int64)
    overlap_latent_cpu = torch.tensor([int(overlap_latent_frames or 0)], dtype=torch.int64)
    if _safetensors_save_file is not None:
        tensors = {"latent_tail": tail_cpu}
        metadata = {"frames": str(int(tail_cpu.shape[2]))}
        if full_cpu is not None:
            tensors["latent_full"] = full_cpu
            metadata["full_frames"] = str(int(full_cpu.shape[2]))
        if reference_cpu is not None:
            tensors["latent_reference"] = reference_cpu
            metadata["reference_frames"] = str(int(reference_cpu.shape[2]))
        tensors["latent_seed_offset"] = seed_offset_cpu
        tensors["latent_overlap_frames"] = overlap_latent_cpu
        metadata["seed_offset_latent_frames"] = str(int(seed_offset_cpu.item()))
        metadata["overlap_latent_frames"] = str(int(overlap_latent_cpu.item()))
        _safetensors_save_file(tensors, str(path), metadata=metadata)
        _write_latent_bridge_manifest(render_id, path)
        return
    payload = {"latent_tail": tail_cpu}
    if full_cpu is not None:
        payload["latent_full"] = full_cpu
    if reference_cpu is not None:
        payload["latent_reference"] = reference_cpu
    payload["latent_seed_offset"] = seed_offset_cpu
    payload["latent_overlap_frames"] = overlap_latent_cpu
    torch.save(payload, str(path))
    _write_latent_bridge_manifest(render_id, path)


def _load_latent_bridge(path: Path) -> dict[str, torch.Tensor | None]:
    if _safetensors_load_file is not None:
        tensors = _safetensors_load_file(str(path))
        return {
            "latent_tail": tensors["latent_tail"],
            "latent_full": tensors.get("latent_full"),
            "latent_reference": tensors.get("latent_reference"),
            "latent_seed_offset": tensors.get("latent_seed_offset"),
            "latent_overlap_frames": tensors.get("latent_overlap_frames"),
        }
    data = torch.load(str(path), map_location="cpu", weights_only=False)
    if torch.is_tensor(data):
        return {"latent_tail": data, "latent_full": None, "latent_reference": None, "latent_seed_offset": None, "latent_overlap_frames": None}
    if isinstance(data, dict):
        if "latent_tail" in data:
            return {
                "latent_tail": data["latent_tail"],
                "latent_full": data.get("latent_full"),
                "latent_reference": data.get("latent_reference"),
                "latent_seed_offset": data.get("latent_seed_offset"),
                "latent_overlap_frames": data.get("latent_overlap_frames"),
            }
        if "samples" in data and torch.is_tensor(data["samples"]):
            return {"latent_tail": data["samples"], "latent_full": data["samples"], "latent_reference": None, "latent_seed_offset": None, "latent_overlap_frames": None}
    raise ValueError(f"Unsupported latent bridge payload: {path}")


def _pixel_frames_to_latent_frames(pixel_frames: int, time_scale_factor: int) -> int:
    pixel_frames = int(pixel_frames)
    if pixel_frames <= 0:
        return 0
    time_scale_factor = max(int(time_scale_factor), 1)
    return 1 + max(0, (pixel_frames - 1) // time_scale_factor)


def _get_time_scale_factor_from_vae(vae) -> int:
    ts = getattr(vae, "downscale_index_formula", None)
    if ts and isinstance(ts, (tuple, list)) and len(ts) >= 1:
        try:
            return int(ts[0])
        except Exception:
            pass
    return 8


def _save_last_frame_png(path: Path, image_tensor: torch.Tensor) -> None:
    from PIL import Image  # type: ignore[import]

    path.parent.mkdir(parents=True, exist_ok=True)
    image = image_tensor.detach().cpu().float().clamp(0, 1)
    arr = (image.numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(str(path))


def _load_png(path: Path) -> torch.Tensor:
    from PIL import Image  # type: ignore[import]

    arr = np.asarray(Image.open(str(path)).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _find_ffmpeg() -> str | None:
    forced = os.environ.get("VHS_FORCE_FFMPEG_PATH")
    if forced and os.path.isfile(forced):
        return forced

    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        ffmpeg_path = get_ffmpeg_exe()
        if ffmpeg_path and os.path.isfile(ffmpeg_path):
            return ffmpeg_path
    except Exception:
        pass

    return shutil.which("ffmpeg")


def _normalize_prompt_keys(prompt):
    if prompt is None:
        return None
    return {str(key): value for key, value in prompt.items()}


def _normalize_outputs_to_execute(outputs_to_execute):
    if outputs_to_execute is None:
        return None
    return [str(item) for item in outputs_to_execute]


def _infer_output_nodes(prompt):
    import nodes  # type: ignore[import]

    normalized_prompt = _normalize_prompt_keys(prompt) or {}
    outputs = []
    for node_id, node in normalized_prompt.items():
        class_type = node.get("class_type")
        class_def = nodes.NODE_CLASS_MAPPINGS.get(class_type)
        if class_def is not None and getattr(class_def, "OUTPUT_NODE", False):
            outputs.append(str(node_id))
    return outputs


def _get_current_queue_item():
    prompt_server = getattr(server.PromptServer, "instance", None)
    if prompt_server is None or getattr(prompt_server, "prompt_queue", None) is None:
        raise RuntimeError("PromptServer prompt queue is unavailable")

    currently_running = getattr(prompt_server.prompt_queue, "currently_running", {})
    if not currently_running:
        raise RuntimeError("No currently running prompt was found")

    current = next(iter(currently_running.values()))
    if len(current) == 6:
        (_, _, prompt, extra_data, outputs_to_execute, sensitive) = current
    else:
        (_, _, prompt, extra_data, outputs_to_execute) = current
        sensitive = {}
    return prompt, extra_data, outputs_to_execute, sensitive


def _enqueue_prompt(prompt, extra_data=None, outputs_to_execute=None, sensitive=None):
    prompt_server = getattr(server.PromptServer, "instance", None)
    if prompt_server is None or getattr(prompt_server, "prompt_queue", None) is None:
        raise RuntimeError("PromptServer prompt queue is unavailable")

    prompt_queue = prompt_server.prompt_queue
    prompt = _normalize_prompt_keys(prompt)

    try:
        _, current_extra_data, current_outputs_to_execute, current_sensitive = _get_current_queue_item()
    except Exception:
        current_extra_data = None
        current_outputs_to_execute = None
        current_sensitive = None

    if extra_data is None:
        extra_data = current_extra_data if current_extra_data is not None else {}
    if sensitive is None:
        sensitive = current_sensitive if current_sensitive is not None else {}
    if outputs_to_execute is None:
        outputs_to_execute = current_outputs_to_execute

    outputs_to_execute = _normalize_outputs_to_execute(outputs_to_execute)
    if outputs_to_execute is None:
        outputs_to_execute = _infer_output_nodes(prompt)
    if not outputs_to_execute:
        raise RuntimeError("No output nodes were found for the requeued prompt")

    number = -prompt_server.number
    prompt_server.number += 1
    prompt_id = str(server.uuid.uuid4())
    log.info("[IAMCCS LTX2] Queueing next segment prompt %s with outputs %s", prompt_id, outputs_to_execute)
    prompt_queue.put((number, prompt_id, prompt, extra_data, outputs_to_execute, sensitive))


def _build_metadata(prompt, extra_pnginfo):
    if args.disable_metadata:
        return None
    metadata = {}
    if extra_pnginfo is not None:
        metadata.update(extra_pnginfo)
    if prompt is not None:
        metadata["prompt"] = prompt
    return metadata or None


def _resolve_output_location(filename_prefix, video):
    import folder_paths  # type: ignore[import]

    width, height = video.get_dimensions()
    full_output_folder, filename, counter, subfolder, _resolved_prefix = folder_paths.get_save_image_path(
        filename_prefix,
        folder_paths.get_output_directory(),
        width,
        height,
    )
    os.makedirs(full_output_folder, exist_ok=True)
    next_counter = max(int(counter or 1), 1)
    while True:
        unique_base_name = f"{filename}_{next_counter:05d}"
        existing_matches = glob.glob(os.path.join(full_output_folder, f"{unique_base_name}*"))
        if not existing_matches:
            break
        next_counter += 1
    return full_output_folder, unique_base_name, subfolder


def _video_extension():
    return Types.VideoContainer.get_extension("auto")


def _segment_filename(base_name, render_id, segment_index):
    return f"{base_name}_{render_id}_seg_{segment_index + 1:04d}.{_video_extension()}"


def _final_filename(base_name, render_id):
    return f"{base_name}_{render_id}_full.{_video_extension()}"


def _concat_segments(segment_paths, output_path):
    ffmpeg_path = _find_ffmpeg()
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg was not found, so segment videos could not be concatenated.")

    missing_paths = [segment_path for segment_path in segment_paths if not os.path.exists(segment_path)]
    if missing_paths:
        missing_lines = "\n".join(missing_paths)
        raise RuntimeError(f"Segment merge aborted because these files are missing:\n{missing_lines}")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
        list_path = handle.name
        for segment_path in segment_paths:
            escaped = segment_path.replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")

    cmd = [
        ffmpeg_path,
        "-nostdin",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path,
        "-fflags",
        "+genpts",
        "-avoid_negative_ts",
        "make_zero",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    try:
        os.remove(list_path)
    except Exception:
        pass
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"ffmpeg concat failed: {stderr}")


def _normalize_audio(audio):
    waveform = None
    sample_rate = None
    if isinstance(audio, dict):
        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate")
    elif hasattr(audio, "get"):
        try:
            waveform = audio.get("waveform")
            sample_rate = audio.get("sample_rate")
        except Exception:
            waveform = None
            sample_rate = None

    if waveform is None and hasattr(audio, "waveform"):
        waveform = getattr(audio, "waveform")
        sample_rate = getattr(audio, "sample_rate", sample_rate)

    if waveform is None and isinstance(audio, (tuple, list)) and audio:
        waveform = audio[0]
        if len(audio) > 1:
            sample_rate = audio[1]

    if waveform is None:
        return None, None

    if not torch.is_tensor(waveform):
        waveform = torch.as_tensor(waveform)

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)

    return waveform.detach().cpu().float(), int(sample_rate or 48000)


def _write_audio_wav(audio, wav_path: str):
    waveform, sample_rate = _normalize_audio(audio)
    if waveform is None:
        return False

    audio_np = waveform.squeeze(0).numpy()
    if audio_np.ndim == 1:
        audio_np = audio_np[np.newaxis, :]
    audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=0.0, neginf=0.0)
    audio_np = np.clip(audio_np, -1.0, 1.0)
    pcm = (audio_np.T * 32767.0).round().astype(np.int16)

    with wave.open(wav_path, "wb") as wav_file:
        wav_file.setnchannels(int(pcm.shape[1]))
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm.tobytes())
    return True


def _trim_audio_leading(audio, trim_frames: int, frame_rate: float):
    waveform, sample_rate = _normalize_audio(audio)
    if waveform is None:
        return audio

    trim_frames = max(0, int(trim_frames))
    frame_rate = float(max(0.001, frame_rate))
    trim_samples = int(round((float(trim_frames) / frame_rate) * float(sample_rate)))
    if trim_samples <= 0:
        return {"waveform": waveform, "sample_rate": sample_rate}

    total_samples = int(waveform.shape[-1])
    trim_samples = min(trim_samples, max(0, total_samples - 1))
    return {
        "waveform": waveform[:, :, trim_samples:],
        "sample_rate": sample_rate,
    }


def _save_segment_from_images(images, audio, frame_rate, output_path):
    ffmpeg_path = _find_ffmpeg()
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg was not found, so the segment video could not be written.")

    if images is None or int(images.shape[0]) <= 0:
        raise ValueError("images input is required when no VIDEO object is provided.")

    frame_rate = float(max(0.001, frame_rate))
    with tempfile.TemporaryDirectory(prefix="iamccs_ltx2_seg_") as temp_dir:
        from PIL import Image  # type: ignore[import]

        for index in range(int(images.shape[0])):
            frame = images[index].detach().cpu().float().clamp(0, 1)
            arr = (frame.numpy() * 255.0).round().astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(temp_dir, f"frame_{index:05d}.png"))

        wav_path = os.path.join(temp_dir, "audio.wav")
        has_audio = audio is not None and _write_audio_wav(audio, wav_path)

        cmd = [
            ffmpeg_path,
            "-nostdin",
            "-y",
            "-framerate",
            f"{frame_rate:.6f}",
            "-i",
            os.path.join(temp_dir, "frame_%05d.png"),
        ]
        if has_audio:
            cmd += ["-i", wav_path]
        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
        ]
        if has_audio:
            cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest"]
        cmd += ["-movflags", "+faststart", output_path]

        result = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if result.returncode != 0:
            stderr = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"ffmpeg segment encode failed: {stderr}")


class IAMCCS_LTX2_LastFrameBridgeLoad:
    # Carmine Cristallo Scalzi AI reasearch (IAMCCS) - patreon.com/IAMCCS
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bridge_name": ("STRING", {"default": "ltx2_detailer_bridge"}),
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
            },
            "optional": {
                "render_id": ("STRING", {"default": ""}),
                "fallback_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("image", "exists", "report")
    FUNCTION = "load"
    CATEGORY = "IAMCCS/LTX-2"

    def load(self, bridge_name, segment_index, render_id="", fallback_image=None):
        segment_index = int(segment_index)
        active_render_id = str(render_id or "").strip()
        if segment_index <= 0 or not active_render_id:
            if fallback_image is None:
                raise ValueError("No bridge available for segment 0. Connect fallback_image for the first pass.")
            return (fallback_image, 0, "LTX2 last-frame bridge: using fallback_image for initial segment")

        bridge_path = _bridge_path(bridge_name, active_render_id)
        if bridge_path.exists():
            image = _load_png(bridge_path)
            return (image, 1, f"LTX2 last-frame bridge: loaded {bridge_path.name}")

        if fallback_image is not None:
            return (fallback_image, 0, f"LTX2 last-frame bridge missing ({bridge_path.name}), using fallback_image")

        raise FileNotFoundError(f"Last-frame bridge not found: {bridge_path}")


class IAMCCS_LTX2_LastFrameBridgeSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "bridge_name": ("STRING", {"default": "ltx2_detailer_bridge"}),
                "render_id": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "save"
    CATEGORY = "IAMCCS/LTX-2"

    def save(self, images, bridge_name, render_id):
        active_render_id = str(render_id or "").strip()
        if not active_render_id:
            return (images, "LTX2 last-frame bridge save disabled: render_id empty")

        if images is None or not torch.is_tensor(images) or images.ndim != 4 or images.shape[0] <= 0:
            raise ValueError("images must be an IMAGE batch [N,H,W,C] with at least one frame")

        last_frame = images[-1]
        bridge_path = _bridge_path(bridge_name, active_render_id)
        _save_last_frame_png(bridge_path, last_frame)
        report = f"LTX2 last-frame bridge: saved {bridge_path.name}"
        log.info("[IAMCCS LTX2] %s", report)
        return (images, report)


class IAMCCS_LTX2_LongVideoWrapperPrep:
    # Carmine Cristallo Scalzi AI reasearch (IAMCCS) - patreon.com/IAMCCS
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_images": ("IMAGE",),
                "song_duration_s": ("FLOAT", {"default": 180.0, "min": 0.01, "max": 36000.0, "step": 0.01}),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.001, "max": 240.0, "step": 0.01}),
                "segment_duration_s": ("FLOAT", {"default": 10.0, "min": 0.01, "max": 3600.0, "step": 0.01}),
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
            },
            "optional": {
                "render_id": ("STRING", {"default": ""}),
                "bridge_name": ("STRING", {"default": "ltx2_detailer_bridge"}),
                "use_bridge_anchor": ("BOOLEAN", {"default": False}),
                "planning_mode": (["manual_segment_seconds", "auto_profile"], {"default": "manual_segment_seconds"}),
                "content_profile": (["videoclip", "monologue"], {"default": "videoclip"}),
                "overlap_frames": ("INT", {"default": 9, "min": 0, "max": 4096, "step": 1}),
                "ltx_round_mode": (["up", "nearest", "down"], {"default": "up"}),
                "head_k_frames": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "head_mode": (["hard_lock", "linear_blend", "ramp"], {"default": "hard_lock"}),
                "head_blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "min_frames": ("INT", {"default": 25, "min": 1, "max": 1000000, "step": 1}),
                "min_frames_mode": (["repeat_last", "error"], {"default": "repeat_last"}),
                "min_frames_ltx_fix": (["none", "up", "down", "nearest"], {"default": "up"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING", "STRING", "INT", "STRING", "INT")
    RETURN_NAMES = ("images", "current_segment", "total_segments", "segment_report", "plan_report", "trim_head_frames", "render_id", "continuation_trim_head_frames")
    FUNCTION = "prepare"
    CATEGORY = "IAMCCS/LTX-2"

    @staticmethod
    def _ensure_image_batch(images: torch.Tensor) -> torch.Tensor:
        if images is None or not torch.is_tensor(images) or images.ndim != 4:
            raise ValueError("source_images must be an IMAGE tensor batch with shape [N,H,W,C]")
        return images

    @staticmethod
    def _extract_range(images: torch.Tensor, start_index: int, end_index: int) -> torch.Tensor:
        total = int(images.shape[0])
        start_index = max(0, min(int(start_index), total - 1))
        end_index = max(start_index + 1, min(int(end_index), total))
        return images[start_index:end_index].clone()

    @staticmethod
    def _first_frame(images: torch.Tensor) -> torch.Tensor:
        return images[:1].clone()

    @staticmethod
    def _resize_to(image: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        if int(image.shape[1]) == target_h and int(image.shape[2]) == target_w:
            return image
        import torch.nn.functional as F

        x = image.permute(0, 3, 1, 2)
        x = F.interpolate(x.float(), size=(target_h, target_w), mode="bilinear", align_corners=False)
        return x.permute(0, 2, 3, 1).clamp(0.0, 1.0).to(image.dtype)

    @classmethod
    def _broadcast_ref(cls, ref: torch.Tensor, count: int, target_h: int, target_w: int) -> torch.Tensor:
        ref = cls._resize_to(ref, target_h, target_w)
        if int(ref.shape[0]) == count:
            return ref
        if int(ref.shape[0]) == 1:
            return ref.repeat((count, 1, 1, 1))
        return ref[:count]

    @staticmethod
    def _blend_weights(count: int, mode: str, max_strength: float) -> list[float]:
        if mode == "hard_lock":
            return [1.0] * count
        if mode == "linear_blend":
            return [max_strength] * count
        if count == 1:
            return [max_strength]
        return [max_strength * float(index + 1) / float(count) for index in range(count)]

    @classmethod
    def _apply_head_anchor(cls, images: torch.Tensor, first_frame: torch.Tensor, k_frames: int, mode: str, blend_strength: float) -> torch.Tensor:
        if first_frame is None:
            return images
        total = int(images.shape[0])
        if total <= 0:
            return images
        k = max(1, min(int(k_frames), total))
        weights = cls._blend_weights(k, str(mode or "hard_lock"), float(max(0.0, min(1.0, blend_strength))))
        out = images.clone()
        ref = cls._broadcast_ref(first_frame, k, int(out.shape[1]), int(out.shape[2]))
        for idx in range(k):
            strength = weights[idx]
            out[idx] = ((1.0 - strength) * out[idx].float() + strength * ref[idx].float()).clamp(0.0, 1.0).to(out.dtype)
        return out

    @staticmethod
    def _fix_ltx_frames(frames: int, mode: str) -> int:
        frames = max(1, int(frames))
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

    @classmethod
    def _ensure_min_frames(cls, images: torch.Tensor, min_frames: int, mode: str, ltx_fix: str) -> tuple[torch.Tensor, int, str]:
        frames_in = int(images.shape[0])
        target_frames = max(frames_in, max(1, int(min_frames)))
        if str(ltx_fix or "none") != "none":
            target_frames = cls._fix_ltx_frames(target_frames, str(ltx_fix))
        if frames_in >= target_frames:
            return images, frames_in, f"EnsureMinFrames: ok ({frames_in}) | min={min_frames} | ltx_fix={ltx_fix}"
        if str(mode or "repeat_last") == "error":
            raise ValueError(f"EnsureMinFrames: got {frames_in} frames, required at least {target_frames}")
        pad = target_frames - frames_in
        out = torch.cat([images, images[-1:, ...].repeat((pad, 1, 1, 1))], dim=0)
        return out, target_frames, f"EnsureMinFrames: repeat_last {frames_in} -> {target_frames} | min={min_frames} | ltx_fix={ltx_fix}"

    def prepare(
        self,
        source_images,
        song_duration_s,
        fps,
        segment_duration_s,
        segment_index,
        render_id="",
        bridge_name="ltx2_detailer_bridge",
        use_bridge_anchor=False,
        planning_mode="manual_segment_seconds",
        content_profile="videoclip",
        overlap_frames=0,
        ltx_round_mode="up",
        head_k_frames=1,
        head_mode="hard_lock",
        head_blend_strength=1.0,
        min_frames=25,
        min_frames_mode="repeat_last",
        min_frames_ltx_fix="up",
    ):
        source_images = self._ensure_image_batch(source_images)

        from .iamccs_ltx2_tools import IAMCCS_SegmentPlanner, IAMCCS_SourceRangeFromSegmentPlan

        plan = IAMCCS_SegmentPlanner().plan(
            song_duration_s=float(song_duration_s),
            fps=float(fps),
            segment_duration_s=float(segment_duration_s),
            planning_mode=str(planning_mode),
            content_profile=str(content_profile),
            overlap_frames=int(overlap_frames),
            ltx_round_mode=str(ltx_round_mode),
            segment_index=int(segment_index),
        )

        total_segments = int(plan[4])
        continuation_raw_frames = int(plan[3])
        current_segment = int(plan[8])
        current_segment_raw_frames = int(plan[9])
        current_segment_unique_frames = int(plan[10])
        current_segment_start_frames = int(plan[11])
        trim_head_frames = max(0, current_segment_raw_frames - current_segment_unique_frames)
        continuation_trim_head_frames = max(0, continuation_raw_frames - current_segment_unique_frames)
        plan_report = str(plan[7])
        plan_segment_report = str(plan[16])

        range_info = IAMCCS_SourceRangeFromSegmentPlan().derive(
            segment_index=current_segment,
            current_segment_raw_frames=current_segment_raw_frames,
            current_segment_unique_frames=current_segment_unique_frames,
            current_segment_start_frames=current_segment_start_frames,
        )
        range_start_index = int(range_info[0])
        range_end_index = int(range_info[1])

        segment_images = self._extract_range(source_images, range_start_index, range_end_index)
        fallback_image = self._first_frame(segment_images)
        bridge_report = "bridge=disabled"
        active_render_id = str(render_id or "").strip() or uuid.uuid4().hex[:10]
        use_bridge_anchor = bool(use_bridge_anchor)
        if use_bridge_anchor and current_segment > 0 and active_render_id:
            bridge_path = _bridge_path(str(bridge_name), active_render_id)
            if bridge_path.exists():
                fallback_image = _load_png(bridge_path)
                bridge_report = f"bridge=loaded:{bridge_path.name}"
            else:
                bridge_report = f"bridge=missing:{bridge_path.name},fallback"
        elif current_segment <= 0:
            bridge_report = "bridge=initial_segment"

        anchored_images = segment_images
        if use_bridge_anchor:
            anchored_images = self._apply_head_anchor(
                segment_images,
                fallback_image,
                k_frames=int(head_k_frames),
                mode=str(head_mode),
                blend_strength=float(head_blend_strength),
            )
        final_images, final_frames, min_report = self._ensure_min_frames(
            anchored_images,
            min_frames=int(min_frames),
            mode=str(min_frames_mode),
            ltx_fix=str(min_frames_ltx_fix),
        )

        segment_report = (
            f"{plan_segment_report} | range=[{range_start_index}..{range_end_index}) | "
            f"bridge={bridge_report} | trim_head={trim_head_frames}f | prepared_frames={final_frames} | {min_report}"
        )
        return (
            final_images,
            current_segment,
            total_segments,
            segment_report,
            plan_report,
            int(trim_head_frames),
            active_render_id,
            int(continuation_trim_head_frames),
        )


class IAMCCS_LTX2_LongVideoWrapperPrepDisk(IAMCCS_LTX2_LongVideoWrapperPrep):
    # Carmine Cristallo Scalzi AI reasearch (IAMCCS) - patreon.com/IAMCCS
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames_dir": ("STRING", {"default": "iamccs_source_frames/source_video"}),
                "song_duration_s": ("FLOAT", {"default": 180.0, "min": 0.01, "max": 36000.0, "step": 0.01}),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.001, "max": 240.0, "step": 0.01}),
                "segment_duration_s": ("FLOAT", {"default": 10.0, "min": 0.01, "max": 3600.0, "step": 0.01}),
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
            },
            "optional": {
                "render_id": ("STRING", {"default": ""}),
                "bridge_name": ("STRING", {"default": "ltx2_detailer_bridge"}),
                "use_bridge_anchor": ("BOOLEAN", {"default": False}),
                "planning_mode": (["manual_segment_seconds", "auto_profile"], {"default": "manual_segment_seconds"}),
                "content_profile": (["videoclip", "monologue"], {"default": "videoclip"}),
                "overlap_frames": ("INT", {"default": 9, "min": 0, "max": 4096, "step": 1}),
                "ltx_round_mode": (["up", "nearest", "down"], {"default": "up"}),
                "head_k_frames": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "head_mode": (["hard_lock", "linear_blend", "ramp"], {"default": "hard_lock"}),
                "head_blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "min_frames": ("INT", {"default": 25, "min": 1, "max": 1000000, "step": 1}),
                "min_frames_mode": (["repeat_last", "error"], {"default": "repeat_last"}),
                "min_frames_ltx_fix": (["none", "up", "down", "nearest"], {"default": "up"}),
            },
        }

    FUNCTION = "prepare_disk"

    def prepare_disk(
        self,
        frames_dir,
        song_duration_s,
        fps,
        segment_duration_s,
        segment_index,
        render_id="",
        bridge_name="ltx2_detailer_bridge",
        use_bridge_anchor=False,
        planning_mode="manual_segment_seconds",
        content_profile="videoclip",
        overlap_frames=0,
        ltx_round_mode="up",
        head_k_frames=1,
        head_mode="hard_lock",
        head_blend_strength=1.0,
        min_frames=25,
        min_frames_mode="repeat_last",
        min_frames_ltx_fix="up",
    ):
        from .iamccs_ltx2_extension_module import IAMCCS_LoadImagesFromDirLite
        from .iamccs_ltx2_tools import IAMCCS_SegmentPlanner, IAMCCS_SourceRangeFromSegmentPlan

        plan = IAMCCS_SegmentPlanner().plan(
            song_duration_s=float(song_duration_s),
            fps=float(fps),
            segment_duration_s=float(segment_duration_s),
            planning_mode=str(planning_mode),
            content_profile=str(content_profile),
            overlap_frames=int(overlap_frames),
            ltx_round_mode=str(ltx_round_mode),
            segment_index=int(segment_index),
        )

        total_segments = int(plan[4])
        continuation_raw_frames = int(plan[3])
        current_segment = int(plan[8])
        current_segment_raw_frames = int(plan[9])
        current_segment_unique_frames = int(plan[10])
        current_segment_start_frames = int(plan[11])
        trim_head_frames = max(0, current_segment_raw_frames - current_segment_unique_frames)
        continuation_trim_head_frames = max(0, continuation_raw_frames - current_segment_unique_frames)
        plan_report = str(plan[7])
        plan_segment_report = str(plan[16])

        range_info = IAMCCS_SourceRangeFromSegmentPlan().derive(
            segment_index=current_segment,
            current_segment_raw_frames=current_segment_raw_frames,
            current_segment_unique_frames=current_segment_unique_frames,
            current_segment_start_frames=current_segment_start_frames,
        )
        range_start_index = int(range_info[0])
        range_end_index = int(range_info[1])

        load_result = IAMCCS_LoadImagesFromDirLite().load(
            directory=str(frames_dir),
            mode="range",
            count=int(current_segment_raw_frames),
            start_index=int(range_start_index),
            end_index=int(range_end_index),
        )
        segment_images = self._ensure_image_batch(load_result[0])
        load_report = str(load_result[2])
        fallback_image = self._first_frame(segment_images)
        bridge_report = "bridge=disabled"
        active_render_id = str(render_id or "").strip() or uuid.uuid4().hex[:10]
        use_bridge_anchor = bool(use_bridge_anchor)
        if use_bridge_anchor and current_segment > 0 and active_render_id:
            bridge_path = _bridge_path(str(bridge_name), active_render_id)
            if bridge_path.exists():
                fallback_image = _load_png(bridge_path)
                bridge_report = f"bridge=loaded:{bridge_path.name}"
            else:
                bridge_report = f"bridge=missing:{bridge_path.name},fallback"
        elif current_segment <= 0:
            bridge_report = "bridge=initial_segment"

        anchored_images = segment_images
        if use_bridge_anchor:
            anchored_images = self._apply_head_anchor(
                segment_images,
                fallback_image,
                k_frames=int(head_k_frames),
                mode=str(head_mode),
                blend_strength=float(head_blend_strength),
            )
        final_images, final_frames, min_report = self._ensure_min_frames(
            anchored_images,
            min_frames=int(min_frames),
            mode=str(min_frames_mode),
            ltx_fix=str(min_frames_ltx_fix),
        )

        segment_report = (
            f"{plan_segment_report} | range=[{range_start_index}..{range_end_index}) | "
            f"load={load_report} | bridge={bridge_report} | trim_head={trim_head_frames}f | "
            f"prepared_frames={final_frames} | {min_report}"
        )
        return (
            final_images,
            current_segment,
            total_segments,
            segment_report,
            plan_report,
            int(trim_head_frames),
            active_render_id,
            int(continuation_trim_head_frames),
        )


class IAMCCS_LTX2_SegmentQueueLoop:
    # Carmine Cristallo Scalzi AI reasearch (IAMCCS) - patreon.com/IAMCCS
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "current_segment": ("INT", {"forceInput": True}),
                "total_segments": ("INT", {"forceInput": True}),
            },
            "optional": {
                "video": ("VIDEO",),
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 0.001, "max": 240.0, "step": 0.01}),
                "bridge_images": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "IAMCCS/LTX2_segment"}),
                "merge_segments": ("BOOLEAN", {"default": True}),
                "keep_segments": ("BOOLEAN", {"default": True}),
                "render_id": ("STRING", {"default": ""}),
                "segment_base_name": ("STRING", {"default": ""}),
                "save_last_frame_bridge": ("BOOLEAN", {"default": True}),
                "bridge_name": ("STRING", {"default": "ltx2_detailer_bridge"}),
                "trim_head_frames": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
                "trim_head_frames_on_continuation": ("INT", {"default": 1, "min": 0, "max": 64, "step": 1}),
                "source_frame_rate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "queue_next"
    OUTPUT_NODE = True
    CATEGORY = "IAMCCS/LTX-2"

    def queue_next(
        self,
        current_segment,
        total_segments,
        video=None,
        images=None,
        audio=None,
        frame_rate=24.0,
        bridge_images=None,
        enabled=True,
        filename_prefix="IAMCCS/LTX2_segment",
        merge_segments=True,
        keep_segments=True,
        render_id="",
        segment_base_name="",
        save_last_frame_bridge=True,
        bridge_name="ltx2_detailer_bridge",
        trim_head_frames=0,
        trim_head_frames_on_continuation=1,
        source_frame_rate=0.0,
        prompt=None,
        unique_id=None,
        extra_pnginfo=None,
    ):
        current_segment = int(current_segment)
        total_segments = int(total_segments)
        enabled = bool(enabled)
        merge_segments = bool(merge_segments)
        keep_segments = bool(keep_segments)
        save_last_frame_bridge = bool(save_last_frame_bridge)
        trim_head_frames = max(0, int(trim_head_frames))
        trim_head_frames_on_continuation = max(0, int(trim_head_frames_on_continuation))
        source_frame_rate = max(0.0, float(source_frame_rate))

        if not render_id and not segment_base_name and current_segment > 0:
            raise RuntimeError(
                f"Fresh run is starting from segment {current_segment + 1} instead of segment 1. "
                "Reset segment_index to 0 or reload the workflow before running again."
            )

        active_render_id = str(render_id or "").strip() or uuid.uuid4().hex[:10]
        if video is not None:
            output_folder, resolved_base_name, _subfolder = _resolve_output_location(filename_prefix, video)
        else:
            try:
                import folder_paths  # type: ignore[import]

                output_folder = folder_paths.get_output_directory()
            except Exception:
                output_folder = os.getcwd()
            prefix_parts = str(filename_prefix or "IAMCCS/LTX2_segment").replace("\\", "/").split("/")
            subfolder_parts = prefix_parts[:-1]
            resolved_base_name = prefix_parts[-1] if prefix_parts[-1] else "LTX2_segment"
            if subfolder_parts:
                output_folder = os.path.join(output_folder, *subfolder_parts)
            os.makedirs(output_folder, exist_ok=True)
        active_base_name = str(segment_base_name or "").strip() or resolved_base_name
        segment_name = _segment_filename(active_base_name, active_render_id, current_segment)
        segment_path = os.path.join(output_folder, segment_name)

        effective_prompt = prompt
        effective_extra_pnginfo = extra_pnginfo
        try:
            live_prompt, live_extra_data, _, _ = _get_current_queue_item()
            if live_prompt is not None:
                effective_prompt = live_prompt
            if effective_extra_pnginfo is None:
                effective_extra_pnginfo = live_extra_data.get("extra_pnginfo", None)
        except Exception:
            pass

        images_to_save = images
        audio_to_save = audio
        effective_trim_head_frames = trim_head_frames if trim_head_frames > 0 else trim_head_frames_on_continuation
        if current_segment > 0 and effective_trim_head_frames > 0 and images is not None:
            scaled_trim_head_frames = effective_trim_head_frames
            if source_frame_rate > 0.0 and frame_rate > 0.0:
                scaled_trim_head_frames = max(
                    0,
                    int(round(float(effective_trim_head_frames) * float(frame_rate) / float(source_frame_rate))),
                )
            available_frames = int(images.shape[0])
            trim_frames = min(scaled_trim_head_frames, max(0, available_frames - 1))
            if trim_frames > 0:
                images_to_save = images[trim_frames:, ...]
                audio_to_save = _trim_audio_leading(audio, trim_frames, frame_rate)
                log.info(
                    "[IAMCCS LTX2] Trimmed %s leading frame(s) from continuation segment %s/%s before save (base=%s, output_fps=%.3f, source_fps=%.3f)",
                    trim_frames,
                    current_segment + 1,
                    total_segments,
                    effective_trim_head_frames,
                    float(frame_rate),
                    float(source_frame_rate),
                )

        metadata = _build_metadata(effective_prompt, effective_extra_pnginfo)
        if video is not None:
            video.save_to(segment_path, format=Types.VideoContainer("auto"), codec="auto", metadata=metadata)
        else:
            _save_segment_from_images(images=images_to_save, audio=audio_to_save, frame_rate=frame_rate, output_path=segment_path)
        log.info("[IAMCCS LTX2] Saved segment %s/%s to %s", current_segment + 1, total_segments, segment_path)

        bridge_images = bridge_images if bridge_images is not None else images
        if save_last_frame_bridge and bridge_images is not None and int(bridge_images.shape[0]) > 0:
            bridge_path = _bridge_path(bridge_name, active_render_id)
            _save_last_frame_png(bridge_path, bridge_images[-1])
            log.info("[IAMCCS LTX2] Saved last-frame bridge %s", bridge_path)

        if not enabled:
            return {"ui": {"text": [f"Saved segment {current_segment + 1}/{total_segments}: {segment_name}"]}}

        next_segment = current_segment + 1
        if next_segment >= total_segments:
            ui_text = [f"Saved segment {current_segment + 1}/{total_segments}: {segment_name}"]
            if merge_segments:
                segment_paths = [
                    os.path.join(output_folder, _segment_filename(active_base_name, active_render_id, index))
                    for index in range(total_segments)
                ]
                final_name = _final_filename(active_base_name, active_render_id)
                final_path = os.path.join(output_folder, final_name)
                _concat_segments(segment_paths, final_path)
                ui_text.append(f"Merged final video: {final_name}")
                if not keep_segments:
                    for path in segment_paths:
                        if os.path.exists(path):
                            os.remove(path)
            return {"ui": {"text": ui_text}}

        base_prompt = effective_prompt
        if base_prompt is None:
            live_prompt, _, _, _ = _get_current_queue_item()
            base_prompt = live_prompt
        prompt_copy = copy.deepcopy(_normalize_prompt_keys(base_prompt))

        loop_updated = False
        updated_segment_nodes = 0
        for node_id, node in prompt_copy.items():
            inputs = node.setdefault("inputs", {})
            if "segment_index" in inputs:
                inputs["segment_index"] = next_segment
                updated_segment_nodes += 1
            if "render_id" in inputs:
                inputs["render_id"] = active_render_id
            if node.get("class_type") == "IAMCCS_LTX2_LongVideoWrapperPrep":
                inputs["render_id"] = active_render_id
            is_current_loop = unique_id is not None and node_id == str(unique_id)
            is_loop_fallback = unique_id is None and node.get("class_type") == "IAMCCS_LTX2_SegmentQueueLoop"
            if is_current_loop or is_loop_fallback:
                inputs["render_id"] = active_render_id
                inputs["segment_base_name"] = active_base_name
                loop_updated = True

        if updated_segment_nodes == 0:
            raise ValueError("IAMCCS_LTX2_SegmentQueueLoop could not find any node with a segment_index input.")
        if not loop_updated:
            raise ValueError("IAMCCS_LTX2_SegmentQueueLoop could not update its own render_id in the prompt.")

        _enqueue_prompt(prompt_copy)
        return {
            "ui": {
                "text": [
                    f"Saved segment {current_segment + 1}/{total_segments}: {segment_name}",
                    f"Queued segment {next_segment + 1}/{total_segments}",
                ]
            }
        }


class IAMCCS_LTX2_LoadLatentBridge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latents": ("LATENT",),
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "render_id": ("STRING", {"default": ""}),
                "temporal_overlap": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "temporal_overlap_cond_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("conditioned_latents", "report")
    FUNCTION = "condition"
    CATEGORY = "IAMCCS/LTX-2"

    def condition(self, latents, segment_index, render_id, temporal_overlap, temporal_overlap_cond_strength, vae=None):
        segment_index = int(segment_index)
        overlap_px = max(0, int(temporal_overlap))
        active_render_id = str(render_id or "").strip()
        if segment_index <= 0 or not active_render_id or overlap_px <= 0:
            return (latents, "latent_bridge=initial_or_disabled")

        bridge_path = _resolve_latent_bridge_payload_path(active_render_id)
        if not bridge_path.exists():
            return (latents, f"latent_bridge=missing:{bridge_path.name}")

        time_scale = _get_time_scale_factor_from_vae(vae) if vae is not None else 8
        overlap_f = _pixel_frames_to_latent_frames(overlap_px, time_scale)
        if overlap_f <= 0:
            return (latents, "latent_bridge=overlap_zero")

        out = {k: v for k, v in latents.items()}
        samples = out["samples"].clone()
        noise_mask = out.get("noise_mask")
        if noise_mask is None or not torch.is_tensor(noise_mask):
            noise_mask = torch.ones(
                (samples.shape[0], 1, samples.shape[2], 1, 1),
                device=samples.device,
                dtype=torch.float32,
            )
        else:
            noise_mask = noise_mask.clone()

        bridge_payload = _load_latent_bridge(bridge_path)
        prev_tail = bridge_payload["latent_tail"]
        if prev_tail is None:
            return (latents, "latent_bridge=empty")
        prev_tail = prev_tail.to(device=samples.device, dtype=samples.dtype)
        saved_overlap = bridge_payload.get("latent_overlap_frames")
        if torch.is_tensor(saved_overlap) and int(saved_overlap.numel()) > 0:
            overlap_f = int(saved_overlap.flatten()[0].item())
        overlap_f = min(int(overlap_f), int(prev_tail.shape[2]), int(samples.shape[2]))
        if overlap_f <= 0:
            return (latents, "latent_bridge=empty")

        samples[:, :, :overlap_f, :, :] = prev_tail[:, :, -overlap_f:, :, :]
        noise_mask[:, :, :overlap_f, :, :] = 1.0 - float(temporal_overlap_cond_strength)
        out["samples"] = samples
        out["noise_mask"] = noise_mask
        prev_full = bridge_payload.get("latent_full")
        if torch.is_tensor(prev_full) and prev_full.ndim == 5:
            out["iamccs_prev_latents"] = {"samples": prev_full.to(device=samples.device, dtype=samples.dtype)}
        reference_latent = bridge_payload.get("latent_reference")
        if torch.is_tensor(reference_latent) and reference_latent.ndim == 5:
            out["iamccs_reference_latents"] = {"samples": reference_latent.to(device=samples.device, dtype=samples.dtype)}
        seed_offset = bridge_payload.get("latent_seed_offset")
        if torch.is_tensor(seed_offset) and int(seed_offset.numel()) > 0:
            out["iamccs_seed_offset"] = int(seed_offset.flatten()[0].item())
        report = f"latent_bridge=loaded:{bridge_path.name} overlap_px={overlap_px} overlap_lat={overlap_f}"
        log.info("[IAMCCS LTX2] %s", report)
        return (out, report)


class IAMCCS_LTX2_SaveLatentBridge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latents": ("LATENT",),
                "render_id": ("STRING", {"default": ""}),
                "temporal_overlap": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            },
            "optional": {
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latents", "report")
    FUNCTION = "save"
    CATEGORY = "IAMCCS/LTX-2"

    def save(self, latents, render_id, temporal_overlap, vae=None):
        active_render_id = str(render_id or "").strip()
        overlap_px = max(0, int(temporal_overlap))
        if not active_render_id or overlap_px <= 0:
            return (latents, "latent_bridge=save_disabled")

        samples = latents.get("samples")
        if not torch.is_tensor(samples) or samples.ndim != 5:
            return (latents, "latent_bridge=invalid_latents")

        time_scale = _get_time_scale_factor_from_vae(vae) if vae is not None else 8
        overlap_f = _pixel_frames_to_latent_frames(overlap_px, time_scale)
        overlap_f = min(max(0, int(overlap_f)), int(samples.shape[2]))
        if overlap_f <= 0:
            return (latents, "latent_bridge=save_overlap_zero")

        tail = samples[:, :, -overlap_f:, :, :]
        bridge_path = _resolve_latent_bridge_payload_path(active_render_id)
        previous_seed_offset = 0
        reference_latent = samples
        if bridge_path.exists():
            try:
                existing_payload = _load_latent_bridge(bridge_path)
                existing_seed_offset = existing_payload.get("latent_seed_offset")
                if torch.is_tensor(existing_seed_offset) and int(existing_seed_offset.numel()) > 0:
                    previous_seed_offset = int(existing_seed_offset.flatten()[0].item())
                existing_reference = existing_payload.get("latent_reference")
                if torch.is_tensor(existing_reference) and existing_reference.ndim == 5:
                    reference_latent = existing_reference.to(dtype=samples.dtype)
            except Exception:
                previous_seed_offset = 0
                reference_latent = samples
        next_seed_offset = previous_seed_offset + max(1, int(samples.shape[2]) - overlap_f)
        _save_latent_bridge(
            active_render_id,
            tail,
            samples,
            reference_latent,
            next_seed_offset,
            overlap_f,
        )
        report = f"latent_bridge=saved:{bridge_path.name} overlap_px={overlap_px} overlap_lat={overlap_f}"
        log.info("[IAMCCS LTX2] %s", report)
        return (latents, report)


class IAMCCS_LTX2_BlendLatentBridge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latents": ("LATENT",),
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "render_id": ("STRING", {"default": ""}),
                "temporal_overlap": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            },
            "optional": {
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latents", "report")
    FUNCTION = "blend"
    CATEGORY = "IAMCCS/LTX-2"

    def blend(self, latents, segment_index, render_id, temporal_overlap, vae=None):
        if latents.get("iamccs_bridge_transition_applied"):
            return (latents, "latent_bridge_blend=handled_by_extend_sampler")

        segment_index = int(segment_index)
        overlap_px = max(0, int(temporal_overlap))
        active_render_id = str(render_id or "").strip()
        if segment_index <= 0 or not active_render_id or overlap_px <= 0:
            return (latents, "latent_bridge_blend=initial_or_disabled")

        bridge_path = _resolve_latent_bridge_payload_path(active_render_id)
        if not bridge_path.exists():
            return (latents, f"latent_bridge_blend=missing:{bridge_path.name}")

        samples = latents.get("samples")
        if not torch.is_tensor(samples) or samples.ndim != 5:
            return (latents, "latent_bridge_blend=invalid_latents")

        time_scale = _get_time_scale_factor_from_vae(vae) if vae is not None else 8
        overlap_f = _pixel_frames_to_latent_frames(overlap_px, time_scale)
        bridge_payload = _load_latent_bridge(bridge_path)
        prev_tail = bridge_payload["latent_tail"]
        if prev_tail is None:
            return (latents, "latent_bridge_blend=empty")
        prev_tail = prev_tail.to(device=samples.device, dtype=samples.dtype)
        saved_overlap = bridge_payload.get("latent_overlap_frames")
        if torch.is_tensor(saved_overlap) and int(saved_overlap.numel()) > 0:
            overlap_f = int(saved_overlap.flatten()[0].item())
        overlap_f = min(int(overlap_f), int(prev_tail.shape[2]), int(samples.shape[2]))
        if overlap_f <= 0:
            return (latents, "latent_bridge_blend=empty")

        out = {k: v for k, v in latents.items()}
        blended_samples = samples.clone()
        alpha = torch.linspace(0.0, 1.0, steps=overlap_f, device=samples.device, dtype=samples.dtype).view(1, 1, overlap_f, 1, 1)
        prev = prev_tail[:, :, -overlap_f:, :, :]
        cur = blended_samples[:, :, :overlap_f, :, :]
        blended_samples[:, :, :overlap_f, :, :] = prev * (1.0 - alpha) + cur * alpha
        out["samples"] = blended_samples
        report = f"latent_bridge_blend=applied:{bridge_path.name} overlap_px={overlap_px} overlap_lat={overlap_f}"
        log.info("[IAMCCS LTX2] %s", report)
        return (out, report)


NODE_CLASS_MAPPINGS = {
    "IAMCCS_LTX2_LastFrameBridgeSave": IAMCCS_LTX2_LastFrameBridgeSave,
    "IAMCCS_LTX2_LastFrameBridgeLoad": IAMCCS_LTX2_LastFrameBridgeLoad,
    "IAMCCS_LTX2_BlendLatentBridge": IAMCCS_LTX2_BlendLatentBridge,
    "IAMCCS_LTX2_LongVideoWrapperPrep": IAMCCS_LTX2_LongVideoWrapperPrep,
    "IAMCCS_LTX2_SegmentQueueLoop": IAMCCS_LTX2_SegmentQueueLoop,
    "IAMCCS_LTX2_LoadLatentBridge": IAMCCS_LTX2_LoadLatentBridge,
    "IAMCCS_LTX2_SaveLatentBridge": IAMCCS_LTX2_SaveLatentBridge,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_LTX2_LastFrameBridgeSave": "LTX-2 Last Frame Bridge Save 🖼️💾",
    "IAMCCS_LTX2_LastFrameBridgeLoad": "LTX-2 Last Frame Bridge Load 🖼️",
    "IAMCCS_LTX2_BlendLatentBridge": "LTX-2 Blend Latent Bridge 🎚️",
    "IAMCCS_LTX2_LongVideoWrapperPrep": "LTX-2 Long Video Wrapper Prep 🧰",
    "IAMCCS_LTX2_SegmentQueueLoop": "LTX-2 Segment Queue Loop 🔁",
    "IAMCCS_LTX2_LoadLatentBridge": "LTX-2 Load Latent Bridge 🧬",
    "IAMCCS_LTX2_SaveLatentBridge": "LTX-2 Save Latent Bridge 💾",
}