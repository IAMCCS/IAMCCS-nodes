# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import copy
import gc
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import uuid
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

import folder_paths

from .iamccs_minimax_h3_shotboard_core import H3_FPS, build_shotplan, plan_json
from .iamccs_prompter import SUPERNODE_LINX_TYPE, apply_prompter_to_minimax
from .iamccs_supernodes_linx import build_stage_linx_payload


SHOTPLAN_TYPE = "IAMCCS_MINIMAX_H3_SHOTPLAN"
CATEGORY = "IAMCCS/MiniMax H3"
LOG = logging.getLogger("IAMCCS.MiniMaxH3.Standalone")

H3_DIMENSION_MULTIPLE = 32
H3_MIN_DIMENSION = 256
H3_MAX_DIMENSION = 5760
def _finite_float(value: Any, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        number = float(default)
    if not math.isfinite(number):
        number = float(default)
    if minimum is not None:
        number = max(float(minimum), number)
    if maximum is not None:
        number = min(float(maximum), number)
    return number


def _h3_legal_dimension(value: Any, default: int) -> int:
    """Round upward to H3's 32-pixel grid (720 -> 736), never returning NaN."""
    number = _finite_float(value, float(default), H3_MIN_DIMENSION, H3_MAX_DIMENSION)
    aligned = int(math.ceil(number / H3_DIMENSION_MULTIPLE) * H3_DIMENSION_MULTIPLE)
    return max(H3_MIN_DIMENSION, min(H3_MAX_DIMENSION, aligned))


def _model_file_available(folder_name: str, filename: Any) -> bool:
    name = str(filename or "").strip()
    if not name:
        return False
    try:
        return bool(folder_paths.get_full_path(folder_name, name))
    except Exception:
        return False


def _resolve_shotplan(value: Any) -> dict[str, Any]:
    """Read the private H3 plan from CineLinX, with legacy-plan tolerance."""
    if isinstance(value, dict) and value.get("schema") == "iamccs.minimax_h3.shotplan":
        return value
    if not isinstance(value, dict):
        raise ValueError("cine_linx MiniMax H3 non valido")

    resources = value.get("resources") if isinstance(value.get("resources"), dict) else {}
    outputs = value.get("outputs") if isinstance(value.get("outputs"), dict) else {}
    payload = resources.get("cine_payload") if isinstance(resources.get("cine_payload"), dict) else {}
    candidates = (
        resources.get("iamccs_minimax_h3_shotplan"),
        resources.get("minimax_h3_shotplan"),
        resources.get("shotplan"),
        outputs.get("shotplan"),
        payload.get("minimax_h3_shotplan"),
        payload.get("shotplan"),
    )
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("schema") == "iamccs.minimax_h3.shotplan":
            return candidate
    raise ValueError("cine_linx non contiene un piano IAMCCS MiniMax H3")


def _timeline_dict(timeline_data: Any) -> dict[str, Any]:
    if isinstance(timeline_data, dict):
        return copy.deepcopy(timeline_data)
    try:
        parsed = json.loads(str(timeline_data or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _build_minimax_cine_linx(
    upstream_linx: Any,
    plan: dict[str, Any],
    timeline_data: Any,
    global_prompt: str,
    report: str,
) -> dict[str, Any]:
    """Pack every former planner output into one standard CineLinX cable."""
    plan_text = plan_json(plan)
    prompt_map_text = json.dumps(plan.get("prompt_map", []), ensure_ascii=False, indent=2)
    timeline = _timeline_dict(timeline_data)
    slots = plan.get("slots") if isinstance(plan.get("slots"), list) else []
    chunks = plan.get("chunks") if isinstance(plan.get("chunks"), list) else []
    audio_segments = timeline.get("audioSegments")
    if not isinstance(audio_segments, list):
        audio_segments = timeline.get("audio_segments")
    if not isinstance(audio_segments, list):
        audio_segments = []
    local_prompts = " | ".join(
        str(slot.get("prompt", "")).strip()
        for slot in slots
        if isinstance(slot, dict) and str(slot.get("prompt", "")).strip()
    )
    segment_lengths = ",".join(
        str(int(chunk.get("frame_count", 0) or 0))
        for chunk in chunks
        if isinstance(chunk, dict)
    )
    effective_duration = float(plan.get("effective_duration_seconds", 0.0) or 0.0)
    total_segments = int(plan.get("total_segments", len(chunks)) or 0)

    payload = {
        "backend_mode": "minimax_h3_shotboard",
        "pipeline_kind": "minimax_h3",
        "schema": "iamccs.minimax_h3.cine_linx",
        "schema_version": 1,
        "shotplan_schema": str(plan.get("schema", "")),
        "global_prompt": str(global_prompt or ""),
        "local_prompts": local_prompts,
        "segment_lengths": segment_lengths,
        "duration_seconds": effective_duration,
        "effective_duration_seconds": effective_duration,
        "frame_rate": int(plan.get("fps", H3_FPS) or H3_FPS),
        "width": int(plan.get("width", 0) or 0),
        "height": int(plan.get("height", 0) or 0),
        "timeline_data": timeline,
        "visual_segments": slots,
        "audioSegments": audio_segments,
        "minimax_h3_shotplan": plan,
    }
    outputs = {
        "shotplan": plan,
        "shotplan_json": plan_text,
        "prompt_map_json": prompt_map_text,
        "total_segments": total_segments,
        "effective_duration": effective_duration,
        "report": report,
        "global_prompt": str(global_prompt or ""),
        "local_prompts": local_prompts,
        "segment_lengths": segment_lengths,
        "timeline_data": json.dumps(timeline, ensure_ascii=False),
        "audio_timeline_json": json.dumps({"audioSegments": audio_segments}, ensure_ascii=False),
    }
    resources = {
        "cine_payload": payload,
        "cine_global_prompt": str(global_prompt or ""),
        "cine_local_prompts": local_prompts,
        "cine_segment_lengths": segment_lengths,
        "cine_duration_seconds": effective_duration,
        "cine_frame_rate": int(plan.get("fps", H3_FPS) or H3_FPS),
        "cine_width": int(plan.get("width", 0) or 0),
        "cine_height": int(plan.get("height", 0) or 0),
        "cine_timeline_data_json": json.dumps(timeline, ensure_ascii=False),
        "cine_visual_segments_json": json.dumps(slots, ensure_ascii=False),
        "cine_audio_timeline_json": json.dumps({"audioSegments": audio_segments}, ensure_ascii=False),
        "iamccs_minimax_h3_shotplan": plan,
        "iamccs_minimax_h3_shotplan_json": plan_text,
        "iamccs_minimax_h3_prompt_map": plan.get("prompt_map", []),
        "iamccs_minimax_h3_prompt_map_json": prompt_map_text,
        "iamccs_minimax_h3_total_segments": total_segments,
        "iamccs_minimax_h3_effective_duration": effective_duration,
        "cine_report": report,
    }
    cine_linx = build_stage_linx_payload(
        upstream_linx,
        stage_name="MiniMax H3 Shotboard",
        stage_kind="minimax_h3_shot_planner",
        payload=payload,
        report=report,
        outputs=outputs,
        resources=resources,
        policies={
            "minimax_h3_chunk_source": "timeline_box_trim",
            "minimax_h3_prompt_source": str(plan.get("prompt_mapping", "global_plus_local")),
            "minimax_h3_transport": "cine_linx",
        },
        downstream_stages=("MiniMax H3 backend", "IAMCCS_CineInfo", "IAMCCS Audioboard"),
    )
    cine_linx["mode"] = "minimax_h3_shotboard"
    return cine_linx


def _node_class(name: str):
    import nodes as comfy_nodes

    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        raise RuntimeError(f"Nodo richiesto non disponibile: {name}. Riavvia ComfyUI e verifica i custom nodes.")
    return cls


def _release_cpu_text_encoder_memory(clip) -> str:
    """Unload CLIP and evict its recently touched GGUF pages before H3 sampling."""
    try:
        import comfy.model_management as model_management

        patcher = getattr(clip, "patcher", None)
        if patcher is not None:
            model_management.unload_model_and_clones(patcher, all_devices=True)
        model_management.soft_empty_cache()
    except Exception as exc:
        LOG.warning("MiniMax H3 text-encoder unload warning: %s", exc)
    gc.collect()
    if os.name == "nt":
        try:
            import ctypes

            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.psapi.EmptyWorkingSet(handle)
            return "cpu text encoder unloaded; Windows working set trimmed"
        except Exception as exc:
            LOG.warning("MiniMax H3 Windows working-set trim warning: %s", exc)
    return "cpu text encoder unloaded"


def _filename_list(*keys: str, suffix: str | None = None) -> list[str]:
    values: set[str] = set()
    for key in keys:
        try:
            values.update(folder_paths.get_filename_list(key))
        except Exception:
            continue
    clean = [value for value in values if not str(value).lower().endswith(".part")]
    if suffix:
        clean = [value for value in clean if str(value).lower().endswith(suffix.lower())]
    return sorted(clean, key=lambda value: ("minimax" not in value.lower() and "h3" not in value.lower(), value.lower()))


def _options(values: list[str], fallback: str) -> list[str]:
    return values or [fallback]


def _prefer(values: list[str], keyword: str) -> list[str]:
    return sorted(values, key=lambda value: (keyword not in value.lower(), value.lower()))


def _resolve_image_path(value: str) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    direct = Path(os.path.expandvars(os.path.expanduser(raw)))
    if direct.is_file():
        return direct
    try:
        annotated = folder_paths.get_annotated_filepath(raw)
        if annotated and os.path.isfile(annotated):
            return Path(annotated)
    except Exception:
        pass
    try:
        candidate = Path(folder_paths.get_input_directory()) / raw
        if candidate.is_file():
            return candidate
    except Exception:
        pass
    raise FileNotFoundError(f"Immagine Shotplanner MiniMax non trovata: {raw}")


def _load_image(value: str) -> torch.Tensor | None:
    path = _resolve_image_path(value)
    if path is None:
        return None
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _black_image(width: int = 64, height: int = 64) -> torch.Tensor:
    return torch.zeros((1, max(1, int(height)), max(1, int(width)), 3), dtype=torch.float32)


def _audio_silence(duration_seconds: float, sample_rate: int = 32000) -> dict[str, Any]:
    samples = max(1, int(round(max(0.01, duration_seconds) * sample_rate)))
    return {"waveform": torch.zeros((1, 2, samples), dtype=torch.float32), "sample_rate": sample_rate}


def _slice_audio(audio: dict[str, Any] | None, start_seconds: float, duration_seconds: float) -> dict[str, Any]:
    if not isinstance(audio, dict) or not torch.is_tensor(audio.get("waveform")):
        return _audio_silence(duration_seconds)
    waveform = audio["waveform"]
    sample_rate = int(audio.get("sample_rate", 32000))
    if waveform.ndim != 3:
        raise ValueError("AUDIO deve avere waveform [B,C,S]")
    start = max(0, int(round(float(start_seconds) * sample_rate)))
    length = max(1, int(round(float(duration_seconds) * sample_rate)))
    end = start + length
    sliced = waveform[:1, :, start:min(end, waveform.shape[-1])]
    if sliced.shape[-1] < length:
        sliced = torch.nn.functional.pad(sliced, (0, length - sliced.shape[-1]))
    return {"waveform": sliced, "sample_rate": sample_rate}


def _normalise_channels(waveform: torch.Tensor, channels: int = 2) -> torch.Tensor:
    waveform = waveform[:1]
    current = waveform.shape[1]
    if current == channels:
        return waveform
    if current == 1 and channels == 2:
        return waveform.repeat(1, 2, 1)
    if current > channels:
        return waveform[:, :channels, :]
    repeats = math.ceil(channels / current)
    return waveform.repeat(1, repeats, 1)[:, :channels, :]


def _resample(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    if source_rate == target_rate:
        return waveform
    import torchaudio

    return torchaudio.functional.resample(waveform, source_rate, target_rate)


def _safe_name(value: str, fallback: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()).strip("._")
    return clean or fallback


def _bridge_directory() -> Path:
    path = Path(folder_paths.get_output_directory()) / "minimax_h3_shotboard" / "bridges"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _bridge_path(render_id: str) -> Path:
    return _bridge_directory() / f"last_frame_{_safe_name(render_id, 'default')}.png"


def _save_frame(path: Path, frame: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (frame.detach().cpu().float().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def _load_frame(path: Path) -> torch.Tensor:
    with Image.open(path) as source:
        array = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _find_ffmpeg() -> str | None:
    forced = os.environ.get("VHS_FORCE_FFMPEG_PATH")
    if forced and os.path.isfile(forced):
        return forced
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        value = get_ffmpeg_exe()
        if value and os.path.isfile(value):
            return value
    except Exception:
        pass
    return shutil.which("ffmpeg")


def _write_wav(audio: dict[str, Any] | None, output: Path) -> bool:
    if not isinstance(audio, dict) or not torch.is_tensor(audio.get("waveform")):
        return False
    waveform = audio["waveform"][:1].detach().cpu().float()
    if waveform.ndim != 3:
        return False
    pcm = np.clip(np.nan_to_num(waveform.squeeze(0).numpy().T), -1.0, 1.0)
    pcm = (pcm * 32767.0).round().astype(np.int16)
    with wave.open(str(output), "wb") as handle:
        handle.setnchannels(int(pcm.shape[1]))
        handle.setsampwidth(2)
        handle.setframerate(int(audio.get("sample_rate", 32000)))
        handle.writeframes(pcm.tobytes())
    return True


def _trim_audio_frames(audio: dict[str, Any] | None, frames: int, fps: float) -> dict[str, Any] | None:
    if not isinstance(audio, dict) or not torch.is_tensor(audio.get("waveform")):
        return audio
    sample_rate = int(audio.get("sample_rate", 32000))
    waveform = audio["waveform"]
    samples = min(max(0, int(round(float(frames) / max(0.001, fps) * sample_rate))), max(0, waveform.shape[-1] - 1))
    return {"waveform": waveform[:, :, samples:], "sample_rate": sample_rate}


def _encode_images(images: torch.Tensor, audio: dict[str, Any] | None, fps: float, output: Path) -> None:
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg non trovato: impossibile salvare i segmenti MiniMax H3")
    if not torch.is_tensor(images) or images.ndim != 4 or images.shape[0] < 1:
        raise ValueError("images deve essere un batch IMAGE [T,H,W,C]")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="minimax_h3_segment_") as temp:
        temp_path = Path(temp)
        for index, frame in enumerate(images):
            _save_frame(temp_path / f"frame_{index:05d}.png", frame)
        wav_path = temp_path / "audio.wav"
        has_audio = _write_wav(audio, wav_path)
        command = [
            ffmpeg, "-nostdin", "-n", "-framerate", f"{float(fps):.6f}",
            "-i", str(temp_path / "frame_%05d.png"),
        ]
        if has_audio:
            command += ["-i", str(wav_path)]
        command += ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]
        if has_audio:
            command += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest"]
        command += ["-movflags", "+faststart", str(output)]
        result = subprocess.run(command, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg segment encode failed: {result.stderr.strip() or result.stdout.strip()}")


def _concat_videos(paths: list[Path], output: Path) -> None:
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg non trovato: impossibile concatenare i segmenti MiniMax H3")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Segmenti mancanti: " + ", ".join(missing))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
        list_path = Path(handle.name)
        for path in paths:
            handle.write("file '" + str(path).replace("'", "'\\''") + "'\n")
    try:
        command = [
            ffmpeg, "-nostdin", "-n", "-f", "concat", "-safe", "0", "-i", str(list_path),
            "-fflags", "+genpts", "-avoid_negative_ts", "make_zero", "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", str(output),
        ]
        result = subprocess.run(command, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {result.stderr.strip() or result.stdout.strip()}")
    finally:
        list_path.unlink(missing_ok=True)


def _output_location(prefix: str) -> tuple[Path, str]:
    parts = [part for part in str(prefix or "MiniMaxH3/segment").replace("\\", "/").split("/") if part and part not in {".", ".."}]
    if not parts:
        parts = ["MiniMaxH3", "segment"]
    base = Path(folder_paths.get_output_directory())
    folder = base.joinpath(*[_safe_name(part, "output") for part in parts[:-1]])
    folder.mkdir(parents=True, exist_ok=True)
    return folder, _safe_name(parts[-1], "segment")


def _next_numbered_render_id(output_folder: Path, base_name: str, requested_render_id: str) -> str:
    """Return a never-reused render id with a monotonically increasing suffix.

    Historical unnumbered outputs count as render 0001, so an installation that
    already contains ``segment_<render>_full.mp4`` continues at 0002.
    """
    root = _safe_name(requested_render_id, "minimax_h3_render")
    escaped_base = re.escape(_safe_name(base_name, "segment"))
    escaped_root = re.escape(root)
    pattern = re.compile(
        rf"^{escaped_base}_{escaped_root}(?:_(\d{{4,}}))?_(?:full|seg_\d{{4,}})\.mp4$",
        re.IGNORECASE,
    )
    used_numbers: set[int] = set()
    try:
        candidates = output_folder.iterdir()
    except FileNotFoundError:
        candidates = ()
    for path in candidates:
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if match is None:
            continue
        suffix = match.group(1)
        used_numbers.add(int(suffix) if suffix is not None else 1)

    next_number = max(used_numbers, default=0) + 1
    while True:
        candidate = f"{root}_{next_number:04d}"
        segment_collision = any(output_folder.glob(f"{_safe_name(base_name, 'segment')}_{candidate}_seg_*.mp4"))
        final_collision = (output_folder / f"{_safe_name(base_name, 'segment')}_{candidate}_full.mp4").exists()
        if not segment_collision and not final_collision:
            return candidate
        next_number += 1


def _require_new_output_path(path: Path) -> None:
    if path.exists():
        raise FileExistsError(
            f"IAMCCS protegge i render esistenti: il file non verrà sovrascritto: {path}"
        )


def _current_prompt():
    import server

    prompt_server = getattr(server.PromptServer, "instance", None)
    running = getattr(getattr(prompt_server, "prompt_queue", None), "currently_running", {})
    if not running:
        raise RuntimeError("Prompt ComfyUI corrente non disponibile")
    current = next(iter(running.values()))
    if len(current) == 6:
        _, _, prompt, extra_data, outputs, sensitive = current
    else:
        _, _, prompt, extra_data, outputs = current
        sensitive = {}
    return prompt, extra_data, outputs, sensitive


def _enqueue(prompt: dict[str, Any], extra_data=None, outputs=None, sensitive=None) -> None:
    import nodes as comfy_nodes
    import server

    prompt_server = server.PromptServer.instance
    if extra_data is None or outputs is None or sensitive is None:
        _, live_extra, live_outputs, live_sensitive = _current_prompt()
        extra_data = live_extra if extra_data is None else extra_data
        outputs = live_outputs if outputs is None else outputs
        sensitive = live_sensitive if sensitive is None else sensitive
    if not outputs:
        outputs = [
            str(node_id) for node_id, node in prompt.items()
            if getattr(comfy_nodes.NODE_CLASS_MAPPINGS.get(node.get("class_type")), "OUTPUT_NODE", False)
        ]
    if not outputs:
        raise RuntimeError("Nessun output node disponibile per rilanciare il segmento successivo")
    number = -prompt_server.number
    prompt_server.number += 1
    prompt_server.prompt_queue.put((number, str(uuid.uuid4()), prompt, extra_data or {}, [str(value) for value in outputs], sensitive or {}))


class IAMCCS_MiniMaxH3GGUFLoader:
    @classmethod
    def INPUT_TYPES(cls):
        unets = _options(_filename_list("unet_gguf", "unet", "diffusion_models", suffix=".gguf"), "NO_H3_GGUF_UNET_FOUND")
        clips = _options(_filename_list("clip_gguf", "clip", "text_encoders", suffix=".gguf"), "NO_H3_GGUF_CLIP_FOUND")
        vaes = _filename_list("vae")
        video_vaes = _options(_prefer(vaes, "video"), "NO_H3_VIDEO_VAE_FOUND")
        audio_vaes = _options(_prefer(vaes, "audio"), "NO_H3_AUDIO_VAE_FOUND")
        return {
            "required": {
                "unet_name": (unets,),
                "clip_name": (clips,),
                "video_vae_name": (video_vaes,),
                "audio_vae_name": (audio_vaes,),
                "model_task": (["fl2va", "ref2va"], {"default": "fl2va"}),
                "acceleration": (["native", "spectrum_conservative", "spectrum_aggressive"], {"default": "spectrum_conservative"}),
                "spectrum_history": (["system_ram", "vram"], {"default": "system_ram"}),
                "spectrum_debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "text_encoder_device": (["cpu_safe_12gb", "auto"], {"default": "cpu_safe_12gb"}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "VAE", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "video_vae", "audio_vae", "model_task", "report")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(
        self,
        unet_name,
        clip_name,
        video_vae_name,
        audio_vae_name,
        model_task,
        acceleration,
        spectrum_history,
        spectrum_debug,
        text_encoder_device="cpu_safe_12gb",
    ):
        if str(unet_name).startswith("NO_") or str(clip_name).startswith("NO_"):
            raise FileNotFoundError("GGUF H3 UNET/CLIP non disponibili. Attendi la fine dei download e riavvia ComfyUI.")
        if str(video_vae_name).startswith("NO_") or str(audio_vae_name).startswith("NO_"):
            raise FileNotFoundError("VAE video/audio MiniMax H3 non disponibili")

        unet_cls = _node_class("UnetLoaderGGUFAdvanced")
        model = unet_cls().load_unet(
            unet_name,
            dequant_dtype="default",
            patch_dtype="default",
            patch_on_device=False,
        )[0]
        clip_cls = _node_class("CLIPLoaderGGUF")
        clip = clip_cls().load_clip(clip_name, type="minimax")[0]
        if str(text_encoder_device) == "cpu_safe_12gb":
            # Qwen3-VL 32B Q2 is ~8 GiB before temporary dequant buffers.
            # Running it on a 12 GiB GPU fails before H3 sampling begins.
            # Use ComfyUI's native device retargeter instead of mutating the
            # GGUF patcher internals so future model-management changes remain
            # compatible.
            from comfy_extras.nodes_multigpu import SelectCLIPDeviceNode

            clip = SelectCLIPDeviceNode.execute(clip=clip, device="cpu")[0]
        vae_cls = _node_class("VAELoader")
        video_vae = vae_cls().load_vae(video_vae_name)[0]
        audio_vae = vae_cls().load_vae(audio_vae_name)[0]

        spectrum_report = "Spectrum OFF"
        if acceleration != "native":
            spectrum_cls = _node_class("SpectrumApplyMiniMaxH3")
            aggressive = acceleration == "spectrum_aggressive"
            model = spectrum_cls().apply(
                model=model,
                enabled=True,
                blend_weight=0.75 if aggressive else 0.50,
                degree=4,
                ridge_lambda=0.10,
                window_size=2.0,
                flex_window=3.0 if aggressive else 0.75,
                warmup_steps=5,
                tail_actual_steps=1,
                max_history=8,
                debug=bool(spectrum_debug),
                history_storage=spectrum_history,
            )[0]
            spectrum_report = f"Spectrum {acceleration} history={spectrum_history}"

        lower_name = str(unet_name).lower()
        warning = ""
        if model_task == "ref2va" and "ref2va" not in lower_name:
            warning = " | ATTENZIONE: task ref2va selezionato ma il filename non contiene ref2va"
        if model_task == "fl2va" and "ref2va" in lower_name:
            warning = " | ATTENZIONE: modello ref2va selezionato per task fl2va"
        report = (
            f"MiniMax H3 GGUF loaded | task={model_task} | unet={unet_name} | clip={clip_name} | "
            f"video_vae={video_vae_name} | audio_vae={audio_vae_name} | "
            f"text_encoder={text_encoder_device} | {spectrum_report}{warning}"
        )
        return model, clip, video_vae, audio_vae, str(model_task), report


class IAMCCS_MiniMaxH3ShotPlanner:
    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers

        samplers = list(comfy.samplers.SAMPLER_NAMES)
        schedulers = list(comfy.samplers.SCHEDULER_NAMES)
        installed_turbo_loras = [
            name for name in folder_paths.get_filename_list("loras")
            if "minimax" in name.lower() and "h3" in name.lower() and "turbo" in name.lower()
        ]
        # Only expose files that really exist. The web migration converts old
        # saved missing filenames to the empty/Base-H3 choice before validation.
        turbo_loras = list(dict.fromkeys(("", *installed_turbo_loras)))
        if "res_multistep" in samplers:
            samplers.remove("res_multistep")
            samplers.insert(0, "res_multistep")
        if "simple" in schedulers:
            schedulers.remove("simple")
            schedulers.insert(0, "simple")
        return {
            "required": {
                "global_prompt": ("STRING", {"default": "one continuous cinematic shot with coherent motion, stable identity, controlled camera and native stereo audio", "multiline": True}),
                "timeline_data": ("STRING", {"default": "", "multiline": True}),
                "duration_seconds": ("FLOAT", {"default": 10.0, "min": 0.01, "max": 36000.0, "step": 0.01}),
                "task_mode": (["auto_from_timeline", "t2va", "i2va", "fl2va", "ref2va"], {"default": "auto_from_timeline"}),
                "audio_mode": (["h3_native_generated", "h3_ref2va_audio", "external_audio_post"], {"default": "h3_native_generated"}),
                "prompt_mapping": (["global_plus_local", "local_only", "global_only"], {"default": "global_plus_local"}),
                "upscale_mode": (["off", "ltx23", "wan22_5b"], {"default": "off"}),
                # 0.5 MP / 16:9 is the practical Dynamic-VRAM default used by
                # the proven H3 reference pipeline on a 12 GiB card.  The UI
                # still lets the user choose any valid H3 resolution.
                "width": ("INT", {"default": 960, "min": 256, "max": 5760, "step": 32}),
                "height": ("INT", {"default": 544, "min": 256, "max": 5760, "step": 32}),
                # Private V3-editor contract. These widgets are hidden by the
                # dedicated MiniMax UI and never become H3 sampler controls.
                "frame_rate": ("INT", {"default": 24, "min": 24, "max": 24, "step": 1}),
                "guide_policy": (["every_checked_row"], {"default": "every_checked_row"}),
                "min_guide_gap_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.0, "step": 0.05}),
                "max_guides": ("INT", {"default": 50, "min": 50, "max": 50, "step": 1}),
                "default_force": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 1.0, "step": 0.01}),
                "promptrelay_epsilon": ("FLOAT", {"default": 0.001, "min": 0.001, "max": 0.001, "step": 0.0001}),
                "ltx_round_mode": (["none"], {"default": "none"}),
                "image_paths": ("STRING", {"default": "", "multiline": True}),
                "image_width": ("INT", {"default": 960, "min": 256, "max": 5760, "step": 32}),
                "image_height": ("INT", {"default": 544, "min": 256, "max": 5760, "step": 32}),
                "image_resize_method": (["crop", "pad", "keep proportion", "stretch", ""], {"default": "crop"}),
                "image_multiple_of": ("INT", {"default": 32, "min": 32, "max": 32, "step": 1}),
                "img_compression": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                # H3-native backend controls.  The dedicated Shotboard UI
                # renders these above the timeline and hides the raw widgets.
                "acceleration": (["auto_3060", "native", "h3_sage", "sage", "sage_sol", "spectrum", "sage_spectrum"], {"default": "auto_3060"}),
                "ref_image_size": (["match", "max"], {"default": "match"}),
                "reference_role_1": (["subject_identity", "keyframe", "composition", "style", "disabled"], {"default": "subject_identity"}),
                "reference_role_2": (["subject_identity", "keyframe", "composition", "style", "disabled"], {"default": "subject_identity"}),
                "reference_role_3": (["subject_identity", "keyframe", "composition", "style", "disabled"], {"default": "composition"}),
                "reference_role_4": (["subject_identity", "keyframe", "composition", "style", "disabled"], {"default": "style"}),
                "reference_video_role": (["off", "motion_camera", "temporal_structure", "video_edit", "continuation"], {"default": "off"}),
                "reference_audio_role": (["off", "voice_timbre", "rhythm_timing", "audio_reuse", "sound_reference"], {"default": "off"}),
                "sol_conditioning": (["exact_kv", "exact_kv_and_rows"], {"default": "exact_kv"}),
                "spectrum_profile": (["conservative_3060", "conservative_quality", "aggressive"], {"default": "conservative_3060"}),
                "vram_clean_before_decode": ("BOOLEAN", {"default": True}),
                "rife_mode": (["off", "rife_48fps", "rife_60fps"], {"default": "off"}),
                "upscale_enabled": ("BOOLEAN", {"default": False}),
                # Post-generation delivery controls.  These values travel in
                # CineLinX and drive the optional lazy LTX/Wan branches in the
                # connected workflow; model files remain selectable in those
                # branches instead of being hardcoded in the Shotboard.
                "upscale_width": ("INT", {"default": 1920, "min": 256, "max": 7680, "step": 8}),
                "upscale_height": ("INT", {"default": 1080, "min": 256, "max": 4320, "step": 8}),
                "upscale_prompt": ("STRING", {"default": "", "multiline": True}),
                "upscale_sage": ("BOOLEAN", {"default": True}),
                "upscale_seed_offset": ("INT", {"default": 10000, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                "wan_upscale_denoise": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                # Qwen3-VL 32B GGUF temporarily expands quantized tensors while
                # encoding.  A 12 GiB GPU cannot hold the full encoder plus its
                # dequantization buffers, so the atomic backend defaults to CPU.
                "text_encoder_device": (["cpu_safe_12gb", "auto"], {"default": "cpu_safe_12gb"}),
                # These values are intentionally owned by the Shotboard.  The
                # generation node keeps legacy widgets only as a compatibility
                # fallback for shotplans created before schema v3.
                "performance_profile": (["rtx3060_draft", "rtx3060_balanced", "rtx3060_turbo", "h3_native_quality", "custom"], {"default": "rtx3060_balanced"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "seed_stride": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                "steps": ("INT", {"default": 16, "min": 1, "max": 100, "step": 1}),
                "sampler_name": (samplers,),
                "scheduler": (schedulers,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                # Turbo is kept inside the typed shotplan so the selected LoRA,
                # effective sampler and audio policy cannot drift apart.
                "turbo_mode": (["off", "early_8_10", "ckpt500_6_8"], {"default": "off"}),
                "turbo_lora_name": (turbo_loras, {"default": ""}),
                "turbo_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "turbo_sampler_mode": (["audio_fixed", "res_multistep_stock"], {"default": "audio_fixed"}),
                # The original reference workflow downsizes inputs before H3.
                # Canvas modes are faster on 12 GiB cards than encoding a 1 MP
                # source only to resize it again inside the native conditioner.
                "reference_resize_policy": (["canvas_crop", "canvas_pad", "total_pixels", "off"], {"default": "canvas_crop"}),
                "reference_resize_megapixels": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.05}),
                "reference_resize_filter": (["area", "bilinear", "bicubic", "nearest-exact"], {"default": "area"}),
            },
            "optional": {
                "cine_linx": (
                    SUPERNODE_LINX_TYPE,
                    {
                        "tooltip": "Connect IAMCCS_Prompter here. Its target is resolved against this Shotboard's real timeline slots.",
                    },
                ),
            },
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE,)
    RETURN_NAMES = ("cine_linx",)
    FUNCTION = "plan"
    CATEGORY = CATEGORY

    def plan(
        self,
        global_prompt,
        timeline_data,
        duration_seconds,
        task_mode,
        audio_mode,
        prompt_mapping,
        upscale_mode,
        width,
        height,
        frame_rate=24,
        guide_policy="every_checked_row",
        min_guide_gap_seconds=0.0,
        max_guides=50,
        default_force=1.0,
        promptrelay_epsilon=0.001,
        ltx_round_mode="none",
        image_paths="",
        image_width=960,
        image_height=544,
        image_resize_method="crop",
        image_multiple_of=32,
        img_compression=0,
        acceleration="auto_3060",
        ref_image_size="match",
        reference_role_1="subject_identity",
        reference_role_2="subject_identity",
        reference_role_3="composition",
        reference_role_4="style",
        reference_video_role="off",
        reference_audio_role="off",
        sol_conditioning="exact_kv",
        spectrum_profile="conservative_3060",
        vram_clean_before_decode=True,
        rife_mode="off",
        upscale_enabled=False,
        upscale_width=1920,
        upscale_height=1080,
        upscale_prompt="",
        upscale_sage=True,
        upscale_seed_offset=10000,
        wan_upscale_denoise=0.2,
        text_encoder_device="cpu_safe_12gb",
        performance_profile="rtx3060_balanced",
        seed=42,
        seed_stride=1,
        steps=16,
        sampler_name="res_multistep",
        scheduler="simple",
        denoise=1.0,
        shift_video=12.0,
        shift_audio=3.0,
        turbo_mode="off",
        turbo_lora_name="",
        turbo_strength=1.0,
        turbo_sampler_mode="audio_fixed",
        reference_resize_policy="canvas_crop",
        reference_resize_megapixels=0.5,
        reference_resize_filter="area",
        cine_linx=None,
    ):
        global_prompt, timeline_data, prompter_injection = apply_prompter_to_minimax(
            cine_linx,
            global_prompt,
            timeline_data,
        )
        width = _h3_legal_dimension(width, 960)
        height = _h3_legal_dimension(height, 544)
        image_width = _h3_legal_dimension(image_width, width)
        image_height = _h3_legal_dimension(image_height, height)
        reference_resize_megapixels = _finite_float(reference_resize_megapixels, 0.5, 0.1, 2.0)

        requested_turbo_mode = str(turbo_mode or "off")
        selected_turbo_lora = str(turbo_lora_name or "").strip()
        turbo_requested = requested_turbo_mode != "off"
        turbo_available = _model_file_available("loras", selected_turbo_lora)
        effective_turbo_mode = requested_turbo_mode if turbo_requested and turbo_available else "off"
        requested_steps = max(1, int(_finite_float(steps, 16, 1, 100)))
        # A missing optional Turbo file must not leave a base-model generation
        # running with an unsafe 6-10 step Turbo schedule.
        effective_steps = max(16, requested_steps) if turbo_requested and not turbo_available else requested_steps
        plan = build_shotplan(
            timeline_data=timeline_data,
            global_prompt=global_prompt,
            duration_seconds=duration_seconds,
            task_mode=task_mode,
            audio_mode=audio_mode,
            prompt_mapping=prompt_mapping,
            upscale_mode=upscale_mode,
            width=width,
            height=height,
            acceleration=acceleration,
            ref_image_size=ref_image_size,
            text_encoder_device=text_encoder_device,
            reference_roles=[reference_role_1, reference_role_2, reference_role_3, reference_role_4],
            reference_video_role=reference_video_role,
            reference_audio_role=reference_audio_role,
            sol_conditioning=sol_conditioning,
            spectrum_profile=spectrum_profile,
            vram_clean_before_decode=vram_clean_before_decode,
            rife_mode=rife_mode,
            upscale_enabled=upscale_enabled,
        )
        plan["prompter_injection"] = prompter_injection
        plan["performance_profile"] = str(performance_profile)
        plan["sampling"] = {
            "seed": int(seed),
            "seed_stride": int(seed_stride),
            "steps": effective_steps,
            "requested_steps": requested_steps,
            "sampler_name": str(sampler_name),
            "scheduler": str(scheduler),
            "denoise": float(denoise),
            "shift_video": float(shift_video),
            "shift_audio": float(shift_audio),
            "source": "shotboard",
        }
        plan["turbo"] = {
            "mode": effective_turbo_mode,
            "requested_mode": requested_turbo_mode,
            "enabled": effective_turbo_mode != "off",
            "requested": turbo_requested,
            "available": turbo_available,
            "fallback_to_base": turbo_requested and not turbo_available,
            "lora_name": selected_turbo_lora,
            "strength": float(turbo_strength),
            "sampler_mode": str(turbo_sampler_mode),
            "source": "shotboard",
        }
        plan["reference_resize"] = {
            "policy": str(reference_resize_policy),
            "megapixels": reference_resize_megapixels,
            "filter": str(reference_resize_filter),
            "multiple_of": 32,
            "downscale_only": True,
            "source": "shotboard",
        }
        plan["upscale_settings"] = {
            "target_width": int(upscale_width),
            "target_height": int(upscale_height),
            "prompt": str(upscale_prompt or "").strip(),
            "sage": bool(upscale_sage),
            "seed_offset": int(upscale_seed_offset),
            "wan_denoise": float(wan_upscale_denoise),
            "source": "shotboard",
        }
        chunk_frames = [int(chunk.get("frame_count", 0) or 0) for chunk in plan.get("chunks", [])]
        max_chunk_frames = max(chunk_frames, default=0)
        native_load = (float(width) * float(height) * max(1, max_chunk_frames)) / (960.0 * 544.0 * 124.0)
        warnings: list[str] = []
        if str(performance_profile).startswith("rtx3060") and max_chunk_frames > 124:
            warnings.append("Low VRAM: trim this timeline box to 124 frames or less; use a following box for continuation")
        if str(performance_profile).startswith("rtx3060") and int(width) * int(height) > 960 * 544:
            warnings.append("Low VRAM: generate at 960x544 or below, then upscale for a 1280-class delivery")
        if str(acceleration) == "sage_sol":
            warnings.append("Sol-Attn is experimental, has a slower first compile, and is not validated for every Low VRAM configuration")
        if str(acceleration) in {"spectrum", "sage_spectrum"} and effective_steps < 14:
            warnings.append("Spectrum saves few transformer calls below 14 steps because warmup and final native steps remain mandatory")
        turbo_enabled = effective_turbo_mode != "off"
        if turbo_requested and not turbo_available:
            missing_name = selected_turbo_lora or "no LoRA selected"
            warnings.append(f"Optional Turbo LoRA unavailable ({missing_name}); using base H3 at {effective_steps} steps")
        if turbo_enabled and not 0.8 <= float(turbo_strength) <= 1.8:
            warnings.append("Turbo LoRA strength is outside the tested 0.8-1.8 range")
        if effective_turbo_mode == "early_8_10" and not 8 <= effective_steps <= 10:
            warnings.append("Early/non-ckpt500 Turbo is normally used at 8-10 steps")
        if effective_turbo_mode == "ckpt500_6_8" and not 6 <= effective_steps <= 8:
            warnings.append("Turbo ckpt500 is normally used at 6-8 steps")
        if turbo_enabled and str(acceleration) in {"spectrum", "sage_spectrum"}:
            warnings.append("Spectrum has little room to forecast at Turbo step counts; Sage-only is the Low VRAM default")
        if turbo_enabled and str(turbo_sampler_mode) == "res_multistep_stock" and int(steps) < 10:
            warnings.append("Stock res_multistep can over-step H3 audio at low Turbo step counts; use Audio-fixed sampler")
        plan["performance"] = {
            "profile": str(performance_profile),
            "relative_native_load_vs_960x544x124": round(native_load, 3),
            "max_chunk_frames": max_chunk_frames,
            "warnings": warnings,
        }
        plan["control_contract"] = {
            "model_route": ["task_mode", "reference roles"],
            "conditioning": ["width", "height", "timeline trim", "prompt mapping", "references", "audio mode"],
            "sampling": ["seed", "steps", "sampler", "scheduler", "denoise", "H3 shifts", "acceleration", "Turbo LoRA", "Turbo audio sampler"],
            "reference_preprocess": ["resize policy", "target megapixels", "filter", "multiple of 32"],
            "delivery": ["VRAM clean", "RIFE", "upscale enabled", "upscale mode", "upscale target", "upscale prompt", "upscale seed"],
            "transport": "one IAMCCS_SUPERNODE_LINX cable; the private H3 plan stays inside CineLinX",
        }
        injection_summary = str(prompter_injection.get("actual_target", "none")) if prompter_injection.get("applied") else "none"
        report = (
            f"MiniMax H3 Shotboard | shots={len(plan['slots'])} | keyframes={plan['total_keyframes']} | chunks={plan['total_segments']} | "
            f"duration={plan['effective_duration_seconds']:.3f}s | grid=17k+5 @24fps | "
            f"task={task_mode} | frames=timeline trim (max 362 per box) | prompt={prompt_mapping} | "
            f"audio={audio_mode} | resolution={width}x{height} | performance={performance_profile} "
            f"load={native_load:.2f}x | sampler={effective_steps}x{sampler_name}+{scheduler} | acceleration={acceleration} | "
            f"turbo={effective_turbo_mode}:{selected_turbo_lora or 'none'}@{float(turbo_strength):.2f}/{turbo_sampler_mode} | "
            f"ref_resize={reference_resize_policy}:{reference_resize_megapixels:.2f}MP/{reference_resize_filter} | "
            f"ref_size={ref_image_size} | text_encoder={text_encoder_device} | "
            f"RIFE={rife_mode} | upscale={'on' if upscale_enabled else 'off'}:{plan['upscale_mode']} "
            f"->{int(upscale_width)}x{int(upscale_height)} sage={'on' if upscale_sage else 'off'} "
            f"wan_denoise={float(wan_upscale_denoise):.2f} | "
            f"prompter={injection_summary} | warnings={'; '.join(warnings) if warnings else 'none'}"
        )
        return (_build_minimax_cine_linx(cine_linx, plan, timeline_data, global_prompt, report),)


def _plan_chunk(cine_linx: dict[str, Any], segment_index: int) -> dict[str, Any]:
    shotplan = _resolve_shotplan(cine_linx)
    chunks = shotplan.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("shotplan MiniMax H3 senza chunks")
    index = int(segment_index)
    if index < 0 or index >= len(chunks):
        raise IndexError(f"segment_index={index} fuori intervallo 0..{len(chunks) - 1}")
    return chunks[index]


class IAMCCS_MiniMaxH3PromptRelayMap:
    """Compatibility node: returns the one static H3 prompt for a chunk.

    MiniMax H3 has no PromptRelay/conditioning schedule in the native ComfyUI
    implementation. The legacy class id is retained so older workflows open.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
            },
            "optional": {"prompt_override": ("STRING", {"default": "", "multiline": True})},
        }

    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT", "INT", "INT", "STRING")
    RETURN_NAMES = ("prompt", "start_seconds", "duration_seconds", "frame_count", "total_segments", "report")
    FUNCTION = "map_prompt"
    CATEGORY = CATEGORY

    def map_prompt(self, cine_linx, segment_index, prompt_override=""):
        shotplan = _resolve_shotplan(cine_linx)
        chunk = _plan_chunk(shotplan, segment_index)
        prompt = str(prompt_override or "").strip() or str(chunk.get("prompt", ""))
        report = (
            f"H3 Static Prompt Map (no Text Relay) | segment={int(segment_index) + 1}/{shotplan['total_segments']} | "
            f"slot={chunk.get('slot_label')} | frames={chunk['frame_count']} | "
            f"mapping={shotplan.get('prompt_mapping')}"
        )
        return (
            prompt,
            float(chunk["timeline_start_seconds"]),
            float(chunk["duration_seconds"]),
            int(chunk["frame_count"]),
            int(shotplan["total_segments"]),
            report,
        )


class IAMCCS_MiniMaxH3LumosPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "instruction": ("STRING", {"default": "Rewrite the shot as a precise MiniMax H3 prompt. Preserve intent. Include camera, motion, continuity and audio timing. Return only the final prompt.", "multiline": True}),
                "max_new_tokens": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {"reference_image": ("IMAGE",)},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "report")
    FUNCTION = "enhance"
    CATEGORY = CATEGORY

    def enhance(self, clip, cine_linx, segment_index, instruction, max_new_tokens, temperature, seed, reference_image=None):
        chunk = _plan_chunk(cine_linx, segment_index)
        source_prompt = str(chunk.get("prompt", ""))
        request = f"{instruction.strip()}\n\nSOURCE SHOT PROMPT:\n{source_prompt}"
        images = [reference_image[:1]] if torch.is_tensor(reference_image) else []
        tokens = clip.tokenize(request, images=images)
        token_ids = clip.generate(
            tokens,
            do_sample=float(temperature) > 0.0,
            max_length=int(max_new_tokens),
            temperature=float(temperature),
            top_k=50,
            top_p=0.92,
            min_p=0.02,
            repetition_penalty=1.05,
            seed=int(seed),
        )
        result = str(clip.decode(token_ids)).strip()
        return result, f"Lumos-compatible Qwen3-VL prompt rewrite | segment={int(segment_index) + 1} | image={'yes' if images else 'no'}"


class IAMCCS_MiniMaxH3PlanFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("first_frame", "last_frame", "has_first", "has_last", "report")
    FUNCTION = "frames"
    CATEGORY = CATEGORY

    def frames(self, cine_linx, segment_index):
        shotplan = _resolve_shotplan(cine_linx)
        chunk = _plan_chunk(shotplan, segment_index)
        first = _load_image(str(chunk.get("first_image", "")))
        last = _load_image(str(chunk.get("last_image", "")))
        width = int(shotplan.get("width", 64))
        height = int(shotplan.get("height", 64))
        return (
            first if first is not None else _black_image(width, height),
            last if last is not None else _black_image(width, height),
            int(first is not None),
            int(last is not None),
            f"Plan frames segment={int(segment_index) + 1} first={'yes' if first is not None else 'bridge/none'} last={'yes' if last is not None else 'none'}",
        )


class IAMCCS_MiniMaxH3Backend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "video_vae": ("VAE",),
                "audio_vae": ("VAE",),
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
            },
            "optional": {
                "model_task": ("STRING", {"default": "auto"}),
                "loaded_model_task": ("STRING", {"forceInput": True}),
                "render_id": ("STRING", {"default": "minimax_h3_render"}),
                "driven_audio": ("AUDIO",),
                "bridge_frame": ("IMAGE",),
                "first_frame_override": ("IMAGE",),
                "last_frame_override": ("IMAGE",),
                "prompt_override": ("STRING", {"default": "", "multiline": True}),
                "release_text_encoder_memory": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "LATENT", "IMAGE", "IMAGE", "AUDIO", "STRING", "INT", "INT", "INT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = (
        "model",
        "positive",
        "latent",
        "first_frame",
        "last_frame",
        "driven_audio_slice",
        "prompt",
        "current_segment",
        "total_segments",
        "trim_head_frames",
        "timeline_start_seconds",
        "upscale_mode",
        "report",
    )
    FUNCTION = "prepare"
    CATEGORY = CATEGORY

    def prepare(
        self,
        model,
        clip,
        video_vae,
        audio_vae,
        cine_linx,
        segment_index,
        model_task="auto",
        loaded_model_task=None,
        render_id="minimax_h3_render",
        driven_audio=None,
        bridge_frame=None,
        first_frame_override=None,
        last_frame_override=None,
        prompt_override="",
        release_text_encoder_memory=True,
    ):
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo

        shotplan = _resolve_shotplan(cine_linx)
        chunk = _plan_chunk(shotplan, segment_index)
        width = int(shotplan.get("width", 960))
        height = int(shotplan.get("height", 544))
        frame_count = int(chunk["frame_count"])
        prompt = str(prompt_override or "").strip() or str(chunk.get("prompt", ""))

        planned_first = _load_image(str(chunk.get("first_image", "")))
        planned_last = _load_image(str(chunk.get("last_image", "")))
        first = first_frame_override if torch.is_tensor(first_frame_override) else planned_first
        use_bridge = bool(chunk.get("uses_bridge_first_frame"))
        if first is None and use_bridge and torch.is_tensor(bridge_frame):
            first = bridge_frame[:1]
        if first is None and use_bridge and str(render_id or "").strip():
            bridge_path = _bridge_path(str(render_id))
            if bridge_path.is_file():
                first = _load_frame(bridge_path)
        if first is None and use_bridge:
            raise FileNotFoundError(
                f"Bridge last-to-first mancante per il chunk {int(segment_index) + 1}. "
                "Avvia la coda dal segmento 0 o collega bridge_frame."
            )
        last = last_frame_override if torch.is_tensor(last_frame_override) else planned_last

        audio_slice = _slice_audio(
            driven_audio,
            float(chunk.get("timeline_start_seconds", 0.0)),
            float(chunk.get("duration_seconds", frame_count / H3_FPS)),
        )
        mode = str(chunk.get("task_mode", "t2va"))
        loaded_task = str(loaded_model_task or model_task or "auto").lower()
        warning = ""

        if mode in {"ref2va_audio", "ref2va_reference"}:
            if loaded_task == "fl2va":
                warning = " | ATTENZIONE: il backend Ref2VA richiede pesi ref2va, non fl2va"
            refs: dict[str, torch.Tensor] = {}
            if first is not None:
                refs["ref_image_1"] = first[:1]
            if last is not None:
                refs[f"ref_image_{len(refs) + 1}"] = last[:1]
            if mode == "ref2va_reference" and not refs:
                raise ValueError("Ref2VA richiede almeno un'immagine di riferimento o bridge_frame")
            ref_audios = None
            tagged = []
            if first is not None:
                tagged.append("Use <Picture 1> as the subject, scene and opening-state reference.")
            if last is not None:
                tagged.append(f"Use <Picture {len(refs)}> as the desired end-state reference.")
            if mode == "ref2va_audio":
                if not isinstance(driven_audio, dict):
                    raise ValueError("ref2va_audio richiede driven_audio")
                ref_audios = {"ref_audio_1": audio_slice}
                tagged.append("Follow <Audio 1> for voice, rhythm, timing and audiovisual action.")
            tagged_prompt = "\n".join(tagged + [prompt])
            result = MiniMaxH3ReferenceToVideo.execute(
                clip=clip,
                vae=video_vae,
                audio_vae=audio_vae,
                prompt=tagged_prompt,
                width=width,
                height=height,
                length=frame_count,
                ref_image_size="match",
                ref_images=refs,
                ref_audios=ref_audios,
            )
            positive, latent = result[0], result[1]
            prompt = tagged_prompt
        else:
            if loaded_task == "ref2va":
                warning = " | ATTENZIONE: FL2VA/T2VA richiede pesi fl2va, non ref2va"
            conditioned_first = first if mode in {"i2v", "i2va", "fl2va"} else None
            conditioned_last = last if mode == "fl2va" else None
            result = MiniMaxH3ImageToVideo.execute(
                clip=clip,
                vae=video_vae,
                prompt=prompt,
                width=width,
                height=height,
                length=frame_count,
                first_frame=conditioned_first,
                last_frame=conditioned_last,
            )
            positive, latent = result[0], result[1]

        memory_report = "text encoder kept"
        clip_device = getattr(getattr(clip, "patcher", None), "load_device", None)
        if bool(release_text_encoder_memory) and str(clip_device) == "cpu":
            memory_report = _release_cpu_text_encoder_memory(clip)

        report = (
            f"MiniMax H3 Backend | segment={int(segment_index) + 1}/{shotplan['total_segments']} | "
            f"mode={mode} | frames={frame_count} | first={'yes' if first is not None else 'no'} | "
            f"last={'yes' if last is not None else 'no'} | audio={shotplan.get('audio_mode')} | "
            f"upscale={shotplan.get('upscale_mode')} | memory={memory_report}{warning}"
        )
        return (
            model,
            positive,
            latent,
            first if first is not None else _black_image(width, height),
            last if last is not None else _black_image(width, height),
            audio_slice,
            prompt,
            int(segment_index),
            int(shotplan["total_segments"]),
            int(chunk.get("trim_head_frames", 0)),
            float(chunk.get("timeline_start_seconds", 0.0)),
            str(shotplan.get("upscale_mode", "off")),
            report,
        )


class IAMCCS_MiniMaxH3RenderBackend:
    """Run one planned H3 chunk and decode both NestedTensor streams."""

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers

        samplers = list(comfy.samplers.SAMPLER_NAMES)
        schedulers = list(comfy.samplers.SCHEDULER_NAMES)
        # Match the stable MiniMax H3 Dynamic-VRAM reference pipeline.
        if "euler" in samplers:
            samplers.remove("euler")
            samplers.insert(0, "euler")
        if "simple" in schedulers:
            schedulers.remove("simple")
            schedulers.insert(0, "simple")
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "latent": ("LATENT",),
                "video_vae": ("VAE",),
                "audio_vae": ("VAE",),
                "chunk_index": ("INT", {"forceInput": True}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "seed_stride": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1}),
                "sampler_name": (samplers,),
                "scheduler": (schedulers,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "LATENT", "MODEL", "STRING")
    RETURN_NAMES = ("frames", "audio", "sampled_latent", "shifted_model", "report")
    FUNCTION = "render"
    CATEGORY = CATEGORY

    def render(
        self,
        model,
        positive,
        latent,
        video_vae,
        audio_vae,
        chunk_index,
        seed,
        seed_stride,
        steps,
        sampler_name,
        scheduler,
        denoise,
        shift_video,
        shift_audio,
    ):
        import nodes as comfy_nodes
        from comfy_extras.nodes_audio import VAEDecodeAudio
        from comfy_extras.nodes_custom_sampler import BasicGuider, BasicScheduler, KSamplerSelect, RandomNoise, SamplerCustomAdvanced
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift

        actual_seed = (int(seed) + int(chunk_index) * int(seed_stride)) & 0xFFFFFFFFFFFFFFFF
        shifted_model = MiniMaxH3SigmaShift.execute(
            model=model,
            shift_video=float(shift_video),
            shift_audio=float(shift_audio),
        )[0]
        noise = RandomNoise.execute(noise_seed=actual_seed)[0]
        guider = BasicGuider.execute(model=shifted_model, conditioning=positive)[0]
        sampler = KSamplerSelect.execute(sampler_name=str(sampler_name))[0]
        sigmas = BasicScheduler.execute(
            model=shifted_model,
            scheduler=str(scheduler),
            steps=int(steps),
            denoise=float(denoise),
        )[0]
        sampled = SamplerCustomAdvanced.execute(
            noise=noise,
            guider=guider,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=latent,
        )[0]
        frames = comfy_nodes.VAEDecode().decode(vae=video_vae, samples=sampled)[0]
        audio = VAEDecodeAudio.execute(vae=audio_vae, samples=sampled)[0]
        report = (
            f"MiniMax H3 Render Backend | chunk={int(chunk_index) + 1} | seed={actual_seed} | steps={int(steps)} | "
            f"sampler={sampler_name} | scheduler={scheduler} | denoise={float(denoise):.2f} | "
            f"shift_video={float(shift_video):.2f} | shift_audio={float(shift_audio):.2f}"
        )
        return frames, audio, sampled, shifted_model, report


class IAMCCS_MiniMaxH3BridgeLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "render_id": ("STRING", {"default": "minimax_h3_render"}),
            },
            "optional": {"fallback_image": ("IMAGE",)},
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("image", "exists", "report")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, segment_index, render_id, fallback_image=None):
        if int(segment_index) <= 0:
            if torch.is_tensor(fallback_image):
                return fallback_image[:1], 0, "MiniMax H3 bridge: segmento iniziale, uso fallback_image"
            return _black_image(), 0, "MiniMax H3 bridge: segmento iniziale senza first frame forzato"

        bridge_path = _bridge_path(str(render_id))
        if bridge_path.is_file():
            return _load_frame(bridge_path), 1, f"MiniMax H3 bridge caricato: {bridge_path.name}"
        if torch.is_tensor(fallback_image):
            return fallback_image[:1], 0, f"MiniMax H3 bridge assente, uso fallback_image: {bridge_path.name}"
        raise FileNotFoundError(f"MiniMax H3 bridge non trovato: {bridge_path}")


class IAMCCS_MiniMaxH3SegmentQueueLoop:
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
                "bridge_images": ("IMAGE",),
                "trim_head_frames": ("INT", {"forceInput": True}),
                "enabled": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "IAMCCS/MiniMaxH3/segment"}),
                "merge_segments": ("BOOLEAN", {"default": True}),
                "keep_segments": ("BOOLEAN", {"default": True}),
                "render_id": ("STRING", {"default": "minimax_h3_render"}),
                "segment_base_name": ("STRING", {"default": ""}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "queue_next"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def queue_next(
        self,
        current_segment,
        total_segments,
        video=None,
        images=None,
        audio=None,
        bridge_images=None,
        trim_head_frames=0,
        enabled=True,
        filename_prefix="IAMCCS/MiniMaxH3/segment",
        merge_segments=True,
        keep_segments=True,
        render_id="minimax_h3_render",
        segment_base_name="",
        prompt=None,
        unique_id=None,
        extra_pnginfo=None,
    ):
        current_segment = int(current_segment)
        total_segments = int(total_segments)
        if current_segment < 0:
            raise ValueError("current_segment non può essere negativo")
        if total_segments < 1 or current_segment >= total_segments:
            raise ValueError(f"Indice segmento non valido: {current_segment + 1}/{total_segments}")
        if not str(render_id or "").strip() and not str(segment_base_name or "").strip() and current_segment > 0:
            raise RuntimeError(
                "Il render riparte da un segmento successivo al primo senza render_id. "
                "Reimposta segment_index a 0 prima di una nuova generazione."
            )

        requested_render_id = _safe_name(
            str(render_id or "").strip() or uuid.uuid4().hex[:10],
            "minimax_h3_render",
        )
        output_folder, resolved_base_name = _output_location(filename_prefix)
        active_base_name = _safe_name(str(segment_base_name or "").strip(), resolved_base_name)
        active_render_id = (
            _next_numbered_render_id(output_folder, active_base_name, requested_render_id)
            if current_segment == 0
            else requested_render_id
        )
        if current_segment == 0:
            LOG.info("MiniMax H3 nuovo render numerato: %s", active_render_id)
        segment_name = f"{active_base_name}_{active_render_id}_seg_{current_segment + 1:04d}.mp4"
        segment_path = output_folder / segment_name
        _require_new_output_path(segment_path)

        images_to_save = images
        audio_to_save = audio
        trim_count = max(0, int(trim_head_frames or 0))
        if trim_count and torch.is_tensor(images) and int(images.shape[0]) > trim_count:
            images_to_save = images[trim_count:, ...]
            audio_to_save = _trim_audio_frames(audio, trim_count, H3_FPS)

        if video is not None:
            try:
                from comfy_api.latest import Types

                video.save_to(str(segment_path), format=Types.VideoContainer("auto"), codec="auto", metadata=None)
            except Exception as exc:
                raise RuntimeError(f"Salvataggio VIDEO MiniMax H3 fallito: {exc}") from exc
        elif torch.is_tensor(images_to_save):
            _encode_images(images_to_save, audio_to_save, H3_FPS, segment_path)
        else:
            raise ValueError("Collega video oppure images al Segment Queue Loop")

        bridge_source = bridge_images if torch.is_tensor(bridge_images) else images
        if torch.is_tensor(bridge_source) and int(bridge_source.shape[0]) > 0:
            bridge_path = _bridge_path(active_render_id)
            _save_frame(bridge_path, bridge_source[-1])
            LOG.info("MiniMax H3 last-frame bridge salvato in %s", bridge_path)

        saved_message = f"Salvato segmento {current_segment + 1}/{total_segments}: {segment_name}"
        if not bool(enabled):
            return {"ui": {"text": [saved_message]}}

        next_segment = current_segment + 1
        if next_segment >= total_segments:
            messages = [saved_message]
            preview = None
            if bool(merge_segments):
                segment_paths = [
                    output_folder / f"{active_base_name}_{active_render_id}_seg_{index + 1:04d}.mp4"
                    for index in range(total_segments)
                ]
                final_name = f"{active_base_name}_{active_render_id}_full.mp4"
                final_path = output_folder / final_name
                _require_new_output_path(final_path)
                _concat_videos(segment_paths, final_path)
                messages.append(f"Video finale concatenato: {final_name}")
                subfolder = os.path.relpath(output_folder, folder_paths.get_output_directory()).replace("\\", "/")
                preview = {"filename": final_name, "subfolder": "" if subfolder == "." else subfolder, "type": "output"}
                if not bool(keep_segments):
                    for path in segment_paths:
                        path.unlink(missing_ok=True)
            ui = {"text": messages}
            if preview is not None:
                ui.update({"images": [preview], "animated": (True,)})
            return {"ui": ui}

        live_extra = None
        live_outputs = None
        live_sensitive = None
        base_prompt = prompt
        try:
            live_prompt, live_extra, live_outputs, live_sensitive = _current_prompt()
            if live_prompt is not None:
                base_prompt = live_prompt
        except Exception:
            pass
        if base_prompt is None:
            raise RuntimeError("Prompt ComfyUI corrente non disponibile per accodare il prossimo segmento")
        prompt_copy = copy.deepcopy({str(key): value for key, value in base_prompt.items()})

        loop_updated = False
        updated_segment_nodes = 0
        for node_id, node in prompt_copy.items():
            inputs = node.setdefault("inputs", {})
            if "segment_index" in inputs:
                inputs["segment_index"] = next_segment
                updated_segment_nodes += 1
            if "render_id" in inputs:
                inputs["render_id"] = active_render_id
            is_current_loop = unique_id is not None and node_id == str(unique_id)
            is_loop_fallback = unique_id is None and node.get("class_type") == "IAMCCS_MiniMaxH3SegmentQueueLoop"
            if is_current_loop or is_loop_fallback:
                inputs["render_id"] = active_render_id
                inputs["segment_base_name"] = active_base_name
                loop_updated = True

        if updated_segment_nodes == 0:
            raise ValueError("Il Segment Queue Loop non trova nessun nodo con input segment_index")
        if not loop_updated:
            raise ValueError("Il Segment Queue Loop non riesce ad aggiornare il proprio render_id")

        _enqueue(prompt_copy, extra_data=live_extra, outputs=live_outputs, sensitive=live_sensitive)
        return {
            "ui": {
                "text": [
                    saved_message,
                    f"Accodato segmento {next_segment + 1}/{total_segments}",
                ]
            }
        }


class IAMCCS_MiniMaxH3AudioConcat:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {f"audio_{index}": ("AUDIO",) for index in range(2, 17)}
        return {
            "required": {
                "audio_1": ("AUDIO",),
                "trim_head_frames": ("INT", {"default": 1, "min": 0, "max": 24, "step": 1}),
                "crossfade_ms": ("FLOAT", {"default": 41.67, "min": 0.0, "max": 1000.0, "step": 1.0}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 1.0}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "report")
    FUNCTION = "concat"
    CATEGORY = CATEGORY

    def concat(self, audio_1, trim_head_frames, crossfade_ms, fps, **kwargs):
        audios = [audio_1] + [kwargs.get(f"audio_{index}") for index in range(2, 17)]
        audios = [audio for audio in audios if isinstance(audio, dict) and torch.is_tensor(audio.get("waveform"))]
        if not audios:
            raise ValueError("Nessun audio da concatenare")
        sample_rate = int(audios[0].get("sample_rate", 32000))
        waves = []
        for audio in audios:
            wave = _normalise_channels(audio["waveform"], 2)
            wave = _resample(wave, int(audio.get("sample_rate", sample_rate)), sample_rate)
            waves.append(wave)

        output = waves[0]
        trim_samples = max(0, int(round((float(trim_head_frames) / max(1.0, float(fps))) * sample_rate)))
        requested_crossfade = max(0, int(round(float(crossfade_ms) / 1000.0 * sample_rate)))
        for wave in waves[1:]:
            trim = min(trim_samples, max(0, wave.shape[-1] - 1))
            overlap = min(trim, requested_crossfade, output.shape[-1], wave.shape[-1])
            if overlap > 0:
                phase = torch.linspace(0.0, math.pi / 2.0, overlap, device=output.device, dtype=output.dtype)
                fade_out = torch.cos(phase).view(1, 1, -1)
                fade_in = torch.sin(phase).view(1, 1, -1)
                blended = output[:, :, -overlap:] * fade_out + wave[:, :, :overlap] * fade_in
                output = torch.cat([output[:, :, :-overlap], blended, wave[:, :, trim:]], dim=-1)
            else:
                output = torch.cat([output, wave[:, :, trim:]], dim=-1)
        audio = {"waveform": output, "sample_rate": sample_rate}
        return audio, f"H3 audio concat | chunks={len(waves)} | sr={sample_rate} | trim={trim_head_frames}f | crossfade={crossfade_ms:.2f}ms"


class IAMCCS_MiniMaxH3AudioPolicy:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3_audio": ("AUDIO",),
                "mode": (["h3_native_generated", "h3_ref2va_audio", "external_audio_post"], {"default": "h3_native_generated"}),
                "h3_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "driven_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {"driven_audio": ("AUDIO",)},
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "report")
    FUNCTION = "select"
    CATEGORY = CATEGORY

    def select(self, h3_audio, mode, h3_gain, driven_gain, driven_audio=None):
        if mode in {"h3_native_generated", "h3_ref2va_audio"}:
            waveform = h3_audio["waveform"] * float(h3_gain)
            return {
                "waveform": torch.clamp(waveform, -1.0, 1.0),
                "sample_rate": int(h3_audio.get("sample_rate", 32000)),
            }, f"Audio policy: {mode} output generated by MiniMax H3"
        if not isinstance(driven_audio, dict) or not torch.is_tensor(driven_audio.get("waveform")):
            raise ValueError(f"Audio policy {mode} richiede driven_audio")
        waveform = driven_audio["waveform"] * float(driven_gain)
        return {
            "waveform": torch.clamp(waveform, -1.0, 1.0),
            "sample_rate": int(driven_audio.get("sample_rate", 32000)),
        }, "Audio policy: external audio after H3 (not a conditioning input)"


class IAMCCS_MiniMaxH3UpscaleRouter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "native_frames": ("IMAGE",),
                "audio": ("AUDIO",),
                "upscale_mode": (["off", "ltx23", "wan22_5b"], {"default": "off"}),
            },
            "optional": {
                "planned_upscale_mode": ("STRING", {"forceInput": True}),
                "ltx23_frames": ("IMAGE",),
                "wan22_frames": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("frames", "audio", "report")
    FUNCTION = "route"
    CATEGORY = CATEGORY

    def route(
        self,
        native_frames,
        audio,
        upscale_mode,
        planned_upscale_mode=None,
        ltx23_frames=None,
        wan22_frames=None,
    ):
        active_mode = str(planned_upscale_mode or upscale_mode or "off")
        if active_mode == "ltx23":
            if not torch.is_tensor(ltx23_frames):
                raise ValueError("upscale_mode=ltx23 ma il ramo LTX 2.3 non è collegato")
            return ltx23_frames, audio, "Upscale router: LTX 2.3"
        if active_mode == "wan22_5b":
            if not torch.is_tensor(wan22_frames):
                raise ValueError("upscale_mode=wan22_5b ma il ramo Wan 2.2 5B non è collegato")
            return wan22_frames, audio, "Upscale router: Wan 2.2 5B"
        return native_frames, audio, "Upscale router: native H3"


NODE_CLASS_MAPPINGS = {
    "IAMCCS_MiniMaxH3GGUFLoader": IAMCCS_MiniMaxH3GGUFLoader,
    "IAMCCS_MiniMaxH3ShotPlanner": IAMCCS_MiniMaxH3ShotPlanner,
    "IAMCCS_MiniMaxH3PromptRelayMap": IAMCCS_MiniMaxH3PromptRelayMap,
    "IAMCCS_MiniMaxH3LumosPrompt": IAMCCS_MiniMaxH3LumosPrompt,
    "IAMCCS_MiniMaxH3PlanFrames": IAMCCS_MiniMaxH3PlanFrames,
    "IAMCCS_MiniMaxH3Backend": IAMCCS_MiniMaxH3Backend,
    "IAMCCS_MiniMaxH3RenderBackend": IAMCCS_MiniMaxH3RenderBackend,
    "IAMCCS_MiniMaxH3BridgeLoad": IAMCCS_MiniMaxH3BridgeLoad,
    "IAMCCS_MiniMaxH3SegmentQueueLoop": IAMCCS_MiniMaxH3SegmentQueueLoop,
    "IAMCCS_MiniMaxH3AudioConcat": IAMCCS_MiniMaxH3AudioConcat,
    "IAMCCS_MiniMaxH3AudioPolicy": IAMCCS_MiniMaxH3AudioPolicy,
    "IAMCCS_MiniMaxH3UpscaleRouter": IAMCCS_MiniMaxH3UpscaleRouter,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_MiniMaxH3GGUFLoader": "MiniMax H3 GGUF Loader + Spectrum",
    "IAMCCS_MiniMaxH3ShotPlanner": "MiniMax H3 Shotboard",
    "IAMCCS_MiniMaxH3PromptRelayMap": "MiniMax H3 Prompt Map",
    "IAMCCS_MiniMaxH3LumosPrompt": "MiniMax H3 Lumos Prompt Enhancer",
    "IAMCCS_MiniMaxH3PlanFrames": "MiniMax H3 Plan Frames",
    "IAMCCS_MiniMaxH3Backend": "MiniMax H3 Shotboard Backend",
    "IAMCCS_MiniMaxH3RenderBackend": "MiniMax H3 Render Backend (Sampler + AV Decode)",
    "IAMCCS_MiniMaxH3BridgeLoad": "MiniMax H3 Last-Frame Bridge",
    "IAMCCS_MiniMaxH3SegmentQueueLoop": "MiniMax H3 Segment Queue + Concat",
    "IAMCCS_MiniMaxH3AudioConcat": "MiniMax H3 Audio Chunk Concat",
    "IAMCCS_MiniMaxH3AudioPolicy": "MiniMax H3 Audio Policy",
    "IAMCCS_MiniMaxH3UpscaleRouter": "MiniMax H3 Optional Upscale Router",
}


# The V2 execution backend remains in its own GPL module.  Keeping registration
# here lets the existing isolated MiniMax package expose it without touching any
# V3/V4 Shotboard module or JavaScript registration.
from .iamccs_minimax_h3_atomic_backend import (
    NODE_CLASS_MAPPINGS as _ATOMIC_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _ATOMIC_NODE_DISPLAY_NAME_MAPPINGS,
)

NODE_CLASS_MAPPINGS.update(_ATOMIC_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(_ATOMIC_NODE_DISPLAY_NAME_MAPPINGS)
