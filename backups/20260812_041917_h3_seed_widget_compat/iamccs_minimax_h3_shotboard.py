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

# R21 keeps every value that shipped in previous workflows and adds one new,
# deliberately distinct route for the v16-style pre-sampler audio latent.
# ``h3_ref2va_audio`` must never be silently reinterpreted as custom drive:
# it selects MiniMax's reference-conditioning path and still generates a new
# output soundtrack. ``h3_custom_audio_drive`` instead reserves a runtime
# AUDIO tensor supplied by CineInfoH3 for an atomic latent-drive backend.
H3_AUDIO_MODES = (
    "h3_native_generated",
    "h3_ref2va_audio",
    "h3_custom_audio_drive",
    "external_audio_post",
)
H3_AUDIO_MODE_ALIASES = {
    "": "h3_native_generated",
    "native": "h3_native_generated",
    "native_generated": "h3_native_generated",
    "h3_native": "h3_native_generated",
    "generated_audio": "h3_native_generated",
    "ref2va_audio": "h3_ref2va_audio",
    "audio_reference": "h3_ref2va_audio",
    # Historical IAMCCS wording. Preserve its old REF2VA meaning.
    "driven_audio": "h3_ref2va_audio",
    "custom_audio": "h3_custom_audio_drive",
    "custom_audio_drive": "h3_custom_audio_drive",
    "audio_driven": "h3_custom_audio_drive",
    "force_audio_latent": "h3_custom_audio_drive",
    "external_post": "external_audio_post",
    "post_audio": "external_audio_post",
}
H3_AUDIO_MODE_CONTRACTS = {
    "h3_native_generated": {
        "label": "Native H3 generated audio",
        "conditioning": "joint_h3_av_sampling",
        "runtime_resource": "none",
        "final_track": "h3_generated_audio",
    },
    "h3_ref2va_audio": {
        "label": "REF2VA audio reference",
        "conditioning": "minimax_ref_audio_conditioning",
        "runtime_resource": "iamccs_minimax_h3_ref_audio",
        "final_track": "h3_generated_audio",
    },
    "h3_custom_audio_drive": {
        "label": "Custom audio drive / forced AV latent",
        "conditioning": "pre_sampler_audio_latent_replacement",
        "runtime_resource": "iamccs_minimax_h3_custom_audio",
        "final_track": "original_custom_audio_preferred",
    },
    "external_audio_post": {
        "label": "External audio post / no video conditioning",
        "conditioning": "none",
        "runtime_resource": "audioboard_or_explicit_post_audio",
        "final_track": "external_post_mix",
    },
}


def _normalise_h3_audio_mode(value: Any) -> str:
    """Return one of the four R21 audio routes without breaking old boards."""
    mode = str(value or "").strip().lower()
    mode = H3_AUDIO_MODE_ALIASES.get(mode, mode)
    return mode if mode in H3_AUDIO_MODES else "h3_native_generated"


def _is_true_4k_delivery(width: Any, height: Any) -> bool:
    """Return True only for UHD/DCI-class delivery canvases.

    The RTX stage is a final delivery pass, not a generic 2x switch.  Requiring
    both a 4K-class long edge and a cinema-usable short edge also covers
    portrait and scope canvases without treating 2048x1152 as "4K".
    """
    target_width = max(0, int(_finite_float(width, 0, 0)))
    target_height = max(0, int(_finite_float(height, 0, 0)))
    return max(target_width, target_height) >= 3840 and min(target_width, target_height) >= 1600


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
    audio_contract = (
        copy.deepcopy(plan.get("audio_contract"))
        if isinstance(plan.get("audio_contract"), dict)
        else {}
    )

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
        "minimax_h3_audio_contract": audio_contract,
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
        "iamccs_minimax_h3_audio_contract": audio_contract,
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
            "minimax_h3_audio_route": str(plan.get("audio_mode", "h3_native_generated")),
            "audioboard_auto_drives_h3": False,
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
    if int(images.shape[-1]) < 3:
        raise ValueError(f"images deve avere almeno tre canali RGB, shape ricevuta: {tuple(images.shape)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="minimax_h3_segment_") as temp:
        temp_path = Path(temp)
        wav_path = temp_path / "audio.wav"
        has_audio = _write_wav(audio, wav_path)
        height = int(images.shape[1])
        width = int(images.shape[2])
        total_frames = int(images.shape[0])
        exact_duration = total_frames / max(0.001, float(fps))
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", f"{width}x{height}",
            "-framerate", f"{float(fps):.6f}", "-i", "pipe:0",
        ]
        if has_audio:
            command += ["-i", str(wav_path)]
        command += ["-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p"]
        if has_audio:
            # The IMAGE frame count owns programme duration.  ``-shortest``
            # can silently drop final video frames when the generated audio
            # is a few AAC samples shorter, so pad/trim audio to that exact
            # duration instead.
            command += [
                "-af",
                f"aresample=48000:async=1:first_pts=0,apad,atrim=duration={exact_duration:.9f}",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            ]
        command += ["-movflags", "+faststart", str(output)]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            if process.stdin is None:
                raise RuntimeError("ffmpeg raw-video stdin non disponibile")
            for index, frame in enumerate(images):
                rgb = (
                    frame[..., :3]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                    .nan_to_num(nan=0.0, posinf=1.0, neginf=0.0)
                    .clamp_(0.0, 1.0)
                    .mul_(255.0)
                    .round_()
                    .to(dtype=torch.uint8)
                    .contiguous()
                    .numpy()
                )
                process.stdin.write(rgb.tobytes())
                if index == 0 or (index + 1) % 24 == 0 or index + 1 == total_frames:
                    LOG.info("MiniMax H3 streaming video encode | %d/%d frames", index + 1, total_frames)
            process.stdin.close()
            error_bytes = process.stderr.read() if process.stderr is not None else b""
            return_code = process.wait()
        except Exception:
            process.kill()
            process.wait()
            raise
        if return_code != 0:
            error = error_bytes.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg segment encode failed: {error or f'exit {return_code}'}")


def _concat_videos(paths: list[Path], output: Path) -> None:
    """Join independent H3 shots with exact video cuts and one audio master.

    Video remains a stream-copy, so a hard cut cannot turn into a dissolve.
    Audio is decoded per shot, conformed to stereo/48 kHz, given a 20 ms
    equal-power edge fade at internal cuts, concatenated without overlap and
    finally normalised once as a complete programme.  The edge treatment
    removes AAC boundary clicks without shortening the edit or mixing two
    different lines of dialogue together.
    """
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg non trovato: impossibile concatenare i segmenti MiniMax H3")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Segmenti mancanti: " + ", ".join(missing))
    output.parent.mkdir(parents=True, exist_ok=True)
    if len(paths) == 1:
        # A one-shot programme is already the exact native master.  Sending it
        # through the concat demuxer needlessly re-encodes AAC and may expose
        # H.264/AAC encoder priming as a one-frame A/V offset.  Preserve the
        # locked custom-audio timing and the native pixels byte-for-byte.
        shutil.copy2(paths[0], output)
        return
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
        list_path = Path(handle.name)
        for path in paths:
            handle.write("file '" + str(path).replace("'", "'\\''") + "'\n")
    try:
        frame_counts: list[int] = []
        try:
            frame_counts = [_read_segment_frame_count(path) for path in paths]
        except Exception:
            # Legacy segments may pre-date the sidecar metadata.  They still
            # concatenate safely, only without boundary-aware micro-fades.
            frame_counts = []

        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-f", "concat", "-safe", "0", "-i", str(list_path),
        ]
        if frame_counts and len(frame_counts) == len(paths):
            for path in paths:
                command += ["-i", str(path)]

            filters: list[str] = []
            labels: list[str] = []
            edge_seconds = 0.020
            for index, frame_count in enumerate(frame_counts):
                duration = max(1.0 / H3_FPS, float(frame_count) / H3_FPS)
                fade = min(edge_seconds, duration / 4.0)
                chain = (
                    f"[{index + 1}:a]aresample=48000:async=1:first_pts=0,"
                    f"aformat=sample_fmts=fltp:channel_layouts=stereo,"
                    f"atrim=duration={duration:.9f},asetpts=PTS-STARTPTS"
                )
                if index > 0:
                    chain += f",afade=t=in:st=0:d={fade:.9f}:curve=qsin"
                if index + 1 < len(paths):
                    chain += (
                        f",afade=t=out:st={max(0.0, duration - fade):.9f}:"
                        f"d={fade:.9f}:curve=qsin"
                    )
                label = f"acut{index}"
                filters.append(f"{chain}[{label}]")
                labels.append(label)
            audio_inputs = "".join(f"[{label}]" for label in labels)
            exact_duration = sum(frame_counts) / H3_FPS
            filters.append(
                f"{audio_inputs}concat=n={len(labels)}:v=0:a=1,"
                "loudnorm=I=-16:LRA=11:TP=-1.5,"
                f"apad,atrim=duration={exact_duration:.9f}[amaster]"
            )
            command += [
                "-filter_complex", ";".join(filters),
                "-map", "0:v:0", "-map", "[amaster]",
                "-fflags", "+genpts", "-avoid_negative_ts", "make_zero",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                "-t", f"{exact_duration:.9f}", "-movflags", "+faststart", str(output),
            ]
        else:
            command += [
                "-fflags", "+genpts", "-avoid_negative_ts", "make_zero", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", str(output),
            ]
        result = subprocess.run(command, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {result.stderr.strip() or result.stdout.strip()}")
    finally:
        list_path.unlink(missing_ok=True)


def _segment_meta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".iamccs.json")


def _write_segment_metadata(path: Path, frame_count: int, fps: float, audio_join_policy: str = "crossfade") -> None:
    metadata = {
        "schema": "iamccs.minimax_h3.segment",
        "frame_count": max(1, int(frame_count)),
        "fps": float(fps),
        "audio_join_policy": str(audio_join_policy or "crossfade"),
    }
    _segment_meta_path(path).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_segment_frame_count(path: Path) -> int:
    meta_path = _segment_meta_path(path)
    if not meta_path.is_file():
        raise FileNotFoundError(f"Metadati frame mancanti per il concat overlap: {meta_path}")
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    count = int(data.get("frame_count", 0))
    if count < 1:
        raise ValueError(f"Frame count non valido nei metadati: {meta_path}")
    return count


def _read_segment_audio_join_policy(path: Path) -> str:
    try:
        data = json.loads(_segment_meta_path(path).read_text(encoding="utf-8"))
        return str(data.get("audio_join_policy", "crossfade") or "crossfade")
    except Exception:
        return "crossfade"


def _concat_videos_overlap(paths: list[Path], output: Path, overlap_frames: int, fps: float) -> None:
    """WAN-style decoded-frame overlap for independently generated H3 chunks.

    MiniMax H3 exposes only one first and one last keyframe per generation, so
    this is a delivery stitch rather than latent continuation.  Corresponding
    tail/head frames are linearly crossfaded and audio uses a synchronized
    crossfade over the same interval.  The exact H3 boundary alternative is
    the one-frame keyframe cut handled by ``_concat_videos``.
    """
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg non trovato: impossibile concatenare i segmenti MiniMax H3 con overlap")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Segmenti mancanti: " + ", ".join(missing))
    if len(paths) < 2:
        return _concat_videos(paths, output)

    frame_counts = [_read_segment_frame_count(path) for path in paths]
    fps = max(1.0, float(fps))
    requested_overlap = max(1, int(overlap_frames))
    output.parent.mkdir(parents=True, exist_ok=True)

    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n"]
    for path in paths:
        command += ["-i", str(path)]

    overlaps = [
        min(requested_overlap, frame_counts[index] - 1, frame_counts[index + 1] - 1)
        for index in range(len(paths) - 1)
    ]
    if any(overlap < 1 for overlap in overlaps):
        raise ValueError(f"Overlap non valido tra i segmenti: {overlaps}")

    filters: list[str] = []
    video_parts: list[str] = []
    first_prefix_end = frame_counts[0] - overlaps[0]
    filters.append(f"[0:v]trim=end_frame={first_prefix_end},setpts=PTS-STARTPTS[vprefix0]")
    video_parts.append("vprefix0")

    for boundary, overlap in enumerate(overlaps):
        left = boundary
        right = boundary + 1
        left_start = frame_counts[left] - overlap
        filters.append(
            f"[{left}:v]trim=start_frame={left_start},setpts=PTS-STARTPTS[vtail{boundary}]"
        )
        filters.append(
            f"[{right}:v]trim=end_frame={overlap},setpts=PTS-STARTPTS[vhead{boundary}]"
        )
        denominator = overlap + 1
        filters.append(
            f"[vtail{boundary}][vhead{boundary}]"
            f"blend=all_expr='A*(1-(N+1)/{denominator})+B*((N+1)/{denominator})':shortest=1"
            f"[vblend{boundary}]"
        )
        video_parts.append(f"vblend{boundary}")

        if right < len(paths) - 1:
            middle_start = overlap
            middle_end = frame_counts[right] - overlaps[right]
            if middle_end > middle_start:
                filters.append(
                    f"[{right}:v]trim=start_frame={middle_start}:end_frame={middle_end},"
                    f"setpts=PTS-STARTPTS[vbody{right}]"
                )
                video_parts.append(f"vbody{right}")
        else:
            filters.append(
                f"[{right}:v]trim=start_frame={overlap},setpts=PTS-STARTPTS[vsuffix{right}]"
            )
            video_parts.append(f"vsuffix{right}")

    video_inputs = "".join(f"[{label}]" for label in video_parts)
    filters.append(f"{video_inputs}concat=n={len(video_parts)}:v=1:a=0[vjoined]")

    trim_locked_audio = all(
        _read_segment_audio_join_policy(path) == "trim_silent_tail"
        for path in paths
    )
    for index in range(len(paths)):
        filters.append(f"[{index}:a]aresample=48000,asetpts=PTS-STARTPTS[a{index}]")
    accumulated_frames = sum(frame_counts) - sum(overlaps)
    if trim_locked_audio:
        # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        # FLF custom-drive owns waveform timing: discard only the silent
        # visual-overlap tail, never crossfade dialogue performances.
        audio_parts = []
        for index, frame_count in enumerate(frame_counts):
            end_seconds = (frame_count - (overlaps[index] if index < len(overlaps) else 0)) / fps
            label = f"atrim{index}"
            filters.append(f"[a{index}]atrim=duration={end_seconds:.9f}[{label}]")
            audio_parts.append(label)
        audio_label = "ajoined"
        filters.append(f"{''.join(f'[{label}]' for label in audio_parts)}concat=n={len(audio_parts)}:v=0:a=1[{audio_label}]")
    else:
        audio_label = "a0"
        for index, overlap in enumerate(overlaps, start=1):
            duration = overlap / fps
            next_audio = f"ax{index}"
            filters.append(
                f"[{audio_label}][a{index}]acrossfade=d={duration:.9f}:c1=tri:c2=tri[{next_audio}]"
            )
            audio_label = next_audio
    exact_duration = accumulated_frames / fps
    filters.append(
        f"[{audio_label}]loudnorm=I=-16:LRA=11:TP=-1.5,"
        f"apad,atrim=duration={exact_duration:.9f}[amaster]"
    )

    command += [
        "-filter_complex", ";".join(filters),
        "-map", "[vjoined]", "-map", "[amaster]",
        "-frames:v", str(accumulated_frames), "-r", f"{fps:.6f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-t", f"{exact_duration:.9f}",
        "-movflags", "+faststart", str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg overlap concat failed: {result.stderr.strip() or result.stdout.strip()}")
    LOG.info(
        "MiniMax H3 overlap concat complete | chunks=%d | overlap=%df | output_frames=%d | fps=%.3f",
        len(paths), requested_overlap, accumulated_frames, fps,
    )


def _output_location(prefix: str) -> tuple[Path, str]:
    parts = [part for part in str(prefix or "MiniMaxH3/segment").replace("\\", "/").split("/") if part and part not in {".", ".."}]
    if not parts:
        parts = ["MiniMaxH3", "segment"]
    base = Path(folder_paths.get_output_directory())
    folder = base.joinpath(*[_safe_name(part, "output") for part in parts[:-1]])
    folder.mkdir(parents=True, exist_ok=True)
    return folder, _safe_name(parts[-1], "segment")


def _decode_video_master(
    path: Path,
    *,
    fps: float,
    fallback_audio: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Decode the completed native master for the one-film upscale pass.

    PyAV is already a ComfyUI/VHS dependency.  Video is kept as uint8 until
    the final tensor conversion to avoid constructing a list of full-size
    float frames.  Audio is conformed to stereo/48 kHz and sample-aligned to
    the decoded picture duration.
    """
    if not path.is_file():
        raise FileNotFoundError(f"MiniMax H3 native master not found: {path}")
    try:
        import av
    except Exception as exc:
        raise RuntimeError(
            "PyAV is required to load the concatenated MiniMax H3 master before upscale"
        ) from exc

    video_arrays: list[np.ndarray] = []
    target_width: int | None = None
    target_height: int | None = None
    resolution_warning_emitted = False
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"Native master has no video stream: {path}")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if target_width is None or target_height is None:
                target_width, target_height = int(frame.width), int(frame.height)
            elif int(frame.width) != target_width or int(frame.height) != target_height:
                if not resolution_warning_emitted:
                    LOG.warning(
                        "MiniMax H3 master contains a resolution change %dx%d -> %dx%d; conforming to the opening canvas",
                        int(frame.width), int(frame.height), target_width, target_height,
                    )
                    resolution_warning_emitted = True
                frame = frame.reformat(width=target_width, height=target_height, format="rgb24")
            video_arrays.append(frame.to_ndarray(format="rgb24"))
    if not video_arrays:
        raise ValueError(f"Native master decoded zero video frames: {path}")
    packed_video = np.stack(video_arrays, axis=0)
    del video_arrays
    frames = torch.from_numpy(packed_video).to(dtype=torch.float32).div_(255.0)
    del packed_video

    sample_rate = 48000
    audio_arrays: list[np.ndarray] = []
    with av.open(str(path)) as container:
        if container.streams.audio:
            audio_stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="fltp", layout="stereo", rate=sample_rate)
            for frame in container.decode(audio_stream):
                converted = resampler.resample(frame)
                if converted is None:
                    continue
                if not isinstance(converted, list):
                    converted = [converted]
                for audio_frame in converted:
                    values = audio_frame.to_ndarray()
                    if values.ndim == 1:
                        values = values.reshape(1, -1)
                    audio_arrays.append(values.astype(np.float32, copy=False))
            flushed = resampler.resample(None)
            if flushed is not None:
                if not isinstance(flushed, list):
                    flushed = [flushed]
                for audio_frame in flushed:
                    values = audio_frame.to_ndarray()
                    if values.ndim == 1:
                        values = values.reshape(1, -1)
                    audio_arrays.append(values.astype(np.float32, copy=False))

    if audio_arrays:
        packed_audio = np.concatenate(audio_arrays, axis=1)
        waveform = torch.from_numpy(packed_audio).to(dtype=torch.float32).unsqueeze(0)
    elif isinstance(fallback_audio, dict) and torch.is_tensor(fallback_audio.get("waveform")):
        waveform = fallback_audio["waveform"][:1].detach().cpu().float()
        source_rate = int(fallback_audio.get("sample_rate", sample_rate))
        if source_rate != sample_rate:
            import torchaudio.functional as AF

            waveform = AF.resample(waveform, source_rate, sample_rate)
    else:
        waveform = torch.zeros((1, 2, 1), dtype=torch.float32)

    if waveform.shape[1] == 1:
        waveform = waveform.repeat(1, 2, 1)
    elif waveform.shape[1] > 2:
        waveform = waveform[:, :2, :]
    target_samples = max(1, int(round(int(frames.shape[0]) / max(1.0, float(fps)) * sample_rate)))
    if int(waveform.shape[-1]) < target_samples:
        waveform = torch.nn.functional.pad(waveform, (0, target_samples - int(waveform.shape[-1])))
    else:
        waveform = waveform[..., :target_samples]
    audio = {"waveform": waveform.contiguous(), "sample_rate": sample_rate}
    LOG.info(
        "MiniMax H3 native master decoded for one-film upscale | frames=%d | resolution=%dx%d | audio_samples=%d",
        int(frames.shape[0]), int(frames.shape[2]), int(frames.shape[1]), target_samples,
    )
    return frames.contiguous(), audio


def _next_numbered_render_id(output_folder: Path, base_name: str, requested_render_id: str) -> str:
    """Return a never-reused render id with a monotonically increasing suffix.

    Historical unnumbered outputs count as render 0001. Every stage label is
    considered a collision, so retakes never overwrite native, delivery, or
    custom-audio checkpoint files created by a prior render.
    """
    root = _safe_name(requested_render_id, "minimax_h3_render")
    escaped_base = re.escape(_safe_name(base_name, "segment"))
    escaped_root = re.escape(root)
    pattern = re.compile(
        rf"^{escaped_base}_{escaped_root}(?:_(\d{{4,}}))?_(?:[a-z0-9][a-z0-9_-]*_)?(?:full|seg_\d{{4,}})\.mp4$",
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
        safe_base = _safe_name(base_name, "segment")
        stage_collision = any(output_folder.glob(f"{safe_base}_{candidate}_*.mp4"))
        if not stage_collision:
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
                "text_encoder_device": (["auto", "cpu_safe_12gb"], {"default": "auto"}),
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
        text_encoder_device="auto",
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
        requested_text_encoder_device = str(text_encoder_device or "auto").lower()
        text_encoder_device = "auto"
        if requested_text_encoder_device == "cpu_safe_12gb":
            text_encoder_device = "auto(gpu-first; migrated legacy cpu_safe_12gb)"
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
        # The LTX finishing slot intentionally exposes the complete ComfyUI
        # LoRA registry. Some useful LTX LoRAs (including community Crisp
        # variants) do not carry reliable "ltx/detail/enhance" tokens in the
        # filename or parent folder. Runtime compatibility remains the user's
        # choice; keeping the historical field name preserves old workflows.
        installed_ltx_detailer_loras = list(folder_paths.get_filename_list("loras"))
        crisp_ltx_loras = sorted(
            (
                name for name in installed_ltx_detailer_loras
                if "ltx" in name.lower() and "crisp" in name.lower()
            ),
            key=lambda name: (
                0 if Path(name).name.lower() == "ltx2.3_crisp_enhance.safetensors" else 1,
                name.lower(),
            ),
        )
        preferred_ltx_detailer_lora = crisp_ltx_loras[0] if crisp_ltx_loras else ""
        # Only expose files that really exist. The web migration converts old
        # saved missing filenames to the empty/Base-H3 choice before validation.
        turbo_loras = list(dict.fromkeys(("", *installed_turbo_loras)))
        ltx_detailer_loras = list(dict.fromkeys(("", *installed_ltx_detailer_loras)))
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
                "task_mode": (["auto_from_timeline", "t2va", "i2va", "fl2va", "ref2va", "v2va_object_swap"], {"default": "auto_from_timeline"}),
                "audio_mode": (list(H3_AUDIO_MODES), {"default": "h3_native_generated"}),
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
                "acceleration": ([
                    "low_vram_auto", "native", "h3_sage", "sol_low_vram", "sol_adaptive_safe",
                    "sol_adaptive_balanced", "adaptive_safe", "spectrum", "sage_spectrum",
                    "auto_3060", "sage", "sage_sol",
                ], {"default": "low_vram_auto"}),
                "ref_image_size": (["match", "max"], {"default": "match"}),
                "reference_role_1": (["subject_identity", "keyframe", "composition", "style", "disabled"], {"default": "subject_identity"}),
                "reference_role_2": (["subject_identity", "keyframe", "composition", "style", "disabled"], {"default": "subject_identity"}),
                "reference_role_3": (["subject_identity", "keyframe", "composition", "style", "disabled"], {"default": "composition"}),
                "reference_role_4": (["subject_identity", "keyframe", "composition", "style", "disabled"], {"default": "style"}),
                "reference_video_role": (["off", "motion_camera", "temporal_structure", "video_edit", "continuation"], {"default": "off"}),
                "reference_audio_role": (["off", "voice_timbre", "rhythm_timing", "audio_reuse", "sound_reference"], {"default": "off"}),
                "sol_conditioning": (["exact_kv_and_rows", "exact_kv"], {"default": "exact_kv_and_rows"}),
                "spectrum_profile": (["low_vram", "quality", "aggressive", "conservative_3060", "conservative_quality"], {"default": "low_vram"}),
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
                # ComfyUI automatic placement is GPU-first. The atomic backend
                # retries on CPU only after a genuine CUDA out-of-memory error.
                "text_encoder_device": (["auto", "cpu_safe_12gb"], {"default": "auto"}),
                # These values are intentionally owned by the Shotboard.  The
                # generation node keeps legacy widgets only as a compatibility
                # fallback for shotplans created before schema v3.
                "performance_profile": ([
                    "low_vram_draft", "low_vram_balanced", "low_vram_turbo", "h3_turbo_quality", "h3_native_quality", "custom",
                    "rtx3060_draft", "rtx3060_balanced", "rtx3060_turbo",
                ], {"default": "low_vram_balanced"}),
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
                # Preserve source pixels by default. The native H3 conditioner
                # performs the single required Lanczos fit to the latent canvas.
                # Pre-resize remains available as an explicit Low VRAM option.
                "reference_resize_policy": (["canvas_crop", "canvas_pad", "total_pixels", "off"], {"default": "off"}),
                "reference_resize_megapixels": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.05}),
                "reference_resize_filter": (["area", "bilinear", "bicubic", "nearest-exact"], {"default": "area"}),
                # Optional LTX finishing controls. All installed LoRAs remain
                # selectable; an installed LTX Crisp variant is preselected.
                # The enable switch still owns whether the LoRA is applied.
                "ltx_detailer_enabled": ("BOOLEAN", {"default": False}),
                "ltx_detailer_lora_name": (
                    ltx_detailer_loras,
                    {"default": preferred_ltx_detailer_lora},
                ),
                "ltx_detailer_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
                "ltx_4k_enabled": ("BOOLEAN", {"default": False}),
                "ltx_4k_quality": (["ULTRA", "HIGH", "MEDIUM", "LOW"], {"default": "ULTRA"}),
                "ltx_seam_safe": ("BOOLEAN", {"default": True}),
                "flf_join_mode": (["h3_keyframe_cut", "wan_overlap_blend"], {"default": "h3_keyframe_cut"}),
                "flf_overlap_frames": ("INT", {"default": 9, "min": 1, "max": 24, "step": 1}),
                # The LTX temporal looper is the delivery sampler for long
                # masters. These settings are Shotboard truth, not workflow
                # constants, so a saved board reproduces its VRAM strategy.
                "ltx_looper_temporal_tile_size": ("INT", {"default": 80, "min": 24, "max": 1000, "step": 8}),
                "ltx_looper_temporal_overlap": ("INT", {"default": 24, "min": 16, "max": 80, "step": 8}),
                "ltx_looper_guiding_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ltx_looper_overlap_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ltx_looper_cond_image_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ltx_looper_horizontal_tiles": ("INT", {"default": 1, "min": 1, "max": 6, "step": 1}),
                "ltx_looper_vertical_tiles": ("INT", {"default": 1, "min": 1, "max": 6, "step": 1}),
                "ltx_looper_spatial_overlap": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
            },
            "optional": {
                # R22 V2V policies are optional for API backward compatibility.
                # Source tensors and their true input FPS remain in CineInfoH3V2V;
                # Shotboard owns only how each timeline segment addresses them.
                "v2v_guide_mode": (["", "raw_only", "raw_pose", "raw_depth", "raw_depth_pose"], {"default": "raw_only"}),
                "v2v_source_range_policy": (["", "timeline_segment", "sequential_requested", "repeat_from_offset"], {"default": "timeline_segment"}),
                "v2v_source_offset_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.01}),
                "v2v_source_fit": (["", "native_adapt", "canvas_pad", "canvas_crop", "stretch"], {"default": "canvas_pad"}),
                "v2v_source_end_policy": (["", "hold_last_for_grid", "error"], {"default": "hold_last_for_grid"}),
                "v2v_audio_pairing": (["", "pair_with_source_video", "standalone_reference", "off"], {"default": "pair_with_source_video"}),
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
        acceleration="low_vram_auto",
        ref_image_size="match",
        reference_role_1="subject_identity",
        reference_role_2="subject_identity",
        reference_role_3="composition",
        reference_role_4="style",
        reference_video_role="off",
        reference_audio_role="off",
        sol_conditioning="exact_kv_and_rows",
        spectrum_profile="low_vram",
        vram_clean_before_decode=True,
        rife_mode="off",
        upscale_enabled=False,
        upscale_width=1920,
        upscale_height=1080,
        upscale_prompt="",
        upscale_sage=True,
        upscale_seed_offset=10000,
        wan_upscale_denoise=0.2,
        text_encoder_device="auto",
        performance_profile="low_vram_balanced",
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
        reference_resize_policy="off",
        reference_resize_megapixels=0.5,
        reference_resize_filter="area",
        ltx_detailer_enabled=False,
        ltx_detailer_lora_name="",
        ltx_detailer_strength=0.6,
        ltx_4k_enabled=False,
        ltx_4k_quality="ULTRA",
        ltx_seam_safe=True,
        flf_join_mode="h3_keyframe_cut",
        flf_overlap_frames=9,
        ltx_looper_temporal_tile_size=80,
        ltx_looper_temporal_overlap=24,
        ltx_looper_guiding_strength=1.0,
        ltx_looper_overlap_strength=0.5,
        ltx_looper_cond_image_strength=1.0,
        ltx_looper_horizontal_tiles=1,
        ltx_looper_vertical_tiles=1,
        ltx_looper_spatial_overlap=1,
        v2v_guide_mode="raw_only",
        v2v_source_range_policy="timeline_segment",
        v2v_source_offset_seconds=0.0,
        v2v_source_fit="canvas_pad",
        v2v_source_end_policy="hold_last_for_grid",
        v2v_audio_pairing="pair_with_source_video",
        cine_linx=None,
    ):
        global_prompt, timeline_data, prompter_injection = apply_prompter_to_minimax(
            cine_linx,
            global_prompt,
            timeline_data,
        )
        audio_mode = _normalise_h3_audio_mode(audio_mode)
        v2v_guide_mode = str(v2v_guide_mode or "raw_only")
        if v2v_guide_mode not in {"raw_only", "raw_pose", "raw_depth", "raw_depth_pose"}:
            v2v_guide_mode = "raw_only"
        v2v_source_range_policy = str(v2v_source_range_policy or "timeline_segment")
        if v2v_source_range_policy not in {"timeline_segment", "sequential_requested", "repeat_from_offset"}:
            v2v_source_range_policy = "timeline_segment"
        v2v_source_fit = str(v2v_source_fit or "canvas_pad")
        if v2v_source_fit not in {"native_adapt", "canvas_pad", "canvas_crop", "stretch"}:
            v2v_source_fit = "canvas_pad"
        v2v_source_end_policy = str(v2v_source_end_policy or "hold_last_for_grid")
        if v2v_source_end_policy not in {"hold_last_for_grid", "error"}:
            v2v_source_end_policy = "hold_last_for_grid"
        v2v_audio_pairing = str(v2v_audio_pairing or "pair_with_source_video")
        if v2v_audio_pairing not in {"pair_with_source_video", "standalone_reference", "off"}:
            v2v_audio_pairing = "pair_with_source_video"
        width = _h3_legal_dimension(width, 960)
        height = _h3_legal_dimension(height, 544)
        image_width = _h3_legal_dimension(image_width, width)
        image_height = _h3_legal_dimension(image_height, height)
        reference_resize_megapixels = _finite_float(reference_resize_megapixels, 0.5, 0.1, 2.0)
        selected_ltx_detailer = str(ltx_detailer_lora_name or "").strip()
        ltx_detailer_requested = bool(ltx_detailer_enabled)
        ltx_detailer_available = _model_file_available("loras", selected_ltx_detailer)
        effective_ltx_detailer = ltx_detailer_requested and ltx_detailer_available
        upscale_target_width = max(256, int(_finite_float(upscale_width, width * 2, 256, 7680)))
        upscale_target_height = max(256, int(_finite_float(upscale_height, height * 2, 256, 4320)))
        ltx_4k_requested = bool(ltx_4k_enabled)
        true_4k_target = _is_true_4k_delivery(upscale_target_width, upscale_target_height)
        effective_ltx_4k = (
            ltx_4k_requested
            and bool(upscale_enabled)
            and str(upscale_mode) == "ltx23"
            and true_4k_target
        )
        ltx_4k_quality = str(ltx_4k_quality or "ULTRA").upper()
        if ltx_4k_quality not in {"ULTRA", "HIGH", "MEDIUM", "LOW"}:
            ltx_4k_quality = "ULTRA"

        requested_turbo_mode = str(turbo_mode or "off")
        selected_turbo_lora = str(turbo_lora_name or "").strip()
        turbo_requested = requested_turbo_mode != "off"
        turbo_available = _model_file_available("loras", selected_turbo_lora)
        effective_turbo_mode = requested_turbo_mode if turbo_requested and turbo_available else "off"
        requested_steps = max(1, int(_finite_float(steps, 16, 1, 100)))
        # A missing optional Turbo file must not leave a base-model generation
        # running with an unsafe 6-10 step Turbo schedule.  Conversely, valid
        # Turbo profiles are clamped to their proven minimum so an imported
        # stale 4-step widget cannot silently degrade a quality render.
        if turbo_requested and not turbo_available:
            effective_steps = max(16, requested_steps)
        elif effective_turbo_mode == "early_8_10":
            effective_steps = max(8, requested_steps)
        elif effective_turbo_mode == "ckpt500_6_8":
            effective_steps = max(6, requested_steps)
        else:
            effective_steps = requested_steps
        plan = build_shotplan(
            timeline_data=timeline_data,
            global_prompt=global_prompt,
            duration_seconds=duration_seconds,
            task_mode=task_mode,
            audio_mode=audio_mode,
            prompt_mapping=prompt_mapping,
            flf_join_mode=flf_join_mode,
            flf_overlap_frames=flf_overlap_frames,
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
        plan["audio_mode"] = audio_mode
        plan["audio_contract"] = {
            "schema": "iamccs.minimax_h3.audio_route",
            "schema_version": 1,
            "selected_mode": audio_mode,
            **copy.deepcopy(H3_AUDIO_MODE_CONTRACTS[audio_mode]),
            "reference_audio_resource": "iamccs_minimax_h3_ref_audio",
            "custom_audio_resource": "iamccs_minimax_h3_custom_audio",
            "timeline_audio_contract": "AudioBoard lanes remain editorial/post metadata until explicitly rendered or published as AUDIO",
            "automatic_audioboard_drive": False,
            "source": "shotboard",
        }
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
        effective_v2v_audio_pairing = (
            "pair_with_source_video"
            if audio_mode == "h3_ref2va_audio"
            else "off"
        )
        plan["v2v_settings"] = {
            "schema": "iamccs.minimax_h3.v2v.settings",
            "schema_version": 1,
            "guide_mode": str(v2v_guide_mode),
            "source_range_policy": str(v2v_source_range_policy),
            "source_offset_seconds": float(_finite_float(v2v_source_offset_seconds, 0.0, 0.0, 86400.0)),
            "source_fit": str(v2v_source_fit),
            "source_end_policy": str(v2v_source_end_policy),
            "audio_behavior": audio_mode,
            "audio_pairing": effective_v2v_audio_pairing,
            "requested_audio_pairing": str(v2v_audio_pairing),
            "source_media": "CineInfoH3V2V",
            "model_family": "ref2va",
            "source": "shotboard",
        }
        plan["upscale_settings"] = {
            "target_width": upscale_target_width,
            "target_height": upscale_target_height,
            "prompt": str(upscale_prompt or "").strip(),
            "sage": bool(upscale_sage),
            "seed_offset": int(upscale_seed_offset),
            "wan_denoise": float(wan_upscale_denoise),
            "ltx_detailer_requested": ltx_detailer_requested,
            "ltx_detailer_enabled": effective_ltx_detailer,
            "ltx_detailer_available": ltx_detailer_available,
            "ltx_detailer_lora_name": selected_ltx_detailer,
            "ltx_detailer_strength": float(ltx_detailer_strength),
            "ltx_4k_requested": ltx_4k_requested,
            "ltx_4k_target_valid": true_4k_target,
            "ltx_4k_enabled": effective_ltx_4k,
            "ltx_4k_quality": ltx_4k_quality,
            "ltx_seam_safe": bool(ltx_seam_safe),
            "ltx_vae_encode_temporal_size": 500 if bool(ltx_seam_safe) else 64,
            "ltx_vae_encode_temporal_overlap": 4 if bool(ltx_seam_safe) else 8,
            "ltx_vae_decode_temporal_size": 64 if bool(ltx_seam_safe) else 16,
            "ltx_vae_decode_temporal_overlap": 4 if bool(ltx_seam_safe) else 1,
            "ltx_vae_decode_spatial_overlap": 4 if bool(ltx_seam_safe) else 1,
            "ltx_looper": {
                "sampler": "LTXVLoopingSampler",
                "temporal_tile_size": max(24, min(1000, int(ltx_looper_temporal_tile_size))),
                "temporal_overlap": max(16, min(80, int(ltx_looper_temporal_overlap))),
                "guiding_strength": max(0.0, min(1.0, float(ltx_looper_guiding_strength))),
                "overlap_strength": max(0.0, min(1.0, float(ltx_looper_overlap_strength))),
                "cond_image_strength": max(0.0, min(1.0, float(ltx_looper_cond_image_strength))),
                "horizontal_tiles": max(1, min(6, int(ltx_looper_horizontal_tiles))),
                "vertical_tiles": max(1, min(6, int(ltx_looper_vertical_tiles))),
                "spatial_overlap": max(1, min(8, int(ltx_looper_spatial_overlap))),
                "source": "shotboard",
            },
            "source": "shotboard",
        }
        chunk_frames = [int(chunk.get("frame_count", 0) or 0) for chunk in plan.get("chunks", [])]
        max_chunk_frames = max(chunk_frames, default=0)
        native_load = (float(width) * float(height) * max(1, max_chunk_frames)) / (960.0 * 544.0 * 124.0)
        warnings: list[str] = []
        if ltx_detailer_requested and not ltx_detailer_available:
            warnings.append(f"Optional LTX detailer unavailable ({selected_ltx_detailer or 'no LoRA selected'}); continuing without it")
        if ltx_4k_requested and not effective_ltx_4k:
            if not true_4k_target:
                warnings.append(
                    f"RTX VSR 4K ignored for {upscale_target_width}x{upscale_target_height}; "
                    "ordinary 2x delivery stays on the direct LTX path"
                )
            else:
                warnings.append("RTX VSR 4K is available only when LTX 2.3 upscale is enabled")
        if effective_ltx_4k:
            warnings.append(
                "4K delivery uses a protected LTX intermediate that never downsamples the native H3 frames, "
                "then NVIDIA RTX VSR reaches the final canvas"
            )
        if effective_steps != requested_steps:
            warnings.append(f"Sampling steps corrected from {requested_steps} to {effective_steps} for the selected Turbo/base contract")
        if str(text_encoder_device).lower() == "cpu_safe_12gb":
            warnings.append("Legacy CPU-safe text encoder setting migrated to GPU-first auto with CPU fallback only after CUDA OOM")
        low_vram_profile = str(performance_profile).startswith(("low_vram", "rtx3060"))
        if low_vram_profile and max_chunk_frames > 124:
            warnings.append("Low VRAM: trim this timeline box to 124 frames or less; use a following box for continuation")
        if low_vram_profile and int(width) * int(height) > 960 * 544:
            warnings.append("Low VRAM: generate at 960x544 or below, then upscale for a 1280-class delivery")
        if str(acceleration) in {"sage_sol", "sol_low_vram", "sol_adaptive_safe", "sol_adaptive_balanced"}:
            warnings.append("Sol-Attn is experimental, has a slower first compile, and is not validated for every Low VRAM configuration")
        if str(acceleration) in {"spectrum", "sage_spectrum"} and effective_steps < 14:
            warnings.append("Spectrum saves few transformer calls below 14 steps because warmup and final native steps remain mandatory")
        turbo_enabled = effective_turbo_mode != "off"
        if turbo_requested and not turbo_available:
            missing_name = selected_turbo_lora or "no LoRA selected"
            warnings.append(f"Optional Turbo LoRA unavailable ({missing_name}); using base H3 at {effective_steps} steps")
        lightx2v_turbo = "lightx2v" in selected_turbo_lora.lower()
        if turbo_enabled and lightx2v_turbo and not 0.6 <= float(turbo_strength) <= 1.0:
            warnings.append("Lightx2v Turbo strength is outside its tested 0.6-1.0 quality range")
        elif turbo_enabled and not lightx2v_turbo and not 0.8 <= float(turbo_strength) <= 1.8:
            warnings.append("Turbo LoRA strength is outside the tested 0.8-1.8 range")
        if effective_turbo_mode == "early_8_10" and not 8 <= effective_steps <= 10:
            warnings.append("Early/non-ckpt500 Turbo is normally used at 8-10 steps")
        if effective_turbo_mode == "ckpt500_6_8" and not 6 <= effective_steps <= 8:
            warnings.append("Turbo ckpt500 is normally used at 6-8 steps")
        if str(acceleration) in {"adaptive_safe", "sol_adaptive_safe", "sol_adaptive_balanced"}:
            warnings.append("Adaptive Cache is approximate; use Safe for faces, hands, dialogue and lip sync")
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
            "model_route": ["task_mode", "reference roles", "V2VA Object Swap -> REF2VA"],
            "conditioning": ["width", "height", "timeline trim", "prompt mapping", "references", "four-way audio route"],
            "sampling": ["seed", "steps", "sampler", "scheduler", "denoise", "H3 shifts", "acceleration", "Turbo LoRA", "Turbo audio sampler"],
            "reference_preprocess": ["resize policy", "target megapixels", "filter", "multiple of 32"],
            "delivery": ["FLF join mode", "FLF overlap frames", "VRAM clean", "RIFE", "upscale enabled", "upscale mode", "upscale target", "upscale prompt", "upscale seed", "LTX seam-safe VAE", "LTX detailer LoRA", "optional RTX VSR 4K"],
            "audio": {
                "selected": audio_mode,
                "native": "joint H3 AV generation",
                "reference": "REF2VA conditioning through CineInfoH3 reference_audio",
                "custom_drive": "distinct CineInfoH3 custom_audio runtime tensor for pre-sampler AV latent replacement",
                "external_post": "post-generation soundtrack; never implied to condition lips or motion",
                "audioboard": "editorial lanes only unless explicitly rendered/published as AUDIO",
            },
            "transport": "one IAMCCS_SUPERNODE_LINX cable; the private H3 plan stays inside CineLinX",
            "v2v": {
                "source_media": "CineInfoH3V2V outside the timeline",
                "shotboard_owns": "prompt, duration and source ranges",
                "guide_mode": str(v2v_guide_mode),
                "source_range_policy": str(v2v_source_range_policy),
                "raw_source_occurrences": 1,
            },
        }
        injection_summary = str(prompter_injection.get("actual_target", "none")) if prompter_injection.get("applied") else "none"
        report = (
            f"MiniMax H3 Shotboard | shots={len(plan['slots'])} | keyframes={plan['total_keyframes']} | chunks={plan['total_segments']} | "
            f"duration={plan['effective_duration_seconds']:.3f}s | grid=17k+5 @24fps | "
            f"task={task_mode} | frames=timeline trim (max 362 per box) | prompt={prompt_mapping} | "
            f"v2v={v2v_guide_mode}/{v2v_source_range_policy}+{float(v2v_source_offset_seconds):.2f}s/"
            f"{v2v_source_fit}/{audio_mode}->{effective_v2v_audio_pairing} | "
            f"join={plan.get('flf_join_mode')}:{plan.get('flf_overlap_frames')}f | "
            f"audio={audio_mode} | resolution={width}x{height} | performance={performance_profile} "
            f"load={native_load:.2f}x | sampler={effective_steps}x{sampler_name}+{scheduler} | acceleration={acceleration} | "
            f"turbo={effective_turbo_mode}:{selected_turbo_lora or 'none'}@{float(turbo_strength):.2f}/{turbo_sampler_mode} | "
            f"ref_resize={reference_resize_policy}:{reference_resize_megapixels:.2f}MP/{reference_resize_filter} | "
            f"ref_size={ref_image_size} | text_encoder={plan.get('text_encoder_device', 'auto')} | "
            f"RIFE={rife_mode} | upscale={'on' if upscale_enabled else 'off'}:{plan['upscale_mode']} "
            f"->{upscale_target_width}x{upscale_target_height} sage={'on' if upscale_sage else 'off'} "
            f"ltx_detailer={'on' if effective_ltx_detailer else 'off'}:{selected_ltx_detailer or 'none'}@{float(ltx_detailer_strength):.2f} "
            f"ltx_seam_safe={'on' if ltx_seam_safe else 'off'} ltx_4k={'on' if effective_ltx_4k else 'off'}:{ltx_4k_quality} "
            f"ltx_looper={int(ltx_looper_temporal_tile_size)}/{int(ltx_looper_temporal_overlap)} "
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
        # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        prompt = str(chunk.get("prompt", "")) or str(prompt_override or "").strip()

        planned_first = _load_image(str(chunk.get("first_image", "")))
        planned_last = _load_image(str(chunk.get("last_image", "")))
        first = planned_first if planned_first is not None else (first_frame_override if torch.is_tensor(first_frame_override) else None)
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
        last = planned_last if planned_last is not None else (last_frame_override if torch.is_tensor(last_frame_override) else None)

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


class IAMCCS_MiniMaxH3NativeCheckpointSave:
    """Persist the native H3 result before any optional upscale branch.

    The node is deliberately a pass-through dependency.  Downstream LTX, Wan,
    or RTX processing cannot begin until the native segment has been encoded,
    so an upscale failure never discards the expensive H3 render.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "current_segment": ("INT", {"forceInput": True}),
                "total_segments": ("INT", {"forceInput": True}),
                "fps": ("INT", {"forceInput": True}),
                "trim_head_frames": ("INT", {"forceInput": True}),
            },
            "optional": {
                "bridge_images": ("IMAGE",),
                "resolved_render_id": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "IAMCCS/MiniMaxH3/segment"}),
                "merge_segments": ("BOOLEAN", {"default": True}),
                "keep_segments": ("BOOLEAN", {"default": True}),
                "render_id": ("STRING", {"default": "minimax_h3_render"}),
                "queue_next_segment": ("BOOLEAN", {"default": False}),
                "stage_label": ("STRING", {"default": "native"}),
                "sampled_latent": ("LATENT",),
                "motion_context": ("IAMCCS_H3_MOTION_CONTEXT",),
                "motion_state": ("IAMCCS_H3_MOTION_CONTEXT",),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("native_frames", "native_audio", "resolved_render_id", "report")
    FUNCTION = "checkpoint"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Saving is intentional on every queued render, even when ComfyUI can
        # reuse the surrounding graph cache.
        return float("nan")

    def checkpoint(
        self,
        images,
        audio,
        current_segment,
        total_segments,
        fps,
        trim_head_frames,
        bridge_images=None,
        resolved_render_id="",
        filename_prefix="IAMCCS/MiniMaxH3/segment",
        merge_segments=True,
        keep_segments=True,
        render_id="minimax_h3_render",
        queue_next_segment=False,
        stage_label="native",
        sampled_latent=None,
        motion_context=None,
        motion_state=None,
        prompt=None,
        unique_id=None,
        extra_pnginfo=None,
    ):
        if not torch.is_tensor(images) or images.ndim != 4 or int(images.shape[0]) < 1:
            raise ValueError("MiniMax H3 native checkpoint expects a non-empty IMAGE frame batch")
        if not isinstance(audio, dict):
            raise ValueError("MiniMax H3 native checkpoint expects the H3 AUDIO output")

        current_segment = int(current_segment)
        total_segments = int(total_segments)
        fps = max(1, int(fps))
        if current_segment < 0 or total_segments < 1 or current_segment >= total_segments:
            raise ValueError(f"Native checkpoint segment index is invalid: {current_segment + 1}/{total_segments}")

        output_folder, base_name = _output_location(filename_prefix)
        requested_render_id = _safe_name(str(render_id or "").strip(), "minimax_h3_render")
        safe_stage_label = _safe_name(str(stage_label or "native").strip(), "native")
        locked_render_id = str(resolved_render_id or "").strip()
        if locked_render_id:
            active_render_id = _safe_name(locked_render_id, requested_render_id)
        else:
            active_render_id = (
                _next_numbered_render_id(output_folder, base_name, requested_render_id)
                if current_segment == 0
                else requested_render_id
            )

        trim_count = max(0, int(trim_head_frames or 0))
        overlap_stitch = trim_count > 1
        images_to_save = images
        audio_to_save = audio
        if trim_count == 1 and int(images.shape[0]) > trim_count:
            images_to_save = images[trim_count:, ...]
            audio_to_save = _trim_audio_frames(audio, trim_count, fps)

        segment_name = f"{base_name}_{active_render_id}_{safe_stage_label}_seg_{current_segment + 1:04d}.mp4"
        segment_path = output_folder / segment_name
        _require_new_output_path(segment_path)
        _encode_images(images_to_save, audio_to_save, fps, segment_path)
        audio_join_policy = "trim_silent_tail" if bool(audio.get("iamccs_flf_locked_audio_handles", False)) else "crossfade"
        _write_segment_metadata(segment_path, int(images_to_save.shape[0]), fps, audio_join_policy)

        bridge_source = bridge_images if torch.is_tensor(bridge_images) else images
        if torch.is_tensor(bridge_source) and int(bridge_source.shape[0]) > 0:
            bridge_path = _bridge_path(active_render_id)
            _save_frame(bridge_path, bridge_source[-1])
            LOG.info("MiniMax H3 last-frame bridge salvato in %s", bridge_path)

        motion_saved = False
        if isinstance(motion_context, dict) and bool(motion_context.get("enabled")) and isinstance(sampled_latent, dict):
            try:
                from .iamccs_minimax_h3_continuity import save_sampled_av
                motion_saved = save_sampled_av(
                    active_render_id,
                    sampled_latent,
                    int(motion_state.get("trim_frames", 0)) if isinstance(motion_state, dict) else 0,
                    int(images.shape[0]),
                )
            except Exception as exc:
                LOG.warning("MiniMax H3 temporal context cache save skipped: %s", exc)

        messages = [f"{safe_stage_label.upper()} checkpoint saved: {segment_name}"]
        preview_path = segment_path
        if current_segment + 1 >= total_segments and bool(merge_segments):
            segment_paths = [
                output_folder / f"{base_name}_{active_render_id}_{safe_stage_label}_seg_{index + 1:04d}.mp4"
                for index in range(total_segments)
            ]
            final_name = f"{base_name}_{active_render_id}_{safe_stage_label}_full.mp4"
            final_path = output_folder / final_name
            _require_new_output_path(final_path)
            if overlap_stitch:
                _concat_videos_overlap(segment_paths, final_path, trim_count, fps)
            else:
                _concat_videos(segment_paths, final_path)
            preview_path = final_path
            messages.append(f"{safe_stage_label.upper()} full video saved: {final_name}")
            if not bool(keep_segments):
                for path in segment_paths:
                    path.unlink(missing_ok=True)
                    _segment_meta_path(path).unlink(missing_ok=True)

        next_segment = current_segment + 1
        if bool(queue_next_segment) and next_segment < total_segments:
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
            updated_segment_nodes = 0
            checkpoint_updated = False
            for node_id, node in prompt_copy.items():
                inputs = node.setdefault("inputs", {})
                if "segment_index" in inputs:
                    inputs["segment_index"] = next_segment
                    updated_segment_nodes += 1

                is_current_checkpoint = unique_id is not None and node_id == str(unique_id)
                if is_current_checkpoint:
                    inputs["render_id"] = active_render_id
                    checkpoint_updated = True
                elif "render_id" in inputs and not isinstance(inputs.get("render_id"), list):
                    inputs["render_id"] = active_render_id
                if node.get("class_type") == "IAMCCS_MiniMaxH3AtomicConditioningBackend":
                    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
                    # The queued FLF chunk resolves its predecessor bridge by
                    # stable render id without creating a graph cycle.
                    inputs["render_id"] = active_render_id

            if updated_segment_nodes == 0:
                raise ValueError("Il checkpoint MiniMax H3 non trova nessun nodo con input segment_index")
            if not checkpoint_updated:
                raise ValueError("Il checkpoint MiniMax H3 non riesce ad aggiornare il proprio render_id")

            _enqueue(prompt_copy, extra_data=live_extra, outputs=live_outputs, sensitive=live_sensitive)
            messages.append(f"Queued native segment {next_segment + 1}/{total_segments} before upscale")

        LOG.info(
            "MiniMax H3 %s checkpoint complete | render=%s | segment=%d/%d | fps=%d | join=%s:%df | motion_context=%s | queued_next=%s",
            safe_stage_label,
            active_render_id,
            current_segment + 1,
            total_segments,
            fps,
            "overlap_blend" if overlap_stitch else ("keyframe_cut" if trim_count == 1 else "cut"),
            trim_count,
            "saved" if motion_saved else "off",
            bool(queue_next_segment) and next_segment < total_segments,
        )
        subfolder = os.path.relpath(
            preview_path.parent,
            folder_paths.get_output_directory(),
        ).replace("\\", "/")
        preview = {
            "filename": preview_path.name,
            "subfolder": "" if subfolder == "." else subfolder,
            "type": "output",
        }
        report = " | ".join(messages)
        return {
            "ui": {
                "text": messages,
                "images": [preview],
                "animated": (True,),
            },
            "result": (images, audio, active_render_id, report),
        }


class IAMCCS_MiniMaxH3NativeMasterLoad:
    """Expose the completed native movie only after the last H3 loop pass.

    On intermediate passes this is a cheap pass-through used only to expose
    ``master_ready=False``.  On the final pass it loads the already stitched
    native movie, so downstream LTX/Wan sees one programme rather than the
    last generated shot or an interleaved H3/upscale sequence.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "current_frames": ("IMAGE",),
                "current_audio": ("AUDIO",),
                "resolved_render_id": ("STRING", {"forceInput": True}),
                "current_segment": ("INT", {"forceInput": True}),
                "total_segments": ("INT", {"forceInput": True}),
                "fps": ("INT", {"forceInput": True}),
            },
            "optional": {
                "filename_prefix": ("STRING", {"default": "IAMCCS/MiniMaxH3/segment"}),
                "stage_label": ("STRING", {"default": "native"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "BOOLEAN", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "master_frames",
        "master_audio",
        "master_ready",
        "master_path",
        "resolved_render_id",
        "report",
    )
    FUNCTION = "load"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    def load(
        self,
        current_frames,
        current_audio,
        resolved_render_id,
        current_segment,
        total_segments,
        fps,
        filename_prefix="IAMCCS/MiniMaxH3/segment",
        stage_label="native",
    ):
        current_segment = int(current_segment)
        total_segments = max(1, int(total_segments))
        active_render_id = _safe_name(str(resolved_render_id or "").strip(), "minimax_h3_render")
        if current_segment + 1 < total_segments:
            report = (
                f"Native master waiting | completed={current_segment + 1}/{total_segments} | "
                "upscale branch locked"
            )
            return current_frames, current_audio, False, "", active_render_id, report

        output_folder, base_name = _output_location(filename_prefix)
        safe_stage = _safe_name(str(stage_label or "native").strip(), "native")
        master_path = output_folder / f"{base_name}_{active_render_id}_{safe_stage}_full.mp4"
        master_frames, master_audio = _decode_video_master(
            master_path,
            fps=max(1, int(fps)),
            fallback_audio=current_audio,
        )
        report = (
            f"Native master ready | segments={total_segments} | frames={int(master_frames.shape[0])} | "
            f"path={master_path.name} | one-film upscale unlocked"
        )
        return master_frames, master_audio, True, str(master_path), active_render_id, report


class IAMCCS_MiniMaxH3FinalMasterSave:
    """Save exactly one delivery movie after native concat and optional upscale."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "master_ready": ("BOOLEAN", {"forceInput": True}),
                "master_path": ("STRING", {"forceInput": True}),
                "resolved_render_id": ("STRING", {"forceInput": True}),
                "fps": ("INT", {"forceInput": True}),
            },
            "optional": {
                "images": ("IMAGE", {"lazy": True}),
                "audio": ("AUDIO", {"lazy": True}),
                "filename_prefix": ("STRING", {"default": "IAMCCS/MiniMaxH3/final"}),
                "stage_label": ("STRING", {"default": "delivery"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    def check_lazy_status(
        self,
        master_ready,
        master_path,
        resolved_render_id,
        fps,
        images=None,
        audio=None,
        **kwargs,
    ):
        if not bool(master_ready):
            return []
        missing = []
        if images is None:
            missing.append("images")
        if audio is None:
            missing.append("audio")
        return missing

    def save(
        self,
        master_ready,
        master_path,
        resolved_render_id,
        fps,
        images=None,
        audio=None,
        filename_prefix="IAMCCS/MiniMaxH3/final",
        stage_label="delivery",
    ):
        if not bool(master_ready):
            return {
                "ui": {
                    "text": [
                        "MiniMax H3 native segment preview saved; waiting for the remaining segments before upscale"
                    ]
                }
            }
        if not torch.is_tensor(images) or images.ndim != 4 or int(images.shape[0]) < 1:
            raise ValueError("Final MiniMax H3 master save requires delivery IMAGE frames")
        if not isinstance(audio, dict) or not torch.is_tensor(audio.get("waveform")):
            raise ValueError("Final MiniMax H3 master save requires the concatenated AUDIO master")

        output_folder, base_name = _output_location(filename_prefix)
        active_render_id = _safe_name(str(resolved_render_id or "").strip(), "minimax_h3_render")
        safe_stage = _safe_name(str(stage_label or "delivery").strip(), "delivery")
        output_path = output_folder / f"{base_name}_{active_render_id}_{safe_stage}_full.mp4"
        _require_new_output_path(output_path)
        _encode_images(images, audio, max(1, int(fps)), output_path)
        subfolder = os.path.relpath(output_path.parent, folder_paths.get_output_directory()).replace("\\", "/")
        preview = {
            "filename": output_path.name,
            "subfolder": "" if subfolder == "." else subfolder,
            "type": "output",
        }
        message = (
            f"Final MiniMax H3 film saved after native concat and one-film upscale: {output_path.name} | "
            f"frames={int(images.shape[0])} | native_master={Path(master_path).name}"
        )
        LOG.info(message)
        return {"ui": {"text": [message], "images": [preview], "animated": (True,)}}


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
                "resolved_render_id": ("STRING", {"forceInput": True}),
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
        resolved_render_id="",
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
        locked_render_id = str(resolved_render_id or "").strip()
        if locked_render_id:
            active_render_id = _safe_name(locked_render_id, requested_render_id)
        else:
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
        overlap_stitch = trim_count > 1
        if trim_count == 1 and torch.is_tensor(images) and int(images.shape[0]) > trim_count:
            images_to_save = images[trim_count:, ...]
            audio_to_save = _trim_audio_frames(audio, trim_count, H3_FPS)

        if video is not None:
            if overlap_stitch:
                raise ValueError("Il concat H3 con overlap richiede l'input images, non il VIDEO opaco")
            try:
                from comfy_api.latest import Types

                video.save_to(str(segment_path), format=Types.VideoContainer("auto"), codec="auto", metadata=None)
            except Exception as exc:
                raise RuntimeError(f"Salvataggio VIDEO MiniMax H3 fallito: {exc}") from exc
        elif torch.is_tensor(images_to_save):
            _encode_images(images_to_save, audio_to_save, H3_FPS, segment_path)
            _write_segment_metadata(segment_path, int(images_to_save.shape[0]), H3_FPS)
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
                if overlap_stitch:
                    _concat_videos_overlap(segment_paths, final_path, trim_count, H3_FPS)
                else:
                    _concat_videos(segment_paths, final_path)
                messages.append(f"Video finale concatenato: {final_name}")
                subfolder = os.path.relpath(output_folder, folder_paths.get_output_directory()).replace("\\", "/")
                preview = {"filename": final_name, "subfolder": "" if subfolder == "." else subfolder, "type": "output"}
                if not bool(keep_segments):
                    for path in segment_paths:
                        path.unlink(missing_ok=True)
                        _segment_meta_path(path).unlink(missing_ok=True)
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
    "IAMCCS_MiniMaxH3NativeCheckpointSave": IAMCCS_MiniMaxH3NativeCheckpointSave,
    "IAMCCS_MiniMaxH3NativeMasterLoad": IAMCCS_MiniMaxH3NativeMasterLoad,
    "IAMCCS_MiniMaxH3FinalMasterSave": IAMCCS_MiniMaxH3FinalMasterSave,
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
    "IAMCCS_MiniMaxH3NativeCheckpointSave": "MiniMax H3 Native Checkpoint Save",
    "IAMCCS_MiniMaxH3NativeMasterLoad": "MiniMax H3 Native Master -> One-Film Upscale",
    "IAMCCS_MiniMaxH3FinalMasterSave": "MiniMax H3 Final Master Save",
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
from .iamccs_minimax_h3_continuity import (
    NODE_CLASS_MAPPINGS as _CONTINUITY_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _CONTINUITY_NODE_DISPLAY_NAME_MAPPINGS,
)
from .iamccs_minimax_h3_cine_info import (
    NODE_CLASS_MAPPINGS as _CINE_INFO_H3_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _CINE_INFO_H3_NODE_DISPLAY_NAME_MAPPINGS,
)
from .iamccs_minimax_h3_audio_drive import (
    NODE_CLASS_MAPPINGS as _AUDIO_DRIVE_R21_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _AUDIO_DRIVE_R21_NODE_DISPLAY_NAME_MAPPINGS,
)
from .iamccs_minimax_h3_audio_timeline import (
    NODE_CLASS_MAPPINGS as _AUDIO_TIMELINE_R21_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _AUDIO_TIMELINE_R21_NODE_DISPLAY_NAME_MAPPINGS,
)
from .iamccs_cine_h3_bus import (
    NODE_CLASS_MAPPINGS as _CINE_H3_AUDIO_BUS_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _CINE_H3_AUDIO_BUS_NODE_DISPLAY_NAME_MAPPINGS,
)
from .iamccs_minimax_h3_v2v_backend import (
    NODE_CLASS_MAPPINGS as _V2V_R22_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _V2V_R22_NODE_DISPLAY_NAME_MAPPINGS,
)

NODE_CLASS_MAPPINGS.update(_ATOMIC_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(_ATOMIC_NODE_DISPLAY_NAME_MAPPINGS)
NODE_CLASS_MAPPINGS.update(_CONTINUITY_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(_CONTINUITY_NODE_DISPLAY_NAME_MAPPINGS)
NODE_CLASS_MAPPINGS.update(_CINE_INFO_H3_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(_CINE_INFO_H3_NODE_DISPLAY_NAME_MAPPINGS)
NODE_CLASS_MAPPINGS.update(_AUDIO_DRIVE_R21_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(_AUDIO_DRIVE_R21_NODE_DISPLAY_NAME_MAPPINGS)
NODE_CLASS_MAPPINGS.update(_AUDIO_TIMELINE_R21_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(_AUDIO_TIMELINE_R21_NODE_DISPLAY_NAME_MAPPINGS)
NODE_CLASS_MAPPINGS.update(_CINE_H3_AUDIO_BUS_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(_CINE_H3_AUDIO_BUS_NODE_DISPLAY_NAME_MAPPINGS)
NODE_CLASS_MAPPINGS.update(_V2V_R22_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(_V2V_R22_NODE_DISPLAY_NAME_MAPPINGS)
