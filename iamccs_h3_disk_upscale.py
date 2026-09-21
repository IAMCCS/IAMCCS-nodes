# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later

"""Standalone, disk-bound H3 upscale stages. No Shotboard/backend routing here.

Stage 1 and Stage 2 must be queued as separate prompts to release the first
generation's model, conditioning and decoded IMAGE batch before tiled refine.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

import folder_paths
import torch
from safetensors.torch import load_file, save_file


CATEGORY = "IAMCCS/MiniMax H3/Disk Upscale (Standalone)"
SCHEMA = "iamccs.h3.disk_upscale.v1"
ROOT_NAME = "IAMCCS/H3_DISK_UPSCALE"


def _safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "").strip()).strip("_-")[:80]
    if not result:
        raise ValueError("Disk Upscale requires a non-empty render_id")
    return result


def _root() -> Path:
    return (Path(folder_paths.get_output_directory()) / ROOT_NAME).resolve()


def _run_dir(render_id: str) -> Path:
    return _root() / _safe_name(render_id)


def _inside_root(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(_root()):
        raise ValueError("Checkpoint must be inside ComfyUI/output/IAMCCS/H3_DISK_UPSCALE")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_new_bytes(path: Path, content: bytes) -> None:
    """Create a new manifest without replacing an existing user artifact."""
    if path.exists():
        raise FileExistsError(f"Disk Upscale refuses to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_new_tensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    if path.exists():
        raise FileExistsError(f"Disk Upscale refuses to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        save_file(tensors, str(temporary))
        os.rename(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _av_streams(latent) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("Expected a MiniMax H3 AV LATENT")
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        streams = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        streams = list(samples)
    else:
        raise ValueError("Expected nested video+audio H3 latent, not a flat image latent")
    if len(streams) != 2 or not all(torch.is_tensor(item) for item in streams):
        raise ValueError("Expected exactly two H3 AV latent tensors")
    video, audio = streams
    if video.ndim != 5 or audio.ndim != 4 or video.shape[0] != 1 or audio.shape[0] != 1:
        raise ValueError(f"Invalid H3 AV dimensions: video={tuple(video.shape)}, audio={tuple(audio.shape)}")
    if video.shape[1] != 24 or audio.shape[1] != 32:
        raise ValueError("H3 AV latent must have 24 video and 32 audio channels")
    return video, audio


def _waveform(audio) -> tuple[torch.Tensor, int]:
    if not isinstance(audio, dict) or not torch.is_tensor(audio.get("waveform")):
        raise ValueError("Connect the native AUDIO output; it is the soundtrack authority")
    wave = audio["waveform"]
    sample_rate = int(audio.get("sample_rate", 0) or 0)
    if wave.ndim != 3 or wave.shape[0] != 1 or wave.shape[1] not in (1, 2) or sample_rate < 8000:
        raise ValueError("AUDIO waveform must be [1,1|2,samples] with a valid sample_rate")
    return wave, sample_rate


def _read_checkpoint(checkpoint_path: str):
    path = _inside_root(checkpoint_path)
    if path.suffix.lower() != ".safetensors" or not path.is_file():
        raise FileNotFoundError(f"H3 Disk Upscale checkpoint missing: {path}")
    manifest_path = path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("checkpoint_name") != path.name:
        raise ValueError("Checkpoint manifest has an invalid schema or file identity")
    if _sha256(path) != manifest.get("checkpoint_sha256"):
        raise ValueError("Checkpoint SHA-256 mismatch; refusing a partial or changed latent")
    tensors = load_file(str(path), device="cpu")
    required = {"video", "audio_latent", "waveform"}
    if set(tensors) != required:
        raise ValueError(f"Checkpoint tensor keys must be {sorted(required)}")
    video, audio_latent = tensors["video"], tensors["audio_latent"]
    _av_streams({"samples": (video, audio_latent)})
    wave, rate = _waveform({"waveform": tensors["waveform"], "sample_rate": manifest["audio_sample_rate"]})
    if int(video.shape[2]) != int(manifest["video_tokens"]):
        raise ValueError("Checkpoint temporal shape does not match its manifest")
    if int(video.shape[2]) < 7 or 17 * ((int(video.shape[2]) - 2) // 5) + 5 < int(manifest["source_frames"]):
        raise ValueError("Checkpoint latent cannot decode the declared source frame count")
    if float(wave.shape[-1]) / rate + 1 / float(manifest["fps"]) < float(manifest["source_frames"]) / float(manifest["fps"]):
        raise ValueError("Checkpoint audio is shorter than its declared video span")
    return path, manifest, video, audio_latent, wave, rate


def _nested_latent(video: torch.Tensor, audio: torch.Tensor):
    import comfy.nested_tensor

    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


def _out0(result):
    try:
        return result[0]
    except (TypeError, KeyError, IndexError):
        return result.result[0]


def _video_frames(path: Path) -> tuple[int, int, int, bool]:
    """Inspect the encoded stream without materializing an IMAGE batch."""
    import av

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in {path}")
        stream = container.streams.video[0]
        width, height = int(stream.width), int(stream.height)
        has_audio = bool(container.streams.audio)
        count = sum(1 for _ in container.decode(video=0))
    if count < 1:
        raise ValueError(f"No decoded frames in {path}")
    return count, width, height, has_audio


class IAMCCS_H3DiskUpscaleCheckpoint:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT",), "native_audio": ("AUDIO",),
            "render_id": ("STRING", {"default": "h3_upscale_test"}),
            "segment_index": ("INT", {"default": 0, "min": 0, "max": 99999}),
            "source_frames": ("INT", {"default": 0, "min": 0, "max": 100000,
                                      "tooltip": "0 = derive the exact decodable H3 frame count from the AV latent"}),
            "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
            "technical_prefix_frames": ("INT", {"default": 0, "min": 0, "max": 10000}),
            "join_overlap_frames": ("INT", {"default": 0, "min": 0, "max": 10000}),
            "join_mode": (["cut", "crossfade"], {"default": "cut"}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("checkpoint_path", "report")
    FUNCTION = "save"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def save(self, latent, native_audio, render_id, segment_index, source_frames, fps,
             technical_prefix_frames, join_overlap_frames, join_mode):
        video, audio_latent = _av_streams(latent)
        wave, rate = _waveform(native_audio)
        run = _safe_name(render_id)
        index = int(segment_index)
        decodable_frames = 17 * ((int(video.shape[2]) - 2) // 5) + 5 if int(video.shape[2]) >= 7 else 0
        frames = int(source_frames) or decodable_frames
        fps = int(fps)
        technical = int(technical_prefix_frames)
        overlap = int(join_overlap_frames)
        if fps != 24:
            raise ValueError("MiniMax H3 disk upscale currently requires 24 fps for frame-accurate joins")
        if decodable_frames < frames:
            raise ValueError("H3 latent cannot decode the declared source frame count")
        if technical + overlap >= frames or (index == 0 and overlap):
            raise ValueError("Invalid technical/join prefix for the segment frame count")
        if join_mode == "crossfade" and index > 0 and overlap < 2:
            raise ValueError("Crossfade requires at least two decoded overlap frames")
        if float(wave.shape[-1]) / rate + 1 / fps < frames / fps:
            raise ValueError("Native audio does not cover source_frames")
        folder = _run_dir(run) / "checkpoints"
        path = folder / f"segment_{index:05d}.safetensors"
        if path.exists() or path.with_suffix(".json").exists():
            raise FileExistsError(f"Checkpoint already exists: {path}")
        tensors = {
            "video": video.detach().to(device="cpu").contiguous(),
            "audio_latent": audio_latent.detach().to(device="cpu").contiguous(),
            "waveform": wave.detach().to(device="cpu").contiguous(),
        }
        _atomic_new_tensors(path, tensors)
        manifest = {
            "schema": SCHEMA, "render_id": run, "segment_index": index,
            "checkpoint_name": path.name, "checkpoint_sha256": _sha256(path),
            "video_tokens": int(video.shape[2]), "native_width": int(video.shape[-1]) * 16,
            "native_height": int(video.shape[-2]) * 16, "source_frames": frames,
            "fps": fps, "audio_sample_rate": rate, "technical_prefix_frames": technical,
            "join_overlap_frames": overlap, "join_mode": join_mode,
        }
        manifest_path = path.with_suffix(".json")
        try:
            _atomic_new_bytes(manifest_path, json.dumps(manifest, indent=2).encode("utf-8"))
        except Exception:
            # The checkpoint is usable only as an authenticated pair. Roll back
            # the exact file created by this call so a retry can resume cleanly.
            if path.exists() and not manifest_path.exists():
                path.unlink()
            raise
        return str(path), f"H3 AV + native audio checkpointed on disk: {path} ({frames} source frames)"


class IAMCCS_H3DiskUpscaleLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"checkpoint_path": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("LATENT", "AUDIO", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("av_latent", "native_audio", "manifest_json", "segment_index", "source_frames", "report")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def load(self, checkpoint_path):
        path, manifest, video, audio, wave, rate = _read_checkpoint(checkpoint_path)
        return (_nested_latent(video, audio), {"waveform": wave, "sample_rate": rate},
                json.dumps(manifest), int(manifest["segment_index"]),
                int(manifest["source_frames"]), f"Verified H3 AV checkpoint {path}")


def _tiled_params(model_name, width, height, tile_width, tile_height, spatial_overlap,
                  temporal_chunk, temporal_overlap, upscaler_device, precision):
    if width < 64 or height < 64 or width % 32 or height % 32:
        raise ValueError("Internal H3 target dimensions must be 32-aligned")
    if tile_width % 32 or tile_height % 32 or min(tile_width, tile_height) < 128:
        raise ValueError("H3 tiles must be 32-aligned and at least 128 pixels")
    if tile_width > width or tile_height > height:
        raise ValueError("H3 tile cannot exceed the internal target canvas")
    if spatial_overlap % 32 or spatial_overlap >= min(tile_width, tile_height):
        raise ValueError("Spatial overlap must be 32-aligned and smaller than each tile")
    if temporal_chunk % 17 or temporal_overlap % 17 or temporal_chunk <= temporal_overlap:
        raise ValueError("Temporal chunk and overlap must be 17-frame multiples, with chunk > overlap")
    if upscaler_device == "cpu" and precision != "fp32":
        raise ValueError("CPU learned lift requires fp32 precision")
    if not model_name or not folder_paths.get_full_path("latent_upscale_models", model_name):
        raise ValueError("Select an installed H3 3D .safetensors latent-upscaler model")
    latent = {"model_name": model_name, "width": width, "height": height,
              "device": upscaler_device, "precision": precision}
    temporal = {"chunk_length": temporal_chunk, "temporal_overlap": temporal_overlap,
                "anchor_strength": 0.999}
    spatial = {
        "tile_width": tile_width, "tile_height": tile_height,
        "spatial_w_overlap": spatial_overlap, "spatial_h_overlap": spatial_overlap,
        "fade_width": min(32, spatial_overlap), "fade_height": min(32, spatial_overlap),
        "min_tile_size": min(256, tile_width, tile_height),
        "overlap_mode": "earlier", "overlap_blend": "smoothstep",
        "tile_size_mode": "specific_size", "masked_area_noise": 0.0,
        "brightness_match": False, "dynamic_fade": "off", "dynamic_fade_min": 32,
    }
    return latent, temporal, spatial


def _delivery_cover_size(native_width: int, native_height: int,
                         delivery_width: int, delivery_height: int,
                         align: int = 32) -> tuple[int, int]:
    """Return an aligned, aspect-preserving canvas that covers delivery.

    The learned H3 model uses one effective scale for both spatial axes.  Asking
    it to map a 5:3 latent directly to a 16:9 canvas introduces anisotropic
    deformation.  Upscale to a cover canvas instead and crop only after decode.
    """
    values = native_width, native_height, delivery_width, delivery_height, align
    if any(int(value) < 1 for value in values):
        raise ValueError("Native, delivery and alignment dimensions must be positive")
    scale = max(delivery_width / native_width, delivery_height / native_height)
    width = math.ceil((native_width * scale) / align) * align
    height = math.ceil((native_height * scale) / align) * align
    return int(width), int(height)


TARGET_PRESETS = {
    "hd_1280x720": (1280, 720),
    "full_hd_1920x1080": (1920, 1080),
    "qhd_2560x1440": (2560, 1440),
    "uhd_3840x2160": (3840, 2160),
}


def _delivery_dimensions(native_width: int, native_height: int, target_preset: str,
                         custom_width: int, custom_height: int) -> tuple[int, int]:
    """Resolve delivery dimensions from the actual low-resolution source."""
    if target_preset in TARGET_PRESETS:
        return TARGET_PRESETS[target_preset]
    source_scales = {"source_1_5x": 1.5, "source_2x": 2.0, "source_3x": 3.0}
    if target_preset in source_scales:
        scale = source_scales[target_preset]
        return (
            max(8, round(native_width * scale / 8) * 8),
            max(8, round(native_height * scale / 8) * 8),
        )
    if target_preset == "custom":
        return int(custom_width), int(custom_height)
    raise ValueError(f"Unknown H3 upscale target preset: {target_preset}")


def _release_external_upscaler(klass) -> None:
    """Release models cached by the third-party upscaler, including OOM paths."""
    module = sys.modules.get(getattr(klass, "__module__", ""))
    cache = getattr(module, "MODEL_CACHE", None)
    if isinstance(cache, dict):
        for model in tuple(cache.values()):
            try:
                model.to("cpu")
            except Exception:
                pass
        cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _learned_3d_temporal_lift(klass, video: torch.Tensor, model_name: str,
                              width: int, height: int, device: str, precision: str,
                              core_tokens: int, halo_tokens: int) -> torch.Tensor:
    """Upscale full spatial frames in small temporal windows and stitch on CPU.

    The provider's built-in temporal chunk is hardcoded to 32 latent tokens and
    expands each segment with halos; at Full HD that still OOMs on 12 GB.  This
    wrapper keeps every spatial operation full-frame (so no texture grid), but
    limits the Conv3D activation peak to ``core + 2*halo`` latent tokens.
    """
    if video.ndim != 5:
        raise ValueError("H3 learned temporal lift expects a 5D B,C,T,H,W latent")
    core_tokens, halo_tokens = int(core_tokens), int(halo_tokens)
    if core_tokens < 1 or halo_tokens < 0:
        raise ValueError("Temporal core must be positive and halo cannot be negative")
    total = int(video.shape[2])
    output = torch.empty(
        int(video.shape[0]), int(video.shape[1]), total,
        int(height) // 16, int(width) // 16,
        dtype=video.dtype, device="cpu",
    )
    try:
        for core_start in range(0, total, core_tokens):
            core_end = min(total, core_start + core_tokens)
            window_start = max(0, core_start - halo_tokens)
            window_end = min(total, core_end + halo_tokens)
            window = video[:, :, window_start:window_end].contiguous()
            payload = _out0(klass.execute(
                latent={"samples": window},
                model_name=model_name,
                mode={"mode": "target dimensions", "width": width, "height": height},
                align=32,
                enable_temporal_chunking=False,
                force_unload=False,
                device=device,
                precision=precision,
            ))
            if not isinstance(payload, dict) or not torch.is_tensor(payload.get("samples")):
                raise TypeError("Minimax H3 3D upscaler returned an invalid temporal window")
            window_out = payload["samples"]
            local_start = core_start - window_start
            local_end = local_start + (core_end - core_start)
            output[:, :, core_start:core_end].copy_(
                window_out[:, :, local_start:local_end].detach().to("cpu")
            )
            del payload, window_out, window
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    except Exception:
        _release_external_upscaler(klass)
        raise
    _release_external_upscaler(klass)
    return output


def _trim_encoded(raw: Path, final: Path, start: int, end: int, width: int, height: int, fps: int):
    from .iamccs_minimax_h3_shotboard import _find_ffmpeg

    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required for exact H3 segment trim/crop")
    if final.exists():
        raise FileExistsError(f"Disk Upscale refuses to overwrite {final}")
    crop = f"crop={width}:{height}:(iw-{width})/2:(ih-{height})/2"
    video_filter = f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS,{crop}"
    audio_filter = f"atrim=start={start / fps:.9f}:end={end / fps:.9f},asetpts=PTS-STARTPTS"
    final.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-i", str(raw),
               "-map", "0:v:0", "-map", "0:a:0", "-vf", video_filter, "-af", audio_filter,
               "-frames:v", str(end - start), "-r", str(fps),
               "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(final)]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError("H3 exact segment trim failed: " + completed.stderr[-2000:])


def _publish_validated_segment(temporary_final: Path, final: Path, result: dict[str, Any]):
    """Publish one validated MP4 and its authenticated metadata without overwrite.

    The encoded bytes exist at ``temporary_final`` until the last atomic rename,
    so their checksum must be computed there.  Hashing ``final`` before the
    rename caused successful GPU renders to fail after streaming decode.
    """
    final_manifest = final.with_suffix(".json")
    sidecar = final.with_suffix(final.suffix + ".iamccs.json")
    if final.exists() or final_manifest.exists() or sidecar.exists():
        raise FileExistsError(f"Upscaled segment already exists: {final}")
    if not temporary_final.is_file():
        raise FileNotFoundError(f"Validated temporary segment is missing: {temporary_final}")

    published = dict(result)
    published["segment_path"] = str(final)
    published["segment_sha256"] = _sha256(temporary_final)
    promoted = False
    try:
        # Prepare both manifests first; the validated MP4 is promoted last.
        # Therefore a visible final segment always has all of its metadata.
        _atomic_new_bytes(sidecar, json.dumps({
            "schema": "iamccs.minimax_h3.segment",
            "frame_count": int(published["frame_count"]),
            "fps": float(published["fps"]),
            "audio_join_policy": "native_audio_locked",
        }, indent=2).encode("utf-8"))
        _atomic_new_bytes(final_manifest, json.dumps(published, indent=2).encode("utf-8"))
        if final.exists():
            raise FileExistsError(f"Upscaled segment already exists: {final}")
        os.rename(temporary_final, final)
        promoted = True
    except Exception:
        if not promoted:
            if final_manifest.exists():
                final_manifest.unlink()
            if sidecar.exists():
                sidecar.unlink()
        raise
    return final_manifest, published


class IAMCCS_H3DiskUpscaleTiledRefine:
    @classmethod
    def INPUT_TYPES(cls):
        try:
            installed = folder_paths.get_filename_list("latent_upscale_models")
        except KeyError:
            installed = []
        models = sorted(
            name for name in installed
            if all(key in name.lower() for key in ("minimax", "h3", "3d"))
            and name.lower().endswith(".safetensors")
        )
        return {"required": {
            "checkpoint_path": ("STRING", {"default": ""}),
            "model": ("MODEL",), "conditioning": ("CONDITIONING",),
            "noise": ("NOISE",), "sampler": ("SAMPLER",), "sigmas": ("SIGMAS",),
            "video_vae": ("VAE",),
            "upscaler_model": ([""] + models, {"default": models[0] if models else ""}),
            "target_width": ("INT", {"default": 1920, "min": 256, "max": 3840, "step": 8}),
            "target_height": ("INT", {"default": 1080, "min": 256, "max": 2160, "step": 8}),
            "tile_width": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 32}),
            "tile_height": ("INT", {"default": 384, "min": 128, "max": 2048, "step": 32}),
            "spatial_overlap": ("INT", {"default": 96, "min": 0, "max": 512, "step": 32}),
            "temporal_chunk_frames": ("INT", {"default": 68, "min": 17, "max": 340, "step": 17}),
            "temporal_overlap_frames": ("INT", {"default": 17, "min": 0, "max": 170, "step": 17}),
            "upscaler_device": (["cuda", "cpu"], {"default": "cuda"}),
            "upscaler_precision": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
            "decode_groups_per_chunk": ("INT", {"default": 1, "min": 1, "max": 8}),
        }, "optional": {
            "fun_control_param": ("H3_FUN_CONTROL_PARAM",),
            "inpaint_param": ("H3_INPAINT_PARAM",),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("segment_path", "segment_manifest_path", "report")
    FUNCTION = "refine"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def refine(self, checkpoint_path, model, conditioning, noise, sampler, sigmas, video_vae,
               upscaler_model, target_width, target_height, tile_width, tile_height,
               spatial_overlap, temporal_chunk_frames, temporal_overlap_frames,
               upscaler_device, upscaler_precision, decode_groups_per_chunk,
               fun_control_param=None, inpaint_param=None):
        import comfy.model_management as mm
        import nodes

        path, manifest, video, audio_latent, wave, rate = _read_checkpoint(checkpoint_path)
        run = str(manifest["render_id"])
        index = int(manifest["segment_index"])
        fps = int(manifest["fps"])
        width, height = int(target_width), int(target_height)
        internal_width = math.ceil(width / 32) * 32
        internal_height = math.ceil(height / 32) * 32
        latent_param, temporal_param, spatial_param = _tiled_params(
            upscaler_model, internal_width, internal_height, int(tile_width), int(tile_height),
            int(spatial_overlap), int(temporal_chunk_frames), int(temporal_overlap_frames),
            upscaler_device, upscaler_precision,
        )
        if fun_control_param is not None:
            if str(fun_control_param.get("control_upscale_mode")) != "per_tile":
                raise ValueError("Disk-safe H3 Fun ControlNet requires per_tile control upscale mode")
            if (int(fun_control_param.get("upscale_width", -1)),
                    int(fun_control_param.get("upscale_height", -1))) != (internal_width, internal_height):
                raise ValueError("Fun ControlNet canvas must match the 32-aligned H3 upscale canvas")
            required_guide_frames = 17 * ((int(video.shape[2]) - 2) // 5) + 5
            if len(fun_control_param.get("control_video", ())) < required_guide_frames:
                raise ValueError("Fun ControlNet guide must cover the full decoded source segment")
        if width < int(manifest["native_width"]) or height < int(manifest["native_height"]):
            raise ValueError("Tiled refine must not downscale the native H3 canvas")
        klass = nodes.NODE_CLASS_MAPPINGS.get("MMH3UltimateUpscale")
        if klass is None:
            raise RuntimeError("Install/enable Comfyui-MMH3-UltimateUpscale and restart ComfyUI")
        segment_dir = _run_dir(run) / "segments"
        final = segment_dir / f"segment_{index:05d}.mp4"
        final_manifest = final.with_suffix(".json")
        sidecar = final.with_suffix(final.suffix + ".iamccs.json")
        if final.exists() or final_manifest.exists() or sidecar.exists():
            raise FileExistsError(f"Upscaled segment already exists: {final}")
        temporary_final = final.with_name(f"{final.stem}.{uuid.uuid4().hex}.tmp.mp4")

        try:
            refined = _out0(klass.execute(
                latent=_nested_latent(video, audio_latent), conditioning=conditioning,
                model=model, noise=noise, sampler=sampler, sigmas=sigmas,
                negative=None, cfg=1.0, latent_upscale_param=latent_param,
                temporal_split_param=temporal_param, spatial_split_param=spatial_param,
                fun_control_param=fun_control_param, inpaint_param=inpaint_param,
            ))
            refined_video, refined_audio = _av_streams(refined)
            # Only the compressed AV latent remains in CPU RAM during VAE decode.
            refined_cpu = _nested_latent(refined_video.detach().to("cpu"), refined_audio.detach().to("cpu"))
            del refined, refined_video, refined_audio, video, audio_latent
            mm.unload_all_models()
            mm.soft_empty_cache()
            gc.collect()

            from .iamccs_minimax_h3_pixel_refine_variant import _provider

            raw = Path(_out0(_provider("nodes_save").MMH3StreamingSave.execute(
                latent=refined_cpu, vae=video_vae, groups_per_chunk=int(decode_groups_per_chunk),
                fps=float(fps), filename_prefix=f"{ROOT_NAME}/{run}/raw/segment_{index:05d}",
                crf=16, audio={"waveform": wave, "sample_rate": rate}, save_metadata=False,
            )))
            del refined_cpu
            mm.unload_all_models()
            mm.soft_empty_cache()
            gc.collect()
            raw_count, raw_width, raw_height, raw_audio = _video_frames(raw)
            if raw_count < int(manifest["source_frames"]) or not raw_audio:
                raise RuntimeError("Streaming decode has missing frames or native audio; segment not accepted")
            if raw_width < width or raw_height < height:
                raise RuntimeError("Streaming decode is smaller than the requested delivery canvas")
            technical = int(manifest["technical_prefix_frames"])
            overlap = int(manifest["join_overlap_frames"])
            join_mode = str(manifest["join_mode"])
            head = technical + (overlap if join_mode == "cut" else 0)
            end = int(manifest["source_frames"])
            _trim_encoded(raw, temporary_final, head, end, width, height, fps)
            frame_count, out_width, out_height, out_audio = _video_frames(temporary_final)
            if (frame_count, out_width, out_height, out_audio) != (end - head, width, height, True):
                raise RuntimeError("Final segment failed exact frame/canvas/audio validation")
            result = {
                "schema": SCHEMA, "render_id": run, "segment_index": index,
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "frame_count": frame_count, "width": width, "height": height, "fps": fps,
                "has_audio": True, "join_mode": join_mode,
                "join_overlap_frames": overlap if join_mode == "crossfade" else 0,
                "technical_prefix_trimmed": technical,
            }
            final_manifest, result = _publish_validated_segment(temporary_final, final, result)
            return str(final), str(final_manifest), (
                f"H3 tiled upscale saved {frame_count} frames at {width}x{height}; "
                f"tile={tile_width}x{tile_height}, temporal={temporal_chunk_frames}/{temporal_overlap_frames}; "
                f"native audio preserved: {final}"
            )
        finally:
            if temporary_final.exists():
                temporary_final.unlink()
            mm.unload_all_models()
            mm.soft_empty_cache()
            gc.collect()


class IAMCCS_H3DiskUpscaleLearned3D:
    """Grid-free H3 learned latent lift followed by disk-bound streaming decode.

    Unlike ``MMH3UltimateUpscale``, this path does not run a new diffusion
    sample independently inside spatial tiles.  Temporal chunking remains
    enabled inside the learned 3D upscaler, while the original audio latent and
    waveform are carried through unchanged.
    """

    @classmethod
    def INPUT_TYPES(cls):
        try:
            installed = folder_paths.get_filename_list("latent_upscale_models")
        except KeyError:
            installed = []
        models = sorted(
            name for name in installed
            if all(key in name.lower() for key in ("minimax", "h3", "3d"))
            and name.lower().endswith(".safetensors")
        )
        return {"required": {
            "checkpoint_path": ("STRING", {"default": ""}),
            "output_render_id": ("STRING", {"default": ""}),
            "video_vae": ("VAE",),
            "upscaler_model": ([""] + models, {"default": models[0] if models else ""}),
            "target_preset": ([
                "full_hd_1920x1080", "hd_1280x720", "qhd_2560x1440",
                "uhd_3840x2160", "source_1_5x", "source_2x", "source_3x", "custom",
            ], {"default": "full_hd_1920x1080"}),
            "target_width": ("INT", {"default": 1920, "min": 256, "max": 3840, "step": 8}),
            "target_height": ("INT", {"default": 1080, "min": 256, "max": 2160, "step": 8}),
            "upscaler_device": (["cuda", "cpu"], {"default": "cuda"}),
            "upscaler_precision": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
            "temporal_core_tokens": ("INT", {"default": 4, "min": 1, "max": 32}),
            "temporal_halo_tokens": ("INT", {"default": 4, "min": 0, "max": 16}),
            "decode_groups_per_chunk": ("INT", {"default": 1, "min": 1, "max": 8}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("segment_path", "segment_manifest_path", "report")
    FUNCTION = "refine"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def refine(self, checkpoint_path, output_render_id, video_vae, upscaler_model, target_preset,
               target_width, target_height, upscaler_device, upscaler_precision,
               temporal_core_tokens, temporal_halo_tokens, decode_groups_per_chunk):
        import comfy.model_management as mm
        import nodes

        path, manifest, video, audio_latent, wave, rate = _read_checkpoint(checkpoint_path)
        source_run = str(manifest["render_id"])
        run = _safe_name(output_render_id) if str(output_render_id).strip() else source_run
        index = int(manifest["segment_index"])
        fps = int(manifest["fps"])
        native_width = int(manifest["native_width"])
        native_height = int(manifest["native_height"])
        width, height = _delivery_dimensions(
            native_width, native_height, str(target_preset),
            int(target_width), int(target_height),
        )
        if width < native_width or height < native_height:
            raise ValueError("Learned H3 upscale must not downscale the native canvas")
        if upscaler_device == "cpu" and upscaler_precision != "fp32":
            raise ValueError("CPU learned lift requires fp32 precision")
        if not upscaler_model or not folder_paths.get_full_path("latent_upscale_models", upscaler_model):
            raise ValueError("Select an installed H3 3D .safetensors latent-upscaler model")

        internal_width, internal_height = _delivery_cover_size(
            native_width, native_height, width, height, 32
        )
        klass = nodes.NODE_CLASS_MAPPINGS.get("MinimaxH3LatentUpscaler3D")
        if klass is None:
            raise RuntimeError("Install/enable Comfyui_Minimax_h3_latent_Upscaler and restart ComfyUI")

        segment_dir = _run_dir(run) / "segments"
        final = segment_dir / f"segment_{index:05d}.mp4"
        final_manifest = final.with_suffix(".json")
        sidecar = final.with_suffix(final.suffix + ".iamccs.json")
        if final.exists() or final_manifest.exists() or sidecar.exists():
            raise FileExistsError(f"Upscaled segment already exists: {final}")
        temporary_final = final.with_name(f"{final.stem}.{uuid.uuid4().hex}.tmp.mp4")

        try:
            # Queue 2 never needs to reload H3 itself: release any resident model
            # before the learned lift and keep the spatial operation full-frame.
            mm.unload_all_models()
            mm.soft_empty_cache()
            gc.collect()
            refined_video = _learned_3d_temporal_lift(
                klass, video, upscaler_model, internal_width, internal_height,
                upscaler_device, upscaler_precision,
                int(temporal_core_tokens), int(temporal_halo_tokens),
            )
            expected_h, expected_w = internal_height // 16, internal_width // 16
            if tuple(refined_video.shape[-2:]) != (expected_h, expected_w):
                raise RuntimeError(
                    "Learned H3 upscale returned an unexpected canvas: "
                    f"{refined_video.shape[-1] * 16}x{refined_video.shape[-2] * 16}"
                )
            refined_cpu = _nested_latent(
                refined_video.detach().to("cpu"), audio_latent.detach().to("cpu")
            )
            del refined_video, video, audio_latent
            mm.unload_all_models()
            mm.soft_empty_cache()
            gc.collect()

            from .iamccs_minimax_h3_pixel_refine_variant import _provider

            raw = Path(_out0(_provider("nodes_save").MMH3StreamingSave.execute(
                latent=refined_cpu, vae=video_vae, groups_per_chunk=int(decode_groups_per_chunk),
                fps=float(fps), filename_prefix=f"{ROOT_NAME}/{run}/raw/segment_{index:05d}",
                crf=16, audio={"waveform": wave, "sample_rate": rate}, save_metadata=False,
            )))
            del refined_cpu
            mm.unload_all_models()
            mm.soft_empty_cache()
            gc.collect()
            raw_count, raw_width, raw_height, raw_audio = _video_frames(raw)
            if raw_count < int(manifest["source_frames"]) or not raw_audio:
                raise RuntimeError("Streaming decode has missing frames or native audio; segment not accepted")
            if (raw_width, raw_height) != (internal_width, internal_height):
                raise RuntimeError("Streaming decode does not match the learned full-frame canvas")
            technical = int(manifest["technical_prefix_frames"])
            overlap = int(manifest["join_overlap_frames"])
            join_mode = str(manifest["join_mode"])
            head = technical + (overlap if join_mode == "cut" else 0)
            end = int(manifest["source_frames"])
            _trim_encoded(raw, temporary_final, head, end, width, height, fps)
            frame_count, out_width, out_height, out_audio = _video_frames(temporary_final)
            if (frame_count, out_width, out_height, out_audio) != (end - head, width, height, True):
                raise RuntimeError("Final segment failed exact frame/canvas/audio validation")
            result = {
                "schema": SCHEMA, "render_id": run, "segment_index": index,
                "source_render_id": source_run,
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "frame_count": frame_count, "width": width, "height": height, "fps": fps,
                "has_audio": True, "join_mode": join_mode,
                "join_overlap_frames": overlap if join_mode == "crossfade" else 0,
                "technical_prefix_trimmed": technical,
                "upscale_method": "learned_3d_grid_free",
                "source_canvas": [native_width, native_height],
                "target_preset": str(target_preset),
                "learned_canvas": [internal_width, internal_height],
                "temporal_core_tokens": int(temporal_core_tokens),
                "temporal_halo_tokens": int(temporal_halo_tokens),
            }
            final_manifest, result = _publish_validated_segment(temporary_final, final, result)
            return str(final), str(final_manifest), (
                f"H3 learned 3D upscale {native_width}x{native_height} -> "
                f"{width}x{height} ({target_preset}); saved {frame_count} frames; "
                f"isotropic latent canvas={internal_width}x{internal_height}, "
                f"temporal core/halo={temporal_core_tokens}/{temporal_halo_tokens}; "
                f"no spatial diffusion tiles; native audio preserved: {final}"
            )
        finally:
            if temporary_final.exists():
                temporary_final.unlink()
            mm.unload_all_models()
            mm.soft_empty_cache()
            gc.collect()


class IAMCCS_H3DiskUpscaleAssemble:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "render_id": ("STRING", {"default": "h3_upscale_test"}),
            "segment_count": ("INT", {"default": 1, "min": 1, "max": 1000}),
            "output_name": ("STRING", {"default": "final_full_hd"}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("film_path", "report")
    FUNCTION = "assemble"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def assemble(self, render_id, segment_count, output_name):
        from .iamccs_minimax_h3_shotboard import _concat_videos, _concat_videos_overlap

        run = _safe_name(render_id)
        directory = _run_dir(run) / "segments"
        paths = [directory / f"segment_{i:05d}.mp4" for i in range(int(segment_count))]
        manifests = []
        for index, path in enumerate(paths):
            if not path.is_file() or not path.with_suffix(".json").is_file():
                raise FileNotFoundError(f"Missing upscaled segment or manifest: {path}")
            data = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            if (data.get("schema") != SCHEMA or data.get("render_id") != run
                    or int(data.get("segment_index", -1)) != index
                    or Path(data.get("segment_path", "")).resolve() != path.resolve()):
                raise ValueError(f"Segment manifest order/schema mismatch: {path}")
            if _sha256(path) != data.get("segment_sha256"):
                raise ValueError(f"Upscaled segment checksum mismatch: {path}")
            count, width, height, audio = _video_frames(path)
            if (count, width, height, audio) != (int(data["frame_count"]), int(data["width"]), int(data["height"]), True):
                raise ValueError(f"Segment frame/canvas/audio mismatch: {path}")
            manifests.append(data)
        width, height, fps = (int(manifests[0][key]) for key in ("width", "height", "fps"))
        if any((int(m["width"]), int(m["height"]), int(m["fps"])) != (width, height, fps) for m in manifests):
            raise ValueError("All H3 upscale segments must have an identical canvas and frame rate")
        if len(manifests) > 1:
            modes = {str(m["join_mode"]) for m in manifests[1:]}
            if len(modes) != 1:
                raise ValueError("Mixed cut/crossfade joins are not supported in this standalone assembler")
            mode = modes.pop()
            overlaps = {int(m["join_overlap_frames"]) for m in manifests[1:]}
            if mode == "crossfade" and (len(overlaps) != 1 or min(overlaps) < 2):
                raise ValueError("Crossfade segments must share one overlap of at least two frames")
            if mode == "crossfade" and any(
                int(m["frame_count"]) <= next(iter(overlaps)) for m in manifests
            ):
                raise ValueError("Crossfade overlap must be shorter than every segment")
        else:
            mode, overlaps = "cut", {0}
        output = _run_dir(run) / f"{_safe_name(output_name)}.mp4"
        if output.exists():
            raise FileExistsError(f"Disk Upscale refuses to overwrite {output}")
        join_frames = next(iter(overlaps)) if mode == "crossfade" else 0
        expected_count = sum(int(m["frame_count"]) for m in manifests) - join_frames * (len(paths) - 1)
        temporary_output = output.with_name(f"{output.stem}.{uuid.uuid4().hex}.tmp.mp4")
        try:
            if mode == "crossfade":
                _concat_videos_overlap(paths, temporary_output, join_frames, fps)
            else:
                _concat_videos(paths, temporary_output)
            count, actual_width, actual_height, has_audio = _video_frames(temporary_output)
            if (count, actual_width, actual_height, has_audio) != (expected_count, width, height, True):
                raise RuntimeError("Assembled film failed exact frame/canvas/audio validation")
            if output.exists():
                raise FileExistsError(f"Disk Upscale refuses to overwrite {output}")
            os.rename(temporary_output, output)
        finally:
            if temporary_output.exists():
                temporary_output.unlink()
        return str(output), f"H3 disk upscale film ready: {count} frames, {width}x{height}, {fps} fps, {len(paths)} segments: {output}"


NODE_CLASS_MAPPINGS = {
    "IAMCCS_H3DiskUpscaleCheckpoint": IAMCCS_H3DiskUpscaleCheckpoint,
    "IAMCCS_H3DiskUpscaleLoad": IAMCCS_H3DiskUpscaleLoad,
    "IAMCCS_H3DiskUpscaleTiledRefine": IAMCCS_H3DiskUpscaleTiledRefine,
    "IAMCCS_H3DiskUpscaleLearned3D": IAMCCS_H3DiskUpscaleLearned3D,
    "IAMCCS_H3DiskUpscaleAssemble": IAMCCS_H3DiskUpscaleAssemble,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_H3DiskUpscaleCheckpoint": "IAMCCS H3 Disk Upscale · 1 Save AV Checkpoint",
    "IAMCCS_H3DiskUpscaleLoad": "IAMCCS H3 Disk Upscale · Load AV Checkpoint",
    "IAMCCS_H3DiskUpscaleTiledRefine": "IAMCCS H3 Disk Upscale · 2 Tiled Refine + Stream",
    "IAMCCS_H3DiskUpscaleLearned3D": "IAMCCS H3 Disk Upscale · 2 Learned 3D (Grid-Free) + Stream",
    "IAMCCS_H3DiskUpscaleAssemble": "IAMCCS H3 Disk Upscale · 3 Assemble Film",
}
