from __future__ import annotations

import importlib.util
import io
import hashlib
import json
import os
import re
import sys
import types
from typing import Any, Dict, List, Optional, Tuple

import comfy.utils
import folder_paths
import numpy as np
import nodes as comfy_nodes
import torch
import torch.nn.functional as F
from comfy_extras.nodes_lt import LTXVAddGuide
from PIL import Image, ImageOps

try:
    from comfy_extras.nodes_lt import _append_guide_attention_entry
except Exception:
    _append_guide_attention_entry = None

from .iamccs_ltx2_cinematic_flf import (
    IAMCCS_LTX2_AudioPromptDirector,
    IAMCCS_LTX2_CinematicLineStacker,
    IAMCCS_LTX2_CinematicMultiGenPlanner,
    IAMCCS_LTX2_CinematicPromptComposer,
    IAMCCS_LTX2_CinematicPromptRelayAdapter,
    IAMCCS_LTX2_CinematicRefLatentControl,
    IAMCCS_LTX2_CinematicShotLineBuilder,
    IAMCCS_LTX2_CinematicShotPlanner,
    IAMCCS_LTX2_CinematicShotAudioSelector,
    IAMCCS_LTX2_CinematicV2VAssetSelector,
    IAMCCS_LTX2_CinematicV2VTimelineLineBuilder,
    IAMCCS_LTX2_CinematicV2VTimelinePlanner,
)
from .iamccs_supernodes_linx import build_stage_linx_payload, linx_output, linx_resource


MAX_CINE_ITEMS = 50
SUPERNODE_LINX_TYPE = "IAMCCS_SUPERNODE_LINX"
CINE1_BACKEND_ID = "CINE_1"
CINE1_SECOND_STAGE_ID = "2nd_Stage_Cine_1"
CINE1_SECOND_STAGE_ALIASES = ("2nd_stage_CINE_1",)
_ORIGINAL_PROMPTRELAY_MODULE = None


def _iamccs_cine_debug_enabled() -> bool:
    return str(os.environ.get("IAMCCS_CINE_DEBUG", "")).strip().lower() in {"1", "true", "yes", "on", "debug"}


def _cine_debug(message: str) -> None:
    if _iamccs_cine_debug_enabled():
        print(message)


def _load_original_promptrelay_module():
    """Load ComfyUI-PromptRelay's own nodes.py under a synthetic package name.

    The upstream folder has a hyphen in its name, so it cannot be imported with a
    normal dotted Python import. Loading it this way preserves its relative
    imports and lets IAMCCS call the original _encode_relay implementation 1:1.
    """
    global _ORIGINAL_PROMPTRELAY_MODULE
    if _ORIGINAL_PROMPTRELAY_MODULE is not None:
        return _ORIGINAL_PROMPTRELAY_MODULE

    custom_nodes_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    promptrelay_dir = os.path.join(custom_nodes_dir, "ComfyUI-PromptRelay")
    nodes_path = os.path.join(promptrelay_dir, "nodes.py")
    if not os.path.exists(nodes_path):
        raise ImportError(f"ComfyUI-PromptRelay nodes.py not found: {nodes_path}")

    package_name = "_iamccs_original_promptrelay"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__file__ = os.path.join(promptrelay_dir, "__init__.py")
        package.__path__ = [promptrelay_dir]
        package.__package__ = package_name
        sys.modules[package_name] = package

    module_name = f"{package_name}.nodes"
    loaded = sys.modules.get(module_name)
    if loaded is not None and hasattr(loaded, "_encode_relay"):
        _ORIGINAL_PROMPTRELAY_MODULE = loaded
        return loaded

    spec = importlib.util.spec_from_file_location(module_name, nodes_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load ComfyUI-PromptRelay module from: {nodes_path}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package_name
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "_encode_relay"):
        raise ImportError("ComfyUI-PromptRelay module loaded, but _encode_relay is missing.")

    _ORIGINAL_PROMPTRELAY_MODULE = module
    return module



def _clamp(value: Any, lo: float, hi: float, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        number = float(default)
    return max(float(lo), min(float(hi), number))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _json_report(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _split_pipe(line: str, expected: int) -> List[str]:
    parts = [part.strip() for part in str(line or "").split("|")]
    while len(parts) < expected:
        parts.append("")
    if len(parts) > expected:
        return parts[: expected - 1] + [" | ".join(parts[expected - 1 :])]
    return parts


def _normalise_label(label: str, fallback: str) -> str:
    raw = str(label or fallback).strip().lower()
    raw = re.sub(r"[^a-z0-9_\-]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw or fallback


def _audio_duration_seconds(audio: Any) -> Optional[float]:
    if not isinstance(audio, dict):
        return None
    waveform = audio.get("waveform")
    sample_rate = _safe_int(audio.get("sample_rate"), 0)
    if waveform is None or sample_rate <= 0:
        return None
    if not isinstance(waveform, torch.Tensor):
        try:
            waveform = torch.tensor(waveform)
        except Exception:
            return None
    if waveform.ndim <= 0:
        return None
    return float(waveform.shape[-1]) / float(sample_rate)


def _round_ltx_frames(frames: int, mode: str) -> int:
    value = max(1, int(frames))
    if mode == "none":
        return value
    rounded = int(round((value - 1) / 8.0) * 8 + 1)
    if mode == "nearest_8n_plus_1":
        return max(1, rounded)
    return max(1, int(np.ceil(max(0, value - 1) / 8.0) * 8 + 1))


def _prompt_bank_lines(text: str) -> List[str]:
    lines = []
    for raw in str(text or "").replace("\r", "\n").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def _pick_cycled(items: List[str], index: int, fallback: str) -> str:
    if not items:
        return fallback
    return items[max(0, int(index)) % len(items)]


def _cine_reference_paths_from_text(image_paths: Any) -> List[str]:
    raw_paths = str(image_paths or "").strip()
    if not raw_paths:
        return []
    parsed_paths: List[str] = []
    try:
        parsed = json.loads(raw_paths)
        if isinstance(parsed, list):
            parsed_paths = [str(item).strip() for item in parsed if str(item).strip()]
        elif isinstance(parsed, dict):
            for key in ("image_paths", "paths", "images", "references"):
                value = parsed.get(key)
                if isinstance(value, list):
                    parsed_paths = [str(item).strip() for item in value if str(item).strip()]
                    break
                if isinstance(value, str):
                    parsed_paths = [p.strip() for p in value.replace("\r", "\n").splitlines() if p.strip()]
                    break
    except Exception:
        parsed_paths = []
    if parsed_paths:
        return parsed_paths
    return [p.strip() for p in raw_paths.replace("\r", "\n").splitlines() if p.strip()]


def _cine_resolve_reference_path(path: str) -> str:
    clean = str(path or "").strip()
    if not clean:
        return ""
    if os.path.exists(clean):
        return os.path.abspath(clean)
    input_path = os.path.join(folder_paths.get_input_directory(), clean)
    if os.path.exists(input_path):
        return os.path.abspath(input_path)
    return os.path.abspath(input_path if not os.path.isabs(clean) else clean)


def _cine_image_path_signature(image_paths: Any) -> List[Dict[str, Any]]:
    signature: List[Dict[str, Any]] = []
    for path in _cine_reference_paths_from_text(image_paths):
        resolved = _cine_resolve_reference_path(path)
        item: Dict[str, Any] = {
            "path": path,
            "resolved": os.path.normcase(os.path.normpath(resolved)) if resolved else "",
        }
        try:
            stat = os.stat(resolved)
            item.update({
                "exists": True,
                "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
                "size": int(stat.st_size),
            })
        except Exception:
            item.update({"exists": False, "mtime_ns": None, "size": None})
        signature.append(item)
    return signature


def _cine_change_fingerprint(payload: Dict[str, Any]) -> str:
    try:
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        encoded = repr(payload)
    return hashlib.sha256(encoded.encode("utf-8", errors="replace")).hexdigest()


class IAMCCS_CineReferenceBoard:
    """Filmmaker-facing reference board for Cine FLF and multi-shot workflows."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_paths": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 768, "min": 0, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 432, "min": 0, "max": 8192, "step": 1}),
                "interpolation": (["lanczos", "nearest", "bilinear", "bicubic", "area", "nearest-exact"], {"default": "lanczos"}),
                "resize_method": (["crop", "pad", "keep proportion", "stretch"], {"default": "crop"}),
                "multiple_of": ("INT", {"default": 32, "min": 0, "max": 512, "step": 1}),
                "img_compression": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",) * 51
    RETURN_NAMES = ("multi_output",) + tuple(f"image_{i + 1}" for i in range(50))
    FUNCTION = "load_images"
    CATEGORY = "IAMCCS/Cine/01 Reference Board"

    @classmethod
    def IS_CHANGED(cls, image_paths, width, height, interpolation, resize_method, multiple_of, img_compression):
        return _cine_change_fingerprint({
            "node": cls.__name__,
            "image_paths": str(image_paths or ""),
            "image_signature": _cine_image_path_signature(image_paths),
            "width": int(width),
            "height": int(height),
            "interpolation": str(interpolation),
            "resize_method": str(resize_method),
            "multiple_of": int(multiple_of),
            "img_compression": int(img_compression),
        })

    def resize_image(self, image, width, height, resize_method="crop", interpolation="lanczos", multiple_of=32):
        max_resolution = 8192
        _, oh, ow, _ = image.shape
        x = y = x2 = y2 = 0
        pad_left = pad_right = pad_top = pad_bottom = 0

        width = int(width) if int(width) > 0 else int(ow)
        height = int(height) if int(height) > 0 else int(oh)

        if multiple_of and int(multiple_of) > 1:
            multiple = int(multiple_of)
            width = max(multiple, width - (width % multiple))
            height = max(multiple, height - (height % multiple))

        if resize_method in {"keep proportion", "pad"}:
            if width == 0 and oh < height:
                width = max_resolution
            elif width == 0 and oh >= height:
                width = ow
            if height == 0 and ow < width:
                height = max_resolution
            elif height == 0 and ow >= width:
                height = oh
            ratio = min(width / ow, height / oh)
            new_width = max(1, round(ow * ratio))
            new_height = max(1, round(oh * ratio))
            if resize_method == "pad":
                pad_left = (width - new_width) // 2
                pad_right = width - new_width - pad_left
                pad_top = (height - new_height) // 2
                pad_bottom = height - new_height - pad_top
            width = new_width
            height = new_height
        elif resize_method == "crop":
            ratio = max(width / ow, height / oh)
            new_width = max(1, round(ow * ratio))
            new_height = max(1, round(oh * ratio))
            x = max(0, (new_width - width) // 2)
            y = max(0, (new_height - height) // 2)
            x2 = x + width
            y2 = y + height
            width = new_width
            height = new_height

        outputs = image.permute(0, 3, 1, 2)
        if interpolation == "lanczos":
            outputs = comfy.utils.lanczos(outputs, width, height)
        else:
            outputs = F.interpolate(outputs, size=(height, width), mode=interpolation)

        if resize_method == "pad" and (pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0):
            outputs = F.pad(outputs, (pad_left, pad_right, pad_top, pad_bottom), value=0)

        outputs = outputs.permute(0, 2, 3, 1)
        if resize_method == "crop" and (x2 > x and y2 > y):
            outputs = outputs[:, y:y2, x:x2, :]

        if multiple_of and int(multiple_of) > 1:
            multiple = int(multiple_of)
            h = outputs.shape[1]
            w = outputs.shape[2]
            crop_h = h - (h % multiple)
            crop_w = w - (w % multiple)
            if crop_h > 0 and crop_w > 0 and (crop_h != h or crop_w != w):
                oy = (h - crop_h) // 2
                ox = (w - crop_w) // 2
                outputs = outputs[:, oy : oy + crop_h, ox : ox + crop_w, :]

        return torch.clamp(outputs, 0, 1)

    def load_images(self, image_paths, width, height, interpolation, resize_method, multiple_of, img_compression):
        results = []
        valid_paths = _cine_reference_paths_from_text(image_paths)

        for path in valid_paths:
            try:
                full_path = path
                if not os.path.exists(full_path):
                    full_path = os.path.join(folder_paths.get_input_directory(), path)
                if not os.path.exists(full_path):
                    print(f"IAMCCS Cine Reference Board warning: image path not found: {path}")
                    continue

                image = Image.open(full_path)
                image = ImageOps.exif_transpose(image).convert("RGB")
                image_np = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)[None,]
                image_tensor = self.resize_image(image_tensor, width, height, resize_method, interpolation, multiple_of)

                if int(img_compression) > 0:
                    img_np = (image_tensor[0].numpy() * 255).clip(0, 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_np)
                    img_byte_arr = io.BytesIO()
                    img_pil.save(img_byte_arr, format="JPEG", quality=max(1, 100 - int(img_compression)))
                    img_pil = Image.open(img_byte_arr).convert("RGB")
                    image_tensor = torch.from_numpy(np.array(img_pil).astype(np.float32) / 255.0)[None,]

                results.append(image_tensor)
            except Exception as exc:
                print(f"IAMCCS Cine Reference Board error loading {path}: {exc}")

        if results:
            first_shape = results[0].shape
            if all(r.shape == first_shape for r in results):
                multi_output = torch.cat(results, dim=0)
            else:
                target_h = int(first_shape[1])
                target_w = int(first_shape[2])
                batch_safe = []
                for r in results:
                    if r.shape != first_shape:
                        rr = r.permute(0, 3, 1, 2)
                        if interpolation == "lanczos":
                            rr = comfy.utils.lanczos(rr, target_w, target_h)
                        else:
                            rr = F.interpolate(rr, size=(target_h, target_w), mode=interpolation)
                        r = rr.permute(0, 2, 3, 1)
                    batch_safe.append(torch.clamp(r, 0, 1))
                print("IAMCCS Cine Reference Board warning: references had different dimensions; resized multi_output to a batch-safe tensor.")
                multi_output = torch.cat(batch_safe, dim=0)
        else:
            multi_output = torch.zeros((1, 64, 64, 3))
            results = [multi_output]

        padded = results + [torch.zeros((1, 64, 64, 3))] * (50 - len(results))
        return (multi_output, *padded[:50])


class IAMCCS_CineLTXSequencer:
    """Cine timeline wrapper around the proven LTX guide insertion logic."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "multi_input": ("IMAGE",),
                "timeline_data": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Edited by the Cine FLF timeline UI. JSON or lines: second | ref | strength | label | camera note",
                }),
                "duration_seconds": ("FLOAT", {"default": 8.0, "min": 0.01, "max": 36000.0, "step": 0.01}),
                "frame_rate": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1}),
                "fallback_num_images": ("INT", {"default": 2, "min": 1, "max": 50, "step": 1}),
                "fallback_strength": ("FLOAT", {"default": 0.82, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tail_safety_frames": ("INT", {"default": 1, "min": 0, "max": 256, "step": 1}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("positive", "negative", "latent", "report")
    FUNCTION = "execute"
    CATEGORY = "IAMCCS/Cine/02 Single Generation"

    @staticmethod
    def _timeline_metadata(timeline_data: str) -> Dict[str, Any]:
        text = str(timeline_data or "").strip()
        if not text:
            return {}
        try:
            data = json.loads(text)
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        meta = data.get("metadata")
        if isinstance(meta, dict):
            merged = dict(data)
            merged.update(meta)
            return merged
        return data

    @staticmethod
    def _parse_timeline(timeline_data: str, duration_seconds: float, fps: int, fallback_num_images: int, fallback_strength: float) -> List[Dict[str, Any]]:
        text = str(timeline_data or "").strip()
        keyframes: List[Dict[str, Any]] = []

        if text:
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    rows = data.get("keyframes") or data.get("rows") or data.get("timeline") or []
                else:
                    rows = data
                if isinstance(rows, list):
                    for idx, row in enumerate(rows):
                        if not isinstance(row, dict):
                            continue
                        second = _safe_float(row.get("second", row.get("time", row.get("seconds", 0.0))), 0.0)
                        ref = _safe_int(row.get("ref", row.get("reference", row.get("reference_index", idx + 1))), idx + 1)
                        strength = _clamp(row.get("force", row.get("strength", fallback_strength)), 0.0, 1.0, fallback_strength)
                        keyframes.append({
                            "second": second,
                            "reference_index": max(1, min(MAX_CINE_ITEMS, ref)),
                            "strength": strength,
                            "label": str(row.get("label", f"key_{idx + 1}")),
                            "camera": str(row.get("camera", row.get("camera_note", ""))),
                        })
            except Exception:
                keyframes = []

            if not keyframes:
                for idx, raw_line in enumerate(text.splitlines()):
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = _split_pipe(line, 5)
                    second = _safe_float(parts[0], 0.0)
                    ref = _safe_int(parts[1], idx + 1)
                    strength = _clamp(parts[2], 0.0, 1.0, fallback_strength)
                    keyframes.append({
                        "second": second,
                        "reference_index": max(1, min(MAX_CINE_ITEMS, ref)),
                        "strength": strength,
                        "label": parts[3] or f"key_{idx + 1}",
                        "camera": parts[4],
                    })

        if not keyframes:
            count = max(1, min(MAX_CINE_ITEMS, int(fallback_num_images)))
            if count == 1:
                seconds = [0.0]
            else:
                seconds = [duration_seconds * (i / (count - 1)) for i in range(count)]
            keyframes = [
                {
                    "second": sec,
                    "reference_index": idx + 1,
                    "strength": float(fallback_strength),
                    "label": f"reference_{idx + 1}",
                    "camera": "auto spaced cine guide",
                }
                for idx, sec in enumerate(seconds)
            ]

        clean = []
        for idx, row in enumerate(keyframes[:MAX_CINE_ITEMS]):
            second = max(0.0, _safe_float(row.get("second"), 0.0))
            raw_frame = row.get("frame", row.get("insert_frame", None))
            frame = _safe_int(raw_frame, int(round(second * max(1, int(fps))))) if raw_frame is not None else int(round(second * max(1, int(fps))))
            clean.append({
                "second": second,
                "frame": frame,
                "reference_index": max(1, min(MAX_CINE_ITEMS, _safe_int(row.get("reference_index"), idx + 1))),
                "strength": _clamp(row.get("strength"), 0.0, 1.0, fallback_strength),
                "label": _normalise_label(str(row.get("label", "")), f"key_{idx + 1}"),
                "camera": str(row.get("camera", "")),
            })
        return sorted(clean, key=lambda item: ((10**9 if int(item["frame"]) < 0 else int(item["frame"])), int(item["reference_index"])))

    @classmethod
    def execute(cls, positive, negative, vae, latent, multi_input, timeline_data, duration_seconds, frame_rate, fallback_num_images, fallback_strength, tail_safety_frames):
        meta = cls._timeline_metadata(timeline_data)
        widget_duration_seconds = float(duration_seconds)
        widget_frame_rate = int(frame_rate)
        duration_seconds = _safe_float(meta.get("duration_seconds", meta.get("duration", duration_seconds)), float(duration_seconds))
        frame_rate = _safe_int(meta.get("frame_rate", meta.get("fps", frame_rate)), int(frame_rate))
        fallback_num_images = _safe_int(meta.get("reference_count", meta.get("images_loaded", fallback_num_images)), int(fallback_num_images))
        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"].clone()
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"].clone()
        else:
            batch, _, latent_frames, _, _ = latent_image.shape
            noise_mask = torch.ones((batch, 1, latent_frames, 1, 1), dtype=torch.float32, device=latent_image.device)

        _, _, latent_length, latent_height, latent_width = latent_image.shape
        batch_size = int(multi_input.shape[0]) if multi_input is not None and torch.is_tensor(multi_input) else 0
        time_scale_factor = int(scale_factors[0]) if scale_factors else 8
        latent_pixel_frames = max(1, (int(latent_length) - 1) * max(1, time_scale_factor) + 1)
        requested_pixel_frames = max(1, int(round(float(duration_seconds) * max(1, int(frame_rate)))))
        pixel_frame_count = min(requested_pixel_frames, latent_pixel_frames)
        max_frame = max(0, pixel_frame_count - 1 - int(tail_safety_frames))
        keyframes = cls._parse_timeline(timeline_data, float(duration_seconds), int(frame_rate), int(fallback_num_images), float(fallback_strength))
        timeline_hash = hashlib.sha1(str(timeline_data or "").encode("utf-8", errors="ignore")).hexdigest()[:12]
        image_shape = [int(v) for v in multi_input.shape] if multi_input is not None and torch.is_tensor(multi_input) else []
        latent_shape = [int(v) for v in latent_image.shape] if torch.is_tensor(latent_image) else []
        _cine_debug(
            "[IAMCCS CineDebug:CineLTXSequencer] "
            f"timeline_hash={timeline_hash} timeline_chars={len(str(timeline_data or ''))} "
            f"duration={float(duration_seconds):.3f}s fps={int(frame_rate)} "
            f"widget_duration={widget_duration_seconds:.3f}s widget_fps={widget_frame_rate} "
            f"duration_source={'timeline_metadata' if meta else 'widget'} "
            f"references_loaded={batch_size} multi_input_shape={image_shape} latent_shape={latent_shape} "
            f"latent_pixel_frames={int(latent_pixel_frames)} requested_pixel_frames={int(requested_pixel_frames)} "
            f"max_frame={int(max_frame)} keyframes={len(keyframes)}"
        )
        for idx, key in enumerate(keyframes[:30]):
            _cine_debug(
                "[IAMCCS CineDebug:CineLTXSequencer] "
                f"key[{idx:02d}] t={float(key.get('second', 0.0)):.3f}s "
                f"frame={int(key.get('frame', 0))} ref={int(key.get('reference_index', 1))} "
                f"strength={float(key.get('strength', 0.0)):.3f} label={key.get('label')} "
                f"camera={str(key.get('camera', ''))[:120]!r}"
            )
        if len(keyframes) > 30:
            _cine_debug(f"[IAMCCS CineDebug:CineLTXSequencer] Keyframe log truncated: {len(keyframes) - 30} more keyframes.")

        applied = []
        skipped = []
        for idx, key in enumerate(keyframes):
            ref_idx = int(key["reference_index"])
            if ref_idx > batch_size:
                skipped.append({"label": key["label"], "reason": "reference index not loaded", "reference_index": ref_idx})
                _cine_debug(
                    "[IAMCCS CineDebug:CineLTXSequencer] SKIP "
                    f"label={key['label']} ref={ref_idx} reason=reference index not loaded batch={batch_size}"
                )
                continue
            img = multi_input[ref_idx - 1 : ref_idx]
            if img is None:
                skipped.append({"label": key["label"], "reason": "empty image", "reference_index": ref_idx})
                _cine_debug(f"[IAMCCS CineDebug:CineLTXSequencer] SKIP label={key['label']} ref={ref_idx} reason=empty image")
                continue

            requested_frame = int(key["frame"])
            f_idx = requested_frame if requested_frame < 0 else min(max(0, requested_frame), max_frame)
            strength = float(key["strength"])
            image_1, t = LTXVAddGuide.encode(vae, latent_width, latent_height, img, scale_factors)
            frame_idx, latent_idx = LTXVAddGuide.get_latent_index(positive, latent_length, len(image_1), f_idx, scale_factors)
            if latent_idx + t.shape[2] > latent_length:
                skipped.append({"label": key["label"], "reason": "guide exceeds latent length", "frame": f_idx})
                _cine_debug(
                    "[IAMCCS CineDebug:CineLTXSequencer] SKIP "
                    f"label={key['label']} frame={f_idx} latent_idx={latent_idx} guide_frames={t.shape[2]} "
                    f"latent_length={latent_length} reason=guide exceeds latent length"
                )
                continue

            positive, negative, latent_image, noise_mask = LTXVAddGuide.append_keyframe(
                positive,
                negative,
                frame_idx,
                latent_image,
                noise_mask,
                t,
                strength,
                scale_factors,
            )
            if _append_guide_attention_entry is not None:
                pre_filter_count = t.shape[2] * t.shape[3] * t.shape[4]
                guide_latent_shape = list(t.shape[2:])
                positive, negative = _append_guide_attention_entry(
                    positive,
                    negative,
                    pre_filter_count,
                    guide_latent_shape,
                    strength=strength,
                )
            applied.append({
                "label": key["label"],
                "reference_index": ref_idx,
                "second": float(key["second"]),
                "frame": int(f_idx),
                "strength": strength,
                "camera": key.get("camera", ""),
            })
            _cine_debug(
                "[IAMCCS CineDebug:CineLTXSequencer] APPLIED "
                f"label={key['label']} ref={ref_idx} second={float(key['second']):.3f}s "
                f"requested_frame={requested_frame} applied_frame={int(f_idx)} strength={strength:.3f}"
            )

        _cine_debug(
            "[IAMCCS CineDebug:CineLTXSequencer] summary "
            f"applied={len(applied)} skipped={len(skipped)}"
        )

        report = _json_report({
            "node": "IAMCCS_CineLTXSequencer",
            "mode": "single_generation_flf_timeline",
            "duration_seconds": float(duration_seconds),
            "frame_rate": int(frame_rate),
            "widget_duration_seconds": float(widget_duration_seconds),
            "widget_frame_rate": int(widget_frame_rate),
            "duration_source": "timeline_metadata" if meta else "widget",
            "references_loaded": batch_size,
            "latent_pixel_frames": int(latent_pixel_frames),
            "applied_keyframes": applied,
            "skipped_keyframes": skipped,
            "truth": "PromptRelay controls temporal text; this node applies image guide keyframes.",
        })
        return positive, negative, {"samples": latent_image, "noise_mask": noise_mask}, report


class IAMCCS_CineAllInOneFLFEngine(IAMCCS_CineLTXSequencer):
    """ShotPlanner-driven FLF engine that applies guides through native LTXVAddGuide nodes.

    This is an experimental AllInOne-style path: it keeps the compact ShotPlanner UI,
    but each guide is applied by calling ComfyUI's LTXVAddGuide.execute exactly as an
    explicit node chain would do.
    """

    CATEGORY = "IAMCCS/Cine/02 Single Generation"

    @classmethod
    def INPUT_TYPES(cls):
        data = IAMCCS_CineLTXSequencer.INPUT_TYPES()
        data["required"]["duration_seconds"] = ("INT", {"default": 19, "min": 1, "max": 36000, "step": 1})
        return data

    @classmethod
    def execute(cls, positive, negative, vae, latent, multi_input, timeline_data, duration_seconds, frame_rate, fallback_num_images, fallback_strength, tail_safety_frames):
        batch_size = int(multi_input.shape[0]) if torch.is_tensor(multi_input) and multi_input.ndim == 4 else 0
        if batch_size <= 0:
            report = _json_report({
                "node": "IAMCCS_CineAllInOneFLFEngine",
                "mode": "allinone_native_ltxvaddguide_chain",
                "applied_keyframes": [],
                "skipped_keyframes": [{"reason": "multi_input is empty or invalid"}],
                "truth": "No image guides were applied because no valid reference batch reached the engine.",
            })
            return positive, negative, latent, report

        fps = max(1, int(frame_rate))
        total_frames = max(1, int(round(float(duration_seconds) * fps)))
        max_frame = max(0, total_frames - 1 - max(0, int(tail_safety_frames)))
        keyframes = cls._parse_timeline(timeline_data, float(duration_seconds), fps, int(fallback_num_images), float(fallback_strength))
        applied = []
        skipped = []
        current_latent = latent

        for key in keyframes:
            ref_idx = int(key["reference_index"])
            if ref_idx > batch_size:
                skipped.append({"label": key["label"], "reason": "reference index not loaded", "reference_index": ref_idx})
                continue
            img = multi_input[ref_idx - 1 : ref_idx]
            if not torch.is_tensor(img) or img.shape[0] <= 0:
                skipped.append({"label": key["label"], "reason": "empty image", "reference_index": ref_idx})
                continue

            requested_frame = int(key["frame"])
            frame_idx = requested_frame if requested_frame < 0 else min(max(0, requested_frame), max_frame)
            strength = float(key["strength"])
            try:
                out = LTXVAddGuide.execute(positive, negative, vae, current_latent, img, frame_idx, strength)
                positive, negative, current_latent = out[0], out[1], out[2]
            except Exception as exc:
                skipped.append({
                    "label": key["label"],
                    "reason": f"native LTXVAddGuide failed: {exc}",
                    "frame": int(frame_idx),
                    "reference_index": ref_idx,
                })
                continue

            applied.append({
                "label": key["label"],
                "reference_index": ref_idx,
                "second": float(key["second"]),
                "frame": int(frame_idx),
                "strength": strength,
                "camera": key.get("camera", ""),
            })

        report = _json_report({
            "node": "IAMCCS_CineAllInOneFLFEngine",
            "mode": "allinone_native_ltxvaddguide_chain",
            "duration_seconds": float(duration_seconds),
            "frame_rate": fps,
            "references_loaded": batch_size,
            "applied_keyframes": applied,
            "skipped_keyframes": skipped,
            "truth": "Experimental path: ShotPlanner timeline is applied through native LTXVAddGuide.execute, matching the AllInOne explicit guide-chain mechanism more closely than the compact legacy sequencer.",
        })
        return positive, negative, current_latent, report


class IAMCCS_CinePromptRelayTimeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "global_prompt": ("STRING", {
                    "default": "cinematic scene, natural motion, no slideshow, coherent camera movement",
                    "multiline": True,
                }),
                "timeline_data": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Edited by the Cine PromptRelay timeline UI. JSON or lines: seconds | local prompt | camera note",
                }),
                "duration_seconds": ("FLOAT", {"default": 8.0, "min": 0.01, "max": 36000.0, "step": 0.01}),
                "frame_rate": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1}),
                "ltx_round_mode": (["up_8n_plus_1", "nearest_8n_plus_1", "none"], {"default": "up_8n_plus_1"}),
                "fallback_segments": ("INT", {"default": 3, "min": 1, "max": 32, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("global_prompt", "local_prompts", "segment_lengths", "max_frames", "report")
    FUNCTION = "build"
    CATEGORY = "IAMCCS/Cine/02 Single Generation"

    @staticmethod
    def _round_frames(frames: int, mode: str) -> int:
        frames = max(1, int(frames))
        if mode == "none":
            return frames
        rem = (frames - 1) % 8
        if rem == 0:
            return frames
        down = max(1, frames - rem)
        up = frames + (8 - rem)
        if mode == "nearest_8n_plus_1":
            return up if (up - frames) <= (frames - down) else down
        return up

    @staticmethod
    def _parse_segments(timeline_data: str, fallback_segments: int, duration_seconds: float) -> List[Dict[str, Any]]:
        text = str(timeline_data or "").strip()
        segments: List[Dict[str, Any]] = []
        if text:
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    rows = data.get("segments") or data.get("rows") or data.get("timeline") or []
                else:
                    rows = data
                if isinstance(rows, list):
                    for idx, row in enumerate(rows):
                        if not isinstance(row, dict):
                            continue
                        seconds = max(0.01, _safe_float(row.get("seconds", row.get("duration", 1.0)), 1.0))
                        prompt = str(row.get("prompt", row.get("local_prompt", "cinematic motion"))).strip()
                        camera = str(row.get("camera", "")).strip()
                        if camera and camera.lower() not in prompt.lower():
                            prompt = f"{prompt}, {camera}"
                        segments.append({"seconds": seconds, "prompt": prompt or "cinematic motion"})
            except Exception:
                segments = []
            if not segments:
                for raw_line in text.splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    seconds, prompt, camera = _split_pipe(line, 3)
                    duration = max(0.01, _safe_float(seconds, 1.0))
                    local = prompt.strip() or "cinematic motion"
                    if camera.strip():
                        local = f"{local}, {camera.strip()}"
                    segments.append({"seconds": duration, "prompt": local})
        if not segments:
            count = max(1, int(fallback_segments))
            seg_seconds = max(0.01, float(duration_seconds) / count)
            segments = [{"seconds": seg_seconds, "prompt": f"cinematic beat {idx + 1}, natural motion"} for idx in range(count)]
        return segments

    def build(self, global_prompt, timeline_data, duration_seconds, frame_rate, ltx_round_mode, fallback_segments):
        fps = max(1, int(frame_rate))
        max_frames = self._round_frames(int(round(float(duration_seconds) * fps)), str(ltx_round_mode))
        segments = self._parse_segments(timeline_data, int(fallback_segments), float(duration_seconds))
        total_seconds = sum(float(seg["seconds"]) for seg in segments)
        if total_seconds <= 0:
            total_seconds = float(duration_seconds)
        lengths = []
        for seg in segments:
            ratio = float(seg["seconds"]) / total_seconds
            lengths.append(max(1, int(round(ratio * max_frames))))
        diff = max_frames - sum(lengths)
        if lengths:
            lengths[-1] = max(1, lengths[-1] + diff)
        local_prompts = " | ".join(seg["prompt"].strip() for seg in segments if seg.get("prompt"))
        segment_lengths = ",".join(str(int(length)) for length in lengths)
        report = _json_report({
            "node": "IAMCCS_CinePromptRelayTimeline",
            "mode": "prompt_relay_timeline_helper",
            "duration_seconds": float(duration_seconds),
            "frame_rate": fps,
            "max_frames": int(max_frames),
            "segments": segments,
            "segment_lengths": lengths,
            "connect_to": "PromptRelayEncodeTimeline or PromptRelayEncode",
        })
        return str(global_prompt or ""), local_prompts, segment_lengths, int(max_frames), report


class IAMCCS_CineShotboardTimelinePro:
    """Filmmaker-facing shotboard that drives FLF guides and PromptRelay timing from one table."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "global_prompt": ("STRING", {
                    "default": "one continuous cinematic shot, coherent motion, no slideshow, no cross dissolve",
                    "multiline": True,
                }),
                "timeline_data": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Edited by the Cine Shotboard Pro UI. Rows contain time, image reference, force, label, camera move, transition, note and guide enable.",
                }),
                "duration_seconds": ("FLOAT", {"default": 8.0, "min": 0.01, "max": 36000.0, "step": 0.01}),
                "frame_rate": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1}),
                "guide_policy": (["safe_core_guides", "prompt_only", "every_checked_row"], {"default": "safe_core_guides"}),
                "min_guide_gap_seconds": ("FLOAT", {"default": 1.75, "min": 0.0, "max": 60.0, "step": 0.05}),
                "max_guides": ("INT", {"default": 5, "min": 0, "max": 50, "step": 1}),
                "default_force": ("FLOAT", {"default": 0.22, "min": 0.0, "max": 1.0, "step": 0.01}),
                "promptrelay_epsilon": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 10.0, "step": 0.01}),
                "ltx_round_mode": (["up_8n_plus_1", "nearest_8n_plus_1", "none"], {"default": "up_8n_plus_1"}),
                "tail_safety_frames": ("INT", {"default": 1, "min": 0, "max": 256, "step": 1}),
                "image_paths": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional internal reference images. Edited by the Shotboard Pro UI. If empty, use the connected multi_input from Cine Reference Board.",
                }),
                "image_width": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 32}),
                "image_height": ("INT", {"default": 432, "min": 64, "max": 8192, "step": 32}),
            },
            "optional": {
                "multi_input": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "STRING", "STRING", "STRING", "STRING", "INT", "FLOAT", "STRING", "IMAGE", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("positive", "negative", "latent", "flf_timeline", "global_prompt", "local_prompts", "segment_lengths", "max_frames", "promptrelay_epsilon", "report", "multi_output", "image_1", "duration_seconds_int", "frame_rate_int")
    FUNCTION = "execute"
    CATEGORY = "IAMCCS/Cine/02 Single Generation"

    @classmethod
    def IS_CHANGED(
        cls,
        positive=None,
        negative=None,
        vae=None,
        latent=None,
        global_prompt=None,
        timeline_data=None,
        duration_seconds=None,
        frame_rate=None,
        guide_policy=None,
        min_guide_gap_seconds=None,
        max_guides=None,
        default_force=None,
        promptrelay_epsilon=None,
        ltx_round_mode=None,
        tail_safety_frames=None,
        image_paths=None,
        image_width=None,
        image_height=None,
        multi_input=None,
        **kwargs,
    ):
        return _cine_change_fingerprint({
            "node": cls.__name__,
            "global_prompt": str(global_prompt or ""),
            "timeline_data": str(timeline_data or ""),
            "duration_seconds": float(duration_seconds or 0),
            "frame_rate": int(frame_rate or 0),
            "guide_policy": str(guide_policy or ""),
            "min_guide_gap_seconds": float(min_guide_gap_seconds or 0),
            "max_guides": int(max_guides or 0),
            "default_force": float(default_force or 0),
            "promptrelay_epsilon": float(promptrelay_epsilon or 0),
            "ltx_round_mode": str(ltx_round_mode or ""),
            "tail_safety_frames": int(tail_safety_frames or 0),
            "image_paths": str(image_paths or ""),
            "image_signature": _cine_image_path_signature(image_paths),
            "image_width": int(image_width or 0),
            "image_height": int(image_height or 0),
        })

    @staticmethod
    def _as_bool(value: Any, default: bool = True) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        return str(value).strip().lower() not in {"0", "false", "no", "off", "disabled"}

    @staticmethod
    def _first_non_empty(*values: Any) -> str:
        for value in values:
            text = str(value or "").strip()
            if text:
                return text
        return ""

    @staticmethod
    def _round_frames(frames: int, mode: str) -> int:
        return IAMCCS_CinePromptRelayTimeline._round_frames(frames, mode)

    @classmethod
    def _parse_rows(cls, timeline_data: str, duration_seconds: float, default_force: float) -> List[Dict[str, Any]]:
        text = str(timeline_data or "").strip()
        rows: List[Dict[str, Any]] = []
        raw_rows: Any = []
        if text:
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    raw_rows = data.get("rows") or data.get("keyframes") or data.get("shotboard") or data.get("timeline") or []
                else:
                    raw_rows = data
            except Exception:
                raw_rows = []

        def _canonical_relay_prompt(row: Dict[str, Any]) -> str:
            return cls._first_non_empty(
                row.get("relay_prompt"),
                row.get("local_prompt"),
                row.get("prompt_beat"),
                row.get("beat_prompt"),
                row.get("video_prompt"),
                row.get("action_prompt"),
                row.get("prompt"),
                row.get("localPrompt"),
                row.get("relayPrompt"),
                row.get("promptLocal"),
            )

        has_any_canonical_relay = any(
            isinstance(row, dict) and bool(_canonical_relay_prompt(row))
            for row in raw_rows
        ) if isinstance(raw_rows, list) else False

        if isinstance(raw_rows, list):
            for idx, row in enumerate(raw_rows):
                if not isinstance(row, dict):
                    continue
                second = _safe_float(row.get("second", row.get("time", row.get("seconds", 0.0))), 0.0)
                raw_frame = row.get("frame", row.get("insert_frame", None))
                ref = _safe_int(row.get("ref", row.get("image_ref", row.get("reference_index", idx + 1))), idx + 1)
                force = _clamp(row.get("force", row.get("strength", default_force)), 0.0, 1.0, default_force)
                note = str(row.get("note", row.get("camera_note", row.get("prompt", "")))).strip()
                camera = str(row.get("camera", row.get("camera_move", "cinematic motion"))).strip()
                transition = str(row.get("transition", row.get("transition_intent", "continuous_motion"))).strip() or "continuous_motion"
                label = _normalise_label(str(row.get("label", row.get("shot_label", ""))), f"shot_{idx + 1}")
                legacy_modifiers = cls._as_bool(row.get("use_relay_modifiers", row.get("use_camera_transition_in_relay", row.get("relay_modifiers", False))), False)
                # Relay text must come from explicit local-prompt fields only.
                # Notes are private/technical shot notes and must not silently
                # become PromptRelay segments when the Relay toggles are enabled.
                relay_prompt = _canonical_relay_prompt(row)
                rows.append({
                    "second": max(0.0, second),
                    "frame": _safe_int(raw_frame, int(round(max(0.0, second) * 24))) if raw_frame is not None else None,
                    "ref": max(1, min(MAX_CINE_ITEMS, ref)),
                    "force": force,
                    "label": label,
                    "camera": camera,
                    "transition": transition,
                    "note": note,
                    "use_guide": cls._as_bool(row.get("use_guide", row.get("guide", True)), True),
                    "use_prompt": cls._as_bool(row.get("use_prompt", row.get("use_relay", row.get("relay", row.get("prompt_relay", True)))), True),
                    "relay_prompt": relay_prompt,
                    "use_relay_modifiers": legacy_modifiers,
                    "camera_relay_mode": str(row.get("camera_relay_mode", row.get("camera_prompt_mode", "before" if legacy_modifiers else "off")) or "off").strip(),
                    "transition_relay_mode": str(row.get("transition_relay_mode", row.get("transition_prompt_mode", "safe_only" if legacy_modifiers else "off")) or "off").strip(),
                    "relay_addon_position": str(row.get("relay_addon_position", row.get("addon_position", "after")) or "after").strip(),
                    "relay_modifier_text": str(row.get("relay_modifier_text", row.get("modifier_text", row.get("relay_addon", ""))) or "").strip(),
                })

        if not rows and text:
            for idx, raw_line in enumerate(text.splitlines()):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                second, ref, force, label, camera, transition, note = _split_pipe(line, 7)
                rows.append({
                    "second": max(0.0, _safe_float(second, idx * 2.0)),
                    "ref": max(1, min(MAX_CINE_ITEMS, _safe_int(ref, idx + 1))),
                    "force": _clamp(force, 0.0, 1.0, default_force),
                    "label": _normalise_label(label, f"shot_{idx + 1}"),
                    "camera": camera.strip() or "cinematic motion",
                    "transition": transition.strip() or "continuous_motion",
                    "note": note.strip(),
                    "use_guide": True,
                    "use_prompt": True,
                    "use_relay_modifiers": False,
                    "camera_relay_mode": "off",
                    "transition_relay_mode": "off",
                    "relay_addon_position": "after",
                    "relay_modifier_text": "",
                })

        if not rows:
            rows = [
                {"second": 0.0, "ref": 1, "force": 0.78, "label": "opening_ref", "camera": "slow push-in", "transition": "continuous_motion", "note": "start from the first reference", "use_guide": True, "use_prompt": True, "use_relay_modifiers": False, "camera_relay_mode": "before", "transition_relay_mode": "off", "relay_addon_position": "after", "relay_modifier_text": ""},
                {"second": max(0.1, duration_seconds * 0.55), "ref": 2, "force": default_force, "label": "middle_ref", "camera": "continuous dolly-in", "transition": "continuous_motion", "note": "midpoint visual target", "use_guide": True, "use_prompt": True, "use_relay_modifiers": False, "camera_relay_mode": "before", "transition_relay_mode": "off", "relay_addon_position": "after", "relay_modifier_text": ""},
                {"second": max(0.2, duration_seconds - 0.4), "ref": 3, "force": default_force, "label": "ending_ref", "camera": "slow push-in", "transition": "continuous_motion", "note": "last visual target", "use_guide": True, "use_prompt": True, "use_relay_modifiers": False, "camera_relay_mode": "before", "transition_relay_mode": "off", "relay_addon_position": "after", "relay_modifier_text": ""},
            ]

        rows = sorted(rows[:MAX_CINE_ITEMS], key=lambda item: (float(item["second"]), int(item["ref"])))
        return rows

    @staticmethod
    def _normalise_mode(value: Any, allowed: set, fallback: str) -> str:
        raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "camera_before": "before",
            "camera_after": "after",
            "before_local": "before",
            "after_local": "after",
            "safe": "safe_only",
            "safeonly": "safe_only",
            "safe_only": "safe_only",
            "append_all": "append",
            "add_on_before": "before",
            "add_on_after": "after",
            "addon_before": "before",
            "addon_after": "after",
        }
        raw = aliases.get(raw, raw)
        return raw if raw in allowed else fallback

    @staticmethod
    def _camera_modifier_part(row: Dict[str, Any]) -> str:
        camera = str(row.get("camera", "") or "").strip().lower().replace("_", " ")
        camera_phrases = {
            "locked-off camera": "locked-off camera, stable framing, subtle subject motion",
            "slow push-in": "slow push-in, gentle forward camera movement",
            "easy-in push": "easy-in push, gradual acceleration into the move",
            "continuous dolly-in": "continuous forward dolly, physical travel and growing parallax",
            "slow pull-back": "slow pull-back, widening space and revealing context",
            "macro push-in": "macro push-in, sharp detail, micro-parallax, stable focus",
            "tracking shot": "tracking shot, smooth camera follow-through",
            "lateral tracking": "lateral tracking movement, side parallax",
            "orbit move": "orbiting camera move, curved parallax around the subject",
            "subtle handheld": "subtle handheld energy, natural micro-movement",
            "pan left": "slow pan left, continuous horizontal camera motion",
            "pan right": "slow pan right, continuous horizontal camera motion",
            "tilt up": "tilt up, continuous vertical camera movement",
            "tilt down": "tilt down, continuous vertical camera movement",
            "crane descent": "crane descent, smooth downward camera travel",
            "aerial descent": "aerial descent, smooth downward travel from above",
            "reverse angle": "reverse angle staging, coherent screen direction",
            "over-the-shoulder": "over-the-shoulder staging, foreground and background depth",
            "wide reveal": "wide reveal, expanding composition and environmental context",
        }
        return camera_phrases.get(camera, "")

    @staticmethod
    def _transition_modifier_parts(row: Dict[str, Any], next_row: Optional[Dict[str, Any]] = None, include_hard: bool = False) -> List[str]:
        parts: List[str] = []
        transition = str(row.get("transition", "continuous_motion") or "continuous_motion").strip()
        if transition == "hard_cut":
            if include_hard:
                parts.append("hard cut staging; do not morph identities inside this single shot")
            return parts
        elif transition == "match_cut":
            parts.append("match movement continuity through shape and camera direction")
        elif transition == "soft_morph":
            parts.append("single continuous transformation, avoid visible cross dissolve")
        else:
            parts.append("continuous physical camera movement, no slideshow, no cross dissolve")

        if next_row is not None and transition != "hard_cut":
            nxt = str(next_row.get("label", "next target")).replace("_", " ")
            parts.append(f"move toward {nxt} without flashing reference frames")
        return parts

    @classmethod
    def _relay_modifier_parts(cls, row: Dict[str, Any], next_row: Optional[Dict[str, Any]] = None) -> List[str]:
        parts: List[str] = []
        camera = cls._camera_modifier_part(row)
        if camera:
            parts.append(camera)
        parts.extend(cls._transition_modifier_parts(row, next_row, include_hard=True))
        return parts

    @classmethod
    def _row_prompt(cls, row: Dict[str, Any], next_row: Optional[Dict[str, Any]] = None) -> str:
        relay_prompt = str(row.get("relay_prompt", "") or "").strip()
        if not relay_prompt:
            return ""
        before_parts: List[str] = []
        after_parts: List[str] = []
        base_parts: List[str] = [relay_prompt]

        legacy_modifiers = bool(row.get("use_relay_modifiers", False))
        camera_mode = cls._normalise_mode(row.get("camera_relay_mode", "before" if legacy_modifiers else "off"), {"off", "before", "after"}, "off")
        transition_mode = cls._normalise_mode(row.get("transition_relay_mode", "safe_only" if legacy_modifiers else "off"), {"off", "safe_only", "append"}, "off")
        addon_position = cls._normalise_mode(row.get("relay_addon_position", "after"), {"before", "after"}, "after")

        camera_text = cls._camera_modifier_part(row)
        if camera_text and camera_mode == "before":
            before_parts.append(camera_text)
        elif camera_text and camera_mode == "after":
            after_parts.append(camera_text)

        if transition_mode != "off":
            after_parts.extend(cls._transition_modifier_parts(row, next_row, include_hard=(transition_mode == "append")))

        modifier_text = str(row.get("relay_modifier_text", "") or "").strip()
        if modifier_text:
            if addon_position == "before":
                before_parts.append(modifier_text)
            else:
                after_parts.append(modifier_text)

        parts = before_parts + base_parts + after_parts
        return ", ".join(part for part in parts if part)

    @classmethod
    def _segments_from_rows(cls, rows: List[Dict[str, Any]], duration_seconds: float, fps: int, ltx_round_mode: str) -> Tuple[str, str, int, List[int]]:
        target_frames = max(1, int(round(float(duration_seconds) * max(1, int(fps)))))
        max_frames = cls._round_frames(target_frames, ltx_round_mode)
        # PromptRelay expects pixel-space frame counts. Use the same LTX-compatible
        # frame budget that the backend will use for the latent, so row timing and
        # PromptRelay segments stay aligned after 8n+1 rounding.
        frame_budget = int(max_frames)
        segments = []
        eligible_rows = [row for row in rows if row.get("use_prompt", True)]
        sorted_rows = []
        for row in eligible_rows:
            if float(row.get("second", 0.0)) >= float(duration_seconds):
                continue
            if not str(row.get("relay_prompt", "") or "").strip():
                continue
            sorted_rows.append(row)
        if not sorted_rows:
            return "", "", int(max_frames), []
        for idx, row in enumerate(sorted_rows):
            start = max(0.0, float(row.get("second", 0.0)))
            if idx + 1 < len(sorted_rows):
                end = max(start + 0.05, float(sorted_rows[idx + 1].get("second", duration_seconds)))
            else:
                end = max(start + 0.05, float(duration_seconds))
            dur = max(0.05, end - start)
            prompt = cls._row_prompt(row, sorted_rows[idx + 1] if idx + 1 < len(sorted_rows) else None)
            if not prompt:
                continue
            segments.append({"seconds": dur, "prompt": prompt})

        total = sum(seg["seconds"] for seg in segments) or float(duration_seconds)
        lengths = [max(1, int(round((seg["seconds"] / total) * frame_budget))) for seg in segments]
        diff = frame_budget - sum(lengths)
        if lengths:
            lengths[-1] = max(1, lengths[-1] + diff)
        local_prompts = " | ".join(seg["prompt"] for seg in segments)
        segment_lengths = ",".join(str(int(length)) for length in lengths)
        return local_prompts, segment_lengths, int(max_frames), lengths

    @classmethod
    @staticmethod
    def _log_text_preview(text: Any, limit: int = 220) -> str:
        preview = str(text or "").replace("\n", "\\n").strip()
        if len(preview) > limit:
            preview = preview[: max(0, limit - 3)] + "..."
        return preview

    @staticmethod
    def _log_text_hash(text: Any) -> str:
        return hashlib.sha1(str(text or "").encode("utf-8", errors="ignore")).hexdigest()[:12]

    @classmethod
    def _log_promptrelay_state(
        cls,
        node_name: str,
        rows: List[Dict[str, Any]],
        local_prompts: str,
        segment_lengths: str,
        duration_seconds: float,
        global_prompt: str = "",
        guide_rows: Optional[List[Dict[str, Any]]] = None,
        max_frames: int = 0,
    ) -> None:
        active_rows = [
            row for row in rows
            if row.get("use_prompt", True)
            and str(row.get("relay_prompt", "") or "").strip()
            and float(row.get("second", 0.0)) < float(duration_seconds)
        ]
        skipped_empty = [
            str(row.get("label", f"row_{idx + 1}"))
            for idx, row in enumerate(rows)
            if row.get("use_prompt", True)
            and not str(row.get("relay_prompt", "") or "").strip()
        ]
        local_count = len([part for part in str(local_prompts or "").split("|") if part.strip()])
        length_count = len([part for part in re.split(r"[,;\s]+", str(segment_lengths or "")) if part.strip()])
        guide_rows = guide_rows or []
        local_parts = [part.strip() for part in str(local_prompts or "").split("|") if part.strip()]
        length_parts = [part for part in re.split(r"[,;\s]+", str(segment_lengths or "")) if part.strip()]
        _cine_debug(
            f"[IAMCCS CineDebug:{node_name}] global_hash={cls._log_text_hash(global_prompt)} "
            f"global_preview={cls._log_text_preview(global_prompt, 260)!r}"
        )
        _cine_debug(
            f"[IAMCCS CineDebug:{node_name}] FLF guides={len(guide_rows)} "
            f"guide_refs={[int(row.get('ref', 1)) for row in guide_rows]} "
            f"guide_seconds={[round(float(row.get('second', 0.0)), 3) for row in guide_rows]} "
            f"guide_forces={[round(float(row.get('force', 0.0)), 3) for row in guide_rows]}"
        )
        _cine_debug(
            f"[IAMCCS CineDebug:{node_name}] PromptRelay local-only enabled={bool(str(local_prompts or '').strip())} "
            f"active_rows={len(active_rows)} local_prompts={local_count} segment_lengths={length_count} "
            f"max_frames={int(max_frames or 0)} segment_lengths_value={segment_lengths or '<empty>'} "
            f"local_hash={cls._log_text_hash(local_prompts)}"
        )
        for idx, row in enumerate(rows[:30]):
            raw_prompt = str(row.get("relay_prompt", "") or "").strip()
            _cine_debug(
                f"[IAMCCS CineDebug:{node_name}] Row {idx:02d} "
                f"t={float(row.get('second', 0.0)):.3f}s ref={row.get('ref')} "
                f"force={float(row.get('force', 0.0)):.3f} "
                f"guide={'ON' if row.get('use_guide', True) else 'off'} "
                f"relay={'ON' if row.get('use_prompt', True) else 'off'} "
                f"label={row.get('label')} "
                f"local_empty={not bool(raw_prompt)} "
                f"local_preview={cls._log_text_preview(raw_prompt, 180)!r}"
            )
        if len(rows) > 30:
            _cine_debug(f"[IAMCCS CineDebug:{node_name}] Row log truncated: {len(rows) - 30} more rows.")
        for idx, prompt in enumerate(local_parts[:30]):
            length = length_parts[idx] if idx < len(length_parts) else "<missing>"
            row = active_rows[idx] if idx < len(active_rows) else {}
            _cine_debug(
                f"[IAMCCS CineDebug:{node_name}] PromptRelay segment {idx:02d} "
                f"length={length} source_t={float(row.get('second', 0.0)):.3f}s "
                f"source_ref={row.get('ref', '<unknown>')} source_label={row.get('label', '<unknown>')} "
                f"text={cls._log_text_preview(prompt, 260)!r}"
            )
        if len(local_parts) > 30:
            _cine_debug(f"[IAMCCS CineDebug:{node_name}] Segment log truncated: {len(local_parts) - 30} more segments.")
        if skipped_empty:
            preview = ", ".join(skipped_empty[:12])
            suffix = "..." if len(skipped_empty) > 12 else ""
            _cine_debug(f"[IAMCCS CineDebug:{node_name}] Relay ON rows skipped because Local prompt is empty: {preview}{suffix}")

    @staticmethod
    def _select_guides(rows: List[Dict[str, Any]], guide_policy: str, min_gap: float, max_guides: int) -> List[Dict[str, Any]]:
        if guide_policy == "prompt_only" or int(max_guides) <= 0:
            return []
        candidates = [row for row in rows if row.get("use_guide", True) and float(row.get("force", 0.0)) > 0.0]
        if guide_policy == "safe_core_guides":
            scored = []
            for idx, row in enumerate(candidates):
                score = float(row.get("force", 0.0))
                if idx == 0:
                    score += 0.35
                if idx == len(candidates) - 1:
                    score += 0.25
                transition = str(row.get("transition", ""))
                if transition == "hard_cut":
                    score -= 0.25
                scored.append((score, idx, row))
            scored.sort(key=lambda item: (-item[0], item[1]))
            chosen = []
            for _, _, row in scored:
                if len(chosen) >= int(max_guides):
                    break
                sec = float(row.get("second", 0.0))
                if any(abs(sec - float(existing.get("second", 0.0))) < float(min_gap) for existing in chosen):
                    continue
                chosen.append(row)
            return sorted(chosen, key=lambda item: float(item.get("second", 0.0)))

        chosen = []
        for row in candidates:
            if len(chosen) >= int(max_guides):
                break
            sec = float(row.get("second", 0.0))
            if any(abs(sec - float(existing.get("second", 0.0))) < float(min_gap) for existing in chosen):
                continue
            chosen.append(row)
        return chosen

    @staticmethod
    def _flf_from_rows(
        rows: List[Dict[str, Any]],
        duration_seconds: Optional[float] = None,
        frame_rate: Optional[int] = None,
        reference_count: Optional[int] = None,
    ) -> str:
        keyframes = []
        for row in rows:
            item = {
                "second": float(row.get("second", 0.0)),
                "ref": int(row.get("ref", 1)),
                "strength": float(row.get("force", 0.0)),
                "label": str(row.get("label", "shot")),
                "camera": str(row.get("camera", "")),
                "note": str(row.get("note", "")),
                "transition": str(row.get("transition", "continuous_motion")),
            }
            if row.get("frame") is not None:
                item["frame"] = int(row.get("frame"))
            keyframes.append(item)
        payload: Dict[str, Any] = {"keyframes": keyframes}
        metadata: Dict[str, Any] = {"source": "IAMCCS_CineShotboardPlannerPro"}
        if duration_seconds is not None:
            metadata["duration_seconds"] = float(duration_seconds)
        if frame_rate is not None:
            metadata["frame_rate"] = int(frame_rate)
        if reference_count is not None:
            metadata["reference_count"] = int(reference_count)
        payload["metadata"] = metadata
        return json.dumps(payload, indent=2)

    def execute(self, positive, negative, vae, latent, global_prompt, timeline_data, duration_seconds, frame_rate, guide_policy, min_guide_gap_seconds, max_guides, default_force, promptrelay_epsilon, ltx_round_mode, tail_safety_frames, image_paths, image_width, image_height, multi_input=None):
        fps = max(1, int(frame_rate))
        if str(image_paths or "").strip():
            try:
                loaded = IAMCCS_CineReferenceBoard().load_images(
                    image_paths,
                    int(image_width),
                    int(image_height),
                    "lanczos",
                    "crop",
                    32,
                    0,
                )
                multi_input = loaded[0]
            except Exception as exc:
                print(f"IAMCCS Cine Shotboard Pro warning: could not load internal image_paths: {exc}")

        if multi_input is not None and torch.is_tensor(multi_input):
            multi_output = multi_input
        else:
            multi_output = torch.zeros((1, 64, 64, 3))
        image_1 = multi_output[0:1] if torch.is_tensor(multi_output) and multi_output.shape[0] > 0 else torch.zeros((1, 64, 64, 3))

        rows = self._parse_rows(timeline_data, float(duration_seconds), float(default_force))
        guide_rows = self._select_guides(rows, str(guide_policy), float(min_guide_gap_seconds), int(max_guides))
        reference_count = int(multi_output.shape[0]) if torch.is_tensor(multi_output) else 0
        flf_timeline = self._flf_from_rows(guide_rows, float(duration_seconds), fps, reference_count)
        local_prompts, segment_lengths, max_frames, latent_lengths_preview = self._segments_from_rows(rows, float(duration_seconds), fps, str(ltx_round_mode))
        self._log_promptrelay_state(
            "ShotboardTimelinePro",
            rows,
            local_prompts,
            segment_lengths,
            float(duration_seconds),
            str(global_prompt or ""),
            guide_rows,
            int(max_frames),
        )

        out_positive = positive
        out_negative = negative
        out_latent = latent
        guide_report = "no image guides applied"
        if guide_rows:
            out_positive, out_negative, out_latent, guide_report = IAMCCS_CineLTXSequencer.execute(
                positive,
                negative,
                vae,
                latent,
                multi_input,
                flf_timeline,
                float(duration_seconds),
                fps,
                len(guide_rows),
                float(default_force),
                int(tail_safety_frames),
            )

        warnings = []
        for idx, row in enumerate(rows[1:], start=1):
            prev = rows[idx - 1]
            if float(row["second"]) - float(prev["second"]) < float(min_guide_gap_seconds):
                warnings.append(f"Rows '{prev['label']}' and '{row['label']}' are closer than min_guide_gap_seconds.")
            if float(row.get("force", 0.0)) > 0.55 and row.get("use_guide", True):
                warnings.append(f"Row '{row['label']}' uses a strong WDC-style FLF anchor; keep it intentional and avoid overcrowded guides.")
            if str(row.get("transition")) == "hard_cut" and str(guide_policy) != "prompt_only":
                warnings.append(f"Row '{row['label']}' is marked hard_cut; multigen is usually cleaner than one continuous FLF shot.")
            if str(row.get("transition")) == "hard_cut" and self._normalise_mode(row.get("transition_relay_mode"), {"off", "safe_only", "append"}, "off") == "append":
                warnings.append(f"Row '{row['label']}' appends hard_cut text into PromptRelay; this can create morphing or visual confusion.")

        report = _json_report({
            "node": "IAMCCS_CineShotboardTimelinePro",
            "mode": "shotboard_pro",
            "duration_seconds": float(duration_seconds),
            "frame_rate": fps,
            "guide_policy": str(guide_policy),
            "rows": rows,
            "guide_rows_applied": guide_rows,
            "promptrelay_resolved_local_prompts": local_prompts,
            "promptrelay_enabled": bool(str(local_prompts or "").strip()),
            "promptrelay_segment_lengths": segment_lengths,
            "promptrelay_pixel_lengths": latent_lengths_preview,
            "promptrelay_latent_lengths_preview": latent_lengths_preview,
            "max_frames": int(max_frames),
            "promptrelay_epsilon": float(promptrelay_epsilon),
            "warnings": warnings,
            "guide_report": guide_report,
            "truth": "For continuous camera moves, prefer prompt_only or safe_core_guides. For true hard cuts, use multigen and final concat.",
        })
        return out_positive, out_negative, out_latent, flf_timeline, str(global_prompt or ""), local_prompts, segment_lengths, int(max_frames), float(promptrelay_epsilon), report, multi_output, image_1, int(round(float(duration_seconds))), fps


class IAMCCS_CineShotboardPlannerPro(IAMCCS_CineShotboardTimelinePro):
    """Planner-only Shotboard Pro. Use upstream of PromptRelay to avoid graph cycles."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "global_prompt": ("STRING", {
                    "default": "one continuous cinematic shot, coherent motion, no slideshow, no cross dissolve",
                    "multiline": True,
                }),
                "timeline_data": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Edited by the Cine Shotboard Pro UI. Rows contain time, image reference, force, label, camera move, transition, note and guide enable.",
                }),
                "duration_seconds": ("FLOAT", {"default": 8.0, "min": 0.01, "max": 36000.0, "step": 0.01}),
                "frame_rate": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1}),
                "guide_policy": (["safe_core_guides", "prompt_only", "every_checked_row"], {"default": "safe_core_guides"}),
                "min_guide_gap_seconds": ("FLOAT", {"default": 1.75, "min": 0.0, "max": 60.0, "step": 0.05}),
                "max_guides": ("INT", {"default": 5, "min": 0, "max": 50, "step": 1}),
                "default_force": ("FLOAT", {"default": 0.22, "min": 0.0, "max": 1.0, "step": 0.01}),
                "promptrelay_epsilon": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 10.0, "step": 0.01}),
                "ltx_round_mode": (["up_8n_plus_1", "nearest_8n_plus_1", "none"], {"default": "up_8n_plus_1"}),
                "image_paths": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Internal reference images. Edited by the Shotboard Pro UI.",
                }),
                "image_width": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 32}),
                "image_height": ("INT", {"default": 432, "min": 64, "max": 8192, "step": 32}),
            },
            "optional": {
                "multi_input": ("IMAGE",),
            },
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE,)
    RETURN_NAMES = ("cine_linx",)
    FUNCTION = "execute"
    CATEGORY = "IAMCCS/Cine/02 Single Generation"

    @staticmethod
    def _build_cine_linx(
        *,
        global_prompt: str,
        timeline_data: str,
        duration_seconds: float,
        frame_rate: int,
        guide_policy: str,
        min_guide_gap_seconds: float,
        max_guides: int,
        default_force: float,
        promptrelay_epsilon: float,
        ltx_round_mode: str,
        image_paths: str,
        image_width: int,
        image_height: int,
        rows: List[Dict[str, Any]],
        guide_rows: List[Dict[str, Any]],
        flf_timeline: str,
        local_prompts: str,
        segment_lengths: str,
        max_frames: int,
        latent_lengths_preview: List[int],
        multi_output: Any,
        image_1: Any,
    ) -> Dict[str, Any]:
        promptrelay_enabled = bool(str(local_prompts or "").strip())
        payload = {
            "backend_id": CINE1_BACKEND_ID,
            "second_stage_id": CINE1_SECOND_STAGE_ID,
            "second_stage_aliases": list(CINE1_SECOND_STAGE_ALIASES),
            "backend_mode": "cine_ltx23_shotboard_promptrelay_flf",
            "promptrelay_enabled": promptrelay_enabled,
            "global_prompt": str(global_prompt or ""),
            "timeline_data": str(timeline_data or ""),
            "flf_timeline": str(flf_timeline or ""),
            "local_prompts": str(local_prompts or ""),
            "segment_lengths": str(segment_lengths or ""),
            "max_frames": int(max_frames),
            "promptrelay_epsilon": float(promptrelay_epsilon),
            "duration_seconds": float(duration_seconds),
            "frame_rate": int(frame_rate),
            "guide_policy": str(guide_policy),
            "min_guide_gap_seconds": float(min_guide_gap_seconds),
            "max_guides": int(max_guides),
            "default_force": float(default_force),
            "ltx_round_mode": str(ltx_round_mode),
            "image_paths": str(image_paths or ""),
            "image_width": int(image_width),
            "image_height": int(image_height),
            "rows": rows,
            "guide_rows": guide_rows,
            "promptrelay_pixel_lengths": latent_lengths_preview,
            "promptrelay_latent_lengths_preview": latent_lengths_preview,
        }
        resources = {
            "cine_payload": payload,
            "cine_flf_timeline": flf_timeline,
            "cine_global_prompt": str(global_prompt or ""),
            "cine_local_prompts": local_prompts,
            "cine_segment_lengths": segment_lengths,
            "cine_promptrelay_enabled": promptrelay_enabled,
            "cine_max_frames": int(max_frames),
            "cine_promptrelay_epsilon": float(promptrelay_epsilon),
            "cine_duration_seconds": float(duration_seconds),
            "cine_frame_rate": int(frame_rate),
            "cine_image_paths": str(image_paths or ""),
            "cine_image_width": int(image_width),
            "cine_image_height": int(image_height),
            "cine_multi_input": multi_output,
            "cine_image_1": image_1,
        }
        return {
            "type": SUPERNODE_LINX_TYPE,
            "pipeline_kind": "i2v_flf",
            "backend_id": CINE1_BACKEND_ID,
            "second_stage_id": CINE1_SECOND_STAGE_ID,
            "second_stage_aliases": list(CINE1_SECOND_STAGE_ALIASES),
            "mode": "cine_ltx23_shotboard_promptrelay_flf",
            "chain": [
                {
                    "role": "planner",
                    "name": "Cine Shotboard Planner Pro",
                    "backend_id": CINE1_BACKEND_ID,
                    "second_stage_id": CINE1_SECOND_STAGE_ID,
                }
            ],
            "stages": [
                {
                    "name": CINE1_BACKEND_ID,
                    "kind": "cine_shotboard_planner",
                    "payload": payload,
                }
            ],
            "policies": {
                "cine_backend": CINE1_BACKEND_ID,
                "cine_second_stage": CINE1_SECOND_STAGE_ID,
                "cine_second_stage_aliases": list(CINE1_SECOND_STAGE_ALIASES),
                "promptrelay_source": "cine_shotboard_planner",
                "flf_source": "cine_shotboard_planner",
            },
            "outputs": {
                "flf_timeline": flf_timeline,
                "global_prompt": str(global_prompt or ""),
                "local_prompts": local_prompts,
                "segment_lengths": segment_lengths,
                "max_frames": int(max_frames),
                "promptrelay_epsilon": float(promptrelay_epsilon),
                "duration_seconds": float(duration_seconds),
                "frame_rate": int(frame_rate),
                "promptrelay_enabled": promptrelay_enabled,
            },
            "resources": resources,
            "resource_keys": sorted(resources.keys()),
            "resource_types": {key: type(value).__name__ for key, value in resources.items()},
        }

    @classmethod
    def IS_CHANGED(
        cls,
        global_prompt=None,
        timeline_data=None,
        duration_seconds=None,
        frame_rate=None,
        guide_policy=None,
        min_guide_gap_seconds=None,
        max_guides=None,
        default_force=None,
        promptrelay_epsilon=None,
        ltx_round_mode=None,
        image_paths=None,
        image_width=None,
        image_height=None,
        multi_input=None,
        **kwargs,
    ):
        return _cine_change_fingerprint({
            "node": cls.__name__,
            "global_prompt": str(global_prompt or ""),
            "timeline_data": str(timeline_data or ""),
            "duration_seconds": float(duration_seconds or 0),
            "frame_rate": int(frame_rate or 0),
            "guide_policy": str(guide_policy or ""),
            "min_guide_gap_seconds": float(min_guide_gap_seconds or 0),
            "max_guides": int(max_guides or 0),
            "default_force": float(default_force or 0),
            "promptrelay_epsilon": None if promptrelay_epsilon is None else float(promptrelay_epsilon or 0),
            "ltx_round_mode": str(ltx_round_mode or ""),
            "image_paths": str(image_paths or ""),
            "image_signature": _cine_image_path_signature(image_paths),
            "image_width": int(image_width or 0),
            "image_height": int(image_height or 0),
        })

    def execute(self, global_prompt, timeline_data, duration_seconds, frame_rate, guide_policy, min_guide_gap_seconds, max_guides, default_force, promptrelay_epsilon, ltx_round_mode, image_paths, image_width, image_height, multi_input=None):
        fps = max(1, int(frame_rate))
        if str(image_paths or "").strip():
            try:
                loaded = IAMCCS_CineReferenceBoard().load_images(
                    image_paths,
                    int(image_width),
                    int(image_height),
                    "lanczos",
                    "crop",
                    32,
                    0,
                )
                multi_input = loaded[0]
            except Exception as exc:
                print(f"IAMCCS Cine Shotboard Planner Pro warning: could not load internal image_paths: {exc}")

        if multi_input is not None and torch.is_tensor(multi_input):
            multi_output = multi_input
        else:
            multi_output = torch.zeros((1, 64, 64, 3))
        image_1 = multi_output[0:1] if torch.is_tensor(multi_output) and multi_output.shape[0] > 0 else torch.zeros((1, 64, 64, 3))

        rows = self._parse_rows(timeline_data, float(duration_seconds), float(default_force))
        guide_rows = self._select_guides(rows, str(guide_policy), float(min_guide_gap_seconds), int(max_guides))
        reference_count = int(multi_output.shape[0]) if torch.is_tensor(multi_output) else 0
        flf_timeline = self._flf_from_rows(guide_rows, float(duration_seconds), fps, reference_count)
        local_prompts, segment_lengths, max_frames, latent_lengths_preview = self._segments_from_rows(rows, float(duration_seconds), fps, str(ltx_round_mode))
        self._log_promptrelay_state(
            "ShotboardPlannerPro",
            rows,
            local_prompts,
            segment_lengths,
            float(duration_seconds),
            str(global_prompt or ""),
            guide_rows,
            int(max_frames),
        )
        cine_linx = self._build_cine_linx(
            global_prompt=str(global_prompt or ""),
            timeline_data=str(timeline_data or ""),
            duration_seconds=float(duration_seconds),
            frame_rate=fps,
            guide_policy=str(guide_policy),
            min_guide_gap_seconds=float(min_guide_gap_seconds),
            max_guides=int(max_guides),
            default_force=float(default_force),
            promptrelay_epsilon=float(promptrelay_epsilon),
            ltx_round_mode=str(ltx_round_mode),
            image_paths=str(image_paths or ""),
            image_width=int(image_width),
            image_height=int(image_height),
            rows=rows,
            guide_rows=guide_rows,
            flf_timeline=flf_timeline,
            local_prompts=local_prompts,
            segment_lengths=segment_lengths,
            max_frames=int(max_frames),
            latent_lengths_preview=latent_lengths_preview,
            multi_output=multi_output,
            image_1=image_1,
        )

        report = _json_report({
            "node": "IAMCCS_CineShotboardPlannerPro",
            "mode": "planner_only_no_cycle",
            "cine_linx": {
                "type": SUPERNODE_LINX_TYPE,
                "backend_id": CINE1_BACKEND_ID,
                "second_stage_id": CINE1_SECOND_STAGE_ID,
                "resource_keys": cine_linx.get("resource_keys", []),
            },
            "duration_seconds": float(duration_seconds),
            "frame_rate": fps,
            "guide_policy": str(guide_policy),
            "rows": rows,
            "guide_rows_for_flf_engine": guide_rows,
            "promptrelay_resolved_local_prompts": local_prompts,
            "promptrelay_enabled": bool(str(local_prompts or "").strip()),
            "promptrelay_segment_lengths": segment_lengths,
            "promptrelay_pixel_lengths": latent_lengths_preview,
            "promptrelay_latent_lengths_preview": latent_lengths_preview,
            "max_frames": int(max_frames),
            "promptrelay_epsilon": float(promptrelay_epsilon),
            "truth": "Planner-only node: connect cine_linx to IAMCCS_CineInfo for classic workflow breakouts. CINE_1 belongs in the SuperNodes Exec Render backend list, not in a separate backend node.",
        })
        cine_linx["outputs"]["report"] = report
        cine_linx["resources"]["cine_report"] = report
        cine_linx["resource_keys"] = sorted(cine_linx["resources"].keys())
        cine_linx["resource_types"] = {key: type(value).__name__ for key, value in cine_linx["resources"].items()}
        return (cine_linx,)


class IAMCCS_CineShotboardPlannerProV2(IAMCCS_CineShotboardPlannerPro):
    """Shotboard Planner Pro V2.

    Same execution contract as Planner Pro, with an upgraded frontend for
    non-destructive reference framing/crop variants.
    """

    CATEGORY = "IAMCCS/Cine/02 Single Generation VIP"


class IAMCCS_CineShotboardLite(IAMCCS_CineShotboardPlannerPro):
    """Public/simple FLF-only shotboard.

    This node intentionally omits PromptRelay controls. It still emits the
    standard cine_linx payload so it can be connected to CineInfo and the same
    FLF backend path used by the Pro nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "global_prompt": ("STRING", {
                    "default": "one continuous cinematic FLF shot, coherent motion through the image guides, no slideshow, no cross dissolve",
                    "multiline": True,
                }),
                "timeline_data": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Edited by the Cine Shotboard Lite UI. FLF-only rows contain time, image reference, force, label and note.",
                }),
                "duration_seconds": ("FLOAT", {"default": 8.0, "min": 0.01, "max": 36000.0, "step": 0.01}),
                "frame_rate": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1}),
                "guide_policy": (["every_checked_row", "safe_core_guides"], {"default": "every_checked_row"}),
                "min_guide_gap_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.05}),
                "max_guides": ("INT", {"default": 12, "min": 1, "max": 50, "step": 1}),
                "default_force": ("FLOAT", {"default": 0.22, "min": 0.0, "max": 1.0, "step": 0.01}),
                "image_paths": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Internal reference images for the Lite shotboard.",
                }),
                "image_width": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 32}),
                "image_height": ("INT", {"default": 432, "min": 64, "max": 8192, "step": 32}),
            },
            "optional": {
                "multi_input": ("IMAGE",),
            },
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE,)
    RETURN_NAMES = ("cine_linx",)
    FUNCTION = "execute"
    CATEGORY = "IAMCCS/Cine/01 Lite"

    @staticmethod
    def _sanitize_lite_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        clean: List[Dict[str, Any]] = []
        for index, row in enumerate(rows):
            item = dict(row)
            item["use_prompt"] = False
            item["relay_prompt"] = ""
            item["use_relay_modifiers"] = False
            item["camera_relay_mode"] = "off"
            item["transition_relay_mode"] = "off"
            item["relay_addon_position"] = "after"
            item["relay_modifier_text"] = ""
            if not str(item.get("camera", "")).strip():
                item["camera"] = "continuous dolly-in"
            if not str(item.get("transition", "")).strip():
                item["transition"] = "continuous_motion"
            if not str(item.get("label", "")).strip():
                item["label"] = f"lite_shot_{index + 1}"
            clean.append(item)
        return clean

    def execute(self, global_prompt, timeline_data, duration_seconds, frame_rate, guide_policy, min_guide_gap_seconds, max_guides, default_force, image_paths, image_width, image_height, multi_input=None):
        fps = max(1, int(frame_rate))
        if str(image_paths or "").strip():
            try:
                loaded = IAMCCS_CineReferenceBoard().load_images(
                    image_paths,
                    int(image_width),
                    int(image_height),
                    "lanczos",
                    "crop",
                    32,
                    0,
                )
                multi_input = loaded[0]
            except Exception as exc:
                print(f"IAMCCS Cine Shotboard Lite warning: could not load internal image_paths: {exc}")

        if multi_input is not None and torch.is_tensor(multi_input):
            multi_output = multi_input
        else:
            multi_output = torch.zeros((1, 64, 64, 3))
        image_1 = multi_output[0:1] if torch.is_tensor(multi_output) and multi_output.shape[0] > 0 else torch.zeros((1, 64, 64, 3))

        rows = self._sanitize_lite_rows(self._parse_rows(timeline_data, float(duration_seconds), float(default_force)))
        guide_rows = self._select_guides(rows, str(guide_policy), float(min_guide_gap_seconds), int(max_guides))
        reference_count = int(multi_output.shape[0]) if torch.is_tensor(multi_output) else 0
        flf_timeline = self._flf_from_rows(guide_rows, float(duration_seconds), fps, reference_count)
        _local_prompts, _segment_lengths, max_frames, _latent_lengths_preview = self._segments_from_rows(
            rows,
            float(duration_seconds),
            fps,
            "up_8n_plus_1",
        )
        local_prompts = ""
        segment_lengths = ""
        latent_lengths_preview: List[int] = []
        _cine_debug(
            "[IAMCCS CineDebug:ShotboardLite] "
            f"FLF-only guides={len(guide_rows)} refs={[int(r.get('ref', 1)) for r in guide_rows]} "
            f"seconds={[float(r.get('second', 0.0)) for r in guide_rows]} "
            f"forces={[float(r.get('force', 0.0)) for r in guide_rows]} "
            f"promptrelay=disabled"
        )
        cine_linx = self._build_cine_linx(
            global_prompt=str(global_prompt or ""),
            timeline_data=str(timeline_data or ""),
            duration_seconds=float(duration_seconds),
            frame_rate=fps,
            guide_policy=str(guide_policy),
            min_guide_gap_seconds=float(min_guide_gap_seconds),
            max_guides=int(max_guides),
            default_force=float(default_force),
            promptrelay_epsilon=0.65,
            ltx_round_mode="up_8n_plus_1",
            image_paths=str(image_paths or ""),
            image_width=int(image_width),
            image_height=int(image_height),
            rows=rows,
            guide_rows=guide_rows,
            flf_timeline=flf_timeline,
            local_prompts=local_prompts,
            segment_lengths=segment_lengths,
            max_frames=int(max_frames),
            latent_lengths_preview=latent_lengths_preview,
            multi_output=multi_output,
            image_1=image_1,
        )
        cine_linx["mode"] = "cine_ltx23_shotboard_lite_flf_only"
        cine_linx["chain"][0]["name"] = "Cine Shotboard Lite"
        cine_linx["stages"][0]["kind"] = "cine_shotboard_lite_flf_only"
        cine_linx["policies"]["promptrelay_source"] = "disabled_by_lite_node"
        cine_linx["outputs"]["promptrelay_enabled"] = False
        cine_linx["resources"]["cine_promptrelay_enabled"] = False
        cine_linx["resources"]["cine_local_prompts"] = ""
        cine_linx["resources"]["cine_segment_lengths"] = ""

        report = _json_report({
            "node": "IAMCCS_CineShotboardLite",
            "mode": "flf_only_lite",
            "duration_seconds": float(duration_seconds),
            "frame_rate": fps,
            "guide_policy": str(guide_policy),
            "rows": rows,
            "guide_rows_for_flf_engine": guide_rows,
            "promptrelay_enabled": False,
            "promptrelay_resolved_local_prompts": "",
            "promptrelay_segment_lengths": "",
            "max_frames": int(max_frames),
            "image_width": int(image_width),
            "image_height": int(image_height),
            "truth": "Lite is FLF-only. Notes stay private and are never emitted as PromptRelay local prompts.",
        })
        cine_linx["outputs"]["report"] = report
        cine_linx["resources"]["cine_report"] = report
        cine_linx["resource_keys"] = sorted(cine_linx["resources"].keys())
        cine_linx["resource_types"] = {key: type(value).__name__ for key, value in cine_linx["resources"].items()}
        return (cine_linx,)


class IAMCCS_CineShotboardPlannerProLegacy(IAMCCS_CineShotboardPlannerPro):
    """Compatibility clone of Shotboard Planner Pro with classic direct outputs.

    The primary Shotboard Planner Pro intentionally exposes only cine_linx.
    Keep this clone for older hand-wired WDC/PromptRelay workflows that still
    expect direct sockets from the planner.
    """

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "INT",
        "FLOAT",
        "STRING",
        "IMAGE",
        "IMAGE",
        "INT",
        "INT",
        "INT",
        "INT",
        SUPERNODE_LINX_TYPE,
    )
    RETURN_NAMES = (
        "flf_timeline",
        "global_prompt",
        "local_prompts",
        "segment_lengths",
        "max_frames",
        "promptrelay_epsilon",
        "report",
        "multi_output",
        "image_1",
        "duration_seconds_int",
        "frame_rate_int",
        "width",
        "height",
        "cine_linx",
    )
    FUNCTION = "execute"
    CATEGORY = "IAMCCS/Cine/02 Single Generation"

    def execute(self, global_prompt, timeline_data, duration_seconds, frame_rate, guide_policy, min_guide_gap_seconds, max_guides, default_force, promptrelay_epsilon, ltx_round_mode, image_paths, image_width, image_height, multi_input=None):
        (cine_linx,) = super().execute(
            global_prompt,
            timeline_data,
            duration_seconds,
            frame_rate,
            guide_policy,
            min_guide_gap_seconds,
            max_guides,
            default_force,
            promptrelay_epsilon,
            ltx_round_mode,
            image_paths,
            image_width,
            image_height,
            multi_input=multi_input,
        )
        outputs = cine_linx.get("outputs", {}) if isinstance(cine_linx, dict) else {}
        resources = cine_linx.get("resources", {}) if isinstance(cine_linx, dict) else {}

        flf_timeline = outputs.get("flf_timeline", resources.get("cine_flf_timeline", ""))
        resolved_global_prompt = outputs.get("global_prompt", resources.get("cine_global_prompt", str(global_prompt or "")))
        local_prompts = outputs.get("local_prompts", resources.get("cine_local_prompts", ""))
        segment_lengths = outputs.get("segment_lengths", resources.get("cine_segment_lengths", ""))
        max_frames = int(outputs.get("max_frames", resources.get("cine_max_frames", 0)) or 0)
        epsilon = float(outputs.get("promptrelay_epsilon", resources.get("cine_promptrelay_epsilon", promptrelay_epsilon)) or 0.0)
        report = outputs.get("report", resources.get("cine_report", ""))
        multi_output = resources.get("cine_multi_input", torch.zeros((1, 64, 64, 3)))
        image_1 = resources.get("cine_image_1", multi_output[0:1] if torch.is_tensor(multi_output) and multi_output.shape[0] > 0 else torch.zeros((1, 64, 64, 3)))
        duration_int = int(round(float(outputs.get("duration_seconds", resources.get("cine_duration_seconds", duration_seconds)) or 0)))
        fps_int = int(outputs.get("frame_rate", resources.get("cine_frame_rate", frame_rate)) or 0)
        width_int = int(resources.get("cine_image_width", image_width) or image_width)
        height_int = int(resources.get("cine_image_height", image_height) or image_height)

        return (
            str(flf_timeline or ""),
            str(resolved_global_prompt or ""),
            str(local_prompts or ""),
            str(segment_lengths or ""),
            max_frames,
            epsilon,
            str(report or ""),
            multi_output,
            image_1,
            duration_int,
            fps_int,
            width_int,
            height_int,
            cine_linx,
        )


class IAMCCS_CineInfo:
    """Small breakout node: one cine_linx input, classic workflow outputs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT", "FLOAT", "STRING", "IMAGE", "IMAGE", "INT", "INT", SUPERNODE_LINX_TYPE, "IMAGE", "INT", "INT")
    RETURN_NAMES = ("flf_timeline", "global_prompt", "local_prompts", "segment_lengths", "max_frames", "promptrelay_epsilon", "report", "multi_output", "image_1", "duration_seconds_int", "frame_rate_int", "cine_linx", "first_stage_preview", "width", "height")
    FUNCTION = "extract"
    CATEGORY = "IAMCCS/Cine/00 Utilities"

    @staticmethod
    def _empty_image():
        return torch.zeros((1, 64, 64, 3))

    @staticmethod
    def _resources(cine_linx: Any) -> Dict[str, Any]:
        if not isinstance(cine_linx, dict):
            return {}
        resources = cine_linx.get("resources")
        return resources if isinstance(resources, dict) else {}

    @staticmethod
    def _outputs(cine_linx: Any) -> Dict[str, Any]:
        if not isinstance(cine_linx, dict):
            return {}
        outputs = cine_linx.get("outputs")
        return outputs if isinstance(outputs, dict) else {}

    @staticmethod
    def _decode_preview_from_resources(resources: Dict[str, Any], max_frames: int = 0) -> Optional[torch.Tensor]:
        latent = resources.get("first_stage_video_latent")
        if latent is None:
            latent = resources.get("video_latent")
        if latent is None:
            return None
        for vae_key in ("taeltx_vae", "vae"):
            vae = resources.get(vae_key)
            if vae is None:
                continue
            try:
                images = comfy_nodes.VAEDecode().decode(vae, latent)[0]
                if torch.is_tensor(images) and images.ndim == 4 and images.shape[0] > 0:
                    if max_frames and images.shape[0] > max_frames:
                        images = images[:max_frames]
                    return images
            except Exception:
                continue
        return None

    def extract(self, cine_linx):
        resources = self._resources(cine_linx)
        outputs = self._outputs(cine_linx)
        payload = resources.get("cine_payload")
        if not isinstance(payload, dict):
            payload = {}

        flf_timeline = str(resources.get("cine_flf_timeline", outputs.get("flf_timeline", payload.get("flf_timeline", ""))) or "")
        global_prompt = str(resources.get("cine_global_prompt", outputs.get("global_prompt", payload.get("global_prompt", ""))) or "")
        local_prompts = str(resources.get("cine_local_prompts", outputs.get("local_prompts", payload.get("local_prompts", ""))) or "")
        segment_lengths = str(resources.get("cine_segment_lengths", outputs.get("segment_lengths", payload.get("segment_lengths", ""))) or "")
        max_frames = _safe_int(resources.get("cine_max_frames", outputs.get("max_frames", payload.get("max_frames", 0))), 0)
        promptrelay_epsilon = _safe_float(resources.get("cine_promptrelay_epsilon", outputs.get("promptrelay_epsilon", payload.get("promptrelay_epsilon", 0.65))), 0.65)
        duration_seconds = _safe_float(resources.get("cine_duration_seconds", outputs.get("duration_seconds", payload.get("duration_seconds", 0.0))), 0.0)
        frame_rate = _safe_int(resources.get("cine_frame_rate", outputs.get("frame_rate", payload.get("frame_rate", 24))), 24)
        width = _safe_int(resources.get("cine_image_width", outputs.get("width", payload.get("image_width", 768))), 768)
        height = _safe_int(resources.get("cine_image_height", outputs.get("height", payload.get("image_height", 432))), 432)

        multi_output = resources.get("cine_multi_input")
        if not torch.is_tensor(multi_output):
            multi_output = self._empty_image()
        image_1 = resources.get("cine_image_1")
        if not torch.is_tensor(image_1):
            image_1 = multi_output[0:1] if torch.is_tensor(multi_output) and multi_output.shape[0] > 0 else self._empty_image()
        first_stage_preview = resources.get("first_stage_preview_images")
        if not torch.is_tensor(first_stage_preview):
            first_stage_preview = resources.get("taeltx_first_stage_preview_images")
        if not torch.is_tensor(first_stage_preview):
            first_stage_preview = self._decode_preview_from_resources(resources, int(max_frames))
        if not torch.is_tensor(first_stage_preview):
            first_stage_preview = self._empty_image()

        report = resources.get("cine_report") or outputs.get("report")
        if not report:
            report = _json_report({
                "node": "IAMCCS_CineInfo",
                "mode": "cine_linx_breakout",
                "backend_id": payload.get("backend_id", cine_linx.get("backend_id") if isinstance(cine_linx, dict) else CINE1_BACKEND_ID),
                "second_stage_id": payload.get("second_stage_id", cine_linx.get("second_stage_id") if isinstance(cine_linx, dict) else CINE1_SECOND_STAGE_ID),
                "duration_seconds": duration_seconds,
                "frame_rate": frame_rate,
                "width": width,
                "height": height,
                "max_frames": max_frames,
                "truth": "Breakout node only: it exposes data produced upstream by Cine Shotboard Planner Pro.",
            })

        local_parts = [part.strip() for part in str(local_prompts or "").split("|") if part.strip()]
        length_parts = [part for part in re.split(r"[,;\s]+", str(segment_lengths or "")) if part.strip()]
        global_hash = hashlib.sha1(str(global_prompt or "").encode("utf-8", errors="ignore")).hexdigest()[:12]
        local_hash = hashlib.sha1(str(local_prompts or "").encode("utf-8", errors="ignore")).hexdigest()[:12]
        flf_hash = hashlib.sha1(str(flf_timeline or "").encode("utf-8", errors="ignore")).hexdigest()[:12]
        image_shape = [int(v) for v in multi_output.shape] if torch.is_tensor(multi_output) else []
        _cine_debug(
            "[IAMCCS CineDebug:CineInfo] "
            f"forwarding flf_hash={flf_hash} flf_chars={len(str(flf_timeline or ''))} "
            f"global_hash={global_hash} local_hash={local_hash} "
            f"local_prompts={len(local_parts)} segment_lengths={len(length_parts)} "
            f"segment_lengths_value={segment_lengths or '<empty>'} max_frames={int(max_frames)} "
            f"duration={duration_seconds:.3f}s fps={int(frame_rate)} size={int(width)}x{int(height)} "
            f"multi_output_shape={image_shape}"
        )
        for idx, prompt in enumerate(local_parts[:20]):
            length = length_parts[idx] if idx < len(length_parts) else "<missing>"
            compact = prompt.replace("\n", "\\n")
            if len(compact) > 220:
                compact = compact[:217] + "..."
            _cine_debug(f"[IAMCCS CineDebug:CineInfo] local[{idx:02d}] length={length} text={compact!r}")
        if len(local_parts) > 20:
            _cine_debug(f"[IAMCCS CineDebug:CineInfo] local log truncated: {len(local_parts) - 20} more prompts.")

        return (
            flf_timeline,
            global_prompt,
            local_prompts,
            segment_lengths,
            int(max_frames),
            float(promptrelay_epsilon),
            str(report),
            multi_output,
            image_1,
            int(round(duration_seconds)),
            int(frame_rate),
            cine_linx,
            first_stage_preview,
            int(width),
            int(height),
        )


class IAMCCS_CineSwitch:
    """Lazy switch between the stable FLF basic prompt path and original PromptRelay.

    Keep PromptRelay out of the active execution path when ShotPlanner has no
    Relay rows. This avoids PromptRelay's required-local-prompt error without
    changing the proven FLF guide pipeline.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "base_model": ("MODEL", {"lazy": True}),
                "base_positive": ("CONDITIONING", {"lazy": True}),
                "relay_model": ("MODEL", {"lazy": True}),
                "relay_positive": ("CONDITIONING", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("model", "positive", "promptrelay_enabled", "prompt_mode", "report")
    FUNCTION = "switch"
    CATEGORY = "IAMCCS/Cine/00 Utilities"

    @staticmethod
    def _resources(cine_linx: Any) -> Dict[str, Any]:
        if not isinstance(cine_linx, dict):
            return {}
        resources = cine_linx.get("resources")
        return resources if isinstance(resources, dict) else {}

    @staticmethod
    def _outputs(cine_linx: Any) -> Dict[str, Any]:
        if not isinstance(cine_linx, dict):
            return {}
        outputs = cine_linx.get("outputs")
        return outputs if isinstance(outputs, dict) else {}

    @staticmethod
    def _payload(cine_linx: Any, resources: Dict[str, Any]) -> Dict[str, Any]:
        payload = resources.get("cine_payload") if isinstance(resources, dict) else None
        if isinstance(payload, dict):
            return payload
        stages = cine_linx.get("stages") if isinstance(cine_linx, dict) else []
        if isinstance(stages, list):
            for stage in stages:
                if isinstance(stage, dict) and isinstance(stage.get("payload"), dict):
                    return stage.get("payload")
        return {}

    @classmethod
    def _state(cls, cine_linx: Any) -> Tuple[bool, str, str, Dict[str, Any]]:
        resources = cls._resources(cine_linx)
        outputs = cls._outputs(cine_linx)
        payload = cls._payload(cine_linx, resources)
        local_prompts = str(resources.get("cine_local_prompts", outputs.get("local_prompts", payload.get("local_prompts", ""))) or "")
        segment_lengths = str(resources.get("cine_segment_lengths", outputs.get("segment_lengths", payload.get("segment_lengths", ""))) or "")
        diagnostics = resources.get("cine_promptrelay_diagnostics", outputs.get("promptrelay_diagnostics", {}))
        rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
        relay_rows = [
            row for row in rows
            if isinstance(row, dict) and str(row.get("use_prompt", False)).strip().lower() in {"1", "true", "yes", "on", "y"}
        ]
        enabled = bool(local_prompts.strip())
        prompt_mode = "PROMPT_RELAY" if enabled else "BASIC_TEXT"
        details = {
            "local_prompt_count": len([part.strip() for part in local_prompts.split("|") if part.strip()]),
            "segment_lengths": segment_lengths,
            "relay_rows": len(relay_rows),
            "diagnostics": diagnostics if isinstance(diagnostics, dict) else {},
            "truth": (
                "PromptRelay requires at least one local prompt. This lazy switch keeps the PromptRelay node unexecuted "
                "when ShotPlanner Relay rows are off, so FLF-only stays on the basic text path."
            ),
        }
        return enabled, prompt_mode, local_prompts, details

    def check_lazy_status(
        self,
        cine_linx,
        base_model=None,
        base_positive=None,
        relay_model=None,
        relay_positive=None,
    ):
        enabled, _, _, _ = self._state(cine_linx)
        needed = []
        if enabled:
            if relay_model is None:
                needed.append("relay_model")
            if relay_positive is None:
                needed.append("relay_positive")
        else:
            if base_model is None:
                needed.append("base_model")
            if base_positive is None:
                needed.append("base_positive")
        return needed

    def switch(
        self,
        cine_linx,
        base_model=None,
        base_positive=None,
        relay_model=None,
        relay_positive=None,
    ):
        enabled, prompt_mode, _, details = self._state(cine_linx)
        report = _json_report({
            "node": "IAMCCS_CineSwitch",
            "prompt_mode": prompt_mode,
            "promptrelay_enabled": enabled,
            "lazy_branch": "PromptRelay" if enabled else "Basic Text",
            **details,
        })
        _cine_debug(f"[IAMCCS CineSwitch] mode={prompt_mode} promptrelay_enabled={enabled}")
        if enabled:
            return relay_model, relay_positive, True, prompt_mode, report
        return base_model, base_positive, False, prompt_mode, report


class IAMCCS_CinePromptRelayLatentShapeSync:
    """Compute the PromptRelay latent shape from the chosen output resolution."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 768, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 448, "min": 1, "max": 8192, "step": 1}),
                "process_scale": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 4.0, "step": 0.01}),
                "round_mode": (["exact_scaled", "floor_to_multiple", "ceil_to_multiple", "nearest_to_multiple"], {"default": "exact_scaled"}),
                "multiple_of": ("INT", {"default": 1, "min": 1, "max": 256, "step": 1}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("relay_width", "relay_height", "process_scale", "report")
    FUNCTION = "sync"
    CATEGORY = "IAMCCS/Cine/00 Utilities"

    @staticmethod
    def _round_value(value: float, mode: str, multiple_of: int) -> int:
        multiple = max(1, int(multiple_of))
        if mode == "floor_to_multiple":
            return max(multiple, int(value // multiple) * multiple)
        if mode == "ceil_to_multiple":
            return max(multiple, int(np.ceil(value / multiple)) * multiple)
        if mode == "nearest_to_multiple":
            return max(multiple, int(round(value / multiple)) * multiple)
        return max(1, int(round(value)))

    def sync(self, width, height, process_scale, round_mode, multiple_of):
        src_w = max(1, int(width))
        src_h = max(1, int(height))
        scale = float(process_scale)
        relay_w = self._round_value(src_w * scale, str(round_mode), int(multiple_of))
        relay_h = self._round_value(src_h * scale, str(round_mode), int(multiple_of))
        report = _json_report({
            "node": "IAMCCS_CinePromptRelayLatentShapeSync",
            "source_width": src_w,
            "source_height": src_h,
            "process_scale": scale,
            "round_mode": str(round_mode),
            "multiple_of": int(multiple_of),
            "relay_width": int(relay_w),
            "relay_height": int(relay_h),
            "truth": "Connect relay_width/relay_height to the PromptRelay EmptyLTXVLatentVideo width/height so PromptRelay uses the same spatial scale as the sampler latent.",
        })
        _cine_debug(
            "[IAMCCS CinePromptRelayLatentShapeSync] "
            f"source={src_w}x{src_h} scale={scale:.4f} mode={round_mode} "
            f"multiple_of={int(multiple_of)} -> relay_shape={int(relay_w)}x{int(relay_h)}"
        )
        return int(relay_w), int(relay_h), scale, report


class IAMCCS_CineFLFLengthCompensator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline_data": ("STRING", {"default": "", "multiline": True}),
                "duration_seconds": ("INT", {"default": 8, "min": 1, "max": 36000, "step": 1}),
                "frame_rate": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1}),
                "crop_guide_pixel_frames": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
                "ltx_round_mode": (["up_8n_plus_1", "nearest_8n_plus_1", "none"], {"default": "up_8n_plus_1"}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("model_length_frames", "target_frames", "guide_count", "report")
    FUNCTION = "compute"
    CATEGORY = "IAMCCS/Cine/00 Utilities"

    @staticmethod
    def _guide_count_from_timeline(timeline_data: Any) -> int:
        try:
            data = json.loads(str(timeline_data or "{}"))
        except Exception:
            return 0
        if isinstance(data, dict):
            rows = data.get("keyframes") or data.get("rows") or data.get("segments") or []
        elif isinstance(data, list):
            rows = data
        else:
            rows = []
        count = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get("use_guide") is False:
                continue
            ref = _safe_int(row.get("ref", row.get("reference_index", row.get("image_ref", 0))), 0)
            strength = _safe_float(row.get("strength", row.get("force", 1.0)), 1.0)
            if ref > 0 and strength > 0:
                count += 1
        return count

    def compute(self, timeline_data, duration_seconds, frame_rate, crop_guide_pixel_frames, ltx_round_mode):
        fps = max(1, _safe_int(frame_rate, 24))
        duration = max(1, _safe_int(duration_seconds, 8))
        target_frames = _round_ltx_frames(int(round(float(duration) * fps)), str(ltx_round_mode))
        guide_count = self._guide_count_from_timeline(timeline_data)
        guide_pad = max(0, _safe_int(crop_guide_pixel_frames, 8))
        model_length_frames = int(target_frames) + int(guide_count) * int(guide_pad)
        if str(ltx_round_mode) != "none":
            model_length_frames = _round_ltx_frames(model_length_frames, str(ltx_round_mode))
        report = _json_report({
            "node": "IAMCCS_CineFLFLengthCompensator",
            "duration_seconds": int(duration),
            "frame_rate": int(fps),
            "target_frames_after_crop": int(target_frames),
            "guide_count": int(guide_count),
            "crop_guide_pixel_frames": int(guide_pad),
            "model_length_frames_before_crop": int(model_length_frames),
            "expected_final_seconds": float(target_frames) / float(fps),
            "why": "LTXVAddGuide appends one latent guide frame per FLF keyframe; LTXVCropGuides removes those guide frames later. This node pads the initial model length so the saved video keeps the requested duration.",
        })
        _cine_debug(
            "[IAMCCS CineFLFLengthCompensator] "
            f"target={target_frames}f guides={guide_count} pad={guide_pad}f "
            f"model_length={model_length_frames}f fps={fps}"
        )
        return int(model_length_frames), int(target_frames), int(guide_count), report


class IAMCCS_CinePromptRelaySafeEncode:
    """PromptRelay 1:1 when active, safe global CLIPTextEncode when inactive.

    This node exists to keep one workflow usable for both modes:
    - Relay ON: call ComfyUI-PromptRelay's original _encode_relay implementation.
    - Relay OFF: do not call PromptRelay, encode only the global prompt.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "latent": ("LATENT",),
                "global_prompt": ("STRING", {"default": "", "multiline": True}),
                "local_prompts": ("STRING", {"default": "", "multiline": True}),
                "segment_lengths": ("STRING", {"default": ""}),
                "epsilon": ("FLOAT", {"default": 1e-3, "min": 1e-6, "max": 0.99, "step": 1e-4}),
            },
            "optional": {
                "relay_options": ("RELAY_OPTIONS",),
                "promptrelay_enabled": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("model", "positive", "promptrelay_enabled", "report")
    FUNCTION = "encode"
    CATEGORY = "IAMCCS/Cine/00 Utilities"

    @staticmethod
    def _basic_encode(clip, text: str):
        result = comfy_nodes.CLIPTextEncode().encode(clip, str(text or ""))
        return result[0] if isinstance(result, tuple) else result

    @staticmethod
    def _split_local_prompts(local_prompts: str) -> List[str]:
        return [part.strip() for part in str(local_prompts or "").split("|") if part.strip()]

    @staticmethod
    def _short_hash(text: str) -> str:
        return hashlib.sha1(str(text or "").encode("utf-8", errors="ignore")).hexdigest()[:12]

    @staticmethod
    def _latent_shape(latent: Any) -> List[int]:
        try:
            samples = latent.get("samples") if isinstance(latent, dict) else None
            if torch.is_tensor(samples):
                return [int(v) for v in samples.shape]
        except Exception:
            pass
        return []

    def encode(
        self,
        model,
        clip,
        latent,
        global_prompt,
        local_prompts,
        segment_lengths,
        epsilon,
        relay_options=None,
        promptrelay_enabled=None,
    ):
        global_prompt = str(global_prompt or "")
        local_prompts = str(local_prompts or "")
        segment_lengths = str(segment_lengths or "")
        locals_list = self._split_local_prompts(local_prompts)
        length_parts = [part for part in re.split(r"[,;\s]+", str(segment_lengths or "")) if part.strip()]
        latent_shape = self._latent_shape(latent)
        local_hash = self._short_hash(local_prompts)
        global_hash = self._short_hash(global_prompt)

        auto_enabled = bool(locals_list)
        requested_enabled = auto_enabled if promptrelay_enabled is None else bool(promptrelay_enabled)
        relay_enabled = bool(requested_enabled and auto_enabled)

        if not relay_enabled:
            conditioning = self._basic_encode(clip, global_prompt)
            report = _json_report({
                "node": "IAMCCS_CinePromptRelaySafeEncode",
                "promptrelay_enabled": False,
                "mode": "BASIC_TEXT_GLOBAL_ONLY",
                "local_prompt_count": 0,
                "local_prompts_present": bool(locals_list),
                "promptrelay_requested": bool(requested_enabled),
                "global_hash": global_hash,
                "latent_shape": latent_shape,
                "model_passthrough_1to1": True,
                "basic_encoder_1to1": "ComfyUI CLIPTextEncode().encode",
                "truth": (
                    "PromptRelay is inactive. The original model is returned unchanged and the "
                    "positive conditioning is produced by ComfyUI's normal CLIPTextEncode using only the global prompt."
                ),
            })
            _cine_debug(
                "[IAMCCS CinePromptRelaySafeEncode] "
                f"mode=BASIC_TEXT_GLOBAL_ONLY promptrelay_enabled=False "
                f"model_passthrough_1to1=True basic_encoder=CLIPTextEncode "
                f"local_prompts_present={bool(locals_list)} promptrelay_requested={bool(requested_enabled)} "
                f"global_hash={global_hash} local_hash={local_hash} "
                f"segment_lengths={segment_lengths or '<empty>'} latent_shape={latent_shape} "
                f"global_preview={global_prompt.replace(chr(10), ' ')[:220]!r}"
            )
            return model, conditioning, False, report

        promptrelay_nodes = _load_original_promptrelay_module()
        patched, conditioning = promptrelay_nodes._encode_relay(
            model,
            clip,
            latent,
            global_prompt,
            local_prompts,
            segment_lengths,
            epsilon,
            relay_options,
        )
        report = _json_report({
            "node": "IAMCCS_CinePromptRelaySafeEncode",
            "promptrelay_enabled": True,
            "mode": "PROMPT_RELAY_ORIGINAL_1TO1",
            "local_prompt_count": len(locals_list),
            "global_hash": global_hash,
            "local_hash": local_hash,
            "segment_lengths": segment_lengths,
            "epsilon": float(epsilon),
            "latent_shape": latent_shape,
            "first_local_prompt": locals_list[0][:220] if locals_list else "",
            "last_local_prompt": locals_list[-1][:220] if locals_list else "",
            "original_promptrelay_module": getattr(promptrelay_nodes, "__file__", ""),
            "original_promptrelay_1to1": True,
            "basic_encoder_1to1": False,
            "truth": (
                "PromptRelay is active. IAMCCS did not reimplement the relay logic; "
                "it called ComfyUI-PromptRelay's original _encode_relay function."
            ),
        })
        _cine_debug(
            "[IAMCCS CinePromptRelaySafeEncode] "
            f"mode=PROMPT_RELAY_ORIGINAL_1TO1 local_prompts={len(locals_list)} "
            f"global_hash={global_hash} local_hash={local_hash} "
            f"segment_lengths_count={len(length_parts)} segment_lengths={segment_lengths or '<auto>'} "
            f"epsilon={float(epsilon):.6f} "
            f"latent_shape={latent_shape} "
            f"global_preview={global_prompt.replace(chr(10), ' ')[:180]!r} "
            f"first_local={locals_list[0][:160]!r}"
        )
        for idx, prompt in enumerate(locals_list[:20]):
            length = length_parts[idx] if idx < len(length_parts) else "<missing>"
            _cine_debug(
                "[IAMCCS CinePromptRelaySafeEncode] "
                f"local[{idx:02d}] length={length} text={prompt.replace(chr(10), ' ')[:260]!r}"
            )
        if len(locals_list) > 20:
            _cine_debug(f"[IAMCCS CinePromptRelaySafeEncode] local log truncated: {len(locals_list) - 20} more prompts.")
        return patched, conditioning, True, report


# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
class IAMCCS_CineRelayOrBypass:
    """Relay-or-bypass conditioning switch driven by cine_linx.

    By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com

    Decision rule (rigoroso):
        - Relay OFF if no ShotPlanner row has BOTH use_prompt=True AND a non-empty
          relay_prompt text.  In that case the node is a pure pass-through: the
          ``positive`` conditioning already encoded upstream is returned unchanged
          and the original model is returned unpatched.
        - Relay ON if at least one row satisfies the rule above.  The node then
          calls ComfyUI-PromptRelay's original ``_encode_relay`` 1:1, exactly as
          IAMCCS_CinePromptRelaySafeEncode does.

    Why this exists instead of IAMCCS_CinePromptRelaySafeEncode:
        - Reading all relay parameters from cine_linx avoids a fan-out of separate
          wires for global_prompt, local_prompts, segment_lengths and epsilon.
        - The explicit ``positive`` bypass input makes the OFF path a true
          zero-re-encode bypass, not a fresh CLIPTextEncode of the global prompt.
        - ``clip`` and ``latent`` are declared lazy so they are not executed when
          the relay is OFF and only the bypass path runs.

    By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    """

    @classmethod
    def INPUT_TYPES(cls):
        # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
            },
            "optional": {
                "global_prompt": ("STRING", {"forceInput": True}),
                "local_prompts": ("STRING", {"forceInput": True}),
                "segment_lengths": ("STRING", {"forceInput": True}),
                "epsilon": ("FLOAT", {"default": 1e-3, "min": 1e-6, "max": 0.99, "step": 1e-4, "forceInput": True}),
                "clip": ("CLIP", {"lazy": True}),
                "latent": ("LATENT", {"lazy": True}),
                "relay_options": ("RELAY_OPTIONS",),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("model", "positive", "promptrelay_enabled", "report")
    FUNCTION = "execute"
    CATEGORY = "IAMCCS/Cine/00 Utilities"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resources(cine_linx: Any) -> Dict[str, Any]:
        if not isinstance(cine_linx, dict):
            return {}
        r = cine_linx.get("resources")
        return r if isinstance(r, dict) else {}

    @staticmethod
    def _outputs(cine_linx: Any) -> Dict[str, Any]:
        if not isinstance(cine_linx, dict):
            return {}
        o = cine_linx.get("outputs")
        return o if isinstance(o, dict) else {}

    @classmethod
    def _payload(cls, cine_linx: Any, resources: Dict[str, Any]) -> Dict[str, Any]:
        p = resources.get("cine_payload")
        if isinstance(p, dict):
            return p
        stages = cine_linx.get("stages") if isinstance(cine_linx, dict) else []
        if isinstance(stages, list):
            for stage in stages:
                if isinstance(stage, dict) and isinstance(stage.get("payload"), dict):
                    return stage["payload"]
        return {}

    @classmethod
    def _relay_active(
        cls,
        cine_linx: Any,
        global_prompt: Optional[str] = None,
        local_prompts: Optional[str] = None,
        segment_lengths: Optional[str] = None,
        epsilon: Optional[float] = None,
    ) -> Tuple[bool, str, str, str, float, str]:
        """Return (active, local_prompts, segment_lengths, global_prompt, epsilon).

        By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com

        Relay is active (rigoroso) only when:
          - at least one row has use_prompt=True, AND
          - that same row has a non-empty relay_prompt string.
        This is equivalent to local_prompts being non-empty, because
        _segments_from_rows already skips rows that fail either condition.
        We read local_prompts directly from cine_linx to stay 1:1 with the
        planner's own computation.
        """
        resources = cls._resources(cine_linx)
        outputs = cls._outputs(cine_linx)
        payload = cls._payload(cine_linx, resources)

        source = "explicit_inputs" if local_prompts is not None else "cine_linx"
        resolved_local_prompts = str(local_prompts if local_prompts is not None else (
            resources.get("cine_local_prompts",
            outputs.get("local_prompts",
            payload.get("local_prompts", "")))
        ) or "")
        resolved_segment_lengths = str(segment_lengths if segment_lengths is not None else (
            resources.get("cine_segment_lengths",
            outputs.get("segment_lengths",
            payload.get("segment_lengths", "")))
        ) or "")
        resolved_global_prompt = str(global_prompt if global_prompt is not None else (
            resources.get("cine_global_prompt",
            outputs.get("global_prompt",
            payload.get("global_prompt", "")))
        ) or "")
        resolved_epsilon = _safe_float(
            epsilon if epsilon is not None else resources.get("cine_promptrelay_epsilon",
            outputs.get("promptrelay_epsilon",
            payload.get("promptrelay_epsilon", 1e-3))),
            1e-3,
        )
        active = bool(resolved_local_prompts.strip())
        return active, resolved_local_prompts, resolved_segment_lengths, resolved_global_prompt, resolved_epsilon, source

    # ------------------------------------------------------------------
    # Lazy evaluation: clip and latent are only needed when relay is ON
    # ------------------------------------------------------------------

    def check_lazy_status(
        self,
        cine_linx,
        model,
        positive,
        global_prompt=None,
        local_prompts=None,
        segment_lengths=None,
        epsilon=None,
        clip=None,
        latent=None,
        relay_options=None,
    ):
        active, _, _, _, _, _ = self._relay_active(cine_linx, global_prompt, local_prompts, segment_lengths, epsilon)
        needed = []
        if active:
            if clip is None:
                needed.append("clip")
            if latent is None:
                needed.append("latent")
        return needed

    # ------------------------------------------------------------------
    # Main execute
    # ------------------------------------------------------------------

    def execute(
        self,
        cine_linx,
        model,
        positive,
        global_prompt=None,
        local_prompts=None,
        segment_lengths=None,
        epsilon=None,
        clip=None,
        latent=None,
        relay_options=None,
    ):
        # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        active, local_prompts, segment_lengths, global_prompt, epsilon, prompt_source = self._relay_active(
            cine_linx,
            global_prompt,
            local_prompts,
            segment_lengths,
            epsilon,
        )

        locals_list = [part.strip() for part in str(local_prompts or "").split("|") if part.strip()]
        length_parts = [p for p in re.split(r"[,;\s]+", str(segment_lengths or "")) if p.strip()]
        global_hash = hashlib.sha1(str(global_prompt or "").encode("utf-8", errors="ignore")).hexdigest()[:12]
        local_hash = hashlib.sha1(str(local_prompts or "").encode("utf-8", errors="ignore")).hexdigest()[:12]

        if not active:
            # ── BYPASS PATH ─────────────────────────────────────────────────────
            report = _json_report({
                "node": "IAMCCS_CineRelayOrBypass",
                "promptrelay_enabled": False,
                "mode": "BYPASS_PASS_THROUGH",
                "local_prompt_count": 0,
                "global_hash": global_hash,
                "prompt_source": prompt_source,
                "truth": (
                    "No ShotPlanner row has both use_prompt=True and a non-empty relay_prompt. "
                    "The pre-encoded positive conditioning is returned unchanged. "
                    "The model is returned unpatched. "
                    "By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS"
                ),
            })
            _cine_debug(
                "[IAMCCS CineRelayOrBypass] "
                f"mode=BYPASS_PASS_THROUGH promptrelay_enabled=False "
                f"prompt_source={prompt_source} global_hash={global_hash} local_hash={local_hash}"
            )
            return model, positive, False, report

        # ── RELAY ACTIVE PATH ────────────────────────────────────────────────────
        if clip is None or latent is None:
            # Safety fallback: relay is ON but clip/latent were not supplied.
            # Return bypass rather than crash; the ComfyUI graph should have
            # provided both when relay is active.
            report = _json_report({
                "node": "IAMCCS_CineRelayOrBypass",
                "promptrelay_enabled": False,
                "mode": "BYPASS_MISSING_INPUTS",
                "warning": "Relay is active but clip or latent inputs are missing. Falling back to bypass.",
                "local_prompt_count": len(locals_list),
                "global_hash": global_hash,
                "local_hash": local_hash,
                "prompt_source": prompt_source,
            })
            print(
                "[IAMCCS CineRelayOrBypass] WARNING mode=BYPASS_MISSING_INPUTS "
                f"clip={'present' if clip is not None else 'MISSING'} "
                f"latent={'present' if latent is not None else 'MISSING'}"
            )
            return model, positive, False, report

        try:
            promptrelay_nodes = _load_original_promptrelay_module()
            patched, conditioning = promptrelay_nodes._encode_relay(
                model,
                clip,
                latent,
                global_prompt,
                local_prompts,
                segment_lengths,
                epsilon,
                relay_options,
            )
        except Exception as exc:
            report = _json_report({
                "node": "IAMCCS_CineRelayOrBypass",
                "promptrelay_enabled": False,
                "mode": "BYPASS_RELAY_ERROR",
                "error": str(exc),
                "local_prompt_count": len(locals_list),
                "global_hash": global_hash,
                "truth": "PromptRelay _encode_relay raised an error; falling back to bypass.",
            })
            print(f"[IAMCCS CineRelayOrBypass] ERROR _encode_relay failed: {exc}. Falling back to bypass.")
            return model, positive, False, report

        report = _json_report({
            "node": "IAMCCS_CineRelayOrBypass",
            "promptrelay_enabled": True,
            "mode": "PROMPT_RELAY_ORIGINAL_1TO1",
            "local_prompt_count": len(locals_list),
            "segment_count": len(length_parts),
            "global_hash": global_hash,
            "local_hash": local_hash,
            "epsilon": float(epsilon),
            "segment_lengths": segment_lengths,
            "prompt_source": prompt_source,
            "first_local": locals_list[0][:220] if locals_list else "",
            "last_local": locals_list[-1][:220] if locals_list else "",
            "truth": (
                "Relay active. Called ComfyUI-PromptRelay _encode_relay 1:1. "
                "By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS"
            ),
        })
        _cine_debug(
            "[IAMCCS CineRelayOrBypass] "
            f"mode=PROMPT_RELAY_ORIGINAL_1TO1 promptrelay_enabled=True "
            f"local_prompts={len(locals_list)} segments={len(length_parts)} "
            f"prompt_source={prompt_source} global_hash={global_hash} local_hash={local_hash} epsilon={float(epsilon):.6f} "
            f"global_preview={global_prompt.replace(chr(10), ' ')[:180]!r} "
            f"first_local={locals_list[0][:140]!r}"
        )
        for idx, prompt in enumerate(locals_list[:20]):
            length = length_parts[idx] if idx < len(length_parts) else "<missing>"
            _cine_debug(
                "[IAMCCS CineRelayOrBypass] "
                f"local[{idx:02d}] length={length} text={prompt.replace(chr(10), ' ')[:260]!r}"
            )
        if len(locals_list) > 20:
            _cine_debug(f"[IAMCCS CineRelayOrBypass] local log truncated: {len(locals_list) - 20} more prompts.")
        return patched, conditioning, True, report


class IAMCCS_CineMusicVideoPlanner:
    """Videoclip-maker planner inspired by audio/music sequencer workflows."""

    DEFAULT_IMAGE_BANK = "\n".join([
        "lead singer in a moody neon rehearsal room, cinematic portrait, expressive eyes",
        "wide shot of the band performing under colored stage lights, smoke haze, high energy",
        "close-up of hands on an electric guitar, strings vibrating, dramatic side light",
        "urban night exterior, performer walking through rain reflections, music video look",
        "abstract close-up of light beams and glass reflections moving with the beat",
        "final heroic performance frame, camera low angle, intense backlight",
    ])

    DEFAULT_ACTION_BANK = "\n".join([
        "the performer breathes with the beat, subtle head movement and eye contact, slow push-in",
        "the camera cuts to a wider energetic performance feeling, lights pulse with the rhythm",
        "hands strike instruments in sync with the beat, sharp close-up motion and quick details",
        "the performer walks forward through the environment, confident rhythm-driven movement",
        "abstract lights and reflections swell on the chorus, dynamic camera drift",
        "the shot builds to a final powerful performance moment, camera pushes forward",
    ])

    @staticmethod
    def _parse_sequence_json(sequence_json: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        text = str(sequence_json or "").strip()
        if not text:
            return [], {}
        try:
            data = json.loads(text)
        except Exception as exc:
            return [], {"parse_error": str(exc)}

        meta = data if isinstance(data, dict) else {}
        if isinstance(data, list):
            candidates = data
        elif isinstance(data, dict):
            candidates = []
            for key in ("shots", "sequence", "segments", "timeline", "items", "clips"):
                value = data.get(key)
                if isinstance(value, list):
                    candidates = value
                    break
        else:
            candidates = []

        shots = [item for item in candidates if isinstance(item, dict)]
        return shots, meta if isinstance(meta, dict) else {}

    @staticmethod
    def _first_number(data: Dict[str, Any], names: Tuple[str, ...]) -> Optional[float]:
        for name in names:
            if name in data:
                try:
                    return float(data.get(name))
                except Exception:
                    continue
        return None

    @staticmethod
    def _first_text(data: Dict[str, Any], names: Tuple[str, ...]) -> str:
        for name in names:
            value = str(data.get(name, "") or "").strip()
            if value:
                return value
        return ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "manual_shot_seconds": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 120.0, "step": 0.05}),
                "manual_total_shots": ("INT", {"default": 8, "min": 1, "max": 512, "step": 1}),
                "duration_mode": (["manual_shot_seconds", "audio_divided_by_total_shots", "max_manual_audio_slice"], {"default": "manual_shot_seconds"}),
                "frame_rounding": (["up_8n_plus_1", "nearest_8n_plus_1", "none"], {"default": "up_8n_plus_1"}),
                "beats_per_shot": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
                "image_model_hint": (["z_image_turbo", "flux_9b", "any_i2i_image_generator"], {"default": "z_image_turbo"}),
                "video_backend_hint": (["ltx23_i2v_audio", "ltx23_i2v", "wan22_i2v", "custom"], {"default": "ltx23_i2v_audio"}),
                "sequence_json": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Optional: connect/paste output from a music-video sequencer. If empty, manual timing is used.",
                }),
                "global_video_prompt": ("STRING", {
                    "default": "cinematic music video, rhythm-driven camera movement, coherent subject identity, no slideshow, no random flashes",
                    "multiline": True,
                }),
                "image_style_modifier": ("STRING", {
                    "default": "photorealistic, cinematic music video still, real-world materials, physically accurate lighting, subtle film grain, high detail, no text, no watermark",
                    "multiline": True,
                }),
                "image_prompt_bank": ("STRING", {"default": cls.DEFAULT_IMAGE_BANK, "multiline": True}),
                "video_action_bank": ("STRING", {"default": cls.DEFAULT_ACTION_BANK, "multiline": True}),
                "negative_prompt": ("STRING", {
                    "default": "text, watermark, logo, subtitles, low detail, plastic CGI, broken anatomy, bad hands, flicker, random objects, incoherent identity",
                    "multiline": True,
                }),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE, "STRING", "STRING", "STRING", "STRING", "STRING", "INT", "INT", "FLOAT", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("music_linx", "image_prompt", "video_global_prompt", "video_local_prompts", "segment_lengths", "negative_prompt", "frame_count_true", "frame_count_adjusted", "start_seconds", "duration_seconds", "total_shots", "report")
    FUNCTION = "plan"
    CATEGORY = "IAMCCS/Cine/05 Videoclip Maker"

    def plan(
        self,
        shot_index,
        fps,
        manual_shot_seconds,
        manual_total_shots,
        duration_mode,
        frame_rounding,
        beats_per_shot,
        image_model_hint,
        video_backend_hint,
        sequence_json,
        global_video_prompt,
        image_style_modifier,
        image_prompt_bank,
        video_action_bank,
        negative_prompt,
        audio=None,
    ):
        fps_value = max(1.0, float(fps))
        shot = max(0, int(shot_index))
        total_shots = max(1, int(manual_total_shots))
        sequence_shots, sequence_meta = self._parse_sequence_json(sequence_json)
        sequence_shot = sequence_shots[shot % len(sequence_shots)] if sequence_shots else None
        sequence_total = _safe_int(sequence_meta.get("total_shots", sequence_meta.get("totalShots", len(sequence_shots))), 0)
        if sequence_total > 0:
            total_shots = sequence_total
        elif sequence_shots:
            total_shots = len(sequence_shots)

        audio_seconds = _audio_duration_seconds(audio)
        manual_duration = max(0.1, float(manual_shot_seconds))
        sequence_duration = self._first_number(sequence_shot or {}, ("duration_seconds", "duration", "shot_seconds", "seconds"))
        sequence_start = self._first_number(sequence_shot or {}, ("start_seconds", "start_time", "start", "time"))
        sequence_frame_count = self._first_number(sequence_shot or {}, ("frame_count", "frames", "frame_count_true", "adjusted_frames"))
        sequence_start_frame = self._first_number(sequence_shot or {}, ("start_frame", "frame_start"))
        sequence_end_frame = self._first_number(sequence_shot or {}, ("end_frame", "frame_end"))

        if sequence_duration and sequence_duration > 0:
            shot_duration = float(sequence_duration)
        elif sequence_frame_count and sequence_frame_count > 0:
            shot_duration = max(0.1, float(sequence_frame_count) / fps_value)
        elif sequence_start_frame is not None and sequence_end_frame is not None and sequence_end_frame > sequence_start_frame:
            shot_duration = max(0.1, float(sequence_end_frame - sequence_start_frame) / fps_value)
        elif str(duration_mode) == "audio_divided_by_total_shots" and audio_seconds:
            shot_duration = max(0.1, float(audio_seconds) / float(total_shots))
        elif str(duration_mode) == "max_manual_audio_slice" and audio_seconds:
            shot_duration = max(manual_duration, float(audio_seconds) / float(total_shots))
        else:
            shot_duration = manual_duration

        if sequence_start is not None:
            start_seconds = float(sequence_start)
        elif sequence_start_frame is not None:
            start_seconds = max(0.0, float(sequence_start_frame) / fps_value)
        else:
            start_seconds = shot * shot_duration

        if sequence_frame_count and sequence_frame_count > 0:
            true_frames = max(1, int(round(sequence_frame_count)))
        elif sequence_start_frame is not None and sequence_end_frame is not None and sequence_end_frame > sequence_start_frame:
            true_frames = max(1, int(round(sequence_end_frame - sequence_start_frame)))
        else:
            true_frames = max(1, int(round(shot_duration * fps_value)))
        adjusted_frames = _round_ltx_frames(true_frames, str(frame_rounding))

        image_lines = _prompt_bank_lines(image_prompt_bank)
        action_lines = _prompt_bank_lines(video_action_bank)
        sequence_image_prompt = self._first_text(sequence_shot or {}, ("image_prompt", "still_prompt", "image", "visual_prompt"))
        sequence_action_prompt = self._first_text(sequence_shot or {}, ("video_prompt", "action_prompt", "local_prompt", "prompt", "action"))
        image_core = sequence_image_prompt or _pick_cycled(image_lines, shot, f"music video shot {shot + 1}, cinematic performer image")
        action_core = sequence_action_prompt or _pick_cycled(action_lines, shot, f"music video action beat {shot + 1}, rhythm-driven camera motion")
        image_prompt = ", ".join(part for part in [image_core, str(image_style_modifier or "").strip()] if part)

        sequence_beat_count = self._first_number(sequence_shot or {}, ("beat_count", "beats", "beats_per_shot"))
        beat_count = max(1, int(sequence_beat_count or beats_per_shot))
        beat_frames = []
        remaining = true_frames
        for beat in range(beat_count):
            if beat == beat_count - 1:
                length = max(1, remaining)
            else:
                length = max(1, int(round(true_frames / beat_count)))
                remaining -= length
            beat_frames.append(length)
        if sum(beat_frames) != true_frames and beat_frames:
            beat_frames[-1] = max(1, beat_frames[-1] + true_frames - sum(beat_frames))

        local_prompts = []
        for beat in range(beat_count):
            if beat == 0:
                local_prompts.append(action_core)
            else:
                local_prompts.append(f"{action_core}, beat variation {beat + 1}, keep same subject and visual style")
        video_local_prompts = " | ".join(local_prompts)
        segment_lengths = ",".join(str(int(item)) for item in beat_frames)

        payload = {
            "backend_id": "CINE_VIDEOCLIP_1",
            "second_stage_id": "2nd_stage_CINE_VIDEOCLIP_1",
            "backend_mode": "music_video_i2v_audio_sequencer",
            "shot_index": shot,
            "total_shots": total_shots,
            "start_seconds": float(start_seconds),
            "duration_seconds": float(shot_duration),
            "fps": float(fps_value),
            "frame_count_true": int(true_frames),
            "frame_count_adjusted": int(adjusted_frames),
            "beats_per_shot": beat_count,
            "image_model_hint": str(image_model_hint),
            "video_backend_hint": str(video_backend_hint),
            "sequence_source": "sequence_json" if sequence_shot else "manual",
            "sequence_shot": sequence_shot,
            "image_prompt": image_prompt,
            "video_global_prompt": str(global_video_prompt or ""),
            "video_local_prompts": video_local_prompts,
            "segment_lengths": segment_lengths,
            "negative_prompt": str(negative_prompt or ""),
            "audio_seconds": audio_seconds,
        }
        music_linx = {
            "type": SUPERNODE_LINX_TYPE,
            "pipeline_kind": "music_video",
            "backend_id": "CINE_VIDEOCLIP_1",
            "second_stage_id": "2nd_stage_CINE_VIDEOCLIP_1",
            "mode": "music_video_i2v_audio_sequencer",
            "chain": [{"role": "planner", "name": "Cine Music Video Planner", "shot_index": shot}],
            "stages": [{"name": "CINE_VIDEOCLIP_1", "kind": "music_video_planner", "payload": payload}],
            "outputs": {
                "image_prompt": image_prompt,
                "video_global_prompt": str(global_video_prompt or ""),
                "video_local_prompts": video_local_prompts,
                "segment_lengths": segment_lengths,
                "negative_prompt": str(negative_prompt or ""),
                "frame_count_true": int(true_frames),
                "frame_count_adjusted": int(adjusted_frames),
                "start_seconds": float(start_seconds),
                "duration_seconds": float(shot_duration),
                "total_shots": total_shots,
            },
            "resources": {
                "music_video_payload": payload,
                "audio": audio,
            },
        }
        report = _json_report({
            "node": "IAMCCS_CineMusicVideoPlanner",
            "mode": "videoclip_maker_planner",
            "shot_index": shot,
            "total_shots": total_shots,
            "start_seconds": float(start_seconds),
            "duration_seconds": float(shot_duration),
            "frame_count_true": int(true_frames),
            "frame_count_adjusted": int(adjusted_frames),
            "segment_lengths": segment_lengths,
            "sequence_source": "sequence_json" if sequence_shot else "manual",
            "image_prompt": image_prompt,
            "video_local_prompts": video_local_prompts,
            "truth": "This is a planner/breakout for music-video workflows. Use it before image generation and LTX I2V+Audio, or later connect music_linx to a dedicated videoclip backend.",
        })
        music_linx["outputs"]["report"] = report
        music_linx["resources"]["report"] = report
        music_linx["resource_keys"] = sorted(music_linx["resources"].keys())
        music_linx["resource_types"] = {key: type(value).__name__ for key, value in music_linx["resources"].items()}
        return (
            music_linx,
            image_prompt,
            str(global_video_prompt or ""),
            video_local_prompts,
            segment_lengths,
            str(negative_prompt or ""),
            int(true_frames),
            int(adjusted_frames),
            float(start_seconds),
            float(shot_duration),
            int(total_shots),
            report,
        )


class IAMCCS_CineShotPlanner(IAMCCS_LTX2_CinematicShotPlanner):
    CATEGORY = "IAMCCS/Cine/02 Single Generation"


class IAMCCS_CineRefLatentControl(IAMCCS_LTX2_CinematicRefLatentControl):
    CATEGORY = "IAMCCS/Cine/02 Single Generation"


class IAMCCS_CineAudioPromptDirector(IAMCCS_LTX2_AudioPromptDirector):
    CATEGORY = "IAMCCS/Cine/02 Single Generation"


class IAMCCS_CinePromptRelayAdapter(IAMCCS_LTX2_CinematicPromptRelayAdapter):
    CATEGORY = "IAMCCS/Cine/02 Single Generation"


class IAMCCS_CinePromptComposer(IAMCCS_LTX2_CinematicPromptComposer):
    CATEGORY = "IAMCCS/Cine/00 Utilities"


class IAMCCS_CineShotLineBuilder(IAMCCS_LTX2_CinematicShotLineBuilder):
    CATEGORY = "IAMCCS/Cine/03 Multi Generation"


class IAMCCS_CineV2VTimelineLineBuilder(IAMCCS_LTX2_CinematicV2VTimelineLineBuilder):
    CATEGORY = "IAMCCS/Cine/04 V2V Timeline"


class IAMCCS_CineLineStacker(IAMCCS_LTX2_CinematicLineStacker):
    CATEGORY = "IAMCCS/Cine/00 Utilities"


class IAMCCS_CineMultiGenDirector(IAMCCS_LTX2_CinematicMultiGenPlanner):
    CATEGORY = "IAMCCS/Cine/03 Multi Generation"


class IAMCCS_CineShotAudioDirector(IAMCCS_LTX2_CinematicShotAudioSelector):
    CATEGORY = "IAMCCS/Cine/03 Multi Generation"


class IAMCCS_CineV2VTimelineDirector(IAMCCS_LTX2_CinematicV2VTimelinePlanner):
    CATEGORY = "IAMCCS/Cine/04 V2V Timeline"


class IAMCCS_CineV2VAssetSelector(IAMCCS_LTX2_CinematicV2VAssetSelector):
    CATEGORY = "IAMCCS/Cine/04 V2V Timeline"


class IAMCCS_CineWorkflowInspector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_mode": (["single_generation_flf", "single_generation_promptrelay", "multigen_indexed", "v2v_timeline"], {"default": "single_generation_flf"}),
                "duration_seconds": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 36000.0, "step": 0.01}),
                "frame_rate": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1}),
                "reference_count": ("INT", {"default": 1, "min": 0, "max": 50, "step": 1}),
                "timeline_data": ("STRING", {"default": "", "multiline": True}),
                "promptrelay_segment_lengths": ("STRING", {"default": "", "multiline": False}),
                "plan_json": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("ok", "report")
    FUNCTION = "inspect"
    CATEGORY = "IAMCCS/Cine/00 Utilities"

    def inspect(self, workflow_mode, duration_seconds, frame_rate, reference_count, timeline_data, promptrelay_segment_lengths, plan_json):
        issues = []
        hints = []
        fps = max(1, int(frame_rate))
        total_frames = int(round(float(duration_seconds) * fps)) if float(duration_seconds) > 0 else 0
        if workflow_mode in {"single_generation_flf", "single_generation_promptrelay"} and int(reference_count) <= 0:
            issues.append("No references loaded for an image-guided workflow.")
        if workflow_mode == "single_generation_promptrelay" and not str(promptrelay_segment_lengths or "").strip():
            issues.append("PromptRelay segment_lengths is empty. Connect Cine PromptRelay Timeline output.")
        if str(timeline_data or "").strip():
            try:
                data = json.loads(str(timeline_data))
                rows = data.get("keyframes") or data.get("segments") or data.get("rows") if isinstance(data, dict) else data
                if isinstance(rows, list) and not rows:
                    issues.append("Timeline JSON is valid but empty.")
            except Exception:
                hints.append("Timeline is not JSON; it will be parsed as pipe lines if the target node supports it.")
        if str(promptrelay_segment_lengths or "").strip():
            lengths = [_safe_int(x, 0) for x in re.split(r"[,;\s]+", str(promptrelay_segment_lengths)) if x.strip()]
            if lengths and total_frames and abs(sum(lengths) - total_frames) > max(8, fps):
                hints.append(f"PromptRelay segment length sum is {sum(lengths)} frames, expected about {total_frames}.")
        if workflow_mode == "multigen_indexed":
            hints.append("For hard cuts, prefer one generation per indexed shot, then concatenate clips.")
        if workflow_mode == "v2v_timeline":
            hints.append("V2V is best when the source video already has useful motion or audio.")
        if str(plan_json or "").strip():
            try:
                json.loads(str(plan_json))
            except Exception:
                hints.append("plan_json is not valid JSON. Check the upstream director node.")
        ok = not issues
        report = _json_report({"ok": ok, "issues": issues, "hints": hints, "mode": workflow_mode})
        return ok, report


