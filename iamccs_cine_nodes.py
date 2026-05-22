from __future__ import annotations

import importlib.util
import base64
import datetime
import io
import hashlib
import json
import math
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


def _wdc_image_guide_strength(row: Dict[str, Any], fallback: float = 1.0) -> float:
    """Single LTX guide strength.

    Mirrors LTX Director semantics: one per-image guide strength controls the
    keyframe conditioning mask. IAMCCS no longer treats Image Lock as a second
    stronger override.
    """
    for key in (
        "guide_strength",
        "guideStrength",
        "strength",
        "force",
        "motion_force",
        "wdc_guide_strength",
        "wdcGuideStrength",
    ):
        if isinstance(row, dict) and row.get(key) is not None:
            return _clamp(row.get(key), 0.0, 1.0, fallback)
    return _clamp(fallback, 0.0, 1.0, 1.0)


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


def _iamccs_cine_resize_method(value: Any) -> str:
    method = str(value or "").strip()
    return method if method in {"crop", "pad", "keep proportion", "stretch"} else "crop"


def _safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    if text in {"0", "false", "no", "off", "n"}:
        return False
    return default


def _json_report(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _safe_json_loads(text: Any, default: Optional[Any] = None) -> Any:
    try:
        return json.loads(str(text or ""))
    except Exception:
        return {} if default is None else default


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


def _iamccs_ltx_compress_image(tensor: torch.Tensor, crf: int) -> torch.Tensor:
    """Mirror LTX Director's guide-image compression path for shotboard-owned images."""
    crf = int(crf or 0)
    if crf <= 0 or not torch.is_tensor(tensor) or tensor.ndim != 4 or tensor.shape[0] < 1:
        return tensor
    img = tensor[0]
    h = (int(img.shape[0]) // 2) * 2
    w = (int(img.shape[1]) // 2) * 2
    if h <= 0 or w <= 0:
        return tensor
    img_np = (img[:h, :w] * 255.0).clamp(0, 255).byte().cpu().numpy()
    try:
        import av

        buffer = io.BytesIO()
        container = av.open(buffer, mode="w", format="mp4")
        stream = container.add_stream("libx264", rate=1)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(max(0, min(51, crf))), "preset": "fast"}
        frame = av.VideoFrame.from_ndarray(img_np, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        buffer.seek(0)
        with av.open(buffer, mode="r") as decoded_container:
            for decoded_frame in decoded_container.decode(video=0):
                arr = decoded_frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
                return torch.from_numpy(arr).unsqueeze(0)
    except Exception as exc:
        print(f"IAMCCS Cine warning: LTX-style image compression failed, using uncompressed guide image: {exc}")
    return tensor


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

    def load_ltx_style_image(self, source, width, height, resize_method="crop", multiple_of=32, img_compression=0):
        image = None
        if isinstance(source, dict):
            image_file = str(source.get("imageFile", source.get("image_file", "")) or "").strip()
            image_b64 = str(source.get("imageB64", source.get("image_b64", "")) or "").strip()
            if image_file:
                full_path = image_file
                if not os.path.exists(full_path):
                    full_path = os.path.join(folder_paths.get_input_directory(), image_file)
                if os.path.exists(full_path):
                    image = Image.open(full_path)
            elif image_b64 and not image_b64.startswith("/view?"):
                if "," in image_b64:
                    image_b64 = image_b64.split(",", 1)[1]
                image = Image.open(io.BytesIO(base64.b64decode(image_b64)))
        else:
            path = str(source or "").strip()
            if path:
                full_path = path
                if not os.path.exists(full_path):
                    full_path = os.path.join(folder_paths.get_input_directory(), path)
                if os.path.exists(full_path):
                    image = Image.open(full_path)
        if image is None:
            raise FileNotFoundError(f"image source not found: {source}")
        image = ImageOps.exif_transpose(image).convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]
        method = str(resize_method or "crop")
        method = {
            "maintain aspect ratio": "keep proportion",
            "stretch to fit": "stretch",
        }.get(method, method)
        image_tensor = self.resize_image(
            image_tensor,
            int(width or 0),
            int(height or 0),
            method,
            "lanczos",
            int(multiple_of or 32),
        )
        return _iamccs_ltx_compress_image(image_tensor, int(img_compression or 0))

    def load_ltx_style_images(self, image_paths, width, height, resize_method="crop", multiple_of=32, img_compression=0):
        results = []
        for path in _cine_reference_paths_from_text(image_paths):
            try:
                results.append(self.load_ltx_style_image(path, width, height, resize_method, multiple_of, img_compression))
            except Exception as exc:
                print(f"IAMCCS Cine Shotboard warning: could not load internal guide image {path}: {exc}")
        if results:
            first_shape = results[0].shape
            batch_safe = []
            for item in results:
                if item.shape != first_shape:
                    item = self.resize_image(item, int(first_shape[2]), int(first_shape[1]), "crop", "lanczos", 32)
                batch_safe.append(torch.clamp(item, 0, 1))
            return torch.cat(batch_safe, dim=0)
        return torch.zeros((1, max(64, int(height or 64)), max(64, int(width or 64)), 3))


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
                        strength = _wdc_image_guide_strength(row, _safe_float(row.get("strength", 1.0), 1.0))
                        motion_force = _safe_float(row.get("motion_force", row.get("force", fallback_strength)), fallback_strength)
                        keyframes.append({
                            "second": second,
                            "reference_index": max(1, min(MAX_CINE_ITEMS, ref)),
                            "strength": strength,
                            "guide_strength": strength,
                            "image_lock_strength": strength,
                            "motion_force": motion_force,
                            "force": motion_force,
                            "label": str(row.get("label", f"key_{idx + 1}")),
                            "camera": str(row.get("camera", row.get("camera_note", ""))),
                            "step_transition_enabled": _safe_bool(row.get("step_transition_enabled"), False),
                            "step_transition_type": str(row.get("step_transition_type", "off") or "off"),
                            "step_transition_duration": _safe_float(row.get("step_transition_duration", 0.0), 0.0),
                            "step_transition_arrival": str(row.get("step_transition_arrival", "auto") or "auto"),
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
                        "guide_strength": strength,
                        "image_lock_strength": strength,
                        "motion_force": fallback_strength,
                        "force": fallback_strength,
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
                    "strength": 1.0,
                    "guide_strength": 1.0,
                    "image_lock_strength": 1.0,
                    "motion_force": float(fallback_strength),
                    "force": float(fallback_strength),
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
                "strength": _wdc_image_guide_strength(row, _safe_float(row.get("strength", 1.0), 1.0)),
                "guide_strength": _wdc_image_guide_strength(row, _safe_float(row.get("strength", 1.0), 1.0)),
                "image_lock_strength": _wdc_image_guide_strength(row, _safe_float(row.get("strength", 1.0), 1.0)),
                "motion_force": _safe_float(row.get("motion_force", row.get("force", fallback_strength)), fallback_strength),
                "force": _safe_float(row.get("motion_force", row.get("force", fallback_strength)), fallback_strength),
                "label": _normalise_label(str(row.get("label", "")), f"key_{idx + 1}"),
                "camera": str(row.get("camera", "")),
                "step_transition_enabled": _safe_bool(row.get("step_transition_enabled"), False),
                "step_transition_type": str(row.get("step_transition_type", "off") or "off"),
                "step_transition_duration": _safe_float(row.get("step_transition_duration", 0.0), 0.0),
                "step_transition_arrival": str(row.get("step_transition_arrival", "auto") or "auto"),
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
            "guide_attention_entries": False,
            "guide_strength_semantics": "clean_noise_mask_without_attention_attenuation",
            "compatibility_mode": "v3_clean_conditioning_guide",
            "applied_keyframes": applied,
            "skipped_keyframes": skipped,
            "truth": "PromptRelay controls temporal text; image guides use the same clean append_keyframe conditioning path as Filmmaker V3.",
        })
        return positive, negative, {"samples": latent_image, "noise_mask": noise_mask}, report


class IAMCCS_CineAllInOneFLFEngine(IAMCCS_CineLTXSequencer):
    """ShotPlanner-driven FLF engine using the same clean guide path as V3."""

    CATEGORY = "IAMCCS/Cine/02 Single Generation"

    @classmethod
    def INPUT_TYPES(cls):
        data = IAMCCS_CineLTXSequencer.INPUT_TYPES()
        data["required"]["duration_seconds"] = ("INT", {"default": 19, "min": 1, "max": 36000, "step": 1})
        return data

    @classmethod
    def execute(cls, positive, negative, vae, latent, multi_input, timeline_data, duration_seconds, frame_rate, fallback_num_images, fallback_strength, tail_safety_frames):
        positive, negative, latent_out, report = IAMCCS_CineLTXSequencer.execute(
            positive,
            negative,
            vae,
            latent,
            multi_input,
            timeline_data,
            duration_seconds,
            frame_rate,
            fallback_num_images,
            fallback_strength,
            tail_safety_frames,
        )
        try:
            data = json.loads(report)
            data["node"] = "IAMCCS_CineAllInOneFLFEngine"
            data["mode"] = "v3_clean_conditioning_guide_compat"
            report = _json_report(data)
        except Exception:
            pass
        return positive, negative, latent_out, report


class IAMCCS_CinePromptRelayTimeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "global_prompt": ("STRING", {
                    "default": "cinematic scene with natural motion and coherent camera movement",
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
                    "default": "one continuous cinematic shot with coherent motion and connected camera travel",
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
                image_lock_strength = _wdc_image_guide_strength(row, _safe_float(row.get("strength", 1.0), 1.0))
                note = str(row.get("note", row.get("camera_note", row.get("prompt", "")))).strip()
                camera = str(row.get("camera", row.get("camera_move", "cinematic motion"))).strip()
                transition = str(row.get("transition", row.get("transition_intent", "continuous_motion"))).strip() or "continuous_motion"
                label = _normalise_label(str(row.get("label", row.get("shot_label", ""))), f"shot_{idx + 1}")
                legacy_modifiers = cls._as_bool(row.get("use_relay_modifiers", row.get("use_camera_transition_in_relay", row.get("relay_modifiers", False))), False)
                camera_relay_mode = str(row.get("camera_relay_mode", row.get("camera_prompt_mode", "before" if legacy_modifiers else "off")) or "off").strip()
                transition_relay_mode = str(row.get("transition_relay_mode", row.get("transition_prompt_mode", "safe_only" if legacy_modifiers else "off")) or "off").strip()
                # Relay text must come from explicit local-prompt fields only.
                # Notes are private/technical shot notes and must not silently
                # become PromptRelay segments when the Relay toggles are enabled.
                relay_prompt = _canonical_relay_prompt(row)
                prompt_requested = cls._as_bool(
                    row.get("use_prompt", row.get("use_relay", row.get("relay", row.get("prompt_relay", True)))),
                    True,
                )
                step_transition_enabled = cls._as_bool(row.get("step_transition_enabled", row.get("stepTransitionEnabled", False)), False)
                option_relay_enabled = any(
                    cls._as_bool(row.get(key, row.get(alias, False)), False)
                    for key, alias in (
                        ("dialogue_pin", "dialoguePin"),
                        ("image_lock", "imageLock"),
                        ("motion_boost", "motionBoost"),
                    )
                )
                modifier_relay_enabled = camera_relay_mode != "off" or transition_relay_mode != "off"
                rows.append({
                    "second": max(0.0, second),
                    "frame": _safe_int(raw_frame, int(round(max(0.0, second) * 24))) if raw_frame is not None else None,
                    "ref": max(1, min(MAX_CINE_ITEMS, ref)),
                    "force": force,
                    "motion_force": force,
                    "image_lock_strength": image_lock_strength,
                    "guide_strength": image_lock_strength,
                    "label": label,
                    "camera": camera,
                    "transition": transition,
                    "note": note,
                    "use_guide": cls._as_bool(row.get("use_guide", row.get("guide", True)), True),
                    "use_prompt": bool((prompt_requested and relay_prompt) or step_transition_enabled or option_relay_enabled or modifier_relay_enabled),
                    "relay_prompt": relay_prompt,
                    "use_relay_modifiers": legacy_modifiers,
                    "camera_relay_mode": camera_relay_mode,
                    "transition_relay_mode": transition_relay_mode,
                    "relay_addon_position": str(row.get("relay_addon_position", row.get("addon_position", "after")) or "after").strip(),
                    "relay_modifier_text": str(row.get("relay_modifier_text", row.get("modifier_text", row.get("relay_addon", ""))) or "").strip(),
                    "dialogue_pin": cls._as_bool(row.get("dialogue_pin", row.get("dialoguePin", False)), False),
                    "image_lock": cls._as_bool(row.get("image_lock", row.get("imageLock", False)), False),
                    "motion_boost": cls._as_bool(row.get("motion_boost", row.get("motionBoost", False)), False),
                    "clean_relay": cls._as_bool(row.get("clean_relay", row.get("cleanRelay", False)), False),
                    "step_transition_enabled": step_transition_enabled,
                    "step_transition_type": str(row.get("step_transition_type", row.get("stepTransitionType", "off")) or "off").strip(),
                    "step_transition_prompt": str(row.get("step_transition_prompt", row.get("stepTransitionPrompt", "")) or "").strip(),
                    "step_transition_easing": str(row.get("step_transition_easing", row.get("stepTransitionEasing", "ease_in_out")) or "ease_in_out").strip(),
                    "step_transition_force_curve": str(row.get("step_transition_force_curve", row.get("stepTransitionForceCurve", "late_target")) or "late_target").strip(),
                    "step_transition_duration": max(0.0, _safe_float(row.get("step_transition_duration", row.get("stepTransitionDuration", row.get("step_seconds", 0.0))), 0.0)),
                    "step_transition_arrival": str(row.get("step_transition_arrival", row.get("stepTransitionArrival", "auto")) or "auto").strip(),
                    "step_transition_auto_fit": cls._as_bool(row.get("step_transition_auto_fit", row.get("stepTransitionAutoFit", True)), True),
                })

        if not rows and text:
            for idx, raw_line in enumerate(text.splitlines()):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                second, ref, force, label, camera, transition, note = _split_pipe(line, 7)
                force_value = _clamp(force, 0.0, 1.0, default_force)
                rows.append({
                    "second": max(0.0, _safe_float(second, idx * 2.0)),
                    "ref": max(1, min(MAX_CINE_ITEMS, _safe_int(ref, idx + 1))),
                    "force": force_value,
                    "motion_force": force_value,
                    "image_lock_strength": 1.0,
                    "guide_strength": 1.0,
                    "label": _normalise_label(label, f"shot_{idx + 1}"),
                    "camera": camera.strip() or "cinematic motion",
                    "transition": transition.strip() or "continuous_motion",
                    "note": note.strip(),
                    "use_guide": True,
                    "use_prompt": bool(note.strip()),
                    "relay_prompt": note.strip(),
                    "use_relay_modifiers": False,
                    "camera_relay_mode": "off",
                    "transition_relay_mode": "off",
                    "relay_addon_position": "after",
                    "relay_modifier_text": "",
                    "dialogue_pin": False,
                    "image_lock": False,
                    "motion_boost": False,
                    "clean_relay": False,
                    "step_transition_enabled": False,
                    "step_transition_type": "off",
                    "step_transition_prompt": "",
                    "step_transition_easing": "ease_in_out",
                    "step_transition_force_curve": "late_target",
                    "step_transition_duration": 0.0,
                    "step_transition_arrival": "auto",
                    "step_transition_auto_fit": True,
                })

        if not rows:
            rows = [
                {"second": 0.0, "ref": 1, "force": 0.78, "label": "opening_ref", "camera": "slow push-in", "transition": "continuous_motion", "note": "start from the first reference", "use_guide": True, "use_prompt": False, "relay_prompt": "", "use_relay_modifiers": False, "camera_relay_mode": "off", "transition_relay_mode": "off", "relay_addon_position": "after", "relay_modifier_text": ""},
                {"second": max(0.1, duration_seconds * 0.55), "ref": 2, "force": default_force, "label": "middle_ref", "camera": "continuous dolly-in", "transition": "continuous_motion", "note": "midpoint visual target", "use_guide": True, "use_prompt": False, "relay_prompt": "", "use_relay_modifiers": False, "camera_relay_mode": "off", "transition_relay_mode": "off", "relay_addon_position": "after", "relay_modifier_text": ""},
                {"second": max(0.2, duration_seconds - 0.4), "ref": 3, "force": default_force, "label": "ending_ref", "camera": "slow push-in", "transition": "continuous_motion", "note": "last visual target", "use_guide": True, "use_prompt": False, "relay_prompt": "", "use_relay_modifiers": False, "camera_relay_mode": "off", "transition_relay_mode": "off", "relay_addon_position": "after", "relay_modifier_text": ""},
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
                parts.append("hard cut staging with clean separated shot identity and stable subject form")
            return parts
        elif transition == "match_cut":
            parts.append("match movement continuity through shape and camera direction")
        elif transition == "soft_morph":
            parts.append("single continuous transformation with a feathered optical blend")
        else:
            parts.append("continuous physical camera movement with stable parallax and connected spatial motion")

        if next_row is not None and transition != "hard_cut":
            nxt = str(next_row.get("label", "next target")).replace("_", " ")
            parts.append(f"move toward {nxt} through one steady camera path")
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
    def _option_modifier_parts(cls, row: Dict[str, Any]) -> List[str]:
        parts: List[str] = []
        clean = cls._as_bool(row.get("clean_relay", False), False)
        if cls._as_bool(row.get("dialogue_pin", False), False):
            parts.append("preserve the exact spoken words and their timing as a pinned dialogue cue")
        if not clean and cls._as_bool(row.get("image_lock", False), False):
            parts.append("preserve the reference image identity, face, wardrobe, composition, lighting, and spatial layout")
        if not clean and cls._as_bool(row.get("motion_boost", False), False):
            parts.append("prioritize physical camera motion, parallax, and visible subject movement while keeping identity stable")
        return parts

    @classmethod
    def _step_transition_parts(cls, row: Dict[str, Any], next_row: Optional[Dict[str, Any]] = None) -> List[str]:
        if not cls._as_bool(row.get("step_transition_enabled", False), False):
            return []
        transition_type = cls._normalise_mode(
            row.get("step_transition_type", "slow_dolly_in"),
            {"off", "action_beat", "slow_dolly_in", "hold_then_push", "orbit_bridge", "match_move", "rack_focus", "soft_push"},
            "slow_dolly_in",
        )
        if transition_type == "off":
            return []
        custom = str(row.get("step_transition_prompt", "") or "").strip()
        next_label = str((next_row or {}).get("label", "next guide") or "next guide").replace("_", " ")
        library = {
            "action_beat": "the action continues through this segment toward the next guide, same subject and location, coherent physical motion",
            "slow_dolly_in": "same continuous shot, physical camera dolly forward from the current framing toward a closer framing of the same subject, continuous lens movement, stable identity, same location and lighting",
            "hold_then_push": "same continuous shot, hold the current framing briefly, then begin a physical camera push forward toward a closer framing of the same subject",
            "orbit_bridge": "same continuous shot, physical camera orbit around the same subject, curved parallax, stable identity and location",
            "match_move": "same continuous shot, match the existing movement direction, preserve scale continuity, stable identity and spatial relation",
            "rack_focus": "same continuous shot, gradual rack focus with stable framing, same subject and same location",
            "soft_push": "same continuous shot, gentle physical camera push forward, stable subject identity and coherent scale change",
        }
        parts = [custom or library.get(transition_type, library["slow_dolly_in"])]
        requested_duration = max(0.0, _safe_float(row.get("step_transition_duration", 0.0), 0.0))
        if requested_duration > 0:
            parts.append(f"use an intentional {requested_duration:.1f} second camera-move window")
        arrival_default = "very_late" if transition_type == "hold_then_push" else "late" if transition_type in {"slow_dolly_in", "orbit_bridge", "soft_push"} else "middle"
        arrival = cls._normalise_mode(row.get("step_transition_arrival", "auto"), {"auto", "early", "middle", "late", "very_late"}, "auto")
        if arrival == "auto":
            arrival = arrival_default
        arrival_text = {
            "early": "reach the closer target framing early while remaining one physical camera shot",
            "middle": "reach the midpoint of the camera move near the middle of the window",
            "late": "hold the source framing longer, then reach the closer target framing late by camera movement",
            "very_late": "preserve the source framing longest, reaching the closer target framing only in the final beats",
        }.get(arrival, "")
        if arrival_text:
            parts.append(arrival_text)
        easing = cls._normalise_mode(
            row.get("step_transition_easing", "ease_in_out"),
            {"linear", "ease_in", "ease_out", "ease_in_out"},
            "ease_in_out",
        )
        easing_text = {
            "linear": "keep the camera move evenly paced across the whole segment",
            "ease_in": "start gently and gradually accelerate the camera move",
            "ease_out": "move early then slow the camera gently into the final closer framing",
            "ease_in_out": "ease in and ease out, with a smooth cinematic middle movement",
        }.get(easing, "")
        if easing_text:
            parts.append(easing_text)
        force_curve = cls._normalise_mode(
            row.get("step_transition_force_curve", "late_target"),
            {"balanced", "late_target", "early_source", "free_motion"},
            "late_target",
        )
        curve_text = {
            "late_target": "use the target guide as the final framing reference near the end of the camera move",
            "early_source": "stay close to the current guide at first before changing scale",
            "free_motion": "allow more physical camera motion while preserving identity",
            "balanced": "balance the current framing and target framing through one camera move",
        }.get(force_curve, "")
        if curve_text:
            parts.append(curve_text)
        return parts

    @classmethod
    def _row_prompt(cls, row: Dict[str, Any], next_row: Optional[Dict[str, Any]] = None) -> str:
        relay_prompt = str(row.get("relay_prompt", "") or "").strip()
        before_parts: List[str] = []
        after_parts: List[str] = []
        base_parts: List[str] = [relay_prompt] if relay_prompt else []

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
        after_parts.extend(cls._option_modifier_parts(row))
        after_parts.extend(cls._step_transition_parts(row, next_row))

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
        all_timed_rows = [
            row for row in rows
            if float(row.get("second", 0.0)) < float(duration_seconds)
        ]
        all_timed_rows = sorted(all_timed_rows, key=lambda item: float(item.get("second", 0.0)))
        sorted_rows = []
        for row in eligible_rows:
            if float(row.get("second", 0.0)) >= float(duration_seconds):
                continue
            sorted_rows.append(row)
        if not sorted_rows:
            return "", "", int(max_frames), []
        for idx, row in enumerate(sorted_rows):
            start = max(0.0, float(row.get("second", 0.0)))
            step_enabled = cls._as_bool(row.get("step_transition_enabled", False), False)
            step_type = cls._normalise_mode(
                row.get("step_transition_type", "off"),
                {"off", "action_beat", "slow_dolly_in", "hold_then_push", "orbit_bridge", "match_move", "rack_focus", "soft_push"},
                "off",
            )
            next_row = sorted_rows[idx + 1] if idx + 1 < len(sorted_rows) else None
            if step_enabled and step_type != "off":
                try:
                    all_idx = all_timed_rows.index(row)
                except ValueError:
                    all_idx = -1
                if 0 <= all_idx < len(all_timed_rows) - 1:
                    next_row = all_timed_rows[all_idx + 1]
            if next_row is not None:
                end = max(start + 0.05, float(next_row.get("second", duration_seconds)))
            elif idx + 1 < len(sorted_rows):
                end = max(start + 0.05, float(sorted_rows[idx + 1].get("second", duration_seconds)))
            else:
                end = max(start + 0.05, float(duration_seconds))
            dur = max(0.05, end - start)
            prompt = cls._row_prompt(row, next_row)
            if not prompt:
                continue
            if step_enabled and step_type != "off" and next_row is not None and dur >= 0.4:
                next_label = str(next_row.get("label", "next guide") or "next guide").replace("_", " ")
                requested = max(0.0, _safe_float(row.get("step_transition_duration", 0.0), 0.0))
                arrival = cls._normalise_mode(row.get("step_transition_arrival", "auto"), {"auto", "early", "middle", "late", "very_late"}, "auto")
                if arrival == "auto":
                    arrival = "very_late" if step_type == "hold_then_push" else "late" if step_type in {"slow_dolly_in", "orbit_bridge", "soft_push"} else "middle"
                window = min(dur, requested if requested > 0 else dur)
                hold = max(0.0, dur - window)
                if requested <= 0 and arrival in {"late", "very_late"} and dur >= 1.0:
                    hold = dur * (0.25 if arrival == "late" else 0.42)
                    window = max(0.1, dur - hold)
                if hold >= 0.18:
                    segments.append({
                        "seconds": hold,
                        "prompt": "same continuous shot, maintain the current guide composition with stable identity and same location, prepare a physical camera move",
                    })
                begin = max(0.08, window * (0.35 if arrival in {"late", "very_late"} else 0.45))
                arrive = max(0.08, window - begin)
                if begin + arrive > window and window > 0:
                    scale = window / (begin + arrive)
                    begin = max(0.05, begin * scale)
                    arrive = max(0.05, arrive * scale)
                segments.append({
                    "seconds": begin,
                    "prompt": f"{prompt}, begin the physical camera move gradually from the current framing, same subject, same space",
                })
                segments.append({
                    "seconds": arrive,
                    "prompt": f"{prompt}, continue the same physical camera move, use {next_label} only as the final framing reference near the end",
                })
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
            and (str(row.get("relay_prompt", "") or "").strip() or cls._as_bool(row.get("step_transition_enabled", False), False))
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
        print(
            f"[IAMCCS {node_name}] PromptRelay requested={bool(str(local_prompts or '').strip())} "
            f"locals={local_count} segments={length_count} rows={len(rows)} guides={len(guide_rows)} "
            f"max_frames={int(max_frames or 0)}"
        )
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
            motion_force = float(row.get("force", row.get("motion_force", 0.0)))
            guide_strength = _wdc_image_guide_strength(row, 1.0)
            item = {
                "second": float(row.get("second", 0.0)),
                "ref": int(row.get("ref", 1)),
                "strength": float(guide_strength),
                "guide_strength": float(guide_strength),
                "image_lock_strength": float(guide_strength),
                "motion_force": float(motion_force),
                "force": float(motion_force),
                "label": str(row.get("label", "shot")),
                "camera": str(row.get("camera", "")),
                "note": str(row.get("note", "")),
                "transition": str(row.get("transition", "continuous_motion")),
                "step_transition_enabled": _safe_bool(row.get("step_transition_enabled"), False),
                "step_transition_type": str(row.get("step_transition_type", "off") or "off"),
                "step_transition_duration": _safe_float(row.get("step_transition_duration", 0.0), 0.0),
                "step_transition_arrival": str(row.get("step_transition_arrival", "auto") or "auto"),
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
                warnings.append(f"Row '{row['label']}' uses a strong FLFreal anchor; keep it intentional and avoid overcrowded guides.")
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
                    "default": "one continuous cinematic shot with coherent motion and connected camera travel",
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
                "image_resize_method": (["crop", "pad", "keep proportion", "stretch", ""], {"default": "crop"}),
                "image_multiple_of": ("INT", {"default": 32, "min": 1, "max": 512, "step": 1}),
                "img_compression": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
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
        image_resize_method: str,
        image_multiple_of: int,
        img_compression: int,
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
        image_resize_method = _iamccs_cine_resize_method(image_resize_method)
        promptrelay_enabled = bool(str(local_prompts or "").strip())
        sorted_visual_rows = sorted(
            [row for row in rows if isinstance(row, dict) and float(row.get("second", 0.0)) < float(duration_seconds)],
            key=lambda item: float(item.get("second", 0.0)),
        )
        reference_paths = _cine_reference_paths_from_text(image_paths)
        visual_segments = []
        for row_index, row in enumerate(sorted_visual_rows):
            start_frame = _safe_int(
                row.get("frame", int(round(float(row.get("second", 0.0)) * max(1, int(frame_rate))))),
                int(round(float(row.get("second", 0.0)) * max(1, int(frame_rate)))),
            )
            if row_index + 1 < len(sorted_visual_rows):
                next_second = float(sorted_visual_rows[row_index + 1].get("second", duration_seconds))
            else:
                next_second = float(duration_seconds)
            end_frame = max(start_frame + 1, int(round(next_second * max(1, int(frame_rate)))))
            ref_index = int(max(1, _safe_int(row.get("ref", row_index + 1), row_index + 1)))
            visual_segments.append({
                "id": f"row_{row_index + 1}",
                "type": "image",
                "start": int(max(0, start_frame)),
                "length": int(max(1, end_frame - start_frame)),
                "ref": ref_index,
                "imageFile": reference_paths[ref_index - 1] if ref_index <= len(reference_paths) else "",
                "label": str(row.get("label", f"ref_{row_index + 1}")),
                "prompt": str(row.get("relay_prompt", "")),
                "camera": str(row.get("camera", "")),
                "transition": str(row.get("transition", "continuous_motion")),
                "guideStrength": float(_safe_float(row.get("motion_force", row.get("force", default_force)), default_force)),
                "imageLockStrength": float(_safe_float(row.get("motion_force", row.get("force", default_force)), default_force)),
                "use_guide": bool(row.get("use_guide", True)),
                "use_prompt": bool(row.get("use_prompt", False)),
                "step_transition_enabled": bool(row.get("step_transition_enabled", False)),
                "step_transition_type": str(row.get("step_transition_type", "off") or "off"),
                "step_transition_prompt": str(row.get("step_transition_prompt", "") or ""),
                "step_transition_duration": float(_safe_float(row.get("step_transition_duration", 0.0), 0.0)),
                "step_transition_arrival": str(row.get("step_transition_arrival", "auto") or "auto"),
                "step_transition_auto_fit": bool(row.get("step_transition_auto_fit", True)),
            })
        visual_segments_json = json.dumps(visual_segments, ensure_ascii=False)
        payload = {
            "backend_id": CINE1_BACKEND_ID,
            "second_stage_id": CINE1_SECOND_STAGE_ID,
            "second_stage_aliases": list(CINE1_SECOND_STAGE_ALIASES),
            "backend_mode": "cine_ltx23_shotboard_promptrelay_flf",
            "promptrelay_enabled": promptrelay_enabled,
            "promptrelay_gate_mode": "toggle_and_non_empty_local_prompt",
            "promptrelay_backend": "IAMCCS_CineRelayOrBypass",
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
            "image_resize_method": image_resize_method,
            "image_multiple_of": int(image_multiple_of or 32),
            "img_compression": int(img_compression or 0),
            "rows": rows,
            "guide_rows": guide_rows,
            "visual_segments": visual_segments,
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
            "cine_image_resize_method": image_resize_method,
            "cine_image_multiple_of": int(image_multiple_of or 32),
            "cine_img_compression": int(img_compression or 0),
            "cine_multi_input": multi_output,
            "cine_image_1": image_1,
            "cine_visual_segments_json": visual_segments_json,
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
                "promptrelay_gate_mode": "toggle_and_non_empty_local_prompt",
                "promptrelay_backend": "IAMCCS_CineRelayOrBypass",
                "flfreal_visual_timeline": "rows_compiled_to_visual_segments",
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
        image_resize_method=None,
        image_multiple_of=None,
        img_compression=None,
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
            "image_resize_method": _iamccs_cine_resize_method(image_resize_method),
            "image_multiple_of": int(image_multiple_of or 0),
            "img_compression": int(img_compression or 0),
        })

    def execute(self, global_prompt, timeline_data, duration_seconds, frame_rate, guide_policy, min_guide_gap_seconds, max_guides, default_force, promptrelay_epsilon, ltx_round_mode, image_paths, image_width, image_height, image_resize_method="crop", image_multiple_of=32, img_compression=0, multi_input=None):
        fps = max(1, int(frame_rate))
        image_resize_method = _iamccs_cine_resize_method(image_resize_method)
        if str(image_paths or "").strip():
            try:
                multi_input = IAMCCS_CineReferenceBoard().load_ltx_style_images(
                    image_paths,
                    int(image_width),
                    int(image_height),
                    image_resize_method,
                    int(image_multiple_of or 32),
                    int(img_compression or 0),
                )
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
            image_resize_method=image_resize_method,
            image_multiple_of=int(image_multiple_of or 32),
            img_compression=int(img_compression or 0),
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
            "promptrelay_gate_mode": "toggle_and_non_empty_local_prompt",
            "promptrelay_backend": "IAMCCS_CineRelayOrBypass",
            "promptrelay_segment_lengths": segment_lengths,
            "promptrelay_pixel_lengths": latent_lengths_preview,
            "promptrelay_latent_lengths_preview": latent_lengths_preview,
            "max_frames": int(max_frames),
            "promptrelay_epsilon": float(promptrelay_epsilon),
            "truth": "Planner-only node: connect cine_linx to IAMCCS_CineInfo for classic workflow breakouts or to IAMCCS_CineRelayOrBypass for internal Relay/bypass. Relay activates only when a row has use_prompt=True and a non-empty local prompt.",
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

    @classmethod
    def _parse_rows(cls, timeline_data: str, duration_seconds: float, default_force: float) -> List[Dict[str, Any]]:
        rows = super()._parse_rows(timeline_data, duration_seconds, default_force)
        for row in rows:
            if not isinstance(row, dict):
                continue
            bridge_text = str(row.get("step_transition_prompt", "") or row.get("note", "") or "").strip()
            if not bridge_text:
                continue
            row["note"] = bridge_text
            row["step_transition_prompt"] = bridge_text
            row["step_transition_enabled"] = True
            if str(row.get("step_transition_type", "off") or "off").strip() == "off":
                row["step_transition_type"] = "action_beat"
            row["use_prompt"] = True
        return rows

    def execute(self, global_prompt, timeline_data, duration_seconds, frame_rate, guide_policy, min_guide_gap_seconds, max_guides, default_force, promptrelay_epsilon, ltx_round_mode, image_paths, image_width, image_height, image_resize_method="crop", image_multiple_of=32, img_compression=0, multi_input=None):
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
            image_resize_method,
            image_multiple_of,
            img_compression,
            multi_input=multi_input,
        )
        if not isinstance(cine_linx, dict):
            return (cine_linx,)

        fps = max(1, _safe_int(frame_rate, 24))
        resources = cine_linx.setdefault("resources", {})
        outputs = cine_linx.setdefault("outputs", {})
        payload = resources.get("cine_payload") if isinstance(resources.get("cine_payload"), dict) else {}
        visual_segments = _safe_json_loads(str(resources.get("cine_visual_segments_json", "[]") or "[]"), [])
        if not isinstance(visual_segments, list):
            visual_segments = []

        promptrelay_requested = False
        for seg in visual_segments:
            if not isinstance(seg, dict) or str(seg.get("type", "image") or "image").strip().lower() == "audio":
                continue
            prompt = str(seg.get("prompt", seg.get("local_prompt", seg.get("relay_prompt", ""))) or "").strip()
            step_enabled = self._as_bool(seg.get("step_transition_enabled", seg.get("stepTransitionEnabled", False)), False)
            if (self._as_bool(seg.get("use_prompt", bool(prompt)), bool(prompt)) and prompt) or step_enabled:
                promptrelay_requested = True
                break

        flfreal_compile = IAMCCS_CineShotboardPlannerV3._compile_flfreal_timeline(
            timeline_data=json.dumps({"segments": visual_segments, "frame_rate": fps}, ensure_ascii=False),
            duration_seconds=float(duration_seconds),
            frame_rate=fps,
            ltx_round_mode=str(ltx_round_mode),
        )
        flfreal_report = flfreal_compile.get("report", {}) if isinstance(flfreal_compile, dict) else {}
        local_prompts = str(flfreal_compile.get("local_prompts", "") if isinstance(flfreal_compile, dict) else "")
        segment_lengths = str(flfreal_compile.get("segment_lengths", "") if isinstance(flfreal_compile, dict) else "")
        if promptrelay_requested and local_prompts.strip():
            resources["cine_local_prompts"] = local_prompts
            resources["cine_segment_lengths"] = segment_lengths
            resources["cine_promptrelay_enabled"] = True
            resources["cine_max_frames"] = int(flfreal_compile.get("max_frames", outputs.get("max_frames", 0)) or 0)
            outputs["local_prompts"] = local_prompts
            outputs["segment_lengths"] = segment_lengths
            outputs["max_frames"] = resources["cine_max_frames"]
            outputs["promptrelay_enabled"] = True
            payload["local_prompts"] = local_prompts
            payload["segment_lengths"] = segment_lengths
            payload["max_frames"] = resources["cine_max_frames"]
            payload["promptrelay_enabled"] = True
        elif not promptrelay_requested:
            resources["cine_local_prompts"] = ""
            resources["cine_segment_lengths"] = ""
            resources["cine_promptrelay_enabled"] = False
            outputs["local_prompts"] = ""
            outputs["segment_lengths"] = ""
            outputs["promptrelay_enabled"] = False
            payload["local_prompts"] = ""
            payload["segment_lengths"] = ""
            payload["promptrelay_enabled"] = False

        payload["backend_mode"] = "cine_ltx23_filmmaker_timeline"
        payload["filmmaker_schema"] = "iamccs.cine.flfreal_visual_timeline"
        payload["flfreal_compile"] = flfreal_report
        payload["flfreal_parity"] = "prov2_uses_same_visual_segment_compiler_as_v3"
        payload["visual_segments"] = visual_segments
        resources["cine_payload"] = payload
        cine_linx["mode"] = "cine_ltx23_filmmaker_timeline"
        if cine_linx.get("chain") and isinstance(cine_linx["chain"][0], dict):
            cine_linx["chain"][0]["name"] = "Cine Shotboard Planner Pro V2"
        if cine_linx.get("stages") and isinstance(cine_linx["stages"][0], dict):
            cine_linx["stages"][0]["kind"] = "cine_filmmaker_timeline_planner"
        policies = cine_linx.setdefault("policies", {})
        policies["flfreal_visual_timeline"] = "prov2_rows_compiled_to_visual_segments_1to1"
        policies["promptrelay_source"] = "flfreal_visual_segments"
        cine_linx["resource_keys"] = sorted(resources.keys())
        cine_linx["resource_types"] = {key: type(value).__name__ for key, value in resources.items()}

        local_parts = [part.strip() for part in str(resources.get("cine_local_prompts", "") or "").split("|") if part.strip()]
        length_parts = [part for part in re.split(r"[,;\s]+", str(resources.get("cine_segment_lengths", "") or "")) if part.strip()]
        print(
            "[IAMCCS FLFReal] "
            "planner=ProV2 "
            f"compile={bool(flfreal_report.get('available', False))} "
            f"promptrelay_requested={bool(promptrelay_requested)} "
            f"locals={len(local_parts)} segments={len(length_parts)} "
            f"visual_segments={len(visual_segments)} "
            f"action_bridges={int(flfreal_report.get('action_bridges', 0) or 0)} "
            "parity=v3_visual_segment_compiler"
        )
        return (cine_linx,)


class IAMCCS_CineShotboardPlannerV3(IAMCCS_CineShotboardPlannerPro):
    """Filmmaker timeline planner.

    V3 keeps the cine_linx contract but stores an editor-grade timeline with
    visual segments and multi-track audio metadata, inspired by LTX-style guide
    data while remaining fully IAMCCS-owned.
    """

    CATEGORY = "IAMCCS/Cine/02 Single Generation VIP"

    @classmethod
    def INPUT_TYPES(cls):
        data = super().INPUT_TYPES()
        data["required"]["global_prompt"] = ("STRING", {
            "default": "one continuous cinematic shot with coherent motion, stable identity and controlled camera movement",
            "multiline": True,
        })
        data["required"]["timeline_data"] = ("STRING", {
            "default": "",
            "multiline": True,
            "tooltip": "Edited by Shotboard Planner V3. JSON contains segments, rows and audioSegments.",
        })
        data["required"]["duration_seconds"] = ("FLOAT", {"default": 20.0, "min": 0.01, "max": 36000.0, "step": 0.01})
        data["required"]["guide_policy"] = (["every_checked_row", "safe_core_guides", "prompt_only"], {"default": "every_checked_row"})
        data["required"]["min_guide_gap_seconds"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.05})
        data["required"]["max_guides"] = ("INT", {"default": 50, "min": 0, "max": 50, "step": 1})
        data["required"]["default_force"] = ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01})
        return data

    @classmethod
    def _parse_rows(cls, timeline_data: str, duration_seconds: float, default_force: float) -> List[Dict[str, Any]]:
        text = str(timeline_data or "").strip()
        if not text:
            return super()._parse_rows(timeline_data, duration_seconds, default_force)
        try:
            data = json.loads(text)
        except Exception:
            return super()._parse_rows(timeline_data, duration_seconds, default_force)
        if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
            return super()._parse_rows(timeline_data, duration_seconds, default_force)

        fps = max(1, _safe_int(data.get("frame_rate", data.get("fps", 24)), 24))
        rows: List[Dict[str, Any]] = []
        image_order = 0
        for idx, seg in enumerate(sorted(data.get("segments", []), key=lambda item: _safe_int(item.get("start", 0), 0))):
            if not isinstance(seg, dict):
                continue
            if cls._as_bool(seg.get("placeholder", False), False):
                continue
            seg_type = str(seg.get("type", "image") or "image").strip().lower()
            if seg_type == "audio":
                continue
            start_frame = max(0, _safe_int(seg.get("start", seg.get("frame", 0)), 0))
            second = start_frame / fps
            if second >= float(duration_seconds):
                continue
            is_text = seg_type == "text"
            if not is_text:
                image_order += 1
            ref = max(1, _safe_int(seg.get("ref", seg.get("reference_index", seg.get("image_ref", image_order or idx + 1))), image_order or idx + 1))
            strength = _clamp(seg.get("guideStrength", seg.get("guide_strength", seg.get("strength", default_force))), 0.0, 1.0, default_force)
            guide_strength = _wdc_image_guide_strength(seg, strength)
            prompt = str(seg.get("prompt", seg.get("local_prompt", "")) or "").strip()
            label = _normalise_label(str(seg.get("label", seg.get("name", "")) or ""), f"{'text' if is_text else 'shot'}_{idx + 1}")
            rows.append({
                "second": max(0.0, second),
                "frame": int(start_frame),
                "ref": ref,
                "force": 0.0 if is_text else float(strength),
                "motion_force": 0.0 if is_text else float(strength),
                "image_lock_strength": 0.0 if is_text else float(guide_strength),
                "guide_strength": 0.0 if is_text else float(guide_strength),
                "label": label,
                "camera": str(seg.get("camera", seg.get("camera_move", "cinematic motion")) or "cinematic motion").strip(),
                "transition": str(seg.get("transition", "continuous_motion") or "continuous_motion").strip(),
                "note": str(seg.get("note", "") or "").strip(),
                "use_guide": (not is_text) and cls._as_bool(seg.get("use_guide", seg.get("guide", True)), True),
                "use_prompt": bool(cls._as_bool(seg.get("use_prompt", bool(prompt)), bool(prompt))),
                "relay_prompt": prompt,
                "use_relay_modifiers": cls._as_bool(seg.get("use_relay_modifiers", False), False),
                "camera_relay_mode": str(seg.get("camera_relay_mode", "off") or "off").strip(),
                "transition_relay_mode": str(seg.get("transition_relay_mode", "off") or "off").strip(),
                "relay_addon_position": str(seg.get("relay_addon_position", "after") or "after").strip(),
                "relay_modifier_text": str(seg.get("relay_modifier_text", "") or "").strip(),
                "step_transition_enabled": False,
                "step_transition_type": "off",
                "step_transition_prompt": "",
                "step_transition_easing": str(seg.get("step_transition_easing", seg.get("stepTransitionEasing", "ease_in_out")) or "ease_in_out").strip(),
                "step_transition_force_curve": str(seg.get("step_transition_force_curve", seg.get("stepTransitionForceCurve", "late_target")) or "late_target").strip(),
                "step_transition_duration": max(0.0, _safe_float(seg.get("step_transition_duration", seg.get("stepTransitionDuration", seg.get("step_seconds", 0.0))), 0.0)),
                "step_transition_arrival": str(seg.get("step_transition_arrival", seg.get("stepTransitionArrival", "auto")) or "auto").strip(),
                "step_transition_auto_fit": cls._as_bool(seg.get("step_transition_auto_fit", seg.get("stepTransitionAutoFit", True)), True),
            })

        if not rows:
            return super()._parse_rows(timeline_data, duration_seconds, default_force)
        return sorted(rows[:MAX_CINE_ITEMS], key=lambda item: (float(item["second"]), int(item["ref"])))


    @classmethod
    def _action_bridge_prompt(cls, seg: Dict[str, Any], next_seg: Optional[Dict[str, Any]] = None) -> str:
        bridge_type = str(seg.get("step_transition_type", seg.get("stepTransitionType", "off")) or "off").strip()
        custom = str(seg.get("step_transition_prompt", seg.get("stepTransitionPrompt", "")) or "").strip()
        next_label = str((next_seg or {}).get("label", (next_seg or {}).get("name", "next frame")) or "next frame").replace("_", " ")
        preset = {
            "action_beat": f"the action continues toward {next_label}; same subject, same space, coherent temporal development before the next guide",
            "slow_dolly_in": f"one continuous physical slow dolly-in toward {next_label}; preserve the same subject and space, arrive on the next guide framing only near the end",
            "hold_then_push": f"hold the current guide composition first, then push physically toward {next_label}; continuous camera move, no editorial dissolve",
            "orbit_bridge": f"one continuous physical orbit move from the current framing toward {next_label}; preserve identity and spatial continuity",
            "match_move": f"match the motion direction into {next_label}; same scene continuity, coherent camera path",
            "rack_focus": f"rack focus through the same scene toward {next_label}; preserve composition and identity",
            "soft_push": f"soft physical push toward {next_label}; continuous cinematic movement and stable subject identity",
        }.get(bridge_type, "")
        if custom and preset:
            return f"{preset}, {custom}"
        return custom or preset

    @classmethod
    def _compile_flfreal_timeline(
        cls,
        *,
        timeline_data: Any,
        duration_seconds: float,
        frame_rate: int,
        ltx_round_mode: str,
    ) -> Dict[str, Any]:
        data = _safe_json_loads(str(timeline_data or "{}"), {})
        if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
            return {"local_prompts": "", "segment_lengths": "", "max_frames": 0, "lengths": [], "report": {"available": False}}
        fps = max(1, int(frame_rate or data.get("frame_rate", 24) or 24))
        target_frames = max(1, int(round(float(duration_seconds) * fps)))
        max_frames = int(cls._round_frames(target_frames, str(ltx_round_mode or "up_8n_plus_1")))
        duration_frames = max_frames
        visual_segments = [
            seg for seg in data.get("segments", [])
            if isinstance(seg, dict)
            and not cls._as_bool(seg.get("placeholder", False), False)
            and str(seg.get("type", "image") or "image").strip().lower() != "audio"
        ]
        visual_segments.sort(key=lambda item: _safe_int(item.get("start", item.get("frame", 0)), 0))
        prompts: List[str] = []
        lengths: List[int] = []
        missing_prompt_labels: List[str] = []
        action_bridge_count = 0
        pending_gap = 0
        cursor = 0

        def append_segment(prompt: str, length: int) -> None:
            prompt = str(prompt or "").strip()
            length = max(1, int(round(length)))
            if not prompt:
                return
            prompts.append(prompt)
            lengths.append(length)

        for index, seg in enumerate(visual_segments):
            start = max(0, _safe_int(seg.get("start", seg.get("frame", 0)), 0))
            if start >= duration_frames:
                break
            if start > cursor:
                gap = min(start, duration_frames) - cursor
                if lengths:
                    lengths[-1] += max(0, gap)
                else:
                    pending_gap += max(0, gap)
            seg_len = max(1, _safe_int(seg.get("length", seg.get("len", fps)), fps))
            clipped_end = min(start + seg_len, duration_frames)
            clipped_len = max(1, clipped_end - start)
            total_len = max(1, clipped_len + pending_gap)
            pending_gap = 0
            cursor = start + seg_len

            prompt = str(seg.get("prompt", seg.get("local_prompt", seg.get("relay_prompt", ""))) or "").strip()
            use_prompt = cls._as_bool(seg.get("use_prompt", bool(prompt)), bool(prompt))
            step_enabled = False
            step_type = "off"
            next_seg = visual_segments[index + 1] if index + 1 < len(visual_segments) else None

            if step_enabled and step_type != "off" and next_seg is not None:
                bridge_prompt = cls._action_bridge_prompt(seg, next_seg)
                requested = max(0.0, _safe_float(seg.get("step_transition_duration", seg.get("stepTransitionDuration", 0.0)), 0.0))
                bridge_len = total_len if requested <= 0 else max(1, min(total_len, int(round(requested * fps))))
                hold_len = max(0, total_len - bridge_len)
                if hold_len > 0 and prompt and use_prompt:
                    append_segment(prompt, hold_len)
                elif hold_len > 0 and not prompt:
                    append_segment("same continuous shot; keep the current guide alive while preparing the next physical camera action", hold_len)
                append_segment(bridge_prompt, bridge_len)
                action_bridge_count += 1
            elif prompt and use_prompt:
                append_segment(prompt, total_len)
            elif use_prompt:
                missing_prompt_labels.append(str(seg.get("label", seg.get("name", f"segment_{index + 1}")) or f"segment_{index + 1}"))
            elif lengths:
                lengths[-1] += total_len
            else:
                pending_gap += total_len

        clamped_cursor = min(cursor, duration_frames)
        if lengths and clamped_cursor < duration_frames:
            lengths[-1] += duration_frames - clamped_cursor
        report = {
            "available": True,
            "mode": "flfreal_visual_timeline_1to1",
            "visual_segments": len(visual_segments),
            "relay_segments": len(prompts),
            "action_bridges": int(action_bridge_count),
            "missing_prompt_labels": missing_prompt_labels,
            "max_frames": int(max_frames),
            "duration_frames": int(duration_frames),
        }
        return {
            "local_prompts": " | ".join(prompts),
            "segment_lengths": ",".join(str(int(length)) for length in lengths),
            "max_frames": int(max_frames),
            "lengths": lengths,
            "report": report,
        }


    def execute(self, global_prompt, timeline_data, duration_seconds, frame_rate, guide_policy, min_guide_gap_seconds, max_guides, default_force, promptrelay_epsilon, ltx_round_mode, image_paths, image_width, image_height, image_resize_method="crop", image_multiple_of=32, img_compression=0, multi_input=None):
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
            image_resize_method,
            image_multiple_of,
            img_compression,
            multi_input=multi_input,
        )
        data = _safe_json_loads(str(timeline_data or "{}"), {})
        audio_segments = data.get("audioSegments", []) if isinstance(data, dict) else []
        visual_segments = data.get("segments", []) if isinstance(data, dict) else []
        if isinstance(visual_segments, list):
            reference_paths = _cine_reference_paths_from_text(image_paths)
            image_order = 0
            enriched_segments = []
            for seg in visual_segments:
                if not isinstance(seg, dict):
                    enriched_segments.append(seg)
                    continue
                item = dict(seg)
                seg_type = str(item.get("type", "image") or "image").strip().lower()
                if seg_type not in {"text", "audio"} and not self._as_bool(item.get("placeholder", False), False):
                    image_order += 1
                    ref = max(1, _safe_int(item.get("ref", item.get("reference_index", item.get("image_ref", image_order))), image_order))
                    if not str(item.get("imageFile", item.get("image_file", "")) or "").strip() and ref <= len(reference_paths):
                        item["imageFile"] = reference_paths[ref - 1]
                enriched_segments.append(item)
            visual_segments = enriched_segments
        flfreal_mode = str(data.get("flfrealMode", data.get("flfreal_mode", "iamccs_enhanced")) if isinstance(data, dict) else "iamccs_enhanced").strip()
        if flfreal_mode not in {"flfreal_parity", "iamccs_enhanced"}:
            flfreal_mode = "iamccs_enhanced"
        director_local_prompts = str(data.get("director_local_prompts", data.get("local_prompts", "")) if isinstance(data, dict) else "")
        director_segment_lengths = str(data.get("director_segment_lengths", data.get("segment_lengths", "")) if isinstance(data, dict) else "")
        director_guide_strength = str(data.get("director_guide_strength", data.get("guide_strength", "")) if isinstance(data, dict) else "")
        director_audio_data = str(data.get("audio_data", "") if isinstance(data, dict) else "")
        global_prompt_only = bool(_safe_bool(data.get("global_prompt_only", data.get("use_global_prompt_only", False)), False)) if isinstance(data, dict) else False
        fps = max(1, _safe_int(frame_rate, 24))
        promptrelay_requested = False
        if isinstance(data, dict):
            promptrelay_requested = self._as_bool(data.get("promptrelay_enabled", data.get("enable_promptrelay", False)), False)
            if global_prompt_only:
                promptrelay_requested = False
            if not global_prompt_only and not promptrelay_requested and flfreal_mode == "flfreal_parity" and director_local_prompts.strip() and director_segment_lengths.strip():
                promptrelay_requested = True
            if not global_prompt_only and not promptrelay_requested:
                for seg in visual_segments if isinstance(visual_segments, list) else []:
                    if not isinstance(seg, dict) or str(seg.get("type", "image") or "image").strip().lower() == "audio":
                        continue
                    prompt = str(seg.get("prompt", seg.get("local_prompt", seg.get("relay_prompt", ""))) or "").strip()
                    if self._as_bool(seg.get("use_prompt", bool(prompt)), bool(prompt)) and prompt:
                        promptrelay_requested = True
                        break
            if not global_prompt_only and not promptrelay_requested:
                for row in data.get("rows", []) if isinstance(data.get("rows"), list) else []:
                    if not isinstance(row, dict):
                        continue
                    prompt = str(row.get("relay_prompt", row.get("local_prompt", row.get("prompt", ""))) or "").strip()
                    if self._as_bool(row.get("use_prompt", False), False) and prompt:
                        promptrelay_requested = True
                        break
        if isinstance(cine_linx, dict):
            resources = cine_linx.setdefault("resources", {})
            outputs = cine_linx.setdefault("outputs", {})
            payload = resources.get("cine_payload") if isinstance(resources.get("cine_payload"), dict) else {}
            payload["backend_mode"] = "cine_ltx23_filmmaker_timeline"
            payload["filmmaker_schema"] = "iamccs.cine.filmmaker_timeline"
            payload["flfreal_mode"] = flfreal_mode
            payload["director_local_prompts"] = director_local_prompts
            payload["director_segment_lengths"] = director_segment_lengths
            payload["director_guide_strength"] = director_guide_strength
            payload["director_audio_data"] = director_audio_data
            payload["global_prompt_only"] = bool(global_prompt_only)
            payload["filmmaker_promptrelay_enabled"] = bool(promptrelay_requested)
            payload["visual_segments"] = visual_segments if isinstance(visual_segments, list) else []
            payload["audioSegments"] = audio_segments if isinstance(audio_segments, list) else []
            payload["use_custom_audio"] = bool(
                _safe_bool(data.get("use_custom_audio", False), False)
                or any(isinstance(seg, dict) and (seg.get("audioFile") or seg.get("audioB64")) for seg in (audio_segments if isinstance(audio_segments, list) else []))
            )
            resources["cine_use_custom_audio"] = payload["use_custom_audio"]
            if isinstance(data, dict):
                payload["audio_sync_mode"] = str(data.get("audioSyncMode", "timeline_audio") or "timeline_audio")
                payload["generation_strategy"] = str(data.get("generationStrategy", "single_timeline") or "single_timeline")
            if not promptrelay_requested:
                payload["local_prompts"] = ""
                payload["segment_lengths"] = ""
                resources["cine_local_prompts"] = ""
                resources["cine_segment_lengths"] = ""
                outputs["local_prompts"] = ""
                outputs["segment_lengths"] = ""
            resources["cine_payload"] = payload
            resources["cine_audio_timeline_json"] = json.dumps(audio_segments if isinstance(audio_segments, list) else [], ensure_ascii=False)
            resources["cine_visual_segments_json"] = json.dumps(visual_segments if isinstance(visual_segments, list) else [], ensure_ascii=False)
            resources["cine_director_local_prompts"] = director_local_prompts
            resources["cine_director_segment_lengths"] = director_segment_lengths
            resources["cine_director_guide_strength"] = director_guide_strength
            resources["cine_director_audio_data"] = director_audio_data
            local_parts = [part.strip() for part in str(resources.get("cine_local_prompts", outputs.get("local_prompts", payload.get("local_prompts", ""))) or "").split("|") if part.strip()]
            length_parts = [part for part in re.split(r"[,;\s]+", str(resources.get("cine_segment_lengths", outputs.get("segment_lengths", payload.get("segment_lengths", ""))) or "")) if part.strip()]
            rebuilt_step_prompts = False
            flfreal_compile = self._compile_flfreal_timeline(
                timeline_data=timeline_data,
                duration_seconds=float(duration_seconds),
                frame_rate=fps,
                ltx_round_mode=str(ltx_round_mode),
            )
            flfreal_report = flfreal_compile.get("report", {}) if isinstance(flfreal_compile, dict) else {}
            has_compiled_action_bridges = int(flfreal_report.get("action_bridges", 0) or 0) > 0
            if (
                flfreal_mode == "flfreal_parity"
                and promptrelay_requested
                and director_local_prompts.strip()
                and director_segment_lengths.strip()
                and not has_compiled_action_bridges
            ):
                resources["cine_local_prompts"] = director_local_prompts
                resources["cine_segment_lengths"] = director_segment_lengths
                resources["cine_promptrelay_enabled"] = True
                resources["cine_max_frames"] = int(flfreal_compile.get("max_frames", outputs.get("max_frames", 0)) or 0)
                outputs["local_prompts"] = resources["cine_local_prompts"]
                outputs["segment_lengths"] = resources["cine_segment_lengths"]
                outputs["max_frames"] = resources["cine_max_frames"]
                outputs["promptrelay_enabled"] = True
                payload["local_prompts"] = resources["cine_local_prompts"]
                payload["segment_lengths"] = resources["cine_segment_lengths"]
                payload["max_frames"] = resources["cine_max_frames"]
                payload["flfreal_compile"] = {**flfreal_report, "source": "director_style_reflection", "mode": "flfreal_parity"}
                payload["promptrelay_enabled"] = True
                local_parts = [part.strip() for part in resources["cine_local_prompts"].split("|") if part.strip()]
                length_parts = [part for part in re.split(r"[,;\s]+", resources["cine_segment_lengths"]) if part.strip()]
                resources["cine_payload"] = payload
            elif promptrelay_requested and str(flfreal_compile.get("local_prompts", "") if isinstance(flfreal_compile, dict) else "").strip():
                resources["cine_local_prompts"] = str(flfreal_compile.get("local_prompts", ""))
                resources["cine_segment_lengths"] = str(flfreal_compile.get("segment_lengths", ""))
                resources["cine_promptrelay_enabled"] = True
                resources["cine_max_frames"] = int(flfreal_compile.get("max_frames", outputs.get("max_frames", 0)) or 0)
                outputs["local_prompts"] = resources["cine_local_prompts"]
                outputs["segment_lengths"] = resources["cine_segment_lengths"]
                outputs["max_frames"] = resources["cine_max_frames"]
                outputs["promptrelay_enabled"] = True
                payload["local_prompts"] = resources["cine_local_prompts"]
                payload["segment_lengths"] = resources["cine_segment_lengths"]
                payload["max_frames"] = resources["cine_max_frames"]
                payload["flfreal_compile"] = {
                    **flfreal_report,
                    "source": "compiled_visual_segments",
                    "mode": flfreal_mode,
                    "director_reflection_bypassed_for_action_bridge": bool(has_compiled_action_bridges),
                }
                payload["promptrelay_enabled"] = True
                local_parts = [part.strip() for part in resources["cine_local_prompts"].split("|") if part.strip()]
                length_parts = [part for part in re.split(r"[,;\s]+", resources["cine_segment_lengths"]) if part.strip()]
                resources["cine_payload"] = payload
            elif isinstance(payload, dict):
                payload["flfreal_compile"] = flfreal_report
                resources["cine_payload"] = payload
            if promptrelay_requested and not local_parts:
                rows = self._parse_rows(timeline_data, float(duration_seconds), float(default_force))
                rebuilt_local_prompts, rebuilt_segment_lengths, rebuilt_max_frames, rebuilt_lengths = self._segments_from_rows(
                    rows,
                    float(duration_seconds),
                    fps,
                    str(ltx_round_mode),
                )
                if str(rebuilt_local_prompts or "").strip():
                    rebuilt_step_prompts = True
                    resources["cine_local_prompts"] = rebuilt_local_prompts
                    resources["cine_segment_lengths"] = rebuilt_segment_lengths
                    resources["cine_promptrelay_enabled"] = True
                    resources["cine_max_frames"] = int(rebuilt_max_frames)
                    outputs["local_prompts"] = rebuilt_local_prompts
                    outputs["segment_lengths"] = rebuilt_segment_lengths
                    outputs["max_frames"] = int(rebuilt_max_frames)
                    outputs["promptrelay_enabled"] = True
                    payload["local_prompts"] = rebuilt_local_prompts
                    payload["segment_lengths"] = rebuilt_segment_lengths
                    payload["max_frames"] = int(rebuilt_max_frames)
                    payload["promptrelay_pixel_lengths"] = rebuilt_lengths
                    payload["promptrelay_latent_lengths_preview"] = rebuilt_lengths
                    payload["promptrelay_enabled"] = True
                    local_parts = [part.strip() for part in str(rebuilt_local_prompts or "").split("|") if part.strip()]
                    length_parts = [part for part in re.split(r"[,;\s]+", str(rebuilt_segment_lengths or "")) if part.strip()]
                    resources["cine_payload"] = payload
            step_count = 0
            flfreal_report_for_log = payload.get("flfreal_compile", {}) if isinstance(payload, dict) else {}
            print(
                "[IAMCCS FLFReal] "
                f"compile={bool(flfreal_report_for_log.get('available', False))} "
                f"promptrelay_requested={bool(promptrelay_requested)} "
                f"locals={len(local_parts)} segments={len(length_parts)} "
                f"visual_segments={len(visual_segments) if isinstance(visual_segments, list) else 0} "
                f"action_bridges={int(flfreal_report_for_log.get('action_bridges', step_count) or 0)} "
                f"rebuilt_legacy_prompts={rebuilt_step_prompts} "
                f"mode={flfreal_mode}"
            )
            if promptrelay_requested and local_parts:
                local_hash = hashlib.sha1(str(resources.get("cine_local_prompts", "") or "").encode("utf-8", errors="ignore")).hexdigest()[:12]
                print(
                    "[IAMCCS FLFReal] "
                    f"LOCAL_PROMPTS_USED source=shotboard_v3_timeline "
                    f"count={len(local_parts)} segments={len(length_parts)} "
                    f"local_hash={local_hash}"
                )
                for idx, prompt in enumerate(local_parts[:50]):
                    compact = str(prompt or "").replace("\n", " ")
                    if len(compact) > 360:
                        compact = compact[:357] + "..."
                    length = length_parts[idx] if idx < len(length_parts) else "<missing>"
                    print(f"[IAMCCS FLFReal] local[{idx:02d}] length={length} prompt={compact!r}")
                if len(local_parts) > 50:
                    print(f"[IAMCCS FLFReal] local prompt log truncated: {len(local_parts) - 50} more prompts.")
            cine_linx["mode"] = "cine_ltx23_filmmaker_timeline"
            cine_linx["chain"][0]["name"] = "Cine Shotboard Planner V3"
            cine_linx["stages"][0]["kind"] = "cine_filmmaker_timeline_planner"
            outputs["audio_timeline_json"] = resources["cine_audio_timeline_json"]
            cine_linx["resource_keys"] = sorted(resources.keys())
            cine_linx["resource_types"] = {key: type(value).__name__ for key, value in resources.items()}
        return (cine_linx,)


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
                    "default": "one continuous cinematic FLF shot with coherent motion through the image guides",
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
            image_resize_method="crop",
            image_multiple_of=32,
            img_compression=0,
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
    Keep this clone for older hand-wired FLFreal/PromptRelay workflows that still
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

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

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
        relay_prompt_log = []
        for idx, prompt in enumerate(locals_list):
            relay_prompt_log.append({
                "index": idx,
                "segment_length": length_parts[idx] if idx < len(length_parts) else "<missing>",
                "prompt": prompt,
            })
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


class IAMCCS_CineInfoV2(IAMCCS_CineInfo):
    """Production breakout: exposes a verified guide plan for the FLF Productor."""

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "INT", "FLOAT", "STRING", "IMAGE", "IMAGE", "INT", "INT", SUPERNODE_LINX_TYPE, "IMAGE", "INT", "INT", "GUIDE_DATA")
    RETURN_NAMES = ("guide_plan_json", "flf_timeline", "global_prompt", "local_prompts", "segment_lengths", "max_frames", "promptrelay_epsilon", "report", "multi_output", "image_1", "duration_seconds_int", "frame_rate_int", "cine_linx", "first_stage_preview", "width", "height", "guide_data")
    CATEGORY = "IAMCCS/Cine/00 Utilities"

    @staticmethod
    def _safe_shape(value: Any) -> List[int]:
        return [int(v) for v in value.shape] if torch.is_tensor(value) else []

    @staticmethod
    def _image_stats(batch: Any, ref_index: int) -> Dict[str, Any]:
        if not torch.is_tensor(batch) or ref_index < 1 or ref_index > int(batch.shape[0]):
            return {"available": False}
        try:
            img = batch[ref_index - 1 : ref_index].detach().float().cpu()
            return {
                "available": True,
                "shape": [int(v) for v in img.shape],
                "mean": round(float(img.mean().item()), 6),
                "std": round(float(img.std().item()), 6),
            }
        except Exception:
            return {"available": True, "stats_error": True}

    @classmethod
    def _build_guide_plan(
        cls,
        *,
        flf_timeline: str,
        image_paths: str,
        multi_output: Any,
        duration_seconds: float,
        frame_rate: int,
        default_force: float = 0.24,
    ) -> Tuple[str, str]:
        fps = max(1, int(frame_rate))
        reference_paths = _cine_reference_paths_from_text(image_paths)
        reference_count = int(multi_output.shape[0]) if torch.is_tensor(multi_output) and multi_output.ndim >= 1 else 0
        keyframes = IAMCCS_CineLTXSequencer._parse_timeline(
            flf_timeline,
            float(duration_seconds),
            fps,
            max(1, reference_count),
            float(default_force),
        )
        guides: List[Dict[str, Any]] = []
        warnings: List[str] = []
        for order, key in enumerate(keyframes):
            ref = max(1, int(key.get("reference_index", order + 1)))
            frame = int(key.get("frame", int(round(float(key.get("second", 0.0)) * fps))))
            guide_strength = _wdc_image_guide_strength(key, _safe_float(key.get("strength", 1.0), 1.0))
            motion_force = _safe_float(key.get("motion_force", key.get("force", default_force)), default_force)
            item = {
                "order": int(order + 1),
                "label": str(key.get("label", f"guide_{order + 1}")),
                "ref": int(ref),
                "reference_index": int(ref),
                "second": float(key.get("second", 0.0)),
                "frame": int(frame),
                "strength": float(guide_strength),
                "guide_strength": float(guide_strength),
                "image_lock_strength": float(guide_strength),
                "motion_force": float(motion_force),
                "force": float(motion_force),
                "camera": str(key.get("camera", "")),
                "step_transition_enabled": _safe_bool(key.get("step_transition_enabled"), False),
                "step_transition_type": str(key.get("step_transition_type", "off") or "off"),
                "step_transition_duration": _safe_float(key.get("step_transition_duration", 0.0), 0.0),
                "step_transition_arrival": str(key.get("step_transition_arrival", "auto") or "auto"),
                "path": reference_paths[ref - 1] if ref - 1 < len(reference_paths) else "",
                "image": cls._image_stats(multi_output, ref),
            }
            if ref > reference_count:
                warnings.append(f"{item['label']}: ref {ref} is not loaded; batch has {reference_count}.")
            if frame < 0:
                item["frame_policy"] = "tail"
            guides.append(item)

        plan = {
            "schema": "iamccs.cine.guide_plan",
            "schema_version": 2,
            "source": "IAMCCS_CineInfoV2",
            "duration_seconds": float(duration_seconds),
            "frame_rate": fps,
            "reference_count": int(reference_count),
            "reference_paths": reference_paths,
            "multi_output_shape": cls._safe_shape(multi_output),
            "guides": guides,
            "warnings": warnings,
            "truth": "This plan is the explicit contract between ShotPlanner and the FLF Productor.",
        }
        report = _json_report({
            "node": "IAMCCS_CineInfoV2",
            "mode": "guide_plan_breakout",
            "duration_seconds": float(duration_seconds),
            "frame_rate": fps,
            "reference_count": int(reference_count),
            "guide_count": len(guides),
            "guide_strength_semantics": "ltx_director_single_guide_strength",
            "warnings": warnings,
            "guide_summary": [
                {
                    "label": item["label"],
                    "ref": item["ref"],
                    "frame": item["frame"],
                    "second": item["second"],
                    "strength": item["strength"],
                    "motion_force": item["motion_force"],
                    "path": item["path"],
                }
                for item in guides
            ],
        })
        return json.dumps(plan, indent=2, ensure_ascii=False), report

    @staticmethod
    def _guide_data_from_plan(guide_plan_json: Any, multi_output: Any) -> Dict[str, Any]:
        plan = _safe_json_loads(str(guide_plan_json or "{}"))
        fps = _safe_int(plan.get("frame_rate", 24), 24) if isinstance(plan, dict) else 24
        guides = plan.get("guides", []) if isinstance(plan, dict) else []
        guide_data: Dict[str, Any] = {
            "images": [],
            "insert_frames": [],
            "strengths": [],
            "frame_rate": int(max(1, fps)),
            "labels": [],
            "reference_indices": [],
            "source": "IAMCCS_CineInfoV2",
        }
        if not torch.is_tensor(multi_output) or multi_output.ndim != 4:
            return guide_data

        batch_size = int(multi_output.shape[0])
        for idx, guide in enumerate(guides):
            if not isinstance(guide, dict):
                continue
            ref = max(1, _safe_int(guide.get("ref", guide.get("reference_index", idx + 1)), idx + 1))
            if ref > batch_size:
                continue
            strength = _wdc_image_guide_strength(guide, _safe_float(guide.get("strength", 1.0), 1.0))
            if strength <= 0:
                continue
            frame = _safe_int(
                guide.get("frame", int(round(_safe_float(guide.get("second", 0.0), 0.0) * max(1, fps)))),
                0,
            )
            guide_data["images"].append(multi_output[ref - 1 : ref])
            guide_data["insert_frames"].append(int(frame))
            guide_data["strengths"].append(float(strength))
            guide_data["labels"].append(str(guide.get("label", f"guide_{idx + 1}")))
            guide_data["reference_indices"].append(int(ref))
            guide_data.setdefault("motion_forces", []).append(
                float(_safe_float(guide.get("motion_force", guide.get("force", 0.0)), 0.0))
            )
            guide_data.setdefault("image_lock_strengths", []).append(float(strength))
            step_type = str(guide.get("step_transition_type", "off") or "off").strip().lower()
            guide_data.setdefault("step_transition_sources", []).append(
                bool(_safe_bool(guide.get("step_transition_enabled"), False) and step_type != "off")
            )
        return guide_data

    def extract(self, cine_linx):
        base = super().extract(cine_linx)
        (
            flf_timeline,
            global_prompt,
            local_prompts,
            segment_lengths,
            max_frames,
            promptrelay_epsilon,
            base_report,
            multi_output,
            image_1,
            duration_seconds_int,
            frame_rate_int,
            cine_linx_out,
            first_stage_preview,
            width,
            height,
        ) = base
        resources = self._resources(cine_linx)
        outputs = self._outputs(cine_linx)
        payload = resources.get("cine_payload") if isinstance(resources.get("cine_payload"), dict) else {}
        duration_seconds = _safe_float(resources.get("cine_duration_seconds", outputs.get("duration_seconds", payload.get("duration_seconds", duration_seconds_int))), float(duration_seconds_int))
        frame_rate = _safe_int(resources.get("cine_frame_rate", outputs.get("frame_rate", payload.get("frame_rate", frame_rate_int))), int(frame_rate_int))
        default_force = _safe_float(resources.get("cine_default_force", payload.get("default_force", 0.24)), 0.24)
        image_paths = str(resources.get("cine_image_paths", payload.get("image_paths", "")) or "")
        guide_plan_json, plan_report = self._build_guide_plan(
            flf_timeline=str(flf_timeline or ""),
            image_paths=image_paths,
            multi_output=multi_output,
            duration_seconds=float(duration_seconds),
            frame_rate=int(frame_rate),
            default_force=float(default_force),
        )
        guide_data = self._guide_data_from_plan(guide_plan_json, multi_output)
        report = _json_report({
            "node": "IAMCCS_CineInfoV2",
            "base_report": _safe_json_loads(str(base_report or "{}")),
            "guide_plan_report": _safe_json_loads(str(plan_report or "{}")),
            "guide_data_report": {
                "schema": "iamccs.GUIDE_DATA.compat",
                "count": len(guide_data.get("images", [])),
                "insert_frames": guide_data.get("insert_frames", []),
                "strengths": guide_data.get("strengths", []),
                "reference_indices": guide_data.get("reference_indices", []),
            },
        })
        return (
            guide_plan_json,
            flf_timeline,
            global_prompt,
            local_prompts,
            segment_lengths,
            int(max_frames),
            float(promptrelay_epsilon),
            report,
            multi_output,
            image_1,
            int(round(duration_seconds)),
            int(frame_rate),
            cine_linx_out,
            first_stage_preview,
            int(width),
            int(height),
            guide_data,
        )


class IAMCCS_CineFLFProductor:
    """Production FLF backend driven by the explicit guide plan from CineInfo V2."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "multi_input": ("IMAGE",),
                "guide_plan_json": ("STRING", {"default": "", "multiline": True}),
                "strength_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "tail_safety_frames": ("INT", {"default": 0, "min": 0, "max": 240, "step": 1}),
            },
            "optional": {
                "timeline_data": ("STRING", {"default": "", "multiline": True}),
                "duration_seconds": ("INT", {"default": 20, "min": 1, "max": 36000, "step": 1}),
                "frame_rate": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1}),
                "guide_data": ("GUIDE_DATA",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("positive", "negative", "latent", "report")
    FUNCTION = "execute"
    CATEGORY = "IAMCCS/Cine/02 Single Generation"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    @staticmethod
    def _parse_plan(guide_plan_json: Any, timeline_data: Any, duration_seconds: Any, frame_rate: Any) -> Dict[str, Any]:
        text = str(guide_plan_json or "").strip()
        if text:
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        fps = max(1, _safe_int(frame_rate, 24))
        duration = max(0.1, _safe_float(duration_seconds, 20.0))
        keyframes = IAMCCS_CineLTXSequencer._parse_timeline(str(timeline_data or ""), duration, fps, MAX_CINE_ITEMS, 0.24)
        return {
            "schema": "iamccs.cine.guide_plan",
            "schema_version": 2,
            "source": "fallback_timeline_data",
            "duration_seconds": duration,
            "frame_rate": fps,
            "reference_count": MAX_CINE_ITEMS,
            "guides": [
                {
                    "order": idx + 1,
                    "label": key.get("label", f"guide_{idx + 1}"),
                    "ref": int(key.get("reference_index", idx + 1)),
                    "reference_index": int(key.get("reference_index", idx + 1)),
                    "second": float(key.get("second", 0.0)),
                    "frame": int(key.get("frame", 0)),
                    "strength": float(_wdc_image_guide_strength(key, _safe_float(key.get("strength", 1.0), 1.0))),
                    "motion_force": float(_safe_float(key.get("motion_force", key.get("force", 0.0)), 0.0)),
                    "camera": str(key.get("camera", "")),
                }
                for idx, key in enumerate(keyframes)
            ],
        }

    @staticmethod
    def _clean_guides(plan: Dict[str, Any], strength_scale: float) -> List[Dict[str, Any]]:
        raw_guides = plan.get("guides")
        if not isinstance(raw_guides, list):
            raw_guides = plan.get("keyframes") if isinstance(plan.get("keyframes"), list) else []
        guides: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_guides):
            if not isinstance(item, dict):
                continue
            ref = max(1, _safe_int(item.get("ref", item.get("reference_index", idx + 1)), idx + 1))
            frame = _safe_int(item.get("frame", int(round(_safe_float(item.get("second", 0.0), 0.0) * _safe_int(plan.get("frame_rate", 24), 24)))), 0)
            strength = _clamp(_wdc_image_guide_strength(item, _safe_float(item.get("strength", 1.0), 1.0)) * float(strength_scale), 0.0, 1.0, 1.0)
            if strength <= 0:
                continue
            motion_force = _safe_float(item.get("motion_force", item.get("force", 0.0)), 0.0)
            guides.append({
                "order": _safe_int(item.get("order", idx + 1), idx + 1),
                "label": str(item.get("label", f"guide_{idx + 1}")),
                "ref": int(ref),
                "frame": int(frame),
                "second": _safe_float(item.get("second", frame / max(1, _safe_int(plan.get("frame_rate", 24), 24))), 0.0),
                "strength": float(strength),
                "guide_strength": float(strength),
                "image_lock_strength": float(strength),
                "motion_force": float(motion_force),
                "force": float(motion_force),
                "camera": str(item.get("camera", "")),
                "step_transition_enabled": _safe_bool(item.get("step_transition_enabled"), False),
                "step_transition_type": str(item.get("step_transition_type", "off") or "off"),
                "step_transition_duration": _safe_float(item.get("step_transition_duration", 0.0), 0.0),
                "step_transition_arrival": str(item.get("step_transition_arrival", "auto") or "auto"),
                "path": str(item.get("path", "")),
            })
        return sorted(guides, key=lambda item: ((10**9 if int(item["frame"]) < 0 else int(item["frame"])), int(item["order"])))

    @staticmethod
    def _execute_guide_data(positive, negative, vae, latent, guide_data: Dict[str, Any], strength_scale: float, tail_safety_frames: int):
        latent_samples = latent.get("samples") if isinstance(latent, dict) else None
        if not torch.is_tensor(latent_samples) or latent_samples.ndim != 5:
            return positive, negative, latent, {
                "applied_guides": [],
                "skipped_guides": [{"reason": "invalid latent samples"}],
                "latent_pixel_frames": 0,
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

        applied = []
        skipped = []
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
            except Exception as exc:
                skipped.append({"label": label, "reference_index": int(ref), "frame": int(frame_idx), "reason": f"guide_data apply failed: {exc}"})
                continue
            applied.append({
                "label": label,
                "reference_index": int(ref),
                "requested_frame": int(requested_frame),
                "frame": int(frame_idx),
                "strength": float(strength),
            })

        return positive, negative, {"samples": latent_image, "noise_mask": noise_mask}, {
            "applied_guides": applied,
            "skipped_guides": skipped,
            "latent_pixel_frames": int(latent_pixel_frames),
        }

    @classmethod
    def _guide_data_from_plan_and_images(cls, plan: Dict[str, Any], multi_input: Any, strength_scale: float = 1.0) -> Dict[str, Any]:
        guides = cls._clean_guides(plan, float(strength_scale))
        guide_data: Dict[str, Any] = {
            "images": [],
            "insert_frames": [],
            "strengths": [],
            "frame_rate": max(1, _safe_int(plan.get("frame_rate", 24), 24)),
            "labels": [],
            "reference_indices": [],
            "source": "IAMCCS_CineFLFProductor.clean_plan_adapter",
        }
        if not torch.is_tensor(multi_input) or multi_input.ndim != 4:
            return guide_data
        batch_size = int(multi_input.shape[0])
        for idx, guide in enumerate(guides):
            ref = int(guide.get("ref", idx + 1))
            if ref < 1 or ref > batch_size:
                continue
            guide_data["images"].append(multi_input[ref - 1 : ref])
            guide_data["insert_frames"].append(int(guide.get("frame", 0)))
            guide_data["strengths"].append(float(guide.get("strength", 0.24)))
            guide_data["labels"].append(str(guide.get("label", f"guide_{idx + 1}")))
            guide_data["reference_indices"].append(int(ref))
            guide_data.setdefault("motion_forces", []).append(float(_safe_float(guide.get("motion_force", guide.get("force", 0.0)), 0.0)))
            guide_data.setdefault("image_lock_strengths", []).append(float(guide.get("strength", 1.0)))
            step_type = str(guide.get("step_transition_type", "off") or "off").strip().lower()
            is_step_source = bool(_safe_bool(guide.get("step_transition_enabled"), False) and step_type != "off")
            prev_step_type = str(guides[idx - 1].get("step_transition_type", "off") or "off").strip().lower() if idx > 0 else "off"
            is_step_target = bool(
                idx > 0
                and _safe_bool(guides[idx - 1].get("step_transition_enabled"), False)
                and prev_step_type != "off"
            )
            guide_data.setdefault("step_transition_sources", []).append(is_step_source)
            guide_data.setdefault("step_transition_targets", []).append(is_step_target)
            guide_data.setdefault("step_transition_protected", []).append(bool(is_step_source or is_step_target))
        return guide_data

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
    ):
        if isinstance(guide_data, dict) and guide_data.get("images"):
            positive, negative, current_latent, guide_data_report = self._execute_guide_data(
                positive,
                negative,
                vae,
                latent,
                guide_data,
                float(strength_scale),
                int(tail_safety_frames),
            )
            report = _json_report({
                "node": "IAMCCS_CineFLFProductor",
                "mode": "flfreal_guide_data_compatible",
                "guide_count": len(guide_data.get("images", [])),
                "applied_count": len(guide_data_report.get("applied_guides", [])),
                "skipped_count": len(guide_data_report.get("skipped_guides", [])),
                "guide_attention_entries": False,
                "guide_strength_semantics": "clean_noise_mask_without_attention_attenuation",
                **guide_data_report,
                "truth": "Consumed GUIDE_DATA directly; image tensors, frames and strengths come from the upstream guide_data contract.",
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
        )

        report = _json_report({
            "node": "IAMCCS_CineFLFProductor",
            "mode": "explicit_clean_guide_productor",
            "plan_source": plan.get("source", ""),
            "reference_count": int(multi_input.shape[0]) if torch.is_tensor(multi_input) and multi_input.ndim == 4 else 0,
            "guide_count": len(clean_guide_data.get("images", [])),
            "applied_count": len(guide_data_report.get("applied_guides", [])),
            "skipped_count": len(guide_data_report.get("skipped_guides", [])),
            "guide_attention_entries": False,
            "guide_strength_semantics": "clean_noise_mask_without_attention_attenuation",
            **guide_data_report,
            "truth": "Each guide is converted to GUIDE_DATA and applied without legacy guide_attention_entries, matching the Filmmaker V3 clean guide behavior.",
        })
        return positive, negative, current_latent, report


class IAMCCS_CineFilmmaker(IAMCCS_CineInfoV2):
    """Filmmaker backend breakout for Shotboard Planner V3."""

    RETURN_TYPES = IAMCCS_CineInfoV2.RETURN_TYPES + ("STRING", "STRING")
    RETURN_NAMES = IAMCCS_CineInfoV2.RETURN_NAMES + ("audio_timeline_json", "visual_segments_json")
    CATEGORY = "IAMCCS/Cine/00 Utilities"

    def extract(self, cine_linx):
        base = super().extract(cine_linx)
        resources = self._resources(cine_linx)
        outputs = self._outputs(cine_linx)
        audio_timeline_json = str(resources.get("cine_audio_timeline_json", outputs.get("audio_timeline_json", "[]")) or "[]")
        visual_segments_json = str(resources.get("cine_visual_segments_json", outputs.get("visual_segments_json", "[]")) or "[]")
        return (*base, audio_timeline_json, visual_segments_json)


class IAMCCS_CineFilmmakerBackend:
    """Filmmaker runtime backend that consumes the FLFreal visual timeline trunk.

    It consumes Shotboard V3 cine_linx and emits the model/conditioning/latent/
    guide_data outputs needed by the LTX sampling chain.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "cine_linx": (SUPERNODE_LINX_TYPE,),
            },
            "optional": {
                "audio_vae": ("VAE",),
                "optional_latent": ("LATENT",),
                "use_custom_audio": ("BOOLEAN", {"default": False}),
                "promptrelay_safety_budget_enabled": ("BOOLEAN", {"default": True}),
                "promptrelay_safety_budget": ("INT", {"default": 24000, "min": 1000, "max": 5000000, "step": 1000}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "LATENT", "LATENT", "GUIDE_DATA", "FLOAT", "AUDIO", "STRING")
    RETURN_NAMES = ("model", "positive", "video_latent", "audio_latent", "guide_data", "frame_rate", "combined_audio", "report")
    FUNCTION = "execute"
    CATEGORY = "IAMCCS/Cine/02 Single Generation"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    @staticmethod
    def _empty_latent(width: int, height: int, pixel_frames: int) -> Dict[str, Any]:
        import comfy.model_management

        latent_w = max(32, (int(width) // 32) * 32)
        latent_h = max(32, (int(height) // 32) * 32)
        latent_t = ((max(1, int(pixel_frames)) - 1) // 8) + 1
        return {
            "samples": torch.zeros(
                [1, 128, latent_t, latent_h // 32, latent_w // 32],
                device=comfy.model_management.intermediate_device(),
            )
        }

    @staticmethod
    def _empty_audio_latent(audio_vae: Any, pixel_frames: int, frame_rate: float) -> Dict[str, Any]:
        if audio_vae is None:
            return {}
        try:
            import comfy.model_management

            z_channels = int(getattr(audio_vae, "latent_channels"))
            audio_freq = int(getattr(audio_vae, "latent_frequency_bins"))
            num_audio_latents = int(audio_vae.num_of_latents_from_frames(max(1, int(pixel_frames)), int(round(float(frame_rate)))))
            audio_latents = torch.zeros(
                (1, z_channels, num_audio_latents, audio_freq),
                device=comfy.model_management.intermediate_device(),
            )
            sample_rate = int(getattr(audio_vae, "sample_rate", 44100))
            return {"samples": audio_latents, "sample_rate": sample_rate, "type": "audio"}
        except Exception:
            pass
        try:
            from comfy_extras.nodes_lt_audio import LTXVEmptyLatentAudio

            result = LTXVEmptyLatentAudio.execute(max(1, int(pixel_frames)), int(round(float(frame_rate))), 1, audio_vae)
            if isinstance(result, tuple):
                result = result[0]
            if isinstance(result, dict) and torch.is_tensor(result.get("samples")):
                return result
            if hasattr(result, "result") and result.result:
                candidate = result.result[0]
                if isinstance(candidate, dict) and torch.is_tensor(candidate.get("samples")):
                    return candidate
        except Exception:
            return {}
        return {}

    @staticmethod
    def _empty_audio(pixel_frames: int, frame_rate: float) -> Dict[str, Any]:
        target_sr = 44100
        safe_fps = max(1.0, float(frame_rate or 24.0))
        total_samples = max(1, int(math.ceil(max(1, int(pixel_frames)) / safe_fps * target_sr)))
        return {"waveform": torch.zeros((1, 2, total_samples), dtype=torch.float32), "sample_rate": target_sr}

    @staticmethod
    def _has_custom_audio(audio_timeline_json: str) -> bool:
        data = _safe_json_loads(str(audio_timeline_json or "[]"), [])
        audio_segments = data if isinstance(data, list) else data.get("audioSegments", []) if isinstance(data, dict) else []
        return any(isinstance(seg, dict) and (seg.get("audioFile") or seg.get("audioB64")) for seg in audio_segments)

    @classmethod
    def _build_combined_audio(cls, audio_timeline_json: str, pixel_frames: int, frame_rate: float) -> Dict[str, Any]:
        target_sr = 44100
        audio_out = cls._empty_audio(pixel_frames, frame_rate)
        data = _safe_json_loads(str(audio_timeline_json or "[]"), [])
        audio_segments = data if isinstance(data, list) else data.get("audioSegments", []) if isinstance(data, dict) else []
        if not isinstance(audio_segments, list) or not audio_segments:
            return audio_out

        try:
            import av
        except Exception:
            return audio_out

        safe_fps = max(1.0, float(frame_rate or 24.0))
        out_waveform = torch.zeros((2, audio_out["waveform"].shape[-1]), dtype=torch.float32)
        input_root = folder_paths.get_input_directory()

        for seg in audio_segments:
            if not isinstance(seg, dict):
                continue
            buffer = None
            audio_file = str(seg.get("audioFile") or "")
            if audio_file:
                file_path = os.path.abspath(os.path.join(input_root, audio_file))
                input_abs = os.path.abspath(input_root)
                try:
                    inside_input = os.path.commonpath([input_abs, file_path]) == input_abs
                except ValueError:
                    inside_input = False
                if inside_input and os.path.exists(file_path):
                    with open(file_path, "rb") as handle:
                        buffer = io.BytesIO(handle.read())
            if buffer is None and seg.get("audioB64"):
                encoded = str(seg.get("audioB64") or "")
                if "," in encoded:
                    encoded = encoded.split(",", 1)[1]
                try:
                    buffer = io.BytesIO(base64.b64decode(encoded))
                except Exception:
                    buffer = None
            if buffer is None:
                continue

            try:
                clip_frames = []
                with av.open(buffer) as container:
                    audio_streams = list(container.streams.audio)
                    if not audio_streams:
                        continue
                    resampler = av.AudioResampler(format="fltp", layout="stereo", rate=target_sr)
                    for frame in container.decode(audio_streams[0]):
                        for resampled_frame in resampler.resample(frame):
                            clip_frames.append(torch.from_numpy(resampled_frame.to_ndarray()))
                    for resampled_frame in resampler.resample(None):
                        clip_frames.append(torch.from_numpy(resampled_frame.to_ndarray()))
                if not clip_frames:
                    continue

                waveform = torch.cat(clip_frames, dim=1).to(torch.float32)
                trim_start_frames = max(0.0, float(seg.get("trimStart", 0) or 0))
                start_frames = max(0.0, float(seg.get("start", 0) or 0))
                length_frames = max(1.0, float(seg.get("length", 1) or 1))
                src_start = max(0, int(trim_start_frames / safe_fps * target_sr))
                src_end = min(waveform.shape[1], src_start + int(length_frames / safe_fps * target_sr))
                if src_end <= src_start:
                    continue
                clip_waveform = waveform[:, src_start:src_end]
                gain = _clamp(seg.get("gain", seg.get("volume", 1.0)), 0.0, 2.0, 1.0)
                if abs(float(gain) - 1.0) > 0.0001:
                    clip_waveform = clip_waveform * float(gain)
                dst_start = int(start_frames / safe_fps * target_sr)
                if dst_start >= out_waveform.shape[1]:
                    continue
                dst_end = min(out_waveform.shape[1], dst_start + clip_waveform.shape[1])
                actual = dst_end - dst_start
                if actual <= 0:
                    continue
                out_waveform[:, dst_start:dst_end] += clip_waveform[:, :actual]
            except Exception:
                continue

        return {"waveform": out_waveform.unsqueeze(0), "sample_rate": target_sr}

    @staticmethod
    def _encode_audio_latent(audio_vae: Any, audio_out: Dict[str, Any]) -> Dict[str, Any]:
        if audio_vae is None or not isinstance(audio_out, dict) or "waveform" not in audio_out:
            return {}
        try:
            import comfy.model_management

            latent_samples = audio_vae.encode(audio_out["waveform"].movedim(1, -1))
            if not torch.is_tensor(latent_samples) or latent_samples.numel() == 0:
                return {}
            mask = torch.full(
                (1, latent_samples.shape[-2], latent_samples.shape[-1]),
                0.0,
                dtype=torch.float32,
                device=comfy.model_management.intermediate_device(),
            )
            return {
                "samples": latent_samples,
                "sample_rate": int(audio_out.get("sample_rate", getattr(audio_vae, "sample_rate", 44100))),
                "type": "audio",
                "noise_mask": mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
            }
        except Exception:
            return {}


    @staticmethod
    def _guide_data_from_visual_segments(
        visual_segments_json: Any,
        multi_output: Any,
        frame_rate: int,
        image_paths: str = "",
        image_width: int = 768,
        image_height: int = 432,
        image_resize_method: str = "crop",
        image_multiple_of: int = 32,
        img_compression: int = 0,
    ) -> Dict[str, Any]:
        raw = _safe_json_loads(str(visual_segments_json or "[]"), [])
        if isinstance(raw, dict):
            raw = raw.get("segments", [])
        guide_data: Dict[str, Any] = {
            "images": [],
            "insert_frames": [],
            "strengths": [],
            "frame_rate": int(max(1, frame_rate)),
            "labels": [],
            "reference_indices": [],
            "source": "IAMCCS_FLFReal_visual_timeline",
        }
        if not isinstance(raw, list):
            return guide_data
        has_batch = torch.is_tensor(multi_output) and multi_output.ndim == 4
        batch_size = int(multi_output.shape[0]) if has_batch else 0
        reference_paths = _cine_reference_paths_from_text(image_paths)
        loader = IAMCCS_CineReferenceBoard()
        image_order = 0
        for idx, seg in enumerate(sorted([s for s in raw if isinstance(s, dict)], key=lambda item: _safe_int(item.get("start", item.get("frame", 0)), 0))):
            seg_type = str(seg.get("type", "image") or "image").strip().lower()
            if seg_type in {"text", "audio"} or _safe_bool(seg.get("placeholder", False), False):
                continue
            image_order += 1
            if _safe_bool(seg.get("use_guide", seg.get("guide", True)), True) is False:
                continue
            ref = max(1, _safe_int(seg.get("ref", seg.get("reference_index", seg.get("image_ref", image_order))), image_order))
            fallback_strength = _safe_float(seg.get("guideStrength", seg.get("guide_strength", seg.get("strength", 1.0))), 1.0)
            strength = _wdc_image_guide_strength(seg, fallback_strength)
            if strength <= 0:
                continue
            img_tensor = None
            source = dict(seg)
            if not str(source.get("imageFile", source.get("image_file", "")) or "").strip() and ref <= len(reference_paths):
                source["imageFile"] = reference_paths[ref - 1]
            if str(source.get("imageFile", source.get("image_file", "")) or "").strip() or str(source.get("imageB64", source.get("image_b64", "")) or "").strip():
                try:
                    img_tensor = loader.load_ltx_style_image(
                        source,
                        int(image_width or 0),
                        int(image_height or 0),
                        _iamccs_cine_resize_method(image_resize_method),
                        int(image_multiple_of or 32),
                        int(img_compression or 0),
                    )
                except Exception as exc:
                    print(f"IAMCCS FilmmakerBackend warning: could not load LTX-style guide image for {source.get('label', idx + 1)}: {exc}")
            if img_tensor is None:
                if not has_batch or ref > batch_size:
                    continue
                img_tensor = multi_output[ref - 1 : ref]
            frame = max(0, _safe_int(seg.get("start", seg.get("frame", 0)), 0))
            guide_data["images"].append(img_tensor)
            guide_data["insert_frames"].append(int(frame))
            guide_data["strengths"].append(float(strength))
            guide_data["labels"].append(str(seg.get("label", f"guide_{idx + 1}")))
            guide_data["reference_indices"].append(int(ref))
            guide_data.setdefault("motion_forces", []).append(float(_safe_float(seg.get("guideStrength", seg.get("motion_force", seg.get("force", 0.0))), 0.0)))
            guide_data.setdefault("image_lock_strengths", []).append(float(strength))
            guide_data.setdefault("image_sources", []).append(str(source.get("imageFile", "")) or f"multi_output[{ref}]")
        return guide_data

    @staticmethod
    def _encode_basic(clip: Any, text: str) -> Any:
        result = comfy_nodes.CLIPTextEncode().encode(clip, str(text or ""))
        return result[0] if isinstance(result, tuple) else result

    def execute(
        self,
        model,
        clip,
        cine_linx,
        audio_vae=None,
        optional_latent=None,
        use_custom_audio=False,
        promptrelay_safety_budget_enabled=True,
        promptrelay_safety_budget=24000,
    ):
        resources = IAMCCS_CineInfo._resources(cine_linx)
        outputs = IAMCCS_CineInfo._outputs(cine_linx)
        payload = resources.get("cine_payload") if isinstance(resources.get("cine_payload"), dict) else {}

        global_prompt = str(resources.get("cine_global_prompt", outputs.get("global_prompt", payload.get("global_prompt", ""))) or "")
        local_prompts = str(resources.get("cine_local_prompts", outputs.get("local_prompts", payload.get("local_prompts", ""))) or "")
        segment_lengths = str(resources.get("cine_segment_lengths", outputs.get("segment_lengths", payload.get("segment_lengths", ""))) or "")
        flf_timeline = str(resources.get("cine_flf_timeline", outputs.get("flf_timeline", payload.get("flf_timeline", ""))) or "")
        audio_timeline_json = str(resources.get("cine_audio_timeline_json", outputs.get("audio_timeline_json", payload.get("audioSegments", "[]"))) or "[]")
        image_paths = str(resources.get("cine_image_paths", payload.get("image_paths", "")) or "")
        duration_seconds = _safe_float(resources.get("cine_duration_seconds", outputs.get("duration_seconds", payload.get("duration_seconds", 20.0))), 20.0)
        frame_rate = _safe_int(resources.get("cine_frame_rate", outputs.get("frame_rate", payload.get("frame_rate", 24))), 24)
        width = _safe_int(resources.get("cine_image_width", outputs.get("width", payload.get("image_width", 768))), 768)
        height = _safe_int(resources.get("cine_image_height", outputs.get("height", payload.get("image_height", 432))), 432)
        image_resize_method = _iamccs_cine_resize_method(resources.get("cine_image_resize_method", payload.get("image_resize_method", "crop")))
        image_multiple_of = _safe_int(resources.get("cine_image_multiple_of", payload.get("image_multiple_of", 32)), 32)
        img_compression = _safe_int(resources.get("cine_img_compression", payload.get("img_compression", 0)), 0)
        max_frames = _safe_int(resources.get("cine_max_frames", outputs.get("max_frames", payload.get("max_frames", 0))), 0)
        if max_frames <= 0:
            max_frames = _round_ltx_frames(int(round(duration_seconds * max(1, frame_rate))), str(payload.get("ltx_round_mode", "up_8n_plus_1")))
        epsilon = _safe_float(resources.get("cine_promptrelay_epsilon", outputs.get("promptrelay_epsilon", payload.get("promptrelay_epsilon", 0.65))), 0.65)

        latent = optional_latent if isinstance(optional_latent, dict) else self._empty_latent(width, height, max_frames)
        promptrelay_requested = _safe_bool(
            payload.get(
                "filmmaker_promptrelay_enabled",
                resources.get(
                    "cine_filmmaker_promptrelay_enabled",
                    payload.get(
                        "promptrelay_enabled",
                        resources.get("cine_promptrelay_enabled", outputs.get("promptrelay_enabled", False)),
                    ),
                ),
            ),
            False,
        )
        safety_budget_enabled = _safe_bool(
            payload.get("promptrelay_safety_budget_enabled", resources.get("cine_promptrelay_safety_budget_enabled", promptrelay_safety_budget_enabled)),
            True,
        )
        safety_budget = max(0, _safe_int(
            payload.get("promptrelay_safety_budget", resources.get("cine_promptrelay_safety_budget", promptrelay_safety_budget)),
            24000,
        ))
        promptrelay_enabled = bool(promptrelay_requested and local_prompts.strip() and segment_lengths.strip())
        local_parts = [part.strip() for part in str(local_prompts or "").split("|") if part.strip()]
        length_parts = [part for part in re.split(r"[,;\s]+", str(segment_lengths or "")) if part.strip()]
        local_hash = hashlib.sha1(str(local_prompts or "").encode("utf-8", errors="ignore")).hexdigest()[:12]
        relay_prompt_log = [
            {
                "index": idx,
                "segment_length": length_parts[idx] if idx < len(length_parts) else "<missing>",
                "prompt": prompt,
            }
            for idx, prompt in enumerate(local_parts)
        ]
        print(
            "[IAMCCS FilmmakerBackend] "
            f"PromptRelay requested={bool(promptrelay_requested)} "
            f"enabled_candidate={bool(promptrelay_enabled)} "
            f"locals={len(local_parts)} segments={len(length_parts)} "
            f"epsilon={float(epsilon):.3f} max_frames={int(max_frames)} "
            f"local_hash={local_hash}"
        )
        relay_error = ""
        if promptrelay_enabled:
            try:
                lengths = [_safe_int(part, 0) for part in re.split(r"[,;\s]+", str(segment_lengths or "")) if part.strip()]
                prompt_budget = sum(max(0, length) for length in lengths) * max(1, len(str(local_prompts or "").split()))
                if safety_budget_enabled and safety_budget > 0 and prompt_budget > safety_budget:
                    raise RuntimeError(f"PromptRelay disabled by optional safety budget: estimated matrix load {prompt_budget} > {safety_budget}")
                promptrelay_nodes = _load_original_promptrelay_module()
                print(
                    "[IAMCCS FilmmakerBackend] "
                    f"PROMPT_RELAY_LOCAL_PROMPTS_USED source=cine_linx "
                    f"count={len(local_parts)} segments={len(length_parts)} "
                    f"global_hash={hashlib.sha1(str(global_prompt or '').encode('utf-8', errors='ignore')).hexdigest()[:12]} "
                    f"local_hash={local_hash}"
                )
                for item in relay_prompt_log[:50]:
                    compact = str(item["prompt"]).replace("\n", " ")
                    if len(compact) > 360:
                        compact = compact[:357] + "..."
                    print(
                        "[IAMCCS FilmmakerBackend] "
                        f"relay[{int(item['index']):02d}] "
                        f"length={item['segment_length']} "
                        f"prompt={compact!r}"
                    )
                if len(relay_prompt_log) > 50:
                    print(f"[IAMCCS FilmmakerBackend] relay prompt log truncated: {len(relay_prompt_log) - 50} more prompts.")
                patched_model, positive = promptrelay_nodes._encode_relay(
                    model,
                    clip,
                    latent,
                    global_prompt,
                    local_prompts,
                    segment_lengths,
                    float(epsilon),
                )
                print(
                    "[IAMCCS FilmmakerBackend] "
                    f"PromptRelay APPLIED locals={len(local_parts)} segments={len(length_parts)} "
                    f"budget_estimate={prompt_budget}"
                )
            except Exception as exc:
                relay_error = str(exc)
                patched_model = model
                positive = self._encode_basic(clip, global_prompt)
                print(f"[IAMCCS FilmmakerBackend] PromptRelay ERROR, falling back to global encode: {relay_error}")
        else:
            patched_model = model
            positive = self._encode_basic(clip, global_prompt)
            reason = "not_requested" if not promptrelay_requested else "missing_local_prompts_or_segment_lengths"
            print(f"[IAMCCS FilmmakerBackend] PromptRelay BYPASS reason={reason}")

        multi_output = resources.get("cine_multi_input")
        if not torch.is_tensor(multi_output):
            multi_output = torch.zeros((1, max(64, int(height)), max(64, int(width)), 3))
        default_force = _safe_float(resources.get("cine_default_force", payload.get("default_force", 0.25)), 0.25)
        visual_segments_json = resources.get("cine_visual_segments_json", "")
        guide_data = self._guide_data_from_visual_segments(
            visual_segments_json,
            multi_output,
            int(frame_rate),
            image_paths=image_paths,
            image_width=int(width),
            image_height=int(height),
            image_resize_method=image_resize_method,
            image_multiple_of=int(image_multiple_of),
            img_compression=int(img_compression),
        )
        guide_data_source = "flfreal_visual_timeline_1to1" if guide_data.get("images") else "cineinfo_v2_fallback"
        guide_plan_json, plan_report = IAMCCS_CineInfoV2._build_guide_plan(
            flf_timeline=flf_timeline,
            image_paths=image_paths,
            multi_output=multi_output,
            duration_seconds=float(duration_seconds),
            frame_rate=int(frame_rate),
            default_force=float(default_force),
        )
        if not guide_data.get("images"):
            guide_data = IAMCCS_CineInfoV2._guide_data_from_plan(guide_plan_json, multi_output)
        has_timeline_audio = self._has_custom_audio(audio_timeline_json)
        custom_audio_requested = bool(_safe_bool(payload.get("use_custom_audio", resources.get("cine_use_custom_audio", use_custom_audio)), False) or has_timeline_audio)
        audio_out = self._build_combined_audio(audio_timeline_json, int(max_frames), float(frame_rate))
        audio_latent = self._encode_audio_latent(audio_vae, audio_out) if custom_audio_requested and has_timeline_audio else {}
        if not audio_latent:
            audio_latent = self._empty_audio_latent(audio_vae, int(max_frames), float(frame_rate))
        report = _json_report({
            "node": "IAMCCS_CineFilmmakerBackend",
            "mode": "shotboard_v3_runtime_backend",
            "promptrelay_requested": bool(promptrelay_requested),
            "promptrelay_enabled": bool(promptrelay_enabled and not relay_error),
            "promptrelay_error": relay_error,
            "promptrelay_safety_budget_enabled": bool(safety_budget_enabled),
            "promptrelay_safety_budget": int(safety_budget),
            "local_hash": local_hash,
            "local_prompts_used": relay_prompt_log,
            "duration_seconds": float(duration_seconds),
            "frame_rate": int(frame_rate),
            "max_frames": int(max_frames),
            "width": int(width),
            "height": int(height),
            "guide_data_count": len(guide_data.get("images", [])),
            "guide_data_source": guide_data_source,
            "image_loading": {
                "mode": "ltx_director_style_internal_load_resize_compress",
                "resize_method": image_resize_method,
                "multiple_of": int(image_multiple_of),
                "img_compression": int(img_compression),
                "sources": guide_data.get("image_sources", []),
            },
            "flfreal_mode": str(payload.get("flfreal_mode", "iamccs_enhanced") if isinstance(payload, dict) else "iamccs_enhanced"),
            "director_style_fields": {
                "local_prompts": bool(str(payload.get("director_local_prompts", "") if isinstance(payload, dict) else "").strip()),
                "segment_lengths": bool(str(payload.get("director_segment_lengths", "") if isinstance(payload, dict) else "").strip()),
                "guide_strength": str(payload.get("director_guide_strength", "") if isinstance(payload, dict) else ""),
                "audio_data": bool(str(payload.get("director_audio_data", "") if isinstance(payload, dict) else "").strip()),
                "use_custom_audio": bool(custom_audio_requested),
            },
            "flfreal_compile": payload.get("flfreal_compile", {}) if isinstance(payload, dict) else {},
            "audio_segments": len(_safe_json_loads(audio_timeline_json, [])) if isinstance(_safe_json_loads(audio_timeline_json, []), list) else len(_safe_json_loads(audio_timeline_json, {}).get("audioSegments", [])) if isinstance(_safe_json_loads(audio_timeline_json, {}), dict) else 0,
            "custom_audio_requested": bool(custom_audio_requested),
            "custom_audio_encoded": bool(custom_audio_requested and has_timeline_audio),
            "audio_latent_valid": bool(isinstance(audio_latent, dict) and torch.is_tensor(audio_latent.get("samples"))),
            "guide_plan_report": _safe_json_loads(plan_report, {}),
            "truth": "This node consumes the FLFreal visual timeline contract from Shotboard Planner V3 cine_linx.",
        })
        return patched_model, positive, latent, audio_latent, guide_data, float(frame_rate), audio_out, report


class IAMCCS_CineShotboardBackendPro:
    """Internal prompt backend for Shotboard Pro / ProV2 cine_linx."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "cine_linx": (SUPERNODE_LINX_TYPE,),
            },
            "optional": {
                "optional_latent": ("LATENT",),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "relay_options": ("RELAY_OPTIONS",),
                "promptrelay_safety_budget_enabled": ("BOOLEAN", {"default": True}),
                "promptrelay_safety_budget": ("INT", {"default": 24000, "min": 1000, "max": 5000000, "step": 1000}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "promptrelay_enabled", "report")
    FUNCTION = "execute"
    CATEGORY = "IAMCCS/Cine/02 Single Generation"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    @staticmethod
    def _encode_basic(clip: Any, text: str) -> Any:
        result = comfy_nodes.CLIPTextEncode().encode(clip, str(text or ""))
        return result[0] if isinstance(result, tuple) else result

    @staticmethod
    def _latent_shape(latent: Any) -> List[int]:
        try:
            samples = latent.get("samples") if isinstance(latent, dict) else None
            if torch.is_tensor(samples):
                return [int(v) for v in samples.shape]
        except Exception:
            pass
        return []

    def execute(
        self,
        model,
        clip,
        cine_linx,
        optional_latent=None,
        negative_prompt="",
        relay_options=None,
        promptrelay_safety_budget_enabled=True,
        promptrelay_safety_budget=24000,
    ):
        resources = IAMCCS_CineInfo._resources(cine_linx)
        outputs = IAMCCS_CineInfo._outputs(cine_linx)
        payload = resources.get("cine_payload") if isinstance(resources.get("cine_payload"), dict) else {}

        global_prompt = str(resources.get("cine_global_prompt", outputs.get("global_prompt", payload.get("global_prompt", ""))) or "")
        local_prompts = str(resources.get("cine_local_prompts", outputs.get("local_prompts", payload.get("local_prompts", ""))) or "")
        segment_lengths = str(resources.get("cine_segment_lengths", outputs.get("segment_lengths", payload.get("segment_lengths", ""))) or "")
        epsilon = _safe_float(resources.get("cine_promptrelay_epsilon", outputs.get("promptrelay_epsilon", payload.get("promptrelay_epsilon", 0.65))), 0.65)
        duration_seconds = _safe_float(resources.get("cine_duration_seconds", outputs.get("duration_seconds", payload.get("duration_seconds", 8.0))), 8.0)
        frame_rate = _safe_int(resources.get("cine_frame_rate", outputs.get("frame_rate", payload.get("frame_rate", 24))), 24)
        width = _safe_int(resources.get("cine_image_width", outputs.get("width", payload.get("image_width", 768))), 768)
        height = _safe_int(resources.get("cine_image_height", outputs.get("height", payload.get("image_height", 432))), 432)
        max_frames = _safe_int(resources.get("cine_max_frames", outputs.get("max_frames", payload.get("max_frames", 0))), 0)
        if max_frames <= 0:
            max_frames = _round_ltx_frames(
                int(round(float(duration_seconds) * max(1, int(frame_rate)))),
                str(payload.get("ltx_round_mode", "up_8n_plus_1")),
            )

        latent = optional_latent if isinstance(optional_latent, dict) else IAMCCS_CineFilmmakerBackend._empty_latent(width, height, max_frames)
        negative = self._encode_basic(clip, str(negative_prompt or ""))

        locals_list = [part.strip() for part in str(local_prompts or "").split("|") if part.strip()]
        length_parts = [part for part in re.split(r"[,;\s]+", str(segment_lengths or "")) if part.strip()]
        relay_requested = bool(locals_list and length_parts)
        safety_budget_enabled = _safe_bool(
            payload.get("promptrelay_safety_budget_enabled", resources.get("cine_promptrelay_safety_budget_enabled", promptrelay_safety_budget_enabled)),
            True,
        )
        safety_budget = max(0, _safe_int(
            payload.get("promptrelay_safety_budget", resources.get("cine_promptrelay_safety_budget", promptrelay_safety_budget)),
            24000,
        ))

        promptrelay_enabled = False
        relay_error = ""
        patched_model = model
        if relay_requested:
            try:
                lengths = [_safe_int(part, 0) for part in length_parts]
                prompt_budget = sum(max(0, length) for length in lengths) * max(1, len(locals_list))
                if safety_budget_enabled and safety_budget > 0 and prompt_budget > safety_budget:
                    raise RuntimeError(f"PromptRelay disabled by optional safety budget: estimated matrix load {prompt_budget} > {safety_budget}")
                promptrelay_nodes = _load_original_promptrelay_module()
                patched_model, positive = promptrelay_nodes._encode_relay(
                    model,
                    clip,
                    latent,
                    global_prompt,
                    local_prompts,
                    segment_lengths,
                    float(epsilon),
                    relay_options,
                )
                promptrelay_enabled = True
            except Exception as exc:
                relay_error = str(exc)
                patched_model = model
                positive = self._encode_basic(clip, global_prompt)
        else:
            positive = self._encode_basic(clip, global_prompt)

        global_hash = hashlib.sha1(str(global_prompt or "").encode("utf-8", errors="ignore")).hexdigest()[:12]
        local_hash = hashlib.sha1(str(local_prompts or "").encode("utf-8", errors="ignore")).hexdigest()[:12]
        report = _json_report({
            "node": "IAMCCS_CineShotboardBackendPro",
            "mode": "INTERNAL_PROMPT_BACKEND",
            "promptrelay_requested": bool(relay_requested),
            "promptrelay_enabled": bool(promptrelay_enabled),
            "promptrelay_error": relay_error,
            "local_prompt_count": len(locals_list),
            "segment_count": len(length_parts),
            "segment_lengths": segment_lengths,
            "epsilon": float(epsilon),
            "latent_shape": self._latent_shape(latent),
            "duration_seconds": float(duration_seconds),
            "frame_rate": int(frame_rate),
            "max_frames": int(max_frames),
            "width": int(width),
            "height": int(height),
            "global_hash": global_hash,
            "local_hash": local_hash,
            "promptrelay_safety_budget_enabled": bool(safety_budget_enabled),
            "promptrelay_safety_budget": int(safety_budget),
            "truth": (
                "Shotboard Pro/ProV2 encoding is handled inside this IAMCCS backend. "
                "Relay OFF returns normal global CLIP conditioning. Relay ON calls the original PromptRelay _encode_relay internally. "
                "No external IAMCCS_CinePromptRelaySafeEncode node is required in new workflows."
            ),
        })
        _cine_debug(
            "[IAMCCS CineShotboardBackendPro] "
            f"mode={'PROMPT_RELAY_ORIGINAL_1TO1' if promptrelay_enabled else 'BASIC_TEXT_GLOBAL_ONLY'} "
            f"local_prompts={len(locals_list)} segments={len(length_parts)} "
            f"global_hash={global_hash} local_hash={local_hash} latent_shape={self._latent_shape(latent)}"
        )
        return patched_model, positive, negative, latent, bool(promptrelay_enabled), report


class IAMCCS_CineFilmmakerGuide(IAMCCS_CineFLFProductor):
    """Guide applicator for the Filmmaker timeline GUIDE_DATA contract."""

    CATEGORY = "IAMCCS/Cine/02 Single Generation"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "guide_data": ("GUIDE_DATA",),
                "strength_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "tail_safety_frames": ("INT", {"default": 0, "min": 0, "max": 240, "step": 1}),
            },
        }

    def execute(self, positive, negative, vae, latent, guide_data, strength_scale, tail_safety_frames):
        positive, negative, out_latent, guide_data_report = self._execute_guide_data(
            positive,
            negative,
            vae,
            latent,
            guide_data if isinstance(guide_data, dict) else {},
            float(strength_scale),
            int(tail_safety_frames),
        )
        report = _json_report({
            "node": "IAMCCS_CineFilmmakerGuide",
            "mode": "pure_guide_data_applicator",
            "guide_count": len(guide_data.get("images", [])) if isinstance(guide_data, dict) else 0,
            "applied_count": len(guide_data_report.get("applied_guides", [])),
            "skipped_count": len(guide_data_report.get("skipped_guides", [])),
            **guide_data_report,
            "truth": "This guide node is driven only by GUIDE_DATA from the Filmmaker backend.",
        })
        return positive, negative, out_latent, report


class IAMCCS_CineFilmmakerGuide1to1:
    """IAMCCS guide applicator matching the LTX Director Guide behavior 1:1.

    This node intentionally does not call or wrap the WhatDreamsCost node. It
    implements the same guide-data contract directly inside IAMCCS so subgraph
    backends can stay fully IAMCCS-owned while preserving LTX Director guide
    semantics.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "guide_data": ("GUIDE_DATA",),
                "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01}),
                "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "bislerp"], {"default": "bicubic"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "IAMCCS/Cine/02 Single Generation"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    @classmethod
    def execute(cls, positive, negative, vae, latent, guide_data, scale_by=1.0, upscale_method="bicubic"):
        scale_factors = vae.downscale_index_formula

        latent_image = latent["samples"].clone()

        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"].clone()
        else:
            batch, _, latent_frames, _, _ = latent_image.shape
            noise_mask = torch.ones(
                (batch, 1, latent_frames, 1, 1),
                dtype=torch.float32,
                device=latent_image.device,
            )

        if scale_by != 1.0:
            batch, channels, frames, height, width = latent_image.shape
            scaled_width = round(width * scale_by)
            scaled_height = round(height * scale_by)

            latent_4d = latent_image.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
            latent_resized_4d = comfy.utils.common_upscale(
                latent_4d,
                scaled_width,
                scaled_height,
                upscale_method,
                "disabled",
            )
            latent_image = latent_resized_4d.reshape(
                batch,
                frames,
                channels,
                scaled_height,
                scaled_width,
            ).permute(0, 2, 1, 3, 4)

            if noise_mask.shape[-1] > 1 or noise_mask.shape[-2] > 1:
                mask_4d = noise_mask.permute(0, 2, 1, 3, 4).reshape(batch * frames, 1, height, width)
                mask_resized_4d = comfy.utils.common_upscale(
                    mask_4d,
                    scaled_width,
                    scaled_height,
                    upscale_method,
                    "disabled",
                )
                noise_mask = mask_resized_4d.reshape(
                    batch,
                    frames,
                    1,
                    scaled_height,
                    scaled_width,
                ).permute(0, 2, 1, 3, 4)

        _, _, latent_length, latent_height, latent_width = latent_image.shape

        images = guide_data.get("images", []) if isinstance(guide_data, dict) else []
        insert_frames = guide_data.get("insert_frames", []) if isinstance(guide_data, dict) else []
        strengths = guide_data.get("strengths", []) if isinstance(guide_data, dict) else []

        for idx, img_tensor in enumerate(images):
            frame = insert_frames[idx] if idx < len(insert_frames) else 0
            strength = strengths[idx] if idx < len(strengths) else 1.0

            image_1, encoded = LTXVAddGuide.encode(vae, latent_width, latent_height, img_tensor, scale_factors)
            frame_idx, latent_idx = LTXVAddGuide.get_latent_index(
                positive,
                latent_length,
                len(image_1),
                frame,
                scale_factors,
            )

            assert latent_idx + encoded.shape[2] <= latent_length, (
                f"Guide image {idx + 1}: conditioning frames exceed the length of the latent sequence."
            )

            positive, negative, latent_image, noise_mask = LTXVAddGuide.append_keyframe(
                positive,
                negative,
                frame_idx,
                latent_image,
                noise_mask,
                encoded,
                strength,
                scale_factors,
            )

        return positive, negative, {"samples": latent_image, "noise_mask": noise_mask}


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

        cine_local_prompts = str(
            resources.get("cine_local_prompts",
            outputs.get("local_prompts",
            payload.get("local_prompts", "")))
            or ""
        )
        cine_segment_lengths = str(
            resources.get("cine_segment_lengths",
            outputs.get("segment_lengths",
            payload.get("segment_lengths", "")))
            or ""
        )
        cine_global_prompt = str(
            resources.get("cine_global_prompt",
            outputs.get("global_prompt",
            payload.get("global_prompt", "")))
            or ""
        )
        cine_epsilon = resources.get("cine_promptrelay_epsilon",
            outputs.get("promptrelay_epsilon",
            payload.get("promptrelay_epsilon", None)))

        if cine_local_prompts.strip():
            source = "cine_linx"
            resolved_local_prompts = cine_local_prompts
            resolved_segment_lengths = cine_segment_lengths
            resolved_global_prompt = cine_global_prompt or str(global_prompt or "")
            resolved_epsilon = _safe_float(cine_epsilon if cine_epsilon is not None else epsilon, 1e-3)
        else:
            source = "explicit_inputs" if local_prompts is not None else "cine_linx"
            resolved_local_prompts = str(local_prompts if local_prompts is not None else cine_local_prompts or "")
            resolved_segment_lengths = str(segment_lengths if segment_lengths is not None else cine_segment_lengths or "")
            resolved_global_prompt = str(global_prompt if global_prompt is not None else cine_global_prompt or "")
            resolved_epsilon = _safe_float(epsilon if epsilon is not None else cine_epsilon, 1e-3)
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
            "local_prompts_used": relay_prompt_log,
            "truth": (
                "Relay active. Called ComfyUI-PromptRelay _encode_relay 1:1. "
                "By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS"
            ),
        })
        print(
            "[IAMCCS CineRelayOrBypass] "
            f"PROMPT_RELAY_LOCAL_PROMPTS_USED source={prompt_source} "
            f"count={len(locals_list)} segments={len(length_parts)} "
            f"global_hash={global_hash} local_hash={local_hash}"
        )
        for item in relay_prompt_log[:50]:
            compact = str(item["prompt"]).replace("\n", " ")
            if len(compact) > 360:
                compact = compact[:357] + "..."
            print(
                "[IAMCCS CineRelayOrBypass] "
                f"relay[{int(item['index']):02d}] "
                f"length={item['segment_length']} "
                f"prompt={compact!r}"
            )
        if len(relay_prompt_log) > 50:
            print(f"[IAMCCS CineRelayOrBypass] relay prompt log truncated: {len(relay_prompt_log) - 50} more prompts.")
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


class IAMCCS_CinePromptArchitect:
    """Creative prompt architect for LTX 2.3 / PromptRelay / Shotboard V3."""

    DEFAULT_BEATS = json.dumps({
        "beats": [
            {
                "label": "opening_contract",
                "duration": 3.0,
                "image_role": "opening anchor",
                "action": "the camera establishes the subject and begins one continuous physical movement",
                "bridge": "carry the same camera movement into the next beat without a hard cut",
            },
            {
                "label": "development",
                "duration": 3.0,
                "image_role": "motion checkpoint",
                "action": "the movement deepens; environment details react physically as the camera advances",
                "bridge": "arrive toward the next visual target only near the end of the beat",
            },
            {
                "label": "arrival",
                "duration": 3.0,
                "image_role": "target keyframe",
                "action": "the shot arrives at the strongest visual target with coherent identity and space",
                "bridge": "",
            },
        ]
    }, indent=2)

    TEMPLATE_RULES = {
        "continuous_dolly": {
            "camera": "one continuous controlled dolly movement",
            "continuity": "no hard cuts, coherent spatial continuity, stable subject identity",
        },
        "future_keyframe": {
            "camera": "movement toward a future visual target",
            "continuity": "arrive at the target framing only near the end, no early reveal",
        },
        "image_text_image": {
            "camera": "visual anchor, timed prompt beat, visual target",
            "continuity": "preserve identity and scene geometry between image guides",
        },
        "dialogue_lipsync": {
            "camera": "performance-aware camera framing",
            "continuity": "natural lip sync, clear diction, expressive face, stable identity",
        },
        "reveal": {
            "camera": "controlled reveal with delayed arrival",
            "continuity": "hide the reveal until the correct timed beat, keep the camera physically motivated",
        },
        "environmental_transition": {
            "camera": "continuous camera movement through a changing environment",
            "continuity": "environment changes physically across time without editorial cuts",
        },
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": ([
                    "continuous_dolly",
                    "future_keyframe",
                    "image_text_image",
                    "dialogue_lipsync",
                    "reveal",
                    "environmental_transition",
                ], {"default": "continuous_dolly"}),
                "subject_identity": ("STRING", {"default": "the same main subject, stable identity", "multiline": True}),
                "environment": ("STRING", {"default": "a cinematic physical environment with readable depth", "multiline": True}),
                "lighting_weather": ("STRING", {"default": "natural realistic lighting, atmospheric depth", "multiline": True}),
                "visual_style": ("STRING", {"default": "cinematic realism, grounded camera physics, detailed texture", "multiline": True}),
                "camera_language": ("STRING", {"default": "slow controlled camera movement", "multiline": True}),
                "continuity_rules": ("STRING", {"default": "no hard cuts, coherent spatial continuity, stable subject identity", "multiline": True}),
                "shot_goal": ("STRING", {"default": "direct one continuous cinematic shot with clear temporal progression", "multiline": True}),
                "movement_path": ("STRING", {"default": "the camera moves forward through the scene with physical parallax", "multiline": True}),
                "target_reveal": ("STRING", {"default": "", "multiline": True}),
                "performance_or_emotion": ("STRING", {"default": "", "multiline": True}),
                "audio_or_dialogue": ("STRING", {"default": "", "multiline": True}),
                "avoid": ("STRING", {"default": "hard cuts, identity drift, duplicated subjects, frozen motion, early reveal", "multiline": True}),
                "duration_seconds": ("FLOAT", {"default": 9.0, "min": 0.1, "max": 600.0, "step": 0.1}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "beat_count": ("INT", {"default": 3, "min": 1, "max": 8, "step": 1}),
                "beat_data": ("STRING", {
                    "default": cls.DEFAULT_BEATS,
                    "multiline": True,
                    "tooltip": "Edited by the CinePrompt Architect UI. JSON with beats: label, duration, image_role, action, bridge.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("global_prompt", "local_prompts", "segment_lengths", "shotboard_timeline_json", "board_json", "markdown_preview", "report")
    FUNCTION = "build"
    CATEGORY = "IAMCCS/Cine/01 Prompting"

    @staticmethod
    def _clean_parts(parts: List[Any]) -> List[str]:
        out: List[str] = []
        seen = set()
        for part in parts:
            text = re.sub(r"\s+", " ", str(part or "").strip(" ,.;\n\t"))
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
        return out

    @classmethod
    def _join_sentence(cls, parts: List[Any]) -> str:
        clean = cls._clean_parts(parts)
        if not clean:
            return ""
        text = ", ".join(clean)
        return text[:1].lower() + text[1:] if text else ""

    @classmethod
    def _parse_beats(cls, beat_data: Any, beat_count: int, duration_seconds: float) -> List[Dict[str, Any]]:
        data = _safe_json_loads(str(beat_data or ""), {})
        raw = data if isinstance(data, list) else data.get("beats", []) if isinstance(data, dict) else []
        if not isinstance(raw, list):
            raw = []
        beat_count = max(1, min(8, int(beat_count or 1)))
        fallback_duration = max(0.1, float(duration_seconds) / float(beat_count))
        beats: List[Dict[str, Any]] = []
        for index in range(beat_count):
            item = raw[index] if index < len(raw) and isinstance(raw[index], dict) else {}
            duration = max(0.1, _safe_float(item.get("duration", item.get("seconds", fallback_duration)), fallback_duration))
            label = _normalise_label(str(item.get("label", item.get("name", "")) or ""), f"beat_{index + 1}")
            action = str(item.get("action", item.get("local_prompt", item.get("prompt", ""))) or "").strip()
            if not action:
                action = "the camera continues one clear cinematic action through this timed beat"
            bridge = str(item.get("bridge", item.get("action_to_next", item.get("step_transition_prompt", ""))) or "").strip()
            beats.append({
                "label": label,
                "duration": duration,
                "image_role": str(item.get("image_role", item.get("image_hint", "")) or "").strip(),
                "action": action,
                "bridge": bridge,
            })
        return beats

    @classmethod
    def _global_prompt(cls, template: str, subject_identity: str, environment: str, lighting_weather: str, visual_style: str, camera_language: str, continuity_rules: str, shot_goal: str, movement_path: str, target_reveal: str, performance_or_emotion: str, audio_or_dialogue: str) -> str:
        rules = cls.TEMPLATE_RULES.get(str(template), cls.TEMPLATE_RULES["continuous_dolly"])
        return cls._join_sentence([
            shot_goal,
            subject_identity,
            environment,
            lighting_weather,
            visual_style,
            camera_language or rules.get("camera"),
            movement_path,
            target_reveal,
            performance_or_emotion,
            audio_or_dialogue,
            continuity_rules or rules.get("continuity"),
        ])

    @classmethod
    def _local_prompt(cls, template: str, beat: Dict[str, Any], index: int, total: int, target_reveal: str, audio_or_dialogue: str) -> str:
        action = str(beat.get("action", "") or "").strip()
        bridge = str(beat.get("bridge", "") or "").strip()
        parts: List[str] = [action]
        if bridge and index < total - 1:
            parts.append(bridge)
        if str(template) in {"future_keyframe", "reveal"} and index < total - 1:
            parts.append("arrive at the next visual target only near the end of this beat")
        if str(template) == "dialogue_lipsync" and audio_or_dialogue:
            parts.append("lips sync naturally to the dialogue, diction is clear, facial expression remains alive and precise")
        if target_reveal and index == total - 1:
            parts.append(target_reveal)
        return cls._join_sentence(parts)

    @classmethod
    def _build_rows(cls, beats: List[Dict[str, Any]], local_prompts: List[str]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        cursor = 0.0
        for index, beat in enumerate(beats):
            duration = max(0.1, float(beat.get("duration", 0.1)))
            prompt = local_prompts[index] if index < len(local_prompts) else str(beat.get("action", ""))
            bridge = str(beat.get("bridge", "") or "").strip()
            rows.append({
                "second": round(cursor, 3),
                "ref": index + 1,
                "force": 0.0,
                "image_lock_strength": 0.0,
                "use_guide": False,
                "use_prompt": bool(prompt),
                "label": str(beat.get("label", f"beat_{index + 1}")),
                "camera": "prompt relay text",
                "transition": "prompt_relay_text",
                "note": str(beat.get("image_role", "")),
                "relay_prompt": prompt,
                "camera_relay_mode": "off",
                "transition_relay_mode": "off",
                "relay_addon_position": "after",
                "relay_modifier_text": "",
                "step_transition_enabled": bool(bridge and index < len(beats) - 1),
                "step_transition_type": "action_beat" if bridge and index < len(beats) - 1 else "off",
                "step_transition_prompt": bridge if index < len(beats) - 1 else "",
                "step_transition_easing": "ease_in_out",
                "step_transition_force_curve": "balanced",
                "step_transition_duration": duration if bridge and index < len(beats) - 1 else 0.0,
                "step_transition_arrival": "auto",
                "step_transition_auto_fit": True,
            })
            cursor += duration
        return rows

    @classmethod
    def _markdown(cls, global_prompt: str, local_prompts: List[str], segment_lengths: List[int], beats: List[Dict[str, Any]], avoid: str) -> str:
        lines = [
            "# CinePrompt Architect Preview",
            "",
            "## Global Prompt",
            "",
            "```text",
            global_prompt,
            "```",
            "",
            "## PromptRelay Beats",
            "",
            "| Beat | Duration | Frames | Image role | Local prompt |",
            "|---:|---:|---:|---|---|",
        ]
        for index, beat in enumerate(beats):
            frames = segment_lengths[index] if index < len(segment_lengths) else 0
            prompt = local_prompts[index] if index < len(local_prompts) else ""
            lines.append(
                f"| {index + 1} | {float(beat.get('duration', 0.0)):.2f}s | {frames} | "
                f"{str(beat.get('image_role', '') or '').replace('|', '/')} | {prompt.replace('|', '/')} |"
            )
        lines.extend([
            "",
            "## Local Prompts",
            "",
            "```text",
            " | ".join(local_prompts),
            "```",
            "",
            "## Segment Lengths",
            "",
            "```text",
            ",".join(str(int(x)) for x in segment_lengths),
            "```",
        ])
        if str(avoid or "").strip():
            lines.extend(["", "## Avoid", "", str(avoid).strip()])
        return "\n".join(lines)

    def build(self, template, subject_identity, environment, lighting_weather, visual_style, camera_language, continuity_rules, shot_goal, movement_path, target_reveal, performance_or_emotion, audio_or_dialogue, avoid, duration_seconds, frame_rate, beat_count, beat_data):
        duration = max(0.1, float(duration_seconds))
        fps = max(1.0, float(frame_rate))
        beats = self._parse_beats(beat_data, int(beat_count), duration)
        total_beat_duration = sum(max(0.1, float(beat.get("duration", 0.1))) for beat in beats) or duration
        scale = duration / total_beat_duration
        segment_lengths = [max(1, int(round(max(0.1, float(beat.get("duration", 0.1))) * scale * fps))) for beat in beats]
        target_frames = max(1, int(round(duration * fps)))
        if segment_lengths:
            segment_lengths[-1] = max(1, segment_lengths[-1] + target_frames - sum(segment_lengths))

        global_prompt = self._global_prompt(str(template), subject_identity, environment, lighting_weather, visual_style, camera_language, continuity_rules, shot_goal, movement_path, target_reveal, performance_or_emotion, audio_or_dialogue)
        local_prompts = [self._local_prompt(str(template), beat, index, len(beats), str(target_reveal or ""), str(audio_or_dialogue or "")) for index, beat in enumerate(beats)]
        rows = self._build_rows(beats, local_prompts)
        timeline = {"rows": rows}
        board = {
            "metadata": {
                "schema": "iamccs.cine.prompt_architect.board",
                "schema_version": 1,
                "node_type": "IAMCCS_CinePromptArchitect",
                "image_storage": "manual_after_import",
                "notes": "Creative prompt/relay board only. Add images, resolution and backend settings inside Shotboard Planner V3/V2.",
            },
            "global_prompt": global_prompt,
            "prompt": global_prompt,
            "timeline_data": json.dumps(timeline, ensure_ascii=False, indent=2),
            "rows": rows,
            "duration_seconds": duration,
            "frame_rate": fps,
            "image_paths": "",
            "images": [],
        }
        markdown = self._markdown(global_prompt, local_prompts, segment_lengths, beats, str(avoid or ""))
        report = _json_report({
            "node": "IAMCCS_CinePromptArchitect",
            "template": str(template),
            "beat_count": len(beats),
            "duration_seconds": duration,
            "frame_rate": fps,
            "local_prompt_count": len([p for p in local_prompts if p.strip()]),
            "segment_lengths": segment_lengths,
            "truth": "Creative prompt architecture node: outputs global/local prompts and Shotboard-importable timing only. Technical generation settings stay in Shotboard/Backend.",
        })
        return (
            global_prompt,
            " | ".join(local_prompts),
            ",".join(str(int(x)) for x in segment_lengths),
            json.dumps(timeline, ensure_ascii=False, indent=2),
            json.dumps(board, ensure_ascii=False, indent=2),
            markdown,
            report,
        )


class IAMCCS_BoardMaker:
    """Planner assistant that exports Shotboard-importable board JSON without images."""

    DEFAULT_ROWS = json.dumps({
        "rows": [
            {
                "label": "box_1",
                "duration": 3.0,
                "local_prompt": "opening beat, establish the subject and camera direction",
                "bridge": "camera continues into the next beat through one connected movement",
                "camera": "slow push-in",
                "transition": "continuous_motion",
                "force": 0.28,
                "use_guide": True,
                "use_prompt": True,
            },
            {
                "label": "box_2",
                "duration": 3.0,
                "local_prompt": "second beat, deepen the action and maintain visual continuity",
                "bridge": "movement carries the viewer toward the final beat",
                "camera": "continuous dolly-in",
                "transition": "continuous_motion",
                "force": 0.24,
                "use_guide": True,
                "use_prompt": True,
            },
            {
                "label": "box_3",
                "duration": 3.0,
                "local_prompt": "final beat, arrive at the strongest visual target",
                "bridge": "",
                "camera": "macro push-in",
                "transition": "continuous_motion",
                "force": 0.32,
                "use_guide": True,
                "use_prompt": True,
            },
        ]
    }, indent=2)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "global_prompt": ("STRING", {
                    "default": "One continuous cinematic shot with coherent camera movement, stable identity, physical parallax and connected visual progression.",
                    "multiline": True,
                }),
                "duration_seconds": ("FLOAT", {"default": 9.0, "min": 0.1, "max": 600.0, "step": 0.1}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "image_width": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 32}),
                "image_height": ("INT", {"default": 432, "min": 64, "max": 8192, "step": 32}),
                "default_force": ("FLOAT", {"default": 0.28, "min": 0.0, "max": 1.0, "step": 0.01}),
                "guide_policy": (["every_checked_row", "first_last", "all", "none"], {"default": "every_checked_row"}),
                "board_name": ("STRING", {"default": "iamccs_boardmaker_board"}),
                "board_data": ("STRING", {
                    "default": cls.DEFAULT_ROWS,
                    "multiline": True,
                    "tooltip": "Edited by the IAMCCS_BoardMaker UI. JSON with rows: duration, local_prompt, bridge, camera, transition, force.",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("board_json", "timeline_data", "row_count", "duration_seconds", "report")
    FUNCTION = "make_board"
    CATEGORY = "IAMCCS/Cinema"

    @staticmethod
    def _rows_from_data(board_data: Any, default_force: float) -> List[Dict[str, Any]]:
        data = _safe_json_loads(str(board_data or ""), {})
        if isinstance(data, list):
            raw_rows = data
        elif isinstance(data, dict):
            raw_rows = data.get("rows") if isinstance(data.get("rows"), list) else []
        else:
            raw_rows = []

        rows: List[Dict[str, Any]] = []
        cursor = 0.0
        for idx, item in enumerate(raw_rows):
            if not isinstance(item, dict):
                continue
            duration = max(0.1, _safe_float(item.get("duration", item.get("seconds", item.get("len", 3.0))), 3.0))
            second = _safe_float(item.get("second", item.get("start", cursor)), cursor)
            force = _clamp(item.get("force", item.get("guide_strength", default_force)), 0.0, 1.0, default_force)
            label = _normalise_label(str(item.get("label", item.get("name", "")) or ""), f"box_{idx + 1}")
            local_prompt = str(item.get("relay_prompt", item.get("local_prompt", item.get("prompt", ""))) or "").strip()
            bridge = str(item.get("step_transition_prompt", item.get("bridge", item.get("action_to_next", item.get("note", "")))) or "").strip()
            transition_type = str(item.get("step_transition_type", "action_beat" if bridge else "off") or "off").strip()
            rows.append({
                "_ui_id": str(item.get("_ui_id", f"boardmaker_{idx + 1:02d}_{label}")),
                "second": round(max(0.0, second), 3),
                "ref": max(1, _safe_int(item.get("ref", item.get("image_ref", idx + 1)), idx + 1)),
                "force": force,
                "image_lock_strength": force,
                "use_guide": IAMCCS_CineShotboardTimelinePro._as_bool(item.get("use_guide", item.get("guide", True)), True),
                "use_prompt": IAMCCS_CineShotboardTimelinePro._as_bool(item.get("use_prompt", item.get("relay", bool(local_prompt or bridge))), bool(local_prompt or bridge)),
                "label": label,
                "camera": str(item.get("camera", item.get("camera_move", "continuous dolly-in")) or "continuous dolly-in").strip(),
                "transition": str(item.get("transition", "continuous_motion") or "continuous_motion").strip(),
                "note": bridge,
                "relay_prompt": local_prompt,
                "use_relay_modifiers": False,
                "camera_relay_mode": str(item.get("camera_relay_mode", "off") or "off").strip(),
                "transition_relay_mode": str(item.get("transition_relay_mode", "off") or "off").strip(),
                "relay_addon_position": str(item.get("relay_addon_position", "after") or "after").strip(),
                "relay_modifier_text": str(item.get("relay_modifier_text", "") or "").strip(),
                "step_transition_enabled": bool(bridge and transition_type != "off"),
                "step_transition_type": transition_type,
                "step_transition_prompt": bridge,
                "step_transition_easing": str(item.get("step_transition_easing", "ease_in_out") or "ease_in_out").strip(),
                "step_transition_force_curve": str(item.get("step_transition_force_curve", "balanced") or "balanced").strip(),
                "step_transition_duration": max(0.0, _safe_float(item.get("step_transition_duration", item.get("bridge_duration", 0.0)), 0.0)),
                "step_transition_arrival": str(item.get("step_transition_arrival", "auto") or "auto").strip(),
                "step_transition_auto_fit": IAMCCS_CineShotboardTimelinePro._as_bool(item.get("step_transition_auto_fit", True), True),
                "duration": duration,
            })
            cursor = max(cursor, second + duration)
        return rows

    @classmethod
    def _build_board(cls, global_prompt: str, duration_seconds: float, frame_rate: float, image_width: int, image_height: int, default_force: float, guide_policy: str, board_name: str, board_data: str) -> Dict[str, Any]:
        rows = cls._rows_from_data(board_data, float(default_force))
        width = max(64, _safe_int(image_width, 768))
        height = max(64, _safe_int(image_height, 432))
        if rows:
            last_end = max(
                float(row.get("second", 0.0)) + max(0.1, _safe_float(row.get("duration", 0.1), 0.1))
                for row in rows
            )
            duration = max(float(duration_seconds), float(last_end))
        else:
            duration = max(0.1, float(duration_seconds))
        timeline_data = json.dumps({"rows": rows}, ensure_ascii=False, indent=2)
        return {
            "metadata": {
                "schema": "iamccs.cine.shotboard.board",
                "schema_version": 1,
                "cine_ui_version": "2026-05-19-boardmaker-1",
                "saved_at": datetime.datetime.now().isoformat(),
                "node_type": "IAMCCS_BoardMaker",
                "board_name": str(board_name or "iamccs_boardmaker_board"),
                "image_storage": "manual_after_import",
                "notes": "BoardMaker creates rows and prompts only. Add/import images manually in Shotboard V2 or V3 after importing this board.",
            },
            "global_prompt": str(global_prompt or ""),
            "prompt": str(global_prompt or ""),
            "timeline_data": timeline_data,
            "rows": rows,
            "settings": {
                "duration_seconds": duration,
                "frame_rate": float(frame_rate),
                "guide_policy": str(guide_policy or "every_checked_row"),
                "min_guide_gap_seconds": 0.0,
                "max_guides": max(1, len(rows)),
                "default_force": float(default_force),
                "promptrelay_epsilon": 0.6,
                "ltx_round_mode": "up_8n_plus_1",
                "image_width": width,
                "image_height": height,
                "image_resize_method": "crop",
                "image_multiple_of": 32,
                "img_compression": 0,
            },
            "duration_seconds": duration,
            "frame_rate": float(frame_rate),
            "guide_policy": str(guide_policy or "every_checked_row"),
            "default_force": float(default_force),
            "image_width": width,
            "image_height": height,
            "image_paths": "",
            "images": [],
        }

    def make_board(self, global_prompt, duration_seconds, frame_rate, image_width, image_height, default_force, guide_policy, board_name, board_data):
        board = self._build_board(global_prompt, duration_seconds, frame_rate, image_width, image_height, default_force, guide_policy, board_name, board_data)
        report = _json_report({
            "node": "IAMCCS_BoardMaker",
            "row_count": len(board.get("rows", [])),
            "duration_seconds": board.get("duration_seconds"),
            "image_policy": "manual_after_import",
            "truth": "Exported JSON can be dragged onto Shotboard V2 or V3. Images are intentionally empty and should be added manually after import.",
        })
        return json.dumps(board, ensure_ascii=False, indent=2), str(board.get("timeline_data", "")), len(board.get("rows", [])), float(board.get("duration_seconds", 0.0)), report


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
                    "default": "cinematic music video with rhythm-driven camera movement, coherent subject identity and continuous visual energy",
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


