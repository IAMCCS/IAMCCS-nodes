import json
import math
import os
import hashlib
import importlib.util
import re
import sys
import types
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

import folder_paths
import comfy.model_management
import node_helpers
from comfy_api.latest import InputImpl


WAN_SHOTBOARD_TYPE = "IAMCCS_WAN_SHOTBOARD"
WAN_TIMELINE_PLAN_TYPE = "IAMCCS_WAN_TIMELINE_PLAN"
WAN_LOOP_STATE_TYPE = "IAMCCS_WAN_LOOP_STATE"
_ORIGINAL_PROMPTRELAY_MODULE = None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return int(default)


def _parse_json(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _load_original_promptrelay_module():
    global _ORIGINAL_PROMPTRELAY_MODULE
    if _ORIGINAL_PROMPTRELAY_MODULE is not None:
        return _ORIGINAL_PROMPTRELAY_MODULE

    custom_nodes_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    promptrelay_dir = os.path.join(custom_nodes_dir, "ComfyUI-PromptRelay")
    nodes_path = os.path.join(promptrelay_dir, "nodes.py")
    if not os.path.exists(nodes_path):
        raise ImportError(f"ComfyUI-PromptRelay nodes.py not found: {nodes_path}")

    package_name = "_iamccs_wan_original_promptrelay"
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


def _split_paths(value: Any) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").replace("\r", "\n")
    parts: List[str] = []
    for line in text.split("\n"):
        clean = line.strip().strip('"')
        if clean:
            parts.append(clean)
    return parts


def _timeline_segments(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("segments", "rows", "keyframes"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    timeline = payload.get("timeline")
    if isinstance(timeline, dict):
        return _timeline_segments(timeline)
    return []


def _timeline_fps(payload: Dict[str, Any], fallback: Any) -> int:
    for key in ("frame_rate", "frameRate", "fps"):
        if key in payload:
            fps = _safe_int(payload.get(key), fallback)
            return max(1, fps)
    timeline = payload.get("timeline")
    if isinstance(timeline, dict):
        return _timeline_fps(timeline, fallback)
    return max(1, _safe_int(fallback, 16))


def _timeline_duration_seconds(payload: Dict[str, Any], fallback: Any) -> float:
    for key in ("duration_seconds", "durationSeconds", "duration"):
        if key in payload:
            return max(0.001, _safe_float(payload.get(key), fallback))
    timeline = payload.get("timeline")
    if isinstance(timeline, dict):
        return _timeline_duration_seconds(timeline, fallback)
    return max(0.001, _safe_float(fallback, 5.0))


def _is_visual_segment(row: Dict[str, Any]) -> bool:
    kind = str(row.get("type") or row.get("kind") or "").strip().lower()
    if kind in {"audio", "sound", "voice", "music"}:
        return False
    return True


def _is_image_segment(row: Dict[str, Any]) -> bool:
    kind = str(row.get("type") or row.get("kind") or "").strip().lower()
    if kind in {"audio", "sound", "voice", "music", "text", "relay", "prompt", "prompt_relay", "text_relay", "transition_relay", "slot_relay", "bridge"}:
        return False
    if kind in {"image", "keyframe", "frame", "still", "shot"}:
        return True
    if _segment_ref(row) is not None:
        return True
    return bool(row.get("path") or row.get("image_path") or row.get("imageFile") or row.get("imageTruthPath"))


def _segment_start(row: Dict[str, Any], fps: int) -> int:
    for key in ("start", "start_frame", "frame", "x"):
        if key in row:
            return max(0, _safe_int(row.get(key), 0))
    for key in ("start_seconds", "startSecond", "startTime", "time"):
        if key in row:
            return max(0, _safe_int(_safe_float(row.get(key), 0.0) * fps, 0))
    return 0


def _segment_length(row: Dict[str, Any], fps: int) -> int:
    for key in ("length", "frames", "duration_frames", "durationFrames"):
        if key in row:
            return max(1, _safe_int(row.get(key), 1))
    for key in ("duration_seconds", "durationSeconds", "seconds", "duration"):
        if key in row:
            return max(1, _safe_int(_safe_float(row.get(key), 0.0) * fps, 1))
    return 1


def _segment_prompt(row: Dict[str, Any]) -> str:
    for key in ("relay_prompt", "local_prompt", "localPrompt", "prompt", "text", "caption", "note"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _segment_prompt_enabled(row: Dict[str, Any]) -> bool:
    if _bool(row.get("relay_manual_off")) or _bool(row.get("promptrelay_manual_off")):
        return False
    if "use_prompt" in row:
        return _bool(row.get("use_prompt"))
    if "usePrompt" in row:
        return _bool(row.get("usePrompt"))
    return bool(_segment_prompt(row))


def _segment_ref(row: Dict[str, Any]) -> Optional[int]:
    for key in ("ref", "image_ref", "imageRef", "index"):
        if row.get(key) is not None:
            value = _safe_int(row.get(key), 0)
            if value > 0:
                return value
    return None


def _segment_path(row: Dict[str, Any], paths: List[str]) -> str:
    for key in ("imageTruthPath", "image_path", "path", "imageFile", "file", "filename"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    ref = _segment_ref(row)
    if ref is not None and 1 <= ref <= len(paths):
        return paths[ref - 1]
    return ""


def _relay_kind(row: Dict[str, Any]) -> str:
    raw = str(row.get("relay_kind") or row.get("relayKind") or "").strip().lower()
    if raw:
        return raw
    if _bool(row.get("slotRelay")) or _bool(row.get("slot_relay")):
        return "slot"
    if _bool(row.get("transitionRelay")) or _bool(row.get("transition_relay")):
        return "transition"
    kind = str(row.get("type") or row.get("kind") or "").strip().lower()
    if kind in {"slot_relay", "in_slot_relay"}:
        return "slot"
    if kind in {"transition_relay", "text", "prompt_relay", "text_relay"}:
        return "transition"
    return ""


def _source_prompt_for_pair(segments: List[Dict[str, Any]], pair: Dict[str, Any]) -> str:
    from_id = str(pair.get("from_id") or "")
    for seg in segments:
        if from_id and str(seg.get("id") or "") == from_id:
            prompt = _segment_prompt(seg)
            if prompt:
                return prompt
            break
    pair_prompt = str(pair.get("prompt") or "").strip()
    return pair_prompt or "maintain the current reference image composition and begin the motion cleanly"


def _image_segment_triplet(image_segments: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if not image_segments:
        return {}, {}, {}
    first = image_segments[0]
    if len(image_segments) == 1:
        return first, first, first
    if len(image_segments) == 2:
        return first, image_segments[1], image_segments[1]
    return first, image_segments[1], image_segments[2]


def _prompt_from_segments(global_prompt: str, *segments: Dict[str, Any], global_only: bool = False) -> str:
    parts = [str(global_prompt or "").strip()]
    if not global_only:
        for segment in segments:
            prompt = str(segment.get("prompt") or "").strip()
            if prompt:
                parts.append(prompt)
    seen = set()
    clean: List[str] = []
    for part in parts:
        key = part.lower()
        if part and key not in seen:
            clean.append(part)
            seen.add(key)
    return "\n".join(clean)


def _board_role_path(board: Dict[str, Any], role: str) -> str:
    role = str(role or "first").strip().lower()
    if role in {"first", "start", "anchor"}:
        return str(board.get("first_image_path") or board.get("start_image_path") or "")
    if role in {"middle", "mid", "bridge"}:
        return str(board.get("middle_image_path") or board.get("end_image_path") or board.get("start_image_path") or "")
    if role in {"last", "final", "end"}:
        return str(board.get("last_image_path") or board.get("end_image_path") or board.get("middle_image_path") or "")
    return str(board.get("start_image_path") or "")


def _flf_pairs(global_prompt: str, image_segments: List[Dict[str, Any]], global_only: bool = False) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    if len(image_segments) < 2:
        return pairs
    for index in range(len(image_segments) - 1):
        start_seg = image_segments[index]
        end_seg = image_segments[index + 1]
        motion = max(1.0, min(2.0, _safe_float(start_seg.get("motion", 1.0), 1.0)))
        chunk_start = int(start_seg.get("start") or 0)
        next_image_start = int(end_seg.get("start") or (chunk_start + int(start_seg.get("length") or 1)))
        chunk_length = max(1, next_image_start - chunk_start)
        pairs.append(
            {
                "index": index,
                "from_index": index + 1,
                "to_index": index + 2,
                "from_id": start_seg.get("id"),
                "to_id": end_seg.get("id"),
                "from_path": start_seg.get("path") or "",
                "to_path": end_seg.get("path") or "",
                "start": chunk_start,
                "length": chunk_length,
                "end": chunk_start + chunk_length,
                "motion": motion,
                "prompt": _prompt_from_segments(global_prompt, start_seg, end_seg, global_only=global_only),
            }
        )
    return pairs


def _relay_prompt_segments_for_pair(segments: List[Dict[str, Any]], pair: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunk_start = int(pair.get("start") or 0)
    chunk_length = max(1, int(pair.get("length") or 1))
    chunk_end = chunk_start + chunk_length
    active: List[Dict[str, Any]] = []
    for seg in sorted(segments, key=lambda item: (int(item.get("start") or 0), int(item.get("index") or 0))):
        prompt = str(seg.get("prompt") or "").strip()
        if not prompt or not _segment_prompt_enabled(seg):
            continue
        seg_start = int(seg.get("start") or 0)
        seg_end = max(seg_start + 1, int(seg.get("end") or (seg_start + int(seg.get("length") or 1))))
        if seg_end <= chunk_start or seg_start >= chunk_end:
            continue
        active.append(
            {
                **seg,
                "prompt": prompt,
                "_clamped_start": max(chunk_start, seg_start),
                "_relay_kind": _relay_kind(seg),
            }
        )
    if not active:
        return []

    relay_segments: List[Dict[str, Any]] = []
    first_start = max(chunk_start, int(active[0].get("_clamped_start") or chunk_start))
    if first_start > chunk_start:
        relay_segments.append(
            {
                "prompt": _source_prompt_for_pair(segments, pair),
                "length": max(1, first_start - chunk_start),
                "start": 0,
                "source_id": pair.get("from_id"),
                "source_type": "image",
                "relay_kind": "implicit_source",
            }
        )
    for index, seg in enumerate(active):
        start = max(chunk_start, int(seg.get("_clamped_start") or chunk_start))
        if index + 1 < len(active):
            end = max(start + 1, min(chunk_end, int(active[index + 1].get("_clamped_start") or chunk_end)))
        else:
            end = chunk_end
        relay_segments.append(
            {
                "prompt": str(seg.get("prompt") or "").strip(),
                "length": max(1, end - start),
                "start": max(0, start - chunk_start),
                "source_id": seg.get("id"),
                "source_type": seg.get("type"),
                "relay_kind": seg.get("_relay_kind") or ("base" if str(seg.get("type") or "") == "image" else "transition"),
                "parent_segment_id": seg.get("parentSegmentId") or seg.get("parent_segment_id") or "",
            }
        )

    diff = chunk_length - sum(int(seg["length"]) for seg in relay_segments)
    if relay_segments:
        relay_segments[-1]["length"] = max(1, int(relay_segments[-1]["length"]) + diff)
    return relay_segments


def _promptrelay_chunks(global_prompt: str, segments: List[Dict[str, Any]], pairs: List[Dict[str, Any]], epsilon: float, fps: int) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for pair in pairs:
        relay_segments = _relay_prompt_segments_for_pair(segments, pair)
        local_prompts = " | ".join(str(seg.get("prompt") or "").strip() for seg in relay_segments if str(seg.get("prompt") or "").strip())
        segment_lengths = ",".join(str(int(seg.get("length") or 1)) for seg in relay_segments)
        chunks.append(
            {
                "index": int(pair.get("index") or 0),
                "chunk_index": int(pair.get("index") or 0),
                "from_path": str(pair.get("from_path") or ""),
                "to_path": str(pair.get("to_path") or ""),
                "global_prompt": str(global_prompt or ""),
                "local_prompts": local_prompts,
                "segment_lengths": segment_lengths,
                "timeline_data": json.dumps({"segments": relay_segments}, ensure_ascii=False),
                "max_frames": max(1, int(pair.get("length") or 1)),
                "epsilon": float(epsilon),
                "fps": float(fps),
                "promptrelay_enabled": bool(local_prompts.strip()),
                "segments": relay_segments,
            }
        )
    return chunks


def _selected_relay_chunk(plan: Dict[str, Any], chunk_index: int) -> Dict[str, Any]:
    board = dict(plan.get("board") or {})
    chunks = plan.get("promptrelay_chunks") or board.get("promptrelay_chunks") or []
    if not isinstance(chunks, list) or not chunks:
        return {}
    selected_index = min(max(0, int(chunk_index or 0)), max(0, len(chunks) - 1))
    chunk = chunks[selected_index]
    return chunk if isinstance(chunk, dict) else {}


def _hash_text(text: Any) -> str:
    return hashlib.sha1(str(text or "").encode("utf-8", errors="ignore")).hexdigest()[:12]


def _image_tensor_from_path(path: str, width: int, height: int, resize_method: str, multiple_of: int) -> torch.Tensor:
    resolved = _resolve_image_path(path)
    dtype = comfy.model_management.intermediate_dtype()
    device = comfy.model_management.intermediate_device()

    components = InputImpl.VideoFromFile(resolved).get_components()
    if components.images.shape[0] > 0:
        images = components.images
        if _shotboard_resize_active(width, height, resize_method):
            return _resize_tensor_images(images, width, height, resize_method, multiple_of, dtype, device)
        return images.to(device=device, dtype=dtype)

    img = node_helpers.pillow(Image.open, resolved)
    output_images = []
    w, h = None, None
    for item in ImageSequence.Iterator(img):
        item = node_helpers.pillow(ImageOps.exif_transpose, item)
        image = item.convert("RGB")
        if _shotboard_resize_active(width, height, resize_method):
            image = _resize_image(image, width, height, resize_method, multiple_of)
        if len(output_images) == 0:
            w = image.size[0]
            h = image.size[1]
        if image.size[0] != w or image.size[1] != h:
            continue
        arr = np.array(image).astype(np.float32) / 255.0
        output_images.append(torch.from_numpy(arr)[None,].to(dtype=dtype))
    return torch.cat(output_images, dim=0).to(device=device, dtype=dtype)


def _shotboard_resize_active(width: int, height: int, resize_method: str) -> bool:
    mode = str(resize_method or "none").strip().lower()
    if mode in {"", "none", "original", "passthrough", "pass-through"}:
        return False
    return int(width or 0) > 0 and int(height or 0) > 0


def _resize_tensor_images(
    images: torch.Tensor,
    width: int,
    height: int,
    resize_method: str,
    multiple_of: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    output_images = []
    for frame in images.detach().cpu():
        arr = frame.float().clamp(0, 1).numpy()
        image = Image.fromarray((arr * 255.0).round().astype(np.uint8), mode="RGB")
        image = _resize_image(image, width, height, resize_method, multiple_of)
        resized = np.array(image).astype(np.float32) / 255.0
        output_images.append(torch.from_numpy(resized)[None,].to(dtype=dtype))
    return torch.cat(output_images, dim=0).to(device=device, dtype=dtype)


def _path_debug(path: str) -> Dict[str, Any]:
    try:
        resolved = _resolve_image_path(path)
        stat = os.stat(resolved)
        with open(resolved, "rb") as handle:
            digest = hashlib.sha1(handle.read()).hexdigest()[:12]
        with Image.open(resolved) as image:
            size = ImageOps.exif_transpose(image).size
        return {
            "path": str(path or ""),
            "resolved": resolved,
            "basename": os.path.basename(resolved),
            "size": list(size),
            "bytes": int(stat.st_size),
            "sha1": digest,
        }
    except Exception as exc:
        return {"path": str(path or ""), "error": str(exc)}


def _normalize_segments(payload: Dict[str, Any], image_paths: List[str], fps: int) -> Tuple[List[Dict[str, Any]], int]:
    normalized: List[Dict[str, Any]] = []
    max_end = 0
    for index, row in enumerate(_timeline_segments(payload)):
        if not _is_visual_segment(row):
            continue
        start = _segment_start(row, fps)
        length = _segment_length(row, fps)
        end = max(start + length, start + 1)
        max_end = max(max_end, end)
        ref = _segment_ref(row)
        normalized_row = {
            "index": index,
            "id": str(row.get("id") or row.get("label") or f"seg_{index:03d}"),
            "type": str(row.get("type") or "image"),
            "label": str(row.get("label") or ""),
            "relay_kind": _relay_kind(row),
            "parentSegmentId": str(row.get("parentSegmentId") or row.get("parent_segment_id") or ""),
            "slotRelay": _bool(row.get("slotRelay") or row.get("slot_relay")),
            "transitionRelay": _bool(row.get("transitionRelay") or row.get("transition_relay")),
            "start": start,
            "length": length,
            "end": end,
            "ref": ref,
            "path": _segment_path(row, image_paths),
            "prompt": _segment_prompt(row),
            "use_guide": _bool(row.get("use_guide") if "use_guide" in row else row.get("useGuide")),
            "use_prompt": _segment_prompt_enabled(row),
            "motion": max(1.0, min(2.0, _safe_float(row.get("motion", row.get("guideStrength", row.get("guide_strength", 1.0))), 1.0))),
            "guide_strength": max(1.0, min(2.0, _safe_float(row.get("motion", row.get("guideStrength", row.get("guide_strength", 1.0))), 1.0))),
        }
        normalized.append(normalized_row)
        nested_relays = row.get("slot_relays") or row.get("slotRelays") or []
        if isinstance(nested_relays, list) and _is_image_segment(normalized_row):
            for relay_index, relay in enumerate(nested_relays):
                if not isinstance(relay, dict):
                    continue
                relay_start = max(start, min(end - 1, _safe_int(relay.get("start"), start)))
                relay_length = max(1, min(end - relay_start, _safe_int(relay.get("length"), max(1, end - relay_start))))
                relay_prompt = _segment_prompt(relay)
                normalized.append(
                    {
                        "index": index + (relay_index + 1) / 1000.0,
                        "id": str(relay.get("id") or f"{normalized_row['id']}_slotrelay_{relay_index + 1}"),
                        "type": "text",
                        "label": str(relay.get("label") or "slot_relay"),
                        "relay_kind": "slot",
                        "parentSegmentId": str(normalized_row["id"]),
                        "slotRelay": True,
                        "transitionRelay": False,
                        "start": relay_start,
                        "length": relay_length,
                        "end": relay_start + relay_length,
                        "ref": None,
                        "path": "",
                        "prompt": relay_prompt,
                        "use_guide": False,
                        "use_prompt": _segment_prompt_enabled(relay),
                        "motion": 1.0,
                        "guide_strength": 1.0,
                    }
                )
    normalized.sort(key=lambda item: (item["start"], item["index"]))
    return normalized, max_end


def _resolve_image_path(path: str) -> str:
    clean = str(path or "").strip().strip('"')
    if not clean:
        raise FileNotFoundError("empty image path")
    if os.path.isabs(clean) and os.path.exists(clean):
        return clean
    if os.path.exists(clean):
        return os.path.abspath(clean)
    input_dir = folder_paths.get_input_directory()
    candidate = os.path.join(input_dir, clean)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(clean)


def _resampling() -> int:
    return getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def _fit_to_multiple(value: int, multiple_of: int) -> int:
    multiple = max(1, int(multiple_of or 1))
    return max(multiple, int(value) // multiple * multiple)


def _resize_image(image: Image.Image, width: int, height: int, mode: str, multiple_of: int) -> Image.Image:
    mode = str(mode or "crop").strip().lower()
    src_w, src_h = image.size
    if mode in {"none", "original", "passthrough", "pass-through"}:
        return image
    target_w = _fit_to_multiple(width if width > 0 else src_w, multiple_of)
    target_h = _fit_to_multiple(height if height > 0 else src_h, multiple_of)
    if src_w == target_w and src_h == target_h:
        return image

    resample = _resampling()
    if mode in {"stretch", "exact"}:
        return image.resize((target_w, target_h), resample)

    if mode in {"pad", "contain", "keep", "keep proportion", "letterbox"}:
        scale = min(target_w / src_w, target_h / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = image.resize((new_w, new_h), resample)
        canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
        return canvas

    scale = max(target_w / src_w, target_h / src_h)
    new_w = max(1, int(math.ceil(src_w * scale)))
    new_h = max(1, int(math.ceil(src_h * scale)))
    resized = image.resize((new_w, new_h), resample)
    left = max(0, (new_w - target_w) // 2)
    top = max(0, (new_h - target_h) // 2)
    return resized.crop((left, top, left + target_w, top + target_h))


class IAMCCS_WanShotboardPlannerPure:
    CATEGORY = "IAMCCS/Wan/PURE"
    RETURN_TYPES = (WAN_SHOTBOARD_TYPE,)
    RETURN_NAMES = ("cine_linx",)
    FUNCTION = "plan"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "global_prompt": ("STRING", {"multiline": True, "default": ""}),
                "timeline_data": ("STRING", {"multiline": True, "default": ""}),
                "duration_seconds": ("FLOAT", {"default": 5.0, "min": 0.0625, "max": 600.0, "step": 0.0625}),
                "frame_rate": ("INT", {"default": 16, "min": 1, "max": 240, "step": 1}),
                "guide_policy": (["every_checked_row", "first_last_only", "none"], {"default": "every_checked_row"}),
                "min_guide_gap_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.1}),
                "max_guides": ("INT", {"default": 50, "min": 0, "max": 200, "step": 1}),
                "default_force": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.01}),
                "promptrelay_epsilon": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 2.0, "step": 0.01}),
                "wan_frame_round_mode": (["none", "floor", "ceil", "nearest"], {"default": "none"}),
                "image_paths": ("STRING", {"multiline": True, "default": ""}),
                "image_width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 8}),
                "image_height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 8}),
                "image_resize_method": (["crop", "pad", "stretch", "none"], {"default": "crop"}),
                "image_multiple_of": ("INT", {"default": 16, "min": 1, "max": 128, "step": 1}),
                "img_compression": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "debug_verbose": ("BOOLEAN", {"default": False}),
            }
        }

    def plan(
        self,
        global_prompt: str,
        timeline_data: str,
        duration_seconds: float,
        frame_rate: int,
        guide_policy: str,
        min_guide_gap_seconds: float,
        max_guides: int,
        default_force: float,
        promptrelay_epsilon: float,
        wan_frame_round_mode: str,
        image_paths: str,
        image_width: int,
        image_height: int,
        image_resize_method: str,
        image_multiple_of: int,
        img_compression: int,
        debug_verbose: bool,
    ):
        payload = _parse_json(timeline_data)
        paths = _split_paths(image_paths)
        timeline_paths = _split_paths(payload.get("image_paths"))
        if timeline_paths:
            paths = timeline_paths

        fps = _timeline_fps(payload, frame_rate)
        fallback_seconds = _timeline_duration_seconds(payload, duration_seconds)
        segments, max_end_frame = _normalize_segments(payload, paths, fps)
        fallback_frames = max(1, _safe_int(fallback_seconds * fps, 1))
        duration_frames = max(1, max_end_frame or fallback_frames)
        truth_seconds = duration_frames / float(fps)
        global_prompt_only = _bool(payload.get("globalPromptOnly") if "globalPromptOnly" in payload else payload.get("global_prompt_only"))

        image_segments = [seg for seg in segments if _is_image_segment(seg)]
        if not image_segments and paths:
            step = max(1, duration_frames // max(1, len(paths) - 1))
            image_segments = []
            for idx, path in enumerate(paths):
                start = min(duration_frames - 1, idx * step)
                end = duration_frames if idx == len(paths) - 1 else min(duration_frames, start + step)
                image_segments.append(
                    {
                        "start": start,
                        "end": max(end, start + 1),
                        "length": max(1, max(end, start + 1) - start),
                        "ref": idx + 1,
                        "path": path,
                        "prompt": "",
                        "id": f"image_{idx + 1}",
                    }
                )

        first_seg, middle_seg, last_seg = _image_segment_triplet(image_segments)
        pair_plan = _flf_pairs(str(global_prompt or ""), image_segments, global_only=global_prompt_only)
        relay_chunks = _promptrelay_chunks(str(global_prompt or ""), segments, pair_plan, float(promptrelay_epsilon), fps)
        start_seg = first_seg
        end_seg = last_seg
        local_prompts = [seg["prompt"] for seg in segments if _segment_prompt_enabled(seg) and str(seg.get("prompt") or "").strip()]
        positive_parts = [str(global_prompt or "").strip()] if global_prompt_only else [str(global_prompt or "").strip()] + local_prompts
        positive_prompt = "\n".join(part for part in positive_parts if part)

        board = {
            "schema": "iamccs.wan.shotboard.pure",
            "schema_version": 1,
            "node": "IAMCCS_WanShotboardPlannerPure",
            "truth": "timeline_dragged_frames",
            "global_prompt": str(global_prompt or ""),
            "positive_prompt": positive_prompt,
            "timeline_data": str(timeline_data or ""),
            "image_paths": paths,
            "segments": segments,
            "timeline_order": image_segments,
            "flf_pairs": pair_plan,
            "promptrelay_chunks": relay_chunks,
            "promptrelay_enabled": any(_bool(chunk.get("promptrelay_enabled")) for chunk in relay_chunks),
            "promptrelay_epsilon": float(promptrelay_epsilon),
            "local_prompts": local_prompts,
            "global_prompt_only": bool(global_prompt_only),
            "globalPromptOnly": bool(global_prompt_only),
            "segment_lengths": [int(seg["length"]) for seg in segments],
            "promptrelay_segment_lengths": [chunk.get("segment_lengths", "") for chunk in relay_chunks],
            "duration_frames": int(duration_frames),
            "duration_seconds": float(truth_seconds),
            "frame_rate": int(fps),
            "first_image_path": str(first_seg.get("path") or ""),
            "middle_image_path": str(middle_seg.get("path") or ""),
            "last_image_path": str(last_seg.get("path") or ""),
            "start_image_path": str(start_seg.get("path") or ""),
            "end_image_path": str(end_seg.get("path") or ""),
            "start_ref": start_seg.get("ref"),
            "end_ref": end_seg.get("ref"),
            "first_prompt": str(first_seg.get("prompt") or ""),
            "middle_prompt": str(middle_seg.get("prompt") or ""),
            "last_prompt": str(last_seg.get("prompt") or ""),
            "positive_prompt_segment_1": _prompt_from_segments(str(global_prompt or ""), first_seg, middle_seg, global_only=global_prompt_only),
            "positive_prompt_segment_2": _prompt_from_segments(str(global_prompt or ""), middle_seg, last_seg, global_only=global_prompt_only),
            "frames_segment_1": int(pair_plan[0]["length"]) if len(pair_plan) > 0 else int(duration_frames),
            "frames_segment_2": int(pair_plan[1]["length"]) if len(pair_plan) > 1 else int(pair_plan[0]["length"]) if pair_plan else int(duration_frames),
            "chunk_count": max(0, len(pair_plan)),
            "image_width": int(image_width),
            "image_height": int(image_height),
            "image_resize_method": str(image_resize_method or "crop"),
            "image_multiple_of": int(image_multiple_of),
            "debug_verbose": bool(debug_verbose),
            "wan_only": True,
            "wan_pure_isolated": True,
        }
        if debug_verbose:
            print(
                "[IAMCCS WAN PURE] "
                f"duration_frames={duration_frames} fps={fps} seconds={truth_seconds:.4f} "
                f"segments={len(segments)} start='{board['start_image_path']}' end='{board['end_image_path']}'"
            )
            print(
                "[IAMCCS WAN PURE][TimelineOrder] "
                + json.dumps(
                    [
                        {
                            "slot": idx,
                            "id": seg.get("id"),
                            "start": seg.get("start"),
                            "length": seg.get("length"),
                            "end": seg.get("end"),
                            "ref": seg.get("ref"),
                            "path": seg.get("path"),
                            "motion": seg.get("motion"),
                            "prompt": str(seg.get("prompt") or "")[:120],
                            "file": _path_debug(str(seg.get("path") or "")),
                        }
                        for idx, seg in enumerate(image_segments)
                    ],
                    ensure_ascii=True,
                )
            )
            print(
                "[IAMCCS WAN PURE][FLFPairs] "
                + json.dumps(
                    [
                        {
                            "index": pair.get("index"),
                            "from_path": pair.get("from_path"),
                            "to_path": pair.get("to_path"),
                            "length": pair.get("length"),
                            "motion": pair.get("motion"),
                            "prompt": str(pair.get("prompt") or "")[:180],
                        }
                        for pair in pair_plan
                    ],
                    ensure_ascii=True,
                )
            )
            print(
                "[IAMCCS WAN PURE][PromptRelayChunks] "
                + json.dumps(
                    [
                        {
                            "index": chunk.get("index"),
                            "from_path": chunk.get("from_path"),
                            "to_path": chunk.get("to_path"),
                            "enabled": chunk.get("promptrelay_enabled"),
                            "global_prompt_len": len(str(chunk.get("global_prompt") or "")),
                            "local_prompts": chunk.get("local_prompts"),
                            "segment_lengths": chunk.get("segment_lengths"),
                            "segments": chunk.get("segments"),
                        }
                        for chunk in relay_chunks
                    ],
                    ensure_ascii=True,
                )
            )
        return (board,)


class IAMCCS_WanCineInfoPure:
    CATEGORY = "IAMCCS/Wan/PURE"
    RETURN_TYPES = (
        WAN_TIMELINE_PLAN_TYPE,
        "FLOAT",
        "INT",
        "INT",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "timeline_plan",
        "frame_rate",
        "chunk_count",
        "duration_frames",
        "timeline_order_json",
        "report",
    )
    FUNCTION = "info"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"cine_linx": (WAN_SHOTBOARD_TYPE,)}}

    def info(self, cine_linx: Dict[str, Any]):
        board = dict(cine_linx or {})
        timeline_order = board.get("timeline_order") or []
        if not isinstance(timeline_order, list):
            timeline_order = []
        flf_pairs = board.get("flf_pairs") or []
        if not isinstance(flf_pairs, list):
            flf_pairs = []
        promptrelay_chunks = board.get("promptrelay_chunks") or []
        if not isinstance(promptrelay_chunks, list):
            promptrelay_chunks = []

        chunk_count = len(flf_pairs)
        active_chunk_count = chunk_count
        timeline_plan = {
            "schema": "iamccs.wan.timeline_plan",
            "schema_version": 1,
            "board": board,
            "timeline_order": timeline_order,
            "flf_pairs": flf_pairs,
            "promptrelay_chunks": promptrelay_chunks,
            "frame_rate": float(board.get("frame_rate") or 16),
            "duration_frames": int(board.get("duration_frames") or 1),
            "image_width": int(board.get("image_width") or 832),
            "image_height": int(board.get("image_height") or 480),
            "image_resize_method": str(board.get("image_resize_method") or "crop"),
            "image_multiple_of": int(board.get("image_multiple_of") or 16),
            "chunk_count": chunk_count,
            "active_chunk_count": active_chunk_count,
        }
        timeline_order_json = json.dumps(
            {
                "timeline_order": timeline_order,
                "flf_pairs": flf_pairs,
                "promptrelay_chunks": promptrelay_chunks,
                "chunk_count": chunk_count,
                "active_chunk_count": active_chunk_count,
            },
            ensure_ascii=True,
        )
        report = json.dumps(
            {
                "schema": board.get("schema"),
                "truth": board.get("truth"),
                "duration_frames": board.get("duration_frames"),
                "frame_rate": board.get("frame_rate"),
                "chunk_count": chunk_count,
                "active_chunk_count": active_chunk_count,
                "promptrelay_enabled": bool(board.get("promptrelay_enabled")),
                "promptrelay_chunks": len(promptrelay_chunks),
                "wan_pure_isolated": board.get("wan_pure_isolated"),
            },
            ensure_ascii=True,
            indent=2,
        )
        if _bool(board.get("debug_verbose")):
            print(f"[IAMCCS WAN PURE][CineInfo] {report}")
            if len(flf_pairs) >= 2:
                continuity = {
                    "pair0_to": flf_pairs[0].get("to_path"),
                    "pair1_from": flf_pairs[1].get("from_path"),
                    "same_string": str(flf_pairs[0].get("to_path") or "") == str(flf_pairs[1].get("from_path") or ""),
                    "pair0_to_file": _path_debug(str(flf_pairs[0].get("to_path") or "")),
                    "pair1_from_file": _path_debug(str(flf_pairs[1].get("from_path") or "")),
                }
                print("[IAMCCS WAN PURE][CineInfo continuity 1->2] " + json.dumps(continuity, ensure_ascii=True))
        return (
            timeline_plan,
            float(timeline_plan["frame_rate"]),
            chunk_count,
            int(timeline_plan["duration_frames"]),
            timeline_order_json,
            report,
        )


class IAMCCS_WanFLFPairFromTimeline:
    CATEGORY = "IAMCCS/Wan/PURE"
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "INT", "FLOAT", "FLOAT", "INT", "INT", "STRING", "STRING", "STRING", "INT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "start_image",
        "end_image",
        "prompt",
        "frames",
        "motion",
        "frame_rate",
        "chunk_index",
        "chunk_count",
        "report",
        "relay_local_prompts",
        "relay_segment_lengths",
        "relay_max_frames",
        "relay_epsilon",
        "relay_timeline_data",
    )
    FUNCTION = "select"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline_plan": (WAN_TIMELINE_PLAN_TYPE,),
                "chunk_index": ("INT", {"default": 0, "min": 0, "max": 999, "step": 1}),
            }
        }

    def select(self, timeline_plan: Dict[str, Any], chunk_index: int = 0):
        plan = dict(timeline_plan or {})
        board = dict(plan.get("board") or {})
        pairs = plan.get("flf_pairs") or []
        if not isinstance(pairs, list):
            pairs = []
        timeline_order = plan.get("timeline_order") or []
        if not isinstance(timeline_order, list):
            timeline_order = []

        first_seg, second_seg, _third_seg = _image_segment_triplet([seg for seg in timeline_order if isinstance(seg, dict)])
        fallback_start = str(first_seg.get("path") or board.get("first_image_path") or board.get("start_image_path") or "")
        fallback_end = str(second_seg.get("path") or board.get("middle_image_path") or board.get("end_image_path") or fallback_start)
        chunk_count = len(pairs)
        requested_index = max(0, int(chunk_index or 0))
        inactive = requested_index >= chunk_count
        selected_index = requested_index if not inactive else requested_index
        selected = pairs[requested_index] if not inactive else {
            "from_path": fallback_start,
            "to_path": fallback_start,
            "prompt": "",
            "length": 1,
            "motion": 1.0,
            "inactive": True,
        }

        width = int(plan.get("image_width") or board.get("image_width") or 832)
        height = int(plan.get("image_height") or board.get("image_height") or 480)
        resize_method = str(plan.get("image_resize_method") or board.get("image_resize_method") or "crop")
        multiple_of = int(plan.get("image_multiple_of") or board.get("image_multiple_of") or 16)
        start_path = str(selected.get("from_path") or fallback_start)
        end_path = str(selected.get("to_path") or fallback_end)
        start_image = _image_tensor_from_path(start_path, width, height, resize_method, multiple_of)
        end_image = _image_tensor_from_path(end_path, width, height, resize_method, multiple_of)
        prompt = "" if inactive else str(selected.get("prompt") or board.get("positive_prompt") or board.get("global_prompt") or "")
        frames = int(selected.get("length") or 1) if inactive else int(selected.get("length") or plan.get("duration_frames") or 1)
        motion = max(1.0, min(2.0, _safe_float(selected.get("motion", 1.0), 1.0)))
        frame_rate = float(plan.get("frame_rate") or board.get("frame_rate") or 16)
        relay_chunk = {} if inactive else _selected_relay_chunk(plan, selected_index)
        relay_local_prompts = "" if inactive else str(relay_chunk.get("local_prompts") or "")
        relay_segment_lengths = "" if inactive else str(relay_chunk.get("segment_lengths") or "")
        relay_max_frames = 1 if inactive else int(relay_chunk.get("max_frames") or frames)
        relay_epsilon = float(_safe_float(relay_chunk.get("epsilon", board.get("promptrelay_epsilon", 0.001)), 0.001))
        relay_timeline_data = str(relay_chunk.get("timeline_data") or json.dumps({"segments": []}, ensure_ascii=False))
        report = json.dumps(
            {
                "chunk_index": selected_index,
                "requested_chunk_index": requested_index,
                "chunk_count": chunk_count,
                "active": not inactive,
                "from_path": start_path,
                "to_path": end_path,
                "frames": frames,
                "motion": motion,
                "frame_rate": frame_rate,
                "shotboard_resolution": {
                    "width": width,
                    "height": height,
                    "resize_method": resize_method,
                    "multiple_of": multiple_of,
                },
                "prompt": prompt,
                "promptrelay_enabled": bool(relay_local_prompts.strip()),
                "relay_local_count": len([part for part in relay_local_prompts.split("|") if part.strip()]),
                "relay_segment_lengths": relay_segment_lengths,
                "from_file": _path_debug(start_path),
                "to_file": _path_debug(end_path),
            },
            ensure_ascii=True,
        )
        if _bool(board.get("debug_verbose")):
            print(f"[IAMCCS WAN PURE][PairFromTimeline] {report}")
        return (
            start_image,
            end_image,
            prompt,
            frames,
            motion,
            frame_rate,
            selected_index,
            chunk_count,
            report,
            relay_local_prompts,
            relay_segment_lengths,
            relay_max_frames,
            relay_epsilon,
            relay_timeline_data,
        )


class IAMCCS_WanChunkGatePure:
    CATEGORY = "IAMCCS/Wan/PURE"
    RETURN_TYPES = ("IMAGE", "BOOLEAN", "STRING")
    RETURN_NAMES = ("images", "active", "report")
    FUNCTION = "gate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline_plan": (WAN_TIMELINE_PLAN_TYPE,),
                "chunk_index": ("INT", {"default": 0, "min": 0, "max": 999, "step": 1}),
            },
            "optional": {
                "source_images": ("IMAGE", {"lazy": True}),
                "generated_images": ("IMAGE", {"lazy": True}),
            },
        }

    @staticmethod
    def _active_count(timeline_plan: Dict[str, Any]) -> int:
        plan = dict(timeline_plan or {})
        try:
            return max(0, int(plan.get("active_chunk_count", plan.get("chunk_count", 0)) or 0))
        except Exception:
            return 0

    def check_lazy_status(self, timeline_plan, chunk_index=0, source_images=None, generated_images=None):
        active = int(chunk_index or 0) < self._active_count(timeline_plan)
        needed = []
        if active:
            if generated_images is None:
                needed.append("generated_images")
        elif source_images is None:
            needed.append("source_images")
        return needed

    def gate(self, timeline_plan, chunk_index=0, source_images=None, generated_images=None):
        plan = dict(timeline_plan or {})
        index = int(chunk_index or 0)
        active_count = self._active_count(plan)
        active = index < active_count
        if active and generated_images is not None:
            images = generated_images
            mode = "ACTIVE_GENERATED"
        elif source_images is not None:
            images = source_images
            mode = "INACTIVE_PASSTHROUGH"
        elif generated_images is not None:
            images = generated_images
            mode = "FALLBACK_GENERATED"
        else:
            raise ValueError("IAMCCS_WanChunkGatePure needs source_images or generated_images.")
        report = json.dumps(
            {
                "node": "IAMCCS_WanChunkGatePure",
                "chunk_index": index,
                "active_chunk_count": active_count,
                "active": bool(active),
                "mode": mode,
            },
            ensure_ascii=True,
        )
        board = dict(plan.get("board") or {})
        if _bool(board.get("debug_verbose")):
            print(f"[IAMCCS WAN PURE][ChunkGate] {report}")
        return images, bool(active), report


class IAMCCS_WanRelayOrBypassPure:
    CATEGORY = "IAMCCS/Wan/PURE"
    RETURN_TYPES = ("MODEL", "CONDITIONING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("model", "positive", "promptrelay_enabled", "report")
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline_plan": (WAN_TIMELINE_PLAN_TYPE,),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "chunk_index": ("INT", {"default": 0, "min": 0, "max": 999, "step": 1}),
            },
            "optional": {
                "clip": ("CLIP", {"lazy": True}),
                "latent": ("LATENT", {"lazy": True}),
                "relay_options": ("RELAY_OPTIONS",),
            },
        }

    @staticmethod
    def _relay_data(timeline_plan: Dict[str, Any], chunk_index: int) -> Tuple[bool, str, str, str, float, Dict[str, Any]]:
        plan = dict(timeline_plan or {})
        board = dict(plan.get("board") or {})
        chunk = _selected_relay_chunk(plan, chunk_index)
        global_prompt = str(chunk.get("global_prompt") or board.get("global_prompt") or "")
        local_prompts = str(chunk.get("local_prompts") or "")
        segment_lengths = str(chunk.get("segment_lengths") or "")
        epsilon = float(_safe_float(chunk.get("epsilon", board.get("promptrelay_epsilon", 0.001)), 0.001))
        active = bool(local_prompts.strip())
        return active, global_prompt, local_prompts, segment_lengths, epsilon, chunk

    def check_lazy_status(
        self,
        timeline_plan,
        model,
        positive,
        chunk_index,
        clip=None,
        latent=None,
        relay_options=None,
    ):
        active, *_ = self._relay_data(timeline_plan, chunk_index)
        needed = []
        if active:
            if clip is None:
                needed.append("clip")
            if latent is None:
                needed.append("latent")
        return needed

    def execute(
        self,
        timeline_plan,
        model,
        positive,
        chunk_index,
        clip=None,
        latent=None,
        relay_options=None,
    ):
        active, global_prompt, local_prompts, segment_lengths, epsilon, chunk = self._relay_data(timeline_plan, chunk_index)
        locals_list = [part.strip() for part in str(local_prompts or "").split("|") if part.strip()]
        length_parts = [part for part in re.split(r"[,;\s]+", str(segment_lengths or "")) if part.strip()]
        report_base = {
            "node": "IAMCCS_WanRelayOrBypassPure",
            "chunk_index": int(chunk_index or 0),
            "global_hash": _hash_text(global_prompt),
            "local_hash": _hash_text(local_prompts),
            "local_prompt_count": len(locals_list),
            "segment_count": len(length_parts),
            "segment_lengths": segment_lengths,
            "epsilon": float(epsilon),
            "from_path": chunk.get("from_path"),
            "to_path": chunk.get("to_path"),
        }
        if not active:
            report = json.dumps({**report_base, "promptrelay_enabled": False, "mode": "BYPASS_NO_LOCAL_PROMPTS"}, ensure_ascii=True)
            return model, positive, False, report

        if clip is None or latent is None:
            report = json.dumps(
                {
                    **report_base,
                    "promptrelay_enabled": False,
                    "mode": "BYPASS_MISSING_INPUTS",
                    "warning": "Relay active but clip or latent is missing.",
                },
                ensure_ascii=True,
            )
            print(
                "[IAMCCS WAN PURE][RelayOrBypass] WARNING mode=BYPASS_MISSING_INPUTS "
                f"chunk={int(chunk_index or 0)} clip={'present' if clip is not None else 'MISSING'} "
                f"latent={'present' if latent is not None else 'MISSING'}"
            )
            return model, positive, False, report

        try:
            promptrelay_nodes = _load_original_promptrelay_module()
            patched_model, relay_positive = promptrelay_nodes._encode_relay(
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
            report = json.dumps(
                {
                    **report_base,
                    "promptrelay_enabled": False,
                    "mode": "BYPASS_RELAY_ERROR",
                    "error": str(exc),
                },
                ensure_ascii=True,
            )
            print(f"[IAMCCS WAN PURE][RelayOrBypass] ERROR _encode_relay failed: {exc}. Falling back to bypass.")
            return model, positive, False, report

        report = json.dumps(
            {
                **report_base,
                "promptrelay_enabled": True,
                "mode": "PROMPT_RELAY_ORIGINAL_1TO1",
                "first_local": locals_list[0][:220] if locals_list else "",
                "last_local": locals_list[-1][:220] if locals_list else "",
            },
            ensure_ascii=True,
        )
        print(
            "[IAMCCS WAN PURE][RelayOrBypass] "
            f"PROMPT_RELAY_APPLIED chunk={int(chunk_index or 0)} "
            f"locals={len(locals_list)} segments={len(length_parts)} "
            f"global_hash={_hash_text(global_prompt)} local_hash={_hash_text(local_prompts)}"
        )
        for index, prompt in enumerate(locals_list[:20]):
            length = length_parts[index] if index < len(length_parts) else "<missing>"
            print(f"[IAMCCS WAN PURE][RelayOrBypass] relay[{index:02d}] length={length} prompt={prompt[:260]!r}")
        return patched_model, relay_positive, True, report


class IAMCCS_WanShotboardLoopInfo:
    CATEGORY = "IAMCCS/Wan/PURE/Loop"
    RETURN_TYPES = (WAN_TIMELINE_PLAN_TYPE, "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("timeline_plan", "chunk_count", "total_frames", "frame_rate", "report")
    FUNCTION = "info"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"cine_linx": (WAN_SHOTBOARD_TYPE,)}}

    def info(self, cine_linx: Dict[str, Any]):
        timeline_plan, frame_rate, chunk_count, duration_frames, _timeline_order_json, cine_report = IAMCCS_WanCineInfoPure().info(cine_linx)
        board = dict(timeline_plan.get("board") or {})
        pairs = timeline_plan.get("flf_pairs") or []
        relay_chunks = timeline_plan.get("promptrelay_chunks") or []
        report = json.dumps(
            {
                "node": "IAMCCS_WanShotboardLoopInfo",
                "schema": board.get("schema"),
                "truth": board.get("truth"),
                "chunk_count": int(chunk_count),
                "total_frames": int(duration_frames),
                "frame_rate": float(frame_rate),
                "timeline_images": len(timeline_plan.get("timeline_order") or []),
                "flf_pairs": len(pairs) if isinstance(pairs, list) else 0,
                "promptrelay_chunks": len(relay_chunks) if isinstance(relay_chunks, list) else 0,
                "easyuse_total": max(1, int(chunk_count)),
                "wan_pure_isolated": board.get("wan_pure_isolated"),
            },
            ensure_ascii=True,
        )
        if _bool(board.get("debug_verbose")):
            print(f"[IAMCCS WAN PURE][LoopInfo] {report}")
        return (timeline_plan, int(chunk_count), int(duration_frames), float(frame_rate), report)


class IAMCCS_WanShotboardLoopChunkSelect:
    CATEGORY = "IAMCCS/Wan/PURE/Loop"
    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "STRING",
        "INT",
        "FLOAT",
        "FLOAT",
        "INT",
        "INT",
        "STRING",
        "STRING",
        "STRING",
        "INT",
        "FLOAT",
        "STRING",
        "STRING",
        "STRING",
        WAN_TIMELINE_PLAN_TYPE,
    )
    RETURN_NAMES = (
        "start_image",
        "end_image",
        "prompt",
        "frames",
        "motion",
        "frame_rate",
        "chunk_index",
        "chunk_count",
        "report",
        "relay_local_prompts",
        "relay_segment_lengths",
        "relay_max_frames",
        "relay_epsilon",
        "relay_timeline_data",
        "start_path",
        "end_path",
        "selected_timeline_plan",
    )
    FUNCTION = "select"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline_plan": (WAN_TIMELINE_PLAN_TYPE,),
                "loop_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1, "lazy": True}),
                "index_mode": (["direct", "clamp"], {"default": "direct"}),
            }
        }

    def check_lazy_status(self, timeline_plan=None, loop_index=None, **kwargs):
        if loop_index is None:
            return ["loop_index"]
        return []

    def select(self, timeline_plan: Dict[str, Any], loop_index: int = 0, index_mode: str = "direct"):
        plan = dict(timeline_plan or {})
        pairs = plan.get("flf_pairs") or []
        if not isinstance(pairs, list):
            pairs = []
        chunk_count = len(pairs)
        requested_index = max(0, int(loop_index or 0))
        if chunk_count <= 0:
            raise ValueError("IAMCCS_WanShotboardLoopChunkSelect: no active FLF chunks. Add at least two image slots.")
        if requested_index >= chunk_count:
            if str(index_mode or "direct") == "clamp":
                selected_index = chunk_count - 1
            else:
                raise IndexError(
                    f"IAMCCS_WanShotboardLoopChunkSelect: loop_index {requested_index} is outside active chunk_count {chunk_count}."
                )
        else:
            selected_index = requested_index

        base = IAMCCS_WanFLFPairFromTimeline().select(plan, selected_index)
        selected = pairs[selected_index]
        start_path = str(selected.get("from_path") or "")
        end_path = str(selected.get("to_path") or "")
        relay_chunk = _selected_relay_chunk(plan, selected_index)
        selected_pair = dict(selected)
        selected_pair["index"] = 0
        selected_pair["original_index"] = selected_index
        selected_relay = dict(relay_chunk or {})
        selected_relay["index"] = 0
        selected_relay["original_index"] = selected_index
        selected_plan = dict(plan)
        selected_plan["flf_pairs"] = [selected_pair]
        selected_plan["promptrelay_chunks"] = [selected_relay] if selected_relay else []
        selected_plan["chunk_count"] = 1
        selected_plan["active_chunk_count"] = 1
        selected_plan["selected_original_chunk_index"] = selected_index
        report_data = {
            "node": "IAMCCS_WanShotboardLoopChunkSelect",
            "requested_loop_index": requested_index,
            "selected_chunk_index": selected_index,
            "chunk_count": chunk_count,
            "from_path": start_path,
            "to_path": end_path,
            "frames": int(base[3]),
            "motion": float(base[4]),
            "relay_local_prompts": str(base[9] or ""),
            "relay_segment_lengths": str(base[10] or ""),
            "relay_max_frames": int(base[11]),
            "start_file": _path_debug(start_path),
            "end_file": _path_debug(end_path),
        }
        board = dict(plan.get("board") or {})
        if _bool(board.get("debug_verbose")):
            print("[IAMCCS WAN PURE][LoopChunkSelect] " + json.dumps(report_data, ensure_ascii=True))
        merged_report = json.dumps(report_data, ensure_ascii=True)
        return (
            base[0],
            base[1],
            base[2],
            base[3],
            base[4],
            base[5],
            base[6],
            base[7],
            merged_report,
            base[9],
            base[10],
            base[11],
            base[12],
            base[13],
            start_path,
            end_path,
            selected_plan,
        )


class IAMCCS_WanShotboardPrevSamplesLoopSelect:
    CATEGORY = "IAMCCS/Wan/PURE/Loop"
    RETURN_TYPES = ("LATENT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("prev_samples", "using_loop_state", "report")
    FUNCTION = "select"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anchor_samples": ("LATENT",),
                "loop_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1, "lazy": True}),
            },
            "optional": {
                "previous_samples": ("LATENT",),
            },
        }

    def check_lazy_status(self, anchor_samples=None, loop_index=None, previous_samples=None, **kwargs):
        needed = []
        if anchor_samples is None:
            needed.append("anchor_samples")
        if loop_index is None:
            needed.append("loop_index")
        return needed

    def select(self, anchor_samples: Dict[str, Any], loop_index: int = 0, previous_samples: Optional[Dict[str, Any]] = None):
        index = max(0, int(loop_index or 0))
        use_previous = index > 0 and isinstance(previous_samples, dict) and previous_samples.get("samples") is not None
        selected = previous_samples if use_previous else anchor_samples
        samples = selected.get("samples") if isinstance(selected, dict) else None
        shape = list(samples.shape) if getattr(samples, "shape", None) is not None else None
        report = json.dumps(
            {
                "node": "IAMCCS_WanShotboardPrevSamplesLoopSelect",
                "loop_index": index,
                "using_loop_state": bool(use_previous),
                "shape": shape,
            },
            ensure_ascii=True,
        )
        print(f"[IAMCCS WAN PURE][PrevSamplesLoopSelect] {report}")
        return (selected, bool(use_previous), report)


class IAMCCS_WanShotboardLoopAccumulator:
    CATEGORY = "IAMCCS/Wan/PURE/Loop"
    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("images", "frame_count", "report")
    FUNCTION = "accumulate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chunk_images": ("IMAGE",),
                "loop_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1, "lazy": True}),
                "trim_first_frames_after_first": ("INT", {"default": 1, "min": 0, "max": 512, "step": 1}),
                "overlap_mode": (["linear_blend", "cut_trim"], {"default": "linear_blend"}),
                "overlap_blend_frames": ("INT", {"default": 5, "min": 0, "max": 512, "step": 1}),
            },
            "optional": {
                "previous_images": ("IMAGE",),
            },
        }

    def check_lazy_status(self, chunk_images=None, loop_index=None, previous_images=None, **kwargs):
        needed = []
        if chunk_images is None:
            needed.append("chunk_images")
        if loop_index is None:
            needed.append("loop_index")
        return needed

    def accumulate(
        self,
        chunk_images,
        loop_index: int = 0,
        trim_first_frames_after_first: int = 1,
        overlap_mode: str = "linear_blend",
        overlap_blend_frames: int = 5,
        previous_images=None,
    ):
        index = max(0, int(loop_index or 0))
        chunk_count = int(chunk_images.shape[0]) if getattr(chunk_images, "shape", None) is not None else 0
        if chunk_count <= 0:
            raise ValueError("IAMCCS_WanShotboardLoopAccumulator: chunk_images is empty.")
        had_previous = previous_images is not None and getattr(previous_images, "shape", None) is not None and int(previous_images.shape[0]) > 0
        previous_count = int(previous_images.shape[0]) if had_previous else 0
        mode = str(overlap_mode or "linear_blend")
        trim = max(0, int(trim_first_frames_after_first or 0)) if index > 0 else 0
        if trim >= chunk_count:
            trim = max(0, chunk_count - 1)
        available_after_trim = max(0, chunk_count - trim)
        blend_frames = max(0, int(overlap_blend_frames or 0)) if index > 0 and had_previous else 0
        blend_frames = min(blend_frames, previous_count, available_after_trim)

        if had_previous:
            previous = previous_images.to(device=chunk_images.device, dtype=chunk_images.dtype)
            if mode == "linear_blend" and blend_frames > 0:
                blend_source = chunk_images[trim:trim + blend_frames]
                if blend_frames == 1:
                    weights = torch.tensor([0.5], device=chunk_images.device, dtype=chunk_images.dtype)
                else:
                    weights = torch.linspace(
                        1.0 / float(blend_frames + 1),
                        float(blend_frames) / float(blend_frames + 1),
                        blend_frames,
                        device=chunk_images.device,
                        dtype=chunk_images.dtype,
                    )
                shape = [blend_frames] + [1] * (chunk_images.ndim - 1)
                weights = weights.reshape(shape)
                blended = previous[-blend_frames:] * (1.0 - weights) + blend_source * weights
                skip = trim + blend_frames
                contribution = chunk_images[skip:]
                output = torch.cat([previous[:-blend_frames], blended, contribution], dim=0)
            else:
                contribution = chunk_images[trim:] if trim > 0 else chunk_images
                output = torch.cat([previous, contribution], dim=0)
        else:
            contribution = chunk_images
            output = contribution
        report = json.dumps(
            {
                "node": "IAMCCS_WanShotboardLoopAccumulator",
                "loop_index": index,
                "previous_frames": previous_count,
                "chunk_frames": chunk_count,
                "trim_first_frames": trim,
                "overlap_mode": mode,
                "overlap_blend_frames": blend_frames,
                "overlap_source_start_after_trim": trim,
                "contributed_frames": int(contribution.shape[0]),
                "total_frames": int(output.shape[0]),
            },
            ensure_ascii=True,
        )
        print(f"[IAMCCS WAN PURE][LoopAccumulator] {report}")
        return (output, int(output.shape[0]), report)


class IAMCCS_WanShotboardLoopAccumulatorLinear:
    CATEGORY = "IAMCCS/Wan/PURE/Loop"
    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("images", "frame_count", "report")
    FUNCTION = "accumulate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chunk_images": ("IMAGE",),
                "loop_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1, "lazy": True}),
                "trim_first_frames_after_first": ("INT", {"default": 1, "min": 0, "max": 512, "step": 1}),
                "overlap_mode": (
                    ["cut", "linear_blend", "ease_in_out", "filmic_crossfade", "perceptual_crossfade"],
                    {"default": "linear_blend"},
                ),
                "overlap_frames": ("INT", {"default": 5, "min": 0, "max": 512, "step": 1}),
            },
            "optional": {
                "previous_images": ("IMAGE",),
            },
        }

    def check_lazy_status(self, chunk_images=None, loop_index=None, previous_images=None, **kwargs):
        needed = []
        if chunk_images is None:
            needed.append("chunk_images")
        if loop_index is None:
            needed.append("loop_index")
        return needed

    def accumulate(
        self,
        chunk_images,
        loop_index: int = 0,
        trim_first_frames_after_first: int = 1,
        overlap_mode: str = "linear_blend",
        overlap_frames: int = 5,
        previous_images=None,
    ):
        index = max(0, int(loop_index or 0))
        chunk_count = int(chunk_images.shape[0]) if getattr(chunk_images, "shape", None) is not None else 0
        if chunk_count <= 0:
            raise ValueError("IAMCCS_WanShotboardLoopAccumulatorLinear: chunk_images is empty.")

        had_previous = previous_images is not None and getattr(previous_images, "shape", None) is not None and int(previous_images.shape[0]) > 0
        previous_count = int(previous_images.shape[0]) if had_previous else 0
        trim = max(0, int(trim_first_frames_after_first or 0)) if index > 0 else 0
        if trim >= chunk_count:
            trim = max(0, chunk_count - 1)

        available_after_trim = max(0, chunk_count - trim)
        mode = str(overlap_mode or "linear_blend")
        valid_modes = {"cut", "linear_blend", "ease_in_out", "filmic_crossfade", "perceptual_crossfade"}
        if mode not in valid_modes:
            mode = "linear_blend"
        blend_frames = max(0, int(overlap_frames or 0)) if index > 0 and had_previous else 0
        blend_frames = min(blend_frames, previous_count, available_after_trim)

        if had_previous:
            previous = previous_images.to(device=chunk_images.device, dtype=chunk_images.dtype)
            if blend_frames > 0:
                blend_src = previous[-blend_frames:]
                blend_dst = chunk_images[trim:trim + blend_frames]
                prefix = previous[:-blend_frames]
                suffix = chunk_images[trim + blend_frames:]
                if mode == "cut":
                    output = torch.cat([prefix, chunk_images[trim:]], dim=0)
                    contribution = chunk_images[trim:]
                else:
                    alpha = torch.linspace(
                        0,
                        1,
                        blend_frames + 2,
                        device=chunk_images.device,
                        dtype=chunk_images.dtype,
                    )[1:-1]
                    shape = [blend_frames] + [1] * (chunk_images.ndim - 1)
                    alpha = alpha.reshape(shape)
                    if mode == "ease_in_out":
                        alpha = 3 * alpha * alpha - 2 * alpha * alpha * alpha
                    if mode == "filmic_crossfade":
                        gamma = 2.2
                        linear_src = torch.pow(torch.clamp(blend_src, 0.0, 1.0), gamma)
                        linear_dst = torch.pow(torch.clamp(blend_dst, 0.0, 1.0), gamma)
                        blended = torch.pow((1.0 - alpha) * linear_src + alpha * linear_dst, 1.0 / gamma)
                    elif mode == "perceptual_crossfade":
                        try:
                            import kornia
                            src_nchw = blend_src.movedim(-1, 1)
                            dst_nchw = blend_dst.movedim(-1, 1)
                            lab_src = kornia.color.rgb_to_lab(src_nchw)
                            lab_dst = kornia.color.rgb_to_lab(dst_nchw)
                            blended_lab = (1.0 - alpha) * lab_src + alpha * lab_dst
                            blended = kornia.color.lab_to_rgb(blended_lab).movedim(1, -1)
                        except Exception:
                            blended = (1.0 - alpha) * blend_src + alpha * blend_dst
                    else:
                        blended = (1.0 - alpha) * blend_src + alpha * blend_dst
                    contribution = suffix
                    output = torch.cat([prefix, blended, suffix], dim=0)
            else:
                contribution = chunk_images[trim:] if trim > 0 else chunk_images
                output = torch.cat([previous, contribution], dim=0)
        else:
            contribution = chunk_images
            output = contribution

        report = json.dumps(
            {
                "node": "IAMCCS_WanShotboardLoopAccumulatorLinear",
                "loop_index": index,
                "previous_frames": previous_count,
                "chunk_frames": chunk_count,
                "trim_first_frames": trim,
                "overlap_mode": mode,
                "overlap_frames_requested": int(overlap_frames or 0),
                "overlap_frames_used": blend_frames,
                "overlap_source_start_after_trim": trim,
                "contributed_frames": int(contribution.shape[0]),
                "total_frames": int(output.shape[0]),
            },
            ensure_ascii=True,
        )
        print(f"[IAMCCS WAN PURE][LoopAccumulatorLinear] {report}")
        return (output, int(output.shape[0]), report)


class IAMCCS_WanShotboardLoopState:
    CATEGORY = "IAMCCS/Wan/PURE/Loop"
    RETURN_TYPES = (WAN_LOOP_STATE_TYPE, "STRING")
    RETURN_NAMES = ("loop_state", "report")
    FUNCTION = "state"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline_plan": (WAN_TIMELINE_PLAN_TYPE,),
                "loop_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1, "lazy": True}),
            },
            "optional": {
                "previous_state": (WAN_LOOP_STATE_TYPE,),
                "chunk_report": ("STRING", {"forceInput": True, "default": ""}),
                "accumulator_report": ("STRING", {"forceInput": True, "default": ""}),
                "prev_samples_report": ("STRING", {"forceInput": True, "default": ""}),
            },
        }

    def state(
        self,
        timeline_plan: Dict[str, Any],
        loop_index: int = 0,
        previous_state: Optional[Dict[str, Any]] = None,
        chunk_report: str = "",
        accumulator_report: str = "",
        prev_samples_report: str = "",
    ):
        plan = dict(timeline_plan or {})
        index = max(0, int(loop_index or 0))
        state = dict(previous_state or {})
        history = list(state.get("history") or [])
        entry = {
            "loop_index": index,
            "chunk_report": str(chunk_report or ""),
            "accumulator_report": str(accumulator_report or ""),
            "prev_samples_report": str(prev_samples_report or ""),
        }
        history.append(entry)
        out = {
            "schema": "iamccs.wan.shotboard.loop_state",
            "schema_version": 1,
            "chunk_count": int(plan.get("chunk_count") or len(plan.get("flf_pairs") or [])),
            "last_loop_index": index,
            "history": history[-256:],
        }
        report = json.dumps(
            {
                "node": "IAMCCS_WanShotboardLoopState",
                "loop_index": index,
                "chunk_count": out["chunk_count"],
                "history_count": len(out["history"]),
            },
            ensure_ascii=True,
        )
        print(f"[IAMCCS WAN PURE][LoopState] {report}")
        return (out, report)


class IAMCCS_WanLoadImageFromPath:
    CATEGORY = "IAMCCS/Wan/PURE"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "width", "height", "report")
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"forceInput": True, "default": ""}),
                "width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 8}),
                "resize_method": (["none", "crop", "pad", "stretch"], {"default": "none"}),
                "multiple_of": ("INT", {"default": 16, "min": 1, "max": 128, "step": 1}),
                "debug_verbose": ("BOOLEAN", {"default": False}),
            }
        }

    def load(self, image_path: str, width: int, height: int, resize_method: str, multiple_of: int, debug_verbose: bool):
        resolved = _resolve_image_path(image_path)
        image = ImageOps.exif_transpose(Image.open(resolved)).convert("RGB")
        image = _resize_image(image, int(width), int(height), str(resize_method), int(multiple_of))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr)[None,].contiguous()
        report = json.dumps(
            {"path": resolved, "width": image.width, "height": image.height, "resize_method": resize_method},
            ensure_ascii=True,
        )
        if debug_verbose:
            print(f"[IAMCCS WAN PURE][LoadImage] {report}")
        return (tensor, int(image.width), int(image.height), report)


class IAMCCS_WanLoadImageFromBoard:
    CATEGORY = "IAMCCS/Wan/PURE"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wan_shotboard": (WAN_SHOTBOARD_TYPE,),
                "role": (["first", "middle", "last"], {"default": "first"}),
                "debug_verbose": ("BOOLEAN", {"default": False}),
            }
        }

    def load(self, wan_shotboard: Dict[str, Any], role: str, debug_verbose: bool):
        board = dict(wan_shotboard or {})
        path = _board_role_path(board, role)
        resolved = _resolve_image_path(path)
        width = int(board.get("image_width") or 832)
        height = int(board.get("image_height") or 480)
        resize_method = str(board.get("image_resize_method") or "crop")
        multiple_of = int(board.get("image_multiple_of") or 16)
        image = ImageOps.exif_transpose(Image.open(resolved)).convert("RGB")
        image = _resize_image(image, width, height, resize_method, multiple_of)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr)[None,].contiguous()
        report = json.dumps(
            {
                "role": role,
                "path": resolved,
                "width": image.width,
                "height": image.height,
                "resize_method": resize_method,
                "multiple_of": multiple_of,
            },
            ensure_ascii=True,
        )
        if debug_verbose or _bool(board.get("debug_verbose")):
            print(f"[IAMCCS WAN PURE][LoadImageFromBoard] {report}")
        return (tensor, report)


NODE_CLASS_MAPPINGS = {
    "IAMCCS_WanShotboardPlannerPure": IAMCCS_WanShotboardPlannerPure,
    "IAMCCS_WanCineInfoPure": IAMCCS_WanCineInfoPure,
    "IAMCCS_WanFLFPairFromTimeline": IAMCCS_WanFLFPairFromTimeline,
    "IAMCCS_WanChunkGatePure": IAMCCS_WanChunkGatePure,
    "IAMCCS_WanRelayOrBypassPure": IAMCCS_WanRelayOrBypassPure,
    "IAMCCS_WanShotboardLoopInfo": IAMCCS_WanShotboardLoopInfo,
    "IAMCCS_WanShotboardLoopChunkSelect": IAMCCS_WanShotboardLoopChunkSelect,
    "IAMCCS_WanShotboardPrevSamplesLoopSelect": IAMCCS_WanShotboardPrevSamplesLoopSelect,
    "IAMCCS_WanShotboardLoopAccumulator": IAMCCS_WanShotboardLoopAccumulator,
    "IAMCCS_WanShotboardLoopAccumulatorLinear": IAMCCS_WanShotboardLoopAccumulatorLinear,
    "IAMCCS_WanShotboardLoopState": IAMCCS_WanShotboardLoopState,
    "IAMCCS_WanLoadImageFromPath": IAMCCS_WanLoadImageFromPath,
    "IAMCCS_WanLoadImageFromBoard": IAMCCS_WanLoadImageFromBoard,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_WanShotboardPlannerPure": "IAMCCS WAN Shotboard Planner PURE",
    "IAMCCS_WanCineInfoPure": "IAMCCS WAN CineInfo PURE",
    "IAMCCS_WanFLFPairFromTimeline": "IAMCCS WAN FLF Pair From Timeline",
    "IAMCCS_WanChunkGatePure": "IAMCCS WAN Chunk Gate PURE",
    "IAMCCS_WanRelayOrBypassPure": "IAMCCS WAN Relay Or Bypass PURE",
    "IAMCCS_WanShotboardLoopInfo": "IAMCCS WAN Shotboard Loop Info",
    "IAMCCS_WanShotboardLoopChunkSelect": "IAMCCS WAN Shotboard Loop Chunk Select",
    "IAMCCS_WanShotboardPrevSamplesLoopSelect": "IAMCCS WAN Prev Samples Loop Select",
    "IAMCCS_WanShotboardLoopAccumulator": "IAMCCS WAN Shotboard Loop Accumulator",
    "IAMCCS_WanShotboardLoopAccumulatorLinear": "IAMCCS WAN Shotboard Loop Accumulator Linear Blend",
    "IAMCCS_WanShotboardLoopState": "IAMCCS WAN Shotboard Loop State",
    "IAMCCS_WanLoadImageFromPath": "IAMCCS WAN Load Image From Path PURE",
    "IAMCCS_WanLoadImageFromBoard": "IAMCCS WAN Load Image From Board PURE",
}
