import json
import math
import os
import hashlib
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
    if kind in {"image", "keyframe", "frame", "still", "shot"}:
        return True
    if row.get("ref") is not None or row.get("image_ref") is not None:
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
    for key in ("prompt", "local_prompt", "localPrompt", "text", "caption"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


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
        pairs.append(
            {
                "index": index,
                "from_index": index + 1,
                "to_index": index + 2,
                "from_id": start_seg.get("id"),
                "to_id": end_seg.get("id"),
                "from_path": start_seg.get("path") or "",
                "to_path": end_seg.get("path") or "",
                "start": int(start_seg.get("start") or 0),
                "length": max(1, int(start_seg.get("length") or 1)),
                "end": int(start_seg.get("end") or 0),
                "motion": motion,
                "prompt": _prompt_from_segments(global_prompt, start_seg, end_seg, global_only=global_only),
            }
        )
    return pairs


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
        normalized.append(
            {
                "index": index,
                "id": str(row.get("id") or row.get("label") or f"seg_{index:03d}"),
                "type": str(row.get("type") or "image"),
                "start": start,
                "length": length,
                "end": end,
                "ref": ref,
                "path": _segment_path(row, image_paths),
                "prompt": _segment_prompt(row),
                "use_guide": _bool(row.get("use_guide") if "use_guide" in row else row.get("useGuide")),
                "motion": max(1.0, min(2.0, _safe_float(row.get("motion", row.get("guideStrength", row.get("guide_strength", 1.0))), 1.0))),
                "guide_strength": max(1.0, min(2.0, _safe_float(row.get("motion", row.get("guideStrength", row.get("guide_strength", 1.0))), 1.0))),
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
        start_seg = first_seg
        end_seg = last_seg
        local_prompts = [seg["prompt"] for seg in segments if str(seg.get("prompt") or "").strip()]
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
            "local_prompts": local_prompts,
            "global_prompt_only": bool(global_prompt_only),
            "globalPromptOnly": bool(global_prompt_only),
            "segment_lengths": [int(seg["length"]) for seg in segments],
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

        chunk_count = len(flf_pairs)
        timeline_plan = {
            "schema": "iamccs.wan.timeline_plan",
            "schema_version": 1,
            "board": board,
            "timeline_order": timeline_order,
            "flf_pairs": flf_pairs,
            "frame_rate": float(board.get("frame_rate") or 16),
            "duration_frames": int(board.get("duration_frames") or 1),
            "image_width": int(board.get("image_width") or 832),
            "image_height": int(board.get("image_height") or 480),
            "image_resize_method": str(board.get("image_resize_method") or "crop"),
            "image_multiple_of": int(board.get("image_multiple_of") or 16),
            "chunk_count": chunk_count,
        }
        timeline_order_json = json.dumps(
            {
                "timeline_order": timeline_order,
                "flf_pairs": flf_pairs,
                "chunk_count": chunk_count,
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
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "INT", "FLOAT", "FLOAT", "INT", "INT", "STRING")
    RETURN_NAMES = ("start_image", "end_image", "prompt", "frames", "motion", "frame_rate", "chunk_index", "chunk_count", "report")
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
        selected_index = min(max(0, int(chunk_index or 0)), max(0, chunk_count - 1))
        selected = pairs[selected_index] if pairs else {
            "from_path": fallback_start,
            "to_path": fallback_end,
            "prompt": str(board.get("positive_prompt") or board.get("global_prompt") or ""),
            "length": int(plan.get("duration_frames") or board.get("duration_frames") or 1),
        }

        width = int(plan.get("image_width") or board.get("image_width") or 832)
        height = int(plan.get("image_height") or board.get("image_height") or 480)
        resize_method = str(plan.get("image_resize_method") or board.get("image_resize_method") or "crop")
        multiple_of = int(plan.get("image_multiple_of") or board.get("image_multiple_of") or 16)
        start_path = str(selected.get("from_path") or fallback_start)
        end_path = str(selected.get("to_path") or fallback_end)
        start_image = _image_tensor_from_path(start_path, width, height, resize_method, multiple_of)
        end_image = _image_tensor_from_path(end_path, width, height, resize_method, multiple_of)
        prompt = str(selected.get("prompt") or board.get("positive_prompt") or board.get("global_prompt") or "")
        frames = int(selected.get("length") or plan.get("duration_frames") or 1)
        motion = max(1.0, min(2.0, _safe_float(selected.get("motion", 1.0), 1.0)))
        frame_rate = float(plan.get("frame_rate") or board.get("frame_rate") or 16)
        report = json.dumps(
            {
                "chunk_index": selected_index,
                "chunk_count": chunk_count,
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
                "from_file": _path_debug(start_path),
                "to_file": _path_debug(end_path),
            },
            ensure_ascii=True,
        )
        if _bool(board.get("debug_verbose")):
            print(f"[IAMCCS WAN PURE][PairFromTimeline] {report}")
        return (start_image, end_image, prompt, frames, motion, frame_rate, selected_index, chunk_count, report)


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
    "IAMCCS_WanLoadImageFromPath": IAMCCS_WanLoadImageFromPath,
    "IAMCCS_WanLoadImageFromBoard": IAMCCS_WanLoadImageFromBoard,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_WanShotboardPlannerPure": "IAMCCS WAN Shotboard Planner PURE",
    "IAMCCS_WanCineInfoPure": "IAMCCS WAN CineInfo PURE",
    "IAMCCS_WanFLFPairFromTimeline": "IAMCCS WAN FLF Pair From Timeline",
    "IAMCCS_WanLoadImageFromPath": "IAMCCS WAN Load Image From Path PURE",
    "IAMCCS_WanLoadImageFromBoard": "IAMCCS WAN Load Image From Board PURE",
}
