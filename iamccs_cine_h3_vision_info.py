# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later

"""Local vision analysis transport for the IAMCCS MiniMax H3 Prompter.

This module is deliberately independent from third-party promptor nodes.  A
native ComfyUI generative vision-language CLIP (normally Qwen3-VL) can inspect
up to four images.  The resulting prompt-ready observations, their semantic
roles, and their Shotboard injection targets travel through CineLinX; image
tensors are not retained in the bus after the analysis pass.
"""

from __future__ import annotations

import json
import gc
import re
import time
from typing import Any

import torch
import torch.nn.functional as torch_functional

try:
    from .iamccs_supernodes_linx import SUPERNODE_LINX_TYPE, build_stage_linx_payload
except ImportError:  # pragma: no cover - standalone schema/test fallback
    SUPERNODE_LINX_TYPE = "IAMCCS_SUPERNODE_LINX"

    def build_stage_linx_payload(
        existing_linx,
        stage_name,
        stage_kind,
        payload,
        report,
        slot_map=None,
        downstream_stages=None,
        policies=None,
        outputs=None,
        resources=None,
        **_kwargs,
    ):
        out = dict(existing_linx) if isinstance(existing_linx, dict) else {}
        out["type"] = SUPERNODE_LINX_TYPE
        chain = list(out.get("chain") or [])
        chain.append({"role": stage_name, "name": stage_name})
        out["chain"] = chain
        stages = list(out.get("stages") or [])
        stages.append({"name": stage_name, "kind": stage_kind, "payload": dict(payload), "report_preview": report[:240]})
        out["stages"] = stages
        out["stage_count"] = len(stages)
        for key, value in (("slot_map", slot_map), ("policies", policies), ("outputs", outputs), ("resources", resources)):
            merged = dict(out.get(key) or {})
            if isinstance(value, dict):
                merged.update({str(k): v for k, v in value.items() if v is not None})
            if merged:
                out[key] = merged
        if downstream_stages is not None:
            out["downstream_stages"] = [str(item) for item in downstream_stages]
        out["active_stage"] = stage_name
        out["active_stage_kind"] = stage_kind
        if isinstance(out.get("resources"), dict):
            out["resource_keys"] = sorted(out["resources"])
            out["resource_types"] = {key: type(value).__name__ for key, value in out["resources"].items()}
        return out


CATEGORY = "IAMCCS/MiniMax H3/Prompting"
STAGE_NAME = "IAMCCS Cine H3 Vision Info"
SCHEMA = "iamccs.minimax_h3.vision_context"
SCHEMA_VERSION = 1
MAX_CONTEXT_CHARACTERS = 5000
MAX_PREPARED_ANALYSIS_CHARACTERS = 5000
MAX_CUSTOM_ANALYSIS_CHARACTERS = 4000
MAX_USER_DIRECTION_CHARACTERS = 4000

ANALYSIS_MODES = [
    "h3_prompt_ready",
    "identity_continuity",
    "subject_performance",
    "scene_geography",
    "camera_light_composition",
    "wardrobe_props",
    "custom",
]
IMAGE_ROLES = [
    "opening_frame",
    "closing_frame",
    "subject_identity",
    "subject_performance",
    "wardrobe_prop",
    "environment",
    "composition_camera",
    "style_lighting",
    "generic_reference",
    "disabled",
]
INJECTION_TARGETS = ["global", "local_auto", "local_1", "local_2", "local_3"]
IMAGE_TARGETS = ["analysis_target", *INJECTION_TARGETS, "disabled"]
TEMPLATE_MODES = ["qwen3_vl_labeled", "model_default"]
PROMPT_ENHANCE_MODES = ["context_only", "enhance_visual_context"]

ROLE_RULES = {
    "opening_frame": "Treat this picture as the exact opening-frame authority; record visible identity, pose, composition, lens perspective, light, and screen geography that must be preserved at 0.00 seconds.",
    "closing_frame": "Treat this picture as the exact closing-frame authority; record the observable final pose, composition, light, and geometry that the preceding motion must reach naturally.",
    "subject_identity": "Extract only stable, visible identity evidence: face, hair, body proportions, age range, distinctive marks, and signature accessories. Do not infer biography or personality.",
    "subject_performance": "Extract visible pose, facial state, gaze, gesture, weight distribution, and physically plausible motion cues useful for directing a performance.",
    "wardrobe_prop": "Extract wardrobe and prop construction, material, colour, wear, scale, hand relationship, and continuity-critical details.",
    "environment": "Extract set geography, architecture, foreground/background layers, weather, time-of-day evidence, practical light sources, and continuity landmarks.",
    "composition_camera": "Extract shot size, camera height and angle, perspective, lens cues, depth of field, eyelines, screen direction, negative space, and framing relationships.",
    "style_lighting": "Extract only observable image-making properties: lighting direction and quality, contrast, palette, texture, medium, and finishing characteristics without turning them into vague quality adjectives.",
    "generic_reference": "Describe the visible subject, setting, composition, light, and any continuity-critical details, explicitly separating observation from uncertain inference.",
}

MODE_RULES = {
    "h3_prompt_ready": "Convert visible evidence into concise MiniMax H3 prompt-ready continuity direction while preserving explicit <Picture N> and <Subject N> references.",
    "identity_continuity": "Prioritize stable identity and continuity facts; omit story invention and unsupported emotion.",
    "subject_performance": "Prioritize observable acting, gaze, pose, gesture, facial state, and physically plausible motion development.",
    "scene_geography": "Prioritize spatial relationships, set geography, screen direction, depth layers, and environmental continuity.",
    "camera_light_composition": "Prioritize framing, camera viewpoint, lens evidence, depth of field, lighting direction, palette, and composition.",
    "wardrobe_props": "Prioritize wardrobe, props, materials, wear, scale, attachment, and hand/prop continuity.",
}


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _bounded_text(value: Any, limit: int, label: str) -> str:
    text = _clean_text(value)
    if len(text) > int(limit):
        raise ValueError(
            f"IAMCCS Cine H3 Vision Info: {label} contains {len(text)} characters; "
            f"the maximum is {int(limit)}"
        )
    return text


def _safe_target(value: Any, fallback: str = "global") -> str:
    clean = _clean_text(value).lower()
    return clean if clean in INJECTION_TARGETS else fallback


def _safe_role(value: Any) -> str:
    clean = _clean_text(value).lower()
    return clean if clean in IMAGE_ROLES else "generic_reference"


def _shape(value: Any) -> list[int]:
    if not torch.is_tensor(value):
        return []
    return [int(item) for item in value.shape]


def _connected_pictures(images: list[Any], roles: list[str], targets: list[str]) -> list[dict[str, Any]]:
    pictures: list[dict[str, Any]] = []
    for slot, (image, role, target) in enumerate(zip(images, roles, targets), start=1):
        role = _safe_role(role)
        if role == "disabled" or not torch.is_tensor(image) or image.ndim != 4 or int(image.shape[0]) < 1:
            continue
        pictures.append({
            "slot": slot,
            "label": f"<Picture {slot}>",
            "role": role,
            "target": _clean_text(target).lower(),
            "shape": _shape(image),
            "source_batch_frames": int(image.shape[0]),
            "tensor": image[0:1].detach(),
        })
    return pictures


def _fit_picture_to_canvas(image: torch.Tensor, canvas_height: int, canvas_width: int) -> torch.Tensor:
    """Aspect-fit a BHWC picture on a neutral canvas without cropping it."""
    source_height, source_width = int(image.shape[1]), int(image.shape[2])
    scale = min(canvas_width / max(1, source_width), canvas_height / max(1, source_height))
    fitted_width = max(1, int(round(source_width * scale)))
    fitted_height = max(1, int(round(source_height * scale)))
    channels_first = image.permute(0, 3, 1, 2).to(dtype=torch.float32)
    resized = torch_functional.interpolate(
        channels_first,
        size=(fitted_height, fitted_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp(0.0, 1.0)
    canvas = torch.full(
        (1, int(image.shape[3]), canvas_height, canvas_width),
        0.5,
        dtype=resized.dtype,
        device=resized.device,
    )
    top = (canvas_height - fitted_height) // 2
    left = (canvas_width - fitted_width) // 2
    canvas[:, :, top:top + fitted_height, left:left + fitted_width] = resized
    return canvas.permute(0, 2, 3, 1)


def _picture_batch(pictures: list[dict[str, Any]], max_side: int) -> tuple[torch.Tensor | None, dict[str, Any]]:
    if not pictures:
        return None, {"canvas": [], "downscaled": False}
    largest_height = max(int(item["tensor"].shape[1]) for item in pictures)
    largest_width = max(int(item["tensor"].shape[2]) for item in pictures)
    limit = max(256, min(1536, int(max_side)))
    scale = min(1.0, limit / max(1, largest_height, largest_width))
    canvas_height = max(32, int(round(largest_height * scale / 32.0)) * 32)
    canvas_width = max(32, int(round(largest_width * scale / 32.0)) * 32)
    fitted = [_fit_picture_to_canvas(item["tensor"], canvas_height, canvas_width) for item in pictures]
    return torch.cat(fitted, dim=0), {
        "canvas": [canvas_height, canvas_width, int(fitted[0].shape[-1])],
        "downscaled": bool(scale < 0.9999),
        "aspect_policy": "fit_with_neutral_padding_no_crop",
    }


def _build_analysis_prompt(
    analysis_mode: str,
    pictures: list[dict[str, Any]],
    custom_analysis_prompt: str,
) -> tuple[str, str]:
    mode = _clean_text(analysis_mode).lower()
    if mode == "custom":
        custom = _bounded_text(
            custom_analysis_prompt,
            MAX_CUSTOM_ANALYSIS_CHARACTERS,
            "custom_analysis_prompt",
        )
        if not custom:
            raise ValueError("IAMCCS Cine H3 Vision Info: custom mode requires a non-empty custom analysis prompt")
        mode_rule = custom
    else:
        mode_rule = MODE_RULES.get(mode, MODE_RULES["h3_prompt_ready"])

    picture_contract = []
    for item in pictures:
        picture_contract.append(
            f"Picture {item['slot']} is {item['label']}; role={item['role']}; "
            f"instruction={ROLE_RULES[item['role']]}"
        )
    system = (
        "You are the IAMCCS visual continuity analyst for MiniMax H3 filmmaking prompts. "
        "Report only evidence visible in the supplied pictures. Never identify a real person, infer private traits, invent backstory, dialogue, logos, text, off-screen objects, or a new scene. "
        "Images are supplied in ascending Picture number. Keep every Picture number stable. "
        "Return plain English only, with one block per supplied image. Start each block exactly with '<Picture N>:' and keep that block specific to that picture. "
        "Within each block write compact prompt-ready observations and continuity locks; clearly mark uncertainty instead of guessing. "
        "Do not emit Markdown headings, JSON, code fences, or an overall preface."
    )
    user = (
        f"Analysis mode: {mode}.\nMode objective: {mode_rule}\n\n"
        "Picture contracts:\n- " + "\n- ".join(picture_contract) + "\n\n"
        "For each picture include only the role-relevant visible facts, followed by one concise 'H3 direction:' sentence that tells a prompt writer what to preserve or animate."
    )
    return system, user


def _qwen3_labeled_chat(system: str, user: str, pictures: list[dict[str, Any]]) -> str:
    vision_token = "<|vision_start|><|image_pad|><|vision_end|>"
    media = "\n".join(f"Picture {item['slot']}: {vision_token}" for item in pictures)
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{media}\n\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _strip_generation_echo(decoded: Any, submitted_prompt: str) -> str:
    text = _clean_text(decoded)
    if submitted_prompt and text.startswith(submitted_prompt):
        text = text[len(submitted_prompt):].lstrip()
    if "<|im_start|>assistant" in text:
        text = text.rsplit("<|im_start|>assistant", 1)[-1]
    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    text = text.replace("<|im_end|>", "").strip()
    return text


def _clip_family_evidence(clip: Any) -> list[str]:
    """Collect bounded class/name evidence without walking model tensors."""
    evidence: list[str] = []
    queue = [clip]
    seen: set[int] = set()
    for _depth in range(3):
        next_queue: list[Any] = []
        for value in queue:
            if value is None or id(value) in seen:
                continue
            seen.add(id(value))
            evidence.append(type(value).__name__)
            for attr in (
                "tokenizer",
                "cond_stage_model",
                "patcher",
                "model",
                "clip_l",
                "qwen3vl_4b",
                "qwen3vl_8b",
                "qwen3vl_32b",
            ):
                child = getattr(value, attr, None)
                if child is not None and not isinstance(child, (str, int, float, bool, bytes)):
                    next_queue.append(child)
        queue = next_queue[:32]
    return evidence


def _validate_template_family(clip: Any, template_mode: str) -> dict[str, Any]:
    mode = _clean_text(template_mode)
    if mode not in TEMPLATE_MODES:
        raise ValueError(
            "IAMCCS Cine H3 Vision Info: template_mode must be one of "
            + ", ".join(TEMPLATE_MODES)
        )
    evidence = _clip_family_evidence(clip)
    normalized = " ".join(evidence).lower().replace("_", "")
    qwen3_vl = "qwen3vl" in normalized
    if mode == "qwen3_vl_labeled" and not qwen3_vl:
        raise ValueError(
            "IAMCCS Cine H3 Vision Info: template_mode=qwen3_vl_labeled requires a Qwen3-VL "
            "generative CLIP. The connected CLIP did not expose Qwen3-VL family markers. "
            "Use qwen3vl_4b_fp8_scaled.safetensors or select template_mode=model_default."
        )
    return {
        "template_mode": mode,
        "qwen3_vl_validated": qwen3_vl,
        "class_evidence": evidence[:16],
    }


def _release_vlm_runtime(enabled: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "requested": bool(enabled),
        "attempted": False,
        "actions": [],
        "errors": [],
    }
    if not enabled:
        result["status"] = "retained_by_user"
        return result
    result["attempted"] = True
    try:
        from comfy import model_management

        for action_name, action in (
            ("unload_all_models", getattr(model_management, "unload_all_models", None)),
            ("cleanup_models", getattr(model_management, "cleanup_models", None)),
        ):
            if callable(action):
                try:
                    action()
                    result["actions"].append(action_name)
                except Exception as exc:  # cleanup must never hide the analysis error
                    result["errors"].append(f"{action_name}: {exc}")
        soft_empty = getattr(model_management, "soft_empty_cache", None)
        if callable(soft_empty):
            try:
                soft_empty(force=True)
                result["actions"].append("soft_empty_cache(force=True)")
            except Exception as exc:
                result["errors"].append(f"soft_empty_cache: {exc}")
    except Exception as exc:
        result["errors"].append(f"comfy.model_management: {exc}")
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            result["actions"].append("torch.cuda.empty_cache")
    except Exception as exc:
        result["errors"].append(f"torch.cuda.empty_cache: {exc}")
    try:
        gc.collect()
        result["actions"].append("gc.collect")
    except Exception as exc:
        result["errors"].append(f"gc.collect: {exc}")
    result["status"] = "released" if not result["errors"] else "release_completed_with_warnings"
    return result


def _native_vision_generate(
    clip: Any,
    image_batch: torch.Tensor,
    pictures: list[dict[str, Any]],
    system: str,
    user: str,
    template_mode: str,
    seed: int,
    temperature: float,
    max_tokens: int,
) -> str:
    missing = [name for name in ("tokenize", "generate", "decode") if not callable(getattr(clip, name, None))]
    if missing:
        raise ValueError(
            "IAMCCS Cine H3 Vision Info requires a generative VLM CLIP with tokenize/generate/decode; "
            f"missing: {', '.join(missing)}. Use a current Qwen3-VL instruct text-encoder repack."
        )
    if _clean_text(template_mode) == "model_default":
        submitted = f"{system}\n\n{user}"
        tokens = clip.tokenize(
            submitted,
            image=image_batch,
            skip_template=False,
            min_length=1,
            thinking=False,
        )
    else:
        submitted = _qwen3_labeled_chat(system, user, pictures)
        tokens = clip.tokenize(
            submitted,
            image=image_batch,
            skip_template=True,
            min_length=1,
            thinking=False,
        )
    generated = clip.generate(
        tokens,
        do_sample=float(temperature) > 0.0,
        max_length=max(64, min(4096, int(max_tokens))),
        temperature=max(0.01, float(temperature)),
        top_k=64,
        top_p=0.9,
        min_p=0.0,
        repetition_penalty=1.05,
        presence_penalty=0.0,
        seed=max(0, int(seed)),
    )
    result = _strip_generation_echo(clip.decode(generated), submitted)
    if not result:
        raise RuntimeError("IAMCCS Cine H3 Vision Info: the VLM returned an empty analysis")
    return result


def _native_vision_generate_with_fallback(
    clip: Any,
    pictures: list[dict[str, Any]],
    analysis_mode: str,
    custom_analysis_prompt: str,
    system: str,
    user: str,
    template_mode: str,
    seed: int,
    temperature: float,
    max_tokens: int,
    max_side: int,
) -> tuple[str, dict[str, Any]]:
    """Prefer joint analysis, then retry each original slot independently.

    Some generative CLIP builds accept a single image but reject a BHWC batch.
    The fallback deliberately re-labels each result with the source slot rather
    than trusting a model that may call every isolated input ``Picture 1``.
    """
    image_batch, batch_meta = _picture_batch(pictures, max_side)
    assert image_batch is not None
    try:
        analysis = _native_vision_generate(
            clip,
            image_batch,
            pictures,
            system,
            user,
            template_mode,
            seed,
            temperature,
            max_tokens,
        )
        return analysis, {
            **batch_meta,
            "generation_strategy": "joint_multi_image" if len(pictures) > 1 else "single_image",
            "fallback_used": False,
        }
    except Exception as joint_error:
        if len(pictures) <= 1:
            raise
        recovered: list[str] = []
        per_picture_batches: list[dict[str, Any]] = []
        for offset, picture in enumerate(pictures):
            single_system, single_user = _build_analysis_prompt(
                analysis_mode,
                [picture],
                custom_analysis_prompt,
            )
            single_batch, single_meta = _picture_batch([picture], max_side)
            assert single_batch is not None
            raw = _native_vision_generate(
                clip,
                single_batch,
                [picture],
                single_system,
                single_user,
                template_mode,
                int(seed) + offset,
                temperature,
                max_tokens,
            )
            blocks = _split_picture_blocks(raw)
            body = blocks.get(int(picture["slot"])) or blocks.get(1) or _clean_text(raw)
            if not body:
                raise RuntimeError(
                    f"IAMCCS Cine H3 Vision Info: per-image fallback returned no text for Picture {picture['slot']}"
                ) from joint_error
            recovered.append(f"<Picture {picture['slot']}>:\n{body}")
            per_picture_batches.append({"picture_slot": int(picture["slot"]), **single_meta})
        return "\n\n".join(recovered), {
            **batch_meta,
            "generation_strategy": "per_image_fallback",
            "fallback_used": True,
            "joint_error": _clean_text(joint_error)[:500],
            "per_picture_batches": per_picture_batches,
        }


def _native_text_generate(
    clip: Any,
    system: str,
    user: str,
    template_mode: str,
    seed: int,
    temperature: float,
    max_tokens: int,
) -> str:
    missing = [name for name in ("tokenize", "generate", "decode") if not callable(getattr(clip, name, None))]
    if missing:
        raise ValueError(
            "IAMCCS Cine H3 Vision Info prompt enhancement requires a generative VLM CLIP; "
            f"missing: {', '.join(missing)}"
        )
    if _clean_text(template_mode) == "qwen3_vl_labeled":
        submitted = (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        tokens = clip.tokenize(
            submitted,
            skip_template=True,
            min_length=1,
            thinking=False,
        )
    else:
        submitted = f"{system}\n\n{user}"
        tokens = clip.tokenize(
            submitted,
            skip_template=False,
            min_length=1,
            thinking=False,
        )
    generated = clip.generate(
        tokens,
        do_sample=float(temperature) > 0.0,
        max_length=max(64, min(4096, int(max_tokens))),
        temperature=max(0.01, float(temperature)),
        top_k=64,
        top_p=0.9,
        min_p=0.0,
        repetition_penalty=1.05,
        presence_penalty=0.0,
        seed=max(0, int(seed)),
    )
    result = _strip_generation_echo(clip.decode(generated), submitted)
    if not result:
        raise RuntimeError("IAMCCS Cine H3 Vision Info: the local VLM returned an empty enhanced context")
    return result


def _split_target_blocks(text: str) -> dict[str, str]:
    clean = _clean_text(text)
    matches = list(
        re.finditer(
            r"(?im)^\s*(?:<\s*)?TARGET\s+(global|local_auto|local_1|local_2|local_3)(?:\s*>)?\s*:\s*",
            clean,
        )
    )
    blocks: dict[str, str] = {}
    for index, match in enumerate(matches):
        target = match.group(1).lower()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(clean)
        body = clean[match.end():end].strip()
        if body:
            blocks[target] = body
    return blocks


def _enhance_target_contexts(
    clip: Any,
    contexts: dict[str, str],
    user_direction: str,
    template_mode: str,
    seed: int,
    temperature: float,
    max_tokens: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Optionally direct the analyzed evidence with the same local VLM.

    This is a second, text-only pass.  It cannot add visual facts and it keeps
    the explicit Shotboard target map so Prompter routing remains deterministic.
    """
    ordered_targets = [target for target in INJECTION_TARGETS if target in contexts]
    if not ordered_targets:
        return contexts, {"status": "skipped_no_contexts", "parse_status": "none"}
    direction = _bounded_text(
        user_direction,
        MAX_USER_DIRECTION_CHARACTERS,
        "user_direction",
    )
    system = (
        "You are the IAMCCS MiniMax H3 visual prompt editor. Rewrite supplied visual evidence into precise, concise filmmaking direction. "
        "Do not invent identity, wardrobe, objects, dialogue, events, or camera facts absent from the evidence. "
        "A user direction may prioritize facts but cannot override the evidence. Preserve Picture and Subject labels. "
        "Return one plain-text block per requested routing target, beginning exactly with '<TARGET target>:'; emit no preface, Markdown, or JSON."
    )
    context_blocks = "\n\n".join(
        f"<TARGET {target}>:\n{contexts[target]}" for target in ordered_targets
    )
    user = (
        "Routing targets to preserve: " + ", ".join(ordered_targets) + "\n"
        + (f"User direction:\n{direction}\n\n" if direction else "")
        + "Visual evidence by target:\n"
        + context_blocks
        + "\n\nRewrite each block for MiniMax H3 while preserving every target boundary and all uncertainty."
    )
    enhanced_text = _native_text_generate(
        clip,
        system,
        user,
        template_mode,
        seed,
        temperature,
        max_tokens,
    )
    parsed = _split_target_blocks(enhanced_text)
    missing = [target for target in ordered_targets if not _clean_text(parsed.get(target))]
    if missing:
        raise RuntimeError(
            "IAMCCS Cine H3 Vision Info: enhancement did not return required target block(s): "
            + ", ".join(missing)
        )
    final = {target: parsed[target] for target in ordered_targets}
    total = sum(len(value) for value in final.values())
    if total > MAX_CONTEXT_CHARACTERS:
        remaining = MAX_CONTEXT_CHARACTERS
        trimmed: dict[str, str] = {}
        for target in ordered_targets:
            if remaining <= 0:
                break
            trimmed[target] = final[target][:remaining].rstrip()
            remaining -= len(trimmed[target])
        final = trimmed
        parse_status = "target_blocks+trimmed"
    else:
        parse_status = "target_blocks"
    return final, {
        "status": "enhanced_with_local_vlm",
        "parse_status": parse_status,
        "user_direction_characters": len(direction),
        "characters": sum(len(value) for value in final.values()),
    }


def _split_picture_blocks(text: str) -> dict[int, str]:
    clean = _clean_text(text)
    matches = list(re.finditer(r"(?im)^\s*(?:<\s*)?Picture\s+(\d+)(?:\s*>)?\s*:\s*", clean))
    blocks: dict[int, str] = {}
    for index, match in enumerate(matches):
        slot = int(match.group(1))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(clean)
        body = clean[match.end():end].strip()
        if body:
            blocks[slot] = body
    return blocks


def _target_contexts(
    analysis_text: str,
    pictures: list[dict[str, Any]],
    analysis_target: str,
) -> tuple[dict[str, str], str]:
    default_target = _safe_target(analysis_target)
    blocks = _split_picture_blocks(analysis_text)
    grouped: dict[str, list[str]] = {}
    if pictures and blocks:
        for item in pictures:
            body = blocks.get(int(item["slot"]))
            if not body:
                continue
            requested = _clean_text(item["target"]).lower()
            target = default_target if requested == "analysis_target" else _safe_target(requested, default_target)
            if requested == "disabled":
                continue
            grouped.setdefault(target, []).append(f"<Picture {item['slot']}> [{item['role']}]\n{body}")
    if not grouped:
        grouped[default_target] = [analysis_text]
        parse_status = "fallback_whole_context"
    else:
        parse_status = "picture_blocks"
    contexts = {target: "\n\n".join(parts).strip() for target, parts in grouped.items() if parts}
    total = sum(len(value) for value in contexts.values())
    if total > MAX_CONTEXT_CHARACTERS:
        remaining = MAX_CONTEXT_CHARACTERS
        trimmed: dict[str, str] = {}
        for target, value in contexts.items():
            if remaining <= 0:
                break
            selection = value[:remaining]
            boundary = selection.rfind("\n")
            if boundary > max(200, len(selection) // 2):
                selection = selection[:boundary]
            trimmed[target] = selection.rstrip()
            remaining -= len(selection)
        contexts = trimmed
        parse_status += "+trimmed"
    return contexts, parse_status


class IAMCCS_CineH3VisionInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "analysis_mode": (ANALYSIS_MODES, {"default": "h3_prompt_ready"}),
                "analysis_target": (INJECTION_TARGETS, {"default": "global"}),
                "context_merge_policy": (["append", "replace"], {"default": "append"}),
                "template_mode": (TEMPLATE_MODES, {"default": "qwen3_vl_labeled"}),
                "image_role_1": (IMAGE_ROLES, {"default": "opening_frame"}),
                "image_target_1": (IMAGE_TARGETS, {"default": "analysis_target"}),
                "image_role_2": (IMAGE_ROLES, {"default": "closing_frame"}),
                "image_target_2": (IMAGE_TARGETS, {"default": "analysis_target"}),
                "image_role_3": (IMAGE_ROLES, {"default": "subject_identity"}),
                "image_target_3": (IMAGE_TARGETS, {"default": "analysis_target"}),
                "image_role_4": (IMAGE_ROLES, {"default": "style_lighting"}),
                "image_target_4": (IMAGE_TARGETS, {"default": "analysis_target"}),
                "analysis_max_side": ("INT", {"default": 768, "min": 256, "max": 1536, "step": 32}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.5, "step": 0.05}),
                "max_tokens": ("INT", {"default": 768, "min": 128, "max": 4096, "step": 64}),
                "prompt_enhance_mode": (PROMPT_ENHANCE_MODES, {"default": "context_only"}),
                "release_vlm_after_analysis": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "clip": ("CLIP",),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "custom_analysis_prompt": ("STRING", {"default": "", "multiline": True}),
                "user_direction": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Used only when prompt_enhance_mode=enhance_visual_context.",
                    },
                ),
                "prepared_analysis": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "forceInput": True,
                        "tooltip": "Optional pre-generated vision description. When connected, it is used instead of running the local VLM.",
                    },
                ),
            },
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE,)
    RETURN_NAMES = ("cine_linx",)
    FUNCTION = "analyze"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        summary: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key == "clip":
                continue
            if key == "cine_linx" and isinstance(value, dict):
                summary[key] = {
                    "active_stage": value.get("active_stage"),
                    "stage_count": value.get("stage_count"),
                    "resource_keys": sorted((value.get("resources") or {}).keys())
                    if isinstance(value.get("resources"), dict)
                    else [],
                }
            else:
                summary[key] = _shape(value) if torch.is_tensor(value) else value
        return json.dumps(summary, ensure_ascii=False, sort_keys=True, default=str)

    def analyze(self, *args, **kwargs):
        """Public Comfy entry point with an error-path VLM release guard."""
        self._iamccs_vlm_release_done = False
        clip = kwargs.get("clip")
        release_requested = bool(kwargs.get("release_vlm_after_analysis", True))
        try:
            return self._analyze_impl(*args, **kwargs)
        except Exception:
            if clip is not None and not self._iamccs_vlm_release_done:
                _release_vlm_runtime(release_requested)
                self._iamccs_vlm_release_done = True
            raise
        finally:
            # Avoid carrying invocation state on a cached Comfy node instance.
            if hasattr(self, "_iamccs_vlm_release_done"):
                delattr(self, "_iamccs_vlm_release_done")

    def _analyze_impl(
        self,
        analysis_mode,
        analysis_target,
        context_merge_policy,
        template_mode,
        image_role_1,
        image_target_1,
        image_role_2,
        image_target_2,
        image_role_3,
        image_target_3,
        image_role_4,
        image_target_4,
        analysis_max_side,
        seed,
        temperature,
        max_tokens,
        prompt_enhance_mode,
        release_vlm_after_analysis,
        cine_linx=None,
        clip=None,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        custom_analysis_prompt="",
        user_direction="",
        prepared_analysis="",
    ):
        started = time.perf_counter()
        enhance_mode = _clean_text(prompt_enhance_mode).lower()
        if enhance_mode not in PROMPT_ENHANCE_MODES:
            raise ValueError(
                "IAMCCS Cine H3 Vision Info: prompt_enhance_mode must be one of "
                + ", ".join(PROMPT_ENHANCE_MODES)
            )
        custom_prompt = _bounded_text(
            custom_analysis_prompt,
            MAX_CUSTOM_ANALYSIS_CHARACTERS,
            "custom_analysis_prompt",
        )
        direction = _bounded_text(
            user_direction,
            MAX_USER_DIRECTION_CHARACTERS,
            "user_direction",
        )
        provided = _bounded_text(
            prepared_analysis,
            MAX_PREPARED_ANALYSIS_CHARACTERS,
            "prepared_analysis",
        )
        pictures = _connected_pictures(
            [image_1, image_2, image_3, image_4],
            [image_role_1, image_role_2, image_role_3, image_role_4],
            [image_target_1, image_target_2, image_target_3, image_target_4],
        )
        system_prompt, analysis_prompt = _build_analysis_prompt(
            str(analysis_mode), pictures, custom_prompt
        )
        batch_meta: dict[str, Any] = {"canvas": [], "downscaled": False}
        template_validation: dict[str, Any] = {
            "template_mode": str(template_mode),
            "qwen3_vl_validated": False,
            "class_evidence": [],
            "status": "not_checked_no_local_vlm_call",
        }
        release_report: dict[str, Any] = {
            "requested": bool(release_vlm_after_analysis),
            "attempted": False,
            "actions": [],
            "errors": [],
            "status": "not_loaded_by_stage",
        }
        enhancement_report: dict[str, Any] = {
            "mode": enhance_mode,
            "status": "context_only",
            "user_direction_characters": len(direction),
        }
        if provided:
            analysis_text = provided
            status = "prepared_analysis_used"
            raw_contexts, parse_status = _target_contexts(
                analysis_text, pictures, str(analysis_target)
            )
            contexts = dict(raw_contexts)
            if enhance_mode == "enhance_visual_context":
                if clip is None:
                    raise ValueError(
                        "IAMCCS Cine H3 Vision Info: enhance_visual_context requires the same local generative VLM CLIP"
                    )
                try:
                    template_validation = _validate_template_family(clip, str(template_mode))
                    try:
                        contexts, enhancement_report = _enhance_target_contexts(
                            clip,
                            raw_contexts,
                            direction,
                            str(template_mode),
                            int(seed) + 104729,
                            float(temperature),
                            int(max_tokens),
                        )
                        enhancement_report["mode"] = enhance_mode
                        status += "+visual_context_enhanced"
                    except Exception as enhancement_error:
                        contexts = dict(raw_contexts)
                        enhancement_report = {
                            "mode": enhance_mode,
                            "status": "enhancement_failed_raw_context_retained",
                            "error": _clean_text(enhancement_error)[:500],
                            "user_direction_characters": len(direction),
                        }
                        status += "+enhancement_failed_raw_context_retained"
                finally:
                    release_report = _release_vlm_runtime(bool(release_vlm_after_analysis))
                    self._iamccs_vlm_release_done = True
            elif clip is not None:
                # A connected CLIPLoader has already materialized its model even
                # though prepared text made inference unnecessary.
                release_report = _release_vlm_runtime(bool(release_vlm_after_analysis))
                self._iamccs_vlm_release_done = True
        else:
            if not pictures:
                raise ValueError("IAMCCS Cine H3 Vision Info requires at least one enabled IMAGE or a prepared_analysis input")
            if clip is None:
                raise ValueError(
                    "IAMCCS Cine H3 Vision Info requires a generative VLM CLIP when prepared_analysis is empty. "
                    "Connect qwen3vl_4b_fp8_scaled.safetensors through CLIPLoader."
                )
            try:
                template_validation = _validate_template_family(clip, str(template_mode))
                analysis_text, batch_meta = _native_vision_generate_with_fallback(
                    clip,
                    pictures,
                    str(analysis_mode),
                    custom_prompt,
                    system_prompt,
                    analysis_prompt,
                    str(template_mode),
                    int(seed),
                    float(temperature),
                    int(max_tokens),
                    int(analysis_max_side),
                )
                status = (
                    "native_vlm_analysis_complete_per_image_fallback"
                    if batch_meta.get("fallback_used")
                    else "native_vlm_analysis_complete"
                )
                raw_contexts, parse_status = _target_contexts(
                    analysis_text, pictures, str(analysis_target)
                )
                contexts = dict(raw_contexts)
                if enhance_mode == "enhance_visual_context":
                    try:
                        contexts, enhancement_report = _enhance_target_contexts(
                            clip,
                            raw_contexts,
                            direction,
                            str(template_mode),
                            int(seed) + 104729,
                            float(temperature),
                            int(max_tokens),
                        )
                        enhancement_report["mode"] = enhance_mode
                        status += "+visual_context_enhanced"
                    except Exception as enhancement_error:
                        contexts = dict(raw_contexts)
                        enhancement_report = {
                            "mode": enhance_mode,
                            "status": "enhancement_failed_raw_context_retained",
                            "error": _clean_text(enhancement_error)[:500],
                            "user_direction_characters": len(direction),
                        }
                        status += "+enhancement_failed_raw_context_retained"
            finally:
                # Always reached after validation, tokenization, generation, or
                # enhancement errors so a failed VLM cannot stay resident.
                release_report = _release_vlm_runtime(bool(release_vlm_after_analysis))
                self._iamccs_vlm_release_done = True

        picture_manifest = [
            {key: value for key, value in item.items() if key != "tensor"}
            for item in pictures
        ]
        manifest = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "engine": "comfy_native_generative_vlm" if not provided else "prepared_analysis",
            "recommended_model_family": "Qwen3-VL Instruct FP8",
            "recommended_clip_file": "qwen3vl_4b_fp8_scaled.safetensors",
            "analysis_mode": str(analysis_mode),
            "analysis_target": _safe_target(analysis_target),
            "context_merge_policy": str(context_merge_policy),
            "template_mode": str(template_mode),
            "template_validation": template_validation,
            "status": status,
            "parse_status": parse_status,
            "prompt_enhance": enhancement_report,
            "pictures": picture_manifest,
            "target_contexts": contexts,
            "raw_target_contexts": raw_contexts,
            "analysis_characters": len(analysis_text),
            "target_context_characters": sum(len(value) for value in contexts.values()),
            "batch": batch_meta,
            "vlm_release": release_report,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "privacy": "image tensors consumed during analysis and not retained in CineLinX",
        }
        report = (
            "IAMCCS Cine H3 Vision Info | "
            f"status={status} | images={len(pictures)}/4 | mode={analysis_mode} | "
            f"targets={','.join(contexts) or 'none'} | parse={parse_status} | "
            f"chars={manifest['target_context_characters']} | tensors_retained=no"
        )
        resources = {
            "iamccs_h3_vision_manifest": manifest,
            "iamccs_h3_vision_manifest_json": json.dumps(manifest, ensure_ascii=False, indent=2),
            "iamccs_h3_vision_context": analysis_text,
            "iamccs_h3_vision_context_by_target": contexts,
            "iamccs_h3_vision_raw_context_by_target": raw_contexts,
            "iamccs_h3_vision_analysis_prompt": analysis_prompt,
            "iamccs_h3_vision_analysis_status": status,
            "iamccs_h3_vision_context_merge_policy": str(context_merge_policy),
        }
        out = build_stage_linx_payload(
            cine_linx,
            stage_name=STAGE_NAME,
            stage_kind="minimax_h3_vision_analysis",
            payload={
                "schema": SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "status": status,
                "analysis_mode": str(analysis_mode),
                "targets": list(contexts),
                "pictures": picture_manifest,
            },
            report=report,
            slot_map={"cine_linx": "IAMCCS Prompter cine_linx input"},
            downstream_stages=("IAMCCS Prompter", "MiniMax H3 Shotboard"),
            policies={
                "vision_context_targets_are_explicit": True,
                "image_tensor_retention": "none_after_analysis",
                "vision_context_merge_policy": str(context_merge_policy),
                "prompt_enhance_mode": enhance_mode,
                "release_vlm_after_analysis": bool(release_vlm_after_analysis),
            },
            outputs={"vision_analysis_status": status, "report": report},
            resources=resources,
        )
        return (out,)


NODE_CLASS_MAPPINGS = {"IAMCCS_CineH3VisionInfo": IAMCCS_CineH3VisionInfo}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_CineH3VisionInfo": "IAMCCS Cine H3 Vision Info - Local VLM",
}
