"""IAMCCS native Qwen-Image 2.1 prompt enhancement / reference routing node.

V3.2: fast PE + queue rerun + multi-reference routing strategies.
Copyright (c) 2026 IAMCCS. MIT License.
"""
from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path

import comfy.utils

LOG = logging.getLogger("iamccs.prompt_q21_enh")
_VISION_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"
_PROMPT_DIR = Path(__file__).resolve().parent / "prompts" / "qwen_image_2_1"
_GENERATION_PROFILES = {
    "official": {"t2i": 16256, "edit": 24000},
    "balanced": {"t2i": 8192, "edit": 8192},
    "fast": {"t2i": 4096, "edit": 4096},
}
_MAX_TOKEN_LIMIT = 32768
_PE_IMAGE_MAX_PIXELS = 1024 * 1024
_SYSTEM_PROMPT_CACHE = {}

_POSE_TEXT_ONLY = """
POSE EXTRACTION MODE — IMPORTANT:
The downstream image generator will receive ONLY <image1>. Other images are visible to you only so you can
ANALYZE their pose. Use <image2> as a pose observation source, then translate that pose into explicit physical
language: torso orientation, shoulders, hips, head angle, gaze direction when visible, arm/elbow/wrist placement,
leg/knee/foot placement, stance, weight distribution and whether the figure is standing/sitting/leaning/crouching.
The FINAL rewritten_prompt MUST NOT contain <image2>, 'image 2', 'second image', 'reference image', or any request
to combine two people. It must read as a SINGLE-IMAGE EDIT of <image1>: re-pose the existing subject according
to the extracted geometry while preserving identity, face, hair, clothing, accessories, body appearance,
background, camera, lighting and style of <image1>. One subject only. Never transfer donor appearance.
Set ratio_follow to <image1> unless the user explicitly requests a different ratio.
If the user asks to keep the donor/background of image 2, this mode is not appropriate; do not pretend that
background can be transferred when the downstream generator will not see it.
"""

_SUBJECT1_ON_CANVAS2 = """
CANVAS-2 SUBJECT TRANSFER MODE — IMPORTANT:
Inputs have been ROUTED so that <image1> is the original user's image 2 and is the OUTPUT CANVAS.
<image2> is the original user's image 1 and is the SUBJECT/APPEARANCE DONOR.
Preserve from <image1>: pose/body placement, background, camera, framing, perspective, lighting and scene layout.
Transfer from <image2>: the requested subject identity and, unless the user says otherwise, face, hair, clothing,
accessories and recognizable appearance. Adapt the transferred subject naturally to the pose already present in
<image1>. Do not place both source people in the result. Do not make a collage, split frame, diptych, panorama,
double exposure or hybrid identity. The final image contains ONE subject in the <image1> canvas.
Set ratio_follow to <image1> unless an explicit target ratio is requested.
"""

_POSE_MULTIREF = """
DIRECT MULTI-REFERENCE POSE MODE (LESS ROBUST):
<image1> is the canvas/identity authority. <image2> is intended only as pose geometry. Preserve identity, face,
hair, clothing, accessories, body appearance, background, camera, lighting and style from <image1>. Transfer
only posture/joint geometry from <image2>. One subject only; no collage, split frame, hybrid or duplicate person.
Set ratio_follow to <image1> unless an explicit target ratio is requested.
"""


def _load_system_prompt(task):
    cached = _SYSTEM_PROMPT_CACHE.get(task)
    if cached is not None:
        return cached
    path = _PROMPT_DIR / f"system_prompt_{task}.txt"
    try:
        prompt = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"Unable to read Qwen-Image 2.1 {task} system prompt: {path}") from exc
    if not prompt:
        raise RuntimeError(f"Qwen-Image 2.1 {task} system prompt is empty: {path}")
    _SYSTEM_PROMPT_CACHE[task] = prompt
    return prompt


def _prepare_pe_image(image):
    """Match the official prompt-enhancer training cap without touching generator inputs."""
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) != 4:
        return image
    height, width = int(shape[1]), int(shape[2])
    if height * width <= _PE_IMAGE_MAX_PIXELS:
        return image
    scale = math.sqrt(_PE_IMAGE_MAX_PIXELS / float(height * width))
    target_width = max(32, math.floor(width * scale / 32) * 32)
    target_height = max(32, math.floor(height * scale / 32) * 32)
    samples = image.movedim(-1, 1)
    return comfy.utils.common_upscale(samples, target_width, target_height, "lanczos", "disabled").movedim(1, -1)


def _chat(system, user):
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def _split_answer(raw):
    raw = str(raw or "").strip()
    if "</think>" in raw:
        before, after = raw.split("</think>", 1)
        return before.split("<think>", 1)[-1].strip(), after.strip()
    if "<think>" in raw:
        return raw.split("<think>", 1)[-1].strip(), ""
    return "", raw


def _json_objects(text):
    start = None
    depth = 0
    quoted = False
    escaped = False
    for index, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start:index + 1]


def _parse(text, edit):
    for candidate in reversed(list(_json_objects(text))):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                import json_repair
                data = json_repair.repair_json(candidate, return_objects=True)
            except (ImportError, ValueError, TypeError):
                continue
        if isinstance(data, list):
            data = next((x for x in data if isinstance(x, dict)), None)
        if not isinstance(data, dict):
            continue
        positive = data.get("rewritten_prompt") or data.get("rewrited_prompt") or data.get("positive_prompt") or data.get("prompt")
        if not isinstance(positive, str) or not positive.strip():
            continue
        ratio = str(data.get("wh_ratio") or "").strip()
        follow = str(data.get("ratio_follow") or "").strip() if edit else ""
        if ratio:
            follow = ""
        return positive.strip(), ratio, follow, True
    fallback = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE).strip()
    return fallback, "", "", False


def _normalize_user_refs(text, slot_to_reference):
    """Convert <imageN>, image N, imageN to routed <imageK> tags without double replacements."""
    text = str(text)
    placeholders = {}
    def repl(match):
        slot = int(match.group(1))
        if slot not in slot_to_reference:
            raise ValueError(f"Prompt references image {slot} but image_{slot} is not connected")
        key = f"__IAMCCS_IMGREF_{len(placeholders)}__"
        placeholders[key] = f"<image{slot_to_reference[slot]}>"
        return key
    patterns = [r"<\s*image\s*(\d+)\s*>", r"\bimage\s+(\d+)\b", r"\bimage(\d+)\b"]
    for pat in patterns:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    for key, value in placeholders.items():
        text = text.replace(key, value)
    return text


def _auto_mode(prompt, connected_slots):
    if len(connected_slots) < 2:
        return "general"
    p = str(prompt).lower()
    pose = any(x in p for x in (
        "pose", "posture", "repose", "body position", "body posture", "match the pose",
        "posa", "postura", "posizione del corpo",
    ))
    if not pose:
        return "general"
    canvas2 = any(x in p for x in (
        "background of image 2", "background from image 2", "keep image 2 background",
        "keep the background of image 2", "sfondo dell'image 2", "sfondo di image 2",
        "mantieni lo sfondo dell'image 2", "mantieni lo sfondo di image 2",
    ))
    return "subject1_on_canvas2" if canvas2 else "pose_text_only"


class IAMCCS_PromptQ21Enh:
    @classmethod
    def INPUT_TYPES(cls):
        images = {f"image_{i}": ("IMAGE",) for i in range(1, 11)}
        return {
            "required": {
                "clip": ("CLIP",),
                "task": (["t2i", "edit"], {"default": "t2i"}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                **images,
                "extra_instructions": ("STRING", {"default": "", "multiline": True}),
                "missing_image_policy": (["error", "pass_through"], {"default": "error"}),
                "reference_mode": (["auto", "general", "pose_text_only", "subject1_on_canvas2", "pose_multiref"], {
                    "default": "auto",
                    "tooltip": "pose_text_only analyzes image_2 in the PE but routes only image_1 to the generator. subject1_on_canvas2 routes original image_2 as generator canvas and original image_1 as donor. pose_multiref passes both and is less robust."
                }),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 20, "min": 1, "max": 200, "step": 1}),
                "presence_penalty": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 5.0, "step": 0.1}),
                "max_new_tokens": ("INT", {"default": 0, "min": 0, "max": 32768, "step": 128,
                                    "tooltip": "0 = selected generation profile. Explicit values override the profile."}),
                "rerun_each_queue": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF, "control_after_generate": True}),
                "generation_profile": (["official", "balanced", "fast"], {
                    "default": "official",
                    "tooltip": "Official uses Qwen's 16256 T2I / 24000 Edit limits. Balanced uses 8192; Fast uses 4096."
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "BOOLEAN", "IMAGE", "IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "wh_ratio", "ratio_follow", "thinking", "parse_ok",
                    "generator_image_1", "generator_image_2", "routing_info", "generator_image_3")
    FUNCTION = "enhance"
    CATEGORY = "IAMCCS/Prompting"
    DESCRIPTION = "Qwen-Image 2.1 prompt enhancer with fast local generation and explicit multi-reference routing."

    @classmethod
    def IS_CHANGED(cls, rerun_each_queue=True, **kwargs):
        return float("nan") if bool(rerun_each_queue) else False

    def enhance(self, clip, task, prompt, extra_instructions="", missing_image_policy="error", reference_mode="auto",
                temperature=1.0, top_p=0.95, top_k=20, presence_penalty=-1.0,
                max_new_tokens=0, rerun_each_queue=True, seed=42, generation_profile="official", **kwargs):
        if not str(prompt).strip():
            return "", "", "", "", "", True, None, None, "empty prompt", None
        if task not in {"t2i", "edit"}:
            raise ValueError(f"Unknown IAMCCS PromptQ21Enh task: {task}")
        edit = task == "edit"
        slot_tensors = {}
        for i in range(1, 11):
            tensor = kwargs.get(f"image_{i}")
            if tensor is not None:
                if getattr(tensor, "ndim", None) != 4 or len(tensor) < 1:
                    raise ValueError(f"image_{i} must be a non-empty IMAGE batch")
                slot_tensors[i] = tensor

        if edit and not slot_tensors:
            if missing_image_policy == "pass_through":
                return str(prompt).strip(), "", "", "", "", False, None, None, "no edit image; pass-through", None
            raise ValueError("Edit mode requires at least one connected image")

        mode = str(reference_mode or "auto").strip().lower()
        if edit and mode == "auto":
            mode = _auto_mode(prompt, sorted(slot_tensors))
        if not edit:
            mode = "general"

        # Determine routed reference order. PE tags and generator outputs share this order.
        if edit and mode == "subject1_on_canvas2":
            if 1 not in slot_tensors or 2 not in slot_tensors:
                raise ValueError("subject1_on_canvas2 requires image_1 and image_2")
            routed_slots = [2, 1] + [i for i in sorted(slot_tensors) if i not in (1, 2)]
        else:
            routed_slots = sorted(slot_tensors)

        slot_to_reference = {slot: idx + 1 for idx, slot in enumerate(routed_slots)}
        images = []
        for slot in routed_slots:
            tensor = slot_tensors[slot]
            images.extend(_prepare_pe_image(tensor[row:row+1]) for row in range(len(tensor)))

        if edit:
            normalized_prompt = _normalize_user_refs(prompt, slot_to_reference)
            references = " ".join(f"<image{i}> {_VISION_TOKEN}" for i in range(1, len(images) + 1))
            user = f"{references}\n{normalized_prompt}"
            system = _load_system_prompt("edit")
            if mode == "pose_text_only":
                if len(routed_slots) < 2:
                    raise ValueError("pose_text_only requires at least image_1 and image_2")
                system += _POSE_TEXT_ONLY
            elif mode == "subject1_on_canvas2":
                system += _SUBJECT1_ON_CANVAS2
            elif mode == "pose_multiref":
                system += _POSE_MULTIREF
        else:
            user = str(prompt)
            system = _load_system_prompt("t2i")

        if str(extra_instructions).strip():
            system += "\nAdditional constraints:\n" + str(extra_instructions).strip()

        token_kwargs = {"skip_template": True, "min_length": 1, "thinking": True}
        if edit:
            token_kwargs["images"] = images
        tokens = clip.tokenize(_chat(system, user), **token_kwargs)
        try:
            requested_tokens = int(max_new_tokens or 0)
        except (TypeError, ValueError):
            requested_tokens = 0
        profile = str(generation_profile or "official").strip().lower()
        if profile not in _GENERATION_PROFILES:
            raise ValueError(f"Unknown IAMCCS PromptQ21Enh generation profile: {generation_profile}")
        generation_limit = _GENERATION_PROFILES[profile][task] if requested_tokens <= 0 else min(max(requested_tokens, 64), _MAX_TOKEN_LIMIT)
        if requested_tokens > _MAX_TOKEN_LIMIT:
            LOG.warning("PromptQ21Enh: max_new_tokens=%s clamped to %s", requested_tokens, generation_limit)
        generated = clip.generate(
            tokens, do_sample=True, max_length=generation_limit,
            temperature=float(temperature), top_k=int(top_k), top_p=float(top_p),
            min_p=0.0, repetition_penalty=1.0,
            presence_penalty=(0.0 if edit else 1.5) if float(presence_penalty) < 0 else float(presence_penalty),
            seed=int(seed),
        )
        thinking, answer = _split_answer(clip.decode(generated))
        positive, ratio, follow, parsed = _parse(answer, edit)
        if not parsed:
            LOG.warning("PromptQ21Enh: response was not valid prompt JSON; returning answer text")

        # Route images for TextEncodeQwenImage21. Connect these outputs instead of the raw loaders.
        gen1 = slot_tensors.get(routed_slots[0]) if routed_slots else None
        gen2 = slot_tensors.get(routed_slots[1]) if len(routed_slots) > 1 else None
        gen3 = slot_tensors.get(routed_slots[2]) if len(routed_slots) > 2 else None
        routing_info = "general"
        if mode == "pose_text_only":
            # Critical: donor image is PE-only, not sent to the image generator.
            gen1 = slot_tensors.get(1)
            gen2 = None
            gen3 = None
            # Final prompt must not accidentally retain donor tags.
            positive = re.sub(r"<image2>|\bimage\s*2\b|\bsecond image\b|\breference image\b", "", positive, flags=re.IGNORECASE)
            positive = re.sub(r"\s{2,}", " ", positive).strip()
            follow = "<image1>" if not ratio else ""
            routing_info = "POSE_TEXT_ONLY: connect generator_image_1 only; generator_image_2 is intentionally None"
        elif mode == "subject1_on_canvas2":
            routing_info = "CANVAS2: generator_image_1 = original image_2 canvas; generator_image_2 = original image_1 subject donor"
        elif mode == "pose_multiref":
            routing_info = "POSE_MULTIREF: both full photos are sent downstream; blending/duplication remains possible"

        return positive, "", ratio, follow, thinking, parsed, gen1, gen2, routing_info, gen3


NODE_CLASS_MAPPINGS = {"IAMCCS_PromptQ21Enh": IAMCCS_PromptQ21Enh}
NODE_DISPLAY_NAME_MAPPINGS = {"IAMCCS_PromptQ21Enh": "IAMCCS PromptQ21Enh V3.2"}
