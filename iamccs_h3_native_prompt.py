"""Request-scoped Qwen3-VL authoring through the installed ClipProj nodes."""
from __future__ import annotations

import base64
from contextlib import contextmanager
from contextvars import ContextVar
import io
import json
import logging
import re
import threading

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
import folder_paths


_session = ContextVar("iamccs_native_prompt", default=None)
_request_lock = threading.Lock()


def native_models():
    names = [name for name in folder_paths.get_filename_list("text_encoders")
             if "qwen3" in name.lower() and "vl" in name.lower()
             and any(size in name.lower() for size in ("4b", "8b", "32b"))
             and name.lower().endswith(".safetensors")]
    return sorted(names, key=lambda name: ("fp8_scaled" not in name.lower(), len(name), name))


def resolve_model(model, profile="4b_v3.1"):
    names = native_models()
    if model and model != "__workflow__":
        if model not in names:
            raise ValueError(f"Qwen encoder is not installed: {model}")
        return model
    if profile not in {"4b_v3.1", "8b_v3.1"}:
        raise ValueError("H3 ClipProj is OFF. Select 4b_v3.1/8b_v3.1 in Settings PRO or choose a Qwen encoder in Prompter.")
    size = "8b" if str(profile).startswith("8b") else "4b"
    candidates = [name for name in names if size in name.lower()]
    if not candidates:
        raise ValueError(f"No Qwen3-VL {size.upper()} safetensors encoder installed. Select an installed encoder in Prompter.")
    return candidates[0]


@contextmanager
def native_request(config=None):
    import comfy.utils

    if not _request_lock.acquire(blocking=False):
        raise RuntimeError("Qwen is already preparing a prompt. Wait for that REQUEST to finish.")
    token = _session.set({"config": config or {}, "clips": {}})
    saved_hook = comfy.utils.PROGRESS_BAR_HOOK

    def progress_hook(*args, **kwargs):
        # REQUEST has no queued node. Leave render progress and its registry alone.
        if _session.get() is None and saved_hook is not None:
            saved_hook(*args, **kwargs)

    comfy.utils.set_progress_bar_global_hook(progress_hook)
    try:
        yield
    finally:
        comfy.utils.set_progress_bar_global_hook(saved_hook)
        _session.get()["clips"].clear()
        _session.reset(token)
        _request_lock.release()


def run_ai_request(payload, function, *args):
    if payload.get("provider") != "qwen_native":
        return function(*args)
    with native_request(payload.get("native_clip")):
        return function(*args)


def _image_tensor(images):
    if not images:
        return None
    pictures = []
    for item in images:
        with Image.open(io.BytesIO(base64.b64decode(item["data"]))) as original:
            picture = ImageOps.exif_transpose(original).convert("RGB")
            picture.thumbnail((768, 768), Image.Resampling.LANCZOS)
            pictures.append(picture)
    if len(pictures) == 1:
        canvas = pictures[0]
    else:
        # ClipProjGenerate accepts one image; label each panel to retain slot roles.
        columns = 2
        width, height = 512, 536
        canvas = Image.new("RGB", (columns * width, ((len(pictures) + 1) // 2) * height), "#202020")
        draw = ImageDraw.Draw(canvas)
        for index, (picture, item) in enumerate(zip(pictures, images)):
            picture.thumbnail((width, height - 24), Image.Resampling.LANCZOS)
            x, y = (index % columns) * width, (index // columns) * height
            canvas.paste(picture, (x + (width - picture.width) // 2, y + 24))
            draw.text((x + 8, y + 5), f"Panel {index + 1} / slot {item.get('slot', index + 1)}", fill="white")
    return torch.from_numpy(np.asarray(canvas).astype(np.float32) / 255.0).unsqueeze(0)


def _validate(value, schema, path="response"):
    kind = schema.get("type")
    expected = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float), "boolean": bool}
    if kind in expected and not isinstance(value, expected[kind]):
        raise ValueError(f"{path} must be {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} has an unsupported value")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                raise ValueError(f"{path} is missing {key}")
        for key, child in schema.get("properties", {}).items():
            if key in value:
                _validate(value[key], child, f"{path}.{key}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", len(value)):
            raise ValueError(f"{path} has the wrong item count")
        for child in value:
            _validate(child, schema.get("items", {}), path + "[]")


def _parse(text, schema):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Qwen returned no complete JSON object")
    value = json.loads(text[start:end + 1])
    _validate(value, schema)
    return value


def native_json_chat(model, system, user, images=None, temperature=.3, schema=None, schema_name=""):
    if _session.get() is None:
        with native_request():
            return native_json_chat(model, system, user, images, temperature, schema, schema_name)
    # Import at use time: ComfyUI registers custom nodes after importing this module.
    import nodes
    import comfy.model_management as mm
    from server import PromptServer

    if PromptServer.instance.prompt_queue.get_current_queue()[0]:
        raise RuntimeError("ComfyUI is rendering. Run Qwen REQUEST when the video generation has finished.")
    state = _session.get()
    name = resolve_model(model, state["config"].get("profile", "4b_v3.1"))
    if name not in state["clips"]:
        loader = nodes.NODE_CLASS_MAPPINGS.get("ClipProjDeviceLoader")
        if loader is None or "ClipProjGenerate" not in nodes.NODE_CLASS_MAPPINGS:
            raise RuntimeError("Native Qwen requires the installed ClipProjDeviceLoader and ClipProjGenerate nodes.")
        logging.info("[IAMCCS Prompter] Loading native Qwen: %s", name)
        state["clips"][name] = loader().load(name, "auto", str(mm.get_torch_device()), mode="streaming", unique_id="IAMCCS_Prompter_REQUEST")[0]
    clip = state["clips"][name]
    schema = schema or {"type": "object"}
    budget = 512 if schema_name == "iamccs_edit_vision" else 384 if schema_name == "iamccs_edit_prompt" else 2048
    system += "\nRespond directly with one complete JSON object. No reasoning, markdown or commentary. Keep text concise. Required JSON schema:\n" + json.dumps(schema, ensure_ascii=False)
    if schema_name == "iamccs_edit_vision":
        system += " Keep each identity attribute under 15 words, or the single observations field under 70 words."
    if images and len(images) > 1:
        user += "\nImage panels in reading order: " + json.dumps([{key: item.get(key, "") for key in ("slot", "role", "name")} for item in images])
    generator = nodes.NODE_CLASS_MAPPINGS["ClipProjGenerate"]()
    text = generator.generate(clip, system, user, budget, min(float(temperature), .5), .9, 40, 0,
                              image=_image_tensor(images), repetition_penalty=1.1)[0]
    try:
        return _parse(text, schema)
    except ValueError as exc:
        if not text.strip():
            raise RuntimeError("Qwen returned an empty response. REQUEST did not change the prompt.") from exc
        # One bounded format repair; never fabricate a prompt when generation failed.
        logging.info("[IAMCCS Prompter] Repairing native JSON format: %s", exc)
        repaired = generator.generate(clip, "Convert the supplied draft to one complete JSON object matching this schema. Return JSON only, no reasoning. " + json.dumps(schema),
                                      "Format error: " + str(exc) + "\nDraft:\n" + text, budget, .1, .9, 40, 0,
                                      repetition_penalty=1.1)[0]
        try:
            return _parse(repaired, schema)
        except ValueError as final:
            raise RuntimeError(f"Native Qwen returned invalid/incomplete JSON ({final}). REQUEST did not change the prompt.") from final
