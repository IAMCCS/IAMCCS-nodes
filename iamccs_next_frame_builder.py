"""IAMCCS NextFrameBuilder.

The visible node is a compact workflow wrapper.  At execution time it expands
to the same Qwen Image Edit 2511 chain used by the reference workflow supplied
for this feature.  Generated frames are committed to ComfyUI's input folder so
they can immediately become the next source frame without copying tensors or
depending on a temporary preview file.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import time
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps, PngImagePlugin

import comfy.samplers
import folder_paths
from comfy_execution.graph_utils import GraphBuilder


LOGGER = logging.getLogger("IAMCCS.NextFrameBuilder")

DEFAULT_GGUF = "qwen-image-edit-2511-Q4_0.gguf"
DEFAULT_NATIVE = "qwen_image_edit_2511_bf16.safetensors"
DEFAULT_CLIP = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
DEFAULT_VAE = "qwen_image_vae.safetensors"
DEFAULT_LIGHTNING_LORA = (
    "Qwen_2511\\Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
)
DEFAULT_NEXT_SCENE_LORA = "next-scene_lora-v2-3000.safetensors"
DEFAULT_LIGHT_LORA = "qwen2511_Add_light_and_shadow.safetensors"
SUPERNODE_LINX_TYPE = "IAMCCS_SUPERNODE_LINX"


def _safe_json_dict(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value or "{}"))
    except Exception:
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def _clean_session(value: Any) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value or "session")).strip("-_")
    return (clean or "session")[:48]


def _frame_path(filename: Any) -> str | None:
    value = str(filename or "").strip()
    if not value:
        return None
    try:
        resolved = folder_paths.get_annotated_filepath(value)
        if resolved and os.path.isfile(resolved):
            return resolved
    except Exception:
        pass
    candidate = os.path.join(folder_paths.get_input_directory(), value)
    return candidate if os.path.isfile(candidate) else None


def _selected_storyboard_frames(board: dict[str, Any]) -> list[dict[str, Any]]:
    frames = [item for item in board.get("frames", []) if isinstance(item, dict)]
    anchor_id = str(board.get("inject_anchor_id", "") or "").strip()
    if anchor_id:
        for index, item in enumerate(frames):
            if str(item.get("id", "") or "") == anchor_id:
                return frames[index:]
    selected = [item for item in frames if bool(item.get("selected", False))]
    return selected or frames


def _injection_start_slot(board: dict[str, Any], selected: list[dict[str, Any]]) -> int:
    frames = [item for item in board.get("frames", []) if isinstance(item, dict)]
    if not selected:
        return 0
    selected_id = str(selected[0].get("id", "") or "")
    for index, frame in enumerate(frames):
        if frame is selected[0] or (selected_id and str(frame.get("id", "") or "") == selected_id):
            return index
    return max(0, int(board.get("inject_anchor_index", 0) or 0))


def _load_storyboard_batch(paths: list[str], width: int, height: int) -> torch.Tensor:
    width = max(32, int(width) - int(width) % 32)
    height = max(32, int(height) - int(height) % 32)
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    images = []
    for filename in paths:
        path = _frame_path(filename)
        if not path:
            LOGGER.warning("NextFrame CineLinX skipped missing image: %s", filename)
            continue
        with Image.open(path) as source:
            image = source.convert("RGB")
            image = ImageOps.fit(image, (width, height), method=resampling, centering=(0.5, 0.5))
            array = np.asarray(image).astype(np.float32) / 255.0
        images.append(torch.from_numpy(array).unsqueeze(0))
    if not images:
        return torch.zeros((1, height, width, 3), dtype=torch.float32)
    return torch.cat(images, dim=0)


def _injection_timeline(
    frames: list[dict[str, Any]],
    slot_seconds: float,
    prompt_text: str,
    negative_prompt: str,
    start_slot: int = 0,
) -> dict[str, Any]:
    fps = 24
    slot_frames = max(1, int(round(max(0.1, float(slot_seconds)) * fps)))
    segments = []
    rows = []
    start_slot = max(0, int(start_slot))
    for index, frame in enumerate(frames):
        filename = str(frame.get("filename", "")).strip()
        prompt = str(frame.get("prompt", "") or "").strip()
        if str(frame.get("role", "")) == "source" and prompt == "Source frame":
            prompt = ""
        absolute_index = start_slot + index
        start = absolute_index * slot_frames
        segment = {
            "id": str(frame.get("id") or f"nextframe_{index + 1}"),
            "type": "image",
            "label": f"NextFrame {absolute_index + 1}",
            "start": start,
            "frame": start,
            "second": start / fps,
            "length": slot_frames,
            "ref": absolute_index + 1,
            "imageFile": filename,
            "image_file": filename,
            "imageTruthPath": filename,
            "fileName": os.path.basename(filename.replace("\\", "/")),
            "prompt": prompt,
            "local_prompt": prompt,
            "relay_prompt": prompt,
            "note": prompt,
            "use_guide": True,
            "use_prompt": bool(prompt),
            "guideStrength": 1.0,
            "guide_strength": 1.0,
            "force": 1.0,
            "source": "IAMCCS_NextFrameBuilder",
            "nextframe_selected": True,
        }
        segments.append(segment)
        rows.append(copy.deepcopy(segment))
    duration_seconds = max(0.1, (start_slot + len(segments)) * max(0.1, float(slot_seconds)))
    return {
        "schema": "iamccs.next_frame_builder.injection_timeline.v1",
        "source": "IAMCCS_NextFrameBuilder",
        "frame_rate": fps,
        "fps": fps,
        "duration_seconds": duration_seconds,
        "global_prompt": str(prompt_text or ""),
        "prompt": str(prompt_text or ""),
        "negative_prompt": str(negative_prompt or ""),
        "start_slot": start_slot,
        "segments": segments,
        "rows": rows,
        "audioSegments": [],
    }


class IAMCCS_NextFrameCommitPreview:
    """Persist a generated frame and expose preview metadata to the wrapper UI."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "source_image": ("STRING", {"default": ""}),
                "prompt_text": ("STRING", {"default": "", "multiline": True}),
                "storyboard_json": ("STRING", {"default": "{}", "multiline": True}),
                "session_id": ("STRING", {"default": "session"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "settings_json": ("STRING", {"default": "{}", "multiline": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "storyboard_json", "generated_filename")
    FUNCTION = "commit"
    CATEGORY = "IAMCCS/Storyboard/Backend"
    OUTPUT_NODE = True

    def commit(
        self,
        images,
        source_image,
        prompt_text,
        storyboard_json,
        session_id,
        seed,
        settings_json,
        prompt=None,
        extra_pnginfo=None,
    ):
        board = _safe_json_dict(storyboard_json)
        frames = board.get("frames")
        if not isinstance(frames, list):
            frames = []

        board["schema"] = "iamccs.next_frame_builder.storyboard.v1"
        board["session_id"] = _clean_session(session_id)
        board["updated_at"] = time.time()

        source_image = str(source_image or "").strip()
        if source_image and not any(
            str(frame.get("filename", "")) == source_image
            for frame in frames
            if isinstance(frame, dict)
        ):
            frames.append(
                {
                    "id": f"{board['session_id']}-source-{int(time.time() * 1000)}",
                    "number": len(frames) + 1,
                    "filename": source_image,
                    "source_image": "",
                    "prompt": "Source frame",
                    "seed": None,
                    "created_at": time.time(),
                    "role": "source",
                    "selected": True,
                }
            )

        input_dir = folder_paths.get_input_directory()
        os.makedirs(input_dir, exist_ok=True)
        session = board["session_id"]
        next_number = max(
            (
                int(frame.get("number", 0) or 0)
                for frame in frames
                if isinstance(frame, dict)
            ),
            default=0,
        ) + 1
        created_frames = []
        settings = _safe_json_dict(settings_json)

        for batch_number, image in enumerate(images):
            frame_number = next_number + batch_number
            stem = f"IAMCCS_NextFrame_{session}_{frame_number:03d}"
            filename = f"{stem}.png"
            suffix = 2
            while os.path.isfile(os.path.join(input_dir, filename)):
                filename = f"{stem}_{suffix:02d}.png"
                suffix += 1

            pixels = (255.0 * image.detach().cpu().numpy()).clip(0, 255).astype(np.uint8)
            pil_image = Image.fromarray(pixels)
            frame = {
                "id": f"{session}-{int(time.time() * 1000)}-{batch_number}",
                "number": frame_number,
                "filename": filename,
                "source_image": str(source_image or ""),
                "prompt": str(prompt_text or "").strip(),
                "seed": int(seed) + batch_number,
                "created_at": time.time(),
                "width": int(pil_image.width),
                "height": int(pil_image.height),
                "settings": settings,
                "selected": True,
            }
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("iamccs_next_frame", json.dumps(frame, ensure_ascii=False))
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt, ensure_ascii=False))
            if isinstance(extra_pnginfo, dict):
                for key, value in extra_pnginfo.items():
                    metadata.add_text(str(key), json.dumps(value, ensure_ascii=False))
            pil_image.save(os.path.join(input_dir, filename), pnginfo=metadata, compress_level=4)
            created_frames.append(frame)

        frames.extend(created_frames)
        board["frames"] = frames[-96:]
        board_json = json.dumps(board, ensure_ascii=False, separators=(",", ":"))
        generated_filename = created_frames[-1]["filename"] if created_frames else ""

        return {
            "ui": {
                "storyboard_json": [board_json],
                "generated_filename": [generated_filename],
                "frames": created_frames,
                "message": [f"Frame {created_frames[-1]['number']} ready" if created_frames else "No frame saved"],
            },
            "result": (images, board_json, generated_filename),
        }


class IAMCCS_NextFrameCineLinxBridge:
    """Publish selected NextFrame cards as a lightweight CineLinX contract."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "storyboard_json": ("STRING", {"default": "{}", "multiline": True}),
                "width": ("INT", {"default": 1920, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 1088, "min": 32, "max": 8192, "step": 32}),
                "inject_slot_seconds": ("FLOAT", {"default": 5.0, "min": 0.25, "max": 60.0, "step": 0.25}),
                "prompt_text": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE, "STRING")
    RETURN_NAMES = ("cine_linx", "report")
    FUNCTION = "publish"
    CATEGORY = "IAMCCS/Storyboard/Backend"

    def publish(
        self,
        storyboard_json,
        width,
        height,
        inject_slot_seconds,
        prompt_text,
        negative_prompt,
    ):
        from .iamccs_supernodes_linx import build_stage_linx_payload

        board = _safe_json_dict(storyboard_json)
        frames = _selected_storyboard_frames(board)
        start_slot = _injection_start_slot(board, frames)
        selected = []
        selected_paths = []
        for frame in frames:
            filename = str(frame.get("filename", "") or "").strip()
            if filename and _frame_path(filename):
                selected.append(frame)
                selected_paths.append(filename)
        timeline = _injection_timeline(
            selected,
            float(inject_slot_seconds),
            str(prompt_text or ""),
            str(negative_prompt or ""),
            start_slot,
        )
        timeline_json = json.dumps(timeline, ensure_ascii=False, separators=(",", ":"))
        image_batch = _load_storyboard_batch(selected_paths, int(width), int(height))
        injection = {
            "schema": "iamccs.next_frame_builder.cine_linx.v1",
            "source": "IAMCCS_NextFrameBuilder",
            "selected_paths": selected_paths,
            "selected_frame_ids": [str(frame.get("id", "")) for frame in selected],
            "start_slot": start_slot,
            "timeline_data": timeline_json,
            "width": int(width),
            "height": int(height),
            "slot_seconds": float(inject_slot_seconds),
            "prompt": str(prompt_text or ""),
            "negative_prompt": str(negative_prompt or ""),
        }
        report = (
            f"IAMCCS NextFrame CineLinX | selected={len(selected_paths)} | "
            f"slots={start_slot + 1}-{start_slot + len(timeline['segments'])} | {int(width)}x{int(height)}"
        )
        resources = {
            "iamccs_next_frame_injection": injection,
            "iamccs_next_frame_storyboard_json": str(storyboard_json or "{}"),
            "iamccs_next_frame_selected_paths": selected_paths,
            "cine_image_paths": json.dumps(selected_paths, ensure_ascii=False),
            "cine_multi_input": image_batch,
            "multi_input": image_batch,
            "cine_board_timeline_data": timeline_json,
            "cine_visual_segments_json": json.dumps(timeline["segments"], ensure_ascii=False),
            "cine_global_prompt": str(prompt_text or ""),
            "cine_negative_prompt": str(negative_prompt or ""),
            "cine_duration_seconds": float(timeline["duration_seconds"]),
            "cine_frame_rate": int(timeline["frame_rate"]),
            "cine_image_width": int(width),
            "cine_image_height": int(height),
            "cine_payload": {"next_frame_injection": injection, "timeline_data": timeline},
        }
        cine_linx = build_stage_linx_payload(
            None,
            "IAMCCS NextFrameBuilder",
            "next_frame_storyboard_injection",
            injection,
            report,
            downstream_stages=[
                "IAMCCS MiniMax H3 Shotboard",
                "IAMCCS Shotboard Planner V3",
                "IAMCCS MiniMax H3 Bridge",
            ],
            policies={
                "nextframe_injection": "explicit_replace_selected_slots",
                "selected_only": True,
                "generation_dependency": "independent_lightweight_branch",
            },
            outputs={"next_frame_injection": injection, "timeline_data": timeline_json},
            resources=resources,
        )
        cine_linx["mode"] = "iamccs_next_frame_injection"
        return cine_linx, report


class IAMCCS_NextFrameBuilder:
    """Professional next-scene storyboard UI backed by a dynamic Qwen graph."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("STRING", {"default": ""}),
                "prompt_text": (
                    "STRING",
                    {
                        "default": (
                            "Next Scene: The camera moves slightly forward to a medium shot as the same "
                            "character completes the next clear action beat. Preserve the exact identity, "
                            "face, hairstyle, wardrobe, body proportions, location geometry, color palette, "
                            "lighting direction and cinematic style of Image 1. Maintain spatial continuity "
                            "and realistic atmospheric depth."
                        ),
                        "multiline": True,
                        "dynamicPrompts": True,
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                    },
                ),
                "storyboard_json": ("STRING", {"default": "{}", "multiline": True}),
                "session_id": ("STRING", {"default": "storyboard"}),
                "run_token": ("STRING", {"default": "ready"}),
                "model_loader": (["GGUF", "Native UNET"], {"default": "GGUF"}),
                "gguf_model": ("STRING", {"default": DEFAULT_GGUF}),
                "native_model": ("STRING", {"default": DEFAULT_NATIVE}),
                "clip_model": ("STRING", {"default": DEFAULT_CLIP}),
                "vae_model": ("STRING", {"default": DEFAULT_VAE}),
                "lightning_lora": ("STRING", {"default": DEFAULT_LIGHTNING_LORA}),
                "lightning_strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "next_scene_lora": ("STRING", {"default": DEFAULT_NEXT_SCENE_LORA}),
                "next_scene_strength": ("FLOAT", {"default": 0.8, "min": -2.0, "max": 2.0, "step": 0.05}),
                "light_lora": ("STRING", {"default": DEFAULT_LIGHT_LORA}),
                "light_strength": ("FLOAT", {"default": 0.8, "min": -2.0, "max": 2.0, "step": 0.05}),
                "width": ("INT", {"default": 1920, "min": 256, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 1088, "min": 256, "max": 4096, "step": 16}),
                "seed": ("INT", {"default": 473146755093516, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "seed_mode": (["randomize", "fixed", "increment"], {"default": "randomize"}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.05}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 3.1, "min": 0.0, "max": 20.0, "step": 0.05}),
                "cfg_norm_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "reference_method": (
                    ["index_timestep_zero", "offset", "index", "uxo/uno"],
                    {"default": "index_timestep_zero"},
                ),
                "conditioning_megapixels": ("FLOAT", {"default": 3.0, "min": 0.25, "max": 16.0, "step": 0.25}),
                "decode_tile_size": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64}),
                "inject_slot_seconds": ("FLOAT", {"default": 5.0, "min": 0.25, "max": 60.0, "step": 0.25}),
            },
            "optional": {
                "reference_image_2": ("STRING", {"default": ""}),
                "reference_image_3": ("STRING", {"default": ""}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", SUPERNODE_LINX_TYPE)
    RETURN_NAMES = ("next_frame", "storyboard_json", "generated_filename", "cine_linx")
    FUNCTION = "build"
    CATEGORY = "IAMCCS/Storyboard"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Two-preview next-scene storyboard builder. It expands to the Qwen Image Edit 2511 "
        "backend from the IAMCCS reference workflow and commits each result as the next reusable frame."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, source_image, prompt_text, **kwargs):
        if not str(source_image or "").strip():
            return "Load a source image in IAMCCS NextFrameBuilder before generating."
        if not folder_paths.exists_annotated_filepath(str(source_image)):
            return f"Source image is not available in ComfyUI input: {source_image}"
        if not str(prompt_text or "").strip():
            return "Write the next-scene prompt before generating."
        return True

    @classmethod
    def IS_CHANGED(cls, run_token, **kwargs):
        return str(run_token or time.time_ns())

    def build(
        self,
        source_image,
        prompt_text,
        negative_prompt,
        storyboard_json,
        session_id,
        run_token,
        model_loader,
        gguf_model,
        native_model,
        clip_model,
        vae_model,
        lightning_lora,
        lightning_strength,
        next_scene_lora,
        next_scene_strength,
        light_lora,
        light_strength,
        width,
        height,
        seed,
        seed_mode,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        shift,
        cfg_norm_strength,
        reference_method,
        conditioning_megapixels,
        decode_tile_size,
        inject_slot_seconds,
        reference_image_2="",
        reference_image_3="",
        unique_id=None,
    ):
        graph = GraphBuilder()

        if model_loader == "Native UNET":
            loader = graph.node("UNETLoader", unet_name=native_model, weight_dtype="default")
        else:
            loader = graph.node("UnetLoaderGGUF", unet_name=gguf_model)
        model = loader.out(0)

        for lora_name, strength in (
            (lightning_lora, lightning_strength),
            (next_scene_lora, next_scene_strength),
            (light_lora, light_strength),
        ):
            if str(lora_name or "").strip() and abs(float(strength)) > 1e-8:
                lora = graph.node(
                    "LoraLoaderModelOnly",
                    model=model,
                    lora_name=str(lora_name),
                    strength_model=float(strength),
                )
                model = lora.out(0)

        sampling = graph.node("ModelSamplingAuraFlow", model=model, shift=float(shift))
        cfg_norm = graph.node(
            "CFGNorm",
            model=sampling.out(0),
            strength=float(cfg_norm_strength),
            pre_cfg=False,
        )
        clip = graph.node("CLIPLoader", clip_name=clip_model, type="qwen_image", device="default")
        vae = graph.node("VAELoader", vae_name=vae_model)

        image_nodes = []
        for filename in (source_image, reference_image_2, reference_image_3):
            if not str(filename or "").strip():
                continue
            loaded = graph.node("LoadImage", image=str(filename))
            scaled = graph.node(
                "ImageScaleToTotalPixels",
                image=loaded.out(0),
                upscale_method="lanczos",
                megapixels=float(conditioning_megapixels),
                resolution_steps=1,
            )
            image_nodes.append(scaled)

        encode_inputs = {
            "clip": clip.out(0),
            "vae": vae.out(0),
            "prompt": str(prompt_text),
        }
        negative_inputs = {"clip": clip.out(0), "vae": vae.out(0), "prompt": str(negative_prompt or "")}
        for index, image_node in enumerate(image_nodes[:3], start=1):
            encode_inputs[f"image{index}"] = image_node.out(0)
            negative_inputs[f"image{index}"] = image_node.out(0)

        positive = graph.node("TextEncodeQwenImageEditPlus", **encode_inputs)
        negative = graph.node("TextEncodeQwenImageEditPlus", **negative_inputs)
        positive_method = graph.node(
            "FluxKontextMultiReferenceLatentMethod",
            conditioning=positive.out(0),
            reference_latents_method=reference_method,
        )
        negative_method = graph.node(
            "FluxKontextMultiReferenceLatentMethod",
            conditioning=negative.out(0),
            reference_latents_method=reference_method,
        )
        latent = graph.node(
            "EmptySD3LatentImage",
            width=int(width),
            height=int(height),
            batch_size=1,
        )
        sampler = graph.node(
            "KSampler",
            model=cfg_norm.out(0),
            positive=positive_method.out(0),
            negative=negative_method.out(0),
            latent_image=latent.out(0),
            seed=int(seed),
            steps=int(steps),
            cfg=float(cfg),
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=float(denoise),
        )
        decoded = graph.node(
            "VAEDecodeTiled",
            samples=sampler.out(0),
            vae=vae.out(0),
            tile_size=int(decode_tile_size),
            overlap=64,
            temporal_size=64,
            temporal_overlap=8,
        )

        settings = {
            "model_loader": model_loader,
            "model": native_model if model_loader == "Native UNET" else gguf_model,
            "clip": clip_model,
            "vae": vae_model,
            "loras": [
                {"name": lightning_lora, "strength": lightning_strength},
                {"name": next_scene_lora, "strength": next_scene_strength},
                {"name": light_lora, "strength": light_strength},
            ],
            "width": int(width),
            "height": int(height),
            "steps": int(steps),
            "cfg": float(cfg),
            "sampler": sampler_name,
            "scheduler": scheduler,
            "denoise": float(denoise),
            "shift": float(shift),
            "reference_method": reference_method,
            "negative_prompt": str(negative_prompt or ""),
        }
        commit = graph.node(
            "IAMCCS_NextFrameCommitPreview",
            images=decoded.out(0),
            source_image=str(source_image),
            prompt_text=str(prompt_text),
            storyboard_json=str(storyboard_json or "{}"),
            session_id=str(session_id or "storyboard"),
            seed=int(seed),
            settings_json=json.dumps(settings, ensure_ascii=False, separators=(",", ":")),
        )
        if unique_id is not None:
            commit.set_override_display_id(str(unique_id))

        cine_linx = graph.node(
            "IAMCCS_NextFrameCineLinxBridge",
            storyboard_json=str(storyboard_json or "{}"),
            width=int(width),
            height=int(height),
            inject_slot_seconds=float(inject_slot_seconds),
            prompt_text=str(prompt_text or ""),
            negative_prompt=str(negative_prompt or ""),
        )

        LOGGER.info(
            "Built Qwen 2511 next-frame graph: source=%s size=%sx%s seed=%s token=%s",
            source_image,
            width,
            height,
            seed,
            run_token,
        )
        return {
            "result": (commit.out(0), commit.out(1), commit.out(2), cine_linx.out(0)),
            "expand": graph.finalize(),
        }


NODE_CLASS_MAPPINGS = {
    "IAMCCS_NextFrameBuilder": IAMCCS_NextFrameBuilder,
    "IAMCCS_NextFrameCommitPreview": IAMCCS_NextFrameCommitPreview,
    "IAMCCS_NextFrameCineLinxBridge": IAMCCS_NextFrameCineLinxBridge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_NextFrameBuilder": "IAMCCS NextFrameBuilder",
    "IAMCCS_NextFrameCommitPreview": "IAMCCS NextFrame Commit Preview",
    "IAMCCS_NextFrameCineLinxBridge": "IAMCCS NextFrame CineLinX Bridge",
}
