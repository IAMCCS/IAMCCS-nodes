"""Local H3 asset inspection and reviewable settings proposals. No model loading."""
from __future__ import annotations

import functools
import hashlib
import json
import math
import re
import struct
from pathlib import Path


def task_family(task):
    task = str(task).lower()
    if task in {"v2va_controlnet", "controlnet_v2v", "h3_fun_controlnet"}:
        return "fl2va"
    return "ref2va" if task.startswith("ref2va") or task in {
        "ref2vid_lipsync", "v2va_object_swap", "v2va_face_swap"
    } else "fl2va"


def named_family(text):
    text = str(text).lower()
    families = [family for token, family in (("ref2v", "ref2va"), ("fl2v", "fl2va")) if token in text]
    return families[0] if len(families) == 1 else ""


@functools.lru_cache(maxsize=128)
def inspect_header(path_text, mtime_ns, size):
    del mtime_ns
    with open(path_text, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        if length > min(size - 8, 32 * 1024 * 1024):
            raise ValueError("Invalid safetensors header length")
        raw = handle.read(length)
        header = json.loads(raw)
    metadata = header.pop("__metadata__", {})
    keys = list(header)
    lora = [key for key in keys if any(marker in key.lower() for marker in (".lora_a.", ".lora_b.", ".lora_down", ".lora_up"))]
    native = any(key.startswith(("blocks.", "diffusion_model.blocks.")) for key in lora)
    return {"metadata": metadata, "native": native, "lora": bool(lora),
            "pdd": all(any(f"final_layer.{kind}_out" in key and ("reshape_weight" in key or "set_weight" in key) for key in keys) for kind in ("video", "audio")),
            "header_sha256": hashlib.sha256(raw).hexdigest(), "bytes": size}


def describe_asset(name, path):
    result = {"name": name, "family": named_family(name), "role": "unknown", "recipe": None,
              "native": False, "conflict": False, "error": "", "bytes": 0,
              "declared_steps": None}
    try:
        stat = Path(path).stat()
        result.update(inspect_header(str(path), stat.st_mtime_ns, stat.st_size))
    except (OSError, ValueError, struct.error, TypeError) as exc:
        result["error"] = str(exc)
        return result
    meta = result["metadata"]
    # Prefer an explicit safetensors declaration.  Filename parsing is only a
    # compatibility fallback for community adapters whose metadata predates a
    # common schema; it never selects an asset by name or enables a LoRA.
    for key in ("steps", "inference_steps", "sampling_steps", "num_steps"):
        try:
            declared_steps = int(str(meta.get(key, "")).strip())
        except (TypeError, ValueError):
            continue
        if 1 <= declared_steps <= 100:
            result["declared_steps"] = declared_steps
            break
    if result["declared_steps"] is None:
        match = re.search(r"(?<!\d)(\d{1,2})[\s_.-]*steps?(?!\d)", str(name), re.IGNORECASE)
        if match:
            declared_steps = int(match.group(1))
            if 1 <= declared_steps <= 100:
                result["declared_steps"] = declared_steps
    declared = named_family(meta.get("base_model", meta.get("ss_base_model_version", "")))
    result["conflict"] = bool(declared and result["family"] and declared != result["family"])
    result["family"] = declared or result["family"]
    lower = name.lower()
    if any(token in lower for token in ("cinema", "360", "studio1939", "mystic", "vbvr")):
        result["role"] = "style"
    elif result["pdd"]:
        result.update(role="pdd", recipe="pdd_8")
    elif meta.get("converted_by") == "lora_convert_h3" and meta.get("adaln_mode") == "drop":
        result.update(role="fasth3", recipe="fasth3_6", family=result["family"] or "fl2va")
    elif "turbo" in lower and result["native"]:
        result["role"] = "turbo"
        if "768p" in lower and "4step" in lower and "_sla_" in lower and "ComfyUI generic LoRA" in str(meta.get("target_format", "")):
            result.update(recipe="lightx_768_sla_4", requires_backend="H3SLAAttention")
        elif "768p" in lower and "8step" in lower and "ComfyUI generic LoRA" in str(meta.get("target_format", "")):
            result["recipe"] = "lightx_768_8"
        elif "lightx2v" in lower and "4step" in lower and "v0.1" in lower and meta.get("converted_layout") == "comfyui_minimax_h3":
            result["recipe"] = "lightx_544_4"
    if result["conflict"]:
        result["error"] = "Filename and base-model metadata disagree; verify provenance before use."
        result["recipe"] = None
    return result


def recipes():
    return json.loads((Path(__file__).parent / "assets" / "iamccs_h3_recipes.json").read_text(encoding="utf-8"))


def preferred_sla_defaults(assets, task, node_names):
    """Defaults for newly created settings / explicit mode presets, never saved values."""
    if task not in {"auto_from_timeline", "t2va", "i2va", "fl2va"} or "H3SLAAttention" not in node_names:
        return {}
    asset = next((a for a in assets if a.get("recipe") == "lightx_768_sla_4"
                  and a.get("family") == "fl2va" and a.get("native") and not a.get("error")), None)
    if not asset:
        return {}
    return {**recipes()["lightx_768_sla_4"]["values"], "turbo_lora_name": asset["name"], "performance_profile": "custom"}


def validate_speed_asset(asset, task, role="turbo"):
    if asset.get("error") or not asset.get("native"):
        raise ValueError(f"H3 {role}: {asset['name']}: {asset.get('error') or 'requires a native ComfyUI adapter'}")
    if asset.get("role") == "style":
        raise ValueError(f"{asset['name']} is a style LoRA; use the secondary creative slot.")
    if asset.get("role") not in {role, "unknown"}:
        raise ValueError(f"{asset['name']} requires the {asset['role']} backend, not {role}.")
    if asset.get("family") and asset["family"] != task_family(task):
        raise ValueError(f"H3 {role}: {asset['name']} belongs to {asset['family']}, task requires {task_family(task)}.")


def validate_settings(settings, task=None):
    import folder_paths
    if task is None:
        task = settings.get("task_mode", "t2va")
        if settings.get("audio_mode") == "h3_ref2va_audio" and not str(task).startswith("longvid"):
            task = "ref2va"
    acceleration = settings.get("acceleration", "native")
    if acceleration == "matlowai_fused_turbo_manual_sigma":
        name = settings.get("fused_turbo_model_name", "")
        if not name or not folder_paths.get_full_path("diffusion_models", name):
            raise ValueError("Fused Fast H3: select an installed fused checkpoint in Fused model, then analyze again.")
        supported = {"t2va", "i2va", "fl2va", "ref2va", "auto_from_timeline"} if "convrot" in str(name).lower() else {"t2va", "auto_from_timeline"}
        if str(task) not in supported or settings.get("secondary_lora_enabled"):
            raise ValueError("This fused checkpoint requires a supported task and no secondary style LoRA.")
    fields = []
    if settings.get("turbo_mode", "off") != "off":
        fields.append(("turbo_lora_name", "turbo"))
    if acceleration in {"pdd_native_8step", "iamccs_progressive_pdd_2stage"}:
        fields.append(("pdd_lora_name", "pdd"))
    if acceleration == "fasth3_dense_6step":
        fields.append(("turbo_lora_name", "fasth3"))
    for field, role in fields:
        name = settings.get(field, "")
        path = folder_paths.get_full_path("loras", name) if name else None
        try:
            if not path:
                raise ValueError(f"H3 {role}: missing adapter '{name}'.")
            validate_speed_asset(describe_asset(name, path), task, role)
        except ValueError as original:
            # Workflows saved before the Settings PRO asset filter may still
            # point FastH3 at the repository's generic diffusers adapter, or
            # PDD at the other model family.  Resolve only the explicitly
            # selected engine to a locally installed native asset of the same
            # role/family; never fall through to a style LoRA.
            compatible = None
            for candidate in folder_paths.get_filename_list("loras"):
                candidate_path = folder_paths.get_full_path("loras", candidate)
                if not candidate_path:
                    continue
                descriptor = describe_asset(candidate, candidate_path)
                try:
                    validate_speed_asset(descriptor, task, role)
                except ValueError:
                    continue
                if descriptor.get("role") == role:
                    compatible = candidate
                    break
            if not compatible:
                raise original
            settings[field] = compatible


def exact_backend_status():
    try:
        from h3_optimizations.memory.embedding import _validate_upstream_forward
        from comfy.ldm.minimax.model import MiniMaxH3Model
        _validate_upstream_forward(MiniMaxH3Model._forward)
    except (ImportError, RuntimeError) as exc:
        return {"available": False, "reason": str(exc)}
    return {"available": True, "reason": "Installed H3 core matches the memory backend compatibility guard."}


def runtime_inventory():
    # Optional runtime imports keep header inspection and proposal tests CPU-only.
    import folder_paths
    import nodes
    import psutil
    import torch
    import comfy.model_management as mm

    device = mm.get_torch_device()
    ram = psutil.virtual_memory()
    hardware = {"device": str(device), "name": str(device), "ram_total": ram.total,
                "ram_free": ram.available, "vram_total": 0, "vram_free": 0,
                "torch": torch.__version__, "capability": None, "nvidia": bool(torch.version.cuda and not torch.version.hip)}
    if device.type != "cpu":
        hardware.update(vram_total=int(mm.get_total_memory(device)), vram_free=int(mm.get_free_memory(device)))
    if device.type == "cuda":
        hardware.update(name=torch.cuda.get_device_name(device), capability=list(torch.cuda.get_device_capability(device)))
    assets = []
    for name in folder_paths.get_filename_list("loras"):
        if "h3" in name.lower() or "minimax" in name.lower():
            assets.append(describe_asset(name, folder_paths.get_full_path("loras", name)))
    models = []
    for category in ("diffusion_models", "text_encoders", "vae", "latent_upscale_models", "checkpoints"):
        for name in folder_paths.get_filename_list(category):
            if any(token in name.lower() for token in ("h3", "minimax", "qwen3", "clipproj", "sam3")):
                path = folder_paths.get_full_path(category, name)
                models.append({"category": category, "name": name, "bytes": Path(path).stat().st_size})
    return {"hardware": hardware, "assets": assets, "models": models, "nodes": sorted(nodes.NODE_CLASS_MAPPINGS),
            "exact_backend": exact_backend_status()}


def inventory_revision(inventory):
    stable = {"hardware": {k: v for k, v in inventory["hardware"].items() if not k.endswith("_free")},
              "assets": inventory["assets"], "models": inventory["models"], "nodes": inventory["nodes"],
              "exact_backend": inventory.get("exact_backend")}
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def propose(request, inventory):
    settings = request.get("settings", {})
    task = str(settings.get("task_mode", "t2va"))
    family = task_family("ref2va" if settings.get("audio_mode") == "h3_ref2va_audio" and not task.startswith("longvid") else task)
    hw = inventory["hardware"]
    nodes = set(inventory["nodes"])
    connected = [item for item in request.get("connected", []) if item.get("mode", 0) == 0]
    active_files = [str(value) for item in connected for value in item.get("values", {}).values() if isinstance(value, str)]
    selected_models = [m for m in inventory["models"] if m["name"] in active_files
                       and (m["category"] != "diffusion_models" or named_family(m["name"]) in {"", family})]
    types = {item.get("type") for item in connected}
    warnings = []
    compatibility_notes = []
    target = int(request.get("target_long_edge", 1280))
    if not 256 <= target <= 5760:
        raise ValueError("Target long edge must be between 256 and 5760 (current H3 canvas limit).")
    width, height = max(32, int(settings.get("width", 768))), max(32, int(settings.get("height", 448)))
    ratio = target / max(width, height)
    native = [max(256, int(math.ceil(axis * ratio / 32)) * 32) for axis in (width, height)]
    target_size = [max(2, round(axis * ratio / 2) * 2) for axis in (width, height)]
    frames = max(22, min(int(settings.get("motion_context_window_frames", 124)), math.ceil(float(settings.get("duration_seconds", 5.17)) * 24)))
    tokens = math.prod(native) / (32 * 32) * math.ceil(frames / 17)
    # Workspace proxy (QKV, FFN and residuals), deliberately not a peak-memory promise.
    workspace_bytes = tokens * 6144 * 2 * 64
    device_free = hw["vram_free"] if hw["vram_total"] else hw["ram_free"]
    budget = max(0, min(device_free, hw["ram_free"]) * 0.5)
    selected_bytes = sum(m["bytes"] for m in selected_models)
    if selected_bytes > hw["ram_free"] + device_free:
        warnings.append("Connected model files exceed currently free RAM + device memory. Offload or paging may dominate; close other workloads or select smaller loaders before benchmarking.")
    scale = min(1.0, math.sqrt(budget / max(1, workspace_bytes)))
    minimum_scale = min(1.0, 256 / min(native))
    modest = [max(256, int(axis * max(minimum_scale, scale) // 32) * 32) for axis in native]
    exact = "H3MemoryOptimization" in nodes and inventory.get("exact_backend", {"available": True})["available"]
    sage = "MiniMaxH3MemoryEfficientSageAttentionPatch" in nodes and hw["capability"] is not None
    memory = "h3_exact" if exact else "h3_sage" if sage else "native"
    if "H3MemoryOptimization" in nodes and not exact:
        message = "H3 Exact was excluded because the installed core failed its compatibility guard; alternative: " + memory + ". " + inventory["exact_backend"]["reason"]
        (warnings if settings.get("acceleration") == "h3_exact" else compatibility_notes).append(message)
    rows = max(256, min(65536, int(budget * 0.08 / (6144 * 2 * 16) // 256) * 256))
    common = {"performance_profile": "custom", "h3_exact_profile": "custom"}
    if memory == "h3_exact":
        common.update(h3_exact_chunk_rows=rows, h3_exact_precision_mode="Preserve native",
                      h3_exact_qkv_streaming="Auto", h3_exact_attention_memory="Standard")
    eligible = [a for a in inventory["assets"] if a.get("recipe") and a.get("native") and not a.get("error") and a.get("family") == family
                and (not a.get("requires_backend") or a["requires_backend"] in nodes)]
    book = recipes()
    current_name = settings.get("turbo_lora_name") if settings.get("turbo_mode", "off") != "off" else settings.get("pdd_lora_name")
    eligible.sort(key=lambda a: (a.get("recipe") != "lightx_768_sla_4", a["name"] != current_name, a["role"] != "turbo",
                                abs((768 if a["recipe"].startswith("lightx_768") else 544) - min(native)), a["name"]))
    choices = []
    route_ok = "IAMCCS_MiniMaxH3UniversalRTXFinalR42" in types and "RTXVideoSuperResolution" in nodes and hw.get("nvidia", False)
    if task == "v2va_face_swap":
        # Mode-specific Face Swap values are intentionally owned by the
        # Settings MODE-SPECIFIC CONTRACT panel. Hardware proposals must not
        # rewrite creative/tracking controls.
        warnings.append("Face Swap adds SAM3 tracking, BiRefNet reference preparation and a tracked crop. These stages have a separate memory cost, so the ordinary full-frame estimate is not calibrated for this mode.")
        if settings.get("upscale_enabled") and settings.get("upscale_mode") not in {"off", "rtx_final"}:
            common.update(upscale_enabled=False, upscale_mode="off")
            warnings.append("The current latent delivery does not match a face crop. Proposals explicitly switch to native output; RTX final is available when connected.")
    fused = settings.get("acceleration") == "matlowai_fused_turbo_manual_sigma"
    fused_modes = {"t2va", "i2va", "fl2va", "ref2va", "auto_from_timeline"} if "convrot" in str(settings.get("fused_turbo_model_name", "")).lower() else {"t2va", "auto_from_timeline"}
    if fused and (task not in fused_modes or settings.get("secondary_lora_enabled")):
        warnings.append("The current fused branch cannot satisfy this task/style selection. Proposals explicitly switch to the connected native base.")
        fused = False
    fused_model = settings.get("fused_turbo_model_name", "")
    installed_diffusion = {m["name"] for m in inventory["models"] if m["category"] == "diffusion_models"}
    if fused and fused_model not in installed_diffusion:
        candidates = [name for name in installed_diffusion if "fused" in name.lower() and "h3" in name.lower()]
        if len(candidates) == 1:
            fused_model = candidates[0]
            warnings.append("The current Fused model is empty or missing. Fused proposals explicitly select the sole installed H3 fused checkpoint; review its filename before applying.")
        else:
            warnings.append("Select an installed fused checkpoint before requesting a Fused proposal. Native-base proposals remain available.")
            fused = False
    if not exact and not sage:
        warnings.append("No registered H3 memory patch. Native execution may exceed available memory.")
    used_assets = set()
    if settings.get("turbo_mode", "off") != "off":
        used_assets.add(settings.get("turbo_lora_name"))
    if settings.get("acceleration") in {"pdd_native_8step", "iamccs_progressive_pdd_2stage"}:
        used_assets.add(settings.get("pdd_lora_name"))
    if settings.get("secondary_lora_enabled"):
        used_assets.add(settings.get("secondary_lora_name"))
    for asset in inventory["assets"]:
        if asset.get("error"):
            message = f"{asset['name']}: {asset['error']}"
            if asset["name"] in used_assets:
                warnings.append("Selected LoRA is invalid: " + message)
            else:
                compatibility_notes.append("Unselected LoRA excluded from proposals: " + message)
    if any("w4a8" in model['name'].lower() for model in selected_models) and eligible:
        warnings.append("W4A8 + distilled adapters: quality needs an A/B render; loading alone does not validate it.")
    if any("Sol" in name for name in nodes):
        message = "Sol is available as an experimental option; its kernel and output quality have not been measured on this system."
        (warnings if "sol" in str(settings.get("acceleration", "")).lower() else compatibility_notes).append(message)
    if not route_ok:
        message = "This workflow has no connected RTX delivery branch. Proposals therefore keep a native output target."
        if settings.get("upscale_enabled") and settings.get("upscale_mode") == "rtx_final":
            warnings.append("RTX Final is selected, but its branch/backend is not available in the active workflow. Select native output or connect the RTX branch.")
        else:
            compatibility_notes.append(message)
    for label, canvas, use_speed in (("Native detail", native, True), ("Lower sampling load + delivery", modest, True), ("Native model baseline", native, False)):
        needs_delivery = max(canvas) < target
        if needs_delivery and not route_ok:
            continue
        values = {**common, "width": canvas[0], "height": canvas[1], "image_width": canvas[0], "image_height": canvas[1]}
        notes = ["Estimate only: no calibrated render time or peak VRAM yet.", "Audio, duration, cuts, creative LoRA and connected model loaders are preserved."]
        if fused and use_speed:
            sigma = settings.get("fused_turbo_sigma_preset", "4_step")
            sigma = sigma if sigma in {"4_step", "6_step", "8_step"} else "4_step"
            values.update(acceleration="matlowai_fused_turbo_manual_sigma", fused_turbo_model_name=fused_model,
                          fused_turbo_sigma_preset=sigma, turbo_mode="off", steps=int(sigma.split("_")[0]),
                          sampler_name="res_multistep" if "convrot" in str(fused_model).lower() else "euler", scheduler="simple", denoise=1.0, shift_video=12.0, shift_audio=3.0)
            notes.append("Complete fused checkpoint + matching manual-sigma recipe. Filename identifies the fused asset; no GPU-specific filename is assumed.")
        elif eligible and use_speed:
            asset = eligible[0]
            recipe = book[asset["recipe"]]
            values.update(recipe["values"])
            values["pdd_lora_name" if asset["role"] == "pdd" else "turbo_lora_name"] = asset["name"]
            if asset["role"] == "turbo" and not asset.get("requires_backend"):
                values["acceleration"] = memory
            notes.append(f"Recipe {asset['recipe']}: {asset['name']}. {recipe['note']}")
            notes.append(recipe["source"])
        else:
            values.update(acceleration=memory, turbo_mode="off", steps=max(20, int(settings.get("steps", 20))), sampler_name="euler", scheduler="simple", shift_video=12.0, shift_audio=3.0, denoise=1.0)
            if not eligible:
                notes.append("No verified task-matched speed recipe; no substitute adapter is selected.")
        if values.get("acceleration", settings.get("acceleration")) != "h3_exact":
            values = {key: value for key, value in values.items() if not key.startswith("h3_exact_")}
        if needs_delivery:
            values.update(upscale_enabled=True, upscale_mode="rtx_final", upscale_width=target_size[0], upscale_height=target_size[1])
            notes.append("RTX delivery adds a separate GPU stage; final size is not equivalent to native detail.")
        elif settings.get("upscale_enabled"):
            # Do not silently discard an authored enhancement route.
            values.update(upscale_width=max(native[0], int(settings.get("upscale_width", native[0]))),
                          upscale_height=max(native[1], int(settings.get("upscale_height", native[1]))))
        if fused and not use_speed:
            notes.append("Uses the externally connected native base in place of the fused branch.")
        choices.append({"id": str(len(choices)), "label": label, "values": values, "notes": notes,
                        "sampling_load_relative": round(math.prod(canvas) / (width * height), 2)})
    return {"proposals": choices, "warnings": warnings, "compatibility_notes": compatibility_notes,
            "memory_backend": memory, "rtx_delivery_available": route_ok, "family": family,
            "workload": {"frames_per_window": frames, "workspace_proxy_bytes": round(workspace_bytes),
                         "budget_proxy_bytes": round(budget), "connected_model_file_bytes": selected_bytes,
                         "target_size": target_size}}


def advice(request):
    inventory = runtime_inventory()
    return {**propose(request, inventory), **inventory, "inventory_revision": inventory_revision(inventory)}
