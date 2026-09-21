# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later

"""Safe, metadata-only library for H3 continuation and RefMod assets."""

from __future__ import annotations

import json
from pathlib import Path
import re

import folder_paths
from safetensors import safe_open
import torch
from PIL import Image

from .iamccs_h3_continuous_av.nodes import (
    H3ContinuousAnalyzeHandoverV11,
    H3ContinuousContinueV11,
    H3ContinuousLoadLatent,
    H3ContinuousSaveLatent,
    H3ContinuousStartV11,
    H3ContinuousStitchOutputV11,
)
from .iamccs_minimax_h3_atomic_backend import SUPERNODE_LINX_TYPE, _resolve_shotplan


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


def _cache_roots():
    """Return only IAMCCS-owned disposable render caches.

    Model folders are deliberately excluded.  Each root has an extension
    allow-list so the web purge API cannot become a general file deleter.
    """
    output = Path(folder_paths.get_output_directory()).resolve()
    h3 = output / "minimax_h3_shotboard"
    return [
        {
            "id": "h3_motion_context",
            "label": "H3 Motion Context AV",
            "root": (h3 / "motion_context").resolve(),
            "extensions": {".pt", ".tmp"},
        },
        {
            "id": "h3_latent_tail",
            "label": "LongVid latent tail",
            "root": (h3 / "latent_tail_experimental").resolve(),
            "extensions": {".pt", ".tmp"},
        },
        {
            "id": "h3_bridge_frames",
            "label": "LongVid bridge frames",
            "root": (h3 / "bridges").resolve(),
            "extensions": {".png", ".tmp"},
        },
        {
            "id": "h3_latent_upres",
            "label": "H3 2-pass latent-upres chunks",
            "root": (h3 / "r38_latent_upres").resolve(),
            "extensions": {".pt", ".tmp"},
        },
        {
            "id": "h3_disk_upscale",
            "label": "H3 Full-HD disk-upscale checkpoints",
            "root": (output / "IAMCCS" / "H3_DISK_UPSCALE").resolve(),
            "extensions": {".safetensors", ".json", ".tmp"},
            "required_part": "checkpoints",
        },
        {
            "id": "h3_latent_go_ahead",
            "label": "H3 LatentGoAhead interval cache",
            "root": (output / "IAMCCS" / "LatentGoAhead").resolve(),
            "extensions": {".safetensors", ".tmp"},
        },
        {
            "id": "iamccs_motion_bridges",
            "label": "IAMCCS motion bridge cache",
            "root": (output / "motion_bridges").resolve(),
            "extensions": {".safetensors", ".png", ".tmp"},
        },
        {
            "id": "ltx2_disk_bridges",
            "label": "LTX disk bridge cache",
            "root": (output / "iamccs_ltx2_bridges").resolve(),
            "extensions": {".pt", ".safetensors", ".json", ".png", ".tmp"},
        },
    ]


def _roots(kind):
    if kind == "continuation":
        return [(Path(folder_paths.get_output_directory()) / "IAMCCS" / "MiniMaxH3" / "CONTINUATION").resolve()]
    if kind == "refmod":
        roots = [Path(folder_paths.models_dir) / "refmods"]
        try:
            roots.extend(Path(p) for p in folder_paths.get_folder_paths("refmods"))
        except KeyError:
            pass
        return list(dict.fromkeys(p.resolve() for p in roots))
    raise ValueError("Unknown H3 asset library")


def _asset(kind, root_index, relative):
    roots = _roots(kind)
    if not isinstance(root_index, int) or not 0 <= root_index < len(roots):
        raise ValueError("Unknown asset root")
    relative = Path(str(relative).replace("\\", "/"))
    if relative.is_absolute() or relative.suffix.lower() != ".safetensors" or ".." in relative.parts:
        raise ValueError("Asset must be a relative .safetensors path")
    root = roots[root_index]
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Asset is outside its library")
    return target


def _cache_asset(root_id, relative):
    spec = next((item for item in _cache_roots() if item["id"] == root_id), None)
    if spec is None:
        raise ValueError("Unknown IAMCCS cache root")
    relative = Path(str(relative).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Cache item must use a relative path")
    if relative.suffix.lower() not in spec["extensions"]:
        raise ValueError("That file type is not purgeable in this cache")
    required_part = spec.get("required_part")
    if required_part and required_part not in relative.parts:
        raise ValueError("That path is not a purgeable checkpoint cache")
    root = spec["root"]
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Cache item is outside its IAMCCS cache root")
    return spec, target


def list_assets(kind):
    items = []
    for root_index, root in enumerate(_roots(kind)):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.safetensors"))[:500]:
            if not path.resolve().is_relative_to(root):
                continue
            try:
                with safe_open(str(path), framework="pt", device="cpu") as reader:
                    meta = reader.metadata() or {}
            except Exception:
                continue
            if kind == "refmod":
                try:
                    refmod = json.loads(meta.get("refmod_meta", "{}"))
                except (TypeError, ValueError):
                    refmod = {}
                if not refmod and path.with_suffix(".json").is_file():
                    try:
                        refmod = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        refmod = {}
                if refmod.get("kind") not in {"image", "video", "audio", "bundle"}:
                    continue
                detail = refmod.get("kind", "refmod")
                facts = [
                    f"kind={refmod.get('kind', 'refmod')}",
                    f"source={refmod.get('source', refmod.get('source_type', 'saved latent'))}",
                ]
                prompt = str(refmod.get("prompt", refmod.get("description", "")) or "").strip()
            else:
                if not str(meta.get("format", "")).startswith("h3_continuous_av"):
                    continue
                visible = meta.get("visible_frame_count", "")
                padding = meta.get("technical_padding_frames", "")
                facts = [f"clip={meta.get('clip_index', '?')}", f"fps={meta.get('fps', '?')}"]
                prompt = ""
                chain_path = path.with_suffix(".chain.json")
                if chain_path.is_file():
                    try:
                        chain = json.loads(chain_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        chain = {}
                    segments = chain.get("segments") if isinstance(chain, dict) else None
                    if isinstance(segments, list) and segments:
                        media = Path(str(segments[-1]))
                        if not media.is_absolute():
                            media = Path(folder_paths.get_output_directory()) / media
                        sidecar = Path(str(media) + ".iamccs.json")
                        try:
                            payload = json.loads(sidecar.read_text(encoding="utf-8"))
                        except (OSError, ValueError):
                            payload = {}
                        generation = payload.get("generation") if isinstance(payload, dict) else {}
                        if not visible and isinstance(payload, dict) and payload.get("frame_count"):
                            visible = str(payload.get("frame_count"))
                        shotplan = generation.get("shotplan") if isinstance(generation, dict) else {}
                        if isinstance(generation, dict):
                            if generation.get("render_id"):
                                facts.append(f"render={generation['render_id']}")
                            if generation.get("stage"):
                                facts.append(f"stage={generation['stage']}")
                        if isinstance(shotplan, dict):
                            mode = shotplan.get("task_mode")
                            width, height = shotplan.get("width"), shotplan.get("height")
                            if mode:
                                facts.append(f"mode={mode}")
                            if width and height:
                                facts.append(f"resolution={width}x{height}")
                            chunks = shotplan.get("chunks")
                            if isinstance(chunks, list) and chunks:
                                prompt = str(chunks[-1].get("prompt", "") or "").strip()
                if not visible:
                    visible = meta.get("frame_count", "?")
                if not padding:
                    try:
                        padding = str(max(
                            0,
                            int(meta.get("frame_count", 0))
                            - int(meta.get("head_context_frames", 0))
                            - int(visible),
                        ))
                    except (TypeError, ValueError):
                        padding = "?"
                detail = (
                    f"{meta.get('frame_count', '?')} technical frames · {visible} visible · "
                    f"{meta.get('head_context_frames', '0')} head context · {padding} grid pad"
                )
            relative = path.relative_to(root).as_posix()
            preview = path.with_suffix(".png").is_file()
            items.append({
                "root": root_index, "path": relative, "name": path.stem,
                "detail": detail, "preview": preview,
                "facts": facts,
                "prompt": prompt[:600],
                "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
            })
    return items


def list_cache_files():
    items = []
    for spec in _cache_roots():
        root = spec["root"]
        if not root.is_dir():
            continue
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file())[:1000]:
            resolved = path.resolve()
            relative = path.relative_to(root)
            if (not resolved.is_relative_to(root)
                    or path.suffix.lower() not in spec["extensions"]
                    or (spec.get("required_part") and spec["required_part"] not in relative.parts)):
                continue
            stat = path.stat()
            items.append({
                "root": spec["id"],
                "category": spec["label"],
                "path": relative.as_posix(),
                "name": path.name,
                "extension": path.suffix.lower(),
                "size_mb": round(stat.st_size / (1024 * 1024), 1),
                "modified_at": int(stat.st_mtime),
            })
    return sorted(items, key=lambda item: item["modified_at"], reverse=True)


def purge_asset(kind, root_index, relative, scope="asset"):
    source = _asset(kind, root_index, relative)
    if scope != "asset":
        raise ValueError("Preview and lineage manifests are protected; only the selected latent asset can be purged")
    # Preview PNGs and lineage/RefMod JSON manifests are deliberately retained.
    # They are small audit/recovery artefacts and the UI must never offer a
    # destructive preview purge beside an otherwise healthy latent card.
    targets = [source]
    existing = [path for path in targets if path.is_file()]
    if not existing:
        raise FileNotFoundError("Asset or cached preview not found")
    removed_bytes = 0
    removed = []
    for path in existing:
        removed_bytes += path.stat().st_size
        path.unlink()
        removed.append(path.name)
    return {"removed": removed, "bytes": removed_bytes}


def purge_cache_file(root_id, relative):
    _, target = _cache_asset(root_id, relative)
    if not target.is_file():
        raise FileNotFoundError("Cache item not found")
    removed_bytes = target.stat().st_size
    target.unlink()
    # Remove empty render-id directories, but never the registered root itself.
    spec = next(item for item in _cache_roots() if item["id"] == root_id)
    parent = target.parent
    while parent != spec["root"] and parent.is_relative_to(spec["root"]):
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent
    return {"removed": [target.name], "bytes": removed_bytes}


def rename_asset(kind, root_index, relative, new_name):
    new_name = str(new_name or "").strip()
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if (not _SAFE_NAME.fullmatch(new_name) or new_name.endswith(".")
            or new_name.split(".")[0].upper() in reserved):
        raise ValueError("Use a short filename with letters, numbers, dot, dash or underscore")
    source = _asset(kind, root_index, relative)
    if not source.is_file():
        raise FileNotFoundError("Asset not found")
    destination = source.with_name(new_name + ".safetensors")
    if destination == source:
        return source.name
    associated = [(source, destination)]
    for companion, target in (
        (source.with_suffix(".png"), destination.with_suffix(".png")),
        (source.with_suffix(".json"), destination.with_suffix(".json")),
        (source.with_suffix(".chain.json"), destination.with_suffix(".chain.json")),
    ):
        if companion.is_file():
            associated.append((companion, target))
    if any(target.exists() for _, target in associated):
        raise FileExistsError("An asset or preview already uses that name")
    moved = []
    try:
        for original, target in associated:
            original.rename(target)
            moved.append((target, original))
    except Exception:
        for target, original in reversed(moved):
            target.rename(original)
        raise
    return destination.name


def register_routes(routes, web):
    @routes.get("/api/iamccs/h3/assets")
    async def h3_assets(request):
        try:
            return web.json_response({"items": list_assets(request.query.get("kind", "continuation"))})
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @routes.get("/api/iamccs/h3/assets/preview")
    async def h3_asset_preview(request):
        try:
            path = _asset(request.query.get("kind", ""), int(request.query.get("root", "-1")), request.query.get("path", ""))
            image = path.with_suffix(".png")
            if not image.is_file():
                return web.Response(status=404, text="No cached preview. Connect the IAMCCS preview-cache node.")
            response = web.FileResponse(image)
            response.headers["Cache-Control"] = "no-store"
            return response
        except (ValueError, TypeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @routes.post("/api/iamccs/h3/assets/rename")
    async def h3_asset_rename(request):
        try:
            payload = await request.json()
            filename = rename_asset(payload.get("kind"), payload.get("root"), payload.get("path"), payload.get("name"))
            return web.json_response({"name": filename})
        except (ValueError, TypeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except FileNotFoundError as exc:
            return web.json_response({"error": str(exc)}, status=404)
        except FileExistsError as exc:
            return web.json_response({"error": str(exc)}, status=409)

    @routes.post("/api/iamccs/h3/assets/purge")
    async def h3_asset_purge(request):
        try:
            payload = await request.json()
            result = purge_asset(
                payload.get("kind"), payload.get("root"), payload.get("path"),
                payload.get("scope", "asset"),
            )
            return web.json_response(result)
        except (ValueError, TypeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except FileNotFoundError as exc:
            return web.json_response({"error": str(exc)}, status=404)

    @routes.get("/api/iamccs/h3/caches")
    async def h3_caches(request):
        del request
        return web.json_response({"items": list_cache_files()})

    @routes.post("/api/iamccs/h3/caches/purge")
    async def h3_cache_purge(request):
        try:
            payload = await request.json()
            result = purge_cache_file(payload.get("root"), payload.get("path"))
            return web.json_response(result)
        except (ValueError, TypeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except FileNotFoundError as exc:
            return web.json_response({"error": str(exc)}, status=404)


def _save_preview(image, path):
    if not torch.is_tensor(image) or image.ndim != 4 or image.shape[0] < 1:
        raise ValueError("Preview must be a nonempty IMAGE batch")
    frame = image[-1, :, :, :3].detach().float().clamp(0, 1).mul(255).round().to(device="cpu", dtype=torch.uint8).numpy()
    pil = Image.fromarray(frame, "RGB")
    pil.thumbnail((640, 640), Image.Resampling.LANCZOS)
    pil.save(path, format="PNG")


class IAMCCS_MiniMaxH3ContinuationSave(H3ContinuousSaveLatent):
    CATEGORY = "IAMCCS/MiniMax H3/Continuation"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"]["filename_prefix"][1]["default"] = "IAMCCS/MiniMaxH3/CONTINUATION/clip"
        inputs["required"]["clip_index"][1]["default"] = 0
        inputs["optional"]["preview_image"] = ("IMAGE",)
        return inputs

    def save(self, latent, filename_prefix="IAMCCS/MiniMaxH3/CONTINUATION/clip", clip_index=0,
             handover=None, head_context_frames=0, preview_image=None, visible_frame_count=0):
        root = _roots("continuation")[0]
        prefix = (Path(folder_paths.get_output_directory()) / str(filename_prefix)).resolve()
        if not prefix.is_relative_to(root):
            raise ValueError("IAMCCS continuation checkpoints must stay inside the CONTINUATION library")
        path, info = super().save(
            latent, filename_prefix, clip_index, handover, head_context_frames,
            visible_frame_count=visible_frame_count,
        )
        if preview_image is not None:
            _save_preview(preview_image, Path(path).with_suffix(".png"))
        return path, info


class IAMCCS_MiniMaxH3ContinuationLoad(H3ContinuousLoadLatent):
    CATEGORY = "IAMCCS/MiniMax H3/Continuation"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"]["latent_path"][1]["default"] = ""
        inputs["required"]["clip_index"][1]["default"] = 0
        inputs["optional"] = {"cine_linx": (SUPERNODE_LINX_TYPE,)}
        return inputs

    def load(self, latent_path, clip_index=0, cine_linx=None):
        if cine_linx is not None:
            plan = _resolve_shotplan(cine_linx)
            configured = plan.get("continuation_settings", {}).get("checkpoint", "")
            latent_path = str(configured or latent_path)
        if not str(latent_path or "").strip():
            raise ValueError("Choose a saved AV checkpoint in Settings PRO or enter its path")
        path = Path(str(latent_path))
        if not path.is_absolute():
            path = Path(folder_paths.get_output_directory()) / path
        path = path.resolve()
        if not any(path.is_relative_to(root) for root in _roots("continuation")):
            raise ValueError("Select a checkpoint inside the IAMCCS CONTINUATION library")
        loaded = super().load(str(path), clip_index)
        latent, resolved_path, info, handover = loaded
        if isinstance(handover, dict) and handover.get("available"):
            return loaded

        # Compatibility recovery for checkpoints created by the short-lived
        # saver that rejected B's visible-frame analyzer metadata because B's
        # technical latent also contained inherited head context. The protected
        # chain/segment sidecars still carry the delivered frame count, so a
        # literal TERMINAL cutoff can be reconstructed without rewriting the
        # (potentially multi-GB) safetensors file.
        try:
            with safe_open(str(resolved_path), framework="pt", device="cpu") as reader:
                metadata = reader.metadata() or {}
            technical = int(metadata.get("frame_count", 0) or 0)
            head = int(metadata.get("head_context_frames", 0) or 0)
            chain = Path(resolved_path).with_suffix(".chain.json")
            payload = json.loads(chain.read_text(encoding="utf-8")) if chain.is_file() else {}
            segments = payload.get("segments") if isinstance(payload, dict) else None
            visible = 0
            if head > 0 and technical > head and isinstance(segments, list) and segments:
                media = Path(str(segments[-1]))
                if not media.is_absolute():
                    media = Path(folder_paths.get_output_directory()) / media
                segment_meta = Path(str(media) + ".iamccs.json")
                segment_payload = json.loads(segment_meta.read_text(encoding="utf-8"))
                visible = int(segment_payload.get("frame_count", 0) or 0)
            end = head + visible
            if visible > 0 and end <= technical:
                terminal = max(0, end - 1)
                recovered = {
                    "available": True,
                    "frame_count": technical,
                    "detector_mode": "iamccs_visible_terminal",
                    "visible_frame_count": visible,
                    "visible_start_frame": head,
                    "visible_end_frame": end,
                    "technical_padding_frames": technical - end,
                    "terminal_target_end_frame": terminal,
                    "phase_aligned_target_end_frame": terminal,
                    "phase_aware_target_end_frame": terminal,
                    "recovered_from_chain_sidecar": True,
                }
                info += (
                    f" | recovered visible terminal from chain: head {head} + "
                    f"visible {visible} + padding {technical - end}"
                )
                return latent, resolved_path, info, recovered
        except Exception:
            pass
        return loaded


class IAMCCS_MiniMaxH3ContinuationStart(H3ContinuousStartV11):
    CATEGORY = "IAMCCS/MiniMax H3/Continuation"


class IAMCCS_MiniMaxH3ContinuationContinue(H3ContinuousContinueV11):
    CATEGORY = "IAMCCS/MiniMax H3/Continuation"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        # External IAMCCS continuation has an explicit terminal mode.  This is
        # intentionally separate from LongerVid's freeze-aware AUTO handover:
        # a saved-AV continuation should normally start from the true terminal
        # latent boundary, not from an earlier anti-freeze landing point.
        required = inputs.get("required", {})
        if "handover_mode" in required:
            required["handover_mode"] = (["terminal", "auto", "manual"], {
                "default": "terminal",
                "tooltip": (
                    "TERMINAL continues from the final AV latent boundary (recommended for external A->B continuation). "
                    "AUTO consumes saved freeze/lock metadata. MANUAL excludes the requested tail."
                ),
            })
        inputs["optional"]["cine_linx"] = (SUPERNODE_LINX_TYPE,)
        return inputs

    def build(self, clip, vae, previous_latent, prompt, width, height, duration,
              context_frames="22", handover_mode="terminal", alignment_mode="phase_aligned_extended",
              manual_landing_tail_frames=0, ref_image_size="match", handover=None,
              last_frame=None, reference_image=None, cine_linx=None):
        if cine_linx is not None:
            settings = _resolve_shotplan(cine_linx).get("continuation_settings", {})
            context_frames = str(settings.get("context_frames", context_frames))
            handover_mode = str(settings.get("handover_mode", handover_mode) or "terminal").strip().lower()
            manual_landing_tail_frames = int(settings.get("manual_tail_frames", manual_landing_tail_frames))

        # TERMINAL is the real external continuation contract: reuse the latest
        # legal H3 AV context ending exactly at the previous clip's final latent
        # boundary.  The generic continuous provider already implements this as
        # a phase-aligned manual slice with desired_tail_frames=0.
        provider_mode = handover_mode
        provider_tail = manual_landing_tail_frames
        provider_handover = handover
        if handover_mode == "terminal":
            terminal_target = None
            if isinstance(handover, dict):
                try:
                    terminal_target = int(handover.get("terminal_target_end_frame"))
                except (TypeError, ValueError):
                    terminal_target = None
            if terminal_target is None:
                provider_mode = "manual"
                provider_tail = 0
                provider_handover = None
            else:
                # B+ checkpoints retain their inherited technical head and a
                # small native grid pad. AUTO is used only as a transport for
                # this explicit terminal cutoff; freeze analysis is not used.
                provider_mode = "auto"
                provider_handover = dict(handover)
                provider_handover["available"] = True
                provider_handover["phase_aligned_target_end_frame"] = terminal_target
                provider_handover["phase_aware_target_end_frame"] = terminal_target
                provider_handover["detector_mode"] = "iamccs_visible_terminal"

        return super().build(
            clip, vae, previous_latent, prompt, width, height, duration,
            context_frames=context_frames, handover_mode=provider_mode,
            alignment_mode=alignment_mode, manual_landing_tail_frames=provider_tail,
            ref_image_size=ref_image_size, handover=provider_handover, last_frame=last_frame,
            reference_image=reference_image,
        )


class IAMCCS_MiniMaxH3ContinuationAnalyze(H3ContinuousAnalyzeHandoverV11):
    CATEGORY = "IAMCCS/MiniMax H3/Continuation"


class IAMCCS_MiniMaxH3ContinuationStitch(H3ContinuousStitchOutputV11):
    CATEGORY = "IAMCCS/MiniMax H3/Continuation"


class IAMCCS_MiniMaxH3RefModPreviewCache:
    """Cache a VAE-decoded first-frame thumbnail beside a saved RefMod."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mods": ("H3_REF_MODS",), "vae": ("VAE",),
                             "index": ("INT", {"default": 0, "min": 0, "max": 255})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("preview_path",)
    FUNCTION = "cache"
    OUTPUT_NODE = True
    CATEGORY = "IAMCCS/MiniMax H3/RefMod"

    def cache(self, mods, vae, index=0):
        if not 0 <= int(index) < len(mods):
            raise ValueError("RefMod preview index is outside the bundle")
        mod = mods[int(index)][0]
        if mod.kind == "audio":
            raise ValueError("Audio RefMods do not have an image thumbnail")
        path = Path(str(mod.path))
        if path.suffix.lower() != ".safetensors":
            path = Path(str(path) + ".safetensors")
        path = path.resolve()
        if path.suffix.lower() != ".safetensors" or not any(path.is_relative_to(root) for root in _roots("refmod")):
            raise ValueError("Save the RefMod inside models/refmods before caching its preview")
        pixels = vae.decode(mod.latent[:, :, :1])
        if pixels.ndim == 5 and pixels.shape[0] == 1:
            pixels = pixels[0]
        _save_preview(pixels, path.with_suffix(".png"))
        return (str(path.with_suffix(".png")),)


NODE_CLASS_MAPPINGS = {
    "IAMCCS_MiniMaxH3ContinuationSave": IAMCCS_MiniMaxH3ContinuationSave,
    "IAMCCS_MiniMaxH3ContinuationLoad": IAMCCS_MiniMaxH3ContinuationLoad,
    "IAMCCS_MiniMaxH3ContinuationStart": IAMCCS_MiniMaxH3ContinuationStart,
    "IAMCCS_MiniMaxH3ContinuationContinue": IAMCCS_MiniMaxH3ContinuationContinue,
    "IAMCCS_MiniMaxH3ContinuationAnalyze": IAMCCS_MiniMaxH3ContinuationAnalyze,
    "IAMCCS_MiniMaxH3ContinuationStitch": IAMCCS_MiniMaxH3ContinuationStitch,
    "IAMCCS_MiniMaxH3RefModPreviewCache": IAMCCS_MiniMaxH3RefModPreviewCache,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_MiniMaxH3ContinuationSave": "H3 Continuation · Save AV Checkpoint + Preview",
    "IAMCCS_MiniMaxH3ContinuationLoad": "H3 Continuation · Load AV Checkpoint",
    "IAMCCS_MiniMaxH3ContinuationStart": "H3 Continuation · Start",
    "IAMCCS_MiniMaxH3ContinuationContinue": "H3 Continuation · Continue",
    "IAMCCS_MiniMaxH3ContinuationAnalyze": "H3 Continuation · Analyze Handover",
    "IAMCCS_MiniMaxH3ContinuationStitch": "H3 Continuation · Stitch",
    "IAMCCS_MiniMaxH3RefModPreviewCache": "H3 RefMod · Cache Latent Preview",
}
