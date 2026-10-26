"""Save a MOSS render, return it to its editor, and publish native H3 media."""
import base64
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import uuid

from aiohttp import web
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import nodes
from server import PromptServer

from .iamccs_moss_frame_designer import image_linx
from .moss_text_layers import compose_text_layers


def notify_moss_queue(json_data):
    prompt = json_data.get("prompt", {})
    for entry in prompt.values():
        if entry.get("class_type") != "IAMCCS_MOSSFrameResult":
            continue
        source = entry.get("inputs", {}).get("project_data")
        if isinstance(source, list) and str(source[0]) in prompt:
            designer = prompt[str(source[0])]
            if designer.get("class_type") == "IAMCCS_MOSSFrameDesigner":
                design = json.loads(designer["inputs"]["design_data"])
                PromptServer.instance.send_sync("iamccs-moss-start", {"designer_id": str(source[0]), "project_id": design.get("project_id")})
    return json_data


PromptServer.instance.add_on_prompt_handler(notify_moss_queue)


def local_project_folder(raw):
    folder = Path(raw)
    if not folder.is_absolute() or not folder.is_dir():
        raise ValueError("Choose an existing absolute folder.")
    if os.name == "nt" and folder.drive.startswith("\\\\"):
        raise ValueError("Choose a local drive for Mboard projects.")
    return folder.resolve()


@PromptServer.instance.routes.get("/iamccs/moss/project-folders")
async def project_folders(request):
    if request.remote not in ("127.0.0.1", "::1"):
        return web.json_response({"error": "Local folder browsing requires localhost."}, status=403)
    raw = request.query.get("path", "")
    if not raw:
        roots = os.listdrives() if os.name == "nt" else ["/"]
        return web.json_response({"path": "", "parent": "", "folders": [{"name": root, "path": root} for root in roots]})
    try:
        folder = local_project_folder(raw)
        children = sorted((child for child in folder.iterdir() if child.is_dir()), key=lambda path: path.name.lower())
    except (ValueError, OSError) as error:
        return web.json_response({"error": str(error)}, status=400)
    return web.json_response({"path": str(folder), "parent": str(folder.parent) if folder.parent != folder else "",
                              "folders": [{"name": child.name, "path": str(child)} for child in children]})


@PromptServer.instance.routes.post("/iamccs/moss/save-mboard")
async def save_mboard(request):
    if request.remote not in ("127.0.0.1", "::1"):
        return web.json_response({"error": "Saving local projects requires localhost."}, status=403)
    if request.content_type != "application/json":
        return web.json_response({"error": "Expected an Mboard JSON document."}, status=400)
    body = await request.json()
    project = body.get("project", {})
    if project.get("format") != "IAMCCS.Mboard" or project.get("edition") != "MOSS" or project.get("version") != 1:
        return web.json_response({"error": "Unsupported Mboard project."}, status=400)
    try:
        folder = local_project_folder(body.get("directory", ""))
        name = str(body.get("name", ""))
        if not name or Path(name).name != name or ":" in name or Path(name).suffix.lower() != ".mboard":
            raise ValueError("Use a filename ending in .Mboard, without folder components.")
        target = folder / name
        if target.is_symlink() or target.is_junction():
            raise ValueError("Choose a regular project file.")
    except (ValueError, OSError) as error:
        return web.json_response({"error": str(error)}, status=400)
    if target.exists() and body.get("overwrite") is not True:
        return web.json_response({"error": "This project already exists. Choose another name or enable Replace existing file."}, status=409)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, suffix=".tmp", delete=False) as file:
            temporary = file.name
            json.dump(project, file, ensure_ascii=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, target)
    except OSError as error:
        return web.json_response({"error": str(error)}, status=400)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return web.json_response({"saved": str(target)})


def clean_project_json(raw):
    design = json.loads(raw)
    design.pop("generated", None)
    if "project_metadata" in design:
        design["project_metadata"].pop("workflow", None)
    return json.dumps(design, ensure_ascii=False)


def render_metadata(prompt, extra_pnginfo):
    clean_prompt, clean_extra = copy.deepcopy(prompt), copy.deepcopy(extra_pnginfo or {})
    for entry in (clean_prompt or {}).values():
        entry.pop("is_changed", None)
        if entry.get("class_type") == "IAMCCS_MOSSFrameDesigner":
            entry["inputs"]["design_data"] = clean_project_json(entry["inputs"]["design_data"])
    for node in clean_extra.get("workflow", {}).get("nodes", []):
        if node.get("type") == "IAMCCS_MOSSFrameDesigner" and node.get("widgets_values"):
            node["widgets_values"][0] = clean_project_json(node["widgets_values"][0])
    return clean_prompt, clean_extra


class IAMCCS_MOSSFrameResult:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Saving and notifying the editor are execution side effects, even when sampling is cached.
        return uuid.uuid4().hex

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "project_data": ("STRING", {"forceInput": True}),
                             "filename_prefix": ("STRING", {"default": "IAMCCS_MOSS/Frame"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("IAMCCS_SUPERNODE_LINX", "IMAGE", "STRING")
    RETURN_NAMES = ("cine_linx", "image", "Mboard_project")
    OUTPUT_NODE = True
    FUNCTION = "publish"
    CATEGORY = "IAMCCS/Cine/MOSS"
    DESCRIPTION = "Connect decoded MOSS images and the designer project_data. Saves PNG, updates the designer canvas, and exports this execution's image to H3 without a graph cycle."

    def publish(self, images, project_data, filename_prefix="IAMCCS_MOSS/Frame", prompt=None, extra_pnginfo=None, unique_id=None):
        design = json.loads(project_data)
        design.pop("generated", None)
        if "project_metadata" in design:
            design["project_metadata"].pop("workflow", None)
        images, text_layers = compose_text_layers(images, design)
        if text_layers:
            design.setdefault("project_metadata", {})["text_compositing"] = text_layers
        project = {"format": "IAMCCS.Mboard", "version": 1, "edition": "MOSS", "design": design}
        clean_prompt, clean_extra = render_metadata(prompt, extra_pnginfo)
        metadata = PngInfo()
        if clean_prompt is not None:
            metadata.add_text("prompt", json.dumps(clean_prompt))
        for key, value in clean_extra.items():
            metadata.add_text(key, json.dumps(value))
        metadata.add_text("Mboard", json.dumps(project, ensure_ascii=False))
        image = Image.fromarray((images[0].detach().cpu().clamp(0, 1).numpy() * 255).round().astype("uint8"))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", pnginfo=metadata)
        design["generated"] = {"data_url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
                               "width": image.width, "height": image.height,
                               "prompt": clean_prompt, "metadata": clean_extra}
        saved = nodes.SaveImage().save_images(images, filename_prefix, clean_prompt, {**clean_extra, "Mboard": project})
        design["generated"]["files"] = saved["ui"]["images"]
        project_json = json.dumps(project, ensure_ascii=False)
        source = (prompt or {}).get(str(unique_id), {}).get("inputs", {}).get("project_data")
        if isinstance(source, list):
            PromptServer.instance.send_sync("iamccs-moss-render", {
                "designer_id": str(source[0]), "project_id": design.get("project_id"), "generated": design["generated"]})
        return {"ui": {**saved["ui"], "moss_project": [project_json]},
                "result": (image_linx(design, images), images, project_json)}


NODE_CLASS_MAPPINGS = {"IAMCCS_MOSSFrameResult": IAMCCS_MOSSFrameResult}
NODE_DISPLAY_NAME_MAPPINGS = {"IAMCCS_MOSSFrameResult": "IAMCCS MOSS Result — Canvas / cine_linx"}
