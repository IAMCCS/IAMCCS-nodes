"""Emit drag-and-drop ComfyUI workflows for IAMCCS_PromptQ21Enh smoke tests."""
from __future__ import annotations

import argparse
import json
import uuid


UNET = "qwen_image_2.1_int8_convrot.safetensors"
CLIP = "qwen3vl_8b_int8_convrot.safetensors"
VAE = "qwen_image_2.1_vae_bf16.safetensors"
PE_T2I = "qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors"
PE_I2I = "qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors"
CONTROLNET = "Qwen-Image-2.1-Fun-Controlnet-Union.safetensors"


class Workflow:
    def __init__(self, name):
        self.name = name
        self.nodes = []
        self.links = []
        self.next_node_id = 1
        self.next_link_id = 1

    def node(self, node_type, pos, size, inputs=(), outputs=(), widgets=(), title=None, properties=None):
        node_id = self.next_node_id
        self.next_node_id += 1
        node = {
            "id": node_id,
            "type": node_type,
            "pos": list(pos),
            "size": list(size),
            "flags": {},
            "order": node_id - 1,
            "mode": 0,
            "inputs": [{"name": name, "type": data_type, "link": None} for name, data_type in inputs],
            "outputs": [{"name": name, "type": data_type, "links": []} for name, data_type in outputs],
            "properties": properties or {"Node name for S&R": node_type},
            "widgets_values": list(widgets),
        }
        if title:
            node["title"] = title
        self.nodes.append(node)
        return node

    def link(self, source, output_name, target, input_name, data_type):
        output_slot = next(i for i, item in enumerate(source["outputs"]) if item["name"] == output_name)
        input_slot = next(i for i, item in enumerate(target["inputs"]) if item["name"] == input_name)
        link_id = self.next_link_id
        self.next_link_id += 1
        self.links.append([link_id, source["id"], output_slot, target["id"], input_slot, data_type])
        source["outputs"][output_slot]["links"].append(link_id)
        target["inputs"][input_slot]["link"] = link_id

    def document(self, groups):
        return {
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"iamccs:{self.name}")),
            "revision": 0,
            "last_node_id": self.next_node_id - 1,
            "last_link_id": self.next_link_id - 1,
            "nodes": self.nodes,
            "links": self.links,
            "groups": groups,
            "config": {},
            "extra": {"ds": {"scale": 0.72, "offset": [280, 220]}},
            "version": 0.4,
        }


def model_properties(node_type, name, directory):
    return {
        "cnr_id": "comfy-core",
        "ver": "0.38.0",
        "Node name for S&R": node_type,
        "models": [{"name": name, "directory": directory}],
    }


def add_common_model_nodes(workflow, pe_model):
    unet = workflow.node(
        "UNETLoader", (-980, -20), (330, 90), outputs=(("MODEL", "MODEL"),),
        widgets=(UNET, "default"), properties=model_properties("UNETLoader", UNET, "diffusion_models"),
    )
    cache = workflow.node(
        "QwenImage21Cache", (-590, -20), (280, 85), inputs=(("model", "MODEL"),),
        outputs=(("MODEL", "MODEL"),), widgets=("auto", "default"),
    )
    clip = workflow.node(
        "CLIPLoader", (-980, 120), (330, 110), outputs=(("CLIP", "CLIP"),),
        widgets=(CLIP, "qwen_image", "default"), properties=model_properties("CLIPLoader", CLIP, "text_encoders"),
    )
    vae = workflow.node(
        "VAELoader", (-980, 270), (330, 70), outputs=(("VAE", "VAE"),), widgets=(VAE,),
        properties=model_properties("VAELoader", VAE, "vae"),
    )
    pe_clip = workflow.node(
        "CLIPLoader", (-980, 410), (330, 110), outputs=(("CLIP", "CLIP"),),
        widgets=(pe_model, "qwen_image", "default"), title="Prompt enhancer checkpoint",
        properties=model_properties("CLIPLoader", pe_model, "text_encoders"),
    )
    workflow.link(unet, "MODEL", cache, "model", "MODEL")
    return cache, clip, vae, pe_clip


def enhancer_node(workflow, task, prompt, reference_mode, pos=(-540, 370)):
    inputs = [("clip", "CLIP")] + [(f"image_{i}", "IMAGE") for i in range(1, 11)]
    outputs = (
        ("positive_prompt", "STRING"), ("negative_prompt", "STRING"), ("wh_ratio", "STRING"),
        ("ratio_follow", "STRING"), ("thinking", "STRING"), ("parse_ok", "BOOLEAN"),
        ("generator_image_1", "IMAGE"), ("generator_image_2", "IMAGE"), ("routing_info", "STRING"),
        ("generator_image_3", "IMAGE"),
    )
    widgets = (
        task, prompt, "", "error", reference_mode, 1.0, 0.95, 20, -1.0, 0, True,
        42, "randomize", "official", "safe_edit", False,
    )
    return workflow.node(
        "IAMCCS_PromptQ21Enh", pos, (540, 560), inputs=inputs, outputs=outputs, widgets=widgets,
        title="IAMCCS PromptQ21Enh · safe edit",
        properties={"aux_id": "IAMCCS/IAMCCS-nodes", "Node name for S&R": "IAMCCS_PromptQ21Enh"},
    )


def text_encoder_node(workflow, image_count, pos=(80, 160)):
    inputs = [("clip", "CLIP"), ("vae", "VAE"), ("prompt", "STRING"), ("negative_prompt", "STRING")]
    inputs.extend((f"images.image_{i}", "IMAGE") for i in range(1, image_count + 1))
    return workflow.node(
        "TextEncodeQwenImage21", pos, (460, 360), inputs=inputs,
        outputs=(("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent", "LATENT")),
        widgets=("", "", 1024), title="Qwen Image 2.1 conditioning",
        properties={"cnr_id": "comfy-core", "ver": "0.38.0", "Node name for S&R": "TextEncodeQwenImage21"},
    )


def finish_graph(workflow, cache, vae, encoder, latent_source, prefix, steps):
    sampler = workflow.node(
        "KSampler", (650, 110), (330, 480),
        inputs=(("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent_image", "LATENT")),
        outputs=(("LATENT", "LATENT"),), widgets=(42, "randomize", steps, 1.0, "euler", "simple", 1.0),
    )
    decode = workflow.node(
        "VAEDecode", (1060, 150), (230, 80), inputs=(("samples", "LATENT"), ("vae", "VAE")),
        outputs=(("IMAGE", "IMAGE"),),
    )
    preview = workflow.node(
        "PreviewImage", (1370, 20), (430, 390), inputs=(("images", "IMAGE"),), title="Smoke-test output",
    )
    save = workflow.node(
        "SaveImage", (1370, 460), (430, 300), inputs=(("images", "IMAGE"),), widgets=(prefix,),
    )
    workflow.link(cache, "MODEL", sampler, "model", "MODEL")
    workflow.link(encoder, "positive", sampler, "positive", "CONDITIONING")
    workflow.link(encoder, "negative", sampler, "negative", "CONDITIONING")
    workflow.link(latent_source, "latent" if latent_source["type"] == "TextEncodeQwenImage21" else "LATENT", sampler, "latent_image", "LATENT")
    workflow.link(sampler, "LATENT", decode, "samples", "LATENT")
    workflow.link(vae, "VAE", decode, "vae", "VAE")
    workflow.link(decode, "IMAGE", preview, "images", "IMAGE")
    workflow.link(decode, "IMAGE", save, "images", "IMAGE")


def note_node(workflow, title, text):
    return workflow.node("MarkdownNote", (-1450, -20), (390, 620), widgets=(text,), title=title)


def common_groups():
    return [
        {"id": 1, "title": "MODELS", "bounding": [-1020, -70, 750, 640], "color": "#3f789e", "font_size": 24, "flags": {}},
        {"id": 2, "title": "IAMCCS PROMPT ENHANCER", "bounding": [-580, 320, 620, 650], "color": "#8c5f2b", "font_size": 24, "flags": {}},
        {"id": 3, "title": "QWEN 2.1 RENDER", "bounding": [30, 50, 1810, 750], "color": "#3b7d5a", "font_size": 24, "flags": {}},
    ]


def build_t2i():
    workflow = Workflow("q21_prompt_enh_t2i_smoke")
    note_node(workflow, "T2I smoke test", "1. Confirm the four model names.\n2. Edit the natural-language prompt in IAMCCS PromptQ21Enh.\n3. Queue the workflow.\n\nThe enhancer output is connected directly to TextEncodeQwenImage21. The default 1 MP latent keeps the smoke test reasonably light; raise ResolutionSelector to 4 MP for native 2K production output.")
    cache, clip, vae, pe_clip = add_common_model_nodes(workflow, PE_T2I)
    enhancer = enhancer_node(workflow, "t2i", "A cinematic editorial portrait of a violin maker working at night in a warm Italian workshop, with readable gold lettering that says \"LIUTERIA\".", "general")
    resolution = workflow.node(
        "ResolutionSelector", (90, -110), (320, 120), outputs=(("width", "INT"), ("height", "INT")),
        widgets=("1:1 (Square)", 1.0, 8),
    )
    latent = workflow.node(
        "EmptyLatentImage", (90, 20), (320, 130), inputs=(("width", "INT"), ("height", "INT")),
        outputs=(("LATENT", "LATENT"),), widgets=(1024, 1024, 1),
    )
    encoder = text_encoder_node(workflow, 0, pos=(90, 200))
    workflow.link(pe_clip, "CLIP", enhancer, "clip", "CLIP")
    workflow.link(clip, "CLIP", encoder, "clip", "CLIP")
    workflow.link(vae, "VAE", encoder, "vae", "VAE")
    workflow.link(enhancer, "positive_prompt", encoder, "prompt", "STRING")
    workflow.link(enhancer, "negative_prompt", encoder, "negative_prompt", "STRING")
    workflow.link(resolution, "width", latent, "width", "INT")
    workflow.link(resolution, "height", latent, "height", "INT")
    finish_graph(workflow, cache, vae, encoder, latent, "IAMCCS_Q21_PE_T2I", 40)
    return workflow.document(common_groups())


def build_edit(mode):
    specs = {
        "image_edit": {
            "name": "q21_prompt_enh_image_edit_smoke",
            "note": "Load one image, edit the instruction in IAMCCS PromptQ21Enh, then queue. generator_image_1 is deliberately connected to TextEncodeQwenImage21 so the enhancer and renderer see the same routed reference.",
            "prompt": "Change the subject's jacket to deep emerald velvet with subtle gold embroidery. Preserve identity, face, pose, hands, background, framing and lighting.",
            "prefix": "IAMCCS_Q21_PE_IMAGE_EDIT",
            "images": 1,
            "reference_mode": "general",
        },
        "outpaint": {
            "name": "q21_prompt_enh_outpaint_smoke",
            "note": "Load one image. ImagePadForOutpaint adds 384 pixels on the left and right before enhancement and encoding. Adjust the four padding controls to choose the expansion direction; describe the desired new content in the enhancer prompt.",
            "prompt": "Outpaint image 1 to a wider cinematic composition, continuing the existing environment naturally on both sides. Preserve the original subject, identity, central composition, camera, perspective, lighting and color grade.",
            "prefix": "IAMCCS_Q21_PE_OUTPAINT",
            "images": 1,
            "reference_mode": "general",
        },
        "two_image_edit": {
            "name": "q21_prompt_enh_two_image_edit_smoke",
            "note": "Image 1 is the subject donor. Image 2 is the output canvas/scene. The subject1_on_canvas2 routing mode swaps them for Qwen so the first generator reference is the canvas while preserving the user's original numbering in the instruction.",
            "prompt": "Transfer the subject from image 1 into the pose and scene of image 2. Preserve the identity, face, hair, clothing and accessories from image 1; preserve pose, placement, background, camera, perspective and lighting from image 2. Produce one subject only.",
            "prefix": "IAMCCS_Q21_PE_TWO_IMAGE_EDIT",
            "images": 2,
            "reference_mode": "subject1_on_canvas2",
        },
    }
    spec = specs[mode]
    workflow = Workflow(spec["name"])
    note_node(workflow, f"{mode.replace('_', ' ').title()} smoke test", spec["note"])
    cache, clip, vae, pe_clip = add_common_model_nodes(workflow, PE_I2I)
    enhancer = enhancer_node(workflow, "edit", spec["prompt"], spec["reference_mode"])
    load_1 = workflow.node(
        "LoadImage", (-990, 650), (330, 330), outputs=(("IMAGE", "IMAGE"), ("MASK", "MASK")),
        widgets=("replace_with_image_1.png", "image"), title="Image 1",
    )
    source_1 = load_1
    if mode == "outpaint":
        pad = workflow.node(
            "ImagePadForOutpaint", (-600, 690), (300, 200), inputs=(("image", "IMAGE"),),
            outputs=(("IMAGE", "IMAGE"), ("MASK", "MASK")), widgets=(384, 0, 384, 0, 64),
            title="Outpaint canvas",
        )
        workflow.link(load_1, "IMAGE", pad, "image", "IMAGE")
        source_1 = pad
    workflow.link(pe_clip, "CLIP", enhancer, "clip", "CLIP")
    workflow.link(source_1, "IMAGE", enhancer, "image_1", "IMAGE")

    load_2 = None
    if spec["images"] == 2:
        load_2 = workflow.node(
            "LoadImage", (-600, 650), (330, 330), outputs=(("IMAGE", "IMAGE"), ("MASK", "MASK")),
            widgets=("replace_with_image_2.png", "image"), title="Image 2 · canvas",
        )
        workflow.link(load_2, "IMAGE", enhancer, "image_2", "IMAGE")

    encoder = text_encoder_node(workflow, spec["images"], pos=(90, 160))
    workflow.link(clip, "CLIP", encoder, "clip", "CLIP")
    workflow.link(vae, "VAE", encoder, "vae", "VAE")
    workflow.link(enhancer, "positive_prompt", encoder, "prompt", "STRING")
    workflow.link(enhancer, "negative_prompt", encoder, "negative_prompt", "STRING")
    workflow.link(enhancer, "generator_image_1", encoder, "images.image_1", "IMAGE")
    if spec["images"] == 2:
        workflow.link(enhancer, "generator_image_2", encoder, "images.image_2", "IMAGE")
    finish_graph(workflow, cache, vae, encoder, encoder, spec["prefix"], 25)
    return workflow.document(common_groups())


def build_unified():
    """Unified T2I/Edit graph using the installed known-good rewrite nodes."""
    w = Workflow("q21_unified_t2i_edit_controlnet_safe")
    note_node(
        w, "IAMCCS Q21 unified - safe defaults",
        "SAFE DEFAULTS:\n- EDIT MODE off = text to image; on = image edit.\n"
        "- AI REFINE is off by default. Turn it on only after checking the rewritten prompt preview.\n"
        "- FUN CONTROLNET is off by default. Load a control image before enabling it.\n"
        "- Prompt generation is capped to 768 tokens for T2I and 512 for Edit.\n"
        "- For two-image edit and outpaint use smoke workflows 03 and 04, which preserve explicit routing.\n\n"
        "The refine nodes below come directly from ComfyUI-Qwen-Image-2.1-Prompt-Enhancer.",
    )
    edit_mode = w.node("PrimitiveBoolean", (-1420, 650), (300, 80), outputs=(("BOOLEAN", "BOOLEAN"),), widgets=(True,), title="EDIT MODE")
    refine = w.node("PrimitiveBoolean", (-1420, 750), (300, 80), outputs=(("BOOLEAN", "BOOLEAN"),), widgets=(False,), title="AI REFINE (default OFF)")
    use_control = w.node("PrimitiveBoolean", (-1420, 850), (300, 80), outputs=(("BOOLEAN", "BOOLEAN"),), widgets=(False,), title="ENABLE FUN CONTROLNET")
    prompt = w.node(
        "PrimitiveStringMultiline", (-1000, -300), (470, 230), outputs=(("STRING", "STRING"),),
        widgets=("Cambia soltanto il colore della giacca in rosso. Mantieni invariati sfondo, volto, posa, inquadratura e illuminazione.",),
        title="PROMPT UTENTE",
    )
    negative = w.node("PrimitiveStringMultiline", (-1000, -40), (470, 120), outputs=(("STRING", "STRING"),), widgets=("",), title="NEGATIVE (optional)")

    unet = w.node("UNETLoader", (-1000, 140), (350, 90), outputs=(("MODEL", "MODEL"),), widgets=(UNET, "default"), properties=model_properties("UNETLoader", UNET, "diffusion_models"))
    cache = w.node("QwenImage21Cache", (-610, 140), (280, 85), inputs=(("model", "MODEL"),), outputs=(("MODEL", "MODEL"),), widgets=("auto", "default"))
    clip = w.node("CLIPLoader", (-1000, 260), (350, 110), outputs=(("CLIP", "CLIP"),), widgets=(CLIP, "qwen_image", "default"), properties=model_properties("CLIPLoader", CLIP, "text_encoders"))
    vae = w.node("VAELoader", (-1000, 400), (350, 70), outputs=(("VAE", "VAE"),), widgets=(VAE,), properties=model_properties("VAELoader", VAE, "vae"))
    pe_t2i = w.node("CLIPLoader", (-1000, 520), (350, 110), outputs=(("CLIP", "CLIP"),), widgets=(PE_T2I, "qwen_image", "default"), title="PE T2I - working package", properties=model_properties("CLIPLoader", PE_T2I, "text_encoders"))
    pe_edit = w.node("CLIPLoader", (-1000, 660), (350, 110), outputs=(("CLIP", "CLIP"),), widgets=(PE_I2I, "qwen_image", "default"), title="PE EDIT - working package", properties=model_properties("CLIPLoader", PE_I2I, "text_encoders"))
    w.link(unet, "MODEL", cache, "model", "MODEL")

    image = w.node("LoadImage", (-610, 650), (350, 330), outputs=(("IMAGE", "IMAGE"), ("MASK", "MASK")), widgets=("replace_with_edit_image.png", "image"), title="EDIT IMAGE / CANVAS")
    control_image = w.node("LoadImage", (-210, 650), (350, 330), outputs=(("IMAGE", "IMAGE"), ("MASK", "MASK")), widgets=("replace_with_control_image.png", "image"), title="CONTROL IMAGE")
    pre = w.node("AIO_Preprocessor", (180, 700), (330, 150), inputs=(("image", "IMAGE"),), outputs=(("IMAGE", "IMAGE"),), widgets=("CannyEdgePreprocessor", 1024), title="Control preprocessor")
    patch = w.node("ModelPatchLoader", (180, 500), (330, 80), outputs=(("MODEL_PATCH", "MODEL_PATCH"),), widgets=(CONTROLNET,), properties=model_properties("ModelPatchLoader", CONTROLNET, "controlnet"))
    apply_control = w.node(
        "QwenImageDiffsynthControlnet", (550, 500), (380, 260),
        inputs=(("model", "MODEL"), ("model_patch", "MODEL_PATCH"), ("vae", "VAE"), ("image", "IMAGE")),
        outputs=(("MODEL", "MODEL"),), widgets=(0.85, 0.0, 1.0), title="Apply Fun Union ControlNet",
    )
    model_switch = w.node(
        "ComfySwitchNode", (970, 500), (300, 160),
        inputs=(("switch", "BOOLEAN"), ("on_false", "MODEL"), ("on_true", "MODEL")),
        outputs=(("output", "MODEL"),), title="ControlNet OFF / ON",
    )
    w.link(control_image, "IMAGE", pre, "image", "IMAGE")
    w.link(cache, "MODEL", apply_control, "model", "MODEL")
    w.link(patch, "MODEL_PATCH", apply_control, "model_patch", "MODEL_PATCH")
    w.link(vae, "VAE", apply_control, "vae", "VAE")
    w.link(pre, "IMAGE", apply_control, "image", "IMAGE")
    w.link(use_control, "BOOLEAN", model_switch, "switch", "BOOLEAN")
    w.link(cache, "MODEL", model_switch, "on_false", "MODEL")
    w.link(apply_control, "MODEL", model_switch, "on_true", "MODEL")

    t2i_rewrite = w.node(
        "QwenImage21_T2IPromptRewrite", (-440, -300), (470, 280),
        inputs=(("clip", "CLIP"), ("prompt", "STRING")),
        outputs=(("positive_prompt", "STRING"), ("negative_prompt", "STRING"), ("wh_ratio", "STRING"), ("thinking", "STRING"), ("parse_ok", "BOOLEAN")),
        widgets=("", 1.0, 0.95, 20, 1.5, 768, 42, "fixed"), title="KNOWN-GOOD T2I REFINE - 768 max",
    )
    edit_rewrite = w.node(
        "QwenImage21_EditPromptRewrite", (-440, 20), (470, 350),
        inputs=(("clip", "CLIP"), ("prompt", "STRING"), ("image_1", "IMAGE")),
        outputs=(("positive_prompt", "STRING"), ("negative_prompt", "STRING"), ("wh_ratio", "STRING"), ("ratio_follow", "STRING"), ("thinking", "STRING"), ("parse_ok", "BOOLEAN")),
        widgets=("", 1.0, 0.95, 0.0, 512, 42, "fixed"), title="KNOWN-GOOD EDIT REFINE - 512 max",
    )
    w.link(pe_t2i, "CLIP", t2i_rewrite, "clip", "CLIP")
    w.link(pe_edit, "CLIP", edit_rewrite, "clip", "CLIP")
    w.link(prompt, "STRING", t2i_rewrite, "prompt", "STRING")
    w.link(prompt, "STRING", edit_rewrite, "prompt", "STRING")
    w.link(image, "IMAGE", edit_rewrite, "image_1", "IMAGE")

    t2i_prompt_switch = w.node("ComfySwitchNode", (80, -260), (300, 160), inputs=(("switch", "BOOLEAN"), ("on_false", "STRING"), ("on_true", "STRING")), outputs=(("output", "STRING"),), title="T2I RAW / REFINED")
    edit_prompt_switch = w.node("ComfySwitchNode", (80, -40), (300, 160), inputs=(("switch", "BOOLEAN"), ("on_false", "STRING"), ("on_true", "STRING")), outputs=(("output", "STRING"),), title="EDIT RAW / REFINED")
    final_prompt = w.node("ComfySwitchNode", (430, -150), (300, 160), inputs=(("switch", "BOOLEAN"), ("on_false", "STRING"), ("on_true", "STRING")), outputs=(("output", "STRING"),), title="FINAL PROMPT BY MODE")
    preview_prompt = w.node("PreviewAny", (780, -180), (520, 220), inputs=(("source", "*"),), title="CHECK FINAL PROMPT BEFORE QUEUE")
    for switch in (t2i_prompt_switch, edit_prompt_switch):
        w.link(refine, "BOOLEAN", switch, "switch", "BOOLEAN")
        w.link(prompt, "STRING", switch, "on_false", "STRING")
    w.link(t2i_rewrite, "positive_prompt", t2i_prompt_switch, "on_true", "STRING")
    w.link(edit_rewrite, "positive_prompt", edit_prompt_switch, "on_true", "STRING")
    w.link(edit_mode, "BOOLEAN", final_prompt, "switch", "BOOLEAN")
    w.link(t2i_prompt_switch, "output", final_prompt, "on_false", "STRING")
    w.link(edit_prompt_switch, "output", final_prompt, "on_true", "STRING")
    w.link(final_prompt, "output", preview_prompt, "source", "*")

    resolution = w.node("ResolutionSelector", (80, 180), (320, 120), outputs=(("width", "INT"), ("height", "INT")), widgets=("1:1 (Square)", 1.0, 8))
    latent = w.node("EmptyLatentImage", (430, 160), (300, 130), inputs=(("width", "INT"), ("height", "INT")), outputs=(("LATENT", "LATENT"),), widgets=(1024, 1024, 1))
    enc_t2i = text_encoder_node(w, 0, pos=(80, 340))
    enc_edit = text_encoder_node(w, 1, pos=(570, 330))
    enc_edit["title"] = "EDIT conditioning - canvas determines size"
    w.link(resolution, "width", latent, "width", "INT")
    w.link(resolution, "height", latent, "height", "INT")
    for enc in (enc_t2i, enc_edit):
        w.link(clip, "CLIP", enc, "clip", "CLIP")
        w.link(vae, "VAE", enc, "vae", "VAE")
        w.link(final_prompt, "output", enc, "prompt", "STRING")
        w.link(negative, "STRING", enc, "negative_prompt", "STRING")
    w.link(image, "IMAGE", enc_edit, "images.image_1", "IMAGE")

    positive_switch = w.node("ComfySwitchNode", (1070, 160), (280, 150), inputs=(("switch", "BOOLEAN"), ("on_false", "CONDITIONING"), ("on_true", "CONDITIONING")), outputs=(("output", "CONDITIONING"),), title="T2I / EDIT positive")
    negative_switch = w.node("ComfySwitchNode", (1070, 330), (280, 150), inputs=(("switch", "BOOLEAN"), ("on_false", "CONDITIONING"), ("on_true", "CONDITIONING")), outputs=(("output", "CONDITIONING"),), title="T2I / EDIT negative")
    latent_switch = w.node("ComfySwitchNode", (1070, 680), (280, 150), inputs=(("switch", "BOOLEAN"), ("on_false", "LATENT"), ("on_true", "LATENT")), outputs=(("output", "LATENT"),), title="T2I / EDIT latent")
    for sw in (positive_switch, negative_switch, latent_switch):
        w.link(edit_mode, "BOOLEAN", sw, "switch", "BOOLEAN")
    w.link(enc_t2i, "positive", positive_switch, "on_false", "CONDITIONING")
    w.link(enc_edit, "positive", positive_switch, "on_true", "CONDITIONING")
    w.link(enc_t2i, "negative", negative_switch, "on_false", "CONDITIONING")
    w.link(enc_edit, "negative", negative_switch, "on_true", "CONDITIONING")
    w.link(latent, "LATENT", latent_switch, "on_false", "LATENT")
    w.link(enc_edit, "latent", latent_switch, "on_true", "LATENT")

    sampler = w.node("KSampler", (1410, 220), (330, 480), inputs=(("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent_image", "LATENT")), outputs=(("LATENT", "LATENT"),), widgets=(42, "randomize", 25, 1.0, "euler", "simple", 1.0))
    decode = w.node("VAEDecode", (1790, 260), (230, 80), inputs=(("samples", "LATENT"), ("vae", "VAE")), outputs=(("IMAGE", "IMAGE"),))
    preview = w.node("PreviewImage", (2060, 20), (440, 390), inputs=(("images", "IMAGE"),), title="GENERATED IMAGE")
    save = w.node("SaveImage", (2060, 450), (440, 300), inputs=(("images", "IMAGE"),), widgets=("IAMCCS_Q21_UNIFIED",))
    w.link(model_switch, "output", sampler, "model", "MODEL")
    w.link(positive_switch, "output", sampler, "positive", "CONDITIONING")
    w.link(negative_switch, "output", sampler, "negative", "CONDITIONING")
    w.link(latent_switch, "output", sampler, "latent_image", "LATENT")
    w.link(sampler, "LATENT", decode, "samples", "LATENT")
    w.link(vae, "VAE", decode, "vae", "VAE")
    w.link(decode, "IMAGE", preview, "images", "IMAGE")
    w.link(decode, "IMAGE", save, "images", "IMAGE")

    groups = [
        {"id": 1, "title": "SAFE CONTROLS", "bounding": [-1460, 620, 380, 350], "color": "#8c5f2b", "font_size": 24, "flags": {}},
        {"id": 2, "title": "PROMPT + KNOWN-GOOD REFINERS", "bounding": [-1030, -350, 1770, 750], "color": "#73518c", "font_size": 24, "flags": {}},
        {"id": 3, "title": "MODELS", "bounding": [-1030, 100, 750, 700], "color": "#3f789e", "font_size": 24, "flags": {}},
        {"id": 4, "title": "FUN CONTROLNET", "bounding": [-250, 450, 1560, 560], "color": "#8c6c2b", "font_size": 24, "flags": {}},
        {"id": 5, "title": "QWEN 2.1 RENDER", "bounding": [40, 120, 2500, 750], "color": "#3b7d5a", "font_size": 24, "flags": {}},
    ]
    return w.document(groups)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("t2i", "image_edit", "outpaint", "two_image_edit", "unified"), required=True)
    args = parser.parse_args()
    if args.mode == "t2i":
        document = build_t2i()
    elif args.mode == "unified":
        document = build_unified()
    else:
        document = build_edit(args.mode)
    print(json.dumps(document, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
