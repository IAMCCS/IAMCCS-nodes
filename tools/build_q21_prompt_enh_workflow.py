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
        42, "randomize", "official",
    )
    return workflow.node(
        "IAMCCS_PromptQ21Enh", pos, (540, 560), inputs=inputs, outputs=outputs, widgets=widgets,
        title="IAMCCS PromptQ21Enh · official profile",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("t2i", "image_edit", "outpaint", "two_image_edit"), required=True)
    args = parser.parse_args()
    document = build_t2i() if args.mode == "t2i" else build_edit(args.mode)
    print(json.dumps(document, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
