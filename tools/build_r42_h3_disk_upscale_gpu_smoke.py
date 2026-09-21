"""Build the dedicated R42 two-prompt H3 learned-3D GPU smoke workflow.

The universal source workflow is read-only. The generated variant disables its
normal delivery terminal and adds one checkpoint terminal plus a disabled
refine terminal, so generation and Full-HD refine can be queued separately.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


SOURCE = Path(r"X:\1_UNIVERSAL_42_43\A_IAMCCS_H3_MINIMAX_R42_UNIVERSAL_180926.json")
OUTPUT = Path(r"X:\1_UNIVERSAL_42_43\A2_IAMCCS_H3_R42_DISK_TILED_FULLHD_GPU_SMOKE_200926.json")
STAGE1_OUTPUT = Path(r"X:\1_UNIVERSAL_42_43\B1_IAMCCS_H3_R42_DISK_CHECKPOINT_GPU_SMOKE_200926.json")
STAGE2_OUTPUT = Path(r"X:\1_UNIVERSAL_42_43\B3_IAMCCS_H3_R42_LEARNED3D_FULLHD_GPU_SMOKE_200926.json")


def socket(name, kind, link=None, *, widget=False, shape=None):
    value = {"localized_name": name, "name": name, "type": kind, "link": link}
    if widget:
        value["widget"] = {"name": name}
    if shape is not None:
        value["shape"] = shape
    return value


def output(name, kind, links=None, slot=None):
    value = {"localized_name": name, "name": name, "type": kind, "links": links or []}
    if slot is not None:
        value["slot_index"] = slot
    return value


def main():
    workflow = json.loads(SOURCE.read_text(encoding="utf-8"))
    nodes = workflow["nodes"]
    links = workflow["links"]
    by_id = {int(node["id"]): node for node in nodes}
    reserved = set(range(900, 907))
    if reserved.intersection(by_id):
        raise RuntimeError("R42 smoke node ids 900..906 are already occupied")
    if any(node.get("type", "").startswith("IAMCCS_H3DiskUpscale") for node in nodes):
        raise RuntimeError("Source workflow already contains H3 disk-upscale nodes")

    # The dedicated variant has one terminal at a time. This prevents the
    # ordinary R42 delivery route from evaluating alongside the memory test.
    by_id[806]["mode"] = 2
    by_id[806]["title"] = "R42 NORMAL DELIVERY DISABLED IN GPU SMOKE VARIANT"

    next_order = max(int(node.get("order", 0)) for node in nodes) + 1
    note = (
        "# R42 H3 LEARNED-3D FULL-HD GPU SMOKE — TWO QUEUES\n\n"
        "This variant deliberately disables the normal R42 Delivery Router. Shotboard and the R42 backend remain the truth.\n\n"
        "## QUEUE 1 — native generation → disk checkpoint\n"
        "1. Keep **1 SAVE AV CHECKPOINT** on `Always`.\n"
        "2. Keep **2 LEARNED 3D GRID-FREE** and **3 ASSEMBLE** on `Never`.\n"
        "3. Use a fresh `render_id`, then Queue. `source_frames=0` reads the exact H3 grid.\n"
        "4. The checkpoint is written to `ComfyUI/output/IAMCCS/H3_DISK_UPSCALE/<render_id>/checkpoints/segment_00000.safetensors`.\n\n"
        "## QUEUE 2 — checkpoint → 1920×1080\n"
        "1. Set **1 SAVE AV CHECKPOINT** to `Never`.\n"
        "2. Paste the absolute safetensors path into **2 LEARNED 3D GRID-FREE**.\n"
        "3. Give Queue 2 a fresh `output_render_id`, choose Full HD/QHD/UHD/source scale/custom and set the node to `Always`.\n"
        "4. The source canvas is read from the checkpoint; the learned lift is isotropic, temporally chunked and spatially full-frame. Decode remains one group at a time.\n\n"
        "The original spatial tiled diffusion refine is intentionally absent: it produced a visible texture grid. "
        "A Full-HD pass is not considered validated until frame count, audio and image detail are checked."
    )
    nodes.extend([
        {
            "id": 900, "type": "MarkdownNote", "pos": [3370, 1340], "size": [1080, 720],
            "flags": {}, "order": next_order, "mode": 0, "inputs": [], "outputs": [],
            "title": "READ FIRST · TWO-PROMPT GPU SMOKE",
            "properties": {"Node name for S&R": "MarkdownNote"},
            "widgets_values": [note], "widgets_values_named": {"text": note},
            "color": "#291c0d", "bgcolor": "#0f1215",
        },
        {
            "id": 901, "type": "IAMCCS_H3DiskUpscaleCheckpoint", "pos": [4580, 1370], "size": [620, 430],
            "flags": {}, "order": next_order + 1, "mode": 0,
            "inputs": [
                socket("latent", "LATENT", 2004), socket("native_audio", "AUDIO", 2005),
                socket("render_id", "STRING", None, widget=True), socket("segment_index", "INT", 2006, widget=True),
                socket("source_frames", "INT", None, widget=True), socket("fps", "INT", None, widget=True),
                socket("technical_prefix_frames", "INT", 2007, widget=True),
                socket("join_overlap_frames", "INT", 2008, widget=True),
                socket("join_mode", "COMBO", None, widget=True),
            ],
            "outputs": [output("checkpoint_path", "STRING", [], 0), output("report", "STRING", [], 1)],
            "title": "1 SAVE AV CHECKPOINT · ALWAYS FOR QUEUE 1",
            "properties": {"aux_id": "IAMCCS/IAMCCS-nodes", "Node name for S&R": "IAMCCS_H3DiskUpscaleCheckpoint"},
            "widgets_values": ["r42_h3_disk_gpu_smoke_01", 0, 0, 24, 0, 0, "cut"],
            "widgets_values_named": {
                "render_id": "r42_h3_disk_gpu_smoke_01", "segment_index": 0, "source_frames": 0,
                "fps": 24, "technical_prefix_frames": 0, "join_overlap_frames": 0, "join_mode": "cut",
            },
            "color": "#6b4318", "bgcolor": "#21170d",
        },
        {
            "id": 903, "type": "RandomNoise", "pos": [4560, 2030], "size": [300, 100],
            "flags": {}, "order": next_order + 2, "mode": 0,
            "inputs": [socket("noise_seed", "INT", None, widget=True)],
            "outputs": [output("NOISE", "NOISE", [2011], 0)],
            "title": "FIXED UPSCALE NOISE", "properties": {"cnr_id": "comfy-core", "Node name for S&R": "RandomNoise"},
            "widgets_values": [20260920, "fixed"], "widgets_values_named": {"noise_seed": 20260920, "control_after_generate": "fixed"},
        },
        {
            "id": 904, "type": "KSamplerSelect", "pos": [4560, 2160], "size": [300, 100],
            "flags": {}, "order": next_order + 3, "mode": 0,
            "inputs": [socket("sampler_name", "COMBO", None, widget=True)],
            "outputs": [output("SAMPLER", "SAMPLER", [2012], 0)],
            "title": "UPSCALE SAMPLER", "properties": {"cnr_id": "comfy-core", "Node name for S&R": "KSamplerSelect"},
            "widgets_values": ["er_sde"], "widgets_values_named": {"sampler_name": "er_sde"},
        },
        {
            "id": 905, "type": "BasicScheduler", "pos": [4560, 2290], "size": [300, 160],
            "flags": {}, "order": next_order + 4, "mode": 0,
            "inputs": [
                socket("model", "MODEL", 2009), socket("scheduler", "COMBO", None, widget=True),
                socket("steps", "INT", None, widget=True), socket("denoise", "FLOAT", None, widget=True),
            ],
            "outputs": [output("SIGMAS", "SIGMAS", [2013], 0)],
            "title": "LOW-DENOISE REFINE", "properties": {"cnr_id": "comfy-core", "Node name for S&R": "BasicScheduler"},
            "widgets_values": ["simple", 2, 0.15],
            "widgets_values_named": {"scheduler": "simple", "steps": 2, "denoise": 0.15},
        },
        {
            "id": 902, "type": "IAMCCS_H3DiskUpscaleLearned3D", "pos": [5350, 1370], "size": [760, 560],
            "flags": {}, "order": next_order + 5, "mode": 2,
            "inputs": [
                socket("checkpoint_path", "STRING", None, widget=True),
                socket("output_render_id", "STRING", None, widget=True),
                socket("video_vae", "VAE", 2015), socket("upscaler_model", "COMBO", None, widget=True),
                socket("target_preset", "COMBO", None, widget=True),
                socket("target_width", "INT", None, widget=True), socket("target_height", "INT", None, widget=True),
                socket("upscaler_device", "COMBO", None, widget=True),
                socket("upscaler_precision", "COMBO", None, widget=True),
                socket("temporal_core_tokens", "INT", None, widget=True),
                socket("temporal_halo_tokens", "INT", None, widget=True),
                socket("decode_groups_per_chunk", "INT", None, widget=True),
            ],
            "outputs": [
                output("segment_path", "STRING", [], 0), output("segment_manifest_path", "STRING", [], 1),
                output("report", "STRING", [], 2),
            ],
            "title": "2 LEARNED 3D GRID-FREE + STREAM · NEVER UNTIL QUEUE 2",
            "properties": {"aux_id": "IAMCCS/IAMCCS-nodes", "Node name for S&R": "IAMCCS_H3DiskUpscaleLearned3D"},
            "widgets_values": [
                r"D:\ComfyUI\ComfyUI\output\IAMCCS\H3_DISK_UPSCALE\r42_h3_disk_gpu_smoke_01\checkpoints\segment_00000.safetensors",
                "r42_h3_learned3d_gpu_smoke_02",
                "minimax_h3_latent_upscaler_3d_fp16.safetensors", "full_hd_1920x1080",
                1920, 1080, "cuda", "fp16", 4, 4, 1,
            ],
            "widgets_values_named": {
                "checkpoint_path": r"D:\ComfyUI\ComfyUI\output\IAMCCS\H3_DISK_UPSCALE\r42_h3_disk_gpu_smoke_01\checkpoints\segment_00000.safetensors",
                "output_render_id": "r42_h3_learned3d_gpu_smoke_02",
                "upscaler_model": "minimax_h3_latent_upscaler_3d_fp16.safetensors",
                "target_preset": "full_hd_1920x1080", "target_width": 1920, "target_height": 1080,
                "upscaler_device": "cuda", "upscaler_precision": "fp16",
                "temporal_core_tokens": 4, "temporal_halo_tokens": 4,
                "decode_groups_per_chunk": 1,
            },
            "color": "#24506b", "bgcolor": "#142d3d",
        },
        {
            "id": 906, "type": "IAMCCS_H3DiskUpscaleAssemble", "pos": [6280, 1370], "size": [520, 230],
            "flags": {}, "order": next_order + 6, "mode": 2,
            "inputs": [
                socket("render_id", "STRING", None, widget=True), socket("segment_count", "INT", None, widget=True),
                socket("output_name", "STRING", None, widget=True),
            ],
            "outputs": [output("film_path", "STRING", [], 0), output("report", "STRING", [], 1)],
            "title": "3 ASSEMBLE · OPTIONAL AFTER ALL SEGMENTS",
            "properties": {"aux_id": "IAMCCS/IAMCCS-nodes", "Node name for S&R": "IAMCCS_H3DiskUpscaleAssemble"},
            "widgets_values": ["r42_h3_learned3d_gpu_smoke_02", 1, "r42_fullhd_learned3d_master"],
            "widgets_values_named": {"render_id": "r42_h3_learned3d_gpu_smoke_02", "segment_count": 1, "output_name": "r42_fullhd_learned3d_master"},
            "color": "#345c75", "bgcolor": "#101820",
        },
    ])

    # The accepted quality path does not execute H3 diffusion a second time.
    # Drop the old noise/sampler/scheduler helpers that belonged to the spatial
    # tiled experiment and would otherwise reload the full H3 stack in Queue 2.
    nodes[:] = [node for node in nodes if int(node["id"]) not in {903, 904, 905}]

    def add_link(link_id, source_id, source_slot, target_id, target_slot, kind):
        links.append([link_id, source_id, source_slot, target_id, target_slot, kind])
        by_source = next(node for node in nodes if int(node["id"]) == source_id)
        by_source["outputs"][source_slot].setdefault("links", []).append(link_id)

    # Take the selected native AV directly from the R42 lazy switch. Do not go
    # through StateCommit/FaceDelivery in Queue 1: committing motion state would
    # make Queue 2 rebuild different conditioning for the same source segment.
    add_link(2004, 619, 3, 901, 0, "LATENT")
    add_link(2005, 619, 1, 901, 1, "AUDIO")
    add_link(2006, 9, 7, 901, 3, "INT")
    add_link(2007, 617, 1, 901, 6, "INT")
    add_link(2008, 9, 9, 901, 7, "INT")
    add_link(2015, 5, 0, 902, 2, "VAE")

    workflow.setdefault("groups", []).append({
        "id": max((int(group.get("id", 0)) for group in workflow.get("groups", [])), default=0) + 1,
        "title": "R42 H3 LEARNED-3D FULL-HD GPU SMOKE · TWO PROMPTS",
        "bounding": [3330, 1280, 3520, 1240], "color": "#9a552e", "flags": {},
    })
    workflow["last_node_id"] = max(int(node["id"]) for node in nodes)
    workflow["last_link_id"] = max(int(link[0]) for link in links)
    workflow.setdefault("extra", {})["iamccs_h3_disk_upscale_gpu_smoke_20260920"] = {
        "source": str(SOURCE), "backend": "R42", "normal_delivery_router_mode": "Never",
        "execution": "two separate queues", "target": [1920, 1080],
        "stage2_method": "learned_3d_grid_free", "source_resolution_policy": "read_from_checkpoint_latent",
        "stage2_initial_mode": "Never", "gpu_validation": "pending",
    }
    OUTPUT.write_text(json.dumps(workflow, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(OUTPUT)
    print(f"nodes={len(nodes)} links={len(links)} last_node_id={workflow['last_node_id']} last_link_id={workflow['last_link_id']}")

    def dedicated_variant(remove_ids, title, note_text, positions, output_path, stage):
        data = deepcopy(workflow)
        data["nodes"] = [node for node in data["nodes"] if int(node["id"]) not in remove_ids]
        retained = {int(node["id"]) for node in data["nodes"]}
        data["links"] = [link for link in data["links"] if int(link[1]) in retained and int(link[3]) in retained]
        valid_links = {int(link[0]) for link in data["links"]}
        for node in data["nodes"]:
            for out in node.get("outputs", []):
                if out.get("links") is not None:
                    out["links"] = [value for value in out["links"] if int(value) in valid_links]
            for inp in node.get("inputs", []):
                if inp.get("link") is not None and int(inp["link"]) not in valid_links:
                    inp["link"] = None
            if int(node["id"]) in positions:
                node["pos"] = positions[int(node["id"])]
        note_node = next(node for node in data["nodes"] if int(node["id"]) == 900)
        note_node["title"] = title
        note_node["widgets_values"] = [note_text]
        note_node["widgets_values_named"] = {"text": note_text}
        if stage == 2:
            next(node for node in data["nodes"] if int(node["id"]) == 902)["mode"] = 0
        smoke_group = next(group for group in data["groups"] if "GPU SMOKE" in str(group.get("title", "")))
        smoke_group["title"] = title
        smoke_group["bounding"] = [-5680, -3780, 3260, 1250]
        data["last_node_id"] = max(int(node["id"]) for node in data["nodes"])
        data["last_link_id"] = max(int(link[0]) for link in data["links"])
        data.setdefault("extra", {})["ds"] = {"scale": 0.65, "offset": [5700, 3850]}
        data["extra"]["iamccs_h3_disk_upscale_gpu_smoke_20260920"]["dedicated_stage"] = stage
        output_path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(output_path)

    dedicated_variant(
        {902, 903, 904, 905, 906},
        "QUEUE 1 ONLY · R42 NATIVE AV → DISK CHECKPOINT",
        "# QUEUE 1 — CREA IL CHECKPOINT H3\n\n"
        "Questo workflow contiene un solo terminale attivo: **SAVE AV CHECKPOINT**. Non devi cercare Always/Never e non devi impostare tile.\n\n"
        "1. Scegli nello Shotboard il segmento da provare. Per il primo smoke usa il segmento 0.\n"
        "2. Nel nodo arancione modifica `render_id` soltanto se il file esiste già.\n"
        "3. Lascia `source_frames = 0`, `fps = 24` e `join_mode = cut`.\n"
        "4. Premi Queue. Il backend R42 genera il native AV e il nodo salva latent video, latent audio e waveform.\n\n"
        "Percorso previsto per il primo segmento:\n"
        "`D:/ComfyUI/ComfyUI/output/IAMCCS/H3_DISK_UPSCALE/r42_h3_disk_gpu_smoke_01/checkpoints/segment_00000.safetensors`\n\n"
        "Dopo il completamento chiudi questo workflow e carica **B3 ... LEARNED3D FULLHD GPU SMOKE**.",
        {900: [-5600, -3650], 901: [-4420, -3470]},
        STAGE1_OUTPUT,
        1,
    )
    dedicated_variant(
        {901, 906},
        "QUEUE 2 ONLY · R42 CHECKPOINT → LEARNED-3D FULL-HD",
        "# QUEUE 2 — UPSCALE LEARNED 3D, SENZA GRIGLIA\n\n"
        "Questo workflow contiene un solo terminale attivo: **LEARNED 3D GRID-FREE + STREAM**. Non ricarica H3, conditioning, noise o sampler.\n\n"
        "1. Nel campo `checkpoint_path` verifica il safetensors creato da B1. La risoluzione sorgente viene letta dal checkpoint: può essere 640, 1280 o un altro canvas H3.\n"
        "2. Usa un `output_render_id` nuovo. Scegli Full HD, HD, QHD, UHD, 1.5x/2x/3x sorgente o custom.\n"
        "3. Primo test 12 GB: `full_hd_1920x1080`, CUDA FP16, temporal core/halo 4/4 latent token, decode group 1. Il lift conserva l'aspect ratio e il crop avviene solo dopo il decode.\n"
        "4. Premi Queue. Il modello learned 3D usa microfinestre temporali full-frame; nessun sampling diffusion separato per tile.\n\n"
        "Output: `ComfyUI/output/IAMCCS/H3_DISK_UPSCALE/<output_render_id>/segments/segment_00000.mp4`.",
        {900: [-5600, -3700], 902: [-4110, -3530]},
        STAGE2_OUTPUT,
        2,
    )


if __name__ == "__main__":
    main()
