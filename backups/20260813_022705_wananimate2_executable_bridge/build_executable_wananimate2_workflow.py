from __future__ import annotations

import json
from pathlib import Path


# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
BACKUP = Path(__file__).with_name("IAMCCS_V2V_SHOTBOARD_EASY_WANANIMATE2_GGUF.json")
DESTINATION = Path(r"D:\ComfyUI\ComfyUI\user\default\workflows\IAMCCS_V2V_SHOTBOARD_EASY\IAMCCS_V2V_SHOTBOARD_EASY_WANANIMATE2_GGUF.json")

# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
BRIDGE_OUTPUTS = [
    ("trim_start_s", "FLOAT"), ("trim_duration_s", "FLOAT"),
    ("target_frames", "INT"), ("width", "INT"), ("height", "INT"),
    ("chunk_frames", "INT"), ("generation_steps", "INT"),
    ("generation_cfg", "FLOAT"), ("generation_seed", "INT"), ("reference_strength", "FLOAT"),
    ("pose_strength", "FLOAT"), ("pose_start_percent", "FLOAT"),
    ("pose_end_percent", "FLOAT"), ("context_windows", "BOOLEAN"),
    ("context_length", "INT"), ("context_overlap", "INT"),
    ("pose_cache", "BOOLEAN"), ("empty_cache_each_chunk", "BOOLEAN"),
    ("positive_prompt", "STRING"), ("negative_prompt", "STRING"),
    ("pose_prompt", "STRING"), ("effective_lora_strength", "FLOAT"),
    ("output_prefix", "STRING"),
]

# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
MAPPINGS = [
    (1, 0, 26, 0, "IAMCCS_SUPERNODE_LINX"), (26, 0, 4, 1, "FLOAT"),
    (26, 1, 4, 2, "FLOAT"), (26, 2, 22, 10, "INT"), (26, 3, 22, 8, "INT"),
    (26, 4, 22, 9, "INT"), (26, 5, 22, 11, "INT"), (26, 6, 21, 2, "INT"),
    (26, 7, 22, 14, "FLOAT"), (26, 8, 22, 12, "INT"), (26, 9, 22, 15, "FLOAT"),
    (26, 10, 22, 16, "FLOAT"), (26, 11, 22, 17, "FLOAT"), (26, 12, 22, 18, "FLOAT"),
    (26, 13, 22, 20, "BOOLEAN"), (26, 14, 22, 21, "INT"), (26, 15, 22, 22, "INT"),
    (26, 16, 22, 25, "BOOLEAN"), (26, 17, 22, 31, "BOOLEAN"), (26, 18, 12, 0, "STRING"),
    (26, 19, 13, 0, "STRING"), (26, 20, 14, 0, "STRING"), (26, 21, 9, 2, "FLOAT"),
    (26, 22, 25, 1, "STRING"),
]


def build() -> None:
    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    workflow = json.loads(BACKUP.read_text(encoding="utf-8"))
    nodes_by_id = {node["id"]: node for node in workflow["nodes"]}
    nodes_by_id[1]["widgets_values"][30:37] = [
        "Wan2.2-Animate-14B-Q4_K_S.gguf",
        "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
        False,
        1.0,
        "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "clip_vision_h.safetensors",
        "Wan2_1_VAE_bf16.safetensors",
    ]
    workflow["links"] = [link for link in workflow["links"] if link[0] != 20]
    nodes_by_id[6]["outputs"][2]["links"] = []
    nodes_by_id[22]["inputs"][10]["link"] = None

    bridge = {
        "id": 26,
        "type": "IAMCCS_WanAnimate2ShotboardBridge",
        "pos": [1460.0, -320.0],
        "size": [430.0, 120.0],
        "flags": {},
        "order": 25,
        "mode": 0,
        "inputs": [{"localized_name": "cine_linx", "name": "cine_linx", "type": "IAMCCS_SUPERNODE_LINX", "link": 33}],
        "outputs": [{"localized_name": name, "name": name, "type": node_type, "links": []} for name, node_type in BRIDGE_OUTPUTS],
        "properties": {"Node name for S&R": "IAMCCS_WanAnimate2ShotboardBridge"},
        "widgets_values": [],
        "title": "WAN ANIMATE 2 EXECUTABLE SHOTBOARD BRIDGE",
        "color": "#1d4853",
        "bgcolor": "#0d252d",
    }
    workflow["nodes"].append(bridge)
    nodes_by_id[26] = bridge

    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    for link_id, (source_id, source_slot, target_id, target_slot, link_type) in enumerate(MAPPINGS, start=33):
        workflow["links"].append([link_id, source_id, source_slot, target_id, target_slot, link_type])
        nodes_by_id[source_id]["outputs"][source_slot]["links"].append(link_id)
        nodes_by_id[target_id]["inputs"][target_slot]["link"] = link_id

    workflow["last_node_id"] = 26
    workflow["last_link_id"] = 32 + len(MAPPINGS)
    workflow["extra"]["iamccs"].update({
        "workflow_version": "1.1.0-executable-bridge",
        "bridge": "IAMCCS_WanAnimate2ShotboardBridge",
    })
    DESTINATION.write_text(json.dumps(workflow, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()