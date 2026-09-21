"""Mechanically connect Atomic motion_state to Fast Latent in R42/R43 variants."""

import json
from pathlib import Path


ROOT = Path(r"X:\1_UNIVERSAL_42_43")
FILES = (
    "A_IAMCCS_H3_MINIMAX_R42_UNIVERSAL_180926.json",
    "B_IAMCCS_H3_R43_KEYFRAME_JOINT_LATENT_NEW_UNIVERSAL_FIXED_MOTION_STATE.json",
    "D_IAMCCS_H3_R42_UNIVERSAL_VIGGLE_VIDEO_EDITOR.json",
    "E_IAMCCS_H3_R43_UNIVERSAL_VIGGLE_VIDEO_EDITOR.json",
    "G_IAMCCS_H3_R42_UNIVERSAL_SCOUT_VIGGLE_VIDEO_EDITOR.json",
    "H_IAMCCS_H3_R43_UNIVERSAL_SCOUT_VIGGLE_VIDEO_EDITOR.json",
)


def connect(path: Path) -> None:
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    source = nodes[9]
    target = nodes[800]
    source_slot = next(i for i, item in enumerate(source["outputs"]) if item["name"] == "motion_state")
    target_slot = next(i for i, item in enumerate(target["inputs"]) if item["name"] == "motion_state")
    existing = target["inputs"][target_slot].get("link")
    if existing is not None:
        link = next(item for item in workflow["links"] if item[0] == existing)
        if link[1:5] != [9, source_slot, 800, target_slot]:
            raise ValueError(f"{path.name}: motion_state already has an unexpected source")
        return
    next_link = max([int(item[0]) for item in workflow.get("links", [])] + [int(workflow.get("last_link_id", 0))]) + 1
    workflow["links"].append([next_link, 9, source_slot, 800, target_slot, "IAMCCS_H3_MOTION_CONTEXT"])
    target["inputs"][target_slot]["link"] = next_link
    links = source["outputs"][source_slot].get("links")
    if not isinstance(links, list):
        links = []
        source["outputs"][source_slot]["links"] = links
    links.append(next_link)
    workflow["last_link_id"] = next_link
    path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"connected {path.name}: link {next_link}")


if __name__ == "__main__":
    for filename in FILES:
        connect(ROOT / filename)
