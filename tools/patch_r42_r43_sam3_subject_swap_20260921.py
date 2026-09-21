"""Mechanically install/update the lazy SAM3 Subject Swap rig in active R42/R43 workflows."""
from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(r"X:\1_UNIVERSAL_42_43")
TARGETS = [
    "A_IAMCCS_H3_MINIMAX_R42_UNIVERSAL_180926.json",
    "B_IAMCCS_H3_R43_KEYFRAME_JOINT_LATENT_NEW_UNIVERSAL_FIXED_MOTION_STATE.json",
    "D_IAMCCS_H3_R42_UNIVERSAL_VIGGLE_VIDEO_EDITOR.json",
    "E_IAMCCS_H3_R43_UNIVERSAL_VIGGLE_VIDEO_EDITOR.json",
    "G_IAMCCS_H3_R42_UNIVERSAL_SCOUT_VIGGLE_VIDEO_EDITOR.json",
    "H_IAMCCS_H3_R43_UNIVERSAL_SCOUT_VIGGLE_VIDEO_EDITOR.json",
    "I_IAMCCS_H3_R42_UNIVERSAL_PROMPTER_AUDIO_TEXT_RIG.json",
]


def node(graph, node_id):
    return next((item for item in graph["nodes"] if int(item.get("id", -1)) == node_id), None)


def patch(path: Path, template: dict) -> None:
    graph = json.loads(path.read_text(encoding="utf-8"))
    planner = node(graph, 7)
    cine_info = node(graph, 586)
    if planner is None or cine_info is None:
        raise RuntimeError(f"{path.name}: expected ShotPlanner 7 and CineInfo 586")

    face = next((item for item in graph["nodes"] if item.get("type") == "IAMCCS_H3FaceSwapInput"), None)
    if face is None:
        if node(graph, 819) is not None or any(int(link[0]) == 1360 for link in graph["links"]):
            raise RuntimeError(f"{path.name}: IDs 819/1360 are occupied")
        face = copy.deepcopy(template)
        graph["nodes"].append(face)
        source_link = next((link for link in graph["links"] if int(link[0]) == 1127), None)
        if source_link is None or int(source_link[1]) != 7 or int(source_link[3]) != 586:
            raise RuntimeError(f"{path.name}: link 1127 is not ShotPlanner -> CineInfo")
        source_link[1] = 819
        source_link[2] = 0
        graph["links"].append([1360, 7, 0, 819, 0, "IAMCCS_SUPERNODE_LINX"])
        for output in planner.get("outputs", []):
            links = output.get("links")
            if isinstance(links, list) and 1127 in links:
                output["links"] = [1360 if int(value) == 1127 else value for value in links]
                break

    face["title"] = "IAMCCS SAM3 SUBJECT SWAP · SOURCE VIDEO + PICTURE 1"
    face["size"] = [500, 220]

    settings = node(graph, 811)
    if settings is not None:
        values = settings.get("widgets_values") or []
        # Current compatible schema keeps the SAM3 threshold immediately after
        # the `head` prompt. This assertion prevents accidental positional edits.
        matches = [i for i, value in enumerate(values[:-1]) if value == "head" and isinstance(values[i + 1], (int, float))]
        if len(matches) != 1:
            raise RuntimeError(f"{path.name}: could not identify SAM3 threshold safely")
        values[matches[0] + 1] = 0.3
        named = (settings.get("properties") or {}).get("widgets_values_named")
        if isinstance(named, dict) and "h3_faceswap_threshold" in named:
            named["h3_faceswap_threshold"] = 0.3

    graph["last_node_id"] = max(int(graph.get("last_node_id", 0)), 819)
    graph["last_link_id"] = max(int(graph.get("last_link_id", 0)), 1360)
    path.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"patched {path.name}")


def main() -> None:
    template_graph = json.loads((ROOT / TARGETS[1]).read_text(encoding="utf-8"))
    template = next(item for item in template_graph["nodes"] if item.get("type") == "IAMCCS_H3FaceSwapInput")
    for name in TARGETS:
        patch(ROOT / name, template)


if __name__ == "__main__":
    main()
