"""Repair saved H3 widget positions and remove the broken Viggle demo branch.

Run with the ComfyUI Python environment. Exact input files are listed below;
each original receives a .pre_schema_fix_20260920.json backup first.
"""

import importlib
import json
import math
from pathlib import Path
import shutil
import sys
import types


REPO = Path(__file__).resolve().parent
ROOT = Path(r"X:\1_UNIVERSAL_42_43")
FILES = (
    ROOT / "SMOKE/CONTROLNET/IAMCCS_H3_CONTROLNET_CANNY_R42_SMOKE_243F.json",
    ROOT / "SMOKE/CONTROLNET/IAMCCS_H3_CONTROLNET_CANNY_REF2VA_R42_SMOKE_124F.json",
    ROOT / "A_IAMCCS_H3_MINIMAX_R42_UNIVERSAL_180926.json",
    ROOT / "B_IAMCCS_H3_R43_KEYFRAME_JOINT_LATENT_NEW_UNIVERSAL_FIXED_MOTION_STATE.json",
    ROOT / "D_IAMCCS_H3_R42_UNIVERSAL_VIGGLE_VIDEO_EDITOR.json",
    ROOT / "E_IAMCCS_H3_R43_UNIVERSAL_VIGGLE_VIDEO_EDITOR.json",
    ROOT / "F_IAMCCS_H3_KEYFRAME_JOINT_LATENT_NEW_EDITOR_JUNCTION_SUBWATER.json",
    ROOT / "G_IAMCCS_H3_R42_UNIVERSAL_SCOUT_VIGGLE_VIDEO_EDITOR.json",
    ROOT / "H_IAMCCS_H3_R43_UNIVERSAL_SCOUT_VIGGLE_VIDEO_EDITOR.json",
    ROOT / "I_IAMCCS_H3_R42_UNIVERSAL_PROMPTER_AUDIO_TEXT_RIG.json",
)


def settings_specs():
    sys.path.insert(0, str(REPO.parents[1]))
    package = types.ModuleType("iamccs_schema_repair")
    package.__path__ = [str(REPO)]
    sys.modules[package.__name__] = package
    module = importlib.import_module(package.__name__ + ".iamccs_minimax_h3_shotboard")
    return {
        "IAMCCS_ShotboardH3Settings": module.IAMCCS_ShotboardH3Settings.INPUT_TYPES(),
        "IAMCCS_ShotboardH3SettingsPro": module.IAMCCS_ShotboardH3SettingsPro.INPUT_TYPES(),
    }


def canonical_value(raw, spec):
    kind, options = spec[0], spec[1]
    default = options.get("default", kind[0] if isinstance(kind, list) else 0 if kind in ("INT", "FLOAT") else "")
    if isinstance(kind, list):
        if raw in kind or (kind == [""] and isinstance(raw, str)):
            return raw
        return default
    if kind in ("INT", "FLOAT"):
        try:
            if raw is None or raw == "" or isinstance(raw, bool):
                raise ValueError("blank or boolean numeric widget")
            value = float(raw)
            if not math.isfinite(value) or (kind == "INT" and not value.is_integer()):
                raise ValueError("non-integral or non-finite widget")
            if value < options.get("min", -math.inf) or value > options.get("max", math.inf):
                raise ValueError("out of bounds widget")
            return int(value) if kind == "INT" else value
        except (TypeError, ValueError):
            return default
    return raw if raw is not None else default


def repair_settings(workflow, schemas):
    changed = []
    for node in workflow["nodes"]:
        if node["type"] not in schemas:
            continue
        schema = schemas[node["type"]]
        specs = {**schema["required"], **schema["optional"]}
        named = node.get("widgets_values_named")
        if not isinstance(named, dict) or not named:
            raise ValueError(f"Settings node {node['id']} has no named values; refusing positional guess")
        normalized = {name: canonical_value(named.get(name, spec[1].get("default")), spec)
                      for name, spec in specs.items()}
        node["widgets_values"] = list(normalized.values())
        node["widgets_values_named"] = normalized
        node.setdefault("properties", {})["iamccs_h3_settings_schema"] = list(specs)
        changed.append((node["id"], len(normalized)))
    return changed


def remove_viggle(workflow):
    selectors = [n for n in workflow["nodes"] if n["type"] == "IAMCCS_MiniMaxH3UniversalViggleSelectorR42"]
    if len(selectors) != 1:
        raise ValueError("Expected one Viggle selector")
    selector = selectors[0]
    selector_id = selector["id"]
    first = selector_id - 13
    remove_ids = set(range(first, selector_id + 1))
    by_id = {n["id"]: n for n in workflow["nodes"]}
    if not all(item in by_id for item in remove_ids):
        raise ValueError("Viggle branch does not match known R42/R43 graph")
    links = workflow["links"]
    standard = {link[4]: link for link in links if link[3] == selector_id and link[4] <= 7}
    if set(standard) != set(range(8)):
        raise ValueError("Viggle selector standard inputs are incomplete")
    for link in links:
        if link[1] == selector_id:
            if link[2] > 7:
                raise ValueError("Viggle report has a live consumer")
            upstream = standard[link[2]]
            link[1], link[2], link[5] = upstream[1], upstream[2], upstream[5]
        elif link[1] in remove_ids and link[3] not in remove_ids:
            raise ValueError(f"Viggle support node {link[1]} has a non-Viggle consumer")
    workflow["links"] = [link for link in links if link[1] not in remove_ids and link[3] not in remove_ids]
    workflow["nodes"] = [node for node in workflow["nodes"] if node["id"] not in remove_ids]

    nodes = {node["id"]: node for node in workflow["nodes"]}
    for node in nodes.values():
        for item in node.get("inputs", []):
            item["link"] = None
        for item in node.get("outputs", []):
            item["links"] = None
    for link in workflow["links"]:
        link_id, source, output, target, input_slot, _ = link
        if source not in nodes or target not in nodes:
            raise ValueError(f"Dangling link {link_id}")
        nodes[source]["outputs"][output].setdefault("links", [])
        if nodes[source]["outputs"][output]["links"] is None:
            nodes[source]["outputs"][output]["links"] = []
        nodes[source]["outputs"][output]["links"].append(link_id)
        nodes[target]["inputs"][input_slot]["link"] = link_id
    return len(remove_ids)


def verify(workflow, schemas):
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow.get("links", [])}
    for node in nodes.values():
        if node["type"] in schemas:
            schema = schemas[node["type"]]
            specs = {**schema["required"], **schema["optional"]}
            named = node["widgets_values_named"]
            if list(named) != list(specs) or node["widgets_values"] != list(named.values()):
                raise ValueError(f"Settings {node['id']} has shifted widget positions")
            for name, spec in specs.items():
                if canonical_value(named[name], spec) != named[name]:
                    raise ValueError(f"Settings {node['id']} has invalid {name}")
        for slot, item in enumerate(node.get("inputs", [])):
            link_id = item.get("link")
            if link_id is not None and (link_id not in links or links[link_id][3:5] != [node["id"], slot]):
                raise ValueError(f"Broken input link {link_id}")
        for slot, item in enumerate(node.get("outputs", [])):
            for link_id in item.get("links") or []:
                if link_id not in links or links[link_id][1:3] != [node["id"], slot]:
                    raise ValueError(f"Broken output link {link_id}")
    for link in links.values():
        if link[1] not in nodes or link[3] not in nodes:
            raise ValueError(f"Dangling link {link[0]}")


def fallback_missing_fasth3(workflow):
    settings = [node for node in workflow["nodes"] if node["type"] in (
        "IAMCCS_ShotboardH3Settings", "IAMCCS_ShotboardH3SettingsPro")]
    if not settings or not any(node["widgets_values_named"].get("acceleration") == "fasth3_dense_6step"
                               and not node["widgets_values_named"].get("turbo_lora_name") for node in settings):
        return False
    for node in workflow["nodes"]:
        if node["type"] not in ("IAMCCS_ShotboardH3Settings", "IAMCCS_ShotboardH3SettingsPro",
                                "IAMCCS_MiniMaxH3ShotPlanner"):
            continue
        named = node.get("widgets_values_named")
        if not isinstance(named, dict):
            continue
        for field, value in (("acceleration", "native"), ("steps", 16), ("turbo_lora_name", "")):
            if field not in named:
                continue
            named[field] = value
            index = list(named).index(field)
            if index < len(node.get("widgets_values", [])):
                node["widgets_values"][index] = value
        if node["type"] == "IAMCCS_MiniMaxH3ShotPlanner" and named.get("timeline_data"):
            timeline = json.loads(named["timeline_data"])
            saved = timeline.get("h3_saved_settings")
            if isinstance(saved, dict):
                saved.update(acceleration="native", steps=16, turbo_lora_name="")
                text = json.dumps(timeline, ensure_ascii=False, indent=2)
                named["timeline_data"] = text
                index = list(named).index("timeline_data")
                node["widgets_values"][index] = text
    return True


def main():
    schemas = settings_specs()
    for path in FILES:
        backup = path.with_name(path.stem + ".pre_schema_fix_20260920.json")
        if backup.exists():
            workflow = json.loads(path.read_text(encoding="utf-8"))
            if fallback_missing_fasth3(workflow):
                path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            verify(workflow, schemas)
            print(f"Verified previously repaired: {path}")
            continue
        workflow = json.loads(path.read_text(encoding="utf-8"))
        settings = repair_settings(workflow, schemas)
        removed = remove_viggle(workflow) if any(n["type"] == "IAMCCS_MiniMaxH3UniversalViggleSelectorR42" for n in workflow["nodes"]) else 0
        for node in workflow["nodes"]:
            if node["type"] in ("IAMCCS_ShotboardVideoEditorRenderV1", "IAMCCS_shotboarder_aud+vid_exporter_PRO"):
                node["mode"] = 2
        fallback_missing_fasth3(workflow)
        verify(workflow, schemas)
        shutil.copy2(path, backup)
        path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{path}: Settings {settings}, Viggle nodes removed {removed}; backup {backup}")


if __name__ == "__main__":
    main()
