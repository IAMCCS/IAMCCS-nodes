"""Build a delivery-only API workflow from a failed H3/LTX API prompt."""

import argparse
from copy import deepcopy
import json
from pathlib import Path


def build(prompt, native_file):
    prompt = deepcopy(prompt)
    native_path = Path(native_file).resolve(strict=True)
    metadata = json.loads(native_path.with_suffix(native_path.suffix + ".iamccs.json").read_text(encoding="utf-8"))
    source = "iamccs_ltx_resume"
    while source in prompt:
        source += "_source"
    sinks = [key for key, node in prompt.items()
             if node.get("class_type") == "IAMCCS_MiniMaxH3UniversalMasterSaveR42"]
    if len(sinks) != 1:
        raise ValueError("Expected one IAMCCS LTX master delivery save node")
    sink = sinks[0]
    prompt[source] = {"class_type": "IAMCCS_MiniMaxH3LTXResumeSource", "inputs": {
        "native_file": str(native_path),
        "upscale_prompt": "Preserve the source video's exact shot order, hard cuts, framing, faces, clothing, lighting, lip motion and performance. Recover fine detail without redesigning subjects or changing camera motion.",
    }, "_meta": {"title": "RESUME SAVED H3 · LTX ONLY"}}
    for node in prompt.values():
        values = node.get("inputs", {})
        if "cine_linx" in values:
            values["cine_linx"] = [source, 2]
        if node.get("class_type") == "IAMCCS_MiniMaxH3PostUpscaleControlV2":
            values.update(native_frames=[source, 0], master_ready=[source, 3], segment_index=0)
    prompt[sink]["inputs"].update(master_audio=[source, 1], master_ready=[source, 3],
                                  resolved_render_id=[source, 4], fps=[source, 5],
                                  current_segment=0, total_segments=1,
                                  filename_prefix="IAMCCS/MiniMaxH3/LTX_RESUME")
    needed = set()
    pending = [sink]
    while pending:
        key = pending.pop()
        if key in needed:
            continue
        needed.add(key)
        for value in prompt[key].get("inputs", {}).values():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in prompt:
                pending.append(str(value[0]))
    result = {key: node for key, node in prompt.items() if key in needed}
    for node in result.values():
        if node.get("class_type") == "SamplerCustomAdvanced":
            node["class_type"] = "IAMCCS_MiniMaxH3LTXTiledSampler"
            node["inputs"]["cine_linx"] = [source, 2]
        if node.get("class_type") == "LTXVConditioning":
            node["inputs"]["frame_rate"] = float(metadata["fps"])
    forbidden = {"IAMCCS_MiniMaxH3AtomicGenerationBackendV2", "IAMCCS_MiniMaxH3AtomicConditioningBackend",
                 "IAMCCS_MiniMaxH3ShotPlanner", "IAMCCS_ShotboardH3SettingsPro"}
    if any(node.get("class_type") in forbidden for node in result.values()):
        raise ValueError("Recovery graph still depends on native H3 generation")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("api_prompt", type=Path)
    parser.add_argument("native_file", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = build(json.loads(args.api_prompt.read_text(encoding="utf-8")), args.native_file)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved delivery-only API workflow: {args.output} ({len(result)} nodes)")
