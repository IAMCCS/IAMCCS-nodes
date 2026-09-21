"""Synchronise smoke A with the user's 864x480 I2V smoke B without touching prompts."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(r"X:\1_UNIVERSAL_42_43\SMOKE_R42_I2V_LATENT_CONTINUATION_GOYAI")
PATHS = [
    ROOT / "A1_IAMCCS_H3_R42_I2V_CLIP_A_SAVE_TERMINAL_AV_SMOKE.json",
    ROOT / "A2_IAMCCS_H3_R42_I2V_CLIP_B_CONTINUE_FROM_AV_SMOKE.json",
]


def update_settings_dict(value):
    if isinstance(value, dict):
        for key in list(value):
            if key in {"width", "image_width"} and value[key] == 640:
                value[key] = 864
            elif key in {"height", "image_height"} and value[key] == 384:
                value[key] = 480
            elif key == "h3_continuation_handover_mode":
                value[key] = "terminal"
            elif key == "h3_faceswap_threshold" and value[key] == 0.5:
                value[key] = 0.3
            else:
                update_settings_dict(value[key])
    elif isinstance(value, list):
        for item in value:
            update_settings_dict(item)


def sync(path):
    graph = json.loads(path.read_text(encoding="utf-8"))
    settings = next(node for node in graph["nodes"] if int(node.get("id", -1)) == 811)
    values = settings["widgets_values"]
    if values[8:12] not in ([640, 384, 640, 384], [864, 480, 864, 480]):
        raise RuntimeError(f"Unexpected A resolution widgets: {values[8:12]}")
    values[8:12] = [864, 480, 864, 480]
    if values[159] not in {"auto", "terminal"}:
        raise RuntimeError(f"Unexpected A handover widget: {values[159]}")
    values[159] = "terminal"
    values[160] = 0
    values[171] = 0.3
    run_and_gun = path.name.startswith("A2_")
    if len(values) == 187:
        values.extend(["latent_only", run_and_gun])
    else:
        values[187:189] = ["latent_only", run_and_gun]
    named = settings.get("widgets_values_named")
    if isinstance(named, dict):
        named.update(
            width=864,
            height=480,
            image_width=864,
            image_height=480,
            h3_continuation_handover_mode="terminal",
            h3_continuation_manual_tail_frames=0,
            h3_continuation_visual_handover="latent_only",
            h3_continuation_run_and_gun_enabled=run_and_gun,
            h3_faceswap_threshold=0.3,
        )
    update_settings_dict(settings.get("properties", {}))

    # Shotboard timeline JSON embeds the same saved settings. Parse only JSON
    # widgets and update named resolution/settings fields; prompts remain exact.
    for node in graph["nodes"]:
        widgets = node.get("widgets_values")
        if not isinstance(widgets, list):
            continue
        for index, raw in enumerate(widgets):
            if not isinstance(raw, str) or not raw.lstrip().startswith("{"):
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            update_settings_dict(payload)
            widgets[index] = json.dumps(payload, ensure_ascii=False, indent=2)

    path.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"synchronised {path.name}")


def main():
    for path in PATHS:
        sync(path)


if __name__ == "__main__":
    main()
