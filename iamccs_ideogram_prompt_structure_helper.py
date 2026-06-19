import json


IDEO_GUIDE = """Ideogram 4 guide for IdeoboardPrompter:
- Use either text_prompt or json_prompt, never both.
- For controlled storyboard sheets, use json_prompt. Ideogram 4 consumes it directly and magic-prompt is disabled.
- Use: high_level_description, style_description, compositional_deconstruction.background, compositional_deconstruction.elements.
- Do not invent a negative_prompt key. Put concise negative guards inside background and each panel desc.
- Panel formula: title + shot scale + subject/action + environment + camera/lens/film language + continuity guard + targeted negative guard.
- Keep negatives targeted: no duplicate protagonist, subject absent from this panel, no poster typography, no extra panels.
- One element per panel. The bbox defines the panel region; the desc defines what happens inside it.
"""


def _colors(raw):
    out = []
    for item in str(raw or "").split(","):
        value = item.strip().upper()
        if not value:
            continue
        if not value.startswith("#"):
            value = "#" + value
        if len(value) == 7:
            out.append(value)
    return out or ["#0A0D0F", "#263238", "#D6D0C2", "#8B1E2D"]


def _bbox(columns, rows, i):
    r = i // columns
    c = i % columns
    return [
        round((r / rows) * 1000),
        round((c / columns) * 1000),
        round(((r + 1) / rows) * 1000),
        round(((c + 1) / columns) * 1000),
    ]


def _fallback_title(i):
    return f"Panel {i + 1}"


def _panel_desc(i, title, prompt):
    title = str(title or _fallback_title(i)).strip()
    prompt = str(prompt or "").strip()
    if not prompt:
        prompt = "shot scale, subject/action, environment, camera/lens/film language, continuity guard, targeted negative guard"
    if prompt.lower().startswith("panel "):
        return prompt
    return f"Panel {i + 1}, {title}: {prompt}"


def _panel_inputs():
    inputs = {}
    for i in range(12):
        n = i + 1
        inputs[f"panel_{n:02d}_title"] = ("STRING", {"default": f"Panel {n}"})
        inputs[f"panel_{n:02d}_prompt"] = ("STRING", {
            "multiline": True,
            "default": "shot scale, subject/action, environment, camera/lens/film language, continuity guard, targeted negative guard",
        })
    return inputs


class IAMCCS_IdeoboardPrompter:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "user_brief": ("STRING", {"multiline": True, "default": "Write the story premise for the storyboard sheet."}),
            "columns": ("INT", {"default": 2, "min": 1, "max": 12}),
            "rows": ("INT", {"default": 3, "min": 1, "max": 12}),
            "high_level_description": ("STRING", {"multiline": True, "default": "2x3 Ideogram 4 storyboard sheet, exactly six panels, cinematic live-action realism."}),
            "background": ("STRING", {"multiline": True, "default": "Clean storyboard sheet, exact panel count, thin grid lines, no UI overlay, no extra panels, no poster typography."}),
            "aesthetics": ("STRING", {"multiline": True, "default": "photoreal live-action cinema, real actors, practical sets, no illustration, no concept art"}),
            "lighting": ("STRING", {"multiline": True, "default": "cinematic lighting, atmospheric haze, controlled contrast, film grain"}),
            "photo": ("STRING", {"multiline": True, "default": "anamorphic 35mm film still, practical set photography, optical diffusion"}),
            "medium": ("STRING", {"default": "photograph"}),
            "negative_rules": ("STRING", {"multiline": True, "default": "no duplicate protagonist, no extra panels, no repeated identical face, no poster typography, no UI overlay"}),
            "palette": ("STRING", {"default": "#0A0D0F, #263238, #D6D0C2, #8B1E2D"}),
        }
        required.update(_panel_inputs())
        return {"required": required}

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "gemma_final_json_request",
        "sheetbuilder_import_json",
        "panel_prompt_templates",
        "grid_columns",
        "grid_rows",
        "bbox_guide",
        "guide_text",
        "usage_notes",
    )
    FUNCTION = "build"
    CATEGORY = "IAMCCS/Cine/Ideogram"

    def build(self, user_brief, columns=2, rows=3, high_level_description="", background="", aesthetics="", lighting="", photo="", medium="photograph", negative_rules="", palette="", **kwargs):
        columns = max(1, int(columns))
        rows = max(1, int(rows))
        total = min(columns * rows, 12)
        colors = _colors(palette)
        elements = []
        panels = []
        bbox_lines = []
        for i in range(total):
            title = kwargs.get(f"panel_{i + 1:02d}_title") or _fallback_title(i)
            prompt = kwargs.get(f"panel_{i + 1:02d}_prompt") or ""
            bb = _bbox(columns, rows, i)
            desc = _panel_desc(i, title, prompt)
            if negative_rules and "no duplicate" not in desc.lower():
                desc = f"{desc}. Targeted guards: {negative_rules}"
            elements.append({"type": "obj", "bbox": bb, "desc": desc, "color_palette": colors[:5]})
            panels.append({
                "panel": i + 1,
                "title": title,
                "prompt": prompt,
                "final_desc": desc,
                "bbox": bb,
                "how_to_write": "title + shot scale + subject/action + environment + camera/lens/film language + continuity guard + targeted negative guard",
            })
            bbox_lines.append(f"Panel {i + 1}: {bb}")

        sheet = {
            "high_level_description": high_level_description,
            "style_description": {
                "aesthetics": aesthetics,
                "lighting": lighting,
                "photo": photo,
                "medium": medium or "photograph",
                "color_palette": colors[:6],
            },
            "compositional_deconstruction": {
                "background": background,
                "elements": elements,
            },
        }
        request = (
            "You are IdeoboardPrompter Gemma Assist. Return one final Ideogram 4 json_prompt object only. "
            "No Markdown, no explanation, no negative_prompt key.\n\n"
            "Rewrite and improve the sheet JSON below using official Ideogram 4 structure:\n"
            "- high_level_description\n- style_description\n- compositional_deconstruction.background\n- compositional_deconstruction.elements\n\n"
            f"User brief:\n{user_brief}\n\n"
            f"Grid: {columns}x{rows}, exactly {total} panels. Keep these exact bboxes:\n" + "\n".join(bbox_lines) + "\n\n"
            "Panel writing rule: title + shot scale + subject/action + environment + camera/lens/film language + continuity guard + targeted negative guard.\n"
            f"Targeted negative guards:\n{negative_rules}\n\n"
            "Draft JSON to improve:\n"
            f"{json.dumps(sheet, indent=2, ensure_ascii=False)}"
        )
        usage = (
            "Connect sheetbuilder_import_json to Sheet Builder import_json. "
            "Connect grid_columns/grid_rows to Sheet Builder and cropper. "
            "Write panel titles and prompts here first; Sheet Builder remains the visual editor/final JSON surface."
        )
        return (
            request,
            json.dumps(sheet, indent=2, ensure_ascii=False),
            json.dumps(panels, indent=2, ensure_ascii=False),
            columns,
            rows,
            "\n".join(bbox_lines),
            IDEO_GUIDE,
            usage,
        )


IAMCCS_IdeogramPromptStructureHelper = IAMCCS_IdeoboardPrompter

NODE_CLASS_MAPPINGS = {
    "IAMCCS_IdeoboardPrompter": IAMCCS_IdeoboardPrompter,
    "IAMCCS_IdeogramPromptStructureHelper": IAMCCS_IdeoboardPrompter,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_IdeoboardPrompter": "IAMCCS IdeoboardPrompter",
    "IAMCCS_IdeogramPromptStructureHelper": "IAMCCS IdeoboardPrompter",
}
