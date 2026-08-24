# SPDX-License-Identifier: GPL-3.0-or-later

"""Structured MiniMax H3 prompt editor and CineLinX injection contract.

The browser editor stores only user-authored project data. The deterministic
path formats MiniMax prompt sections and carries an injection request through
CineLinX. The optional assistant is implemented locally in this module and can
call Ollama or a user-selected compatible provider without wrapping another
custom-node package.
"""

from __future__ import annotations

import copy
import asyncio
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


SUPERNODE_LINX_TYPE = "IAMCCS_SUPERNODE_LINX"
CATEGORY = "IAMCCS/MiniMax H3/Prompting"
PROJECT_SCHEMA = "iamccs.minimax_h3.prompter_project"
PROJECT_VERSION = 3
H3_ABSOLUTE_CHAR_LIMIT = 7000
AI_IMAGE_LIMIT = 4
AI_IMAGE_MAX_BYTES = 16 * 1024 * 1024
AUDIO_HANDOFF_AUTHORING_RULE = (
    "Never carry dialogue or a new vocalisation across two independently generated chunks. "
    "For every non-final chunk, finish all dialogue and shouts at least 1.00 second before the end; "
    "reserve the final 1.00 second for continuous ambience, physical action sounds and quiet natural breathing. "
    "Every following chunk must also reserve its first 1.00 second for that same ambience and continued physical action before any new line starts. "
    "Do not impose this restriction on the final or only chunk."
)
TASK_MODE_ALIASES = {
    "v2v_object_swap": "v2va_object_swap",
    "v2va": "v2va_object_swap",
    "object_swap": "v2va_object_swap",
}


def _normalise_task_mode(value: Any) -> str:
    mode = str(value or "t2va").strip().lower()
    return TASK_MODE_ALIASES.get(mode, mode)


MODE_SECTIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "t2va": (
        ("scene", "SCENE"),
        ("shot_list", "SHOT LIST"),
        ("acting", "ACTING"),
        ("dialogue", "DIALOGUE"),
        ("light_and_image", "LIGHT AND IMAGE"),
        ("camera", "CAMERA"),
        ("production_sound", "PRODUCTION SOUND"),
        ("non_diegetic_music", "NON-DIEGETIC MUSIC"),
        ("negatives", "NEGATIVES"),
    ),
    "i2va": (
        ("reference_use", "REFERENCE USE"),
        ("identity_continuity_locks", "IDENTITY / CONTINUITY LOCKS"),
        ("scene", "SCENE"),
        ("shot_list", "SHOT LIST"),
        ("acting", "ACTING"),
        ("dialogue", "DIALOGUE"),
        ("light_and_image", "LIGHT AND IMAGE"),
        ("camera", "CAMERA"),
        ("production_sound", "PRODUCTION SOUND"),
        ("non_diegetic_music", "NON-DIEGETIC MUSIC"),
        ("negatives", "NEGATIVES"),
    ),
    "fl2va": (
        ("boundary_frames", "BOUNDARY FRAMES"),
        ("reference_use", "REFERENCE USE"),
        ("identity_continuity_locks", "IDENTITY / CONTINUITY LOCKS"),
        ("action", "ACTION"),
        ("shot_list", "SHOT LIST"),
        ("acting", "ACTING"),
        ("dialogue", "DIALOGUE"),
        ("light_and_image", "LIGHT AND IMAGE"),
        ("camera", "CAMERA"),
        ("production_sound", "PRODUCTION SOUND"),
        ("non_diegetic_music", "NON-DIEGETIC MUSIC"),
        ("negatives", "NEGATIVES"),
    ),
    "ref2va": (
        ("subject_definitions", "subject_definitions"),
        ("summary", "summary"),
        ("retention_analysis", "retention_analysis"),
        ("detailed_description", "detailed_description"),
        ("overall_soundscape", "overall_soundscape"),
        ("non_diegetic_music", "non_diegetic_music"),
    ),
    "v2va_object_swap": (
        ("v2va_subject_definitions", "SUBJECT DEFINITIONS"),
        ("v2va_source_video_authority", "SOURCE VIDEO 1 AUTHORITY"),
        ("v2va_replacement_retention", "REPLACEMENT / RETENTION ANALYSIS"),
        ("v2va_interval_edits", "INTERVAL EDIT INSTRUCTIONS"),
        ("v2va_sound_policy", "SOUND POLICY"),
        ("v2va_exclusions", "EXCLUSIONS / CONTINUITY SAFEGUARDS"),
    ),
    "audio_driven": (
        ("audio_drive_contract", "AUDIO DRIVE CONTRACT"),
        ("audio_subject_map", "SUBJECT / SPEAKER MAP"),
        ("audio_scene_intent", "SCENE INTENT"),
        ("audio_timed_performance", "TIMED PERFORMANCE"),
        ("audio_dialogue_map", "DIALOGUE MAP"),
        ("audio_visual_sync", "VISIBLE SYNC CUES"),
        ("audio_camera_sync", "CAMERA"),
        ("audio_environment", "ENVIRONMENT SOUND"),
        ("audio_continuity_locks", "CONTINUITY SAFEGUARDS"),
    ),
}


DEFAULT_SECTIONS = {
    "scene": "A rain-polished railway platform before sunrise. <Subject 1>, a tired courier in a charcoal coat, waits beside a silver case while an empty train approaches through blue mist.",
    "shot_list": "0.00-2.00s: hold a medium-wide profile. 2.00-4.50s: the train enters and throws moving reflections across the platform. 4.50s-end: <Subject 1> turns toward camera and grips the case.",
    "acting": "Restrained performance: shoulders tense first, then the eyes react, then one deliberate turn. Preserve natural blinking and breathing.",
    "dialogue": "<Subject 1> (S1): <d>[English] Not this train.</d>",
    "light_and_image": "Cool dawn ambience, practical sodium lamps, wet reflections, restrained contrast, realistic skin texture, cinematic depth without artificial glow.",
    "camera": "One slow lateral tracking move at chest height with mild foreground parallax; no cut and no change of lens language.",
    "production_sound": "Distant rail vibration, light rain on metal roofing, one approaching brake squeal, coat movement, clear close dialogue with matching platform reverb.",
    "non_diegetic_music": "A sparse low cello pulse enters only after the train becomes visible; keep it separate from the physical scene sound.",
    "negatives": "No identity drift, no duplicate people, no wardrobe change, no warped hands, no sudden zoom, no jump cut, no subtitles, no logo.",
    "reference_use": "Use <Picture 1> as the complete opening-frame authority for identity, wardrobe, composition, lens perspective, lighting direction and visible environment. Animate from it rather than redesigning it.",
    "identity_continuity_locks": "Keep <Subject 1>'s face, hairline, coat, silver case, body proportions and screen side unchanged. Preserve the platform geometry and time of day.",
    "boundary_frames": "Open exactly on <Picture 1> and arrive naturally at <Picture 2> as the final composition. Treat both pictures as full-frame boundaries, not loose style references.",
    "action": "The character crosses the connected space in one continuous action. Movement should develop physically toward the final pose with stable identity and coherent screen direction.",
    "subject_definitions": "<Subject 1>: the principal performer shown in <Picture 1>; preserve face, body proportions, wardrobe and signature accessories.\n<Subject 2>: the compact silver case; preserve its shape, scale, surface marks and position relative to <Subject 1>.",
    "summary": "A tense cinematic beat in which <Subject 1> notices an approaching threat while protecting <Subject 2>. The result should feel observational, grounded and continuous.",
    "retention_analysis": "Retain identity and wardrobe from <Picture 1>. Retain the physical timing and camera rhythm from <Video 1> only where supplied. Use <Audio 1> for voice character or cadence only when it is connected; do not invent an unseen speaker.",
    "detailed_description": "Begin with the supplied reference composition. <Subject 1> hears the approaching train, tightens one hand around <Subject 2>, then turns with a controlled breath. Use a single lateral camera move and preserve spatial geography. If dialogue is desired: <Subject 1> (S1): <d>[English] Not this train.</d>",
    "overall_soundscape": "Layer the location ambience, contact sounds, movement and dialogue in chronological order. Keep perspective and reverberation consistent with camera distance; avoid wall-to-wall effects.",
    # V2VA Object Swap is a text/reference contract. It deliberately makes no
    # ControlNet, mask, depth, pose, segmentation, or tracker claim.
    "v2va_subject_definitions": (
        "<Picture 1> defines the replacement <Subject 1>: [write only visible identity, body, wardrobe, material or object facts that must be preserved].\n"
        "<Video 1> contains source <Subject 2>: [identify exactly what is being replaced]. Each additional <Picture N> may define only the named <Subject N> or continuity attribute."
    ),
    "v2va_source_video_authority": (
        "<Video 1> is the temporal source authority for duration, action timing, body or object motion, camera path, framing, occlusion order, environment and edit rhythm. "
        "Preserve those source relationships unless an interval instruction below explicitly changes one."
    ),
    "v2va_replacement_retention": (
        "Replace source <Subject 2> from <Video 1> with replacement <Subject 1> from <Picture 1>. "
        "Retain [list source environment, secondary subjects, interactions, contact points, lighting response and camera behavior]. "
        "Change only [list the requested identity, object, clothing or appearance attributes]."
    ),
    "v2va_interval_edits": (
        "[Define source-time intervals from <Video 1>, for example 00:00.00-00:02.50, and state the visible replacement action or retained event in each interval. "
        "Leave this field untimed when the edit applies uniformly to the complete source video.]"
    ),
    "v2va_sound_policy": (
        "[State whether connected source-video audio is retained, replaced, muted or supplemented. Name <Audio 1> only when an audio reference is actually connected. "
        "Keep dialogue wording, lip timing, contact sounds and ambience consistent with the chosen policy.]"
    ),
    "v2va_exclusions": (
        "Do not change unselected subjects, source environment, camera trajectory, duration, occlusion order or interactions. "
        "No identity blending between <Subject 1> and <Subject 2>, duplicate replacement, geometry drift, temporal jump, subtitle, logo or invented reference."
    ),
    # R21 audio-drive fields are deliberately content-free. They are a
    # reusable authoring scaffold, not a hardcoded character, story or line.
    "audio_drive_contract": "Treat the connected custom audio as the timing authority. Preserve its order, pauses, breaths and duration; do not invent, remove or reorder speech.",
    "audio_subject_map": "<Subject 1> (S1): [describe the visible speaker and the identity/reference facts that must remain stable].",
    "audio_scene_intent": "[Describe location, time, dramatic purpose and the visible starting situation.]",
    "audio_timed_performance": "[Map audible phrases, pauses and breaths to chronological facial expression, gaze, gesture and body action.]",
    "audio_dialogue_map": "<Subject 1> (S1): <d>[Language] ...</d>",
    "audio_visual_sync": "[Describe visible mouth articulation, breath, contact or musical actions that must synchronize with the connected audio.]",
    "audio_camera_sync": "[Describe one coherent framing and camera move that supports the timed performance without hiding the speaker.]",
    "audio_environment": "[Describe only environmental ambience and contact sounds not already fixed by the custom audio.]",
    "audio_continuity_locks": "[List identity, wardrobe, anatomy, prop, geography, eyeline and lip-visibility facts that cannot drift.]",
}


HEADING_ALIASES = {
    "identity_continuity_locks": "identity_continuity_locks",
    "identity_locks": "identity_continuity_locks",
    "continuity_locks": "identity_continuity_locks",
    "light_image": "light_and_image",
    "light_and_image": "light_and_image",
    "camera_and_sound": "camera",
    "sound": "production_sound",
    **{key: key for key in DEFAULT_SECTIONS},
}


def default_project() -> dict[str, Any]:
    return {
        "schema": PROJECT_SCHEMA,
        "schema_version": PROJECT_VERSION,
        "project_name": "Platform at Blue Hour",
        "task_mode": "t2va",
        "injection_target": "global",
        "writing_mode": "guided",
        "merge_policy": "replace",
        "ai_direction": "",
        "ai_scope": "active_field",
        "ai_visual_roles": {},
        "sections": copy.deepcopy(DEFAULT_SECTIONS),
    }


def _safe_project(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        source = copy.deepcopy(value)
    else:
        raw = str(value or "").strip()
        if not raw:
            source = default_project()
        else:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"IAMCCS_Prompter project JSON non valido: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError("IAMCCS_Prompter project_data deve essere un oggetto JSON")
            source = parsed
    project = default_project()
    project.update({key: value for key, value in source.items() if key != "sections"})
    sections = source.get("sections")
    if isinstance(sections, dict):
        project["sections"].update({str(key): str(value or "") for key, value in sections.items()})
        # R21 audio-drive projects created before the Final Draft field rename
        # used the aliases below.  Preserve their authored content instead of
        # silently falling back to the current bracketed example placeholders.
        def _legacy_text(*keys: str) -> str:
            return "\n".join(
                str(sections.get(key) or "").strip()
                for key in keys
                if str(sections.get(key) or "").strip()
            )

        legacy_audio_aliases = {
            "audio_timed_performance": _legacy_text("audio_timing_map", "audio_performance"),
            "audio_visual_sync": _legacy_text("audio_visible_sync"),
            "audio_camera_sync": _legacy_text("audio_camera"),
            "audio_environment": _legacy_text("audio_ambience"),
            "audio_continuity_locks": _legacy_text("audio_continuity"),
        }
        for current_key, migrated_value in legacy_audio_aliases.items():
            if migrated_value and not str(sections.get(current_key) or "").strip():
                project["sections"][current_key] = migrated_value
    project["ai_direction"] = str(project.get("ai_direction") or "")
    project["ai_scope"] = str(project.get("ai_scope") or "active_field")
    visual_roles = project.get("ai_visual_roles")
    project["ai_visual_roles"] = visual_roles if isinstance(visual_roles, dict) else {}
    project["schema"] = PROJECT_SCHEMA
    project["schema_version"] = PROJECT_VERSION
    project["task_mode"] = _normalise_task_mode(project.get("task_mode"))
    return project


def _normalise_heading(value: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return HEADING_ALIASES.get(clean, clean)


def _parse_assistant_draft(value: str) -> dict[str, str]:
    """Parse the headings emitted by common H3 prompting assistants.

    Manual fields always win.  This parser therefore only needs to recover
    recognisable sections so an optional assistant can fill blank boxes.
    """
    text = str(value or "").strip()
    if not text:
        return {}
    matches = list(
        re.finditer(
            r"(?m)^\s*(?:\[([^\]\r\n]+)\]|([A-Za-z][A-Za-z0-9 _/\-]{1,60})\s*:)\s*$",
            text,
        )
    )
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        key = _normalise_heading(match.group(1) or match.group(2) or "")
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if key and body:
            sections[key] = body
    if not sections:
        sections["detailed_description"] = text
        sections["scene"] = text
    return sections


def _compose_prompt(project: dict[str, Any], task_mode: str, writing_mode: str, assistant_draft: str) -> tuple[str, dict[str, Any]]:
    mode = _normalise_task_mode(task_mode or project.get("task_mode") or "t2va")
    if mode not in MODE_SECTIONS:
        mode = "t2va"
    manual = project.get("sections") if isinstance(project.get("sections"), dict) else {}
    assisted = _parse_assistant_draft(assistant_draft) if writing_mode == "assistant_fill" else {}
    resolved: dict[str, str] = {}
    assistant_fills: list[str] = []
    for key, _label in MODE_SECTIONS[mode]:
        value = str(manual.get(key, "") or "").strip()
        if not value and str(assisted.get(key, "") or "").strip():
            value = str(assisted[key]).strip()
            assistant_fills.append(key)
        resolved[key] = value

    def join_fields(keys: tuple[str, ...]) -> str:
        return "\n".join(resolved.get(key, "").strip() for key in keys if resolved.get(key, "").strip())

    blocks: list[str] = []
    if any(resolved.values()):
        if mode == "ref2va":
            # MiniMax full-reference mode has six mandatory headings in this
            # exact order. Blank authoring fields become an explicit N/A rather
            # than silently changing the prompt grammar.
            for key, label in MODE_SECTIONS[mode]:
                body = resolved.get(key, "").strip() or "N/A"
                if key == "summary" and body != "N/A" and not body.lower().startswith("[reference"):
                    body = f"[reference generation] {body}"
                blocks.append(f"{label}:\n{body}")
        elif mode == "v2va_object_swap":
            detail = join_fields(("v2va_interval_edits", "v2va_exclusions")) or "N/A"
            summary = resolved.get("v2va_source_video_authority", "").strip() or "N/A"
            if summary != "N/A" and not summary.lower().startswith("[reference"):
                summary = f"[reference generation + video reference] {summary}"
            blocks = [
                f"subject_definitions:\n{resolved.get('v2va_subject_definitions', '').strip() or 'N/A'}",
                f"summary:\n{summary}",
                f"retention_analysis:\n{resolved.get('v2va_replacement_retention', '').strip() or 'N/A'}",
                f"detailed_description:\n{detail}",
                f"overall_soundscape:\n{resolved.get('v2va_sound_policy', '').strip() or 'N/A'}",
                "non_diegetic_music:\nN/A",
            ]
        else:
            if mode == "t2va":
                detail_keys = ("scene", "shot_list", "acting", "dialogue", "light_and_image", "camera", "negatives")
                alignment = ""
                sound = resolved.get("production_sound", "").strip()
                music = resolved.get("non_diegetic_music", "").strip()
            elif mode == "i2va":
                detail_keys = ("reference_use", "identity_continuity_locks", "scene", "shot_list", "acting", "dialogue", "light_and_image", "camera", "negatives")
                alignment = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
                sound = resolved.get("production_sound", "").strip()
                music = resolved.get("non_diegetic_music", "").strip()
            elif mode == "fl2va":
                detail_keys = ("reference_use", "identity_continuity_locks", "action", "shot_list", "acting", "dialogue", "light_and_image", "camera", "negatives")
                alignment = resolved.get("boundary_frames", "").strip()
                sound = resolved.get("production_sound", "").strip()
                music = resolved.get("non_diegetic_music", "").strip()
            else:  # audio_driven
                detail_keys = (
                    "audio_drive_contract", "audio_subject_map", "audio_scene_intent",
                    "audio_timed_performance", "audio_dialogue_map", "audio_visual_sync",
                    "audio_camera_sync", "audio_continuity_locks",
                )
                alignment = ""
                sound = resolved.get("audio_environment", "").strip()
                music = "N/A"
            detail = join_fields(detail_keys)
            if detail and not re.match(r"^\s*\[Shot\s+1\]", detail, flags=re.I):
                detail = f"[Shot 1] {detail}"
            if alignment:
                blocks.append(alignment)
            blocks.extend([
                f"integrated_multimodal_description:\n{detail or 'N/A'}",
                f"overall_soundscape:\n{sound or 'N/A'}",
                f"non_diegetic_music:\n{music or 'N/A'}",
            ])
    prompt = "\n\n".join(blocks).strip()
    return prompt, {
        "task_mode": mode,
        "included_sections": [key for key, _label in MODE_SECTIONS[mode] if resolved.get(key)],
        "missing_sections": [key for key, _label in MODE_SECTIONS[mode] if not resolved.get(key)],
        "assistant_fills": assistant_fills,
    }


def _merge_text(existing: str, incoming: str, policy: str) -> str:
    old = str(existing or "").strip()
    new = str(incoming or "").strip()
    if not old or str(policy or "replace") == "replace":
        return new
    if not new:
        return old
    return f"{old}\n\n{new}"


def _normalise_ai_images(value: Any) -> list[dict[str, str]]:
    images: list[dict[str, str]] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict) or len(images) >= AI_IMAGE_LIMIT:
            continue
        data = str(item.get("data") or "").strip()
        if data.startswith("data:") and "," in data:
            header, data = data.split(",", 1)
            guessed = header[5:].split(";", 1)[0]
        else:
            guessed = ""
        data = re.sub(r"\s+", "", data)
        if not data:
            continue
        estimated_bytes = (len(data) * 3) // 4
        if estimated_bytes > AI_IMAGE_MAX_BYTES:
            raise ValueError(f"AI reference image exceeds {AI_IMAGE_MAX_BYTES // (1024 * 1024)} MB")
        mime_type = str(item.get("mime_type") or guessed or "image/png").strip().lower()
        if not mime_type.startswith("image/"):
            mime_type = "image/png"
        role = str(item.get("role") or "reference").strip().lower()
        if role not in {"opening", "closing", "identity", "composition", "style", "reference"}:
            role = "reference"
        images.append({
            "data": data,
            "mime_type": mime_type,
            "name": str(item.get("name") or f"Picture {len(images) + 1}").strip(),
            "role": role,
            "slot": str(item.get("slot") or len(images) + 1),
        })
    return images


def _assistant_instruction(
    task_mode: str,
    sections: dict[str, str],
    user_direction: str = "",
    target_keys: Any = None,
    images: Any = None,
) -> tuple[str, str]:
    mode = _normalise_task_mode(task_mode or "t2va")
    if mode not in MODE_SECTIONS:
        mode = "t2va"
    allowed = [key for key, _label in MODE_SECTIONS[mode]]
    rough = {key: str(sections.get(key, "") or "").strip() for key in allowed}
    filled = {key: value for key, value in rough.items() if value}
    selected = [str(key) for key in (target_keys if isinstance(target_keys, list) else []) if str(key) in allowed]
    if not selected:
        selected = list(filled)
    if not selected:
        raise ValueError("Select a MiniMax prompt section or write a rough idea before calling the AI")
    visuals = _normalise_ai_images(images)
    mode_rules = {
        "t2va": "Build the requested event from text. Keep the action chronological, filmable and compatible with one continuous audiovisual clip.",
        "i2va": "Treat <Picture 1> as the exact opening-frame authority. Animate from it without redesigning identity, wardrobe, composition or screen geography.",
        "fl2va": "Treat the opening and closing pictures as exact boundary frames. Describe one physically continuous path from the first frame to the last; do not solve the transition with a cut, dissolve or unrelated redesign.",
        "ref2va": "Use explicit <Picture N>, <Video N>, <Audio N> and <Subject N> references. State what each reference contributes and what must be ignored; preserve the lowercase REF2VA section semantics.",
        "v2va_object_swap": (
            "Write a MiniMax H3 video-to-video object/subject replacement contract. Use <Picture N> only for connected replacement/identity references, <Video 1> for the connected source video's temporal motion, camera and environment authority, and stable <Subject N> labels. "
            "Separate what is replaced from what remains, then express user-supplied interval edits in source-video time. Do not claim a mask, tracker, ControlNet, depth, pose or segmentation signal unless the user's connected workflow explicitly provides and names it."
        ),
        "audio_driven": (
            "Treat the connected custom audio as immutable timing authority for visible performance. "
            "Map the user-supplied transcript with stable speaker notation such as <Subject 1> (S1): "
            "<d>[Language] ...</d>. Never invent words that are not supplied by the user, never call custom "
            "drive audio <Audio 1> unless it is also explicitly connected as a REF2VA reference, and keep the "
            "speaker's mouth visible when lip synchronization is requested."
        ),
    }[mode]
    system = (
        "You are the autonomous IAMCCS MiniMax H3 prompt editor. Improve the user's own direction; do not replace it with a different story. "
        "Return one JSON object only, with plain-string values and no markdown. Valid keys are "
        f"{allowed}. Return only the selected keys {selected}; never create a blank or unselected section. "
        "Write concise production-ready English optimized for MiniMax H3 audiovisual generation. Preserve exact identity facts, reference tags, requested timing, language and quoted dialogue unless the user explicitly asks to change them. "
        "Use chronological visible action, realistic body mechanics, stable screen geography and one coherent camera language. Prefer one motivated camera move over a list of conflicting moves. "
        "Return only the content of each selected authoring field, never repeat its JSON key as a heading. The IAMCCS composer will assemble base modes into MiniMax's official integrated_multimodal_description, overall_soundscape and non_diegetic_music grammar, and full-reference modes into the official six-section grammar. "
        "When rewriting shot_list, do not add a second [Shot 1] marker because the composer supplies it; mark only a real later cut as [Shot N] At MM:SS.mmm, and do not invent cuts merely to make the description longer. "
        "Separate diegetic ambience, dialogue and contact effects from non-diegetic score. Use <Subject N> consistently and keep dialogue inside <d>[Language] ...</d> with stable speaker labels such as (S1) when those tags are present. "
        f"Chunk-boundary sound rule: {AUDIO_HANDOFF_AUTHORING_RULE} "
        "Do not invent extra characters, products, dialogue, scene changes, cuts, subtitles or logos. Turn negative wishes into concrete continuity safeguards, not vague quality adjectives. "
        f"Mode rule: {mode_rules} "
        "When images are attached, analyze only the contribution named by each image role. An opening image governs the first frame; a closing image governs the last frame; identity, composition and style images govern only those named attributes. "
        "Never mention unavailable media or claim to have seen a detail that is not visible."
    )
    if len(system) > 24000:
        raise RuntimeError("MiniMax assistant system prompt exceeds the 7000-token safety envelope")
    user = json.dumps({
        "task_mode": mode,
        "selected_sections": selected,
        "user_direction": str(user_direction or "").strip(),
        "rough_sections": {key: rough[key] for key in selected},
        "visual_context": [
            {"slot": item["slot"], "name": item["name"], "role": item["role"]}
            for item in visuals
        ],
    }, ensure_ascii=False, indent=2)
    return system, user


def _http_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        str(url),
        data=data,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=max(5.0, min(300.0, float(timeout)))) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1200]
        raise RuntimeError(f"AI provider HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"AI provider connection failed: {exc.reason}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("AI provider returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("AI provider returned an unsupported response")
    return parsed


def _http_get_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    request = urllib.request.Request(str(url), headers={"Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=max(2.0, min(30.0, float(timeout)))) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1200]
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama connection failed: {exc.reason}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Ollama returned an unsupported response")
    return parsed


def _ollama_native_base(value: Any) -> str:
    """Return the Ollama native API root accepted by /api/chat and /api/tags.

    Ollama also exposes an OpenAI-compatible ``/v1`` surface.  Users commonly
    paste that URL after using another local client, but native endpoints must
    be addressed from the server root.  Normalising here keeps model discovery
    and rewrite requests on the same endpoint contract.
    """
    root = str(value or "http://127.0.0.1:11434").strip().rstrip("/")
    if root.lower().endswith("/v1"):
        root = root[:-3].rstrip("/")
    return root or "http://127.0.0.1:11434"


def _extract_json_object(text: str) -> dict[str, str]:
    clean = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", str(text or "").strip(), flags=re.I | re.S)
    start = clean.find("{")
    end = clean.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("The AI response did not contain a JSON object")
    try:
        value = json.loads(clean[start:end + 1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"The AI response JSON is invalid: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("The AI response must be a JSON object")
    return {str(key): str(item or "").strip() for key, item in value.items() if isinstance(item, (str, int, float))}


def rewrite_sections_with_ai(
    provider: str,
    base_url: str,
    model: str,
    api_key: str,
    task_mode: str,
    sections: dict[str, str],
    temperature: float = 0.35,
    timeout: float = 120.0,
    user_direction: str = "",
    target_keys: Any = None,
    images: Any = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    provider = str(provider or "ollama").strip().lower()
    model = str(model or "").strip()
    if not model:
        raise ValueError("Select an AI model before rewriting")
    visual_inputs = _normalise_ai_images(images)
    system, user = _assistant_instruction(task_mode, sections, user_direction, target_keys, visual_inputs)
    api_key = str(api_key or "").strip()
    if not api_key:
        api_key = {
            "openai_compatible": os.environ.get("OPENAI_API_KEY", ""),
            "gemini": os.environ.get("GEMINI_API_KEY", ""),
            "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
        }.get(provider, "")
    content = ""

    if provider == "ollama":
        root = _ollama_native_base(base_url)
        result = _http_json(
            f"{root}/api/chat",
            {
                "model": model,
                "stream": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": user,
                        **({"images": [item["data"] for item in visual_inputs]} if visual_inputs else {}),
                    },
                ],
                "options": {"temperature": float(temperature)},
            },
            {},
            timeout,
        )
        content = str((result.get("message") or {}).get("content") or "")
    elif provider == "openai_compatible":
        root = str(base_url or "https://api.openai.com/v1").rstrip("/")
        url = root if root.endswith("/chat/completions") else f"{root}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        openai_user: Any = user
        if visual_inputs:
            openai_user = [{"type": "text", "text": user}] + [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{item['mime_type']};base64,{item['data']}"},
                }
                for item in visual_inputs
            ]
        result = _http_json(
            url,
            {
                "model": model,
                "temperature": float(temperature),
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": openai_user}],
            },
            headers,
            timeout,
        )
        choices = result.get("choices") or []
        content = str(((choices[0] if choices else {}).get("message") or {}).get("content") or "")
    elif provider == "gemini":
        root = str(base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        encoded_model = urllib.parse.quote(model, safe="-._")
        suffix = f"/models/{encoded_model}:generateContent"
        url = f"{root}{suffix}?key={urllib.parse.quote(api_key)}"
        gemini_parts: list[dict[str, Any]] = [{"text": user}]
        gemini_parts.extend(
            {"inlineData": {"mimeType": item["mime_type"], "data": item["data"]}}
            for item in visual_inputs
        )
        result = _http_json(
            url,
            {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": gemini_parts}],
                "generationConfig": {"temperature": float(temperature), "responseMimeType": "application/json"},
            },
            {},
            timeout,
        )
        candidates = result.get("candidates") or []
        parts = (((candidates[0] if candidates else {}).get("content") or {}).get("parts") or [])
        content = "".join(str(part.get("text") or "") for part in parts if isinstance(part, dict))
    elif provider == "anthropic":
        root = str(base_url or "https://api.anthropic.com/v1").rstrip("/")
        url = root if root.endswith("/messages") else f"{root}/messages"
        anthropic_user: Any = user
        if visual_inputs:
            anthropic_user = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": item["mime_type"],
                        "data": item["data"],
                    },
                }
                for item in visual_inputs
            ] + [{"type": "text", "text": user}]
        result = _http_json(
            url,
            {
                "model": model,
                "max_tokens": 4096,
                "temperature": float(temperature),
                "system": system,
                "messages": [{"role": "user", "content": anthropic_user}],
            },
            {"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            timeout,
        )
        content = "".join(str(item.get("text") or "") for item in (result.get("content") or []) if isinstance(item, dict))
    else:
        raise ValueError(f"Unsupported AI provider: {provider}")

    rewritten = _extract_json_object(content)
    allowed = {key for key, _label in MODE_SECTIONS.get(_normalise_task_mode(task_mode), MODE_SECTIONS["t2va"])}
    supplied = {key for key, value in sections.items() if key in allowed and str(value or "").strip()}
    requested = {str(key) for key in target_keys} if isinstance(target_keys, list) else supplied
    requested = requested & allowed
    if not requested:
        requested = supplied
    filtered = {key: value for key, value in rewritten.items() if key in requested and value}
    if not filtered:
        raise RuntimeError("The AI did not return any valid filled MiniMax section")
    return filtered, {
        "provider": provider,
        "model": model,
        "rewritten_sections": sorted(filtered),
        "preserved_blank_sections": sorted(allowed - supplied),
        "selected_sections": sorted(requested),
        "visual_references": [
            {"slot": item["slot"], "name": item["name"], "role": item["role"]}
            for item in visual_inputs
        ],
        "system_prompt_characters": len(system),
        "audio_handoff_authoring_rule": AUDIO_HANDOFF_AUTHORING_RULE,
    }


def _linx_resources(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    resources = value.get("resources")
    return resources if isinstance(resources, dict) else {}


VISION_CONTEXT_SCHEMA = "iamccs.minimax_h3.vision_context"
VISION_CONTEXT_RESOURCE = "iamccs_h3_vision_context_by_target"
VISION_MANIFEST_RESOURCE = "iamccs_h3_vision_manifest"


def _format_vision_context(task_mode: str, context: Any) -> str:
    text = str(context or "").strip()
    if not text:
        return ""
    if str(task_mode or "").strip().lower() == "ref2va":
        return f"visual_reference_analysis:\n{text}"
    return f"[VISUAL REFERENCE ANALYSIS]\n{text}"


def _vision_context_requests(
    cine_linx: Any,
    task_mode: str,
    primary_target: str,
) -> tuple[dict[str, str], str, dict[str, Any]]:
    resources = _linx_resources(cine_linx)
    manifest = resources.get(VISION_MANIFEST_RESOURCE)
    contexts = resources.get(VISION_CONTEXT_RESOURCE)
    if not isinstance(manifest, dict) or manifest.get("schema") != VISION_CONTEXT_SCHEMA:
        return {}, "append", {"consumed": False, "reason": "no_h3_vision_manifest"}
    if not isinstance(contexts, dict):
        contexts = manifest.get("target_contexts")
    if not isinstance(contexts, dict):
        return {}, "append", {"consumed": False, "reason": "no_h3_vision_target_contexts"}

    allowed_targets = {"global", "local_auto", "local_1", "local_2", "local_3"}
    clean: dict[str, str] = {}
    for target, context in contexts.items():
        target_name = str(target or "").strip().lower()
        if target_name not in allowed_targets:
            continue
        formatted = _format_vision_context(task_mode, context)
        if formatted:
            clean[target_name] = formatted[:H3_ABSOLUTE_CHAR_LIMIT]
    policy = str(
        resources.get(
            "iamccs_h3_vision_context_merge_policy",
            manifest.get("context_merge_policy", "append"),
        )
        or "append"
    ).strip().lower()
    policy = policy if policy in {"append", "replace"} else "append"
    return clean, policy, {
        "consumed": bool(clean),
        "schema_version": manifest.get("schema_version"),
        "status": manifest.get("status"),
        "analysis_mode": manifest.get("analysis_mode"),
        "available_targets": sorted(clean),
        "primary_target": str(primary_target),
        "merge_policy": policy,
        "pictures": [
            {
                "slot": item.get("slot"),
                "role": item.get("role"),
                "target": item.get("target"),
            }
            for item in manifest.get("pictures", [])
            if isinstance(item, dict)
        ],
    }


def _append_prompter_stage(
    upstream_linx: Any,
    injection: dict[str, Any],
    injections: list[dict[str, Any]],
    final_prompt: str,
    project_json: str,
    report: str,
    mode: str,
    injection_target: str,
    character_count: int,
) -> dict[str, Any]:
    out = dict(upstream_linx) if isinstance(upstream_linx, dict) else {}
    out["type"] = SUPERNODE_LINX_TYPE
    out["mode"] = "iamccs_minimax_h3_prompter"
    out["active_stage"] = "iamccs_prompter"
    out["active_stage_kind"] = "prompt_authoring"

    chain = [dict(item) for item in (out.get("chain") or []) if isinstance(item, dict)]
    chain.append({"role": "prompt_author", "name": "IAMCCS_Prompter"})
    out["chain"] = chain
    stages = [dict(item) for item in (out.get("stages") or []) if isinstance(item, dict)]
    stages.append({
        "name": "iamccs_prompter",
        "kind": "prompt_authoring",
        "payload": {
            "task_mode": mode,
            "target": str(injection_target),
            "characters": character_count,
            "injection_count": len(injections),
        },
    })
    out["stages"] = stages
    out["stage_count"] = len(stages)

    resources = dict(_linx_resources(out))
    resources.update({
        # Keep the singular contract for every older Shotboard/workflow.
        "iamccs_prompter_injection": injection,
        # New contract: deterministic ordered requests can address global and
        # independent local prompt targets in one CineLinX pass.
        "iamccs_prompter_injections": [dict(item) for item in injections],
        "iamccs_prompter_prompt": final_prompt,
        "iamccs_prompter_project_json": project_json,
        "iamccs_prompter_audio_handoff_rule": AUDIO_HANDOFF_AUTHORING_RULE,
        "iamccs_prompter_audio_driven_dialogue_template": "<Subject 1> (S1): <d>[Language] ...</d>",
        "cine_global_prompt": final_prompt if str(injection_target) == "global" else "",
        "cine_report": report,
    })
    out["resources"] = resources

    outputs = dict(out.get("outputs") or {})
    outputs.update({
        "final_prompt": final_prompt,
        "project_json": project_json,
        "injection_target": str(injection_target),
        "report": report,
    })
    out["outputs"] = outputs
    out["resource_keys"] = sorted(resources)
    out["resource_types"] = {key: type(value).__name__ for key, value in resources.items()}
    return out


def _apply_one_prompter_request(
    global_prompt: str,
    timeline_data: Any,
    request: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    """Apply one explicit request without reading mutable state from CineLinX."""

    prompt = str(request.get("prompt", "") or "").strip()
    target = str(request.get("target", "global") or "global").strip().lower()
    policy = str(request.get("merge_policy", "replace") or "replace").strip().lower()
    if not prompt:
        return str(global_prompt or ""), str(timeline_data or ""), {"applied": False, "reason": "empty_prompter_prompt"}

    if target == "global":
        merged = _merge_text(str(global_prompt or ""), prompt, policy)
        return merged, str(timeline_data or ""), {
            "applied": True,
            "requested_target": target,
            "actual_target": "global",
            "merge_policy": policy,
        }

    raw_timeline = str(timeline_data or "").strip()
    try:
        timeline = json.loads(raw_timeline) if raw_timeline else {}
    except json.JSONDecodeError:
        timeline = {}
    if not isinstance(timeline, dict):
        timeline = {}

    rows: list[dict[str, Any]] | None = None
    row_key = ""
    for candidate in ("segments", "rows", "slots", "shots"):
        value = timeline.get(candidate)
        if isinstance(value, list):
            rows = value
            row_key = candidate
            break
    visual: list[tuple[int, dict[str, Any]]] = []
    if isinstance(rows, list):
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict) or bool(row.get("placeholder", False)):
                continue
            row_type = str(row.get("type", "image") or "image").strip().lower()
            if row_type in {"audio", "motion", "video"}:
                continue
            visual.append((row_index, row))

    if not visual:
        merged = _merge_text(str(global_prompt or ""), prompt, "append" if str(global_prompt or "").strip() else "replace")
        return merged, raw_timeline, {
            "applied": True,
            "requested_target": target,
            "actual_target": "global_fallback_no_local_slots",
            "merge_policy": "append" if str(global_prompt or "").strip() else "replace",
        }

    limit = min(3, len(visual))
    selected_position: int | None = None
    effective_policy = policy
    if target == "local_auto":
        for position in range(limit):
            row = visual[position][1]
            existing = str(row.get("prompt", row.get("local_prompt", row.get("relay_prompt", ""))) or "").strip()
            if not existing:
                selected_position = position
                break
        if selected_position is None:
            selected_position = limit - 1
            # Auto must never silently destroy three completed local prompts.
            effective_policy = "append"
    else:
        match = re.fullmatch(r"local_([123])", target)
        requested_position = int(match.group(1)) - 1 if match else 0
        if requested_position < limit:
            selected_position = requested_position
        else:
            # Detect what is actually present and select the first empty slot,
            # otherwise the final available slot without creating fake timing.
            selected_position = next(
                (
                    position
                    for position in range(limit)
                    if not str(visual[position][1].get("prompt", visual[position][1].get("local_prompt", "")) or "").strip()
                ),
                limit - 1,
            )

    assert selected_position is not None
    actual_row_index, row = visual[selected_position]
    existing = str(row.get("prompt", row.get("local_prompt", row.get("relay_prompt", ""))) or "")
    merged = _merge_text(existing, prompt, effective_policy)
    row["prompt"] = merged
    row["local_prompt"] = merged
    row["relay_prompt"] = merged
    row["use_prompt"] = True
    row["relay_manual_off"] = False
    row["promptrelay_manual_off"] = False
    timeline[row_key] = rows
    return str(global_prompt or ""), json.dumps(timeline, ensure_ascii=False), {
        "applied": True,
        "requested_target": target,
        "actual_target": f"local_{selected_position + 1}",
        "timeline_row_index": actual_row_index,
        "merge_policy": effective_policy,
        "available_local_slots": len(visual),
    }


def apply_prompter_to_minimax(
    cine_linx: Any,
    global_prompt: str,
    timeline_data: Any,
) -> tuple[str, str, dict[str, Any]]:
    """Apply legacy or multi-target Prompter requests to a known Shotboard."""
    resources = _linx_resources(cine_linx)
    requests = resources.get("iamccs_prompter_injections")
    if not isinstance(requests, list):
        legacy = resources.get("iamccs_prompter_injection")
        requests = [legacy] if isinstance(legacy, dict) else []
    requests = [dict(item) for item in requests if isinstance(item, dict)]
    if not requests:
        return str(global_prompt or ""), str(timeline_data or ""), {
            "applied": False,
            "reason": "no_prompter_cine_linx",
        }

    resolved_global = str(global_prompt or "")
    resolved_timeline = str(timeline_data or "")
    applications: list[dict[str, Any]] = []
    for request_index, request in enumerate(requests):
        resolved_global, resolved_timeline, result = _apply_one_prompter_request(
            resolved_global,
            resolved_timeline,
            request,
        )
        applications.append({"request_index": request_index, **result})

    applied = [item for item in applications if item.get("applied")]
    actual_targets = [str(item.get("actual_target")) for item in applied if item.get("actual_target")]
    return resolved_global, resolved_timeline, {
        "applied": bool(applied),
        "requested_count": len(requests),
        "applied_count": len(applied),
        "actual_target": ",".join(actual_targets) if actual_targets else "none",
        "actual_targets": actual_targets,
        "applications": applications,
        "multi_target_contract": len(requests) > 1,
    }


class IAMCCS_Prompter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_data": (
                    "STRING",
                    {
                        "default": json.dumps(default_project(), ensure_ascii=False),
                        "multiline": True,
                    },
                ),
                "task_mode": (
                    ["t2va", "i2va", "fl2va", "ref2va", "v2va_object_swap", "audio_driven"],
                    {"default": "t2va"},
                ),
                "injection_target": (
                    ["global", "local_auto", "local_1", "local_2", "local_3"],
                    {"default": "global"},
                ),
                "writing_mode": (
                    ["manual", "guided", "assistant_fill"],
                    {"default": "guided"},
                ),
                "merge_policy": (["replace", "append"], {"default": "replace"}),
                "character_budget": ("INT", {"default": 6800, "min": 1000, "max": H3_ABSOLUTE_CHAR_LIMIT, "step": 100}),
            },
            "optional": {
                "cine_linx": (
                    SUPERNODE_LINX_TYPE,
                    {
                        "tooltip": (
                            "Optional upstream IAMCCS Cine H3 Vision Info. Its analyzed visual context is "
                            "routed to the declared global/local targets without embedding image tensors in this node."
                        ),
                    },
                ),
                "assistant_draft": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "forceInput": True,
                        "tooltip": "Optional structured draft from any text source. In Assistant Fill mode it fills only empty structured boxes.",
                    },
                ),
            },
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("cine_linx", "final_prompt", "project_json", "report")
    FUNCTION = "compose"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return json.dumps(kwargs, ensure_ascii=False, sort_keys=True, default=str)

    def compose(
        self,
        project_data,
        task_mode,
        injection_target,
        writing_mode,
        merge_policy,
        character_budget,
        assistant_draft="",
        cine_linx=None,
    ):
        project = _safe_project(project_data)
        mode = _normalise_task_mode(task_mode or project.get("task_mode") or "t2va")
        project["task_mode"] = mode
        project["injection_target"] = str(injection_target)
        project["writing_mode"] = str(writing_mode)
        project["merge_policy"] = str(merge_policy)
        final_prompt, details = _compose_prompt(project, mode, str(writing_mode), str(assistant_draft or ""))
        if not final_prompt:
            raise ValueError("IAMCCS_Prompter: compila almeno un box prima di accodare il workflow")

        primary_target = str(injection_target or "global").strip().lower()
        vision_contexts, vision_merge_policy, vision_report = _vision_context_requests(
            cine_linx,
            mode,
            primary_target,
        )
        primary_vision_context = vision_contexts.pop(primary_target, "")
        if primary_vision_context:
            final_prompt = _merge_text(final_prompt, primary_vision_context, vision_merge_policy)
        char_count = len(final_prompt)
        budget = min(H3_ABSOLUTE_CHAR_LIMIT, max(1000, int(character_budget)))
        if char_count > H3_ABSOLUTE_CHAR_LIMIT:
            raise ValueError(
                f"IAMCCS_Prompter: prompt di {char_count} caratteri; MiniMax H3 richiede massimo "
                f"{H3_ABSOLUTE_CHAR_LIMIT}. Riduci i box di almeno {char_count - H3_ABSOLUTE_CHAR_LIMIT} caratteri."
            )

        injection = {
            "schema": "iamccs.minimax_h3.prompt_injection",
            "schema_version": 1,
            "prompt": final_prompt,
            "target": str(injection_target),
            "merge_policy": str(merge_policy),
            "task_mode": mode,
            "project_name": str(project.get("project_name") or "Untitled Prompt"),
            "source": "iamccs_prompter",
        }

        injections = [injection]
        for target, context in vision_contexts.items():
            if not str(context or "").strip():
                continue
            if len(context) > H3_ABSOLUTE_CHAR_LIMIT:
                raise ValueError(
                    f"IAMCCS_Prompter: visual context for {target} contains {len(context)} characters; "
                    f"the MiniMax H3 request limit is {H3_ABSOLUTE_CHAR_LIMIT}."
                )
            injections.append({
                "schema": "iamccs.minimax_h3.prompt_injection",
                "schema_version": 1,
                "prompt": context,
                "target": target,
                "merge_policy": vision_merge_policy,
                "task_mode": mode,
                "project_name": injection["project_name"],
                "source": "iamccs_cine_h3_vision_info",
            })
        project_json = json.dumps(project, ensure_ascii=False, indent=2)
        report_data = {
            "node": "IAMCCS_Prompter",
            "project_name": injection["project_name"],
            "task_mode": mode,
            "writing_mode": str(writing_mode),
            "requested_target": str(injection_target),
            "merge_policy": str(merge_policy),
            "characters": char_count,
            "character_budget": budget,
            "within_recommended_budget": char_count <= budget,
            "injection_count": len(injections),
            "injection_targets": [item["target"] for item in injections],
            "vision_context": vision_report,
            "audio_handoff_authoring_rule": AUDIO_HANDOFF_AUTHORING_RULE,
            "audio_driven_dialogue_template": "<Subject 1> (S1): <d>[Language] ...</d>",
            **details,
            "truth": "The MiniMax Shotboard resolves local_auto only after reading its own timeline slots.",
        }
        report = json.dumps(report_data, ensure_ascii=False, indent=2)
        out_linx = _append_prompter_stage(
            cine_linx,
            injection,
            injections,
            final_prompt,
            project_json,
            report,
            mode,
            str(injection_target),
            char_count,
        )
        return out_linx, final_prompt, project_json, report


def _register_prompter_routes() -> None:
    """Register the interactive AI rewrite endpoint without adding a dependency."""
    try:
        from aiohttp import web
        from server import PromptServer

        routes = PromptServer.instance.routes

        @routes.get("/iamccs/prompter/ollama/models")
        async def iamccs_prompter_ollama_models(request):
            try:
                base_url = _ollama_native_base(request.query.get("base_url"))
                payload = await asyncio.to_thread(_http_get_json, f"{base_url}/api/tags", 10.0)
                models = []
                for item in payload.get("models") if isinstance(payload.get("models"), list) else []:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("name") or item.get("model") or "").strip()
                    if name:
                        models.append({
                            "name": name,
                            "size": int(item.get("size") or 0),
                            "modified_at": str(item.get("modified_at") or ""),
                        })
                return web.json_response({"ok": True, "models": models})
            except Exception as exc:
                return web.json_response({"ok": False, "error": str(exc)}, status=400)

        @routes.post("/iamccs/prompter/rewrite")
        async def iamccs_prompter_rewrite(request):
            try:
                payload = await request.json()
                sections = payload.get("sections") if isinstance(payload, dict) else None
                if not isinstance(sections, dict):
                    raise ValueError("sections must be a JSON object")
                rewritten, report = await asyncio.to_thread(
                    rewrite_sections_with_ai,
                    str(payload.get("provider", "ollama")),
                    str(payload.get("base_url", "")),
                    str(payload.get("model", "")),
                    str(payload.get("api_key", "")),
                    str(payload.get("task_mode", "t2va")),
                    {str(key): str(value or "") for key, value in sections.items()},
                    float(payload.get("temperature", 0.35)),
                    float(payload.get("timeout", 120.0)),
                    str(payload.get("user_direction", "")),
                    payload.get("target_keys"),
                    payload.get("images"),
                )
                return web.json_response({"ok": True, "sections": rewritten, "report": report})
            except Exception as exc:
                safe_error = re.sub(
                    r"(?i)(api[_ -]?key|authorization)[^,;\n]*",
                    r"\1=[redacted]",
                    str(exc),
                )
                return web.json_response({"ok": False, "error": safe_error}, status=400)
    except Exception:
        # Schema discovery and headless tests can import before PromptServer.
        return


_register_prompter_routes()


NODE_CLASS_MAPPINGS = {"IAMCCS_Prompter": IAMCCS_Prompter}
NODE_DISPLAY_NAME_MAPPINGS = {"IAMCCS_Prompter": "IAMCCS Prompter — MiniMax H3 Screenplay"}
