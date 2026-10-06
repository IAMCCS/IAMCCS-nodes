import json

import iamccs_prompter as prompter


def test_evolving_natural_language_infers_boundaries_and_frames():
    beats = prompter.parse_evolving_timeline(
        "0-5 seconds: walks through the city.\n"
        "At 5 seconds she stops and looks up.\n"
        "A 10 secondi inizia a correre.",
        duration_seconds=20,
        fps=24,
    )
    assert [(beat["start_frame"], beat["end_frame"]) for beat in beats] == [
        (0, 120), (120, 240), (240, 480),
    ]
    assert beats[2]["action"] == "inizia a correre"


def test_evolving_rejects_untimed_prose():
    try:
        prompter.parse_evolving_timeline("She walks and later starts running.", duration_seconds=20)
    except ValueError as exc:
        assert "second marker" in str(exc)
    else:
        raise AssertionError("untimed evolving prose must fail instead of silently becoming continuous")


def test_prompter_exports_evolving_policy_and_demo_text():
    project = prompter.default_project()
    project["task_mode"] = "fl2va"
    project["sections"]["action"] = "The woman crosses the city in one uninterrupted take."
    project["extended_conditioning_policy"] = "evolving"
    project["evolving_timeline"] = prompter.EVOLVING_DEMO_TIMELINE
    output = prompter.IAMCCS_Prompter().compose(
        json.dumps(project), "fl2va", "global", "guided", "replace", 6800
    )
    result = output["result"]
    linx = result[0]
    request = linx["resources"]["iamccs_prompter_injection"]
    report = json.loads(result[3])
    assert request["extended_conditioning_policy"] == "evolving"
    assert "At 10 seconds" in request["evolving_timeline"]
    assert report["extended_conditioning"]["evolving_event_count"] == 4
    assert report["extended_conditioning"]["required_shotboard_mode"] == "fl2va_extended_av"


def test_existing_projects_without_policy_migrate_to_default():
    project = prompter._safe_project({"schema_version": 4, "sections": {"scene": "A street."}})
    assert project["schema_version"] == 7
    assert project["extended_conditioning_policy"] == "default"
    assert project["evolving_timeline"] == ""


def test_fl2va_continuous_is_one_untimed_global_injection_without_local_prompt():
    project = prompter.default_project()
    project["task_mode"] = "fl2va"
    project["extended_conditioning_policy"] = "continuous"
    project["sections"].update({
        "boundary_frames": "Picture 1 is the exact opening visual authority.",
        "identity_continuity_locks": "Preserve the same woman and bicycle.",
        "action": "0-8 seconds: the woman pedals steadily toward the bridge.",
        "shot_list": "At 8 seconds she keeps pedaling steadily toward the bridge.",
        "camera": "One steady backward tracking camera.",
    })
    output = prompter.IAMCCS_Prompter().compose(
        json.dumps(project), "fl2va", "local_1", "guided", "replace", 6800
    )
    linx, final_prompt, _project_json, report_json, *_ = output["result"]
    report = json.loads(report_json)
    injections = linx["resources"]["iamccs_prompter_injections"]
    assert [item["target"] for item in injections] == ["global"]
    assert "continuous_action:" in final_prompt
    assert "the woman pedals steadily toward the bridge" in final_prompt
    assert "0-8 seconds" not in final_prompt
    assert "At 8 seconds" not in final_prompt
    assert linx["resources"]["iamccs_prompter_local_prompt"] == ""
    assert report["local_characters"] == 0


def test_fl2va_default_keeps_normal_global_and_local_contract():
    project = prompter.default_project()
    project["task_mode"] = "fl2va"
    project["sections"].update({
        "identity_continuity_locks": "Preserve the same woman.",
        "action": "The woman turns toward the window.",
    })
    output = prompter.IAMCCS_Prompter().compose(
        json.dumps(project), "fl2va", "local_1", "guided", "replace", 6800
    )
    linx = output["result"][0]
    assert project["extended_conditioning_policy"] == "default"
    assert [item["target"] for item in linx["resources"]["iamccs_prompter_injections"]] == ["global", "local_1"]


def test_fl2va_backend_separates_static_global_from_action_local():
    project = prompter.default_project()
    project["task_mode"] = "fl2va"
    project["sections"].update({
        "boundary_frames": "Picture 1 opens the take and Picture 2 defines the final composition.",
        "identity_continuity_locks": "Preserve the same soldier woman and army throughout.",
        "action": "The soldier woman walks toward camera and raises her sword.",
        "shot_list": "At 4 seconds she raises her sword.",
        "camera": "One steady backward tracking shot; no cuts or camera reset.",
        "production_sound": "Footsteps and one sword draw at 4 seconds.",
        "non_diegetic_music": "No score.",
    })
    final_prompt, details = prompter._compose_prompt(project, "fl2va", "guided", "")
    assert "raises her sword" not in final_prompt
    assert "At 4 seconds" not in final_prompt
    assert "no cuts" not in final_prompt.lower()
    assert "Preserve the same soldier woman" in final_prompt
    assert "raises her sword" in details["local_prompt"]
    assert "At 4 seconds" in details["local_prompt"]


def test_final_prompt_override_is_preserved_by_backend():
    project = prompter.default_project()
    project["task_mode"] = "fl2va"
    project["sections"]["identity_continuity_locks"] = "Generated identity continuity."
    project["sections"]["action"] = "Generated local action."
    project["final_prompt_override_enabled"] = True
    project["final_prompt_override"] = "MANUALLY EDITED GLOBAL"
    project["final_local_prompt_override_enabled"] = True
    project["final_local_prompt_override"] = "MANUALLY EDITED LOCAL"
    output = prompter.IAMCCS_Prompter().compose(json.dumps(project), "fl2va", "global", "guided", "replace", 6800)
    linx, final_prompt, _project_json, report_json, *_ = output["result"]
    report = json.loads(report_json)
    assert final_prompt == "MANUALLY EDITED GLOBAL"
    assert linx["resources"]["iamccs_prompter_local_prompt"] == "MANUALLY EDITED LOCAL"
    assert report["final_prompt_override"] is True
    assert report["final_local_prompt_override"] is True


def test_canonical_evolving_repairs_malformed_ai_timeline_before_queue():
    raw = (
        "0-12 seconds: the soldier woman takes a deep breath and walks toward the camera while the army follows behind her in the same formation and direction. "
        "[ONSET_once] 12-22 seconds: the soldier woman gives one brief rage scream toward the camera and the soldiers immediately answer with one brief collective scream "
        "[SUSTAIN] after both screams have finished, they continue advancing toward the camera in the same formation without repeating either scream before\n"
        "22 seconds. 22-30 seconds: the soldier woman stops and starts to laugh while the army behind her starts to sing epic battle songs; continue this final performance naturally through the end"
    )
    canonical = prompter._validate_canonical_evolving_timeline(raw)
    lines = canonical.splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("0-12 seconds: [ONSET ONCE]")
    assert "[THEN SUSTAIN] the soldier woman walks toward the camera" in lines[0]
    assert lines[1].startswith("12-22 seconds: [ONSET ONCE]")
    assert "[THEN SUSTAIN] they continue advancing toward the camera" in lines[1]
    assert "after both screams" not in lines[1].lower()
    assert "without repeating" not in lines[1].lower()
    assert lines[2].startswith("22-30 seconds: [ONSET ONCE] the soldier woman stops")
    assert "continues laughing" in lines[2]
    assert "continues singing" in lines[2]
    assert "[ONSET_once]" not in canonical
    assert "[SUSTAIN]" not in canonical


def test_canonical_evolving_timestamp_precedes_all_tags():
    raw = "[ONSET_once] 12-22 seconds: she shouts once [SUSTAIN] she keeps marching"
    canonical = prompter._validate_canonical_evolving_timeline(raw)
    assert canonical == "12-22 seconds: [ONSET ONCE] she shouts once; [THEN SUSTAIN] she keeps marching"

