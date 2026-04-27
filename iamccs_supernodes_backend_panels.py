def backend_panel_input_types():
    return {
        "backend_panel_mode": (["compact", "show_backend_settings"],),
        "backend_profile_preset": (
            ["inherit", "quality_first", "balanced", "speed_first", "low_vram_safe", "audio_sync_strict"],
        ),
        "backend_memory_mode": (
            ["inherit", "auto", "normal_vram", "low_ram_disk", "cpu_offload_safe"],
        ),
        "backend_quality_mode": (
            ["inherit", "draft", "balanced", "high_detail", "identity_strict", "temporal_stable"],
        ),
        "backend_settings_override": ("STRING", {"default": "", "multiline": True}),
    }


def merge_backend_panel_inputs(required_inputs):
    merged = dict(required_inputs)
    merged.update(backend_panel_input_types())
    return merged


def _sanitize_override(value):
    return str(value or "").replace(";", ",").strip() or "none"


def build_backend_panel_payload(
    backend_panel_mode,
    backend_profile_preset,
    backend_memory_mode,
    backend_quality_mode,
    backend_settings_override,
):
    sanitized_override = _sanitize_override(backend_settings_override)
    payload = (
        f"backend_panel_mode={backend_panel_mode}; backend_profile_preset={backend_profile_preset}; "
        f"backend_memory_mode={backend_memory_mode}; backend_quality_mode={backend_quality_mode}; "
        f"backend_settings_override={sanitized_override}"
    )
    summary = (
        f"backend_panel={backend_panel_mode} | preset={backend_profile_preset} | "
        f"memory={backend_memory_mode} | quality={backend_quality_mode}"
    )
    return payload, summary


def append_backend_panel_to_payload_and_report(
    payload,
    report,
    backend_panel_mode,
    backend_profile_preset,
    backend_memory_mode,
    backend_quality_mode,
    backend_settings_override,
):
    panel_payload, panel_summary = build_backend_panel_payload(
        backend_panel_mode,
        backend_profile_preset,
        backend_memory_mode,
        backend_quality_mode,
        backend_settings_override,
    )
    if payload:
        payload = f"{payload}; {panel_payload}"
    else:
        payload = panel_payload
    if report:
        report = f"{report} | {panel_summary}"
    else:
        report = panel_summary
    return payload, report