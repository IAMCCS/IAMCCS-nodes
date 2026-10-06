from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_rtx_probe_exposes_runtime_diagnostics():
    source = (ROOT / "iamccs_rtx_vfx.py").read_text(encoding="utf-8")
    for key in ("python_executable", "runtime_source", "runtime_path", "module_path", "video_super_res"):
        assert f'"{key}"' in source


def test_exporter_has_manual_rtx_check_and_guide_buttons():
    source = (ROOT / "web" / "iamccs_shotboarder_exporter_pro_ui.js").read_text(encoding="utf-8")
    assert "CHECK RTX INSTALLATION" in source
    assert "OPEN OFFICIAL INSTALL GUIDE" in source
    assert "data-rtx-runtime-badge" in source
    assert "data-rtx-python" in source
    assert "data-rtx-vsr" in source


def test_face_refine_presets_are_grouped_and_documented():
    source = (ROOT / "web" / "iamccs_h3_settings_pro_ui.js").read_text(encoding="utf-8")
    assert "h3p-face-refine-presets-grid" in source
    assert "FACE_REFINE_PRESET_HELP" in source
    assert "Does not change Single/Multi Face mode" in source
    assert '"facesize_small": { face_detailer_profile:' not in source


def test_face_validation_runs_after_settings_override():
    source = (ROOT / "iamccs_minimax_h3_face_delivery_variant.py").read_text(encoding="utf-8")
    override = source.index('canvas_width = _face_setting(face, "canvas_width"')
    validation = source.index("if window_overlap >= window_frames")
    assert validation > override
