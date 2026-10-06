from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_face_refine_recipe_groups_are_independent_before_generic_memory_group():
    source = (ROOT / "web" / "iamccs_h3_settings_pro_ui.js").read_text(encoding="utf-8")
    func = source[source.index("function recipeGroup(parent)"):source.index("function rememberedRecipe", source.index("function recipeGroup(parent)"))]
    lines = [line for line in func.splitlines() if "if (parent?.classList?.contains" in line]
    assert "h3p-face-refine-quality-recipes" in lines[0]
    assert "h3p-face-refine-size-recipes" in lines[1]
    assert "h3p-face-refine-definition-recipes" in lines[2]
    assert "h3p-memory-recipes" in lines[3]
    assert "ACTIVE PRESET STACK" in source
    assert "RESET PRESETS · CURRENT DEFAULTS" in source


def test_face_refine_preset_families_are_orthogonal():
    source = (ROOT / "web" / "iamccs_h3_settings_pro_ui.js").read_text(encoding="utf-8")
    tuning = source[source.index("function applyFaceRefineTuning"):source.index("function resetFaceRefinePresets")]
    quality = tuning[tuning.index('"faceq_safe"'):tuning.index('"facesize_wide"')]
    definition = tuning[tuning.index('"facedef_soft"'):tuning.index("};", tuning.index('"facedef_strong"'))]
    assert "face_detailer_denoise" not in quality
    assert "face_detailer_blend" not in quality
    assert "face_detailer_strength_large_face" not in definition


def test_video_editor_audio_slave_exports_realtime_track_meters_and_smooth_gain():
    source = (ROOT / "web" / "iamccs_shotboard_video_editor_v1_ui.js").read_text(encoding="utf-8")
    assert "editorAudioMeterSnapshot" in source
    assert "meters: editorAudioMeterSnapshot()" in source
    assert "smoothAudioPreviewVolume" in source
    assert "refreshAudioPreviewGains" in source
    assert "faderOnlyUpdate" in source


def test_exporter_rtx_buttons_use_compact_labels_with_full_tooltips():
    source = (ROOT / "web" / "iamccs_shotboarder_exporter_pro_ui.js").read_text(encoding="utf-8")
    assert 'data-rtx-check title="CHECK RTX INSTALLATION">CHECK RTX<' in source
    assert 'data-rtx-guide title="OPEN OFFICIAL INSTALL GUIDE">INSTALL GUIDE<' in source
    assert "white-space:normal" in source
