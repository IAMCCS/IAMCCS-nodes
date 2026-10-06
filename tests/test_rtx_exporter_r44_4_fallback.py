from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_exporter_has_nonfatal_missing_runtime_fallback():
    source = (ROOT / "iamccs_shotboarder_exporter_pro.py").read_text(encoding="utf-8")
    assert "rtx_vfx_runtime_probe" in source
    assert "RTX was requested but the NVIDIA RTX VFX runtime is unavailable" in source
    assert '"rtx_requested": rtx_requested' in source
    assert '"rtx_effective": rtx_active' in source


def test_rtx_probe_is_non_throwing_contract():
    source = (ROOT / "iamccs_rtx_vfx.py").read_text(encoding="utf-8")
    assert "def rtx_vfx_runtime_probe()" in source
    assert '"available": False' in source
    assert "_import_video_super_res()" in source


def test_exporter_ui_queries_runtime_instead_of_claiming_active_blindly():
    source = (ROOT / "web" / "iamccs_shotboarder_exporter_pro_ui.js").read_text(encoding="utf-8")
    assert '/api/iamccs/rtx_vfx/status' in source
    assert "runtime missing · export will fall back to native frames" in source
