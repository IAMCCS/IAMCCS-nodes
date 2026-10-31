"""Pika recipe, adapter family and lazy branch tests without inference."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pika_profile_test", ROOT / "iamccs_h3_pika_flex.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def plan(mode="t2va"):
    return {"acceleration": P.PROFILE, "task_mode": mode, "sampling": {"seed": 42}}


def settings():
    return {"turbo_mode": "off", "pika_flex_fl2va_lora": "fl.safetensors",
            "pika_flex_ref2va_lora": "ref.safetensors"}


def test_disabled_profile_preserves_sampling():
    p = {"acceleration": "native", "sampling": {"steps": 20}}
    P.configure(p, {})
    assert p["sampling"] == {"steps": 20}
    assert not P.active(p)


@pytest.mark.parametrize("mode", ["t2va", "i2va", "fl2va", "ref2va"])
def test_exact_six_nfe_contract_preserves_seed(mode):
    p = plan(mode)
    P.configure(p, settings())
    assert p["sampling"]["seed"] == 42
    assert p["sampling"]["steps"] == 6
    assert p["sampling"]["sampler_name"] == "lcm"
    assert p["sampling"]["shift_video"] == 12
    assert len(P.SIGMAS.split(",")) == 7
    assert float(P.SIGMAS.split(",")[-1]) == 0


@pytest.mark.parametrize("change", [{"h3_omni_enabled": True}, {"turbo_mode": "lightx2v"}])
def test_rejects_stacked_speed_profiles(change):
    with pytest.raises(ValueError, match="disable"):
        P.configure(plan(), {**settings(), **change})


def test_experimental_continuation_is_explicitly_unavailable():
    with pytest.raises(ValueError, match="continuation"):
        P.configure(plan("fl2va_extended_av"), settings())


@pytest.mark.parametrize("mode,expected", [("i2va", "fl.safetensors"), ("ref2va", "ref.safetensors")])
def test_matching_adapter_is_applied_once(monkeypatch, mode, expected):
    loader = Mock(return_value=("patched",))
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(get_full_path=lambda *a: "registered-file"))
    monkeypatch.setitem(sys.modules, "nodes", SimpleNamespace(LoraLoaderModelOnly=lambda: SimpleNamespace(load_lora_model_only=loader)))
    p = plan(mode)
    P.configure(p, settings())
    assert P.apply_adapter("base", p, mode)[0] == "patched"
    loader.assert_called_once_with(model="base", lora_name=expected, strength_model=1.0)
    loader.reset_mock()
    assert P.apply_adapter("base", {"pika_flex": {"enabled": False}}, mode)[0] == "base"
    loader.assert_not_called()


def test_lazy_router_requests_only_selected_six_outputs(monkeypatch):
    package = ModuleType("pika_router_test")
    package.__path__ = []
    atomic = ModuleType("pika_router_test.iamccs_minimax_h3_atomic_backend")
    atomic._resolve_shotplan = lambda p: p
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, atomic.__name__, atomic)
    monkeypatch.setitem(P.IAMCCS_MiniMaxH3PikaFlexLazyRouter.names.__globals__, "__package__", package.__name__)
    router = P.IAMCCS_MiniMaxH3PikaFlexLazyRouter()
    for enabled, prefix in [(True, "variant"), (False, "legacy")]:
        p = {"pika_flex": {"enabled": enabled}}
        names = router.check_lazy_status(p)
        assert len(names) == 6 and all(n.startswith(prefix) for n in names)
        values = {name: object() for name in names}
        selected = router.select(p, **values)
        assert selected == tuple(values[name] for name in names)
        assert router.check_lazy_status(p, **values) == []
