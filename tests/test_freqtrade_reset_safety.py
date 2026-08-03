from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_checker():
    path = PROJECT_ROOT / "tools" / "check_freqtrade_reset.py"
    spec = importlib.util.spec_from_file_location("check_freqtrade_reset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_freqtrade_reset_config_is_fail_closed():
    report = _load_checker().validate()

    assert report["passed"] is True
    assert report["orders_enabled"] is False
    assert report["promotion_authority"] is False


def test_pulse_strategy_contains_runtime_dry_run_guard():
    source = (
        PROJECT_ROOT
        / "freqtrade_reset"
        / "user_data"
        / "strategies"
        / "DeterministicPulseStrategy.py"
    ).read_text(encoding="utf-8")

    assert 'self.config.get("dry_run") is not True' in source
    assert "requires dry_run=true" in source
