from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "freqtrade_pilot"


def test_upstream_lock_matches_submodule() -> None:
    lock = json.loads((PILOT / "upstream.lock.json").read_text(encoding="utf-8"))
    gitmodules = (ROOT / ".gitmodules").read_text(encoding="utf-8")

    assert lock["repository"] in gitmodules
    assert lock["submodule_path"] in gitmodules
    assert lock["commit"] == "52bc96f4480b1a0da6a9b455bd00b17fbb6786a5"
    assert lock["tag"] == lock["package_version"] == "2026.7"


def test_pilot_config_is_dry_run_only() -> None:
    config = json.loads(
        (PILOT / "user_data" / "config.dryrun.json").read_text(encoding="utf-8")
    )

    assert config["dry_run"] is True
    assert config["trading_mode"] == "futures"
    assert config["margin_mode"] == "isolated"
    assert config["max_open_trades"] == 2
    assert config["stake_amount"] == 10
    assert config["force_entry_enable"] is False
    assert config["exchange"]["name"] == "bitget"
    assert config["exchange"]["key"] == ""
    assert config["exchange"]["secret"] == ""
    assert config["exchange"]["password"] == ""
    assert len(config["exchange"]["pair_whitelist"]) == 8


def test_adapter_has_live_guard_and_server_side_stop() -> None:
    source = (
        PILOT / "user_data" / "strategies" / "UpstreamSampleDryRunStrategy.py"
    ).read_text(encoding="utf-8")

    assert "class UpstreamSampleDryRunStrategy(SampleStrategy)" in source
    assert 'self.config.get("dry_run") is not True' in source
    assert '"stoploss_on_exchange": True' in source
    assert '"stoploss_price_type": "mark"' in source


def test_runner_keeps_promotion_authority_disabled() -> None:
    source = (ROOT / "tools" / "run_freqtrade_pilot.py").read_text(encoding="utf-8")

    assert '"orders_enabled": False' in source
    assert '"promotion_authority": False' in source
    assert '"eligible_for_paper": False' in source
    assert '"eligible_for_live": False' in source
    assert 'env["PYTHONUTF8"] = "1"' in source


def test_pinned_docker_image_matches_upstream_release() -> None:
    compose = (PILOT / "docker-compose.yml").read_text(encoding="utf-8")
    strategy = (
        PILOT / "user_data" / "strategies" / "UpstreamSampleDryRunStrategy.py"
    ).read_text(encoding="utf-8")

    assert "freqtradeorg/freqtrade:2026.7" in compose
    assert "--userdir /freqtrade/user_data" in compose
    assert "sqlite:////freqtrade/user_data/pilot.dryrun.sqlite" in compose
    assert 'Path("/freqtrade/upstream_templates")' in strategy


def test_generated_report_rejects_sample_strategy_when_present() -> None:
    report_path = ROOT / "Reports" / "FreqtradePilot" / "pilot_report_v1.json"
    if not report_path.exists():
        return

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["eligible_for_paper"] is False
    assert report["eligible_for_live"] is False
    assert report["safety"]["orders_enabled"] is False
    assert report["safety"]["promotion_authority"] is False
