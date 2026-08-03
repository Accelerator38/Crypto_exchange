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


def test_long_horizon_candidate_is_fixed_and_fail_closed() -> None:
    candidate = json.loads(
        (PILOT / "candidates" / "long_horizon_trend_v1.json").read_text(
            encoding="utf-8"
        )
    )
    source = (
        PILOT / "user_data" / "strategies" / "LongHorizonTrendStrategyV1.py"
    ).read_text(encoding="utf-8")

    assert candidate["stage"] == "fixed_retrospective_only"
    assert candidate["entry_contract"] == {
        "ema_fast_hours": 72,
        "ema_slow_hours": 336,
        "breakout_hours": 168,
        "momentum_hours": 672,
        "min_abs_momentum": 0.05,
        "directions": ["LONG", "SHORT"],
    }
    assert candidate["paper_allowed"] is False
    assert candidate["live_allowed"] is False
    assert 'self.config.get("dry_run") is not True' in source
    assert "can_short = True" in source
    assert 'timeframe = "1h"' in source


def test_historical_override_is_explicitly_zero_funding() -> None:
    override = json.loads(
        (PILOT / "user_data" / "config.historical.json").read_text(
            encoding="utf-8"
        )
    )
    runner = (ROOT / "tools" / "run_freqtrade_long_horizon_v1.py").read_text(
        encoding="utf-8"
    )

    assert override == {"futures_funding_rate": 0}
    assert '"paper_allowed": False' in runner
    assert '"live_allowed": False' in runner
    assert '"promotion_authority": False' in runner


def test_long_horizon_report_never_promotes_when_present() -> None:
    report_path = ROOT / "Reports" / "FreqtradePilot" / "long_horizon_trend_v1.json"
    if not report_path.exists():
        return

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["paper_allowed"] is False
    assert report["live_allowed"] is False
    assert report["orders_enabled"] is False
    assert report["promotion_authority"] is False
    assert report["status"] == "TERMINAL_REJECTED_RETROSPECTIVE"
    assert len(report["failed_windows"]) == 8
    assert report["source"]["dataset_sha256"] == (
        "f1479f4e3b2d32a51dd588b8c9c241a8ef71d85da33a27d8dfd5b3a18314481e"
    )
    assert all(
        item["rows"] == 39744 for item in report["conversion"]["pairs"].values()
    )


def test_regime_attribution_is_diagnostic_and_lagged() -> None:
    runner = (ROOT / "tools" / "run_freqtrade_long_horizon_v1.py").read_text(
        encoding="utf-8"
    )

    assert 'decision_timestamp = int(trade["open_timestamp"]) - 3_600_000' in runner
    assert '"used_by_strategy": False' in runner
    assert 'REGIME_ORDER = ("bullish", "bearish", "volatile_mixed"' in runner

    report_path = ROOT / "Reports" / "FreqtradePilot" / "long_horizon_trend_v1.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["regime_diagnostic"]["used_by_strategy"] is False
        assert report["regime_diagnostic"]["attribution_time"] == (
            "last_closed_hour_before_trade_entry"
        )
        assert all(
            values[cost]["market_regimes"]["unknown"]["closed_trades"] == 0
            for values in report["evaluations"].values()
            for cost in ("base", "stress")
        )
