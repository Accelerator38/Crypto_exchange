from __future__ import annotations

import importlib.util
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from panteon_v2.policy.carryflow_adapter import CARRYFLOW_CONFIG_FIELDS
from panteon_v2.policy.manifest import parse_manifest_payload, seal_manifest_payload


ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = ROOT / "tools" / "run_policy_replay_v1.py"
COLLECTOR_PATH = ROOT / "tools" / "collect_bitget_carryflow_tape.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("run_policy_replay_v1", TOOL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_collector():
    spec = importlib.util.spec_from_file_location(
        "collect_bitget_carryflow_tape_for_replay_test",
        COLLECTOR_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tool_builds_exact_research_only_carryflow_manifest():
    tool = _load_tool()

    payload = tool.build_replay_manifest_payload(
        policy_id="carryflow-test",
        symbols=("ADA", "BNB"),
        regimes=("neutral",),
        created_at=datetime(2026, 7, 12, tzinfo=timezone.utc),
        source_revision="d863585",
        runtime_fingerprint_sha256="a" * 64,
        round_trip_fee_bps=8.0,
        slippage_bps=4.0,
        safety_buffer_bps=4.0,
        signal_slope_bps_per_unit=100.0,
        signal_lcb_haircut_bps=4.0,
        signal_min_feature=0.2,
        initial_capital_usd=1000.0,
        capital_fraction=0.01,
        max_notional_usd=10.0,
        max_daily_loss_usd=5.0,
        assumed_spread_bps=2.0,
        stride_minutes=60,
    )
    manifest = parse_manifest_payload(seal_manifest_payload(payload))

    assert manifest.target.value == "replay"
    assert manifest.data.bar_interval_seconds == 3600
    assert manifest.data.max_derivatives_age_seconds == 1200
    assert manifest.evidence == ()
    assert set(dict(manifest.actor_config)) == set(CARRYFLOW_CONFIG_FIELDS)
    assert {rule.symbol for rule in manifest.rules} == {"ADA", "BNB"}
    assert {rule.regime for rule in manifest.rules} == {"neutral"}
    assert {rule.direction.value for rule in manifest.rules} == {"SHORT"}
    assert all(rule.min_expected_move_bps == 16.0 for rule in manifest.rules)


def test_tool_uses_unified_tape_as_authoritative_input(tmp_path):
    tool = _load_tool()
    collector = _load_collector()
    symbols = ("BTC", "ETH")
    tape_path = tmp_path / "tape.jsonl"
    base_close = datetime.now(timezone.utc).replace(
        minute=0,
        second=0,
        microsecond=0,
    ) - timedelta(hours=3)
    revision = "d863585"
    for index in range(3):
        bar_close = base_close + timedelta(hours=index)
        observed = bar_close + timedelta(seconds=30)
        bar_close_ms = int(bar_close.timestamp() * 1000)
        candle_start = bar_close_ms - 3_600_000
        candles = {
            symbol: {
                "candle_timestamp_ms": candle_start,
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
                "volume": 500.0,
            }
            for symbol in symbols
        }
        derivatives = {
            symbol: {
                "funding_rate": 0.0002,
                "open_interest_usdt": 1_000_000.0 + index,
                "long_ratio": 0.65,
                "short_ratio": 0.35,
                "mark_price": 101.0,
                "index_price": 100.0,
                "last_price": 101.0,
                "next_funding_ts": 0,
                "long_short_ratio_ts": int(observed.timestamp() * 1000),
                "long_short_ratio_age_sec": 0.0,
                "long_short_source": "account_long_short_v2",
                "updated_ts": observed.timestamp() - 5.0,
                "context_complete": True,
            }
            for symbol in symbols
        }
        payload = collector.build_tape_payload(
            collector_run_id="replay-tool-test",
            source_revision=revision,
            collector_code_sha256="1" * 64,
            funding_fetcher_code_sha256="2" * 64,
            observed_at=observed,
            bar_interval_seconds=3600,
            bar_close_timestamp_ms=bar_close_ms,
            max_bar_close_lag_seconds=120,
            max_derivatives_age_seconds=1200,
            symbols=symbols,
            candles=candles,
            derivatives=derivatives,
        )
        from panteon_v2.policy import append_tape_sample

        append_tape_sample(tape_path, payload)

    summary = tool.run(
        Namespace(
            data="unused.csv",
            evidence_tape=str(tape_path),
            derivatives_context_csv=None,
            allow_legacy_split_input=False,
            out_dir=str(tmp_path / "report"),
            policy_id="carryflow-tape-test",
            symbols=",".join(symbols),
            regimes="neutral",
            stride_minutes=60,
            max_snapshots=None,
            oos_fraction=0.35,
            use_live_regime_detector=False,
            initial_capital_usd=1000.0,
            capital_fraction=0.01,
            max_notional_usd=10.0,
            max_daily_loss_usd=5.0,
            exchange_min_notional_usd=5.0,
            round_trip_fee_bps=8.0,
            slippage_bps=4.0,
            safety_buffer_bps=4.0,
            assumed_spread_bps=2.0,
            signal_slope_bps_per_unit=100.0,
            signal_lcb_haircut_bps=4.0,
            signal_min_feature=0.2,
        )
    )

    assert summary["input_contract"] == "unified_hash_chained_tape"
    assert summary["robust_input_contract"] is True
    assert summary["evidence_tape"]["samples"] == 3
    assert summary["derivatives_context_coverage_pct"] == 100.0
    assert summary["parity_passed"] is True
    assert summary["evidence_source_revision"] == revision
    assert summary["policy_source_revision"] == tool._git_revision(ROOT)
    assert summary["cross_revision_replay"] is True
