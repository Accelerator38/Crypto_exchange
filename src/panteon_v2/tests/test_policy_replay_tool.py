from __future__ import annotations

import importlib.util
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from panteon_v2.policy.manifest import parse_manifest_payload, seal_manifest_payload
from panteon_v2.policy.warmup_seed import write_warmup_seed


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
        profile_id="screened_short_v1",
        created_at=datetime(2026, 7, 12, tzinfo=timezone.utc),
        source_revision="d863585",
        runtime_fingerprint_sha256="a" * 64,
        round_trip_fee_bps=8.0,
        slippage_bps=4.0,
        safety_buffer_bps=4.0,
        capital_fraction=0.01,
        max_notional_usd=10.0,
        max_daily_loss_usd=5.0,
        stride_minutes=60,
    )
    manifest = parse_manifest_payload(seal_manifest_payload(payload))

    assert manifest.target.value == "replay"
    assert manifest.data.bar_interval_seconds == 3600
    assert manifest.data.max_derivatives_age_seconds == 1200
    assert manifest.evidence == ()
    assert manifest.signal_model.feature == "diagnostic.edge"
    assert dict(manifest.actor_config) == {"PROFILE_ID": "screened_short_v1"}
    assert {rule.symbol for rule in manifest.rules} == {"ADA", "BNB"}
    assert "neutral" in {rule.regime for rule in manifest.rules}
    assert "range_low_vol" in {rule.regime for rule in manifest.rules}
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

    seed_path = tmp_path / "warmup.json"
    first_close_ms = int(base_close.timestamp() * 1000)
    interval_ms = 3_600_000
    first_seed_start = first_close_ms - 54 * interval_ms
    warmup_bars = []
    for index in range(53):
        start = first_seed_start + index * interval_ms
        price = 95.0 + index * 0.1
        warmup_bars.append(
            {
                "bar_close_timestamp_ms": start + interval_ms,
                "symbols": {
                    symbol: {
                        "candle_timestamp_ms": start,
                        "open": price,
                        "high": price + 1.0,
                        "low": price - 1.0,
                        "close": price,
                        "volume": 500.0,
                    }
                    for symbol in symbols
                },
            }
        )
    write_warmup_seed(
        seed_path,
        collector.build_warmup_seed_payload(
            collector_run_id="replay-tool-test",
            source_revision=revision,
            collector_code_sha256="1" * 64,
            created_at=base_close - timedelta(hours=55),
            bar_interval_seconds=3600,
            intended_first_evidence_bar_close_timestamp_ms=first_close_ms,
            symbols=symbols,
            bars=warmup_bars,
        ),
    )

    args = Namespace(
        data="unused.csv",
        evidence_tape=str(tape_path),
        warmup_seed=str(seed_path),
        derivatives_context_csv=None,
        allow_legacy_split_input=False,
        out_dir=str(tmp_path / "report"),
        policy_id="carryflow-tape-test",
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
        profile="screened_short_v1",
    )
    summary = tool.run(args)

    assert summary["input_contract"] == "unified_hash_chained_tape"
    assert summary["robust_input_contract"] is True
    assert summary["evidence_tape"]["samples"] == 3
    assert summary["symbols"] == list(symbols)
    assert summary["bar_interval_minutes"] == 60
    assert summary["input_settings_source"] == "evidence_tape_contract"
    assert summary["warmup_bars"] == 53
    assert summary["continuous_warmup_bars_avoided"] == 53
    assert summary["flatten_end"] is True
    assert summary["end_positions_censored"] == 0
    assert summary["warmup_seed"]["prospective_for_tape"] is True
    assert summary["research_hypothesis"]["profile_id"] == "screened_short_v1"
    assert summary["research_hypothesis"]["runtime_knob_count"] == 1
    assert "not_enough_history" not in summary["actor_diagnostic_reasons"]
    assert summary["derivatives_context_coverage_pct"] == 100.0
    assert summary["parity_passed"] is True
    assert summary["evidence_source_revision"] == revision
    assert summary["policy_source_revision"] == tool._git_revision(ROOT)
    assert summary["cross_revision_replay"] is True

    mismatched_symbols = dict(vars(args))
    mismatched_symbols["symbols"] = "BTC"
    with pytest.raises(ValueError, match="symbol set mismatch"):
        tool.run(Namespace(**mismatched_symbols))

    mismatched_stride = dict(vars(args))
    mismatched_stride["stride_minutes"] = 5
    with pytest.raises(ValueError, match="does not match"):
        tool.run(Namespace(**mismatched_stride))
