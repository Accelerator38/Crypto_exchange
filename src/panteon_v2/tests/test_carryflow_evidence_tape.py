from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from panteon_v2.policy import (
    CarryFlowWarmupSeed,
    MarketDataPolicy,
    WarmupSeedError,
    write_warmup_seed,
)
from panteon_v2.policy.evidence_tape import (
    CarryFlowEvidenceTape,
    EvidenceTapeError,
    append_tape_sample,
)


ROOT = Path(__file__).resolve().parents[3]
COLLECTOR_PATH = ROOT / "tools" / "collect_bitget_carryflow_tape.py"
NOW = datetime(2026, 7, 12, 12, 0, 30, tzinfo=timezone.utc)
SYMBOLS = ("BTC", "ETH")


def _load_collector():
    spec = importlib.util.spec_from_file_location(
        "collect_bitget_carryflow_tape",
        COLLECTOR_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candles(bar_close_ms: int):
    start = bar_close_ms - 3_600_000
    return {
        symbol: {
            "candle_timestamp_ms": start,
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": 101.0,
            "volume": 500.0,
        }
        for symbol in SYMBOLS
    }


def _derivatives(observed_at: datetime, *, complete: bool = True):
    return {
        symbol: {
            "funding_rate": 0.0002,
            "open_interest_usdt": 1_000_000.0,
            "long_ratio": 0.65,
            "short_ratio": 0.35,
            "mark_price": 101.0,
            "index_price": 100.0,
            "last_price": 101.0,
            "next_funding_ts": 0,
            "long_short_ratio_ts": int(observed_at.timestamp() * 1000),
            "long_short_ratio_age_sec": 0.0,
            "long_short_source": "account_long_short_v2",
            "updated_ts": observed_at.timestamp() - 5.0,
            "context_complete": complete,
        }
        for symbol in SYMBOLS
    }


def _payload(
    index: int,
    *,
    derivatives_complete: bool = True,
    observation_delay_seconds: float = 30.0,
):
    collector = _load_collector()
    observed_at = (
        NOW
        + timedelta(hours=index)
        + timedelta(seconds=observation_delay_seconds - 30.0)
    )
    bar_close_ms = int(
        (datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc) + timedelta(hours=index)).timestamp()
        * 1000
    )
    return collector.build_tape_payload(
        collector_run_id="run-1",
        source_revision="d863585",
        collector_code_sha256="1" * 64,
        funding_fetcher_code_sha256="2" * 64,
        observed_at=observed_at,
        bar_interval_seconds=3600,
        bar_close_timestamp_ms=bar_close_ms,
        max_bar_close_lag_seconds=120,
        max_derivatives_age_seconds=1200,
        symbols=SYMBOLS,
        candles=_candles(bar_close_ms),
        derivatives=_derivatives(
            observed_at,
            complete=derivatives_complete,
        ),
    )


def _policy(**overrides):
    values = {
        "bar_interval_seconds": 3600,
        "cadence_tolerance_seconds": 0,
        "max_bar_close_lag_seconds": 120,
        "max_derivatives_age_seconds": 1200,
        "required_context_coverage_pct": 95.0,
    }
    values.update(overrides)
    return MarketDataPolicy(**values)


def test_tape_hash_chain_builds_exact_market_and_context_bundle(tmp_path):
    path = tmp_path / "tape.jsonl"
    first = append_tape_sample(path, _payload(0))
    second = append_tape_sample(path, _payload(1))

    tape = CarryFlowEvidenceTape.from_jsonl(path, expected_symbols=SYMBOLS)
    bundle = tape.replay_bundle(_policy())

    assert second["previous_sample_sha256"] == first["sample_sha256"]
    assert tape.head_sha256 == second["sample_sha256"]
    assert tape.describe()["complete_samples"] == 2
    assert tape.describe()["context_coverage_pct"] == 100.0
    assert len(bundle.snapshots) == 2
    assert bundle.snapshots[0].prices == {"BTC": 101.0, "ETH": 101.0}
    assert bundle.snapshots[0].bar_ohlc("BTC") == (100.0, 102.0, 99.0, 101.0)
    bundle.derivatives_context.advance(bundle.snapshots[1].timestamp)
    assert bundle.derivatives_context.get("BTC")["context_complete"] is True


def test_replay_cadence_uses_bar_closes_not_network_observation_jitter(tmp_path):
    path = tmp_path / "jittered.jsonl"
    append_tape_sample(path, _payload(0, observation_delay_seconds=51.8))
    append_tape_sample(path, _payload(1, observation_delay_seconds=48.5))

    tape = CarryFlowEvidenceTape.from_jsonl(path, expected_symbols=SYMBOLS)
    bundle = tape.replay_bundle(_policy(cadence_tolerance_seconds=0))

    assert len(bundle.snapshots) == 2
    assert (
        bundle.snapshots[1].cadence_timestamp
        - bundle.snapshots[0].cadence_timestamp
    ).total_seconds() == 3600
    assert (
        bundle.snapshots[1].timestamp - bundle.snapshots[0].timestamp
    ).total_seconds() != 3600


def test_tape_tampering_and_missing_bar_fail_closed(tmp_path):
    path = tmp_path / "tape.jsonl"
    append_tape_sample(path, _payload(0))
    lines = path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[0])
    tampered["symbols"]["BTC"]["market"]["decision_price"] = 999.0
    path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")

    with pytest.raises(EvidenceTapeError, match="hash mismatch"):
        CarryFlowEvidenceTape.from_jsonl(path)

    path = tmp_path / "gap.jsonl"
    append_tape_sample(path, _payload(0))
    with pytest.raises(EvidenceTapeError, match="missing market bar"):
        append_tape_sample(path, _payload(2))


def test_tape_retains_incomplete_context_but_cannot_fake_coverage(tmp_path):
    path = tmp_path / "tape.jsonl"
    sample = append_tape_sample(
        path,
        _payload(0, derivatives_complete=False),
    )
    tape = CarryFlowEvidenceTape.from_jsonl(path)
    bundle = tape.replay_bundle(_policy())

    assert sample["sample_complete"] is False
    assert sample["incomplete_symbols"] == ["BTC", "ETH"]
    assert tape.describe()["context_coverage_pct"] == 0.0
    bundle.derivatives_context.advance(bundle.snapshots[0].timestamp)
    assert bundle.derivatives_context.get("BTC")["context_complete"] is False


def test_tape_manifest_contract_mismatch_is_rejected(tmp_path):
    path = tmp_path / "tape.jsonl"
    append_tape_sample(path, _payload(0))
    tape = CarryFlowEvidenceTape.from_jsonl(path)

    with pytest.raises(EvidenceTapeError, match="bar interval mismatch"):
        tape.replay_bundle(_policy(bar_interval_seconds=300))


def test_public_collector_writes_one_atomic_complete_sample_with_fake_client(tmp_path):
    collector = _load_collector()

    class Exchange:
        def fetch_ohlcv(self, market, timeframe, since, limit):
            assert market in {"BTC/USDT:USDT", "ETH/USDT:USDT"}
            assert timeframe == "1h"
            assert limit == 2
            return [[since, 100.0, 102.0, 99.0, 101.0, 500.0]]

    class Fetcher:
        exchange = Exchange()

        def fetch_all(self, symbols):
            assert tuple(symbols) == SYMBOLS
            return _derivatives(NOW)

    times = iter((NOW, NOW, NOW))
    progress = []
    output = tmp_path / "collector.jsonl"
    status = tmp_path / "collector_status.json"
    summary = collector.collect_tape_samples(
        fetcher=Fetcher(),
        symbols=SYMBOLS,
        output=output,
        samples=1,
        bar_interval_seconds=3600,
        max_bar_close_lag_seconds=120,
        max_derivatives_age_seconds=1200,
        alignment_delay_seconds=30,
        align=False,
        collector_run_id="run-1",
        source_revision="d863585",
        now_fn=lambda: next(times),
        sleep_fn=lambda _seconds: None,
        progress_fn=progress.append,
        status_path=status,
    )

    assert summary["orders_enabled"] is False
    assert summary["complete_samples"] == 1
    assert progress[0]["complete_symbols"] == 2
    assert CarryFlowEvidenceTape.from_jsonl(output).describe()["samples"] == 1
    status_payload = json.loads(status.read_text(encoding="utf-8"))
    assert status_payload["run_state"] == "completed"
    assert status_payload["orders_enabled"] is False


def test_collector_tolerates_subsecond_source_clock_skew():
    collector = _load_collector()
    derivatives = _derivatives(NOW)
    derivatives["ETH"]["updated_ts"] = NOW.timestamp() + 0.5
    bar_close_ms = int(
        datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc).timestamp() * 1000
    )

    payload = collector.build_tape_payload(
        collector_run_id="run-1",
        source_revision="d863585",
        collector_code_sha256="1" * 64,
        funding_fetcher_code_sha256="2" * 64,
        observed_at=NOW,
        bar_interval_seconds=3600,
        bar_close_timestamp_ms=bar_close_ms,
        max_bar_close_lag_seconds=120,
        max_derivatives_age_seconds=1200,
        symbols=SYMBOLS,
        candles=_candles(bar_close_ms),
        derivatives=derivatives,
    )

    assert payload["sample_complete"] is True
    assert payload["symbols"]["ETH"]["derivatives"]["age_sec"] == 0.0


def test_warmup_seed_is_sealed_and_immediately_precedes_tape(tmp_path):
    collector = _load_collector()
    first_close_ms = int(
        datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc).timestamp() * 1000
    )

    class Exchange:
        def fetch_ohlcv(self, market, timeframe, since, limit):
            assert market in {"BTC/USDT:USDT", "ETH/USDT:USDT"}
            assert timeframe == "1h"
            assert limit == 7
            return [
                [
                    since + (index + 1) * 3_600_000,
                    100.0 + index,
                    102.0 + index,
                    99.0 + index,
                    101.0 + index,
                    500.0,
                ]
                for index in range(3)
            ]

    bars = collector.fetch_warmup_bars(
        exchange=Exchange(),
        symbols=SYMBOLS,
        bar_interval_seconds=3600,
        intended_first_evidence_bar_close_timestamp_ms=first_close_ms,
        warmup_bars=3,
    )
    payload = collector.build_warmup_seed_payload(
        collector_run_id="run-1",
        source_revision="d863585",
        collector_code_sha256="1" * 64,
        created_at=NOW - timedelta(minutes=1),
        bar_interval_seconds=3600,
        intended_first_evidence_bar_close_timestamp_ms=first_close_ms,
        symbols=SYMBOLS,
        bars=bars,
    )
    seed_path = tmp_path / "warmup.json"
    write_warmup_seed(seed_path, payload)
    tape_path = tmp_path / "tape.jsonl"
    append_tape_sample(tape_path, _payload(0))
    tape = CarryFlowEvidenceTape.from_jsonl(tape_path)
    seed = CarryFlowWarmupSeed.from_json(seed_path, expected_symbols=SYMBOLS)

    seed.validate_for_tape(tape)
    snapshots = seed.snapshots()
    assert [snapshot.bar for snapshot in snapshots] == [-2, -1, 0]
    assert snapshots[-1].timestamp == datetime(
        2026, 7, 12, 11, 0, tzinfo=timezone.utc
    )
    assert seed.is_prospective_for_tape(tape) is True


def test_warmup_seed_tampering_fails_closed(tmp_path):
    collector = _load_collector()
    first_close_ms = int(
        datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc).timestamp() * 1000
    )
    bars = []
    for index in range(2):
        candle_start = first_close_ms - (3 - index) * 3_600_000
        bars.append(
            {
                "bar_close_timestamp_ms": candle_start + 3_600_000,
                "symbols": {
                    symbol: {
                        "candle_timestamp_ms": candle_start,
                        "open": 100.0,
                        "high": 102.0,
                        "low": 99.0,
                        "close": 101.0,
                        "volume": 500.0,
                    }
                    for symbol in SYMBOLS
                },
            }
        )
    path = tmp_path / "warmup.json"
    write_warmup_seed(
        path,
        collector.build_warmup_seed_payload(
            collector_run_id="run-1",
            source_revision="d863585",
            collector_code_sha256="1" * 64,
            created_at=NOW,
            bar_interval_seconds=3600,
            intended_first_evidence_bar_close_timestamp_ms=first_close_ms,
            symbols=SYMBOLS,
            bars=bars,
        ),
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["bars"][0]["symbols"]["BTC"]["close"] = 100.5
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(WarmupSeedError, match="hash mismatch"):
        CarryFlowWarmupSeed.from_json(path)


def test_collector_resume_appends_one_scheduled_sample_without_process_uptime(
    tmp_path,
    monkeypatch,
):
    collector = _load_collector()
    output = tmp_path / "scheduled.jsonl"
    status = tmp_path / "status.json"
    now = datetime.now(timezone.utc).replace(
        minute=0,
        second=30,
        microsecond=0,
    )
    current_close_ms = collector.expected_bar_close_ms(now, 3600)
    prior_close_ms = current_close_ms - 3_600_000
    prior_observed = datetime.fromtimestamp(
        prior_close_ms / 1000.0 + 30.0,
        timezone.utc,
    )
    source_revision = collector._git_revision(ROOT)
    collector_sha = collector._sha256_file(COLLECTOR_PATH)
    funding_sha = collector._sha256_file(
        ROOT / "src" / "panteon_runtime" / "bitget_funding.py"
    )
    append_tape_sample(
        output,
        collector.build_tape_payload(
            collector_run_id="scheduled-run",
            source_revision=source_revision,
            collector_code_sha256=collector_sha,
            funding_fetcher_code_sha256=funding_sha,
            observed_at=prior_observed,
            bar_interval_seconds=3600,
            bar_close_timestamp_ms=prior_close_ms,
            max_bar_close_lag_seconds=120,
            max_derivatives_age_seconds=1200,
            symbols=SYMBOLS,
            candles=_candles(prior_close_ms),
            derivatives=_derivatives(prior_observed),
        ),
    )

    class Exchange:
        def fetch_ohlcv(self, market, timeframe, since, limit):
            return [[since, 100.0, 102.0, 99.0, 101.0, 500.0]]

    class Fetcher:
        exchange = Exchange()

        def __init__(self, symbols):
            self.symbols = tuple(symbols)

        def fetch_all(self, symbols):
            return _derivatives(now)

    monkeypatch.setattr(collector, "BitgetFundingDataFetcher", Fetcher)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now if tz is not None else now.replace(tzinfo=None)

    monkeypatch.setattr(collector, "datetime", FixedDateTime)

    result = collector.main(
        [
            "--output",
            str(output),
            "--status-path",
            str(status),
            "--resume",
            "--warmup-bars",
            "0",
            "--symbols",
            ",".join(SYMBOLS),
        ]
    )

    tape = CarryFlowEvidenceTape.from_jsonl(output, expected_symbols=SYMBOLS)
    assert result == 0
    assert tape.collector_run_id == "scheduled-run"
    assert len(tape.samples) == 2
