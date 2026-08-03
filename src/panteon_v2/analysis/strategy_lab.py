from __future__ import annotations

import csv
import hashlib
import json
import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Mapping, Sequence

import numpy as np

from panteon_v2.analysis.bitget_funding_history import load_funding_snapshot
from panteon_v2.policy.strategy_candidate_registry import (
    StrategyCandidateRegistry,
)


STRATEGY_LAB_SCHEMA_VERSION = "panteon.strategy_lab_evaluation.v1"


class StrategyLabError(ValueError):
    pass


@dataclass(frozen=True)
class MarketBar:
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class MarketFrame:
    timestamp_ms: int
    bars: dict[str, MarketBar]


@dataclass(frozen=True)
class SymbolSeries:
    symbol: str
    timestamps: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    true_range: np.ndarray
    atr14: np.ndarray
    ema24: np.ndarray
    ema96: np.ndarray
    volume_median24: np.ndarray
    bollinger_width24: np.ndarray


@dataclass(frozen=True)
class EntrySignal:
    symbol: str
    direction: str
    signal_index: int
    score: float
    atr: float
    regime: str
    range_high: float | None = None
    range_low: float | None = None


@dataclass
class OpenPosition:
    signal: EntrySignal
    entry_index: int
    entry_timestamp_ms: int
    entry_price: float
    stop_price: float
    target_price: float
    pending_exit_reason: str = ""
    funding_bps: float = 0.0


@dataclass(frozen=True)
class PairSignal:
    long_symbol: str
    short_symbol: str
    signal_index: int
    score_spread: float
    momentum_spread_bps: float


@dataclass
class PairPosition:
    signal: PairSignal
    entry_index: int
    entry_timestamp_ms: int
    long_entry_price: float
    short_entry_price: float
    pending_exit_reason: str = ""


@dataclass(frozen=True)
class PortfolioSignal:
    long_symbols: tuple[str, ...]
    short_symbols: tuple[str, ...]
    signal_index: int
    score_dispersion: float
    momentum_dispersion_bps: float


@dataclass
class PortfolioPosition:
    signal: PortfolioSignal
    entry_index: int
    entry_timestamp_ms: int
    entry_prices: dict[str, float]


def verify_registered_dataset(
    registry: StrategyCandidateRegistry,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root)
    dataset = registry.payload["dataset"]
    data_dir = root / Path(str(dataset["data_dir"]))
    manifest_path = data_dir / "integrity_manifest.json"
    if not manifest_path.is_file():
        raise StrategyLabError(f"integrity manifest missing: {manifest_path}")
    manifest_sha = _sha256_file(manifest_path)
    if manifest_sha != dataset["integrity_manifest_sha256"]:
        raise StrategyLabError("integrity manifest SHA-256 mismatch")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StrategyLabError("cannot read integrity manifest") from exc
    if manifest.get("dataset_sha256") != dataset["dataset_sha256"]:
        raise StrategyLabError("dataset SHA-256 mismatch")

    expected_files = manifest.get("files")
    if not isinstance(expected_files, list) or not expected_files:
        raise StrategyLabError("integrity manifest has no files")
    verified_files: list[dict[str, Any]] = []
    for row in expected_files:
        path = data_dir / Path(str(row.get("path") or "")).name
        if not path.is_file():
            raise StrategyLabError(f"registered dataset file missing: {path}")
        actual_sha = _sha256_file(path)
        if actual_sha != str(row.get("sha256") or "").lower():
            raise StrategyLabError(f"dataset file SHA-256 mismatch: {path.name}")
        verified_files.append(
            {"file": path.name, "bytes": path.stat().st_size, "sha256": actual_sha}
        )
    return {
        "dataset_id": dataset["dataset_id"],
        "dataset_sha256": dataset["dataset_sha256"],
        "integrity_manifest_sha256": manifest_sha,
        "data_dir": str(dataset["data_dir"]),
        "files": verified_files,
        "verified": True,
    }


def load_aligned_hourly_market(
    registry: StrategyCandidateRegistry,
    *,
    repository_root: str | Path,
) -> tuple[list[MarketFrame], dict[str, SymbolSeries]]:
    dataset = registry.payload["dataset"]
    data_dir = Path(repository_root) / Path(str(dataset["data_dir"]))
    symbols = tuple(str(item) for item in dataset["symbols"])
    expected = set(symbols)
    by_timestamp: dict[int, dict[str, MarketBar]] = {}

    for path in sorted(data_dir.glob("crypto_*_*_all_symbols.csv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                symbol = _base_symbol(row.get("symbol"))
                if symbol not in expected:
                    continue
                try:
                    timestamp_ms = int(row["timestamp"])
                    bar = MarketBar(
                        timestamp_ms=timestamp_ms,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise StrategyLabError(f"invalid OHLCV row in {path.name}") from exc
                if (
                    min(bar.open, bar.high, bar.low, bar.close) <= 0.0
                    or bar.volume < 0.0
                    or bar.high < max(bar.open, bar.close, bar.low)
                    or bar.low > min(bar.open, bar.close, bar.high)
                ):
                    raise StrategyLabError(
                        f"invalid OHLCV values in {path.name} at {timestamp_ms}"
                    )
                bucket = by_timestamp.setdefault(timestamp_ms, {})
                if symbol in bucket:
                    raise StrategyLabError(
                        f"duplicate OHLCV row for {symbol} at {timestamp_ms}"
                    )
                bucket[symbol] = bar

    frames: list[MarketFrame] = []
    previous: int | None = None
    for timestamp_ms in sorted(by_timestamp):
        bars = by_timestamp[timestamp_ms]
        if set(bars) != expected:
            raise StrategyLabError(
                f"incomplete full8 frame at {timestamp_ms}; "
                f"missing={sorted(expected - set(bars))}"
            )
        if previous is not None and timestamp_ms - previous != 3_600_000:
            raise StrategyLabError(
                f"hourly cadence gap between {previous} and {timestamp_ms}"
            )
        frames.append(MarketFrame(timestamp_ms=timestamp_ms, bars=bars))
        previous = timestamp_ms
    if not frames:
        raise StrategyLabError("registered dataset has no aligned frames")
    return frames, {
        symbol: _build_symbol_series(symbol, frames) for symbol in symbols
    }


def evaluate_registered_candidates(
    registry: StrategyCandidateRegistry,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    provenance = verify_registered_dataset(registry, repository_root=root)
    frames, series = load_aligned_hourly_market(
        registry, repository_root=repository_root
    )
    evaluation = registry.payload["common_evaluation"]
    timestamps = [frame.timestamp_ms for frame in frames]
    baseline = _baseline_candidate()
    baseline_splits = _evaluate_splits(
        baseline,
        frames=frames,
        series=series,
        timestamps=timestamps,
        evaluation=evaluation,
    )
    baseline_summary = _summarize_candidate(
        baseline,
        baseline_splits,
        evaluation=evaluation,
        baseline_splits=None,
    )

    candidate_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    for candidate in registry.payload["candidates"]:
        if not candidate["historical_evaluation_allowed"]:
            blocked_rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "status": candidate["status"],
                    "missing_fields": candidate["data_requirements"]["missing_fields"],
                    "profile_sha256": candidate["profile_sha256"],
                    "orders_enabled": False,
                    "promotion_authority": False,
                }
            )
            continue
        splits = _evaluate_splits(
            candidate,
            frames=frames,
            series=series,
            timestamps=timestamps,
            evaluation=evaluation,
        )
        candidate_rows.append(
            _summarize_candidate(
                candidate,
                splits,
                evaluation=evaluation,
                baseline_splits=baseline_splits,
            )
        )

    return {
        "schema_version": STRATEGY_LAB_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "registry_path": _relative_report_path(registry.path, root),
        "dataset": provenance,
        "frame_count": len(frames),
        "first_timestamp_ms": frames[0].timestamp_ms,
        "last_timestamp_ms": frames[-1].timestamp_ms,
        "costs": evaluation["costs"],
        "portfolio": evaluation["portfolio"],
        "gates": evaluation["gates"],
        "baseline": baseline_summary,
        "candidates": candidate_rows,
        "blocked_candidates": blocked_rows,
        "operational_candidate_id": None,
        "runtime_actor_created": False,
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }


def evaluate_funding_recent_screen(
    registry: StrategyCandidateRegistry,
    *,
    repository_root: str | Path,
    funding_snapshot_dir: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    candidate = registry.candidate("funding_carry_hourly_v1")
    if candidate["status"] != "blocked_missing_data":
        raise StrategyLabError("funding candidate must remain data-blocked")
    symbols = tuple(str(item) for item in registry.payload["dataset"]["symbols"])
    funding, funding_manifest = load_funding_snapshot(
        funding_snapshot_dir,
        expected_symbols=symbols,
    )
    frames, series = load_aligned_hourly_market(
        registry,
        repository_root=root,
    )
    timestamps = [frame.timestamp_ms for frame in frames]
    common_window = funding_manifest["observed_common_window"]
    start_index = bisect_left(
        timestamps,
        int(common_window["start_timestamp_ms"]),
    )
    end_index = bisect_right(
        timestamps,
        int(common_window["end_timestamp_ms"]),
    ) - 1
    if end_index < start_index:
        raise StrategyLabError("funding snapshot has no overlap with OHLCV")
    evaluation = registry.payload["common_evaluation"]
    split = _simulate_split(
        candidate,
        split_name="recent_screen",
        frames=frames,
        series=series,
        start_index=start_index,
        end_index=end_index,
        costs=evaluation["costs"],
        portfolio=evaluation["portfolio"],
        funding_by_symbol=funding,
    )
    activation = _funding_activation_funnel(
        candidate,
        frames=frames,
        series=series,
        start_index=start_index,
        end_index=end_index,
        funding_by_symbol=funding,
    )
    cost_floor_bps = (
        float(evaluation["costs"]["round_trip_fee_bps"])
        + float(evaluation["costs"]["slippage_bps"])
        + float(evaluation["costs"]["safety_buffer_bps"])
    )
    early_activation_failure = (
        int(split["candidate_signals"]) == 0
        and float(activation["maxima"]["projected_bps"]) < cost_floor_bps
    )
    return {
        "schema_version": STRATEGY_LAB_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "screen_type": "funding_carry_recent_diagnostic",
        "candidate_id": candidate["candidate_id"],
        "profile_sha256": candidate["profile_sha256"],
        "registry_path": _relative_report_path(registry.path, root),
        "funding_snapshot": {
            "dataset_sha256": funding_manifest["dataset_sha256"],
            "manifest_sha256": funding_manifest["manifest_sha256"],
            "observed_common_window": funding_manifest[
                "observed_common_window"
            ],
            "registered_oos_coverage_complete": funding_manifest[
                "registered_oos_coverage_complete"
            ],
            "allowed_use": funding_manifest["allowed_use"],
        },
        "ohlcv_dataset": {
            "dataset_id": registry.payload["dataset"]["dataset_id"],
            "dataset_sha256": registry.payload["dataset"]["dataset_sha256"],
        },
        "overlap": {
            "start_timestamp_ms": frames[start_index].timestamp_ms,
            "end_timestamp_ms": frames[end_index].timestamp_ms,
            "bars": end_index - start_index + 1,
        },
        "costs": evaluation["costs"],
        "portfolio": evaluation["portfolio"],
        "screen": split,
        "activation_funnel": activation,
        "early_rejection": {
            "applied": early_activation_failure,
            "reason": (
                "zero_signals_and_max_projected_carry_below_cost_floor"
                if early_activation_failure
                else ""
            ),
            "cost_floor_bps": cost_floor_bps,
            "max_projected_carry_bps": activation["maxima"][
                "projected_bps"
            ],
        },
        "verdict": (
            "terminal_rejected_recent_activation_and_cost_floor"
            if early_activation_failure
            else "recent_screen_only_registered_oos_still_blocked"
        ),
        "registered_historical_oos_passed": False,
        "continuation_allowed": not early_activation_failure,
        "profile_registration_allowed": False,
        "runtime_actor_created": False,
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }


def evaluate_cross_sectional_development(
    contract: Mapping[str, Any],
    registry: StrategyCandidateRegistry,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    contract_dataset = contract["dataset"]
    registered_dataset = registry.payload["dataset"]
    for key in (
        "dataset_id",
        "data_dir",
        "dataset_sha256",
        "integrity_manifest_sha256",
        "symbols",
    ):
        if contract_dataset[key] != registered_dataset[key]:
            raise StrategyLabError(
                f"development dataset mismatch: {key}"
            )
    provenance = verify_registered_dataset(
        registry,
        repository_root=root,
    )
    hourly_frames, _ = load_aligned_hourly_market(
        registry,
        repository_root=root,
    )
    candidate = contract["candidate"]
    aggregate_bars = int(candidate["event_contract"]["aggregate_bars"])
    frames = aggregate_market_frames(
        hourly_frames,
        bars_per_frame=aggregate_bars,
    )
    symbols = tuple(str(item) for item in contract_dataset["symbols"])
    series = {
        symbol: _build_symbol_series(symbol, frames) for symbol in symbols
    }
    timestamps = [frame.timestamp_ms for frame in frames]
    protocol = contract["protocol"]
    start_index = bisect_left(
        timestamps,
        _timestamp_ms(protocol["development_start_at"]),
    )
    end_index = bisect_right(
        timestamps,
        _timestamp_ms(protocol["development_end_at"]),
    ) - 1
    if end_index < start_index:
        raise StrategyLabError("development split has no 4h bars")
    split = _simulate_split(
        candidate,
        split_name="development",
        frames=frames,
        series=series,
        start_index=start_index,
        end_index=end_index,
        costs=protocol["costs"],
        portfolio=protocol["portfolio"],
    )
    baseline = _cross_sectional_baseline_candidate()
    baseline_split = _simulate_split(
        baseline,
        split_name="development",
        frames=frames,
        series=series,
        start_index=start_index,
        end_index=end_index,
        costs=protocol["costs"],
        portfolio=protocol["portfolio"],
    )
    verdict = _development_verdict(
        candidate,
        split=split,
        baseline_split=baseline_split,
        gates=protocol["gates"],
    )
    return {
        "schema_version": STRATEGY_LAB_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "screen_type": "cross_sectional_trend_4h_development",
        "stage": "development_only",
        "sealed_windows": list(protocol["sealed_windows"]),
        "candidate_id": candidate["candidate_id"],
        "profile_sha256": candidate["profile_sha256"],
        "contract_path": "configs/strategy_candidate_p3_cross_sectional_v1.json",
        "dataset": provenance,
        "aggregation": {
            "source_timeframe": "1h",
            "bars_per_frame": aggregate_bars,
            "result_timeframe": "4h",
            "frame_count": len(frames),
        },
        "development_window": {
            "start_timestamp_ms": frames[start_index].timestamp_ms,
            "end_timestamp_ms": frames[end_index].timestamp_ms,
            "bars": end_index - start_index + 1,
        },
        "costs": protocol["costs"],
        "portfolio": protocol["portfolio"],
        "gates": protocol["gates"],
        "candidate": split,
        "baseline": baseline_split,
        **verdict,
        "runtime_actor_created": False,
        "validation_opened": False,
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }


def evaluate_market_neutral_pair_development(
    contract: Mapping[str, Any],
    registry: StrategyCandidateRegistry,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    contract_dataset = contract["dataset"]
    registered_dataset = registry.payload["dataset"]
    for key in (
        "dataset_id",
        "data_dir",
        "dataset_sha256",
        "integrity_manifest_sha256",
        "symbols",
    ):
        if contract_dataset[key] != registered_dataset[key]:
            raise StrategyLabError(f"paired development dataset mismatch: {key}")
    provenance = verify_registered_dataset(registry, repository_root=root)
    hourly_frames, _ = load_aligned_hourly_market(
        registry,
        repository_root=root,
    )
    candidate = contract["candidate"]
    aggregate_bars = int(candidate["event_contract"]["aggregate_bars"])
    frames = aggregate_market_frames(
        hourly_frames,
        bars_per_frame=aggregate_bars,
    )
    symbols = tuple(str(item) for item in contract_dataset["symbols"])
    series = {
        symbol: _build_symbol_series(symbol, frames) for symbol in symbols
    }
    timestamps = [frame.timestamp_ms for frame in frames]
    protocol = contract["protocol"]
    start_index = bisect_left(
        timestamps,
        _timestamp_ms(protocol["development_start_at"]),
    )
    end_index = bisect_right(
        timestamps,
        _timestamp_ms(protocol["development_end_at"]),
    ) - 1
    if end_index < start_index:
        raise StrategyLabError("paired development split has no 4h bars")

    split = _simulate_pair_split(
        candidate,
        split_name="development",
        frames=frames,
        series=series,
        start_index=start_index,
        end_index=end_index,
        costs=protocol["costs"],
        portfolio=protocol["portfolio"],
    )
    baseline_candidate = _paired_baseline_candidate(candidate)
    baseline_split = _simulate_pair_split(
        baseline_candidate,
        split_name="development",
        frames=frames,
        series=series,
        start_index=start_index,
        end_index=end_index,
        costs=protocol["costs"],
        portfolio=protocol["portfolio"],
    )
    verdict = _paired_development_verdict(
        split=split,
        baseline_split=baseline_split,
        gates=protocol["gates"],
    )
    return {
        "schema_version": STRATEGY_LAB_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "screen_type": "market_neutral_relative_momentum_4h_development",
        "stage": "development_only",
        "sealed_windows": list(protocol["sealed_windows"]),
        "candidate_id": candidate["candidate_id"],
        "profile_sha256": candidate["profile_sha256"],
        "contract_path": (
            "configs/strategy_candidate_p4_market_neutral_pair_v1.json"
        ),
        "dataset": provenance,
        "aggregation": {
            "source_timeframe": "1h",
            "bars_per_frame": aggregate_bars,
            "result_timeframe": "4h",
            "frame_count": len(frames),
        },
        "development_window": {
            "start_timestamp_ms": frames[start_index].timestamp_ms,
            "end_timestamp_ms": frames[end_index].timestamp_ms,
            "bars": end_index - start_index + 1,
        },
        "costs": protocol["costs"],
        "portfolio": protocol["portfolio"],
        "gates": protocol["gates"],
        "candidate": split,
        "baseline": baseline_split,
        **verdict,
        "runtime_actor_created": False,
        "validation_opened": False,
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }


def evaluate_market_neutral_portfolio_development(
    contract: Mapping[str, Any],
    registry: StrategyCandidateRegistry,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    contract_dataset = contract["dataset"]
    registered_dataset = registry.payload["dataset"]
    for key in (
        "dataset_id",
        "data_dir",
        "dataset_sha256",
        "integrity_manifest_sha256",
        "symbols",
    ):
        if contract_dataset[key] != registered_dataset[key]:
            raise StrategyLabError(
                f"portfolio development dataset mismatch: {key}"
            )
    provenance = verify_registered_dataset(registry, repository_root=root)
    hourly_frames, _ = load_aligned_hourly_market(
        registry,
        repository_root=root,
    )
    candidate = contract["candidate"]
    aggregate_bars = int(candidate["event_contract"]["aggregate_bars"])
    frames = aggregate_market_frames(
        hourly_frames,
        bars_per_frame=aggregate_bars,
    )
    symbols = tuple(str(item) for item in contract_dataset["symbols"])
    series = {
        symbol: _build_symbol_series(symbol, frames) for symbol in symbols
    }
    timestamps = [frame.timestamp_ms for frame in frames]
    protocol = contract["protocol"]
    start_index = bisect_left(
        timestamps,
        _timestamp_ms(protocol["development_start_at"]),
    )
    end_index = bisect_right(
        timestamps,
        _timestamp_ms(protocol["development_end_at"]),
    ) - 1
    if end_index < start_index:
        raise StrategyLabError("portfolio development split has no 4h bars")

    split = _simulate_portfolio_split(
        candidate,
        split_name="development",
        frames=frames,
        series=series,
        start_index=start_index,
        end_index=end_index,
        costs=protocol["costs"],
        portfolio=protocol["portfolio"],
    )
    baseline_candidate = _portfolio_baseline_candidate(candidate)
    baseline_split = _simulate_portfolio_split(
        baseline_candidate,
        split_name="development",
        frames=frames,
        series=series,
        start_index=start_index,
        end_index=end_index,
        costs=protocol["costs"],
        portfolio=protocol["portfolio"],
    )
    verdict = _portfolio_development_verdict(
        split=split,
        baseline_split=baseline_split,
        gates=protocol["gates"],
    )
    return {
        "schema_version": STRATEGY_LAB_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "screen_type": "weekly_top2_bottom2_4h_development",
        "stage": "development_only",
        "sealed_windows": list(protocol["sealed_windows"]),
        "candidate_id": candidate["candidate_id"],
        "profile_sha256": candidate["profile_sha256"],
        "contract_path": (
            "configs/strategy_candidate_p5_market_neutral_portfolio_v1.json"
        ),
        "dataset": provenance,
        "aggregation": {
            "source_timeframe": "1h",
            "bars_per_frame": aggregate_bars,
            "result_timeframe": "4h",
            "frame_count": len(frames),
        },
        "development_window": {
            "start_timestamp_ms": frames[start_index].timestamp_ms,
            "end_timestamp_ms": frames[end_index].timestamp_ms,
            "bars": end_index - start_index + 1,
        },
        "costs": protocol["costs"],
        "portfolio": protocol["portfolio"],
        "gates": protocol["gates"],
        "candidate": split,
        "baseline": baseline_split,
        **verdict,
        "runtime_actor_created": False,
        "validation_opened": False,
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }


def aggregate_market_frames(
    frames: Sequence[MarketFrame],
    *,
    bars_per_frame: int,
) -> list[MarketFrame]:
    width = int(bars_per_frame)
    if width <= 1:
        raise StrategyLabError("aggregate width must be greater than one")
    result: list[MarketFrame] = []
    for offset in range(0, len(frames) - width + 1, width):
        window = frames[offset:offset + width]
        if any(
            current.timestamp_ms - previous.timestamp_ms != 3_600_000
            for previous, current in zip(window, window[1:])
        ):
            raise StrategyLabError("cannot aggregate discontinuous hourly bars")
        symbols = set(window[0].bars)
        if any(set(frame.bars) != symbols for frame in window):
            raise StrategyLabError("cannot aggregate incomplete full8 frame")
        bars: dict[str, MarketBar] = {}
        for symbol in symbols:
            rows = [frame.bars[symbol] for frame in window]
            bars[symbol] = MarketBar(
                timestamp_ms=window[0].timestamp_ms,
                open=rows[0].open,
                high=max(row.high for row in rows),
                low=min(row.low for row in rows),
                close=rows[-1].close,
                volume=sum(row.volume for row in rows),
            )
        result.append(
            MarketFrame(
                timestamp_ms=window[0].timestamp_ms,
                bars=bars,
            )
        )
    if not result:
        raise StrategyLabError("aggregation produced no frames")
    return result


def write_strategy_lab_report(
    report: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "strategy_lab_summary.json"
    markdown_path = target / "strategy_lab_summary.md"
    compact_report = _compact_report(report)
    json_path.write_text(
        json.dumps(compact_report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def write_funding_recent_screen_report(
    report: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "funding_carry_recent_screen.json"
    markdown_path = target / "funding_carry_recent_screen.md"
    json_path.write_text(
        json.dumps(
            _compact_report(report),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        _render_funding_screen_markdown(report),
        encoding="utf-8",
    )
    return json_path, markdown_path


def write_cross_sectional_development_report(
    report: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "cross_sectional_trend_development.json"
    markdown_path = target / "cross_sectional_trend_development.md"
    json_path.write_text(
        json.dumps(
            _compact_report(report),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        _render_cross_sectional_development_markdown(report),
        encoding="utf-8",
    )
    return json_path, markdown_path


def write_market_neutral_pair_development_report(
    report: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "market_neutral_pair_development.json"
    markdown_path = target / "market_neutral_pair_development.md"
    json_path.write_text(
        json.dumps(
            _compact_report(report),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        _render_market_neutral_pair_development_markdown(report),
        encoding="utf-8",
    )
    return json_path, markdown_path


def write_market_neutral_portfolio_development_report(
    report: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "market_neutral_portfolio_development.json"
    markdown_path = target / "market_neutral_portfolio_development.md"
    json_path.write_text(
        json.dumps(
            _compact_report(report),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        _render_market_neutral_portfolio_development_markdown(report),
        encoding="utf-8",
    )
    return json_path, markdown_path


def _compact_report(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _compact_report(item)
            for key, item in value.items()
            if not (
                key in {"trades", "leg_trades"}
                and isinstance(item, list)
            )
        }
    if isinstance(value, list):
        return [_compact_report(item) for item in value]
    return value


def _relative_report_path(path: str | Path, root: Path) -> str:
    source = Path(path).resolve()
    try:
        return source.relative_to(root).as_posix()
    except ValueError:
        return source.as_posix()


def detect_entry_signals(
    candidate: Mapping[str, Any],
    *,
    index: int,
    series: Mapping[str, SymbolSeries],
    funding_by_symbol: Mapping[str, Mapping[int, float]] | None = None,
) -> list[EntrySignal]:
    candidate_id = str(candidate["candidate_id"])
    if candidate_id == "regime_pullback_hourly_v1":
        return _pullback_signals(candidate, index=index, series=series)
    if candidate_id == "ohlcv_compression_transition_hourly_v1":
        return _compression_signals(candidate, index=index, series=series)
    if candidate_id.startswith("donchian_"):
        return _donchian_signals(candidate, index=index, series=series)
    if candidate_id == "funding_carry_hourly_v1":
        return _funding_carry_signals(
            candidate,
            index=index,
            series=series,
            funding_by_symbol=funding_by_symbol,
        )
    if candidate_id == "cross_sectional_trend_4h_v1":
        return _cross_sectional_trend_signals(
            candidate,
            index=index,
            series=series,
        )
    raise StrategyLabError(f"unsupported candidate: {candidate_id}")


def _evaluate_splits(
    candidate: Mapping[str, Any],
    *,
    frames: Sequence[MarketFrame],
    series: Mapping[str, SymbolSeries],
    timestamps: Sequence[int],
    evaluation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in evaluation["splits"]:
        start_index = bisect_left(timestamps, _timestamp_ms(split["start_at"]))
        end_index = bisect_right(timestamps, _timestamp_ms(split["end_at"])) - 1
        if end_index < start_index:
            raise StrategyLabError(f"split has no bars: {split['name']}")
        rows.append(
            _simulate_split(
                candidate,
                split_name=str(split["name"]),
                frames=frames,
                series=series,
                start_index=start_index,
                end_index=end_index,
                costs=evaluation["costs"],
                portfolio=evaluation["portfolio"],
            )
        )
    return rows


def _simulate_split(
    candidate: Mapping[str, Any],
    *,
    split_name: str,
    frames: Sequence[MarketFrame],
    series: Mapping[str, SymbolSeries],
    start_index: int,
    end_index: int,
    costs: Mapping[str, Any],
    portfolio: Mapping[str, Any],
    funding_by_symbol: Mapping[str, Mapping[int, float]] | None = None,
) -> dict[str, Any]:
    real_cost_bps = float(costs["round_trip_fee_bps"]) + float(
        costs["slippage_bps"]
    )
    stressed_cost_bps = max(
        real_cost_bps * float(costs["cost_stress_multiplier"]),
        real_cost_bps + float(costs.get("safety_buffer_bps", 0.0)),
    )
    notional = float(portfolio["notional_per_trade_usd"])
    position: OpenPosition | None = None
    pending_entry: EntrySignal | None = None
    cooldowns: dict[str, int] = {}
    trades: list[dict[str, Any]] = []
    candidate_signals = selected_signals = entry_fills = exit_fills = 0

    for index in range(start_index, end_index + 1):
        frame = frames[index]
        if pending_entry is not None and position is None:
            bar = frame.bars[pending_entry.symbol]
            position = _open_position(candidate, pending_entry, bar, index=index)
            cooldowns[pending_entry.symbol] = index
            pending_entry = None
            entry_fills += 1

        if position is not None:
            bar = frame.bars[position.signal.symbol]
            funding_rate = _funding_rate_at(
                funding_by_symbol,
                symbol=position.signal.symbol,
                timestamp_ms=frame.timestamp_ms,
            )
            if funding_rate is not None:
                position.funding_bps += (
                    funding_rate * 10_000.0
                    if position.signal.direction == "SHORT"
                    else -funding_rate * 10_000.0
                )
            exit_price: float | None = None
            exit_reason = ""
            if position.pending_exit_reason:
                exit_price, exit_reason = bar.open, position.pending_exit_reason
            elif index - position.entry_index >= int(
                candidate["exit_contract"]["max_holding_bars"]
            ):
                exit_price, exit_reason = bar.open, "max_holding"
            else:
                exit_price, exit_reason = protective_exit(position, bar)
            if exit_price is not None:
                trades.append(
                    _close_trade(
                        position,
                        exit_index=index,
                        exit_timestamp_ms=frame.timestamp_ms,
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        real_cost_bps=real_cost_bps,
                        stressed_cost_bps=stressed_cost_bps,
                        notional_usd=notional,
                    )
                )
                position = None
                exit_fills += 1

        if position is not None and not position.pending_exit_reason:
            reason = _close_signal_reason(
                candidate,
                position,
                index=index,
                series=series,
                funding_by_symbol=funding_by_symbol,
            )
            if reason and index < end_index:
                position.pending_exit_reason = reason

        signals = detect_entry_signals(
            candidate,
            index=index,
            series=series,
            funding_by_symbol=funding_by_symbol,
        )
        candidate_signals += len(signals)
        if position is None and pending_entry is None and index < end_index:
            cooldown_bars = int(candidate["selection_contract"]["cooldown_bars"])
            eligible = [
                signal
                for signal in signals
                if index - cooldowns.get(signal.symbol, -10**9) >= cooldown_bars
            ]
            if eligible:
                pending_entry = sorted(
                    eligible,
                    key=lambda item: (-item.score, item.symbol, item.direction),
                )[0]
                selected_signals += 1

    metrics = _metrics(trades, net_field="net_bps")
    return {
        "split": split_name,
        "start_timestamp_ms": frames[start_index].timestamp_ms,
        "end_timestamp_ms": frames[end_index].timestamp_ms,
        "bars": end_index - start_index + 1,
        "candidate_signals": candidate_signals,
        "selected_signals": selected_signals,
        "entry_fills": entry_fills,
        "exit_fills": exit_fills,
        "filled_orders": entry_fills + exit_fills,
        "closed_trades": len(trades),
        "remaining_open_positions": int(position is not None),
        "right_censored_positions": int(position is not None),
        "metrics": metrics,
        "cost_stress_metrics": _metrics(trades, net_field="stressed_net_bps"),
        "per_symbol": _group_metrics(trades, "symbol"),
        "per_direction": _group_metrics(trades, "direction"),
        "per_regime": _group_metrics(trades, "regime"),
        "trades": trades,
    }


def _simulate_pair_split(
    candidate: Mapping[str, Any],
    *,
    split_name: str,
    frames: Sequence[MarketFrame],
    series: Mapping[str, SymbolSeries],
    start_index: int,
    end_index: int,
    costs: Mapping[str, Any],
    portfolio: Mapping[str, Any],
) -> dict[str, Any]:
    real_cost_bps = float(costs["round_trip_fee_bps"]) + float(
        costs["slippage_bps"]
    )
    stressed_cost_bps = max(
        real_cost_bps * float(costs["cost_stress_multiplier"]),
        real_cost_bps + float(costs.get("safety_buffer_bps", 0.0)),
    )
    notional_per_leg = float(portfolio["notional_per_trade_usd"])
    position: PairPosition | None = None
    pending_entry: PairSignal | None = None
    last_exit_index = -10**9
    pair_trades: list[dict[str, Any]] = []
    leg_trades: list[dict[str, Any]] = []
    candidate_signals = selected_signals = 0
    entry_fills = exit_fills = 0

    for index in range(start_index, end_index + 1):
        frame = frames[index]
        if pending_entry is not None and position is None:
            position = PairPosition(
                signal=pending_entry,
                entry_index=index,
                entry_timestamp_ms=frame.timestamp_ms,
                long_entry_price=frame.bars[pending_entry.long_symbol].open,
                short_entry_price=frame.bars[pending_entry.short_symbol].open,
            )
            pending_entry = None
            entry_fills += 2

        if position is not None:
            exit_reason = ""
            if position.pending_exit_reason:
                exit_reason = position.pending_exit_reason
            elif index - position.entry_index >= int(
                candidate["exit_contract"]["max_holding_bars"]
            ):
                exit_reason = "max_holding"
            if exit_reason:
                pair_trade, legs = _close_pair_trade(
                    position,
                    frame=frame,
                    exit_index=index,
                    exit_reason=exit_reason,
                    real_cost_bps=real_cost_bps,
                    stressed_cost_bps=stressed_cost_bps,
                    notional_per_leg_usd=notional_per_leg,
                )
                pair_trades.append(pair_trade)
                leg_trades.extend(legs)
                position = None
                exit_fills += 2
                last_exit_index = index

        if position is not None and not position.pending_exit_reason:
            pair_mark_bps = _pair_mark_gross_bps(position, frame)
            if pair_mark_bps <= float(
                candidate["exit_contract"]["pair_stop_bps"]
            ):
                if index < end_index:
                    position.pending_exit_reason = "pair_stop"
            elif (
                candidate["exit_contract"]["rank_inversion_exit"] is True
                and _pair_decision_index(candidate, index=index, series=series)
            ):
                snapshot = _relative_momentum_snapshot(
                    candidate,
                    index=index,
                    series=series,
                )
                long_row = snapshot.get(position.signal.long_symbol)
                short_row = snapshot.get(position.signal.short_symbol)
                if (
                    long_row is not None
                    and short_row is not None
                    and long_row["rank_score"] <= short_row["rank_score"]
                    and index < end_index
                ):
                    position.pending_exit_reason = "rank_inversion"

        signal = _market_neutral_pair_signal(
            candidate,
            index=index,
            series=series,
        )
        if signal is not None:
            candidate_signals += 1
        if (
            signal is not None
            and position is None
            and pending_entry is None
            and index < end_index
            and index - last_exit_index
            >= int(candidate["selection_contract"]["cooldown_bars"])
        ):
            pending_entry = signal
            selected_signals += 1

    metrics = _pair_metrics(pair_trades, net_field="net_bps")
    return {
        "split": split_name,
        "start_timestamp_ms": frames[start_index].timestamp_ms,
        "end_timestamp_ms": frames[end_index].timestamp_ms,
        "bars": end_index - start_index + 1,
        "candidate_signals": candidate_signals,
        "selected_signals": selected_signals,
        "entry_fills": entry_fills,
        "exit_fills": exit_fills,
        "filled_orders": entry_fills + exit_fills,
        "closed_trades": len(pair_trades),
        "closed_pairs": len(pair_trades),
        "remaining_open_positions": 2 if position is not None else 0,
        "right_censored_positions": 2 if position is not None else 0,
        "paired_fill_atomicity_violations": 0,
        "real_cost_bps_per_leg": real_cost_bps,
        "total_pair_round_trip_cost_bps": real_cost_bps * 2.0,
        "metrics": metrics,
        "cost_stress_metrics": _pair_metrics(
            pair_trades,
            net_field="stressed_net_bps",
        ),
        "per_symbol": _group_metrics(leg_trades, "symbol"),
        "per_direction": _group_metrics(leg_trades, "direction"),
        "per_exit_reason": _group_metrics(pair_trades, "exit_reason"),
        "trades": pair_trades,
    }


def _market_neutral_pair_signal(
    candidate: Mapping[str, Any],
    *,
    index: int,
    series: Mapping[str, SymbolSeries],
) -> PairSignal | None:
    if not _pair_decision_index(candidate, index=index, series=series):
        return None
    snapshot = _relative_momentum_snapshot(
        candidate,
        index=index,
        series=series,
    )
    if len(snapshot) < 2:
        return None
    long_symbol, long_row = max(
        snapshot.items(),
        key=lambda item: (item[1]["rank_score"], item[0]),
    )
    short_symbol, short_row = min(
        snapshot.items(),
        key=lambda item: (item[1]["rank_score"], item[0]),
    )
    if long_symbol == short_symbol:
        return None
    momentum_spread_bps = (
        long_row["momentum"] - short_row["momentum"]
    ) * 10_000.0
    if momentum_spread_bps < float(
        candidate["event_contract"]["min_cross_sectional_spread_bps"]
    ):
        return None
    return PairSignal(
        long_symbol=long_symbol,
        short_symbol=short_symbol,
        signal_index=index,
        score_spread=long_row["rank_score"] - short_row["rank_score"],
        momentum_spread_bps=momentum_spread_bps,
    )


def _pair_decision_index(
    candidate: Mapping[str, Any],
    *,
    index: int,
    series: Mapping[str, SymbolSeries],
) -> bool:
    first_series = next(iter(series.values()))
    timestamp_ms = int(first_series.timestamps[index])
    close_hour_utc = (
        timestamp_ms // 3_600_000
        + int(candidate["event_contract"]["aggregate_bars"])
    ) % 24
    return close_hour_utc == 0


def _relative_momentum_snapshot(
    candidate: Mapping[str, Any],
    *,
    index: int,
    series: Mapping[str, SymbolSeries],
) -> dict[str, dict[str, float]]:
    event = candidate["event_contract"]
    lookback = int(event["momentum_lookback_bars"])
    skip_recent = int(event["skip_recent_bars"])
    volatility_lookback = int(event["volatility_lookback_bars"])
    if index < max(lookback, volatility_lookback + skip_recent):
        return {}
    rank_method = str(candidate["selection_contract"]["portfolio_rank"])
    rows: dict[str, dict[str, float]] = {}
    for symbol, item in series.items():
        past = float(item.close[index - lookback])
        skipped = float(item.close[index - skip_recent])
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (past, skipped)
        ):
            continue
        momentum = skipped / past - 1.0
        volatility_end = index - skip_recent + 1
        prices = item.close[
            volatility_end - volatility_lookback:volatility_end
        ]
        if prices.size < 3 or np.any(prices <= 0.0):
            continue
        realized_vol = float(np.std(np.diff(np.log(prices))))
        if not math.isfinite(realized_vol) or realized_vol <= 0.0:
            continue
        if rank_method == (
            "signed_skip_one_day_momentum_divided_by_realized_volatility"
        ):
            rank_score = momentum / realized_vol
        elif rank_method == "signed_skip_one_day_momentum":
            rank_score = momentum
        else:
            raise StrategyLabError(f"unsupported pair rank: {rank_method}")
        rows[symbol] = {
            "momentum": momentum,
            "realized_volatility": realized_vol,
            "rank_score": rank_score,
        }
    return rows


def _pair_mark_gross_bps(
    position: PairPosition,
    frame: MarketFrame,
) -> float:
    long_close = frame.bars[position.signal.long_symbol].close
    short_close = frame.bars[position.signal.short_symbol].close
    long_bps = (
        long_close / max(position.long_entry_price, 1e-12) - 1.0
    ) * 10_000.0
    short_bps = (
        position.short_entry_price / max(short_close, 1e-12) - 1.0
    ) * 10_000.0
    return (long_bps + short_bps) / 2.0


def _close_pair_trade(
    position: PairPosition,
    *,
    frame: MarketFrame,
    exit_index: int,
    exit_reason: str,
    real_cost_bps: float,
    stressed_cost_bps: float,
    notional_per_leg_usd: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    long_exit = frame.bars[position.signal.long_symbol].open
    short_exit = frame.bars[position.signal.short_symbol].open
    long_gross_bps = (
        long_exit / max(position.long_entry_price, 1e-12) - 1.0
    ) * 10_000.0
    short_gross_bps = (
        position.short_entry_price / max(short_exit, 1e-12) - 1.0
    ) * 10_000.0
    pair_gross_bps = (long_gross_bps + short_gross_bps) / 2.0
    pair_net_bps = pair_gross_bps - real_cost_bps
    pair_stressed_net_bps = pair_gross_bps - stressed_cost_bps
    pair_notional = notional_per_leg_usd * 2.0
    common = {
        "direction": "PAIR",
        "regime": "market_neutral_relative_momentum",
        "signal_index": position.signal.signal_index,
        "entry_index": position.entry_index,
        "exit_index": exit_index,
        "entry_timestamp_ms": position.entry_timestamp_ms,
        "exit_timestamp_ms": frame.timestamp_ms,
        "holding_bars": exit_index - position.entry_index,
        "exit_reason": exit_reason,
        "funding_bps": 0.0,
    }
    pair_trade = {
        **common,
        "symbol": (
            f"{position.signal.long_symbol}|{position.signal.short_symbol}"
        ),
        "long_symbol": position.signal.long_symbol,
        "short_symbol": position.signal.short_symbol,
        "long_gross_bps": long_gross_bps,
        "short_gross_bps": short_gross_bps,
        "gross_bps": pair_gross_bps,
        "cost_bps": real_cost_bps,
        "stressed_cost_bps": stressed_cost_bps,
        "net_bps": pair_net_bps,
        "stressed_net_bps": pair_stressed_net_bps,
        "net_pnl_usd": pair_notional * pair_net_bps / 10_000.0,
        "stressed_net_pnl_usd": (
            pair_notional * pair_stressed_net_bps / 10_000.0
        ),
    }
    legs = [
        _pair_leg_trade(
            common,
            symbol=position.signal.long_symbol,
            direction="LONG",
            gross_bps=long_gross_bps,
            real_cost_bps=real_cost_bps,
            stressed_cost_bps=stressed_cost_bps,
            notional_usd=notional_per_leg_usd,
        ),
        _pair_leg_trade(
            common,
            symbol=position.signal.short_symbol,
            direction="SHORT",
            gross_bps=short_gross_bps,
            real_cost_bps=real_cost_bps,
            stressed_cost_bps=stressed_cost_bps,
            notional_usd=notional_per_leg_usd,
        ),
    ]
    return pair_trade, legs


def _pair_leg_trade(
    common: Mapping[str, Any],
    *,
    symbol: str,
    direction: str,
    gross_bps: float,
    real_cost_bps: float,
    stressed_cost_bps: float,
    notional_usd: float,
) -> dict[str, Any]:
    net_bps = gross_bps - real_cost_bps
    stressed_net_bps = gross_bps - stressed_cost_bps
    return {
        **common,
        "symbol": symbol,
        "direction": direction,
        "gross_bps": gross_bps,
        "cost_bps": real_cost_bps,
        "stressed_cost_bps": stressed_cost_bps,
        "net_bps": net_bps,
        "stressed_net_bps": stressed_net_bps,
        "net_pnl_usd": notional_usd * net_bps / 10_000.0,
        "stressed_net_pnl_usd": (
            notional_usd * stressed_net_bps / 10_000.0
        ),
    }


def _pair_metrics(
    trades: Sequence[Mapping[str, Any]],
    *,
    net_field: str,
) -> dict[str, Any]:
    result = _metrics(trades, net_field=net_field)
    result["fills"] = len(trades) * 4
    return result


def _simulate_portfolio_split(
    candidate: Mapping[str, Any],
    *,
    split_name: str,
    frames: Sequence[MarketFrame],
    series: Mapping[str, SymbolSeries],
    start_index: int,
    end_index: int,
    costs: Mapping[str, Any],
    portfolio: Mapping[str, Any],
) -> dict[str, Any]:
    real_cost_bps = float(costs["round_trip_fee_bps"]) + float(
        costs["slippage_bps"]
    )
    stressed_cost_bps = max(
        real_cost_bps * float(costs["cost_stress_multiplier"]),
        real_cost_bps + float(costs.get("safety_buffer_bps", 0.0)),
    )
    notional_per_leg = float(portfolio["notional_per_trade_usd"])
    position: PortfolioPosition | None = None
    rebalance_due = False
    pending_target: PortfolioSignal | None = None
    portfolio_trades: list[dict[str, Any]] = []
    leg_trades: list[dict[str, Any]] = []
    candidate_signals = selected_signals = 0
    entry_fills = exit_fills = 0

    for index in range(start_index, end_index + 1):
        frame = frames[index]
        if rebalance_due:
            if position is not None:
                portfolio_trade, legs = _close_portfolio_trade(
                    position,
                    frame=frame,
                    exit_index=index,
                    real_cost_bps=real_cost_bps,
                    stressed_cost_bps=stressed_cost_bps,
                    notional_per_leg_usd=notional_per_leg,
                )
                portfolio_trades.append(portfolio_trade)
                leg_trades.extend(legs)
                position = None
                exit_fills += 4
            if pending_target is not None:
                symbols = (
                    pending_target.long_symbols
                    + pending_target.short_symbols
                )
                position = PortfolioPosition(
                    signal=pending_target,
                    entry_index=index,
                    entry_timestamp_ms=frame.timestamp_ms,
                    entry_prices={
                        symbol: frame.bars[symbol].open for symbol in symbols
                    },
                )
                entry_fills += 4
            pending_target = None
            rebalance_due = False

        if _portfolio_decision_index(
            candidate,
            index=index,
            series=series,
        ):
            signal = _market_neutral_portfolio_signal(
                candidate,
                index=index,
                series=series,
            )
            if signal is not None:
                candidate_signals += 1
            if index < end_index:
                pending_target = signal
                rebalance_due = True
                if signal is not None:
                    selected_signals += 1

    metrics = _portfolio_metrics(portfolio_trades, net_field="net_bps")
    return {
        "split": split_name,
        "start_timestamp_ms": frames[start_index].timestamp_ms,
        "end_timestamp_ms": frames[end_index].timestamp_ms,
        "bars": end_index - start_index + 1,
        "candidate_signals": candidate_signals,
        "selected_signals": selected_signals,
        "entry_fills": entry_fills,
        "exit_fills": exit_fills,
        "filled_orders": entry_fills + exit_fills,
        "closed_trades": len(portfolio_trades),
        "closed_portfolios": len(portfolio_trades),
        "remaining_open_positions": 4 if position is not None else 0,
        "right_censored_positions": 4 if position is not None else 0,
        "portfolio_fill_atomicity_violations": 0,
        "long_notional_usd": notional_per_leg * 2.0,
        "short_notional_usd": notional_per_leg * 2.0,
        "net_notional_usd": 0.0,
        "real_cost_bps_per_leg": real_cost_bps,
        "total_portfolio_round_trip_cost_bps": real_cost_bps * 4.0,
        "metrics": metrics,
        "cost_stress_metrics": _portfolio_metrics(
            portfolio_trades,
            net_field="stressed_net_bps",
        ),
        "per_symbol": _group_metrics(leg_trades, "symbol"),
        "per_direction": _group_metrics(leg_trades, "direction"),
        "per_exit_reason": _group_metrics(portfolio_trades, "exit_reason"),
        "trades": portfolio_trades,
        "leg_trades": leg_trades,
    }


def _market_neutral_portfolio_signal(
    candidate: Mapping[str, Any],
    *,
    index: int,
    series: Mapping[str, SymbolSeries],
) -> PortfolioSignal | None:
    if not _portfolio_decision_index(candidate, index=index, series=series):
        return None
    snapshot = _relative_momentum_snapshot(
        candidate,
        index=index,
        series=series,
    )
    if len(snapshot) < 4:
        return None
    ordered = sorted(
        snapshot,
        key=lambda symbol: (snapshot[symbol]["rank_score"], symbol),
    )
    short_symbols = tuple(ordered[:2])
    long_symbols = tuple(reversed(ordered[-2:]))
    long_momentum = mean(
        snapshot[symbol]["momentum"] for symbol in long_symbols
    )
    short_momentum = mean(
        snapshot[symbol]["momentum"] for symbol in short_symbols
    )
    momentum_dispersion_bps = (
        long_momentum - short_momentum
    ) * 10_000.0
    if momentum_dispersion_bps < float(
        candidate["event_contract"]["min_cross_sectional_dispersion_bps"]
    ):
        return None
    long_score = mean(
        snapshot[symbol]["rank_score"] for symbol in long_symbols
    )
    short_score = mean(
        snapshot[symbol]["rank_score"] for symbol in short_symbols
    )
    return PortfolioSignal(
        long_symbols=long_symbols,
        short_symbols=short_symbols,
        signal_index=index,
        score_dispersion=long_score - short_score,
        momentum_dispersion_bps=momentum_dispersion_bps,
    )


def _portfolio_decision_index(
    candidate: Mapping[str, Any],
    *,
    index: int,
    series: Mapping[str, SymbolSeries],
) -> bool:
    event = candidate["event_contract"]
    first_series = next(iter(series.values()))
    close_timestamp_ms = int(first_series.timestamps[index]) + int(
        event["aggregate_bars"]
    ) * 3_600_000
    close_time = datetime.fromtimestamp(
        close_timestamp_ms / 1000.0,
        tz=timezone.utc,
    )
    return (
        close_time.weekday() == int(event["rebalance_weekday_utc"])
        and close_time.hour == int(event["rebalance_close_hour_utc"])
    )


def _close_portfolio_trade(
    position: PortfolioPosition,
    *,
    frame: MarketFrame,
    exit_index: int,
    real_cost_bps: float,
    stressed_cost_bps: float,
    notional_per_leg_usd: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gross_by_leg: list[tuple[str, str, float]] = []
    for symbol in position.signal.long_symbols:
        gross_by_leg.append(
            (
                symbol,
                "LONG",
                (
                    frame.bars[symbol].open
                    / max(position.entry_prices[symbol], 1e-12)
                    - 1.0
                )
                * 10_000.0,
            )
        )
    for symbol in position.signal.short_symbols:
        gross_by_leg.append(
            (
                symbol,
                "SHORT",
                (
                    position.entry_prices[symbol]
                    / max(frame.bars[symbol].open, 1e-12)
                    - 1.0
                )
                * 10_000.0,
            )
        )
    portfolio_gross_bps = mean(row[2] for row in gross_by_leg)
    portfolio_net_bps = portfolio_gross_bps - real_cost_bps
    stressed_net_bps = portfolio_gross_bps - stressed_cost_bps
    total_notional = notional_per_leg_usd * 4.0
    common = {
        "direction": "PORTFOLIO",
        "regime": "weekly_market_neutral_relative_momentum",
        "signal_index": position.signal.signal_index,
        "entry_index": position.entry_index,
        "exit_index": exit_index,
        "entry_timestamp_ms": position.entry_timestamp_ms,
        "exit_timestamp_ms": frame.timestamp_ms,
        "holding_bars": exit_index - position.entry_index,
        "exit_reason": "scheduled_rebalance",
        "funding_bps": 0.0,
    }
    portfolio_trade = {
        **common,
        "symbol": (
            "L:"
            + ",".join(position.signal.long_symbols)
            + "|S:"
            + ",".join(position.signal.short_symbols)
        ),
        "long_symbols": list(position.signal.long_symbols),
        "short_symbols": list(position.signal.short_symbols),
        "gross_bps": portfolio_gross_bps,
        "cost_bps": real_cost_bps,
        "stressed_cost_bps": stressed_cost_bps,
        "net_bps": portfolio_net_bps,
        "stressed_net_bps": stressed_net_bps,
        "net_pnl_usd": total_notional * portfolio_net_bps / 10_000.0,
        "stressed_net_pnl_usd": (
            total_notional * stressed_net_bps / 10_000.0
        ),
    }
    legs = [
        _pair_leg_trade(
            common,
            symbol=symbol,
            direction=direction,
            gross_bps=gross_bps,
            real_cost_bps=real_cost_bps,
            stressed_cost_bps=stressed_cost_bps,
            notional_usd=notional_per_leg_usd,
        )
        for symbol, direction, gross_bps in gross_by_leg
    ]
    return portfolio_trade, legs


def _portfolio_metrics(
    trades: Sequence[Mapping[str, Any]],
    *,
    net_field: str,
) -> dict[str, Any]:
    result = _metrics(trades, net_field=net_field)
    result["fills"] = len(trades) * 8
    return result


def _summarize_candidate(
    candidate: Mapping[str, Any],
    splits: Sequence[Mapping[str, Any]],
    *,
    evaluation: Mapping[str, Any],
    baseline_splits: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    all_trades = [trade for split in splits for trade in split["trades"]]
    evidence_trades = [
        trade
        for split in splits
        if split["split"] in {"validation", "oos", "sanity"}
        for trade in split["trades"]
    ]
    aggregate = _metrics(all_trades, net_field="net_bps")
    evidence = _metrics(evidence_trades, net_field="net_bps")
    stress = _metrics(evidence_trades, net_field="stressed_net_bps")
    gates = evaluation["gates"]
    failures: list[str] = []

    if len(all_trades) < int(gates["min_total_closed_trades"]):
        failures.append("total_closed_trades_below_50")
    for split in splits:
        if int(split["filled_orders"]) < int(gates["min_window_fills"]):
            failures.append(f"{split['split']}:fills_below_20")
        if int(split["closed_trades"]) < int(gates["min_window_closed_trades"]):
            failures.append(f"{split['split']}:closed_trades_below_10")
        if split["split"] in {"validation", "oos", "sanity"}:
            metrics = split["metrics"]
            if (metrics["mean_net_bps"] or 0.0) <= 0.0:
                failures.append(f"root_expectancy_collapse:{split['split']}")
            if (metrics["lcb_95_net_bps"] or 0.0) <= 0.0:
                failures.append(f"root_lcb_collapse:{split['split']}")
    if (evidence["mean_net_bps"] or 0.0) <= float(
        gates["min_costed_expectancy_bps"]
    ):
        failures.append("nonpositive_costed_expectancy")
    if (evidence["lcb_95_net_bps"] or 0.0) <= float(gates["min_lcb_95_bps"]):
        failures.append("nonpositive_lcb")
    if (stress["mean_net_bps"] or 0.0) <= 0.0:
        failures.append("cost_stress_nonpositive_expectancy")
    if (stress["lcb_95_net_bps"] or 0.0) <= 0.0:
        failures.append("cost_stress_nonpositive_lcb")
    if float(evidence["max_drawdown_usd"]) > float(gates["max_drawdown_usd"]):
        failures.append("max_drawdown_exceeded")

    per_direction = _group_metrics(evidence_trades, "direction")
    for direction in candidate["hypothesis"].get("directions", []):
        metrics = per_direction.get(direction)
        if (
            metrics is None
            or int(metrics["trades"]) < int(gates["min_window_closed_trades"])
            or (metrics["mean_net_bps"] or 0.0) <= 0.0
            or (metrics["lcb_95_net_bps"] or 0.0) <= 0.0
        ):
            failures.append(f"direction_collapse:{direction}")

    per_regime = _group_metrics(evidence_trades, "regime")
    for regime, metrics in per_regime.items():
        if int(metrics["trades"]) >= int(gates["min_window_closed_trades"]) and (
            (metrics["mean_net_bps"] or 0.0) <= 0.0
            or (metrics["lcb_95_net_bps"] or 0.0) <= 0.0
        ):
            failures.append(f"regime_collapse:{regime}")

    per_symbol = _group_metrics(evidence_trades, "symbol")
    promotable_symbols = sorted(
        symbol
        for symbol, metrics in per_symbol.items()
        if int(metrics["trades"]) >= int(gates["min_window_closed_trades"])
        and (metrics["mean_net_bps"] or 0.0) > 0.0
        and (metrics["lcb_95_net_bps"] or 0.0) > 0.0
    )
    if len(promotable_symbols) < int(gates["min_promotable_symbols"]):
        failures.append("promotable_symbols_below_2")

    baseline_comparison: dict[str, Any] = {}
    if baseline_splits is not None:
        baseline_trades = [
            trade
            for split in baseline_splits
            if split["split"] in {"validation", "oos", "sanity"}
            for trade in split["trades"]
        ]
        baseline_metrics = _metrics(baseline_trades, net_field="net_bps")
        beats_baseline = (
            (evidence["mean_net_bps"] or -math.inf)
            > (baseline_metrics["mean_net_bps"] or -math.inf)
            and float(evidence["net_sum_bps"])
            > float(baseline_metrics["net_sum_bps"])
        )
        baseline_comparison = {
            "baseline_id": gates["baseline_id"],
            "candidate_mean_net_bps": evidence["mean_net_bps"],
            "baseline_mean_net_bps": baseline_metrics["mean_net_bps"],
            "candidate_net_sum_bps": evidence["net_sum_bps"],
            "baseline_net_sum_bps": baseline_metrics["net_sum_bps"],
            "beats_baseline": beats_baseline,
        }
        if not beats_baseline:
            failures.append("does_not_beat_baseline")

    unique_failures = sorted(set(failures))
    passed = not unique_failures
    return {
        "candidate_id": candidate["candidate_id"],
        "profile_sha256": candidate.get("profile_sha256", ""),
        "verdict": (
            "accepted_for_runtime_implementation"
            if passed
            else "terminal_rejected_historical_oos"
        ),
        "passed": passed,
        "failures": unique_failures,
        "aggregate": aggregate,
        "evidence_aggregate": evidence,
        "cost_stress_aggregate": stress,
        "per_symbol": per_symbol,
        "per_direction": per_direction,
        "per_regime": per_regime,
        "promotable_symbols": promotable_symbols,
        "baseline_comparison": baseline_comparison,
        "splits": list(splits),
        "runtime_actor_created": False,
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }


def _pullback_signals(
    candidate: Mapping[str, Any],
    *,
    index: int,
    series: Mapping[str, SymbolSeries],
) -> list[EntrySignal]:
    event = candidate["event_contract"]
    slow_slope_bars = int(event["slow_slope_bars"])
    pullback_bars = int(event["pullback_window_bars"])
    required = max(
        int(event["ema_slow_bars"]) + slow_slope_bars,
        int(event["volume_median_bars"]),
        int(event["atr_bars"]),
        pullback_bars,
    )
    if index < required:
        return []
    signals: list[EntrySignal] = []
    for symbol, item in series.items():
        atr = float(item.atr14[index - 1])
        fast = float(item.ema24[index])
        slow = float(item.ema96[index])
        previous_fast = float(item.ema24[index - 1])
        volume_median = float(item.volume_median24[index - 1])
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (atr, fast, slow, volume_median)
        ):
            continue
        volume_ratio = float(item.volume[index]) / volume_median
        if volume_ratio < float(event["min_volume_ratio"]):
            continue
        window = slice(index - pullback_bars, index)
        touch_atr = float(event["pullback_touch_atr"])
        slow_slope = slow - float(item.ema96[index - slow_slope_bars])
        close = float(item.close[index])
        previous_close = float(item.close[index - 1])
        if (
            fast > slow
            and slow_slope > 0.0
            and previous_close <= previous_fast
            and close > fast
            and float(np.min(np.abs(item.low[window] - item.ema24[window])))
            <= touch_atr * atr
        ):
            signals.append(
                EntrySignal(
                    symbol=symbol,
                    direction="LONG",
                    signal_index=index,
                    score=max(0.0, (close - fast) / atr) * volume_ratio,
                    atr=atr,
                    regime="bullish",
                )
            )
        elif (
            fast < slow
            and slow_slope < 0.0
            and previous_close >= previous_fast
            and close < fast
            and float(np.min(np.abs(item.high[window] - item.ema24[window])))
            <= touch_atr * atr
        ):
            signals.append(
                EntrySignal(
                    symbol=symbol,
                    direction="SHORT",
                    signal_index=index,
                    score=max(0.0, (fast - close) / atr) * volume_ratio,
                    atr=atr,
                    regime="bearish",
                )
            )
    return signals


def _compression_signals(
    candidate: Mapping[str, Any],
    *,
    index: int,
    series: Mapping[str, SymbolSeries],
) -> list[EntrySignal]:
    event = candidate["event_contract"]
    range_bars = int(event["range_bars"])
    percentile_bars = int(event["compression_percentile_lookback_bars"])
    if index < max(percentile_bars + 1, range_bars + 1):
        return []
    signals: list[EntrySignal] = []
    for symbol, item in series.items():
        previous_width = float(item.bollinger_width24[index - 1])
        history = item.bollinger_width24[
            index - percentile_bars - 1:index - 1
        ]
        history = history[np.isfinite(history)]
        atr = float(item.atr14[index - 1])
        volume_median = float(item.volume_median24[index - 1])
        if (
            not math.isfinite(previous_width)
            or history.size < percentile_bars // 2
            or not math.isfinite(atr)
            or atr <= 0.0
            or not math.isfinite(volume_median)
            or volume_median <= 0.0
        ):
            continue
        if float(np.mean(history <= previous_width)) > float(
            event["max_compression_percentile"]
        ):
            continue
        volume_ratio = float(item.volume[index]) / volume_median
        if volume_ratio < float(event["min_volume_ratio"]):
            continue
        if float(item.true_range[index]) <= atr:
            continue
        prior_high = float(np.max(item.high[index - range_bars:index]))
        prior_low = float(np.min(item.low[index - range_bars:index]))
        close = float(item.close[index])
        if close > prior_high:
            signals.append(
                EntrySignal(
                    symbol=symbol,
                    direction="LONG",
                    signal_index=index,
                    score=((close - prior_high) / atr) * volume_ratio,
                    atr=atr,
                    regime="compression_transition",
                    range_high=prior_high,
                    range_low=prior_low,
                )
            )
        elif close < prior_low:
            signals.append(
                EntrySignal(
                    symbol=symbol,
                    direction="SHORT",
                    signal_index=index,
                    score=((prior_low - close) / atr) * volume_ratio,
                    atr=atr,
                    regime="compression_transition",
                    range_high=prior_high,
                    range_low=prior_low,
                )
            )
    return signals


def _cross_sectional_trend_signals(
    candidate: Mapping[str, Any],
    *,
    index: int,
    series: Mapping[str, SymbolSeries],
) -> list[EntrySignal]:
    event = candidate["event_contract"]
    lookback = int(event["momentum_lookback_bars"])
    skip_recent = int(event["skip_recent_bars"])
    volatility_lookback = int(event["volatility_lookback_bars"])
    required = max(lookback, volatility_lookback + 1, 96)
    if index < required:
        return []
    first_series = next(iter(series.values()))
    timestamp_ms = int(first_series.timestamps[index])
    close_hour_utc = (
        timestamp_ms // 3_600_000 + int(event["aggregate_bars"])
    ) % 24
    if close_hour_utc != 0:
        return []

    long_rows: list[EntrySignal] = []
    short_rows: list[EntrySignal] = []
    for symbol, item in series.items():
        close = float(item.close[index])
        fast = float(item.ema24[index])
        slow = float(item.ema96[index])
        atr = float(item.atr14[index - 1])
        past = float(item.close[index - lookback])
        skipped = float(item.close[index - skip_recent])
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (close, fast, slow, atr, past, skipped)
        ):
            continue
        momentum = skipped / past - 1.0
        momentum_bps = momentum * 10_000.0
        if abs(momentum_bps) < float(event["min_momentum_bps"]):
            continue
        volatility_end = index - skip_recent + 1
        prices = item.close[
            volatility_end - volatility_lookback:volatility_end
        ]
        if prices.size < 3 or np.any(prices <= 0.0):
            continue
        returns = np.diff(np.log(prices))
        realized_vol = float(np.std(returns))
        if not math.isfinite(realized_vol) or realized_vol <= 0.0:
            continue
        signal = EntrySignal(
            symbol=symbol,
            direction="LONG" if momentum > 0.0 else "SHORT",
            signal_index=index,
            score=abs(momentum) / realized_vol,
            atr=atr,
            regime="cross_sectional_trend",
        )
        if momentum > 0.0 and close > fast > slow:
            long_rows.append(signal)
        elif momentum < 0.0 and close < fast < slow:
            short_rows.append(signal)
    selected: list[EntrySignal] = []
    if long_rows:
        selected.append(max(long_rows, key=lambda row: (row.score, row.symbol)))
    if short_rows:
        selected.append(max(short_rows, key=lambda row: (row.score, row.symbol)))
    return selected


def _funding_carry_signals(
    candidate: Mapping[str, Any],
    *,
    index: int,
    series: Mapping[str, SymbolSeries],
    funding_by_symbol: Mapping[str, Mapping[int, float]] | None,
) -> list[EntrySignal]:
    if funding_by_symbol is None:
        return []
    signals: list[EntrySignal] = []
    for symbol, item in series.items():
        signal, _, _ = _funding_carry_signal_decision(
            candidate,
            symbol=symbol,
            item=item,
            index=index,
            funding_by_symbol=funding_by_symbol,
        )
        if signal is not None:
            signals.append(signal)
    return signals


def _funding_carry_signal_decision(
    candidate: Mapping[str, Any],
    *,
    symbol: str,
    item: SymbolSeries,
    index: int,
    funding_by_symbol: Mapping[str, Mapping[int, float]],
) -> tuple[EntrySignal | None, str, dict[str, float]]:
    event = candidate["event_contract"]
    history_windows = int(event["funding_history_windows"])
    trend_bars = int(event["trend_guard_bars"])
    diagnostics = {
        "last_abs_bps": 0.0,
        "median_abs_bps": 0.0,
        "projected_bps": 0.0,
        "adverse_trend_atr": 0.0,
    }
    if index < max(int(event["atr_bars"]), trend_bars):
        return None, "indicator_warmup", diagnostics
    timestamp_ms = int(item.timestamps[index])
    symbol_funding = funding_by_symbol.get(symbol, {})
    if timestamp_ms not in symbol_funding:
        return None, "not_settlement", diagnostics
    rates = _settled_rates_through(
        funding_by_symbol,
        symbol=symbol,
        timestamp_ms=timestamp_ms,
    )
    if len(rates) < history_windows:
        return None, "funding_history_short", diagnostics
    trailing = [rate for _, rate in rates[-history_windows:]]
    last_rate = float(trailing[-1])
    if last_rate == 0.0 or any(
        rate == 0.0
        or math.copysign(1.0, rate) != math.copysign(1.0, last_rate)
        for rate in trailing
    ):
        return None, "sign_not_persistent", diagnostics
    last_bps = abs(last_rate) * 10_000.0
    median_rate = float(median(trailing))
    median_bps = abs(median_rate) * 10_000.0
    projected_bps = median_bps * int(event["projected_windows"])
    diagnostics.update(
        {
            "last_abs_bps": last_bps,
            "median_abs_bps": median_bps,
            "projected_bps": projected_bps,
        }
    )
    if last_bps < float(event["min_last_settled_funding_bps"]):
        return None, "last_below_threshold", diagnostics
    if median_bps < float(event["min_trailing_median_funding_bps"]):
        return None, "median_below_threshold", diagnostics
    if projected_bps < float(event["min_projected_carry_bps"]):
        return None, "projected_below_threshold", diagnostics
    atr = float(item.atr14[index - 1])
    if not math.isfinite(atr) or atr <= 0.0:
        return None, "atr_unavailable", diagnostics
    close = float(item.close[index])
    previous = float(item.close[index - trend_bars])
    direction = (
        str(event["positive_funding_direction"])
        if last_rate > 0.0
        else str(event["negative_funding_direction"])
    )
    adverse_trend_atr = (
        (close - previous) / atr
        if direction == "SHORT"
        else (previous - close) / atr
    )
    diagnostics["adverse_trend_atr"] = adverse_trend_atr
    if adverse_trend_atr > float(event["max_adverse_trend_atr"]):
        return None, "adverse_trend", diagnostics
    return (
        EntrySignal(
            symbol=symbol,
            direction=direction,
            signal_index=index,
            score=projected_bps,
            atr=atr,
            regime=(
                "positive_funding"
                if last_rate > 0.0
                else "negative_funding"
            ),
        ),
        "signal",
        diagnostics,
    )


def _funding_activation_funnel(
    candidate: Mapping[str, Any],
    *,
    frames: Sequence[MarketFrame],
    series: Mapping[str, SymbolSeries],
    start_index: int,
    end_index: int,
    funding_by_symbol: Mapping[str, Mapping[int, float]],
) -> dict[str, Any]:
    reasons: dict[str, int] = {}
    per_symbol: dict[str, dict[str, int]] = {}
    maxima = {
        "last_abs_bps": 0.0,
        "median_abs_bps": 0.0,
        "projected_bps": 0.0,
    }
    settlement_observations = 0
    for index in range(start_index, end_index + 1):
        timestamp_ms = frames[index].timestamp_ms
        for symbol, item in series.items():
            if timestamp_ms not in funding_by_symbol.get(symbol, {}):
                continue
            settlement_observations += 1
            _, reason, diagnostics = _funding_carry_signal_decision(
                candidate,
                symbol=symbol,
                item=item,
                index=index,
                funding_by_symbol=funding_by_symbol,
            )
            reasons[reason] = reasons.get(reason, 0) + 1
            symbol_reasons = per_symbol.setdefault(symbol, {})
            symbol_reasons[reason] = symbol_reasons.get(reason, 0) + 1
            for key in maxima:
                maxima[key] = max(maxima[key], float(diagnostics[key]))
    history_ready = settlement_observations - reasons.get(
        "indicator_warmup", 0
    ) - reasons.get("funding_history_short", 0)
    sign_persistent = history_ready - reasons.get("sign_not_persistent", 0)
    last_rate_pass = sign_persistent - reasons.get("last_below_threshold", 0)
    median_rate_pass = last_rate_pass - reasons.get(
        "median_below_threshold", 0
    )
    projected_pass = median_rate_pass - reasons.get(
        "projected_below_threshold", 0
    )
    trend_guard_pass = (
        projected_pass
        - reasons.get("atr_unavailable", 0)
        - reasons.get("adverse_trend", 0)
    )
    return {
        "settlement_observations": settlement_observations,
        "stage_pass_counts": {
            "history_ready": history_ready,
            "sign_persistent": sign_persistent,
            "last_rate_pass": last_rate_pass,
            "median_rate_pass": median_rate_pass,
            "projected_carry_pass": projected_pass,
            "trend_guard_pass": trend_guard_pass,
            "signals": reasons.get("signal", 0),
        },
        "first_rejection_reasons": dict(sorted(reasons.items())),
        "per_symbol_first_rejection_reasons": {
            symbol: dict(sorted(values.items()))
            for symbol, values in sorted(per_symbol.items())
        },
        "maxima": maxima,
    }


def _donchian_signals(
    candidate: Mapping[str, Any],
    *,
    index: int,
    series: Mapping[str, SymbolSeries],
) -> list[EntrySignal]:
    range_bars = int(candidate["event_contract"]["range_bars"])
    if index < max(range_bars, 14):
        return []
    signals: list[EntrySignal] = []
    for symbol, item in series.items():
        atr = float(item.atr14[index - 1])
        if not math.isfinite(atr) or atr <= 0.0:
            continue
        prior_high = float(np.max(item.high[index - range_bars:index]))
        prior_low = float(np.min(item.low[index - range_bars:index]))
        close = float(item.close[index])
        if close > prior_high:
            signals.append(
                EntrySignal(
                    symbol=symbol,
                    direction="LONG",
                    signal_index=index,
                    score=(close - prior_high) / atr,
                    atr=atr,
                    regime="donchian_breakout",
                )
            )
        elif close < prior_low:
            signals.append(
                EntrySignal(
                    symbol=symbol,
                    direction="SHORT",
                    signal_index=index,
                    score=(prior_low - close) / atr,
                    atr=atr,
                    regime="donchian_breakout",
                )
            )
    return signals


def _open_position(
    candidate: Mapping[str, Any],
    signal: EntrySignal,
    bar: MarketBar,
    *,
    index: int,
) -> OpenPosition:
    stop_distance = float(candidate["exit_contract"]["stop_atr"]) * signal.atr
    target_atr = candidate["exit_contract"].get("target_atr")
    target_distance = (
        float(target_atr) * signal.atr if target_atr is not None else math.inf
    )
    if signal.direction == "LONG":
        stop, target = bar.open - stop_distance, bar.open + target_distance
    else:
        stop = bar.open + stop_distance
        target = (
            bar.open - target_distance
            if math.isfinite(target_distance)
            else -math.inf
        )
    return OpenPosition(
        signal=signal,
        entry_index=index,
        entry_timestamp_ms=bar.timestamp_ms,
        entry_price=bar.open,
        stop_price=stop,
        target_price=target,
    )


def protective_exit(
    position: OpenPosition,
    bar: MarketBar,
) -> tuple[float | None, str]:
    if position.signal.direction == "LONG":
        if bar.open <= position.stop_price:
            return bar.open, "stop_gap"
        if bar.low <= position.stop_price:
            return position.stop_price, "stop_loss"
        if bar.open >= position.target_price:
            return bar.open, "target_gap"
        if bar.high >= position.target_price:
            return position.target_price, "take_profit"
    else:
        if bar.open >= position.stop_price:
            return bar.open, "stop_gap"
        if bar.high >= position.stop_price:
            return position.stop_price, "stop_loss"
        if bar.open <= position.target_price:
            return bar.open, "target_gap"
        if bar.low <= position.target_price:
            return position.target_price, "take_profit"
    return None, ""


def _close_signal_reason(
    candidate: Mapping[str, Any],
    position: OpenPosition,
    *,
    index: int,
    series: Mapping[str, SymbolSeries],
    funding_by_symbol: Mapping[str, Mapping[int, float]] | None = None,
) -> str:
    item = series[position.signal.symbol]
    candidate_id = str(candidate["candidate_id"])
    if candidate_id == "regime_pullback_hourly_v1":
        fast, slow = float(item.ema24[index]), float(item.ema96[index])
        if position.signal.direction == "LONG" and fast <= slow:
            return "trend_invalidation"
        if position.signal.direction == "SHORT" and fast >= slow:
            return "trend_invalidation"
    elif candidate_id == "ohlcv_compression_transition_hourly_v1":
        close = float(item.close[index])
        if (
            position.signal.direction == "LONG"
            and position.signal.range_high is not None
            and close <= position.signal.range_high
        ):
            return "failed_breakout"
        if (
            position.signal.direction == "SHORT"
            and position.signal.range_low is not None
            and close >= position.signal.range_low
        ):
            return "failed_breakout"
    elif candidate_id == "funding_carry_hourly_v1":
        timestamp_ms = int(item.timestamps[index])
        funding_rate = _funding_rate_at(
            funding_by_symbol,
            symbol=position.signal.symbol,
            timestamp_ms=timestamp_ms,
        )
        if funding_rate is None:
            return ""
        exit_contract = candidate["exit_contract"]
        if abs(funding_rate) * 10_000.0 < float(
            exit_contract["exit_below_abs_funding_bps"]
        ):
            return "funding_below_floor"
        if (
            position.signal.direction == "SHORT" and funding_rate <= 0.0
        ) or (
            position.signal.direction == "LONG" and funding_rate >= 0.0
        ):
            return "funding_sign_flip"
    elif candidate_id == "cross_sectional_trend_4h_v1":
        close = float(item.close[index])
        fast = float(item.ema24[index])
        slow = float(item.ema96[index])
        if (
            position.signal.direction == "LONG"
            and (close <= fast or fast <= slow)
        ):
            return "trend_invalidation"
        if (
            position.signal.direction == "SHORT"
            and (close >= fast or fast >= slow)
        ):
            return "trend_invalidation"
    return ""


def _close_trade(
    position: OpenPosition,
    *,
    exit_index: int,
    exit_timestamp_ms: int,
    exit_price: float,
    exit_reason: str,
    real_cost_bps: float,
    stressed_cost_bps: float,
    notional_usd: float,
) -> dict[str, Any]:
    if position.signal.direction == "LONG":
        price_gross_bps = (
            exit_price / max(position.entry_price, 1e-12) - 1.0
        ) * 10_000.0
    else:
        price_gross_bps = (
            position.entry_price / max(exit_price, 1e-12) - 1.0
        ) * 10_000.0
    gross_bps = price_gross_bps + position.funding_bps
    net_bps = gross_bps - real_cost_bps
    stressed_net_bps = gross_bps - stressed_cost_bps
    return {
        "symbol": position.signal.symbol,
        "direction": position.signal.direction,
        "regime": position.signal.regime,
        "signal_index": position.signal.signal_index,
        "entry_index": position.entry_index,
        "exit_index": exit_index,
        "entry_timestamp_ms": position.entry_timestamp_ms,
        "exit_timestamp_ms": exit_timestamp_ms,
        "entry_price": position.entry_price,
        "exit_price": exit_price,
        "holding_bars": exit_index - position.entry_index,
        "exit_reason": exit_reason,
        "price_gross_bps": price_gross_bps,
        "funding_bps": position.funding_bps,
        "gross_bps": gross_bps,
        "cost_bps": real_cost_bps,
        "stressed_cost_bps": stressed_cost_bps,
        "net_bps": net_bps,
        "stressed_net_bps": stressed_net_bps,
        "net_pnl_usd": notional_usd * net_bps / 10_000.0,
        "stressed_net_pnl_usd": notional_usd * stressed_net_bps / 10_000.0,
    }


def _metrics(
    trades: Sequence[Mapping[str, Any]],
    *,
    net_field: str,
) -> dict[str, Any]:
    values = [float(item[net_field]) for item in trades]
    gross = [float(item["gross_bps"]) for item in trades]
    pnl_field = (
        "stressed_net_pnl_usd"
        if net_field == "stressed_net_bps"
        else "net_pnl_usd"
    )
    pnl = [float(item[pnl_field]) for item in trades]
    lcb = (
        mean(values) - 1.96 * stdev(values) / math.sqrt(len(values))
        if len(values) >= 2
        else None
    )
    equity = peak = max_drawdown = 0.0
    for value in pnl:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    return {
        "trades": len(values),
        "fills": len(values) * 2,
        "positive_trades": sum(value > 0.0 for value in values),
        "positive_rate": (
            sum(value > 0.0 for value in values) / len(values) if values else 0.0
        ),
        "mean_gross_bps": mean(gross) if gross else None,
        "mean_net_bps": mean(values) if values else None,
        "lcb_95_net_bps": lcb,
        "gross_sum_bps": sum(gross),
        "net_sum_bps": sum(values),
        "net_pnl_usd": sum(pnl),
        "max_drawdown_usd": max_drawdown,
    }


def _group_metrics(
    trades: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for trade in trades:
        groups.setdefault(str(trade[key]), []).append(trade)
    return {
        label: _metrics(rows, net_field="net_bps")
        for label, rows in sorted(groups.items())
    }


def _funding_rate_at(
    funding_by_symbol: Mapping[str, Mapping[int, float]] | None,
    *,
    symbol: str,
    timestamp_ms: int,
) -> float | None:
    if funding_by_symbol is None:
        return None
    value = funding_by_symbol.get(symbol, {}).get(int(timestamp_ms))
    return float(value) if value is not None else None


def _settled_rates_through(
    funding_by_symbol: Mapping[str, Mapping[int, float]],
    *,
    symbol: str,
    timestamp_ms: int,
) -> list[tuple[int, float]]:
    return [
        (int(timestamp), float(rate))
        for timestamp, rate in sorted(funding_by_symbol.get(symbol, {}).items())
        if int(timestamp) <= int(timestamp_ms)
    ]


def _build_symbol_series(
    symbol: str,
    frames: Sequence[MarketFrame],
) -> SymbolSeries:
    timestamps = np.array([item.timestamp_ms for item in frames], dtype=np.int64)
    opens = np.array([item.bars[symbol].open for item in frames], dtype=float)
    highs = np.array([item.bars[symbol].high for item in frames], dtype=float)
    lows = np.array([item.bars[symbol].low for item in frames], dtype=float)
    closes = np.array([item.bars[symbol].close for item in frames], dtype=float)
    volumes = np.array([item.bars[symbol].volume for item in frames], dtype=float)
    previous_close = np.concatenate(([closes[0]], closes[:-1]))
    true_range = np.maximum(
        highs - lows,
        np.maximum(np.abs(highs - previous_close), np.abs(lows - previous_close)),
    )
    return SymbolSeries(
        symbol=symbol,
        timestamps=timestamps,
        open=opens,
        high=highs,
        low=lows,
        close=closes,
        volume=volumes,
        true_range=true_range,
        atr14=_rolling_mean(true_range, 14),
        ema24=_ema(closes, 24),
        ema96=_ema(closes, 96),
        volume_median24=_rolling_median(volumes, 24),
        bollinger_width24=_bollinger_width(closes, 24, 2.0),
    )


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    alpha = 2.0 / (period + 1.0)
    result[0] = float(values[0])
    for index in range(1, len(values)):
        result[index] = (
            alpha * float(values[index])
            + (1.0 - alpha) * result[index - 1]
        )
    return result


def _rolling_mean(values: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=float)
    if len(values) < period:
        return result
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    result[period - 1:] = (
        cumulative[period:] - cumulative[:-period]
    ) / period
    return result


def _rolling_median(values: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=float)
    for index in range(period - 1, len(values)):
        result[index] = median(values[index - period + 1:index + 1])
    return result


def _bollinger_width(
    values: np.ndarray,
    period: int,
    standard_deviations: float,
) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=float)
    for index in range(period - 1, len(values)):
        window = values[index - period + 1:index + 1]
        middle = float(np.mean(window))
        if middle > 0.0:
            result[index] = (
                2.0 * standard_deviations * float(np.std(window)) / middle
            )
    return result


def _baseline_candidate() -> dict[str, Any]:
    return {
        "candidate_id": "donchian_24_atr_exit_v1",
        "profile_sha256": "",
        "hypothesis": {"directions": ["LONG", "SHORT"]},
        "event_contract": {"range_bars": 24},
        "exit_contract": {
            "stop_atr": 1.5,
            "target_atr": 3.0,
            "max_holding_bars": 36,
        },
        "selection_contract": {"cooldown_bars": 24},
    }


def _cross_sectional_baseline_candidate() -> dict[str, Any]:
    return {
        "candidate_id": "donchian_42_4h_atr_exit_v1",
        "profile_sha256": "",
        "hypothesis": {"directions": ["LONG", "SHORT"]},
        "event_contract": {"range_bars": 42},
        "exit_contract": {
            "stop_atr": 3.0,
            "max_holding_bars": 42,
        },
        "selection_contract": {"cooldown_bars": 42},
    }


def _paired_baseline_candidate(
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    selection = dict(candidate["selection_contract"])
    selection["portfolio_rank"] = "signed_skip_one_day_momentum"
    return {
        "candidate_id": "paired_raw_momentum_30d_v1",
        "profile_sha256": "",
        "hypothesis": {
            "directions": ["LONG", "SHORT"],
            "portfolio_unit": "synchronized_pair",
        },
        "event_contract": dict(candidate["event_contract"]),
        "exit_contract": dict(candidate["exit_contract"]),
        "selection_contract": selection,
    }


def _portfolio_baseline_candidate(
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    selection = dict(candidate["selection_contract"])
    selection["portfolio_rank"] = "signed_skip_one_day_momentum"
    return {
        "candidate_id": "weekly_top2_bottom2_raw_momentum_v1",
        "profile_sha256": "",
        "hypothesis": {
            "directions": ["LONG", "SHORT"],
            "portfolio_unit": "four_leg_dollar_neutral_basket",
        },
        "event_contract": dict(candidate["event_contract"]),
        "exit_contract": dict(candidate["exit_contract"]),
        "selection_contract": selection,
    }


def _development_verdict(
    candidate: Mapping[str, Any],
    *,
    split: Mapping[str, Any],
    baseline_split: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = split["metrics"]
    stress = split["cost_stress_metrics"]
    baseline = baseline_split["metrics"]
    failures: list[str] = []
    if int(split["filled_orders"]) < int(gates["min_filled_orders"]):
        failures.append("filled_orders_below_minimum")
    if int(split["closed_trades"]) < int(gates["min_closed_trades"]):
        failures.append("closed_trades_below_minimum")
    if (metrics["mean_net_bps"] or -math.inf) <= float(
        gates["min_mean_net_bps"]
    ):
        failures.append("nonpositive_costed_expectancy")
    if (metrics["lcb_95_net_bps"] or -math.inf) <= float(
        gates["min_lcb_95_net_bps"]
    ):
        failures.append("nonpositive_lcb")
    if float(metrics["max_drawdown_usd"]) > float(
        gates["max_drawdown_usd"]
    ):
        failures.append("max_drawdown_exceeded")
    if (stress["mean_net_bps"] or -math.inf) <= 0.0:
        failures.append("cost_stress_nonpositive_expectancy")
    if (stress["lcb_95_net_bps"] or -math.inf) <= 0.0:
        failures.append("cost_stress_nonpositive_lcb")

    for direction in candidate["hypothesis"]["directions"]:
        row = split["per_direction"].get(direction)
        if (
            row is None
            or int(row["trades"]) < int(gates["min_direction_trades"])
            or (row["mean_net_bps"] or -math.inf) <= 0.0
            or (row["lcb_95_net_bps"] or -math.inf) <= 0.0
        ):
            failures.append(f"direction_collapse:{direction}")

    beats_baseline = (
        (metrics["mean_net_bps"] or -math.inf)
        > (baseline["mean_net_bps"] or -math.inf)
        and float(metrics["net_sum_bps"]) > float(baseline["net_sum_bps"])
    )
    if not beats_baseline:
        failures.append("does_not_beat_baseline")
    unique_failures = sorted(set(failures))
    passed = not unique_failures
    return {
        "verdict": (
            "accepted_for_sealed_validation"
            if passed
            else "terminal_rejected_development"
        ),
        "passed": passed,
        "failures": unique_failures,
        "baseline_comparison": {
            "baseline_id": gates["baseline_id"],
            "candidate_mean_net_bps": metrics["mean_net_bps"],
            "baseline_mean_net_bps": baseline["mean_net_bps"],
            "candidate_net_sum_bps": metrics["net_sum_bps"],
            "baseline_net_sum_bps": baseline["net_sum_bps"],
            "beats_baseline": beats_baseline,
        },
        "continuation_allowed": passed,
    }


def _paired_development_verdict(
    *,
    split: Mapping[str, Any],
    baseline_split: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = split["metrics"]
    stress = split["cost_stress_metrics"]
    baseline = baseline_split["metrics"]
    failures: list[str] = []
    if int(split["filled_orders"]) < int(gates["min_filled_orders"]):
        failures.append("filled_orders_below_minimum")
    if int(split["closed_pairs"]) < int(gates["min_closed_trades"]):
        failures.append("closed_pairs_below_minimum")
    if int(split["paired_fill_atomicity_violations"]) != 0:
        failures.append("paired_fill_atomicity_violation")
    if (metrics["mean_net_bps"] or -math.inf) <= float(
        gates["min_mean_net_bps"]
    ):
        failures.append("nonpositive_costed_expectancy")
    if (metrics["lcb_95_net_bps"] or -math.inf) <= float(
        gates["min_lcb_95_net_bps"]
    ):
        failures.append("nonpositive_lcb")
    if float(metrics["max_drawdown_usd"]) > float(
        gates["max_drawdown_usd"]
    ):
        failures.append("max_drawdown_exceeded")
    if (stress["mean_net_bps"] or -math.inf) <= 0.0:
        failures.append("cost_stress_nonpositive_expectancy")
    if (stress["lcb_95_net_bps"] or -math.inf) <= 0.0:
        failures.append("cost_stress_nonpositive_lcb")

    beats_baseline = (
        (metrics["mean_net_bps"] or -math.inf)
        > (baseline["mean_net_bps"] or -math.inf)
        and float(metrics["net_sum_bps"]) > float(baseline["net_sum_bps"])
    )
    if not beats_baseline:
        failures.append("does_not_beat_baseline")
    unique_failures = sorted(set(failures))
    passed = not unique_failures
    return {
        "verdict": (
            "accepted_for_sealed_validation"
            if passed
            else "terminal_rejected_development"
        ),
        "passed": passed,
        "failures": unique_failures,
        "baseline_comparison": {
            "baseline_id": gates["baseline_id"],
            "candidate_mean_net_bps": metrics["mean_net_bps"],
            "baseline_mean_net_bps": baseline["mean_net_bps"],
            "candidate_net_sum_bps": metrics["net_sum_bps"],
            "baseline_net_sum_bps": baseline["net_sum_bps"],
            "beats_baseline": beats_baseline,
        },
        "continuation_allowed": passed,
    }


def _portfolio_development_verdict(
    *,
    split: Mapping[str, Any],
    baseline_split: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = split["metrics"]
    stress = split["cost_stress_metrics"]
    baseline = baseline_split["metrics"]
    failures: list[str] = []
    if int(split["filled_orders"]) < int(gates["min_filled_orders"]):
        failures.append("filled_orders_below_minimum")
    if int(split["closed_portfolios"]) < int(gates["min_closed_trades"]):
        failures.append("closed_portfolios_below_minimum")
    if int(split["portfolio_fill_atomicity_violations"]) != 0:
        failures.append("portfolio_fill_atomicity_violation")
    if float(split["net_notional_usd"]) != 0.0:
        failures.append("portfolio_not_dollar_neutral")
    if (metrics["mean_net_bps"] or -math.inf) <= float(
        gates["min_mean_net_bps"]
    ):
        failures.append("nonpositive_costed_expectancy")
    if (metrics["lcb_95_net_bps"] or -math.inf) <= float(
        gates["min_lcb_95_net_bps"]
    ):
        failures.append("nonpositive_lcb")
    if float(metrics["max_drawdown_usd"]) > float(
        gates["max_drawdown_usd"]
    ):
        failures.append("max_drawdown_exceeded")
    if (stress["mean_net_bps"] or -math.inf) <= 0.0:
        failures.append("cost_stress_nonpositive_expectancy")
    if (stress["lcb_95_net_bps"] or -math.inf) <= 0.0:
        failures.append("cost_stress_nonpositive_lcb")

    beats_baseline = (
        (metrics["mean_net_bps"] or -math.inf)
        > (baseline["mean_net_bps"] or -math.inf)
        and float(metrics["net_sum_bps"]) > float(baseline["net_sum_bps"])
    )
    if not beats_baseline:
        failures.append("does_not_beat_baseline")
    unique_failures = sorted(set(failures))
    passed = not unique_failures
    return {
        "verdict": (
            "accepted_for_sealed_validation"
            if passed
            else "terminal_rejected_development"
        ),
        "passed": passed,
        "failures": unique_failures,
        "baseline_comparison": {
            "baseline_id": gates["baseline_id"],
            "candidate_mean_net_bps": metrics["mean_net_bps"],
            "baseline_mean_net_bps": baseline["mean_net_bps"],
            "candidate_net_sum_bps": metrics["net_sum_bps"],
            "baseline_net_sum_bps": baseline["net_sum_bps"],
            "beats_baseline": beats_baseline,
        },
        "continuation_allowed": passed,
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Strategy Lab V1 historical evaluation",
        "",
        "## Safety",
        "",
        "- Orders enabled: `false`",
        "- Promotion authority: `false`",
        "- Runtime actor created: `false`",
        "",
        "## Candidates",
        "",
        "| Candidate | Verdict | Trades | Evidence mean bps | LCB bps | "
        "Stress mean bps | Failures |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in report["candidates"]:
        evidence = row["evidence_aggregate"]
        stress = row["cost_stress_aggregate"]
        lines.append(
            "| {candidate} | {verdict} | {trades} | {mean} | {lcb} | "
            "{stress} | {failures} |".format(
                candidate=row["candidate_id"],
                verdict=row["verdict"],
                trades=row["aggregate"]["trades"],
                mean=_fmt(evidence["mean_net_bps"]),
                lcb=_fmt(evidence["lcb_95_net_bps"]),
                stress=_fmt(stress["mean_net_bps"]),
                failures=", ".join(row["failures"]) or "none",
            )
        )
    lines.extend(["", "## Split metrics", ""])
    for row in report["candidates"]:
        lines.extend(
            [
                f"### {row['candidate_id']}",
                "",
                "| Split | Signals | Fills | Closed | Mean bps | LCB bps | DD USD |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for split in row["splits"]:
            metrics = split["metrics"]
            lines.append(
                "| {split} | {signals} | {fills} | {closed} | {mean} | "
                "{lcb} | {dd} |".format(
                    split=split["split"],
                    signals=split["candidate_signals"],
                    fills=split["filled_orders"],
                    closed=split["closed_trades"],
                    mean=_fmt(metrics["mean_net_bps"]),
                    lcb=_fmt(metrics["lcb_95_net_bps"]),
                    dd=_fmt(metrics["max_drawdown_usd"]),
                )
            )
        lines.append("")
    if report["blocked_candidates"]:
        lines.extend(["## Data-blocked", ""])
        for row in report["blocked_candidates"]:
            lines.append(
                f"- `{row['candidate_id']}`: missing "
                f"`{', '.join(row['missing_fields'])}`."
            )
        lines.append("")
    return "\n".join(lines)


def _render_funding_screen_markdown(report: Mapping[str, Any]) -> str:
    screen = report["screen"]
    metrics = screen["metrics"]
    stress = screen["cost_stress_metrics"]
    funding = report["funding_snapshot"]
    activation = report["activation_funnel"]
    lines = [
        "# Funding Carry recent diagnostic screen",
        "",
        "## Verdict",
        "",
        f"- Result: `{report['verdict']}`",
        f"- Continuation allowed: `{str(report['continuation_allowed']).lower()}`",
        "- Registered historical OOS passed: `false`",
        "- Paper/live allowed: `false`",
        "- Orders enabled: `false`",
        "- Promotion authority: `false`",
        "",
        "The public Bitget funding history window does not cover the registered "
        "validation/OOS/sanity splits. This screen is diagnostic only.",
        "",
        "## Evidence",
        "",
        f"- Funding dataset SHA: `{funding['dataset_sha256']}`",
        f"- Funding manifest SHA: `{funding['manifest_sha256']}`",
        f"- Allowed use: `{funding['allowed_use']}`",
        f"- Overlap bars: `{report['overlap']['bars']}`",
        f"- Candidate signals: `{screen['candidate_signals']}`",
        f"- Filled orders: `{screen['filled_orders']}`",
        f"- Closed trades: `{screen['closed_trades']}`",
        f"- Mean net expectancy: `{_fmt(metrics['mean_net_bps'])} bps`",
        f"- LCB 95: `{_fmt(metrics['lcb_95_net_bps'])} bps`",
        f"- Stress mean: `{_fmt(stress['mean_net_bps'])} bps`",
        f"- Max drawdown: `{_fmt(metrics['max_drawdown_usd'])} USD`",
        f"- Cost floor: `{_fmt(report['early_rejection']['cost_floor_bps'])} bps`",
        "- Max projected carry: "
        f"`{_fmt(report['early_rejection']['max_projected_carry_bps'])} bps`",
        "",
        "## Activation funnel",
        "",
        "| Stage | Observations |",
        "|---|---:|",
        f"| settlements | {activation['settlement_observations']} |",
    ]
    for stage, count in activation["stage_pass_counts"].items():
        lines.append(f"| {stage} | {count} |")
    lines.extend(
        [
            "",
            "First rejection reasons: "
            + ", ".join(
                f"`{key}={value}`"
                for key, value in activation[
                    "first_rejection_reasons"
                ].items()
            ),
            "",
            "## Per symbol",
            "",
            "| Symbol | Trades | Mean net bps | LCB bps |",
            "|---|---:|---:|---:|",
        ]
    )
    for symbol, row in screen["per_symbol"].items():
        lines.append(
            f"| {symbol} | {row['trades']} | "
            f"{_fmt(row['mean_net_bps'])} | {_fmt(row['lcb_95_net_bps'])} |"
        )
    return "\n".join(lines)


def _render_cross_sectional_development_markdown(
    report: Mapping[str, Any],
) -> str:
    candidate = report["candidate"]
    metrics = candidate["metrics"]
    stress = candidate["cost_stress_metrics"]
    baseline = report["baseline"]["metrics"]
    lines = [
        "# Cross-sectional trend 4h development screen",
        "",
        "## Safety",
        "",
        "- Stage: `development_only`",
        "- Validation/OOS/sanity opened: `false`",
        "- Runtime actor created: `false`",
        "- Paper/live allowed: `false`",
        "- Orders enabled: `false`",
        "- Promotion authority: `false`",
        "",
        "## Verdict",
        "",
        f"- Result: `{report['verdict']}`",
        f"- Continuation allowed: `{str(report['continuation_allowed']).lower()}`",
        "- Failures: "
        + (", ".join(f"`{item}`" for item in report["failures"]) or "none"),
        "",
        "## Metrics",
        "",
        "| Candidate | Signals | Fills | Closed | Mean bps | LCB bps | "
        "Stress mean | DD USD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| {report['candidate_id']} | {candidate['candidate_signals']} | "
        f"{candidate['filled_orders']} | {candidate['closed_trades']} | "
        f"{_fmt(metrics['mean_net_bps'])} | "
        f"{_fmt(metrics['lcb_95_net_bps'])} | "
        f"{_fmt(stress['mean_net_bps'])} | "
        f"{_fmt(metrics['max_drawdown_usd'])} |",
        f"| {report['baseline_comparison']['baseline_id']} | "
        f"{report['baseline']['candidate_signals']} | "
        f"{report['baseline']['filled_orders']} | "
        f"{report['baseline']['closed_trades']} | "
        f"{_fmt(baseline['mean_net_bps'])} | "
        f"{_fmt(baseline['lcb_95_net_bps'])} | n/a | "
        f"{_fmt(baseline['max_drawdown_usd'])} |",
        "",
        "## Direction",
        "",
        "| Direction | Trades | Mean net bps | LCB bps |",
        "|---|---:|---:|---:|",
    ]
    for direction, row in candidate["per_direction"].items():
        lines.append(
            f"| {direction} | {row['trades']} | "
            f"{_fmt(row['mean_net_bps'])} | "
            f"{_fmt(row['lcb_95_net_bps'])} |"
        )
    return "\n".join(lines)


def _render_market_neutral_pair_development_markdown(
    report: Mapping[str, Any],
) -> str:
    candidate = report["candidate"]
    metrics = candidate["metrics"]
    stress = candidate["cost_stress_metrics"]
    baseline = report["baseline"]["metrics"]
    lines = [
        "# Market-neutral relative momentum 4h development screen",
        "",
        "## Safety",
        "",
        "- Stage: `development_only`",
        "- Validation/OOS/sanity opened: `false`",
        "- Runtime actor created: `false`",
        "- Paper/live allowed: `false`",
        "- Orders enabled: `false`",
        "- Promotion authority: `false`",
        "",
        "## Verdict",
        "",
        f"- Result: `{report['verdict']}`",
        f"- Continuation allowed: `{str(report['continuation_allowed']).lower()}`",
        "- Failures: "
        + (", ".join(f"`{item}`" for item in report["failures"]) or "none"),
        "",
        "## Pair metrics",
        "",
        "| Candidate | Signals | Fills | Closed pairs | Mean bps | LCB bps | "
        "Stress mean | DD USD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| {report['candidate_id']} | {candidate['candidate_signals']} | "
        f"{candidate['filled_orders']} | {candidate['closed_pairs']} | "
        f"{_fmt(metrics['mean_net_bps'])} | "
        f"{_fmt(metrics['lcb_95_net_bps'])} | "
        f"{_fmt(stress['mean_net_bps'])} | "
        f"{_fmt(metrics['max_drawdown_usd'])} |",
        f"| {report['baseline_comparison']['baseline_id']} | "
        f"{report['baseline']['candidate_signals']} | "
        f"{report['baseline']['filled_orders']} | "
        f"{report['baseline']['closed_pairs']} | "
        f"{_fmt(baseline['mean_net_bps'])} | "
        f"{_fmt(baseline['lcb_95_net_bps'])} | n/a | "
        f"{_fmt(baseline['max_drawdown_usd'])} |",
        "",
        f"Round-trip cost per leg: `{candidate['real_cost_bps_per_leg']} bps`; "
        f"total pair cost: `{candidate['total_pair_round_trip_cost_bps']} bps`.",
        "",
        "## Leg diagnostics",
        "",
        "| Direction | Legs | Mean net bps | LCB bps |",
        "|---|---:|---:|---:|",
    ]
    for direction, row in candidate["per_direction"].items():
        lines.append(
            f"| {direction} | {row['trades']} | "
            f"{_fmt(row['mean_net_bps'])} | "
            f"{_fmt(row['lcb_95_net_bps'])} |"
        )
    lines.extend(
        [
            "",
            "## Exit reasons",
            "",
            "| Reason | Pairs | Mean net bps |",
            "|---|---:|---:|",
        ]
    )
    for reason, row in candidate["per_exit_reason"].items():
        lines.append(
            f"| {reason} | {row['trades']} | "
            f"{_fmt(row['mean_net_bps'])} |"
        )
    return "\n".join(lines)


def _render_market_neutral_portfolio_development_markdown(
    report: Mapping[str, Any],
) -> str:
    candidate = report["candidate"]
    metrics = candidate["metrics"]
    stress = candidate["cost_stress_metrics"]
    baseline = report["baseline"]["metrics"]
    lines = [
        "# Weekly top2/bottom2 market-neutral development screen",
        "",
        "## Safety",
        "",
        "- Stage: `development_only`",
        "- Development run budget: `1`",
        "- Parameter sweep allowed: `false`",
        "- Validation/OOS/sanity opened: `false`",
        "- Runtime actor created: `false`",
        "- Paper/live allowed: `false`",
        "- Orders enabled: `false`",
        "- Promotion authority: `false`",
        "",
        "## Verdict",
        "",
        f"- Result: `{report['verdict']}`",
        f"- Continuation allowed: `{str(report['continuation_allowed']).lower()}`",
        "- Failures: "
        + (", ".join(f"`{item}`" for item in report["failures"]) or "none"),
        "",
        "## Portfolio metrics",
        "",
        "| Candidate | Signals | Fills | Closed portfolios | Mean bps | "
        "LCB bps | Stress mean | DD USD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| {report['candidate_id']} | {candidate['candidate_signals']} | "
        f"{candidate['filled_orders']} | {candidate['closed_portfolios']} | "
        f"{_fmt(metrics['mean_net_bps'])} | "
        f"{_fmt(metrics['lcb_95_net_bps'])} | "
        f"{_fmt(stress['mean_net_bps'])} | "
        f"{_fmt(metrics['max_drawdown_usd'])} |",
        f"| {report['baseline_comparison']['baseline_id']} | "
        f"{report['baseline']['candidate_signals']} | "
        f"{report['baseline']['filled_orders']} | "
        f"{report['baseline']['closed_portfolios']} | "
        f"{_fmt(baseline['mean_net_bps'])} | "
        f"{_fmt(baseline['lcb_95_net_bps'])} | n/a | "
        f"{_fmt(baseline['max_drawdown_usd'])} |",
        "",
        f"Long notional: `${candidate['long_notional_usd']}`; short notional: "
        f"`${candidate['short_notional_usd']}`; net: "
        f"`${candidate['net_notional_usd']}`.",
        "",
        "## Leg diagnostics",
        "",
        "| Direction | Legs | Mean net bps | LCB bps |",
        "|---|---:|---:|---:|",
    ]
    for direction, row in candidate["per_direction"].items():
        lines.append(
            f"| {direction} | {row['trades']} | "
            f"{_fmt(row['mean_net_bps'])} | "
            f"{_fmt(row['lcb_95_net_bps'])} |"
        )
    return "\n".join(lines)


def _base_symbol(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text.split("/", 1)[0].split("-", 1)[0]


def _timestamp_ms(value: Any) -> int:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise StrategyLabError("split timestamp must be timezone-aware")
    return int(parsed.timestamp() * 1000)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"
