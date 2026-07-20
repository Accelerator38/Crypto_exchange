"""Atomic, hash-chained Bitget evidence tape for CarryFlow replay."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..analysis.technical_indicators import TechnicalIndicatorState
from ..app.regime_detector import PriceRegimeDetector
from ..domain.types import MarketSnapshot
from .derivatives_context import (
    DerivativesContextRecord,
    HistoricalDerivativesContext,
    canonical_derivatives_symbol,
)
from .manifest import MarketDataPolicy


TAPE_SCHEMA_VERSION = "panteon.bitget_carryflow_tape.v1"
ZERO_SHA256 = "0" * 64
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{7,40}$")

_SAMPLE_KEYS = frozenset(
    {
        "schema_version",
        "collector_run_id",
        "source_revision",
        "collector_code_sha256",
        "funding_fetcher_code_sha256",
        "source_exchange",
        "market_type",
        "observed_at",
        "observed_at_ms",
        "bar_interval_seconds",
        "bar_close_timestamp_ms",
        "collection_lag_seconds",
        "max_bar_close_lag_seconds",
        "max_derivatives_age_seconds",
        "symbol_set",
        "symbols",
        "complete_symbols",
        "incomplete_symbols",
        "sample_complete",
        "sample_reasons",
        "previous_sample_sha256",
        "sample_sha256",
    }
)
_SYMBOL_KEYS = frozenset({"market", "derivatives", "complete", "reasons"})
_MARKET_KEYS = frozenset(
    {
        "market_symbol",
        "candle_timestamp_ms",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "decision_price",
        "complete",
    }
)
_DERIVATIVES_KEYS = frozenset(
    {
        "funding_rate",
        "open_interest_usdt",
        "long_ratio",
        "short_ratio",
        "mark_price",
        "index_price",
        "last_price",
        "next_funding_ts",
        "long_short_ratio_ts",
        "long_short_ratio_age_sec",
        "long_short_source",
        "source_updated_ts",
        "age_sec",
        "context_complete",
    }
)


class EvidenceTapeError(ValueError):
    """The tape cannot be trusted as replay evidence."""


@dataclass(frozen=True)
class TapeReplayBundle:
    snapshots: tuple[MarketSnapshot, ...]
    derivatives_context: HistoricalDerivativesContext


class CarryFlowEvidenceTape:
    def __init__(self, *, path: Path, samples: Sequence[Mapping[str, Any]]) -> None:
        if not samples:
            raise EvidenceTapeError("evidence tape contains no samples")
        self.path = path
        self.samples = tuple(_json_copy(sample) for sample in samples)
        first = self.samples[0]
        self.bar_interval_seconds = int(first["bar_interval_seconds"])
        self.max_bar_close_lag_seconds = int(first["max_bar_close_lag_seconds"])
        self.max_derivatives_age_seconds = int(
            first["max_derivatives_age_seconds"]
        )
        self.symbols = tuple(str(item) for item in first["symbol_set"])
        self.source_revision = str(first["source_revision"])
        self.collector_code_sha256 = str(first["collector_code_sha256"])
        self.funding_fetcher_code_sha256 = str(
            first["funding_fetcher_code_sha256"]
        )
        self.collector_run_id = str(first["collector_run_id"])

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        expected_symbols: Iterable[str] | None = None,
    ) -> "CarryFlowEvidenceTape":
        source = Path(path)
        if not source.is_file():
            raise EvidenceTapeError(f"evidence tape not found: {source}")
        selected = (
            tuple(canonical_derivatives_symbol(item) for item in expected_symbols)
            if expected_symbols is not None
            else None
        )
        samples: list[dict[str, Any]] = []
        previous_sha = ZERO_SHA256
        first_contract: tuple[Any, ...] | None = None
        previous_observed_ms = 0
        previous_bar_close_ms = 0
        with source.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    raise EvidenceTapeError(
                        f"blank evidence tape line: {line_number}"
                    )
                try:
                    raw = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise EvidenceTapeError(
                        f"invalid evidence tape JSON at line {line_number}: {exc}"
                    ) from exc
                sample = validate_tape_sample(
                    raw,
                    expected_previous_sha256=previous_sha,
                )
                contract = (
                    int(sample["bar_interval_seconds"]),
                    int(sample["max_bar_close_lag_seconds"]),
                    int(sample["max_derivatives_age_seconds"]),
                    tuple(sample["symbol_set"]),
                    str(sample["source_revision"]),
                    str(sample["collector_code_sha256"]),
                    str(sample["funding_fetcher_code_sha256"]),
                    str(sample["collector_run_id"]),
                )
                if first_contract is None:
                    first_contract = contract
                elif contract != first_contract:
                    raise EvidenceTapeError(
                        f"evidence tape contract changed at line {line_number}"
                    )
                if selected is not None and tuple(sample["symbol_set"]) != selected:
                    raise EvidenceTapeError("evidence tape symbol set mismatch")
                observed_ms = int(sample["observed_at_ms"])
                bar_close_ms = int(sample["bar_close_timestamp_ms"])
                if previous_observed_ms and observed_ms <= previous_observed_ms:
                    raise EvidenceTapeError(
                        "evidence tape observations are not strictly increasing"
                    )
                if previous_bar_close_ms:
                    expected_close = (
                        previous_bar_close_ms
                        + int(sample["bar_interval_seconds"]) * 1000
                    )
                    if bar_close_ms != expected_close:
                        raise EvidenceTapeError(
                            "evidence tape has a duplicate or missing market bar"
                        )
                previous_observed_ms = observed_ms
                previous_bar_close_ms = bar_close_ms
                previous_sha = str(sample["sample_sha256"])
                samples.append(sample)
        return cls(path=source.resolve(), samples=samples)

    @property
    def head_sha256(self) -> str:
        return str(self.samples[-1]["sample_sha256"])

    @property
    def file_sha256(self) -> str:
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def describe(self) -> dict[str, Any]:
        symbol_observations = len(self.samples) * len(self.symbols)
        complete_symbol_observations = sum(
            bool(sample["symbols"][symbol]["complete"])
            for sample in self.samples
            for symbol in self.symbols
        )
        complete_samples = sum(bool(sample["sample_complete"]) for sample in self.samples)
        return {
            "schema_version": TAPE_SCHEMA_VERSION,
            "path": str(self.path),
            "file_sha256": self.file_sha256,
            "head_sha256": self.head_sha256,
            "collector_run_id": self.collector_run_id,
            "source_revision": self.source_revision,
            "collector_code_sha256": self.collector_code_sha256,
            "funding_fetcher_code_sha256": self.funding_fetcher_code_sha256,
            "bar_interval_seconds": self.bar_interval_seconds,
            "max_bar_close_lag_seconds": self.max_bar_close_lag_seconds,
            "max_derivatives_age_seconds": self.max_derivatives_age_seconds,
            "symbols": list(self.symbols),
            "samples": len(self.samples),
            "complete_samples": complete_samples,
            "incomplete_samples": len(self.samples) - complete_samples,
            "symbol_observations": symbol_observations,
            "complete_symbol_observations": complete_symbol_observations,
            "context_coverage_pct": (
                complete_symbol_observations / symbol_observations * 100.0
                if symbol_observations
                else 0.0
            ),
            "first_observed_at": self.samples[0]["observed_at"],
            "last_observed_at": self.samples[-1]["observed_at"],
            "first_bar_close_timestamp_ms": self.samples[0][
                "bar_close_timestamp_ms"
            ],
            "last_bar_close_timestamp_ms": self.samples[-1][
                "bar_close_timestamp_ms"
            ],
        }

    def replay_bundle(
        self,
        data_policy: MarketDataPolicy,
        *,
        warmup_seed: Any | None = None,
    ) -> TapeReplayBundle:
        if int(data_policy.bar_interval_seconds) != self.bar_interval_seconds:
            raise EvidenceTapeError("manifest/tape bar interval mismatch")
        if (
            int(data_policy.max_bar_close_lag_seconds)
            != self.max_bar_close_lag_seconds
        ):
            raise EvidenceTapeError("manifest/tape bar close lag mismatch")
        if (
            int(data_policy.max_derivatives_age_seconds)
            != self.max_derivatives_age_seconds
        ):
            raise EvidenceTapeError("manifest/tape derivatives age mismatch")
        previous_bar_close_ms: int | None = None
        detector = PriceRegimeDetector(
            exchange_name="BITGET",
            poll_interval_sec=float(self.bar_interval_seconds),
            horizons_sec=(
                self.bar_interval_seconds,
                self.bar_interval_seconds * 3,
                self.bar_interval_seconds * 12,
                self.bar_interval_seconds * 24,
            ),
        )
        technicals = TechnicalIndicatorState()
        snapshots: list[MarketSnapshot] = []
        records: list[DerivativesContextRecord] = []
        if warmup_seed is not None:
            warmup_seed.validate_for_tape(self)
            for seed_bar in warmup_seed.bars:
                prices: dict[str, float] = {}
                volumes: dict[str, float] = {}
                for symbol in self.symbols:
                    candle = seed_bar["symbols"][symbol]
                    close = float(candle["close"])
                    prices[symbol] = close
                    volumes[symbol] = float(candle["volume"])
                    technicals.update_symbol(
                        symbol,
                        high=float(candle["high"]),
                        low=float(candle["low"]),
                        close=close,
                    )
                detector.update(
                    prices,
                    volumes=volumes,
                    funding={symbol: 0.0 for symbol in self.symbols},
                )
        for bar, sample in enumerate(self.samples, start=1):
            timestamp = _parse_datetime(sample["observed_at"])
            bar_close_ms = int(sample["bar_close_timestamp_ms"])
            cadence_timestamp = datetime.fromtimestamp(
                bar_close_ms / 1000.0,
                timezone.utc,
            )
            if previous_bar_close_ms is not None:
                delta = (bar_close_ms - previous_bar_close_ms) / 1000.0
                expected = float(data_policy.bar_interval_seconds)
                tolerance = float(data_policy.cadence_tolerance_seconds)
                if abs(delta - expected) > tolerance:
                    raise EvidenceTapeError(
                        "evidence tape bar-close cadence violates manifest"
                    )
            previous_bar_close_ms = bar_close_ms
            lag = float(sample["collection_lag_seconds"])
            if lag < 0.0 or lag > data_policy.max_bar_close_lag_seconds:
                raise EvidenceTapeError(
                    "evidence tape sample exceeds manifest bar close lag"
                )
            prices: dict[str, float] = {}
            volumes: dict[str, float] = {}
            funding: dict[str, float] = {}
            bar_opens: dict[str, float] = {}
            bar_highs: dict[str, float] = {}
            bar_lows: dict[str, float] = {}
            bar_closes: dict[str, float] = {}
            technicals_by_symbol = {}
            for symbol in self.symbols:
                row = sample["symbols"][symbol]
                market = row["market"]
                derivatives = row["derivatives"]
                if not bool(market["complete"]):
                    raise EvidenceTapeError(
                        f"market data incomplete for {symbol} at bar {bar}"
                    )
                price = float(market["decision_price"])
                bar_high = float(market["high"])
                bar_low = float(market["low"])
                indicator_high = max(bar_high, price)
                indicator_low = min(bar_low, price)
                prices[symbol] = price
                volumes[symbol] = float(market["volume"])
                funding[symbol] = float(derivatives["funding_rate"])
                bar_opens[symbol] = float(market["open"])
                bar_highs[symbol] = bar_high
                bar_lows[symbol] = bar_low
                bar_closes[symbol] = float(market["close"])
                technicals_by_symbol[symbol] = technicals.update_symbol(
                    symbol,
                    high=indicator_high,
                    low=indicator_low,
                    close=price,
                )
                records.append(
                    DerivativesContextRecord(
                        timestamp=timestamp,
                        symbol=symbol,
                        funding_rate=float(derivatives["funding_rate"]),
                        open_interest_usdt=float(
                            derivatives["open_interest_usdt"]
                        ),
                        long_ratio=float(derivatives["long_ratio"]),
                        short_ratio=float(derivatives["short_ratio"]),
                        mark_price=float(derivatives["mark_price"]),
                        index_price=float(derivatives["index_price"]),
                        last_price=float(derivatives["last_price"]),
                        next_funding_ts=int(derivatives["next_funding_ts"]),
                        long_short_ratio_ts=int(
                            derivatives["long_short_ratio_ts"]
                        ),
                        source_updated_ts=float(
                            derivatives["source_updated_ts"]
                        ),
                        context_complete=bool(
                            derivatives["context_complete"]
                        ),
                    )
                )
            regime = detector.update(prices, volumes=volumes, funding=funding)
            snapshots.append(
                MarketSnapshot(
                    bar=bar,
                    timestamp=timestamp,
                    regime=regime,
                    regime_confidence=float(detector.confidence),
                    prices=prices,
                    volumes=volumes,
                    funding=funding,
                    month=timestamp.month,
                    technicals_by_symbol=technicals_by_symbol,
                    regimes_by_symbol=detector.symbol_regimes,
                    regime_features_by_symbol=detector.symbol_stats,
                    bar_opens=bar_opens,
                    bar_highs=bar_highs,
                    bar_lows=bar_lows,
                    bar_closes=bar_closes,
                    cadence_timestamp=cadence_timestamp,
                )
            )
        return TapeReplayBundle(
            snapshots=tuple(snapshots),
            derivatives_context=HistoricalDerivativesContext(records),
        )


def compute_sample_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("sample_sha256", None)
    raw = json.dumps(
        canonical,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def seal_tape_sample(payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = _json_copy(payload)
    sealed["sample_sha256"] = compute_sample_sha256(sealed)
    return validate_tape_sample(sealed)


def validate_tape_sample(
    payload: Mapping[str, Any],
    *,
    expected_previous_sha256: str | None = None,
) -> dict[str, Any]:
    sample = _json_copy(payload)
    _exact_keys(sample, _SAMPLE_KEYS, "sample")
    if sample["schema_version"] != TAPE_SCHEMA_VERSION:
        raise EvidenceTapeError("unsupported evidence tape schema")
    if str(sample["source_exchange"]).strip().upper() != "BITGET":
        raise EvidenceTapeError("evidence tape exchange must be BITGET")
    if str(sample["market_type"]).strip().lower() != "swap":
        raise EvidenceTapeError("evidence tape market_type must be swap")
    if not str(sample["collector_run_id"]).strip():
        raise EvidenceTapeError("collector_run_id is required")
    revision = str(sample["source_revision"]).strip().lower()
    if not _REVISION_RE.fullmatch(revision):
        raise EvidenceTapeError("source_revision is invalid")
    sample["source_revision"] = revision
    for key in ("collector_code_sha256", "funding_fetcher_code_sha256"):
        digest = str(sample[key]).strip().lower()
        if not _SHA256_RE.fullmatch(digest):
            raise EvidenceTapeError(f"{key} is invalid")
        sample[key] = digest
    interval = _integer(sample["bar_interval_seconds"], "bar_interval_seconds")
    if not 60 <= interval <= 24 * 60 * 60 or interval % 60 != 0:
        raise EvidenceTapeError("bar_interval_seconds is invalid")
    max_lag = _integer(
        sample["max_bar_close_lag_seconds"],
        "max_bar_close_lag_seconds",
    )
    if not 1 <= max_lag <= 300:
        raise EvidenceTapeError("max_bar_close_lag_seconds is invalid")
    max_derivatives_age = _integer(
        sample["max_derivatives_age_seconds"],
        "max_derivatives_age_seconds",
    )
    if not 60 <= max_derivatives_age <= 60 * 60:
        raise EvidenceTapeError("max_derivatives_age_seconds is invalid")
    observed_ms = _integer(sample["observed_at_ms"], "observed_at_ms")
    bar_close_ms = _integer(
        sample["bar_close_timestamp_ms"], "bar_close_timestamp_ms"
    )
    observed_at = _parse_datetime(sample["observed_at"])
    if abs(observed_at.timestamp() * 1000.0 - observed_ms) > 1.0:
        raise EvidenceTapeError("observed_at does not match observed_at_ms")
    collection_lag = _finite(
        sample["collection_lag_seconds"], "collection_lag_seconds"
    )
    expected_lag = (observed_ms - bar_close_ms) / 1000.0
    if abs(collection_lag - expected_lag) > 0.001:
        raise EvidenceTapeError("collection_lag_seconds is inconsistent")

    symbol_set = sample["symbol_set"]
    if not isinstance(symbol_set, list) or not symbol_set:
        raise EvidenceTapeError("symbol_set must be a non-empty list")
    symbols = tuple(canonical_derivatives_symbol(item) for item in symbol_set)
    if any(not item for item in symbols) or len(symbols) != len(set(symbols)):
        raise EvidenceTapeError("symbol_set contains invalid or duplicate symbols")
    if list(symbols) != symbol_set:
        raise EvidenceTapeError("symbol_set must use canonical symbols")
    raw_symbols = sample["symbols"]
    if not isinstance(raw_symbols, dict) or set(raw_symbols) != set(symbols):
        raise EvidenceTapeError("symbols payload does not match symbol_set")

    complete_symbols: list[str] = []
    incomplete_symbols: list[str] = []
    for symbol in symbols:
        row = raw_symbols[symbol]
        if not isinstance(row, dict):
            raise EvidenceTapeError(f"symbol payload must be an object: {symbol}")
        _exact_keys(row, _SYMBOL_KEYS, f"symbols.{symbol}")
        market = row["market"]
        derivatives = row["derivatives"]
        if not isinstance(market, dict) or not isinstance(derivatives, dict):
            raise EvidenceTapeError(f"invalid market/derivatives payload: {symbol}")
        _exact_keys(market, _MARKET_KEYS, f"symbols.{symbol}.market")
        _exact_keys(
            derivatives,
            _DERIVATIVES_KEYS,
            f"symbols.{symbol}.derivatives",
        )
        market_complete = _validate_market(
            market,
            interval=interval,
            bar_close_ms=bar_close_ms,
            symbol=symbol,
        )
        derivatives_complete = _validate_derivatives(
            derivatives,
            symbol=symbol,
            max_age_seconds=max_derivatives_age,
        )
        declared_complete = _boolean(row["complete"], f"symbols.{symbol}.complete")
        expected_complete = market_complete and derivatives_complete
        if declared_complete != expected_complete:
            raise EvidenceTapeError(f"symbol completeness mismatch: {symbol}")
        reasons = _string_list(row["reasons"], f"symbols.{symbol}.reasons")
        if declared_complete and reasons:
            raise EvidenceTapeError(f"complete symbol has failure reasons: {symbol}")
        if not declared_complete and not reasons:
            raise EvidenceTapeError(f"incomplete symbol lacks failure reason: {symbol}")
        if declared_complete:
            complete_symbols.append(symbol)
        else:
            incomplete_symbols.append(symbol)

    if sample["complete_symbols"] != complete_symbols:
        raise EvidenceTapeError("complete_symbols is inconsistent")
    if sample["incomplete_symbols"] != incomplete_symbols:
        raise EvidenceTapeError("incomplete_symbols is inconsistent")
    lag_ok = 0.0 <= collection_lag <= max_lag
    expected_sample_complete = not incomplete_symbols and lag_ok
    declared_sample_complete = _boolean(sample["sample_complete"], "sample_complete")
    if declared_sample_complete != expected_sample_complete:
        raise EvidenceTapeError("sample_complete is inconsistent")
    sample_reasons = _string_list(sample["sample_reasons"], "sample_reasons")
    expected_sample_reasons = []
    if incomplete_symbols:
        expected_sample_reasons.append("incomplete_symbols")
    if not lag_ok:
        expected_sample_reasons.append("bar_close_lag_exceeded")
    if sample_reasons != expected_sample_reasons:
        raise EvidenceTapeError("sample_reasons is inconsistent")

    previous = str(sample["previous_sample_sha256"]).lower()
    digest = str(sample["sample_sha256"]).lower()
    if not _SHA256_RE.fullmatch(previous) or not _SHA256_RE.fullmatch(digest):
        raise EvidenceTapeError("sample hash is invalid")
    if expected_previous_sha256 is not None and previous != expected_previous_sha256:
        raise EvidenceTapeError("evidence tape hash chain is broken")
    if digest != compute_sample_sha256(sample):
        raise EvidenceTapeError("evidence tape sample hash mismatch")
    sample["previous_sample_sha256"] = previous
    sample["sample_sha256"] = digest
    return sample


def append_tape_sample(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    previous_sha = ZERO_SHA256
    previous_sample: Mapping[str, Any] | None = None
    if output.exists() and output.stat().st_size > 0:
        existing = CarryFlowEvidenceTape.from_jsonl(output)
        previous_sha = existing.head_sha256
        previous_sample = existing.samples[-1]
    draft = _json_copy(payload)
    draft["previous_sample_sha256"] = previous_sha
    draft.pop("sample_sha256", None)
    sealed = seal_tape_sample(draft)
    if previous_sample is not None:
        for key in (
            "collector_run_id",
            "source_revision",
            "collector_code_sha256",
            "funding_fetcher_code_sha256",
            "bar_interval_seconds",
            "max_bar_close_lag_seconds",
            "max_derivatives_age_seconds",
            "symbol_set",
        ):
            if sealed[key] != previous_sample[key]:
                raise EvidenceTapeError(f"cannot append changed tape contract: {key}")
        expected_close = (
            int(previous_sample["bar_close_timestamp_ms"])
            + int(sealed["bar_interval_seconds"]) * 1000
        )
        if int(sealed["bar_close_timestamp_ms"]) != expected_close:
            raise EvidenceTapeError("cannot append duplicate or missing market bar")
        if int(sealed["observed_at_ms"]) <= int(previous_sample["observed_at_ms"]):
            raise EvidenceTapeError("cannot append non-increasing observation")
    encoded = (
        json.dumps(
            sealed,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(output, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise EvidenceTapeError("partial evidence tape write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return sealed


def _validate_market(
    market: Mapping[str, Any],
    *,
    interval: int,
    bar_close_ms: int,
    symbol: str,
) -> bool:
    declared = _boolean(market["complete"], f"market.complete.{symbol}")
    candle_ms = _integer(market["candle_timestamp_ms"], "candle_timestamp_ms")
    values = {
        name: _finite(market[name], f"market.{name}.{symbol}")
        for name in ("open", "high", "low", "close", "volume", "decision_price")
    }
    values_complete = bool(
        candle_ms > 0
        and candle_ms + interval * 1000 == bar_close_ms
        and values["open"] > 0.0
        and values["high"] > 0.0
        and values["low"] > 0.0
        and values["close"] > 0.0
        and values["decision_price"] > 0.0
        and values["volume"] >= 0.0
        and values["high"] >= values["low"]
        and values["low"] <= values["open"] <= values["high"]
        and values["low"] <= values["close"] <= values["high"]
    )
    if declared and not values_complete:
        raise EvidenceTapeError(f"declared complete market row is invalid: {symbol}")
    return declared and values_complete


def _validate_derivatives(
    derivatives: Mapping[str, Any],
    *,
    symbol: str,
    max_age_seconds: int,
) -> bool:
    values = {
        name: _finite(derivatives[name], f"derivatives.{name}.{symbol}")
        for name in (
            "funding_rate",
            "open_interest_usdt",
            "long_ratio",
            "short_ratio",
            "mark_price",
            "index_price",
            "last_price",
            "long_short_ratio_age_sec",
            "source_updated_ts",
            "age_sec",
        )
    }
    _integer(derivatives["next_funding_ts"], "next_funding_ts")
    _integer(derivatives["long_short_ratio_ts"], "long_short_ratio_ts")
    if not str(derivatives["long_short_source"]).strip():
        raise EvidenceTapeError(f"long_short_source is missing: {symbol}")
    declared = _boolean(
        derivatives["context_complete"],
        f"derivatives.context_complete.{symbol}",
    )
    values_complete = bool(
        values["open_interest_usdt"] > 0.0
        and 0.0 < values["long_ratio"] <= 1.0
        and 0.0 < values["short_ratio"] <= 1.0
        and values["mark_price"] > 0.0
        and values["index_price"] > 0.0
        and values["last_price"] > 0.0
        and 0.0 <= values["age_sec"] <= max_age_seconds
    )
    if declared and not values_complete:
        raise EvidenceTapeError(
            f"declared complete derivatives row is invalid: {symbol}"
        )
    return declared and values_complete


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        parts = []
        if missing:
            parts.append("missing=" + ",".join(missing))
        if extra:
            parts.append("extra=" + ",".join(extra))
        raise EvidenceTapeError(f"{label} keys invalid: {'; '.join(parts)}")


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(
            json.dumps(value, ensure_ascii=True, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise EvidenceTapeError(f"evidence tape payload is not canonical JSON: {exc}") from exc


def _finite(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceTapeError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed):
        raise EvidenceTapeError(f"{label} must be finite")
    return parsed


def _integer(value: Any, label: str) -> int:
    parsed = _finite(value, label)
    if int(parsed) != parsed:
        raise EvidenceTapeError(f"{label} must be an integer")
    return int(parsed)


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise EvidenceTapeError(f"{label} must be boolean")
    return value


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise EvidenceTapeError(f"{label} must be a string list")
    return list(value)


def _parse_datetime(value: Any) -> datetime:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise EvidenceTapeError("invalid observed_at") from exc
    if parsed.tzinfo is None:
        raise EvidenceTapeError("observed_at must include timezone")
    return parsed.astimezone(timezone.utc)
