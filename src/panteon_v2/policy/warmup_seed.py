"""Immutable closed-candle seed used to warm policy indicators before replay."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..domain.types import MarketSnapshot, Regime
from .derivatives_context import canonical_derivatives_symbol


WARMUP_SEED_SCHEMA_VERSION = "panteon.carryflow_warmup_seed.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{7,40}$")
_PAYLOAD_KEYS = frozenset(
    {
        "schema_version",
        "collector_run_id",
        "source_revision",
        "collector_code_sha256",
        "source_exchange",
        "market_type",
        "created_at",
        "bar_interval_seconds",
        "intended_first_evidence_bar_close_timestamp_ms",
        "symbol_set",
        "bars",
        "seed_sha256",
    }
)
_BAR_KEYS = frozenset({"bar_close_timestamp_ms", "symbols"})
_CANDLE_KEYS = frozenset(
    {
        "candle_timestamp_ms",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
)


class WarmupSeedError(ValueError):
    """The warm-up seed is incomplete, inconsistent, or has been modified."""


@dataclass(frozen=True)
class CarryFlowWarmupSeed:
    path: Path
    payload: Mapping[str, Any]

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        expected_symbols: Sequence[str] | None = None,
    ) -> "CarryFlowWarmupSeed":
        source = Path(path)
        if not source.is_file():
            raise WarmupSeedError(f"warm-up seed not found: {source}")
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise WarmupSeedError(f"invalid warm-up seed JSON: {exc}") from exc
        payload = validate_warmup_seed(raw)
        if expected_symbols is not None:
            selected = tuple(
                canonical_derivatives_symbol(symbol)
                for symbol in expected_symbols
            )
            if tuple(payload["symbol_set"]) != selected:
                raise WarmupSeedError("warm-up seed symbol set mismatch")
        return cls(path=source.resolve(), payload=payload)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.payload["symbol_set"])

    @property
    def bar_interval_seconds(self) -> int:
        return int(self.payload["bar_interval_seconds"])

    @property
    def seed_sha256(self) -> str:
        return str(self.payload["seed_sha256"])

    @property
    def bars(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.payload["bars"])

    def validate_for_tape(self, tape: Any) -> None:
        if tuple(tape.symbols) != self.symbols:
            raise WarmupSeedError("warm-up seed/tape symbol set mismatch")
        if int(tape.bar_interval_seconds) != self.bar_interval_seconds:
            raise WarmupSeedError("warm-up seed/tape bar interval mismatch")
        if str(tape.collector_run_id) != str(self.payload["collector_run_id"]):
            raise WarmupSeedError("warm-up seed/tape collector run mismatch")
        if str(tape.source_revision) != str(self.payload["source_revision"]):
            raise WarmupSeedError("warm-up seed/tape source revision mismatch")
        first_close = int(tape.samples[0]["bar_close_timestamp_ms"])
        intended_close = int(
            self.payload["intended_first_evidence_bar_close_timestamp_ms"]
        )
        if first_close != intended_close:
            raise WarmupSeedError(
                "warm-up seed does not immediately precede the evidence tape"
            )

    def is_prospective_for_tape(self, tape: Any) -> bool:
        created_at = _parse_datetime(self.payload["created_at"])
        first_observed = _parse_datetime(tape.samples[0]["observed_at"])
        return created_at <= first_observed

    def snapshots(self) -> tuple[MarketSnapshot, ...]:
        total = len(self.bars)
        result: list[MarketSnapshot] = []
        for index, row in enumerate(self.bars, start=1):
            timestamp = datetime.fromtimestamp(
                int(row["bar_close_timestamp_ms"]) / 1000.0,
                timezone.utc,
            )
            prices = {
                symbol: float(row["symbols"][symbol]["close"])
                for symbol in self.symbols
            }
            volumes = {
                symbol: float(row["symbols"][symbol]["volume"])
                for symbol in self.symbols
            }
            result.append(
                MarketSnapshot(
                    bar=index - total,
                    timestamp=timestamp,
                    cadence_timestamp=timestamp,
                    regime=Regime.NEUTRAL,
                    regime_confidence=1.0,
                    prices=prices,
                    volumes=volumes,
                    funding={symbol: 0.0 for symbol in self.symbols},
                    month=timestamp.month,
                )
            )
        return tuple(result)

    def describe(self) -> dict[str, Any]:
        return {
            "schema_version": WARMUP_SEED_SCHEMA_VERSION,
            "path": str(self.path),
            "seed_sha256": self.seed_sha256,
            "collector_run_id": self.payload["collector_run_id"],
            "source_revision": self.payload["source_revision"],
            "created_at": self.payload["created_at"],
            "bar_interval_seconds": self.bar_interval_seconds,
            "symbols": list(self.symbols),
            "bars": len(self.bars),
            "first_bar_close_timestamp_ms": self.bars[0][
                "bar_close_timestamp_ms"
            ],
            "last_bar_close_timestamp_ms": self.bars[-1][
                "bar_close_timestamp_ms"
            ],
            "intended_first_evidence_bar_close_timestamp_ms": self.payload[
                "intended_first_evidence_bar_close_timestamp_ms"
            ],
        }


def compute_warmup_seed_sha256(payload: Mapping[str, Any]) -> str:
    canonical = _json_copy(payload)
    canonical.pop("seed_sha256", None)
    raw = json.dumps(
        canonical,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def seal_warmup_seed(payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = _json_copy(payload)
    sealed["seed_sha256"] = compute_warmup_seed_sha256(sealed)
    return validate_warmup_seed(sealed)


def validate_warmup_seed(payload: Mapping[str, Any]) -> dict[str, Any]:
    seed = _json_copy(payload)
    _exact_keys(seed, _PAYLOAD_KEYS, "warm-up seed")
    if seed["schema_version"] != WARMUP_SEED_SCHEMA_VERSION:
        raise WarmupSeedError("unsupported warm-up seed schema")
    if str(seed["source_exchange"]).upper() != "BITGET":
        raise WarmupSeedError("warm-up seed exchange must be BITGET")
    if str(seed["market_type"]).lower() != "swap":
        raise WarmupSeedError("warm-up seed market_type must be swap")
    if not str(seed["collector_run_id"]).strip():
        raise WarmupSeedError("warm-up seed collector_run_id is required")
    revision = str(seed["source_revision"]).strip().lower()
    if not _REVISION_RE.fullmatch(revision):
        raise WarmupSeedError("warm-up seed source_revision is invalid")
    seed["source_revision"] = revision
    code_sha = str(seed["collector_code_sha256"]).strip().lower()
    if not _SHA256_RE.fullmatch(code_sha):
        raise WarmupSeedError("warm-up seed collector_code_sha256 is invalid")
    seed["collector_code_sha256"] = code_sha
    _parse_datetime(seed["created_at"])
    interval = _integer(seed["bar_interval_seconds"], "bar_interval_seconds")
    if not 60 <= interval <= 86_400 or interval % 60:
        raise WarmupSeedError("warm-up seed bar interval is invalid")
    seed["bar_interval_seconds"] = interval
    intended_close = _integer(
        seed["intended_first_evidence_bar_close_timestamp_ms"],
        "intended_first_evidence_bar_close_timestamp_ms",
    )
    if intended_close <= 0 or intended_close % (interval * 1000):
        raise WarmupSeedError("intended first evidence bar close is misaligned")
    seed["intended_first_evidence_bar_close_timestamp_ms"] = intended_close
    raw_symbols = seed["symbol_set"]
    if not isinstance(raw_symbols, list) or not raw_symbols:
        raise WarmupSeedError("warm-up seed symbol_set must be non-empty")
    symbols = [canonical_derivatives_symbol(item) for item in raw_symbols]
    if any(not symbol for symbol in symbols) or len(set(symbols)) != len(symbols):
        raise WarmupSeedError("warm-up seed symbol_set is invalid")
    seed["symbol_set"] = symbols
    bars = seed["bars"]
    if not isinstance(bars, list) or not bars:
        raise WarmupSeedError("warm-up seed bars must be non-empty")
    previous_close = 0
    normalized_bars: list[dict[str, Any]] = []
    for index, raw_bar in enumerate(bars, start=1):
        if not isinstance(raw_bar, Mapping):
            raise WarmupSeedError(f"warm-up bar {index} must be an object")
        bar = _json_copy(raw_bar)
        _exact_keys(bar, _BAR_KEYS, f"warm-up bar {index}")
        close_ms = _integer(
            bar["bar_close_timestamp_ms"],
            f"bar {index} close timestamp",
        )
        if close_ms <= 0 or close_ms % (interval * 1000):
            raise WarmupSeedError(f"warm-up bar {index} is misaligned")
        if previous_close and close_ms != previous_close + interval * 1000:
            raise WarmupSeedError("warm-up seed has a duplicate or missing bar")
        previous_close = close_ms
        raw_rows = bar["symbols"]
        if not isinstance(raw_rows, Mapping) or set(raw_rows) != set(symbols):
            raise WarmupSeedError(
                f"warm-up bar {index} does not contain the exact symbol set"
            )
        rows: dict[str, dict[str, Any]] = {}
        for symbol in symbols:
            row = _json_copy(raw_rows[symbol])
            _exact_keys(row, _CANDLE_KEYS, f"warm-up candle {index}/{symbol}")
            candle_ms = _integer(
                row["candle_timestamp_ms"],
                f"warm-up candle {index}/{symbol} timestamp",
            )
            if candle_ms != close_ms - interval * 1000:
                raise WarmupSeedError(
                    f"warm-up candle {index}/{symbol} is misaligned"
                )
            open_price = _finite(row["open"], f"{index}/{symbol} open")
            high = _finite(row["high"], f"{index}/{symbol} high")
            low = _finite(row["low"], f"{index}/{symbol} low")
            close = _finite(row["close"], f"{index}/{symbol} close")
            volume = _finite(row["volume"], f"{index}/{symbol} volume")
            if not (
                open_price > 0.0
                and high >= low > 0.0
                and low <= open_price <= high
                and low <= close <= high
                and volume >= 0.0
            ):
                raise WarmupSeedError(
                    f"warm-up candle {index}/{symbol} has invalid OHLCV"
                )
            rows[symbol] = {
                "candle_timestamp_ms": candle_ms,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        normalized_bars.append(
            {"bar_close_timestamp_ms": close_ms, "symbols": rows}
        )
    if previous_close + interval * 1000 != intended_close:
        raise WarmupSeedError(
            "warm-up seed does not end immediately before intended evidence"
        )
    seed["bars"] = normalized_bars
    digest = str(seed["seed_sha256"]).strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        raise WarmupSeedError("warm-up seed SHA256 is invalid")
    seed["seed_sha256"] = digest
    if compute_warmup_seed_sha256(seed) != digest:
        raise WarmupSeedError("warm-up seed hash mismatch")
    return seed


def write_warmup_seed(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    target = Path(path)
    if target.exists():
        raise WarmupSeedError(f"refusing to overwrite warm-up seed: {target}")
    sealed = seal_warmup_seed(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(sealed, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return sealed


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise WarmupSeedError(f"warm-up seed is not canonical JSON: {exc}") from exc


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise WarmupSeedError(f"{name} keys are invalid ({'; '.join(details)})")


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise WarmupSeedError(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise WarmupSeedError(f"{name} must be an integer") from exc
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise WarmupSeedError(f"{name} must be an integer") from exc
    if not math.isfinite(numeric) or numeric != parsed:
        raise WarmupSeedError(f"{name} must be an integer")
    return parsed


def _finite(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise WarmupSeedError(f"{name} must be finite") from exc
    if not math.isfinite(parsed):
        raise WarmupSeedError(f"{name} must be finite")
    return parsed


def _parse_datetime(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise WarmupSeedError("warm-up seed datetime is invalid") from exc
    if parsed.tzinfo is None:
        raise WarmupSeedError("warm-up seed datetime must be timezone-aware")
    return parsed.astimezone(timezone.utc)
