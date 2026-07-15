"""Point-in-time derivatives context for deterministic policy replay."""

from __future__ import annotations

import csv
import math
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


DERIVATIVES_CONTEXT_CSV_FIELDS = (
    "timestamp_ms",
    "datetime_utc",
    "symbol",
    "source_exchange",
    "market_type",
    "funding_rate",
    "open_interest_usdt",
    "long_ratio",
    "short_ratio",
    "mark_price",
    "index_price",
    "last_price",
    "next_funding_ts",
    "long_short_ratio_ts",
    "source_updated_ts",
    "context_complete",
    "retrieval_status",
)

_REQUIRED_CSV_FIELDS = frozenset(
    {
        "timestamp_ms",
        "datetime_utc",
        "symbol",
        "source_exchange",
        "market_type",
        "funding_rate",
        "open_interest_usdt",
        "long_ratio",
        "short_ratio",
        "mark_price",
        "index_price",
        "context_complete",
    }
)


class DerivativesContextError(ValueError):
    """Raised when replay derivatives data is ambiguous or malformed."""


def canonical_derivatives_symbol(value: str) -> str:
    text = str(value or "").strip().upper()
    if ":" in text:
        text = text.split(":", 1)[0]
    if "/" in text:
        text = text.split("/", 1)[0]
    if text.endswith("_USDT"):
        text = text[:-5]
    elif text.endswith("USDT") and len(text) > 4:
        text = text[:-4]
    return text


@dataclass(frozen=True)
class DerivativesContextRecord:
    timestamp: datetime
    symbol: str
    funding_rate: float
    open_interest_usdt: float
    long_ratio: float
    short_ratio: float
    mark_price: float
    index_price: float
    last_price: float
    next_funding_ts: int
    long_short_ratio_ts: int
    source_updated_ts: float
    context_complete: bool

    def actor_payload(self, replay_timestamp: datetime) -> dict[str, Any]:
        age_sec = (replay_timestamp - self.timestamp).total_seconds()
        return {
            "symbol": self.symbol,
            "funding_rate": self.funding_rate,
            "open_interest_usdt": self.open_interest_usdt,
            "long_ratio": self.long_ratio,
            "short_ratio": self.short_ratio,
            "mark_price": self.mark_price,
            "index_price": self.index_price,
            "last_price": self.last_price,
            "next_funding_ts": self.next_funding_ts,
            "long_short_ratio_ts": self.long_short_ratio_ts,
            "source_updated_ts": self.source_updated_ts,
            "updated_ts": self.timestamp.timestamp(),
            "age_sec": age_sec,
            "context_timestamp": self.timestamp.isoformat(),
            "context_complete": self.context_complete,
        }


class HistoricalDerivativesContext:
    """Expose only the latest context record available at each replay bar.

    ``advance`` is monotonic and uses ``record.timestamp <= replay timestamp``.
    This makes future rows inaccessible and turns accidental time reversal into
    an explicit replay failure.
    """

    def __init__(self, records: Iterable[DerivativesContextRecord]) -> None:
        grouped: dict[str, list[DerivativesContextRecord]] = {}
        seen: set[tuple[str, datetime]] = set()
        for record in records:
            key = (record.symbol, record.timestamp)
            if key in seen:
                raise DerivativesContextError(
                    f"duplicate derivatives context row: {record.symbol} "
                    f"{record.timestamp.isoformat()}"
                )
            seen.add(key)
            grouped.setdefault(record.symbol, []).append(record)
        if not grouped:
            raise DerivativesContextError("derivatives context contains no rows")
        for rows in grouped.values():
            rows.sort(key=lambda item: item.timestamp)
        self._records = {symbol: tuple(rows) for symbol, rows in grouped.items()}
        self._timestamps = {
            symbol: tuple(row.timestamp for row in rows)
            for symbol, rows in self._records.items()
        }
        self._current: dict[str, DerivativesContextRecord] = {}
        self._replay_timestamp: datetime | None = None

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        symbols: Iterable[str] | None = None,
    ) -> "HistoricalDerivativesContext":
        source = Path(path)
        if not source.is_file():
            raise DerivativesContextError(
                f"derivatives context CSV not found: {source}"
            )
        selected = (
            {canonical_derivatives_symbol(item) for item in symbols}
            if symbols is not None
            else None
        )
        records: list[DerivativesContextRecord] = []
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            missing = sorted(_REQUIRED_CSV_FIELDS - fields)
            if missing:
                raise DerivativesContextError(
                    "derivatives context CSV missing fields: " + ", ".join(missing)
                )
            for line_number, raw in enumerate(reader, start=2):
                symbol = canonical_derivatives_symbol(raw.get("symbol", ""))
                if not symbol:
                    raise DerivativesContextError(
                        f"empty derivatives context symbol at line {line_number}"
                    )
                if selected is not None and symbol not in selected:
                    continue
                try:
                    record = _parse_record(raw, symbol=symbol)
                except (TypeError, ValueError) as exc:
                    raise DerivativesContextError(
                        f"invalid derivatives context at line {line_number}: {exc}"
                    ) from exc
                records.append(record)
        return cls(records)

    @property
    def row_count(self) -> int:
        return sum(len(rows) for rows in self._records.values())

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self._records))

    @property
    def first_timestamp(self) -> datetime:
        return min(rows[0].timestamp for rows in self._records.values())

    @property
    def last_timestamp(self) -> datetime:
        return max(rows[-1].timestamp for rows in self._records.values())

    def describe(self) -> dict[str, Any]:
        return {
            "rows": self.row_count,
            "symbols": list(self.symbols),
            "first_timestamp": self.first_timestamp.isoformat(),
            "last_timestamp": self.last_timestamp.isoformat(),
        }

    def advance(self, replay_timestamp: datetime) -> None:
        timestamp = _utc_datetime(replay_timestamp)
        if self._replay_timestamp is not None and timestamp < self._replay_timestamp:
            raise DerivativesContextError(
                "derivatives context replay timestamp moved backwards"
            )
        self._replay_timestamp = timestamp
        for symbol, rows in self._records.items():
            index = bisect_right(self._timestamps[symbol], timestamp) - 1
            if index >= 0:
                self._current[symbol] = rows[index]
            else:
                self._current.pop(symbol, None)

    def get(self, symbol: str) -> dict[str, Any]:
        if self._replay_timestamp is None:
            raise DerivativesContextError(
                "advance(replay_timestamp) must be called before get(symbol)"
            )
        record = self._current.get(canonical_derivatives_symbol(symbol))
        if record is None:
            return {}
        return record.actor_payload(self._replay_timestamp)

    def get_global(self) -> dict[str, Any]:
        if self._replay_timestamp is None:
            return {}
        payloads = [row.actor_payload(self._replay_timestamp) for row in self._current.values()]
        if not payloads:
            return {
                "timestamp": self._replay_timestamp.timestamp(),
                "total_oi_usdt": 0.0,
                "avg_funding": 0.0,
                "complete_symbols": 0,
            }
        return {
            "timestamp": self._replay_timestamp.timestamp(),
            "total_oi_usdt": sum(row["open_interest_usdt"] for row in payloads),
            "avg_funding": sum(row["funding_rate"] for row in payloads) / len(payloads),
            "complete_symbols": sum(bool(row["context_complete"]) for row in payloads),
        }


def _parse_record(
    raw: Mapping[str, Any],
    *,
    symbol: str,
) -> DerivativesContextRecord:
    timestamp_ms = _required_int(raw.get("timestamp_ms"), "timestamp_ms")
    if timestamp_ms <= 0:
        raise ValueError("timestamp_ms must be positive")
    timestamp = datetime.fromtimestamp(timestamp_ms / 1000.0, timezone.utc)
    declared_timestamp = _parse_datetime(raw.get("datetime_utc"))
    if abs((declared_timestamp - timestamp).total_seconds()) > 0.001:
        raise ValueError("datetime_utc does not match timestamp_ms")
    if str(raw.get("source_exchange") or "").strip().upper() != "BITGET":
        raise ValueError("source_exchange must be BITGET")
    if str(raw.get("market_type") or "").strip().lower() != "swap":
        raise ValueError("market_type must be swap")
    # Missing values remain an explicit incomplete row. They are never
    # promoted to a complete actor context, but retaining them preserves data
    # gap evidence without invalidating unrelated symbols in the same sample.
    funding_rate = _optional_float(raw.get("funding_rate"))
    open_interest = _optional_float(raw.get("open_interest_usdt"))
    long_ratio = _optional_float(raw.get("long_ratio"))
    short_ratio = _optional_float(raw.get("short_ratio"))
    mark_price = _optional_float(raw.get("mark_price"))
    index_price = _optional_float(raw.get("index_price"))
    declared_complete = _parse_bool(raw.get("context_complete"))
    values_complete = bool(
        open_interest > 0.0
        and 0.0 < long_ratio <= 1.0
        and 0.0 < short_ratio <= 1.0
        and mark_price > 0.0
        and index_price > 0.0
    )
    return DerivativesContextRecord(
        timestamp=timestamp,
        symbol=symbol,
        funding_rate=funding_rate,
        open_interest_usdt=open_interest,
        long_ratio=long_ratio,
        short_ratio=short_ratio,
        mark_price=mark_price,
        index_price=index_price,
        last_price=_optional_float(raw.get("last_price")),
        next_funding_ts=_optional_int(raw.get("next_funding_ts")),
        long_short_ratio_ts=_optional_int(raw.get("long_short_ratio_ts")),
        source_updated_ts=_optional_float(raw.get("source_updated_ts")),
        context_complete=declared_complete and values_complete,
    )


def _required_float(value: Any, name: str) -> float:
    if value is None or str(value).strip() == "":
        raise ValueError(f"{name} is required")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _optional_float(value: Any) -> float:
    if value is None or str(value).strip() == "":
        return 0.0
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("optional numeric value must be finite")
    return parsed


def _required_int(value: Any, name: str) -> int:
    parsed = _required_float(value, name)
    if int(parsed) != parsed:
        raise ValueError(f"{name} must be an integer")
    return int(parsed)


def _optional_int(value: Any) -> int:
    if value is None or str(value).strip() == "":
        return 0
    parsed = _optional_float(value)
    if int(parsed) != parsed:
        raise ValueError("optional timestamp must be an integer")
    return int(parsed)


def _parse_bool(value: Any) -> bool:
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError("context_complete must be true or false")


def _parse_datetime(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("datetime_utc is required")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError("datetime_utc must include a timezone")
    return parsed.astimezone(timezone.utc)


def _utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
