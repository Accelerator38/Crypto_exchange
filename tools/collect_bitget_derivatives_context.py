"""Collect public Bitget derivatives context for future exact replay."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RUNTIME = SRC / "panteon_runtime"
for path in (SRC, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bitget_funding import BitgetFundingDataFetcher  # noqa: E402
from panteon_v2.policy.derivatives_context import (  # noqa: E402
    DERIVATIVES_CONTEXT_CSV_FIELDS,
    canonical_derivatives_symbol,
)


DEFAULT_SYMBOLS = ("BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK")
DEFAULT_OUTPUT = (
    ROOT
    / "Retrodate"
    / "bitget_derivatives_context_live"
    / "bitget_derivatives_context.csv"
)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _context_complete(data: Mapping[str, Any]) -> bool:
    return bool(
        data.get("context_complete")
        and "funding_rate" in data
        and _finite(data.get("open_interest_usdt")) > 0.0
        and 0.0 < _finite(data.get("long_ratio")) <= 1.0
        and 0.0 < _finite(data.get("short_ratio")) <= 1.0
        and _finite(data.get("mark_price")) > 0.0
        and _finite(data.get("index_price")) > 0.0
    )


def build_sample_rows(
    *,
    observed_at: datetime,
    symbols: Sequence[str],
    cache: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=timezone.utc)
    observed_at = observed_at.astimezone(timezone.utc)
    timestamp_ms = int(observed_at.timestamp() * 1000)
    normalized_cache = {
        canonical_derivatives_symbol(symbol): dict(data)
        for symbol, data in cache.items()
    }
    rows: list[dict[str, Any]] = []
    for raw_symbol in symbols:
        symbol = canonical_derivatives_symbol(raw_symbol)
        data = normalized_cache.get(symbol, {})
        complete = _context_complete(data)
        status = "complete" if complete else ("incomplete" if data else "missing")
        rows.append(
            {
                "timestamp_ms": timestamp_ms,
                "datetime_utc": observed_at.isoformat(),
                "symbol": symbol,
                "source_exchange": "BITGET",
                "market_type": "swap",
                "funding_rate": data.get("funding_rate", ""),
                "open_interest_usdt": data.get("open_interest_usdt", ""),
                "long_ratio": data.get("long_ratio", ""),
                "short_ratio": data.get("short_ratio", ""),
                "mark_price": data.get("mark_price", ""),
                "index_price": data.get("index_price", ""),
                "last_price": data.get("last_price", ""),
                "next_funding_ts": data.get("next_funding_ts", ""),
                "long_short_ratio_ts": data.get("long_short_ratio_ts", ""),
                "source_updated_ts": data.get("updated_ts", ""),
                "context_complete": "true" if complete else "false",
                "retrieval_status": status,
            }
        )
    return rows


def append_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists() and output.stat().st_size > 0
    if exists:
        with output.open("r", encoding="utf-8-sig", newline="") as handle:
            existing_fields = next(csv.reader(handle), [])
        if tuple(existing_fields) != DERIVATIVES_CONTEXT_CSV_FIELDS:
            raise ValueError("existing derivatives context CSV has incompatible header")
    with output.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DERIVATIVES_CONTEXT_CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()


def collect_samples(
    *,
    fetcher: Any,
    symbols: Sequence[str],
    output: str | Path,
    samples: int,
    interval_sec: float,
    now_fn: Callable[[], datetime] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if samples < 1:
        raise ValueError("samples must be at least 1")
    if interval_sec < 0.0:
        raise ValueError("interval_sec must be non-negative")
    now_fn = now_fn or (lambda: datetime.now(timezone.utc))
    complete_rows = 0
    incomplete_rows = 0
    missing_rows = 0
    first_timestamp = ""
    last_timestamp = ""
    for index in range(samples):
        cache = fetcher.fetch_all(symbols)
        observed_at = now_fn()
        rows = build_sample_rows(
            observed_at=observed_at,
            symbols=symbols,
            cache=cache,
        )
        append_rows(output, rows)
        first_timestamp = first_timestamp or observed_at.isoformat()
        last_timestamp = observed_at.isoformat()
        for row in rows:
            status = row["retrieval_status"]
            complete_rows += status == "complete"
            incomplete_rows += status == "incomplete"
            missing_rows += status == "missing"
        if index + 1 < samples:
            sleep_fn(interval_sec)
    return {
        "source": "BITGET_PUBLIC",
        "orders_enabled": False,
        "output": str(Path(output).resolve()),
        "symbols": list(symbols),
        "samples": samples,
        "rows": samples * len(symbols),
        "complete_rows": complete_rows,
        "incomplete_rows": incomplete_rows,
        "missing_rows": missing_rows,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Append public Bitget funding/OI/account-ratio snapshots for exact replay."
        )
    )
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--interval-sec", type=float, default=60.0)
    args = parser.parse_args(argv)
    symbols = tuple(
        dict.fromkeys(
            canonical_derivatives_symbol(item)
            for item in args.symbols.split(",")
            if canonical_derivatives_symbol(item)
        )
    )
    if not symbols:
        raise ValueError("symbols must be non-empty")
    fetcher = BitgetFundingDataFetcher(symbols=symbols)
    summary = collect_samples(
        fetcher=fetcher,
        symbols=symbols,
        output=args.output,
        samples=args.samples,
        interval_sec=args.interval_sec,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["complete_rows"] == summary["rows"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
