from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


RETRODATE_FIELDS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
    "datetime",
    "source_exchange",
    "market_type",
)


def normalize_ohlcv_row(
    *,
    exchange: str,
    symbol: str,
    ohlcv: Sequence[Any],
) -> dict[str, object]:
    timestamp = int(float(ohlcv[0]))
    dt = datetime.fromtimestamp(timestamp / 1000.0, timezone.utc)
    return {
        "timestamp": timestamp,
        "open": str(ohlcv[1]),
        "high": str(ohlcv[2]),
        "low": str(ohlcv[3]),
        "close": str(ohlcv[4]),
        "volume": str(ohlcv[5]),
        "symbol": normalize_symbol(symbol),
        "datetime": dt.strftime("%Y-%m-%d %H:%M:%S+00:00"),
        "source_exchange": str(exchange or "").strip().upper(),
        "market_type": "futures",
    }


def normalize_symbol(symbol: str) -> str:
    clean = str(symbol or "").strip().upper()
    if ":" in clean:
        clean = clean.split(":", 1)[0]
    clean = clean.replace("-", "/").replace("_", "/")
    if "/" in clean:
        base, quote, *_ = clean.split("/")
        return f"{base}/{quote}"
    if clean.endswith("USDT"):
        return f"{clean[:-4]}/USDT"
    if clean.endswith("USDC"):
        return f"{clean[:-4]}/USDC"
    return clean


def write_year_files(output_dir: str | Path, rows: Iterable[Mapping[str, Any]]) -> dict[int, int]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    buckets: dict[int, dict[tuple[int, str, str], dict[str, object]]] = {}
    for row in rows:
        try:
            timestamp = int(row.get("timestamp") or 0)
        except (TypeError, ValueError):
            continue
        if timestamp <= 0:
            continue
        symbol = normalize_symbol(str(row.get("symbol") or ""))
        if not symbol:
            continue
        year = datetime.fromtimestamp(timestamp / 1000.0, timezone.utc).year
        exchange = str(row.get("source_exchange") or "").strip().upper()
        buckets.setdefault(year, {})[(timestamp, symbol, exchange)] = {
            field: row.get(field, "")
            for field in RETRODATE_FIELDS
        } | {"timestamp": timestamp, "symbol": symbol}

    counts: dict[int, int] = {}
    for year, keyed_rows in sorted(buckets.items()):
        path = output / f"crypto_1m_{year}_all_symbols.csv"
        if path.exists():
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for existing in reader:
                    try:
                        timestamp = int(existing.get("timestamp") or 0)
                    except (TypeError, ValueError):
                        continue
                    symbol = normalize_symbol(str(existing.get("symbol") or ""))
                    exchange = str(existing.get("source_exchange") or "").strip().upper()
                    if timestamp <= 0 or not symbol or not exchange:
                        continue
                    keyed_rows.setdefault(
                        (timestamp, symbol, exchange),
                        {
                            field: existing.get(field, "")
                            for field in RETRODATE_FIELDS
                        } | {"timestamp": timestamp, "symbol": symbol},
                    )
        rows_sorted = [
            keyed_rows[key]
            for key in sorted(keyed_rows, key=lambda item: (item[0], item[1], item[2]))
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RETRODATE_FIELDS)
            writer.writeheader()
            writer.writerows(rows_sorted)
        counts[year] = len(rows_sorted)
    return counts


def fetch_exchange_ohlcv_rows(
    *,
    exchange: str,
    symbols: Sequence[str],
    start: date,
    end: date,
    timeframe: str,
    limit: int,
    sleep_ms: int,
) -> list[dict[str, object]]:
    try:
        import ccxt  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("ccxt is required to download futures OHLCV data") from exc

    exchange_id = str(exchange or "").strip().lower()
    exchange_cls = getattr(ccxt, exchange_id)
    client = exchange_cls({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    client.load_markets()
    since = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp() * 1000)
    until = int(
        (
            datetime(end.year, end.month, end.day, tzinfo=timezone.utc)
            + timedelta(days=1)
        ).timestamp()
        * 1000
    )
    rows: list[dict[str, object]] = []
    for raw_symbol in symbols:
        market_symbol = resolve_market_symbol(client.markets, raw_symbol)
        cursor = since
        while cursor < until:
            batch = client.fetch_ohlcv(
                market_symbol,
                timeframe=timeframe,
                since=cursor,
                limit=int(limit),
                params={},
            )
            if not batch:
                break
            advanced = False
            for item in batch:
                ts = int(item[0])
                if ts >= until:
                    continue
                if ts >= cursor:
                    advanced = True
                rows.append(
                    normalize_ohlcv_row(
                        exchange=exchange,
                        symbol=market_symbol,
                        ohlcv=item,
                    )
                )
            last_ts = int(batch[-1][0])
            next_cursor = last_ts + 1
            if not advanced or next_cursor <= cursor:
                break
            cursor = next_cursor
            if sleep_ms > 0:
                time.sleep(float(sleep_ms) / 1000.0)
    return rows


def resolve_market_symbol(markets: Mapping[str, Any], symbol: str) -> str:
    normalized = normalize_symbol(symbol)
    preferred = (
        f"{normalized}:USDT",
        normalized,
        normalized.replace("/", ""),
    )
    for candidate in preferred:
        if candidate in markets:
            return candidate
    for market_symbol in markets:
        if normalize_symbol(market_symbol) == normalized:
            return str(market_symbol)
    raise ValueError(f"futures market not found for {symbol!r}")


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _parse_symbols(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(raw or "").replace(";", ",").split(",") if part.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download public MEXC/BITGET futures OHLCV into Retrodate CSV format."
    )
    parser.add_argument("--exchange", required=True, choices=("MEXC", "BITGET", "mexc", "bitget"))
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g. BTC/USDT,ETH/USDT")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--sleep-ms", type=int, default=250)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", default="")
    args = parser.parse_args(argv)

    start = _parse_date(args.start_date)
    end = _parse_date(args.end_date)
    if end < start:
        raise ValueError("end-date must be >= start-date")
    rows = fetch_exchange_ohlcv_rows(
        exchange=str(args.exchange).upper(),
        symbols=_parse_symbols(args.symbols),
        start=start,
        end=end,
        timeframe=args.timeframe,
        limit=max(1, int(args.limit)),
        sleep_ms=max(0, int(args.sleep_ms)),
    )
    counts = write_year_files(args.output_dir, rows)
    report = {
        "exchange": str(args.exchange).upper(),
        "symbols": list(_parse_symbols(args.symbols)),
        "start_date": str(start),
        "end_date": str(end),
        "timeframe": args.timeframe,
        "rows": len(rows),
        "year_counts": counts,
        "output_dir": str(Path(args.output_dir)),
    }
    report_path = Path(args.report) if args.report else Path(args.output_dir) / "build_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
