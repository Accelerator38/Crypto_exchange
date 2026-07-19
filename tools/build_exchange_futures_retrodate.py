from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


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

BITGET_V3_HISTORY_ENDPOINT = "https://api.bitget.com/api/v3/market/history-candles"
BITGET_MAX_QUERY_SPAN_MS = 90 * 24 * 60 * 60 * 1000
TIMEFRAME_MS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


class _RequestRateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        self._interval = 1.0 / max(1.0, float(requests_per_second))
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            scheduled = max(now, self._next_at)
            self._next_at = scheduled + self._interval
        delay = scheduled - now
        if delay > 0.0:
            time.sleep(delay)


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


def fetch_bitget_v3_history_rows(
    *,
    symbols: Sequence[str],
    start: date,
    end: date,
    timeframe: str,
    limit: int = 100,
    sleep_ms: int = 75,
    max_retries: int = 5,
    parallel_workers: int = 1,
    request_get: Callable[..., Any] | None = None,
    _rate_limiter: _RequestRateLimiter | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Download closed Bitget futures candles by paginating backwards.

    Bitget's v3 history endpoint returns the latest page before ``endTime``.
    Each request is kept inside the documented 90-day maximum range.  The
    endpoint is public and never receives credentials or order permissions.
    """
    workers = max(1, min(int(parallel_workers), len(symbols) or 1))
    if workers > 1 and len(symbols) > 1 and request_get is None:
        limiter = _rate_limiter or _RequestRateLimiter(18.0)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    fetch_bitget_v3_history_rows,
                    symbols=(symbol,),
                    start=start,
                    end=end,
                    timeframe=timeframe,
                    limit=limit,
                    sleep_ms=sleep_ms,
                    max_retries=max_retries,
                    parallel_workers=1,
                    _rate_limiter=limiter,
                )
                for symbol in symbols
            ]
            results = [future.result() for future in futures]
        rows = [row for result_rows, _report in results for row in result_rows]
        reports = [report for _result_rows, report in results]
        return rows, {
            "endpoint": BITGET_V3_HISTORY_ENDPOINT,
            "public_only": True,
            "category": "USDT-FUTURES",
            "interval": reports[0]["interval"] if reports else timeframe,
            "page_limit": min(100, max(1, int(limit))),
            "max_query_span_days": 90,
            "parallel_workers": workers,
            "global_rate_limit_requests_per_second": 18.0,
            "requests": sum(int(report.get("requests", 0)) for report in reports),
            "per_symbol": {
                label: payload
                for report in reports
                for label, payload in dict(report.get("per_symbol", {})).items()
            },
        }

    normalized_timeframe = str(timeframe or "").strip().lower()
    if normalized_timeframe not in TIMEFRAME_MS:
        raise ValueError(f"unsupported Bitget timeframe: {timeframe!r}")
    interval = (
        normalized_timeframe[:-1] + "H"
        if normalized_timeframe.endswith("h")
        else normalized_timeframe[:-1] + "D"
        if normalized_timeframe.endswith("d")
        else normalized_timeframe
    )
    page_limit = min(100, max(1, int(limit)))
    start_ms = int(
        datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp()
        * 1000
    )
    end_exclusive_ms = int(
        (
            datetime(end.year, end.month, end.day, tzinfo=timezone.utc)
            + timedelta(days=1)
        ).timestamp()
        * 1000
    )
    if request_get is None:
        try:
            import requests  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("requests is required to download Bitget history") from exc
        session = requests.Session()
        request_get = session.get

    all_rows: list[dict[str, object]] = []
    request_count = 0
    per_symbol: dict[str, dict[str, object]] = {}
    for raw_symbol in symbols:
        normalized_symbol = normalize_symbol(raw_symbol)
        api_symbol = normalized_symbol.replace("/", "")
        # Bitget treats an interval-aligned endTime as a strict upper bound.
        # Passing the exact next boundary includes the final requested candle;
        # subtracting 1 ms is rounded down and skips that candle.
        cursor_end = end_exclusive_ms
        keyed_rows: dict[int, dict[str, object]] = {}
        symbol_requests = 0
        while cursor_end >= start_ms:
            window_start = max(
                start_ms,
                cursor_end - BITGET_MAX_QUERY_SPAN_MS + 1,
            )
            params = {
                "category": "USDT-FUTURES",
                "symbol": api_symbol,
                "interval": interval,
                "startTime": str(window_start),
                "endTime": str(cursor_end),
                "type": "market",
                "limit": str(page_limit),
            }
            payload: Mapping[str, Any] | None = None
            last_error: Exception | None = None
            for attempt in range(max(1, int(max_retries))):
                request_count += 1
                symbol_requests += 1
                try:
                    if _rate_limiter is not None:
                        _rate_limiter.wait()
                    response = request_get(
                        BITGET_V3_HISTORY_ENDPOINT,
                        params=params,
                        timeout=30,
                    )
                    raise_for_status = getattr(response, "raise_for_status", None)
                    if callable(raise_for_status):
                        raise_for_status()
                    raw_payload = response.json()
                    if not isinstance(raw_payload, Mapping):
                        raise RuntimeError("Bitget returned a non-object payload")
                    if str(raw_payload.get("code") or "") != "00000":
                        raise RuntimeError(
                            "Bitget history error "
                            f"{raw_payload.get('code')}: {raw_payload.get('msg')}"
                        )
                    payload = raw_payload
                    break
                except Exception as exc:  # network/runtime specific
                    last_error = exc
                    if attempt + 1 >= max(1, int(max_retries)):
                        break
                    time.sleep(max(0.25, sleep_ms / 1000.0) * (2 ** attempt))
            if payload is None:
                raise RuntimeError(
                    f"Bitget history download failed for {api_symbol}: {last_error}"
                )
            raw_batch = payload.get("data")
            batch = raw_batch if isinstance(raw_batch, list) else []
            valid_timestamps: list[int] = []
            for item in batch:
                if not isinstance(item, (list, tuple)) or len(item) < 6:
                    continue
                try:
                    timestamp = int(item[0])
                except (TypeError, ValueError):
                    continue
                valid_timestamps.append(timestamp)
                if start_ms <= timestamp < end_exclusive_ms:
                    keyed_rows[timestamp] = normalize_ohlcv_row(
                        exchange="BITGET",
                        symbol=normalized_symbol,
                        ohlcv=item,
                    )
            if not valid_timestamps:
                if window_start <= start_ms:
                    break
                cursor_end = window_start
            else:
                earliest = min(valid_timestamps)
                if earliest <= start_ms:
                    break
                # The next page must end exactly at the earliest candle.  The
                # endpoint returns candles strictly before this boundary.
                cursor_end = earliest if earliest < cursor_end else window_start
            if sleep_ms > 0:
                time.sleep(float(sleep_ms) / 1000.0)
            if symbol_requests % 100 == 0:
                print(
                    f"[{api_symbol}] requests={symbol_requests} rows={len(keyed_rows)}",
                    file=sys.stderr,
                    flush=True,
                )
        rows = [keyed_rows[key] for key in sorted(keyed_rows)]
        all_rows.extend(rows)
        per_symbol[normalized_symbol] = {
            "rows": len(rows),
            "requests": symbol_requests,
            "first_timestamp": int(rows[0]["timestamp"]) if rows else None,
            "last_timestamp": int(rows[-1]["timestamp"]) if rows else None,
        }
    return all_rows, {
        "endpoint": BITGET_V3_HISTORY_ENDPOINT,
        "public_only": True,
        "category": "USDT-FUTURES",
        "interval": interval,
        "page_limit": page_limit,
        "max_query_span_days": 90,
        "requests": request_count,
        "per_symbol": per_symbol,
    }


def build_integrity_manifest(
    output_dir: str | Path,
    *,
    exchange: str,
    symbols: Sequence[str],
    start: date,
    end: date,
    timeframe: str,
    source: Mapping[str, Any],
) -> dict[str, object]:
    """Describe persisted coverage and hashes, without trusting fetch counters."""
    output = Path(output_dir)
    step_ms = TIMEFRAME_MS[str(timeframe or "").strip().lower()]
    requested_start_ms = int(
        datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp()
        * 1000
    )
    requested_end_exclusive_ms = int(
        (
            datetime(end.year, end.month, end.day, tzinfo=timezone.utc)
            + timedelta(days=1)
        ).timestamp()
        * 1000
    )
    files: list[dict[str, object]] = []
    timestamps_by_symbol: dict[str, list[int]] = {
        normalize_symbol(symbol): [] for symbol in symbols
    }
    duplicate_rows = 0
    seen: set[tuple[int, str, str]] = set()
    for path in sorted(output.glob("crypto_1m_*_all_symbols.csv")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        row_count = 0
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("source_exchange") or "").strip().upper() != str(exchange).upper():
                    continue
                try:
                    timestamp = int(row.get("timestamp") or 0)
                except (TypeError, ValueError):
                    continue
                symbol = normalize_symbol(str(row.get("symbol") or ""))
                key = (timestamp, symbol, str(exchange).upper())
                if key in seen:
                    duplicate_rows += 1
                    continue
                seen.add(key)
                timestamps_by_symbol.setdefault(symbol, []).append(timestamp)
                row_count += 1
        files.append({
            "path": str(path),
            "bytes": path.stat().st_size,
            "rows": row_count,
            "sha256": digest,
        })

    coverage: dict[str, dict[str, object]] = {}
    total_missing_bars = 0
    total_off_grid = 0
    total_irregular = 0
    for symbol in sorted(timestamps_by_symbol):
        timestamps = sorted(set(timestamps_by_symbol[symbol]))
        missing_bars = 0
        irregular_intervals = 0
        for previous, current in zip(timestamps, timestamps[1:]):
            delta = current - previous
            if delta > step_ms:
                missing_bars += max(0, delta // step_ms - 1)
            if delta <= 0 or delta % step_ms:
                irregular_intervals += 1
        off_grid = sum(1 for timestamp in timestamps if timestamp % step_ms)
        total_missing_bars += missing_bars
        total_off_grid += off_grid
        total_irregular += irregular_intervals
        first = timestamps[0] if timestamps else None
        last = timestamps[-1] if timestamps else None
        expected_inside = (
            ((last - first) // step_ms + 1)
            if first is not None and last is not None
            else 0
        )
        coverage[symbol] = {
            "rows": len(timestamps),
            "first_timestamp": first,
            "first_utc": (
                datetime.fromtimestamp(first / 1000.0, timezone.utc).isoformat()
                if first is not None else None
            ),
            "last_timestamp": last,
            "last_utc": (
                datetime.fromtimestamp(last / 1000.0, timezone.utc).isoformat()
                if last is not None else None
            ),
            "missing_bars_inside_coverage": missing_bars,
            "irregular_intervals": irregular_intervals,
            "off_grid_timestamps": off_grid,
            "coverage_pct_inside_first_last": (
                len(timestamps) / expected_inside * 100.0
                if expected_inside else 0.0
            ),
            "head_missing_bars_from_requested_start": (
                max(0, (first - requested_start_ms) // step_ms)
                if first is not None else None
            ),
            "tail_missing_bars_to_requested_end": (
                max(0, (requested_end_exclusive_ms - step_ms - last) // step_ms)
                if last is not None else None
            ),
        }
    dataset_fingerprint = hashlib.sha256(
        "\n".join(str(item["sha256"]) for item in files).encode("ascii")
    ).hexdigest()
    manifest: dict[str, object] = {
        "schema_version": "panteon.bitget_history_manifest.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "exchange": str(exchange).upper(),
        "market_type": "futures",
        "timeframe": timeframe,
        "requested_start_date": str(start),
        "requested_end_date": str(end),
        "requested_symbols": [normalize_symbol(symbol) for symbol in symbols],
        "source": dict(source),
        "files": files,
        "dataset_sha256": dataset_fingerprint,
        "coverage": coverage,
        "validation": {
            "passed": (
                duplicate_rows == 0
                and total_missing_bars == 0
                and total_off_grid == 0
                and total_irregular == 0
                and all(item["rows"] for item in coverage.values())
            ),
            "duplicate_rows": duplicate_rows,
            "missing_bars_inside_coverage": total_missing_bars,
            "off_grid_timestamps": total_off_grid,
            "irregular_intervals": total_irregular,
        },
    }
    return manifest


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
    parser.add_argument(
        "--source-api",
        choices=("auto", "ccxt", "bitget-v3"),
        default="auto",
        help="auto uses the public Bitget v3 history endpoint for BITGET.",
    )
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument("--integrity-manifest", default="")
    args = parser.parse_args(argv)

    start = _parse_date(args.start_date)
    end = _parse_date(args.end_date)
    if end < start:
        raise ValueError("end-date must be >= start-date")
    exchange = str(args.exchange).upper()
    symbols = _parse_symbols(args.symbols)
    source_api = (
        "bitget-v3"
        if args.source_api == "auto" and exchange == "BITGET"
        else "ccxt"
        if args.source_api == "auto"
        else args.source_api
    )
    if source_api == "bitget-v3":
        if exchange != "BITGET":
            raise ValueError("bitget-v3 source is valid only for BITGET")
        rows, fetch_report = fetch_bitget_v3_history_rows(
            symbols=symbols,
            start=start,
            end=end,
            timeframe=args.timeframe,
            limit=max(1, int(args.limit)),
            sleep_ms=max(0, int(args.sleep_ms)),
            max_retries=max(1, int(args.max_retries)),
            parallel_workers=max(1, int(args.parallel_workers)),
        )
    else:
        rows = fetch_exchange_ohlcv_rows(
            exchange=exchange,
            symbols=symbols,
            start=start,
            end=end,
            timeframe=args.timeframe,
            limit=max(1, int(args.limit)),
            sleep_ms=max(0, int(args.sleep_ms)),
        )
        fetch_report = {
            "adapter": "ccxt",
            "public_only": True,
            "exchange": exchange,
        }
    counts = write_year_files(args.output_dir, rows)
    manifest = build_integrity_manifest(
        args.output_dir,
        exchange=exchange,
        symbols=symbols,
        start=start,
        end=end,
        timeframe=args.timeframe,
        source=fetch_report,
    )
    manifest_path = (
        Path(args.integrity_manifest)
        if args.integrity_manifest
        else Path(args.output_dir) / "integrity_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "exchange": exchange,
        "symbols": list(symbols),
        "start_date": str(start),
        "end_date": str(end),
        "timeframe": args.timeframe,
        "source_api": source_api,
        "fetch": fetch_report,
        "rows": len(rows),
        "year_counts": counts,
        "output_dir": str(Path(args.output_dir)),
        "integrity_manifest": str(manifest_path),
        "integrity_passed": bool(manifest["validation"]["passed"]),
    }
    report_path = Path(args.report) if args.report else Path(args.output_dir) / "build_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
