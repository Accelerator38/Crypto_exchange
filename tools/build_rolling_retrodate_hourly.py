from __future__ import annotations

import argparse
import csv
import subprocess
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence


DEFAULT_SYMBOLS = (
    "ADA/USDT",
    "APT/USDT",
    "ATOM/USDT",
    "AVAX/USDT",
    "BNB/USDT",
    "BTC/USDT",
    "DOGE/USDT",
    "DOT/USDT",
    "ETC/USDT",
    "ETH/USDT",
    "FIL/USDT",
    "ICP/USDT",
    "LINK/USDT",
    "LTC/USDT",
    "MATIC/USDT",
    "NEAR/USDT",
    "SOL/USDT",
    "TRX/USDT",
    "XLM/USDT",
    "XRP/USDT",
)
BINANCE_SPOT_BASE = "https://data.binance.vision/data/spot"
RETRODATE_FIELDS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
    "datetime",
)
HOUR_MS = 60 * 60 * 1000


@dataclass(frozen=True)
class DownloadPlan:
    source: str
    symbol: str
    url: str
    cache_path: Path


def iter_months(start: date, end: date) -> Iterator[tuple[int, int]]:
    current = date(start.year, start.month, 1)
    final = date(end.year, end.month, 1)
    while current <= final:
        yield current.year, current.month
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


def iter_dates(start: date, end: date) -> Iterator[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def parse_binance_kline_row(symbol: str, row: Sequence[str]) -> dict[str, object] | None:
    if not row or str(row[0]).strip().lower() in {"open_time", "open time"}:
        return None
    try:
        timestamp = normalize_timestamp_ms(int(row[0]))
        open_price = str(row[1])
        high = str(row[2])
        low = str(row[3])
        close = str(row[4])
        volume = str(row[5])
    except (IndexError, TypeError, ValueError):
        return None
    dt = datetime.fromtimestamp(timestamp / 1000.0, timezone.utc)
    return {
        "timestamp": timestamp,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "symbol": normalize_symbol(symbol),
        "datetime": dt.strftime("%Y-%m-%d %H:%M:%S+00:00"),
    }


def normalize_timestamp_ms(raw_timestamp: int) -> int:
    timestamp = int(raw_timestamp)
    # Binance started publishing some daily klines with microsecond timestamps.
    if timestamp >= 10_000_000_000_000:
        timestamp //= 1000
    return timestamp


def normalize_symbol(symbol: str) -> str:
    clean = str(symbol or "").strip().upper().replace("/", "")
    if clean.endswith("USDT"):
        return f"{clean[:-4]}/USDT"
    return clean


def binance_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper().replace("/", "")


def monthly_plan(symbol: str, year: int, month: int, cache_dir: Path) -> DownloadPlan:
    pair = binance_symbol(symbol)
    name = f"{pair}-1m-{year:04d}-{month:02d}.zip"
    return DownloadPlan(
        source="monthly",
        symbol=pair,
        url=f"{BINANCE_SPOT_BASE}/monthly/klines/{pair}/1m/{name}",
        cache_path=cache_dir / "monthly" / pair / name,
    )


def daily_plan(symbol: str, day: date, cache_dir: Path) -> DownloadPlan:
    pair = binance_symbol(symbol)
    name = f"{pair}-1m-{day:%Y-%m-%d}.zip"
    return DownloadPlan(
        source="daily",
        symbol=pair,
        url=f"{BINANCE_SPOT_BASE}/daily/klines/{pair}/1m/{name}",
        cache_path=cache_dir / "daily" / pair / name,
    )


def download_with_curl(plan: DownloadPlan) -> bool:
    if plan.cache_path.exists() and plan.cache_path.stat().st_size > 0:
        return True
    plan.cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = plan.cache_path.with_suffix(plan.cache_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    result = subprocess.run(
        [
            "curl.exe",
            "-L",
            "--fail",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "20",
            "--max-time",
            "180",
            "-o",
            str(tmp),
            plan.url,
        ],
        check=False,
    )
    if result.returncode != 0:
        if tmp.exists():
            tmp.unlink()
        return False
    tmp.replace(plan.cache_path)
    return True


def rows_from_zip(plan: DownloadPlan) -> Iterator[dict[str, object]]:
    with zipfile.ZipFile(plan.cache_path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        if not names:
            return
        with archive.open(names[0]) as handle:
            text = (line.decode("utf-8").strip() for line in handle)
            for row in csv.reader(text):
                parsed = parse_binance_kline_row(plan.symbol, row)
                if parsed is not None:
                    yield parsed


def read_existing_rows(
    data_dir: Path,
    *,
    start: date,
    end: date,
) -> Iterator[dict[str, object]]:
    for year in range(start.year, end.year + 1):
        path = data_dir / f"crypto_1m_{year}_all_symbols.csv"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    timestamp = int(row.get("timestamp") or 0)
                except ValueError:
                    continue
                if timestamp % HOUR_MS != 0:
                    continue
                row_date = datetime.fromtimestamp(
                    timestamp / 1000.0,
                    timezone.utc,
                ).date()
                if start <= row_date <= end:
                    yield {
                        "timestamp": timestamp,
                        "open": row.get("open", ""),
                        "high": row.get("high", ""),
                        "low": row.get("low", ""),
                        "close": row.get("close", ""),
                        "volume": row.get("volume", ""),
                        "symbol": normalize_symbol(str(row.get("symbol") or "")),
                        "datetime": row.get("datetime")
                        or datetime.fromtimestamp(
                            timestamp / 1000.0,
                            timezone.utc,
                        ).strftime("%Y-%m-%d %H:%M:%S+00:00"),
                    }


def add_rows(
    rows_by_year: dict[int, dict[tuple[int, str], dict[str, object]]],
    rows: Iterable[dict[str, object]],
    *,
    start: date,
    end: date,
) -> int:
    added = 0
    for row in rows:
        try:
            timestamp = int(row["timestamp"])
        except (KeyError, TypeError, ValueError):
            continue
        if timestamp % HOUR_MS != 0:
            continue
        row_date = datetime.fromtimestamp(timestamp / 1000.0, timezone.utc).date()
        if not start <= row_date <= end:
            continue
        symbol = normalize_symbol(str(row.get("symbol") or ""))
        if not symbol:
            continue
        bucket = rows_by_year[row_date.year]
        key = (timestamp, symbol)
        if key not in bucket:
            added += 1
        bucket[key] = dict(row, timestamp=timestamp, symbol=symbol)
    return added


def write_year_files(
    output_dir: Path,
    rows_by_year: dict[int, dict[tuple[int, str], dict[str, object]]],
) -> dict[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[int, int] = {}
    for year, rows in sorted(rows_by_year.items()):
        path = output_dir / f"crypto_1m_{year}_all_symbols.csv"
        ordered = [row for _key, row in sorted(rows.items())]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RETRODATE_FIELDS)
            writer.writeheader()
            for row in ordered:
                writer.writerow({field: row.get(field, "") for field in RETRODATE_FIELDS})
        counts[year] = len(ordered)
    return counts


def build_dataset(
    *,
    source_data_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    start: date,
    end: date,
    symbols: Sequence[str],
) -> dict[str, object]:
    rows_by_year: dict[int, dict[tuple[int, str], dict[str, object]]] = defaultdict(dict)
    existing_added = add_rows(
        rows_by_year,
        read_existing_rows(source_data_dir, start=start, end=end),
        start=start,
        end=end,
    )

    downloaded = 0
    missing = 0
    downloaded_rows = 0
    # Fill the historical gap before the local 2022 data starts.
    for symbol in symbols:
        for year, month in iter_months(start, min(end, date(2021, 12, 31))):
            plan = monthly_plan(symbol, year, month, cache_dir)
            if not download_with_curl(plan):
                missing += 1
                continue
            downloaded += 1
            downloaded_rows += add_rows(
                rows_by_year,
                rows_from_zip(plan),
                start=start,
                end=end,
            )

    # Fill the latest local gap after the existing 2026 file.
    if end >= date(2026, 5, 19):
        latest_start = max(start, date(2026, 5, 19))
        for symbol in symbols:
            for day in iter_dates(latest_start, end):
                plan = daily_plan(symbol, day, cache_dir)
                if not download_with_curl(plan):
                    missing += 1
                    continue
                downloaded += 1
                downloaded_rows += add_rows(
                    rows_by_year,
                    rows_from_zip(plan),
                    start=start,
                    end=end,
                )

    counts = write_year_files(output_dir, rows_by_year)
    return {
        "source_data_dir": str(source_data_dir),
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "symbols": [normalize_symbol(symbol) for symbol in symbols],
        "existing_rows_added": existing_added,
        "downloaded_files": downloaded,
        "missing_files": missing,
        "downloaded_rows_added": downloaded_rows,
        "year_counts": counts,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a strict rolling hourly Retrodate dataset from local data plus Binance ZIPs.",
    )
    parser.add_argument("--source-data-dir", default="Retrodate/hourly")
    parser.add_argument("--output-dir", default="Retrodate/rolling_5y_hourly")
    parser.add_argument("--cache-dir", default="Retrodate/binance_cache")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated symbols, e.g. BTC/USDT,ETH/USDT.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    symbols = tuple(item.strip() for item in str(args.symbols).split(",") if item.strip())
    summary = build_dataset(
        source_data_dir=Path(args.source_data_dir),
        output_dir=Path(args.output_dir),
        cache_dir=Path(args.cache_dir),
        start=date.fromisoformat(args.start_date),
        end=date.fromisoformat(args.end_date),
        symbols=symbols,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
