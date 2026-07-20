"""Collect one synchronized, hash-chained Bitget CarryFlow evidence tape."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RUNTIME = SRC / "panteon_runtime"
for path in (SRC, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bitget_funding import BitgetFundingDataFetcher  # noqa: E402
from panteon_v2.policy.evidence_tape import (  # noqa: E402
    TAPE_SCHEMA_VERSION,
    CarryFlowEvidenceTape,
    append_tape_sample,
)
from panteon_v2.policy.warmup_seed import (  # noqa: E402
    WARMUP_SEED_SCHEMA_VERSION,
    CarryFlowWarmupSeed,
    write_warmup_seed,
)
from panteon_v2.policy.derivatives_context import (  # noqa: E402
    canonical_derivatives_symbol,
)


DEFAULT_SYMBOLS = ("BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK")
DEFAULT_DIR = ROOT / "Retrodate" / "bitget_carryflow_tape"
_TIMEFRAMES = {
    60: "1m",
    300: "5m",
    900: "15m",
    1800: "30m",
    3600: "1h",
    14_400: "4h",
}


class CollectorLock:
    def __init__(self, output: Path) -> None:
        self.path = output.with_suffix(output.suffix + ".lock")
        self.token = f"{os.getpid()}:{uuid.uuid4().hex}"

    def __enter__(self) -> "CollectorLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                self.path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
        except FileExistsError as exc:
            if not self._remove_stale_lock():
                raise RuntimeError(
                    f"another tape collector owns lock: {self.path}"
                ) from exc
            descriptor = os.open(
                self.path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
        try:
            os.write(descriptor, self.token.encode("ascii"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        try:
            if self.path.read_text(encoding="ascii").strip() == self.token:
                self.path.unlink()
        except FileNotFoundError:
            pass

    def _remove_stale_lock(self) -> bool:
        try:
            raw = self.path.read_text(encoding="ascii").strip()
            pid = int(raw.split(":", 1)[0])
        except (OSError, ValueError):
            return False
        if _pid_exists(pid):
            return False
        try:
            self.path.unlink()
        except OSError:
            return False
        return True


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            process_query_limited_information,
            False,
            pid,
        )
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
            return True
        error = ctypes.windll.kernel32.GetLastError()  # type: ignore[attr-defined]
        return error == 5
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _git_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    revision = result.stdout.strip().lower()
    if result.returncode != 0 or len(revision) < 7:
        raise RuntimeError("git revision unavailable")
    return revision


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _market_symbol(symbol: str) -> str:
    return f"{canonical_derivatives_symbol(symbol)}/USDT:USDT"


def expected_bar_close_ms(now: datetime, bar_interval_seconds: int) -> int:
    timestamp_ms = int(now.astimezone(timezone.utc).timestamp() * 1000)
    interval_ms = int(bar_interval_seconds) * 1000
    return timestamp_ms - timestamp_ms % interval_ms


def intended_first_bar_close_ms(
    *,
    now: datetime,
    bar_interval_seconds: int,
    alignment_delay_seconds: int,
    align: bool,
) -> int:
    close_ms = expected_bar_close_ms(now, bar_interval_seconds)
    if not align:
        return close_ms
    close_at = close_ms / 1000.0 + int(alignment_delay_seconds)
    if close_at <= now.astimezone(timezone.utc).timestamp():
        close_ms += int(bar_interval_seconds) * 1000
    return close_ms


def fetch_warmup_bars(
    *,
    exchange: Any,
    symbols: Sequence[str],
    bar_interval_seconds: int,
    intended_first_evidence_bar_close_timestamp_ms: int,
    warmup_bars: int,
) -> list[dict[str, Any]]:
    if warmup_bars < 1:
        raise ValueError("warmup_bars must be at least 1")
    if warmup_bars > 500:
        raise ValueError("warmup_bars cannot exceed 500")
    timeframe = _TIMEFRAMES.get(int(bar_interval_seconds))
    if timeframe is None:
        raise ValueError("unsupported bar interval for Bitget OHLCV")
    interval_ms = int(bar_interval_seconds) * 1000
    first_start = (
        int(intended_first_evidence_bar_close_timestamp_ms)
        - (warmup_bars + 1) * interval_ms
    )
    expected_starts = tuple(
        first_start + index * interval_ms for index in range(warmup_bars)
    )
    query_start = first_start - interval_ms
    rows_by_symbol: dict[str, dict[int, Sequence[Any]]] = {}
    for raw_symbol in symbols:
        symbol = canonical_derivatives_symbol(raw_symbol)
        selected: dict[int, Sequence[Any]] = {}
        for attempt in range(3):
            try:
                rows = exchange.fetch_ohlcv(
                    _market_symbol(symbol),
                    timeframe=timeframe,
                    since=query_start,
                    limit=warmup_bars + 4,
                ) or []
            except Exception:
                rows = []
            selected = {
                int(_finite(row[0])): row
                for row in rows
                if isinstance(row, (list, tuple))
                and len(row) >= 6
                and int(_finite(row[0])) in expected_starts
            }
            if all(timestamp in selected for timestamp in expected_starts):
                break
            if attempt < 2:
                time.sleep(0.5)
        missing = [
            timestamp for timestamp in expected_starts if timestamp not in selected
        ]
        if missing:
            raise RuntimeError(
                f"warm-up OHLCV incomplete for {symbol}: "
                f"missing {len(missing)}/{warmup_bars} closed bars"
            )
        rows_by_symbol[symbol] = selected
    result: list[dict[str, Any]] = []
    normalized_symbols = [canonical_derivatives_symbol(item) for item in symbols]
    for candle_start in expected_starts:
        symbol_rows: dict[str, dict[str, Any]] = {}
        for symbol in normalized_symbols:
            row = rows_by_symbol[symbol][candle_start]
            symbol_rows[symbol] = {
                "candle_timestamp_ms": candle_start,
                "open": _finite(row[1]),
                "high": _finite(row[2]),
                "low": _finite(row[3]),
                "close": _finite(row[4]),
                "volume": max(0.0, _finite(row[5])),
            }
        result.append(
            {
                "bar_close_timestamp_ms": candle_start + interval_ms,
                "symbols": symbol_rows,
            }
        )
    return result


def build_warmup_seed_payload(
    *,
    collector_run_id: str,
    source_revision: str,
    collector_code_sha256: str,
    created_at: datetime,
    bar_interval_seconds: int,
    intended_first_evidence_bar_close_timestamp_ms: int,
    symbols: Sequence[str],
    bars: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": WARMUP_SEED_SCHEMA_VERSION,
        "collector_run_id": collector_run_id,
        "source_revision": source_revision,
        "collector_code_sha256": collector_code_sha256,
        "source_exchange": "BITGET",
        "market_type": "swap",
        "created_at": created_at.astimezone(timezone.utc).isoformat(),
        "bar_interval_seconds": int(bar_interval_seconds),
        "intended_first_evidence_bar_close_timestamp_ms": int(
            intended_first_evidence_bar_close_timestamp_ms
        ),
        "symbol_set": [canonical_derivatives_symbol(item) for item in symbols],
        "bars": [dict(row) for row in bars],
        "seed_sha256": "0" * 64,
    }


def fetch_closed_candles(
    *,
    exchange: Any,
    symbols: Sequence[str],
    bar_interval_seconds: int,
    bar_close_timestamp_ms: int,
) -> dict[str, dict[str, Any]]:
    timeframe = _TIMEFRAMES.get(int(bar_interval_seconds))
    if timeframe is None:
        raise ValueError("unsupported bar interval for Bitget OHLCV")
    expected_start = bar_close_timestamp_ms - int(bar_interval_seconds) * 1000
    result: dict[str, dict[str, Any]] = {}
    for raw_symbol in symbols:
        symbol = canonical_derivatives_symbol(raw_symbol)
        selected = None
        for attempt in range(3):
            try:
                rows = exchange.fetch_ohlcv(
                    _market_symbol(symbol),
                    timeframe=timeframe,
                    since=expected_start,
                    limit=2,
                ) or []
            except Exception:
                rows = []
            selected = next(
                (
                    row
                    for row in rows
                    if isinstance(row, (list, tuple))
                    and len(row) >= 6
                    and int(_finite(row[0])) == expected_start
                ),
                None,
            )
            if selected is not None:
                break
            if attempt < 2:
                time.sleep(0.5)
        if selected is None:
            result[symbol] = {}
            continue
        result[symbol] = {
            "candle_timestamp_ms": int(_finite(selected[0])),
            "open": _finite(selected[1]),
            "high": _finite(selected[2]),
            "low": _finite(selected[3]),
            "close": _finite(selected[4]),
            "volume": max(0.0, _finite(selected[5])),
        }
    return result


def build_tape_payload(
    *,
    collector_run_id: str,
    source_revision: str,
    collector_code_sha256: str,
    funding_fetcher_code_sha256: str,
    observed_at: datetime,
    bar_interval_seconds: int,
    bar_close_timestamp_ms: int,
    max_bar_close_lag_seconds: int,
    max_derivatives_age_seconds: int,
    symbols: Sequence[str],
    candles: Mapping[str, Mapping[str, Any]],
    derivatives: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    observed_at = observed_at.astimezone(timezone.utc)
    observed_at_ms = int(observed_at.timestamp() * 1000)
    lag_seconds = (observed_at_ms - int(bar_close_timestamp_ms)) / 1000.0
    normalized_candles = {
        canonical_derivatives_symbol(symbol): dict(row)
        for symbol, row in candles.items()
    }
    normalized_derivatives = {
        canonical_derivatives_symbol(symbol): dict(row)
        for symbol, row in derivatives.items()
    }
    symbol_set = [canonical_derivatives_symbol(item) for item in symbols]
    rows: dict[str, dict[str, Any]] = {}
    complete_symbols: list[str] = []
    incomplete_symbols: list[str] = []
    expected_start = (
        int(bar_close_timestamp_ms) - int(bar_interval_seconds) * 1000
    )
    for symbol in symbol_set:
        candle = normalized_candles.get(symbol, {})
        context = normalized_derivatives.get(symbol, {})
        candle_timestamp_ms = int(_finite(candle.get("candle_timestamp_ms")))
        open_price = _finite(candle.get("open"))
        high = _finite(candle.get("high"))
        low = _finite(candle.get("low"))
        close = _finite(candle.get("close"))
        volume = max(0.0, _finite(candle.get("volume")))
        decision_price = _finite(context.get("last_price"))
        market_complete = bool(
            candle_timestamp_ms == expected_start
            and open_price > 0.0
            and high >= low > 0.0
            and low <= open_price <= high
            and low <= close <= high
            and close > 0.0
            and decision_price > 0.0
        )
        source_updated_ts = _finite(context.get("updated_ts"))
        raw_context_age = observed_at.timestamp() - source_updated_ts
        context_age = 0.0 if -1.0 <= raw_context_age < 0.0 else raw_context_age
        derivatives_complete = bool(
            context.get("context_complete")
            and 0.0 <= context_age <= max_derivatives_age_seconds
        )
        reasons: list[str] = []
        if not market_complete:
            reasons.append("market_bar_missing_or_misaligned")
        if not context.get("context_complete"):
            reasons.append("derivatives_context_incomplete")
        elif not 0.0 <= context_age <= max_derivatives_age_seconds:
            reasons.append("derivatives_context_stale")
        complete = market_complete and derivatives_complete
        rows[symbol] = {
            "market": {
                "market_symbol": _market_symbol(symbol),
                "candle_timestamp_ms": candle_timestamp_ms,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "decision_price": decision_price,
                "complete": market_complete,
            },
            "derivatives": {
                "funding_rate": _finite(context.get("funding_rate")),
                "open_interest_usdt": _finite(
                    context.get("open_interest_usdt")
                ),
                "long_ratio": _finite(context.get("long_ratio")),
                "short_ratio": _finite(context.get("short_ratio")),
                "mark_price": _finite(context.get("mark_price")),
                "index_price": _finite(context.get("index_price")),
                "last_price": _finite(context.get("last_price")),
                "next_funding_ts": int(_finite(context.get("next_funding_ts"))),
                "long_short_ratio_ts": int(
                    _finite(context.get("long_short_ratio_ts"))
                ),
                "long_short_ratio_age_sec": _finite(
                    context.get("long_short_ratio_age_sec")
                ),
                "long_short_source": str(
                    context.get("long_short_source") or "missing"
                ),
                "source_updated_ts": source_updated_ts,
                "age_sec": context_age,
                "context_complete": derivatives_complete,
            },
            "complete": complete,
            "reasons": reasons,
        }
        if complete:
            complete_symbols.append(symbol)
        else:
            incomplete_symbols.append(symbol)
    lag_ok = 0.0 <= lag_seconds <= max_bar_close_lag_seconds
    sample_reasons: list[str] = []
    if incomplete_symbols:
        sample_reasons.append("incomplete_symbols")
    if not lag_ok:
        sample_reasons.append("bar_close_lag_exceeded")
    return {
        "schema_version": TAPE_SCHEMA_VERSION,
        "collector_run_id": collector_run_id,
        "source_revision": source_revision,
        "collector_code_sha256": collector_code_sha256,
        "funding_fetcher_code_sha256": funding_fetcher_code_sha256,
        "source_exchange": "BITGET",
        "market_type": "swap",
        "observed_at": observed_at.isoformat(),
        "observed_at_ms": observed_at_ms,
        "bar_interval_seconds": int(bar_interval_seconds),
        "bar_close_timestamp_ms": int(bar_close_timestamp_ms),
        "collection_lag_seconds": lag_seconds,
        "max_bar_close_lag_seconds": int(max_bar_close_lag_seconds),
        "max_derivatives_age_seconds": int(max_derivatives_age_seconds),
        "symbol_set": symbol_set,
        "symbols": rows,
        "complete_symbols": complete_symbols,
        "incomplete_symbols": incomplete_symbols,
        "sample_complete": not incomplete_symbols and lag_ok,
        "sample_reasons": sample_reasons,
        "previous_sample_sha256": "0" * 64,
    }


def seconds_until_aligned_collection(
    *,
    now: datetime,
    bar_interval_seconds: int,
    alignment_delay_seconds: int,
) -> float:
    current = now.astimezone(timezone.utc).timestamp()
    interval = int(bar_interval_seconds)
    boundary = math.floor(current / interval) * interval
    target = boundary + int(alignment_delay_seconds)
    if target <= current:
        target += interval
    return max(0.0, target - current)


def write_collector_status(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "schema_version": "panteon.bitget_carryflow_collector_status.v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "orders_enabled": False,
        **dict(payload),
    }
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def collect_tape_samples(
    *,
    fetcher: Any,
    symbols: Sequence[str],
    output: str | Path,
    samples: int,
    bar_interval_seconds: int,
    max_bar_close_lag_seconds: int,
    max_derivatives_age_seconds: int,
    alignment_delay_seconds: int,
    align: bool,
    collector_run_id: str,
    source_revision: str,
    collector_code_sha256: str | None = None,
    funding_fetcher_code_sha256: str | None = None,
    now_fn: Callable[[], datetime] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    progress_fn: Callable[[Mapping[str, Any]], None] | None = None,
    status_path: str | Path | None = None,
) -> dict[str, Any]:
    if samples < 1:
        raise ValueError("samples must be at least 1")
    if bar_interval_seconds not in _TIMEFRAMES:
        raise ValueError("unsupported bar interval")
    now_fn = now_fn or (lambda: datetime.now(timezone.utc))
    collector_code_sha256 = collector_code_sha256 or _sha256_file(
        Path(__file__).resolve()
    )
    funding_fetcher_code_sha256 = funding_fetcher_code_sha256 or _sha256_file(
        RUNTIME / "bitget_funding.py"
    )
    output_path = Path(output)
    complete_samples = 0
    first_observed_at = ""
    last_observed_at = ""
    with CollectorLock(output_path):
        for index in range(samples):
            if align:
                wait_started_at = now_fn()
                wait = seconds_until_aligned_collection(
                    now=wait_started_at,
                    bar_interval_seconds=bar_interval_seconds,
                    alignment_delay_seconds=alignment_delay_seconds,
                )
                if status_path is not None:
                    write_collector_status(
                        status_path,
                        {
                            "run_state": "waiting_for_bar_close",
                            "collector_run_id": collector_run_id,
                            "output": str(output_path.resolve()),
                            "sample_index": index,
                            "samples_target": samples,
                            "next_collection_at": (
                                wait_started_at.astimezone(timezone.utc)
                                + timedelta(seconds=wait)
                            ).isoformat(),
                        },
                    )
                sleep_fn(wait)
            cutoff = now_fn().astimezone(timezone.utc)
            if status_path is not None:
                write_collector_status(
                    status_path,
                    {
                        "run_state": "collecting",
                        "collector_run_id": collector_run_id,
                        "output": str(output_path.resolve()),
                        "sample_index": index + 1,
                        "samples_target": samples,
                    },
                )
            bar_close_ms = expected_bar_close_ms(cutoff, bar_interval_seconds)
            candles = fetch_closed_candles(
                exchange=fetcher.exchange,
                symbols=symbols,
                bar_interval_seconds=bar_interval_seconds,
                bar_close_timestamp_ms=bar_close_ms,
            )
            derivatives = fetcher.fetch_all(symbols)
            incomplete_context = [
                symbol
                for symbol in symbols
                if not derivatives.get(symbol, {}).get("context_complete")
            ]
            if incomplete_context:
                sleep_fn(1.0)
                derivatives.update(fetcher.fetch_all(incomplete_context))
            observed_at = now_fn().astimezone(timezone.utc)
            payload = build_tape_payload(
                collector_run_id=collector_run_id,
                source_revision=source_revision,
                collector_code_sha256=collector_code_sha256,
                funding_fetcher_code_sha256=funding_fetcher_code_sha256,
                observed_at=observed_at,
                bar_interval_seconds=bar_interval_seconds,
                bar_close_timestamp_ms=bar_close_ms,
                max_bar_close_lag_seconds=max_bar_close_lag_seconds,
                max_derivatives_age_seconds=max_derivatives_age_seconds,
                symbols=symbols,
                candles=candles,
                derivatives=derivatives,
            )
            sealed = append_tape_sample(output_path, payload)
            complete_samples += bool(sealed["sample_complete"])
            first_observed_at = first_observed_at or sealed["observed_at"]
            last_observed_at = sealed["observed_at"]
            if progress_fn is not None:
                progress_fn(
                    {
                        "sample": index + 1,
                        "samples": samples,
                        "observed_at": sealed["observed_at"],
                        "bar_close_timestamp_ms": sealed[
                            "bar_close_timestamp_ms"
                        ],
                        "complete": sealed["sample_complete"],
                        "complete_symbols": len(sealed["complete_symbols"]),
                        "sample_sha256": sealed["sample_sha256"],
                    }
                )
            if status_path is not None:
                write_collector_status(
                    status_path,
                    {
                        "run_state": (
                            "completed" if index + 1 == samples else "running"
                        ),
                        "collector_run_id": collector_run_id,
                        "output": str(output_path.resolve()),
                        "sample_index": index + 1,
                        "samples_target": samples,
                        "last_sample_complete": sealed["sample_complete"],
                        "last_complete_symbols": len(sealed["complete_symbols"]),
                        "last_sample_sha256": sealed["sample_sha256"],
                        "last_observed_at": sealed["observed_at"],
                    },
                )
    return {
        "source": "BITGET_PUBLIC",
        "orders_enabled": False,
        "output": str(output_path.resolve()),
        "collector_run_id": collector_run_id,
        "source_revision": source_revision,
        "symbols": list(symbols),
        "bar_interval_seconds": bar_interval_seconds,
        "samples": samples,
        "complete_samples": complete_samples,
        "incomplete_samples": samples - complete_samples,
        "first_observed_at": first_observed_at,
        "last_observed_at": last_observed_at,
    }


def _print_progress(payload: Mapping[str, Any]) -> None:
    print(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect synchronized public Bitget CarryFlow evidence tape."
    )
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--output")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--bar-interval-sec", type=int, default=3600)
    parser.add_argument("--max-bar-close-lag-sec", type=int, default=120)
    parser.add_argument("--max-derivatives-age-sec", type=int, default=1200)
    parser.add_argument("--alignment-delay-sec", type=int, default=30)
    parser.add_argument("--align", action="store_true")
    parser.add_argument("--collector-run-id")
    parser.add_argument("--status-path")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Append to an existing exact-cadence tape using its immutable run "
            "contract. Intended for hourly one-shot scheduled collection."
        ),
    )
    parser.add_argument(
        "--warmup-bars",
        type=int,
        default=53,
        help=(
            "Closed OHLCV bars sealed before collection so indicator warm-up "
            "does not consume the continuous evidence window; use 0 to disable."
        ),
    )
    parser.add_argument("--warmup-seed-output")
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
    if args.resume and not args.output:
        raise ValueError("--resume requires an explicit --output")
    source_revision = _git_revision(ROOT)
    collector_code_sha256 = _sha256_file(Path(__file__).resolve())
    funding_fetcher_code_sha256 = _sha256_file(RUNTIME / "bitget_funding.py")
    requested_run_id = args.collector_run_id or (
        f"carryflow-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    output = Path(args.output) if args.output else (
        DEFAULT_DIR / requested_run_id / "carryflow_evidence_tape.jsonl"
    )
    existing_tape = None
    run_id = requested_run_id
    if args.resume and output.is_file():
        existing_tape = CarryFlowEvidenceTape.from_jsonl(
            output,
            expected_symbols=symbols,
        )
        run_id = existing_tape.collector_run_id
        if args.collector_run_id and args.collector_run_id != run_id:
            raise ValueError("collector-run-id does not match resumed tape")
        if existing_tape.source_revision != source_revision:
            raise ValueError(
                "resumed tape source revision differs; start a new evidence root"
            )
        if existing_tape.collector_code_sha256 != collector_code_sha256:
            raise ValueError(
                "resumed tape collector code differs; start a new evidence root"
            )
        if (
            existing_tape.funding_fetcher_code_sha256
            != funding_fetcher_code_sha256
        ):
            raise ValueError(
                "resumed tape funding fetcher differs; start a new evidence root"
            )
        if existing_tape.bar_interval_seconds != args.bar_interval_sec:
            raise ValueError("resumed tape bar interval differs")
        if existing_tape.max_bar_close_lag_seconds != args.max_bar_close_lag_sec:
            raise ValueError("resumed tape max bar close lag differs")
        if (
            existing_tape.max_derivatives_age_seconds
            != args.max_derivatives_age_sec
        ):
            raise ValueError("resumed tape derivatives age limit differs")
    status_path = Path(args.status_path) if args.status_path else (
        output.parent / "collector_status.json"
    )
    warmup_seed_path = (
        Path(args.warmup_seed_output)
        if args.warmup_seed_output
        else output.parent / "carryflow_warmup_seed.json"
    )
    fetcher = BitgetFundingDataFetcher(symbols=symbols)
    warmup_seed_description: dict[str, Any] | None = None
    try:
        if args.warmup_bars < 0:
            raise ValueError("warmup-bars must be non-negative")
        if existing_tape is not None:
            if warmup_seed_path.is_file():
                existing_seed = CarryFlowWarmupSeed.from_json(
                    warmup_seed_path,
                    expected_symbols=symbols,
                )
                existing_seed.validate_for_tape(existing_tape)
                warmup_seed_description = existing_seed.describe()
            elif args.warmup_bars:
                raise ValueError(
                    "resumed tape has no prospective warm-up seed; use "
                    "--warmup-bars 0 or start a new evidence root"
                )
        elif args.warmup_bars:
            seed_created_at = datetime.now(timezone.utc)
            intended_close_ms = intended_first_bar_close_ms(
                now=seed_created_at,
                bar_interval_seconds=args.bar_interval_sec,
                alignment_delay_seconds=args.alignment_delay_sec,
                align=args.align,
            )
            warmup_rows = fetch_warmup_bars(
                exchange=fetcher.exchange,
                symbols=symbols,
                bar_interval_seconds=args.bar_interval_sec,
                intended_first_evidence_bar_close_timestamp_ms=intended_close_ms,
                warmup_bars=args.warmup_bars,
            )
            warmup_payload = build_warmup_seed_payload(
                collector_run_id=run_id,
                source_revision=source_revision,
                collector_code_sha256=collector_code_sha256,
                created_at=seed_created_at,
                bar_interval_seconds=args.bar_interval_sec,
                intended_first_evidence_bar_close_timestamp_ms=intended_close_ms,
                symbols=symbols,
                bars=warmup_rows,
            )
            write_warmup_seed(warmup_seed_path, warmup_payload)
            warmup_seed_description = CarryFlowWarmupSeed.from_json(
                warmup_seed_path,
                expected_symbols=symbols,
            ).describe()
            if args.align:
                collection_deadline = (
                    intended_close_ms / 1000.0 + args.alignment_delay_sec
                )
                if datetime.now(timezone.utc).timestamp() >= collection_deadline:
                    raise RuntimeError(
                        "warm-up fetch missed the intended first evidence bar; "
                        "start a new evidence root"
                    )
        summary = collect_tape_samples(
            fetcher=fetcher,
            symbols=symbols,
            output=output,
            samples=args.samples,
            bar_interval_seconds=args.bar_interval_sec,
            max_bar_close_lag_seconds=args.max_bar_close_lag_sec,
            max_derivatives_age_seconds=args.max_derivatives_age_sec,
            alignment_delay_seconds=args.alignment_delay_sec,
            align=args.align,
            collector_run_id=run_id,
            source_revision=source_revision,
            collector_code_sha256=collector_code_sha256,
            funding_fetcher_code_sha256=funding_fetcher_code_sha256,
            progress_fn=_print_progress,
            status_path=status_path,
        )
        summary["warmup_seed"] = warmup_seed_description
        summary["resumed"] = existing_tape is not None
        completed_tape = CarryFlowEvidenceTape.from_jsonl(
            output,
            expected_symbols=symbols,
        )
        summary["tape"] = completed_tape.describe()
        if args.resume and args.samples == 1:
            last_sample = completed_tape.samples[-1]
            next_close_ms = (
                int(last_sample["bar_close_timestamp_ms"])
                + args.bar_interval_sec * 1000
            )
            next_collection_at = datetime.fromtimestamp(
                next_close_ms / 1000.0 + max(1, args.alignment_delay_sec),
                timezone.utc,
            ).isoformat()
            write_collector_status(
                status_path,
                {
                    "run_state": "scheduled_idle",
                    "collection_mode": "scheduled_one_shot",
                    "collector_run_id": run_id,
                    "output": str(output.resolve()),
                    "sample_index": len(completed_tape.samples),
                    "samples_target": 0,
                    "next_collection_at": next_collection_at,
                    "last_sample_complete": last_sample["sample_complete"],
                    "last_complete_symbols": len(
                        last_sample["complete_symbols"]
                    ),
                    "last_sample_sha256": last_sample["sample_sha256"],
                    "last_observed_at": last_sample["observed_at"],
                    "warmup_seed_sha256": (
                        warmup_seed_description["seed_sha256"]
                        if warmup_seed_description
                        else ""
                    ),
                },
            )
    except Exception as exc:
        write_collector_status(
            status_path,
            {
                "run_state": "failed",
                "collector_run_id": run_id,
                "output": str(output.resolve()),
                "warmup_seed_output": str(warmup_seed_path.resolve()),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["complete_samples"] == summary["samples"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
