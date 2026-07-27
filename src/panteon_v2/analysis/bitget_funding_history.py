from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


BITGET_FUNDING_HISTORY_ENDPOINT = (
    "https://api.bitget.com/api/v3/market/history-fund-rate"
)
BITGET_INSTRUMENTS_ENDPOINT = "https://api.bitget.com/api/v3/market/instruments"
FUNDING_HISTORY_SCHEMA_VERSION = "panteon.bitget_funding_history.v1"
ALLOWED_FUNDING_INTERVAL_HOURS = frozenset({1, 2, 4, 8})


class FundingHistoryError(ValueError):
    pass


def build_or_verify_funding_snapshot(
    output_dir: str | Path,
    *,
    symbols: Sequence[str],
    required_start_at: str,
    required_end_at: str,
    request_get: Callable[..., Any] | None = None,
    page_limit: int = 100,
    max_pages: int = 100,
    max_retries: int = 4,
    sleep_seconds: float = 0.06,
) -> dict[str, Any]:
    target = Path(output_dir)
    manifest_path = target / "integrity_manifest.json"
    if manifest_path.is_file():
        return verify_funding_snapshot(target, expected_symbols=symbols)
    if target.exists() and any(target.iterdir()):
        raise FundingHistoryError(
            "funding snapshot directory is non-empty but not sealed"
        )
    target.mkdir(parents=True, exist_ok=True)

    if request_get is None:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise FundingHistoryError(
                "requests is required to fetch Bitget funding history"
            ) from exc
        request_get = requests.Session().get

    normalized_symbols = tuple(_normalize_symbol(item) for item in symbols)
    instruments_payload, instrument_requests = _request_payload(
        request_get,
        BITGET_INSTRUMENTS_ENDPOINT,
        params={"category": "USDT-FUTURES"},
        max_retries=max_retries,
    )
    instruments = _parse_instruments(
        instruments_payload,
        expected_symbols=normalized_symbols,
    )
    rows_by_symbol: dict[str, list[dict[str, Any]]] = {}
    request_count = instrument_requests
    per_symbol_requests: dict[str, int] = {}
    limit = min(100, max(1, int(page_limit)))

    for symbol in normalized_symbols:
        keyed: dict[int, dict[str, Any]] = {}
        requests_for_symbol = 0
        page_exhausted = False
        for cursor in range(1, max(1, int(max_pages)) + 1):
            payload, attempts = _request_payload(
                request_get,
                BITGET_FUNDING_HISTORY_ENDPOINT,
                params={
                    "category": "USDT-FUTURES",
                    "symbol": f"{symbol}USDT",
                    "limit": str(limit),
                    "cursor": str(cursor),
                },
                max_retries=max_retries,
            )
            request_count += attempts
            requests_for_symbol += attempts
            batch = _parse_funding_page(payload, expected_symbol=symbol)
            for row in batch:
                timestamp_ms = int(row["funding_time_ms"])
                previous = keyed.get(timestamp_ms)
                if previous is not None and previous != row:
                    raise FundingHistoryError(
                        f"conflicting funding settlement for {symbol} "
                        f"at {timestamp_ms}"
                    )
                keyed[timestamp_ms] = row
            if len(batch) < limit:
                page_exhausted = True
                break
            if sleep_seconds > 0:
                time.sleep(float(sleep_seconds))
        if not page_exhausted:
            raise FundingHistoryError(
                f"funding pagination limit reached for {symbol}"
            )
        rows = [keyed[key] for key in sorted(keyed)]
        if not rows:
            raise FundingHistoryError(f"Bitget returned no funding rows for {symbol}")
        rows_by_symbol[symbol] = rows
        per_symbol_requests[symbol] = requests_for_symbol

    files = _write_symbol_files(target, rows_by_symbol)
    coverage = {
        symbol: _coverage(rows, current_interval_hours=instruments[symbol])
        for symbol, rows in rows_by_symbol.items()
    }
    if not all(item["validation_passed"] for item in coverage.values()):
        raise FundingHistoryError("funding snapshot continuity validation failed")
    dataset_sha = _dataset_sha(files)
    required_start_ms = _timestamp_ms(required_start_at)
    required_end_ms = _timestamp_ms(required_end_at)
    common_start_ms = max(int(item["first_timestamp_ms"]) for item in coverage.values())
    common_end_ms = min(int(item["last_timestamp_ms"]) for item in coverage.values())
    full_registered_window = (
        common_start_ms <= required_start_ms and common_end_ms >= required_end_ms
    )
    manifest = {
        "schema_version": FUNDING_HISTORY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "exchange": "BITGET",
        "category": "USDT-FUTURES",
        "symbols": list(normalized_symbols),
        "required_window": {
            "start_at": required_start_at,
            "end_at": required_end_at,
        },
        "observed_common_window": {
            "start_timestamp_ms": common_start_ms,
            "start_at": _iso_timestamp(common_start_ms),
            "end_timestamp_ms": common_end_ms,
            "end_at": _iso_timestamp(common_end_ms),
        },
        "source": {
            "history_endpoint": BITGET_FUNDING_HISTORY_ENDPOINT,
            "instruments_endpoint": BITGET_INSTRUMENTS_ENDPOINT,
            "public_only": True,
            "page_limit": limit,
            "max_pages": int(max_pages),
            "requests": request_count,
            "per_symbol_requests": per_symbol_requests,
            "observed_rolling_window_only": not full_registered_window,
        },
        "current_interval_hours": instruments,
        "coverage": coverage,
        "files": files,
        "dataset_sha256": dataset_sha,
        "registered_oos_coverage_complete": full_registered_window,
        "allowed_use": (
            "registered_historical_oos"
            if full_registered_window
            else "recent_diagnostic_screen_only"
        ),
        "orders_enabled": False,
        "promotion_authority": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return verify_funding_snapshot(target, expected_symbols=normalized_symbols)


def verify_funding_snapshot(
    snapshot_dir: str | Path,
    *,
    expected_symbols: Sequence[str],
) -> dict[str, Any]:
    target = Path(snapshot_dir)
    manifest_path = target / "integrity_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FundingHistoryError("cannot read funding integrity manifest") from exc
    if manifest.get("schema_version") != FUNDING_HISTORY_SCHEMA_VERSION:
        raise FundingHistoryError("unsupported funding snapshot schema")
    if manifest.get("exchange") != "BITGET":
        raise FundingHistoryError("funding snapshot exchange mismatch")
    if (
        manifest.get("orders_enabled") is not False
        or manifest.get("promotion_authority") is not False
    ):
        raise FundingHistoryError("funding snapshot has trading authority")
    normalized_symbols = tuple(_normalize_symbol(item) for item in expected_symbols)
    if tuple(manifest.get("symbols") or ()) != normalized_symbols:
        raise FundingHistoryError("funding snapshot symbol set mismatch")

    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != len(normalized_symbols):
        raise FundingHistoryError("funding snapshot file list is invalid")
    seen_symbols: set[str] = set()
    verified_files: list[dict[str, Any]] = []
    coverage: dict[str, dict[str, Any]] = {}
    for file_row in files:
        if not isinstance(file_row, Mapping):
            raise FundingHistoryError("funding file manifest row is invalid")
        name = str(file_row.get("path") or "")
        path = target / name
        if Path(name).name != name or not path.is_file():
            raise FundingHistoryError(f"funding snapshot file missing: {name}")
        actual_sha = _sha256_file(path)
        if actual_sha != str(file_row.get("sha256") or "").lower():
            raise FundingHistoryError(f"funding snapshot SHA mismatch: {name}")
        rows = _read_symbol_file(path)
        if not rows:
            raise FundingHistoryError(f"funding snapshot file is empty: {name}")
        symbol = str(rows[0]["symbol"])
        if symbol in seen_symbols or any(row["symbol"] != symbol for row in rows):
            raise FundingHistoryError(f"funding snapshot symbol mismatch: {name}")
        seen_symbols.add(symbol)
        expected_interval = int(
            (manifest.get("current_interval_hours") or {}).get(symbol, 0)
        )
        row_coverage = _coverage(
            rows,
            current_interval_hours=expected_interval,
        )
        if not row_coverage["validation_passed"]:
            raise FundingHistoryError(f"funding continuity invalid: {symbol}")
        coverage[symbol] = row_coverage
        verified_files.append(
            {
                "path": name,
                "bytes": path.stat().st_size,
                "rows": len(rows),
                "sha256": actual_sha,
            }
        )
    if seen_symbols != set(normalized_symbols):
        raise FundingHistoryError("funding snapshot files do not cover full8")
    if _dataset_sha(verified_files) != manifest.get("dataset_sha256"):
        raise FundingHistoryError("funding snapshot dataset SHA mismatch")
    if coverage != manifest.get("coverage"):
        raise FundingHistoryError("funding snapshot coverage metadata mismatch")
    return {
        **manifest,
        "manifest_sha256": _sha256_file(manifest_path),
        "snapshot_dir": str(target.resolve()),
        "verified": True,
    }


def load_funding_snapshot(
    snapshot_dir: str | Path,
    *,
    expected_symbols: Sequence[str],
) -> tuple[dict[str, dict[int, float]], dict[str, Any]]:
    manifest = verify_funding_snapshot(
        snapshot_dir,
        expected_symbols=expected_symbols,
    )
    target = Path(snapshot_dir)
    result: dict[str, dict[int, float]] = {}
    for row in manifest["files"]:
        records = _read_symbol_file(target / row["path"])
        symbol = str(records[0]["symbol"])
        result[symbol] = {
            int(item["funding_time_ms"]): float(item["funding_rate"])
            for item in records
        }
    return result, manifest


def _request_payload(
    request_get: Callable[..., Any],
    endpoint: str,
    *,
    params: Mapping[str, Any],
    max_retries: int,
) -> tuple[Mapping[str, Any], int]:
    last_error: Exception | None = None
    attempts = 0
    for attempt in range(max(1, int(max_retries))):
        attempts += 1
        try:
            response = request_get(endpoint, params=dict(params), timeout=30)
            raise_for_status = getattr(response, "raise_for_status", None)
            if callable(raise_for_status):
                raise_for_status()
            payload = response.json()
            if not isinstance(payload, Mapping):
                raise FundingHistoryError("Bitget returned a non-object payload")
            if str(payload.get("code") or "") != "00000":
                raise FundingHistoryError(
                    f"Bitget funding error {payload.get('code')}: "
                    f"{payload.get('msg')}"
                )
            return payload, attempts
        except Exception as exc:  # network clients expose varied exceptions
            last_error = exc
            if attempt + 1 < max(1, int(max_retries)):
                time.sleep(0.25 * (2**attempt))
    raise FundingHistoryError(f"Bitget funding request failed: {last_error}")


def _parse_instruments(
    payload: Mapping[str, Any],
    *,
    expected_symbols: Sequence[str],
) -> dict[str, int]:
    raw = payload.get("data")
    rows = raw if isinstance(raw, list) else []
    expected = set(expected_symbols)
    result: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        raw_symbol = str(row.get("symbol") or "").upper()
        if not raw_symbol.endswith("USDT"):
            continue
        symbol = _normalize_symbol(raw_symbol)
        if symbol not in expected:
            continue
        try:
            interval = int(row.get("fundInterval"))
        except (TypeError, ValueError):
            continue
        if interval not in ALLOWED_FUNDING_INTERVAL_HOURS:
            raise FundingHistoryError(
                f"unsupported funding interval for {symbol}: {interval}"
            )
        if str(row.get("status") or "").lower() != "online":
            raise FundingHistoryError(f"funding instrument is not online: {symbol}")
        result[symbol] = interval
    if set(result) != expected:
        raise FundingHistoryError(
            f"instrument metadata missing for {sorted(expected - set(result))}"
        )
    return {symbol: result[symbol] for symbol in expected_symbols}


def _parse_funding_page(
    payload: Mapping[str, Any],
    *,
    expected_symbol: str,
) -> list[dict[str, Any]]:
    data = payload.get("data")
    raw_rows = data.get("resultList") if isinstance(data, Mapping) else None
    rows = raw_rows if isinstance(raw_rows, list) else []
    parsed: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise FundingHistoryError("invalid funding history row")
        symbol = _normalize_symbol(row.get("symbol"))
        if symbol != expected_symbol:
            raise FundingHistoryError(
                f"unexpected funding symbol {symbol}; expected {expected_symbol}"
            )
        try:
            timestamp_ms = int(row.get("fundingRateTimestamp"))
            rate = float(row.get("fundingRate"))
        except (TypeError, ValueError) as exc:
            raise FundingHistoryError("invalid funding settlement value") from exc
        if timestamp_ms <= 0 or timestamp_ms % 3_600_000:
            raise FundingHistoryError("funding settlement is off the hourly grid")
        if not math.isfinite(rate) or abs(rate) > 0.1:
            raise FundingHistoryError("funding settlement rate is invalid")
        parsed.append(
            {
                "symbol": symbol,
                "funding_time_ms": timestamp_ms,
                "funding_rate": rate,
            }
        )
    return parsed


def _write_symbol_files(
    target: Path,
    rows_by_symbol: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for symbol in sorted(rows_by_symbol):
        path = target / f"funding_{symbol.lower()}usdt.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("symbol", "funding_time_ms", "funding_rate"),
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows_by_symbol[symbol]:
                writer.writerow(row)
        files.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "rows": len(rows_by_symbol[symbol]),
                "sha256": _sha256_file(path),
            }
        )
    return files


def _read_symbol_file(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            try:
                symbol = _normalize_symbol(raw.get("symbol"))
                timestamp_ms = int(raw["funding_time_ms"])
                rate = float(raw["funding_rate"])
            except (KeyError, TypeError, ValueError) as exc:
                raise FundingHistoryError(
                    f"invalid persisted funding row: {path.name}"
                ) from exc
            if (
                timestamp_ms <= 0
                or timestamp_ms % 3_600_000
                or not math.isfinite(rate)
                or abs(rate) > 0.1
            ):
                raise FundingHistoryError(
                    f"invalid persisted funding value: {path.name}"
                )
            rows.append(
                {
                    "symbol": symbol,
                    "funding_time_ms": timestamp_ms,
                    "funding_rate": rate,
                }
            )
    timestamps = [int(row["funding_time_ms"]) for row in rows]
    if timestamps != sorted(set(timestamps)):
        raise FundingHistoryError(
            f"funding rows are not unique and ordered: {path.name}"
        )
    return rows


def _coverage(
    rows: Sequence[Mapping[str, Any]],
    *,
    current_interval_hours: int,
) -> dict[str, Any]:
    timestamps = [int(row["funding_time_ms"]) for row in rows]
    gaps = [
        (current - previous) // 3_600_000
        for previous, current in zip(timestamps, timestamps[1:])
    ]
    gap_counts = Counter(gaps)
    invalid_gaps = [
        gap
        for gap in gaps
        if gap not in ALLOWED_FUNDING_INTERVAL_HOURS
    ]
    return {
        "rows": len(rows),
        "first_timestamp_ms": timestamps[0],
        "first_at": _iso_timestamp(timestamps[0]),
        "last_timestamp_ms": timestamps[-1],
        "last_at": _iso_timestamp(timestamps[-1]),
        "current_interval_hours": int(current_interval_hours),
        "observed_gap_hours": {
            str(key): int(value) for key, value in sorted(gap_counts.items())
        },
        "invalid_gap_count": len(invalid_gaps),
        "validation_passed": (
            bool(rows)
            and int(current_interval_hours) in ALLOWED_FUNDING_INTERVAL_HOURS
            and not invalid_gaps
        ),
    }


def _dataset_sha(files: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "path": str(item["path"]),
            "rows": int(item["rows"]),
            "sha256": str(item["sha256"]),
        }
        for item in sorted(files, key=lambda value: str(value["path"]))
    ]
    raw = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _normalize_symbol(value: Any) -> str:
    text = str(value or "").strip().upper()
    text = text.split(":", 1)[0].replace("/", "").replace("-", "")
    if text.endswith("USDT"):
        text = text[:-4]
    if not text or not text.isalnum():
        raise FundingHistoryError(f"invalid funding symbol: {value!r}")
    return text


def _timestamp_ms(value: Any) -> int:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise FundingHistoryError(f"invalid timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise FundingHistoryError("funding required window must be timezone-aware")
    return int(parsed.timestamp() * 1000)


def _iso_timestamp(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(
        timestamp_ms / 1000.0,
        timezone.utc,
    ).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
