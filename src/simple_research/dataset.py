from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
)
FEATURE_COLUMNS = (
    "return_1",
    "log_return_1",
    "range_bps",
    "body_bps",
    "volume_z_48",
    "volume_ratio_24",
    "realized_vol_24",
    "realized_vol_96",
    "compression_ratio",
    "ema_12",
    "ema_48",
    "ema_96",
    "atr_14",
    "close_z_24",
    "donchian_high_20",
    "donchian_low_20",
    "donchian_high_55",
    "donchian_low_55",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files(source_dir: Path) -> tuple[Path, ...]:
    files = tuple(sorted(source_dir.glob("crypto_*.csv")))
    if not files:
        raise FileNotFoundError(f"no crypto_*.csv files in {source_dir}")
    return files


def _load_source(source_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in _source_files(source_dir)]
    frame = pd.concat(frames, ignore_index=True)
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"missing source columns: {missing}")
    frame = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="raise").astype(
        "int64"
    )
    for name in ("open", "high", "low", "close", "volume"):
        frame[name] = pd.to_numeric(frame[name], errors="raise").astype("float64")
    frame["symbol"] = frame["symbol"].astype(str)
    return frame.sort_values(["symbol", "timestamp"], kind="stable").reset_index(
        drop=True
    )


def validate_ohlcv(
    frame: pd.DataFrame,
    *,
    timeframe_ms: int,
    expected_symbols: Iterable[str] | None = None,
) -> dict[str, object]:
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    duplicates = int(frame.duplicated(["symbol", "timestamp"]).sum())
    finite = np.isfinite(frame[["open", "high", "low", "close", "volume"]]).all(
        axis=None
    )
    price_invalid = int(
        (
            (frame["open"] <= 0)
            | (frame["high"] <= 0)
            | (frame["low"] <= 0)
            | (frame["close"] <= 0)
            | (frame["volume"] < 0)
            | (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
            | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
        ).sum()
    )
    interval_violations = 0
    symbol_rows: dict[str, int] = {}
    for symbol, group in frame.groupby("symbol", sort=True):
        diff = group["timestamp"].diff().dropna()
        interval_violations += int((diff != timeframe_ms).sum())
        symbol_rows[str(symbol)] = int(len(group))
    actual_symbols = tuple(sorted(symbol_rows))
    expected = tuple(sorted(str(value) for value in (expected_symbols or actual_symbols)))
    checks = {
        "nonempty": not frame.empty,
        "no_duplicates": duplicates == 0,
        "finite_ohlcv": bool(finite),
        "valid_ohlcv": price_invalid == 0,
        "continuous": interval_violations == 0,
        "symbols_match": actual_symbols == expected,
    }
    failed = [name for name, passed in checks.items() if not passed]
    report = {
        "passed": not failed,
        "checks": checks,
        "failed": failed,
        "rows": int(len(frame)),
        "symbols": list(actual_symbols),
        "rows_by_symbol": symbol_rows,
        "duplicates": duplicates,
        "price_invalid": price_invalid,
        "interval_violations": interval_violations,
    }
    if failed:
        raise ValueError(f"OHLCV validation failed: {json.dumps(report)}")
    return report


def _with_features(frame: pd.DataFrame) -> pd.DataFrame:
    groups: list[pd.DataFrame] = []
    for _, source in frame.groupby("symbol", sort=True):
        group = source.copy()
        close = group["close"]
        previous_close = close.shift(1)
        log_return = np.log(close).diff()
        group["return_1"] = close.pct_change()
        group["log_return_1"] = log_return
        group["range_bps"] = (group["high"] - group["low"]) / close * 10_000.0
        group["body_bps"] = (close - group["open"]) / group["open"] * 10_000.0
        log_volume = np.log1p(group["volume"])
        volume_mean = log_volume.rolling(48, min_periods=24).mean()
        volume_std = log_volume.rolling(48, min_periods=24).std(ddof=0)
        group["volume_z_48"] = (log_volume - volume_mean) / volume_std.replace(0, np.nan)
        group["volume_ratio_24"] = group["volume"] / group["volume"].rolling(
            24, min_periods=12
        ).mean()
        group["realized_vol_24"] = log_return.rolling(24, min_periods=12).std(ddof=0)
        group["realized_vol_96"] = log_return.rolling(96, min_periods=48).std(ddof=0)
        group["compression_ratio"] = group["realized_vol_24"] / group[
            "realized_vol_96"
        ].replace(0, np.nan)
        for span in (12, 48, 96):
            group[f"ema_{span}"] = close.ewm(span=span, adjust=False).mean()
        true_range = pd.concat(
            [
                group["high"] - group["low"],
                (group["high"] - previous_close).abs(),
                (group["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        group["atr_14"] = true_range.rolling(14, min_periods=7).mean()
        close_mean = close.rolling(24, min_periods=12).mean()
        close_std = close.rolling(24, min_periods=12).std(ddof=0)
        group["close_z_24"] = (close - close_mean) / close_std.replace(0, np.nan)
        for window in (20, 55):
            group[f"donchian_high_{window}"] = (
                group["high"].shift(1).rolling(window, min_periods=window).max()
            )
            group[f"donchian_low_{window}"] = (
                group["low"].shift(1).rolling(window, min_periods=window).min()
            )
        groups.append(group)
    result = pd.concat(groups, ignore_index=True)
    result["datetime"] = pd.to_datetime(result["timestamp"], unit="ms", utc=True)
    return result.sort_values(["symbol", "timestamp"], kind="stable").reset_index(
        drop=True
    )


def build_feature_tape(
    *,
    source_dir: Path,
    output_path: Path,
    timeframe: str,
    timeframe_ms: int,
    expected_symbols: Iterable[str] | None = None,
) -> dict[str, object]:
    source_dir = source_dir.resolve()
    output_path = output_path.resolve()
    source = _load_source(source_dir)
    validation = validate_ohlcv(
        source,
        timeframe_ms=timeframe_ms,
        expected_symbols=expected_symbols,
    )
    tape = _with_features(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tape.to_parquet(output_path, index=False, compression="zstd")
    files = [
        {
            "path": str(path.relative_to(source_dir)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in _source_files(source_dir)
    ]
    source_integrity_path = source_dir / "integrity_manifest.json"
    source_integrity: dict[str, object] | None = None
    if source_integrity_path.is_file():
        source_integrity_payload = json.loads(
            source_integrity_path.read_text(encoding="utf-8")
        )
        source_integrity = {
            "path": str(source_integrity_path),
            "sha256": _sha256(source_integrity_path),
            "dataset_sha256": source_integrity_payload.get("dataset_sha256"),
            "validation_passed": source_integrity_payload.get("validation", {}).get(
                "passed"
            ),
        }
    manifest: dict[str, object] = {
        "schema_version": "simple_research.feature_tape.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timeframe": timeframe,
        "timeframe_ms": timeframe_ms,
        "source_dir": str(source_dir),
        "source_files": files,
        "source_integrity": source_integrity,
        "output_path": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": _sha256(output_path),
        "rows": int(len(tape)),
        "symbols": validation["symbols"],
        "first_timestamp": int(tape["timestamp"].min()),
        "last_timestamp": int(tape["timestamp"].max()),
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_semantics": "bar-close observations; execution must be delayed to next open",
        "future_labels_included": False,
        "validation": validation,
        "orders_enabled": False,
        "promotion_authority": False,
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_feature_tape(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = [name for name in (*REQUIRED_COLUMNS, *FEATURE_COLUMNS) if name not in frame]
    if missing:
        raise ValueError(f"feature tape missing columns: {missing}")
    return frame.sort_values(["symbol", "timestamp"], kind="stable").reset_index(
        drop=True
    )
