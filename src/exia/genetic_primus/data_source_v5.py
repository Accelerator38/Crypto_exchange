from __future__ import annotations

import hashlib
import io
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from zipfile import ZIP_STORED, ZipFile

import numpy as np
import pandas as pd

from .contracts import sha256_file
from .contracts_v5 import PrimusV5Contract


RAW_METRIC_COLUMNS = (
    "create_time",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)

V5_DERIVATIVE_COLUMNS = (
    "timestamp",
    "available_at_ms",
    "symbol",
    "source_symbol",
    "samples",
    "sample_coverage",
    "complete_bar",
    "oi_open",
    "oi_close",
    "oi_change_4h",
    "oi_change_12h",
    "oi_change_24h",
    "oi_value_open",
    "oi_value_close",
    "oi_value_change_4h",
    "account_long_short_log",
    "top_account_long_short_log",
    "top_position_long_short_log",
    "taker_imbalance_mean",
    "taker_imbalance_std",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_checksum(payload: bytes, expected_name: str) -> str:
    text = payload.decode("utf-8").strip()
    parts = text.split()
    if len(parts) < 2 or parts[1].lstrip("*") != expected_name:
        raise ValueError(f"unexpected official checksum payload for {expected_name}")
    digest = parts[0].lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"invalid official SHA-256 for {expected_name}")
    return digest


def read_metric_zip(path_or_bytes: Path | bytes, *, expected_symbol: str | None = None) -> pd.DataFrame:
    source: Path | io.BytesIO
    source = path_or_bytes if isinstance(path_or_bytes, Path) else io.BytesIO(path_or_bytes)
    with ZipFile(source) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if len(members) != 1 or not members[0].endswith(".csv"):
            raise ValueError("V5 raw metric ZIP must contain exactly one CSV")
        with archive.open(members[0]) as handle:
            frame = pd.read_csv(handle)
    missing = set(RAW_METRIC_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"V5 raw metric columns missing: {sorted(missing)}")
    frame = frame.loc[:, list(RAW_METRIC_COLUMNS)].copy()
    if expected_symbol is not None and set(frame["symbol"].dropna().astype(str)) != {expected_symbol}:
        raise ValueError(f"V5 raw metric symbol mismatch: {expected_symbol}")
    frame["create_time"] = pd.to_datetime(frame["create_time"], utc=True, errors="raise")
    for column in RAW_METRIC_COLUMNS[2:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame["create_time"].duplicated().any():
        raise ValueError("V5 raw metric timestamps must be unique")
    source_was_unsorted = not frame["create_time"].is_monotonic_increasing
    if source_was_unsorted:
        frame = frame.sort_values("create_time", kind="stable").reset_index(drop=True)
    frame.attrs["source_was_unsorted"] = source_was_unsorted
    return frame


def aggregate_symbol_metrics(
    raw: pd.DataFrame,
    *,
    source_symbol: str,
    target_symbol: str,
    timeframe_minutes: int = 240,
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=V5_DERIVATIVE_COLUMNS)
    local = raw.sort_values("create_time", kind="stable").copy()
    if set(local["symbol"].dropna().astype(str)) != {source_symbol}:
        raise ValueError(f"unexpected source symbol while aggregating {source_symbol}")
    taker_ratio = local["sum_taker_long_short_vol_ratio"].where(
        local["sum_taker_long_short_vol_ratio"] > 0.0
    )
    local["taker_imbalance"] = (taker_ratio - 1.0) / (taker_ratio + 1.0)
    local["bar_open"] = local["create_time"].dt.floor(f"{int(timeframe_minutes)}min")
    grouped = local.groupby("bar_open", sort=True, observed=True).agg(
        samples=("create_time", "size"),
        oi_open=("sum_open_interest", "first"),
        oi_close=("sum_open_interest", "last"),
        oi_value_open=("sum_open_interest_value", "first"),
        oi_value_close=("sum_open_interest_value", "last"),
        account_long_short_ratio=("count_long_short_ratio", "last"),
        top_account_long_short_ratio=("count_toptrader_long_short_ratio", "last"),
        top_position_long_short_ratio=("sum_toptrader_long_short_ratio", "last"),
        taker_imbalance_mean=("taker_imbalance", "mean"),
        taker_imbalance_std=("taker_imbalance", "std"),
    )
    expected_samples = int(timeframe_minutes // 5)
    grouped["sample_coverage"] = grouped["samples"] / expected_samples
    grouped["complete_bar"] = grouped["samples"] >= math.ceil(expected_samples * 0.95)
    grouped["oi_change_4h"] = grouped["oi_close"] / grouped["oi_open"].replace(0.0, np.nan) - 1.0
    grouped["oi_change_12h"] = grouped["oi_close"] / grouped["oi_close"].shift(3).replace(0.0, np.nan) - 1.0
    grouped["oi_change_24h"] = grouped["oi_close"] / grouped["oi_close"].shift(6).replace(0.0, np.nan) - 1.0
    grouped["oi_value_change_4h"] = (
        grouped["oi_value_close"] / grouped["oi_value_open"].replace(0.0, np.nan) - 1.0
    )
    for source_column, output_column in (
        ("account_long_short_ratio", "account_long_short_log"),
        ("top_account_long_short_ratio", "top_account_long_short_log"),
        ("top_position_long_short_ratio", "top_position_long_short_log"),
    ):
        grouped[output_column] = np.log(grouped[source_column].where(grouped[source_column] > 0.0))
    grouped = grouped.reset_index()
    grouped["timestamp"] = grouped["bar_open"].map(lambda value: int(value.timestamp() * 1000)).astype("int64")
    grouped["available_at_ms"] = grouped["timestamp"] + int(timeframe_minutes * 60_000)
    grouped["symbol"] = target_symbol
    grouped["source_symbol"] = source_symbol
    grouped["taker_imbalance_std"] = grouped["taker_imbalance_std"].fillna(0.0)
    return grouped.loc[:, list(V5_DERIVATIVE_COLUMNS)].reset_index(drop=True)


def pack_raw_directory(raw_dir: Path, archive_path: Path) -> None:
    temporary = archive_path.with_suffix(archive_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with ZipFile(temporary, "w", compression=ZIP_STORED, allowZip64=True) as archive:
        for path in sorted(raw_dir.rglob("*.zip")):
            archive.write(path, path.relative_to(raw_dir).as_posix())
    temporary.replace(archive_path)


def validate_dataset_manifest(
    contract: PrimusV5Contract,
    manifest_path: Path,
    *,
    root: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "exia.genetic_primus_derivatives_dataset.v5":
        raise ValueError("unexpected V5 derivatives dataset schema")
    if manifest.get("contract_sha256") != contract.sha256:
        raise ValueError("V5 derivatives dataset contract SHA mismatch")
    if manifest.get("source_id") != contract.source["source_id"]:
        raise ValueError("V5 derivatives dataset source ID mismatch")
    for flag in ("paper_allowed", "live_allowed", "orders_enabled", "promotion_authority"):
        if manifest.get(flag) is not False:
            raise ValueError("V5 derivatives dataset safety flags must be false")
    artifacts = manifest.get("artifacts", {})
    for key in ("dataset", "raw_catalog", "raw_archive"):
        item = artifacts.get(key, {})
        path = root / str(item.get("path", ""))
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise ValueError(f"V5 derivatives artifact mismatch: {key}")
    return manifest
