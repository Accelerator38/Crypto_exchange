from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd

from .contracts import sha256_file
from .contracts_v7 import PrimusV7Contract


LIQUIDATION_COLUMNS = (
    "time",
    "side",
    "order_type",
    "time_in_force",
    "original_quantity",
    "price",
    "average_price",
    "order_status",
    "last_fill_quantity",
    "accumulated_fill_quantity",
)


def four_hour_grid(start_date: str, end_date: str) -> pd.DataFrame:
    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(hours=20)
    timestamps = pd.date_range(start, end, freq="4h")
    frame = pd.DataFrame(
        {"timestamp": pd.Series([timestamp.value // 1_000_000 for timestamp in timestamps], dtype="int64")}
    )
    frame["date"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    return frame


def read_liquidation_zip(path_or_bytes: Path | bytes, *, expected_date: str) -> pd.DataFrame:
    source = path_or_bytes if isinstance(path_or_bytes, Path) else io.BytesIO(path_or_bytes)
    with ZipFile(source) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if len(members) != 1 or not members[0].endswith(".csv"):
            raise ValueError("V7 liquidation ZIP must contain exactly one CSV")
        with archive.open(members[0]) as handle:
            frame = pd.read_csv(handle)
    if tuple(frame.columns) != LIQUIDATION_COLUMNS:
        raise ValueError("V7 liquidation schema mismatch")
    numeric = (
        "time",
        "original_quantity",
        "price",
        "average_price",
        "last_fill_quantity",
        "accumulated_fill_quantity",
    )
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["time"] = frame["time"].astype("int64")
    if not set(frame["side"].astype(str)).issubset({"BUY", "SELL"}):
        raise ValueError("V7 liquidation side is invalid")
    if (frame[["original_quantity", "accumulated_fill_quantity", "average_price"]] < 0).any().any():
        raise ValueError("V7 liquidation numeric values must be nonnegative")
    dates = pd.to_datetime(frame["time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    if not dates.eq(expected_date).all():
        raise ValueError("V7 liquidation timestamp is outside archive date")
    duplicate_count = int(frame.duplicated(keep="first").sum())
    frame = frame.drop_duplicates(keep="first").sort_values("time", kind="stable").reset_index(drop=True)
    frame.attrs["exact_duplicates_removed"] = duplicate_count
    return frame


def aggregate_liquidations(
    raw: pd.DataFrame,
    *,
    target_symbol: str,
    source_symbol: str,
    contract_size_usd: float,
) -> pd.DataFrame:
    local = raw.copy()
    local["timestamp"] = (local["time"] // 14_400_000) * 14_400_000
    local["notional_usd"] = local["accumulated_fill_quantity"] * float(contract_size_usd)
    local["sell_liquidation_usd"] = local["notional_usd"].where(local["side"] == "SELL", 0.0)
    local["buy_liquidation_usd"] = local["notional_usd"].where(local["side"] == "BUY", 0.0)
    local["sell_liquidation_count"] = (local["side"] == "SELL").astype("int64")
    local["buy_liquidation_count"] = (local["side"] == "BUY").astype("int64")
    grouped = local.groupby("timestamp", as_index=False, sort=True).agg(
        sell_liquidation_usd=("sell_liquidation_usd", "sum"),
        buy_liquidation_usd=("buy_liquidation_usd", "sum"),
        sell_liquidation_count=("sell_liquidation_count", "sum"),
        buy_liquidation_count=("buy_liquidation_count", "sum"),
    )
    grouped["liquidation_total_usd"] = grouped["sell_liquidation_usd"] + grouped["buy_liquidation_usd"]
    grouped["symbol"] = target_symbol
    grouped["source_symbol"] = source_symbol
    return grouped


def validate_v7_dataset_manifest(
    contract: PrimusV7Contract,
    manifest_path: Path,
    *,
    root: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "exia.genetic_primus_liquidation_dataset.v7":
        raise ValueError("unexpected V7 liquidation dataset schema")
    if manifest.get("contract_sha256") != contract.sha256:
        raise ValueError("V7 liquidation contract SHA mismatch")
    for flag in ("paper_allowed", "live_allowed", "orders_enabled", "promotion_authority"):
        if manifest.get(flag) is not False:
            raise ValueError("V7 liquidation dataset must be fail-closed")
    for key in ("feature_panel", "event_bars", "raw_catalog", "source_listing", "raw_archive", "workbook"):
        item = manifest.get("artifacts", {}).get(key, {})
        path = root / str(item.get("path", ""))
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise ValueError(f"V7 artifact mismatch: {key}")
    return manifest
