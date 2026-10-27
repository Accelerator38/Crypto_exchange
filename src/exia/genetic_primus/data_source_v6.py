from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .contracts import sha256_file
from .contracts_v6 import PrimusV6Contract


KLINE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)


def read_kline_zip(path_or_bytes: Path | bytes) -> pd.DataFrame:
    source = path_or_bytes if isinstance(path_or_bytes, Path) else io.BytesIO(path_or_bytes)
    with ZipFile(source) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if len(members) != 1 or not members[0].endswith(".csv"):
            raise ValueError("V6 kline ZIP must contain exactly one CSV")
        with archive.open(members[0]) as handle:
            payload = handle.read()
    frame = pd.read_csv(io.BytesIO(payload))
    source_header_missing = False
    if tuple(frame.columns) != KLINE_COLUMNS:
        numeric_header = len(frame.columns) == len(KLINE_COLUMNS)
        if numeric_header:
            try:
                [float(value) for value in frame.columns]
            except (TypeError, ValueError):
                numeric_header = False
        if not numeric_header:
            raise ValueError("V6 kline schema mismatch")
        frame = pd.read_csv(io.BytesIO(payload), header=None, names=KLINE_COLUMNS)
        source_header_missing = True
    for column in KLINE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["open_time"] = frame["open_time"].astype("int64")
    frame["close_time"] = frame["close_time"].astype("int64")
    if frame["open_time"].duplicated().any() or not frame["open_time"].is_monotonic_increasing:
        raise ValueError("V6 kline timestamps must be unique and sorted")
    interval_ms = 4 * 60 * 60 * 1000
    if (frame["open_time"] % interval_ms != 0).any():
        raise ValueError("V6 kline open timestamp is off the four-hour grid")
    if (frame["close_time"] != frame["open_time"] + interval_ms - 1).any():
        raise ValueError("V6 kline close timestamp mismatch")
    if (frame["quote_volume"] <= 0.0).any():
        raise ValueError("V6 kline quote volume must be positive")
    frame.attrs["source_header_missing"] = source_header_missing
    return frame


def build_flow_frame(raw: pd.DataFrame, *, source_symbol: str, target_symbol: str) -> pd.DataFrame:
    imbalance = 2.0 * raw["taker_buy_quote_volume"] / raw["quote_volume"] - 1.0
    if ((imbalance < -1.0000001) | (imbalance > 1.0000001)).any():
        raise ValueError("V6 kline taker imbalance outside [-1, 1]")
    return pd.DataFrame(
        {
            "timestamp": raw["open_time"].astype("int64"),
            "available_at_ms": raw["close_time"].astype("int64") + 1,
            "symbol": target_symbol,
            "source_symbol": source_symbol,
            "kline_taker_imbalance_4h": imbalance.astype(float),
            "kline_quote_volume": raw["quote_volume"].astype(float),
            "kline_trade_count": raw["count"].astype("int64"),
        }
    )


def validate_v6_dataset_manifest(
    contract: PrimusV6Contract,
    manifest_path: Path,
    *,
    root: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "exia.genetic_primus_derivatives_dataset.v6":
        raise ValueError("unexpected V6 derivatives dataset schema")
    if manifest.get("contract_sha256") != contract.sha256:
        raise ValueError("V6 derivatives contract SHA mismatch")
    for flag in ("paper_allowed", "live_allowed", "orders_enabled", "promotion_authority"):
        if manifest.get(flag) is not False:
            raise ValueError("V6 derivatives dataset must be fail-closed")
    for key in ("combined_dataset", "flow_dataset", "raw_catalog", "raw_archive", "workbook"):
        item = manifest.get("artifacts", {}).get(key, {})
        path = root / str(item.get("path", ""))
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise ValueError(f"V6 artifact mismatch: {key}")
    return manifest
