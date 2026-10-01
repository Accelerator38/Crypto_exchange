"""Immutable local-data commitment for the prospective evaluator."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .prospective_contract_v1 import file_sha256


SCHEMA = "exia.genetic_primus.prospective_data_manifest/1"
REQUIRED_COLUMNS = ("timestamp", "symbol", "open", "high", "low", "close", "volume")


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("snapshot file must remain inside the workspace")
    return resolved


def _read_parquet(path: Path) -> pd.DataFrame:
    if path.suffix.lower() != ".parquet":
        raise ValueError("only immutable parquet snapshots are accepted")
    frame = pd.read_parquet(path)
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"snapshot columns missing: {sorted(missing)}")
    return frame[list(REQUIRED_COLUMNS)].copy()


def _validate_panel(panel: pd.DataFrame, symbols: tuple[str, ...], timeframe_ms: int) -> dict[str, Any]:
    if panel.empty:
        raise ValueError("empty market snapshot")
    panel["symbol"] = panel["symbol"].astype(str)
    panel["timestamp"] = pd.to_numeric(panel["timestamp"], errors="raise").astype("int64")
    numeric = list(REQUIRED_COLUMNS[2:])
    panel[numeric] = panel[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(panel[numeric].to_numpy(dtype=float)).all():
        raise ValueError("non-finite OHLCV value")
    if (panel[["open", "high", "low", "close"]] <= 0).any(axis=None):
        raise ValueError("non-positive price")
    if (panel["volume"] < 0).any():
        raise ValueError("negative volume")
    if panel.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("duplicate timestamp/symbol")
    observed = tuple(sorted(panel["symbol"].unique()))
    if observed != tuple(sorted(symbols)):
        raise ValueError("snapshot universe differs from contract")
    if (panel["timestamp"] % timeframe_ms != 0).any():
        raise ValueError("timestamp is not aligned to timeframe")
    counts = panel.groupby("timestamp", observed=True)["symbol"].nunique()
    if not (counts == len(symbols)).all():
        raise ValueError("incomplete universe at a timestamp")
    timestamps = np.sort(panel["timestamp"].unique())
    if len(timestamps) > 1 and not np.all(np.diff(timestamps) == timeframe_ms):
        raise ValueError("market grid has a missing candle")
    return {
        "row_count": int(len(panel)),
        "timestamp_count": int(len(timestamps)),
        "coverage_start": int(timestamps[0]),
        "coverage_end_exclusive": int(timestamps[-1] + timeframe_ms),
        "symbols": list(symbols),
        "timeframe_ms": int(timeframe_ms),
        "columns": list(REQUIRED_COLUMNS),
    }


def build_manifest(
    paths: Iterable[Path], *, root: Path, contract_path: Path, contract: dict[str, Any],
    created_at: datetime | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    resolved = tuple(sorted({_inside(path, root) for path in paths}, key=lambda item: item.as_posix()))
    if not resolved:
        raise ValueError("at least one parquet snapshot is required")
    entries = [
        {"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}
        for path in resolved
    ]
    panel = pd.concat((_read_parquet(path) for path in resolved), ignore_index=True)
    timeframe_ms = int(contract["training"]["timeframe_minutes"]) * 60_000
    metadata = _validate_panel(panel, tuple(contract["universe"]), timeframe_ms)
    created = (created_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    revealed_diagnostic = (
        contract.get("data_already_revealed") is True
        and contract.get("selection_use") == "engineering_diagnostic_only_not_validation_not_tuning_not_promotion"
    )
    if not revealed_diagnostic and created >= datetime.fromisoformat(
        contract["outer"]["end_exclusive_utc"].replace("Z", "+00:00")
    ):
        raise ValueError("manifest was not committed before the full outer interval ended")
    return {
        "schema_version": SCHEMA,
        "created_at_utc": created.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "contract_path": contract_path.resolve().relative_to(root).as_posix(),
        "contract_sha256": file_sha256(contract_path),
        "snapshot_files": entries,
        "snapshot_sha256": _canonical_sha(entries),
        "metadata": metadata,
        "data_already_revealed": revealed_diagnostic,
        "immutable": True,
        "market_values_validated_by_builder": True,
        "outcome_metrics_computed_by_builder": False,
        "network_used": False,
        "safety": dict(contract["safety"]),
    }


def save_manifest_exclusive(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def load_verified_snapshot(
    manifest_path: Path, *, root: Path, contract_path: Path, contract: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    root = root.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA or manifest.get("immutable") is not True:
        raise ValueError("wrong or mutable manifest")
    if manifest.get("contract_path") != contract_path.resolve().relative_to(root).as_posix():
        raise ValueError("manifest points to another contract")
    if manifest.get("contract_sha256") != file_sha256(contract_path):
        raise ValueError("contract changed after data commitment")
    if any(manifest.get("safety", {}).get(key) is not False for key in contract["safety"]):
        raise ValueError("manifest safety flags changed")
    entries = manifest.get("snapshot_files")
    if not isinstance(entries, list) or not entries or _canonical_sha(entries) != manifest.get("snapshot_sha256"):
        raise ValueError("snapshot commitment is malformed")
    frames = []
    for entry in entries:
        path = _inside(root / entry["path"], root)
        if path.stat().st_size != entry["size_bytes"] or file_sha256(path) != entry["sha256"]:
            raise ValueError("snapshot bytes changed after commitment")
        frames.append(_read_parquet(path))
    panel = pd.concat(frames, ignore_index=True)
    metadata = _validate_panel(
        panel, tuple(contract["universe"]), int(contract["training"]["timeframe_minutes"]) * 60_000
    )
    if metadata != manifest.get("metadata"):
        raise ValueError("manifest metadata does not match snapshot")
    return manifest, panel.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)
