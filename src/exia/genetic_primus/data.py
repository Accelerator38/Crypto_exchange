from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .contracts import PrimusConfig, sha256_file


REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume", "symbol")


def load_panel(config: PrimusConfig, *, root: Path) -> pd.DataFrame:
    manifest_path = root / str(config.dataset["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_sha256") != config.dataset["dataset_sha256"]:
        raise ValueError("Genetic_Primus dataset SHA mismatch")
    if manifest.get("validation", {}).get("passed") is not True:
        raise ValueError("Genetic_Primus dataset manifest did not pass validation")
    if manifest.get("timeframe") != "240m":
        raise ValueError("Genetic_Primus source must be the immutable 240m panel")
    frames: list[pd.DataFrame] = []
    for item in manifest.get("files", []):
        path = root / str(item["path"]).replace("\\", "/")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"dataset file SHA mismatch: {path}")
        frames.append(pd.read_parquet(path, columns=list(REQUIRED_COLUMNS)))
    if not frames:
        raise ValueError("dataset manifest contains no files")
    frame = pd.concat(frames, ignore_index=True)
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="raise").astype("int64")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
    frame["symbol"] = frame["symbol"].astype(str)
    frame = frame.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)
    if frame.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("dataset contains duplicate timestamp/symbol rows")
    expected = set(str(value) for value in config.dataset["symbols"])
    if set(frame["symbol"].unique()) != expected:
        raise ValueError("dataset symbol set does not match Genetic_Primus contract")
    counts = frame.groupby("timestamp", sort=False)["symbol"].nunique()
    if not (counts == len(expected)).all():
        raise ValueError("dataset contains incomplete full8 bars")
    return frame


def split_frame(frame: pd.DataFrame, window: dict[str, str]) -> pd.DataFrame:
    start = int(pd.Timestamp(window["start"]).timestamp() * 1000)
    end = int(pd.Timestamp(window["end"]).timestamp() * 1000)
    return frame.loc[(frame["timestamp"] >= start) & (frame["timestamp"] < end)].copy()
