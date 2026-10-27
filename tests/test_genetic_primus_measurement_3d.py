from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from exia.genetic_primus.fitness_v4 import contiguous_block_lcb
from exia.genetic_primus.measurement import (
    build_bootstrap_bank,
    label_purge_mask,
    load_measurement_protocol,
    save_bootstrap_bank,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/genetic_primus_measurement_3d_v1.json"


def test_three_day_protocol_purges_seven_bars() -> None:
    protocol = load_measurement_protocol(PROTOCOL)
    bar_ms = 240 * 60 * 1000
    timestamps = pd.Series(
        [protocol.cutoff_ms - 8 * bar_ms, protocol.cutoff_ms - 7 * bar_ms]
    )
    assert label_purge_mask(timestamps, protocol).tolist() == [True, False]
    assert protocol.payload["deployment_horizon_days"] == 3


def test_bootstrap_banks_are_common_deterministic_and_independent() -> None:
    protocol = load_measurement_protocol(PROTOCOL)
    start = int(pd.Timestamp("2024-01-01T00:00:00Z").timestamp() * 1000)
    days = pd.date_range("2024-01-01", periods=16, freq="1D", tz="UTC")
    frame = pd.DataFrame(
        [
            {"timestamp": int(day.timestamp() * 1000), "symbol": symbol}
            for symbol in ("BTC/USDT", "LINK/USDT")
            for day in days
        ]
    )
    folds = (
        {
            "name": "toy",
            "start": "2024-01-01T00:00:00Z",
            "end": "2024-01-17T00:00:00Z",
        },
    )
    common = {
        "frame": frame,
        "start_timestamp": start,
        "cutoff_timestamp": int(pd.Timestamp("2024-01-17T00:00:00Z").timestamp() * 1000),
        "folds": folds,
        "samples": 100,
        "block_days": 7,
    }
    first = build_bootstrap_bank(**common, bank_id="selection", seed=101)
    repeated = build_bootstrap_bank(**common, bank_id="selection", seed=101)
    audit = build_bootstrap_bank(**common, bank_id="audit", seed=202)
    assert first.content_sha256 == repeated.content_sha256
    assert first.content_sha256 != audit.content_sha256
    values = np.arange(len(first.scope_timestamps["overall"]), dtype=float)
    indices = first.indices_for("overall", first.scope_timestamps["overall"])
    left = contiguous_block_lcb(
        values,
        samples=100,
        alpha=0.05,
        block_size=7,
        seed=1,
        resample_indices=indices,
    )
    right = contiguous_block_lcb(
        values,
        samples=100,
        alpha=0.05,
        block_size=7,
        seed=999,
        resample_indices=indices,
    )
    assert left == right
    metadata = first.metadata()
    assert metadata["cutoff_ms"] == common["cutoff_timestamp"]
    assert metadata["folds_sha256"]
    assert metadata["content_sha256"] == first.content_sha256
    with tempfile.TemporaryDirectory() as directory:
        first_path = Path(directory) / "first.npz"
        second_path = Path(directory) / "second.npz"
        save_bootstrap_bank(first_path, first)
        save_bootstrap_bank(second_path, first)
        first_sha = hashlib.sha256(first_path.read_bytes()).hexdigest()
        second_sha = hashlib.sha256(second_path.read_bytes()).hexdigest()
        assert first_sha == second_sha
