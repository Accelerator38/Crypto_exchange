from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from exia.genetic_primus.prospective_contract_v1 import load_and_validate
from exia.genetic_primus.prospective_evaluator_v1 import evaluate_panel_origin, origin_preflight
from exia.genetic_primus.prospective_manifest_v1 import (
    build_manifest,
    load_verified_snapshot,
    save_manifest_exclusive,
)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/genetic_primus_prospective_3d_v1.json"


def _panel(contract, timestamp_count=3):
    start = 1_640_995_200_000
    step = 14_400_000
    rows = []
    for offset in range(timestamp_count):
        for symbol_index, symbol in enumerate(contract["universe"]):
            price = 100.0 + symbol_index + offset
            rows.append({
                "timestamp": start + offset * step,
                "symbol": symbol,
                "open": price,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price + 0.25,
                "volume": 10.0 + offset,
            })
    return pd.DataFrame(rows)


def test_manifest_round_trip_and_byte_tamper_detection(tmp_path):
    contract, _ = load_and_validate(CONFIG)
    local_config = tmp_path / "contract.json"
    local_config.write_bytes(CONFIG.read_bytes())
    snapshot = tmp_path / "snapshot.parquet"
    _panel(contract).to_parquet(snapshot, index=False)
    manifest = build_manifest(
        [snapshot], root=tmp_path, contract_path=local_config, contract=contract,
        created_at=datetime(2026, 8, 30, 12, tzinfo=timezone.utc),
    )
    path = tmp_path / "manifest.json"
    save_manifest_exclusive(path, manifest)
    loaded, panel = load_verified_snapshot(
        path, root=tmp_path, contract_path=local_config, contract=contract
    )
    assert loaded["snapshot_sha256"] == manifest["snapshot_sha256"]
    assert len(panel) == 3 * len(contract["universe"])
    with snapshot.open("ab") as stream:
        stream.write(b"tamper")
    try:
        load_verified_snapshot(path, root=tmp_path, contract_path=local_config, contract=contract)
    except ValueError as error:
        assert "bytes changed" in str(error)
    else:
        raise AssertionError("tampered snapshot was accepted")


def test_preflight_rejects_early_or_incomplete_origin():
    contract, _ = load_and_validate(CONFIG)
    manifest = {"metadata": {"coverage_start": 1_640_995_200_000, "coverage_end_exclusive": 1_788_480_000_000}}
    try:
        origin_preflight(contract, manifest, 0, now=datetime(2026, 9, 3, tzinfo=timezone.utc))
    except ValueError as error:
        assert "not complete" in str(error)
    else:
        raise AssertionError("unfinished origin was accepted")
    incomplete = deepcopy(manifest)
    incomplete["metadata"]["coverage_end_exclusive"] = 1_788_307_200_000
    try:
        origin_preflight(contract, incomplete, 0, now=datetime(2026, 9, 4, tzinfo=timezone.utc))
    except ValueError as error:
        assert "does not cover" in str(error)
    else:
        raise AssertionError("incomplete snapshot was accepted")


def test_synthetic_origin_evaluates_fixed_candidates_without_search():
    contract, _ = load_and_validate(CONFIG)
    step = 14_400_000
    start = 1_640_995_200_000
    end = 1_788_480_000_000
    timestamps = np.arange(start, end, step, dtype=np.int64)
    grid = pd.MultiIndex.from_product(
        [timestamps, contract["universe"]], names=["timestamp", "symbol"]
    ).to_frame(index=False)
    symbol_code = grid["symbol"].map({symbol: i for i, symbol in enumerate(contract["universe"])}).to_numpy()
    time_code = ((grid["timestamp"].to_numpy() - start) // step).astype(float)
    close = 100.0 + 3.0 * symbol_code + 0.004 * time_code + np.sin(time_code / 13.0) * 0.3
    panel = grid.assign(
        open=close - 0.02, high=close + 0.10, low=close - 0.10, close=close,
        volume=1000.0 + symbol_code + np.cos(time_code / 17.0),
    )
    manifest = {
        "snapshot_sha256": "synthetic_snapshot_for_test_only",
        "metadata": {
            "coverage_start": int(timestamps[0]),
            "coverage_end_exclusive": int(timestamps[-1] + step),
        },
    }
    result = evaluate_panel_origin(
        panel, contract, manifest, 0, now=datetime(2026, 9, 4, tzinfo=timezone.utc)
    )
    json.dumps(result, allow_nan=False)
    assert result["search_evaluations"] == 0
    assert set(result["evaluations"]) == {"normal", "stress"}
    for evaluation in result["evaluations"].values():
        assert set(evaluation["candidates"]) == {
            "NoTrade", "EMA_TrendConsensus", "GA_overlay_f7a4ae11"
        }
        assert all(len(item["daily_returns"]) == 3 for item in evaluation["candidates"].values())
        assert all(len(item["ledger"]) == 18 * 8 for item in evaluation["candidates"].values())
