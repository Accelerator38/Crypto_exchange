"""Offline, fixed-candidate evaluator for one completed prospective origin."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from .measurement_lab_v2 import (
    ALLOW_ALL,
    NO_TRADE,
    LabScaler,
    PreparedWindow,
    build_lab_features,
    daily_returns,
)
from .prospective_contract_v1 import derive_origins, validate_contract


OVERLAY = (0, 0, 1, 1, -1)
GENOMES = {
    "NoTrade": NO_TRADE,
    "EMA_TrendConsensus": ALLOW_ALL,
    "GA_overlay_f7a4ae11": OVERLAY,
}


def _frame_sha(frame: pd.DataFrame) -> str:
    time_column = "timestamp" if "timestamp" in frame.columns else "execution_timestamp"
    if time_column not in frame.columns or "symbol" not in frame.columns:
        raise ValueError("hashed frame needs time and symbol columns")
    ordered = frame.sort_values([time_column, "symbol"], kind="stable").reset_index(drop=True)
    prefix = json.dumps(
        {"columns": list(ordered.columns), "dtypes": [str(value) for value in ordered.dtypes]},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    hashes = pd.util.hash_pandas_object(ordered, index=False, categorize=True).to_numpy(dtype="<u8")
    return hashlib.sha256(prefix + hashes.tobytes()).hexdigest()


def _as_utc(value: datetime | None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return current.astimezone(timezone.utc)


def origin_preflight(
    contract: dict[str, Any], manifest: dict[str, Any], origin_index: int, *, now: datetime | None = None,
) -> tuple[int, int]:
    origins = validate_contract(contract)
    if origin_index not in range(len(origins)):
        raise ValueError("origin index is not registered")
    start, end = origins[origin_index]
    if _as_utc(now) < end:
        raise ValueError("origin is not complete")
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    metadata = manifest["metadata"]
    if metadata["coverage_start"] > int(datetime.fromisoformat(
        contract["training"]["initial_history_start_utc"].replace("Z", "+00:00")
    ).timestamp() * 1000):
        raise ValueError("snapshot does not cover registered training history")
    if metadata["coverage_end_exclusive"] < end_ms:
        raise ValueError("snapshot does not cover completed origin")
    return start_ms, end_ms


def _simulation_payload(
    simulation, *, origin_id: str, candidate_id: str, proposal: pd.DataFrame,
    data_snapshot_sha256: str, train_rows_sha256: str, scaler: LabScaler,
) -> dict[str, Any]:
    decisions = simulation.decisions.merge(
        proposal, on=["execution_timestamp", "symbol"], how="left", validate="one_to_one"
    )
    if decisions["proposal"].isna().any():
        raise ValueError("decision/proposal ledger mismatch")
    decisions["origin_id"] = origin_id
    decisions["candidate_id"] = candidate_id
    decisions["action"] = decisions["desired_position"].astype(int)
    decisions["data_snapshot_sha256"] = data_snapshot_sha256
    decisions["train_rows_sha256"] = train_rows_sha256
    decisions["scaler_fit_values_sha256"] = scaler.fit_values_sha256
    required = (
        "origin_id", "signal_timestamp", "execution_timestamp", "symbol", "candidate_id",
        "proposal", "action", "model_sha256", "data_snapshot_sha256",
        "train_rows_sha256", "scaler_fit_values_sha256",
    )
    if decisions["model_sha256"].nunique() != 1:
        raise ValueError("model SHA changed inside candidate/origin")
    return {
        "metrics": simulation.metrics,
        "daily_returns": daily_returns(simulation).tolist(),
        "portfolio": simulation.portfolio.to_dict("records"),
        "trades": simulation.trades.to_dict("records"),
        "ledger": decisions[list(required)].to_dict("records"),
        "ledger_sha256": _frame_sha(decisions[list(required)]),
        "model_sha256": str(decisions["model_sha256"].iloc[0]),
    }


def evaluate_panel_origin(
    panel: pd.DataFrame, contract: dict[str, Any], manifest: dict[str, Any], origin_index: int,
    *, now: datetime | None = None,
) -> dict[str, Any]:
    start_ms, end_ms = origin_preflight(contract, manifest, origin_index, now=now)
    return evaluate_panel_bounds(
        panel, contract, manifest, origin_index, start_ms=start_ms, end_ms=end_ms
    )


def evaluate_panel_bounds(
    panel: pd.DataFrame, contract: dict[str, Any], manifest: dict[str, Any], origin_index: int,
    *, start_ms: int, end_ms: int,
) -> dict[str, Any]:
    """Pure accounting engine; caller is responsible for temporal authorization."""
    timeframe_ms = int(contract["training"]["timeframe_minutes"]) * 60_000
    purge_ms = int(contract["training"]["purge_bars"]) * timeframe_ms
    history_start_ms = int(datetime.fromisoformat(
        contract["training"]["initial_history_start_utc"].replace("Z", "+00:00")
    ).timestamp() * 1000)
    train_start_ms = history_start_ms + timeframe_ms
    train_end_ms = start_ms - purge_ms
    if train_end_ms <= train_start_ms:
        raise ValueError("insufficient training history after purge")

    causal_panel = panel[panel["timestamp"] < end_ms].copy()
    features = build_lab_features(causal_panel)
    symbols = tuple(contract["universe"])
    train = PreparedWindow.from_frame(features, symbols, train_start_ms, train_end_ms, timeframe_ms)
    deploy = PreparedWindow.from_frame(features, symbols, start_ms, end_ms, timeframe_ms)
    scaler = LabScaler.fit(train.values)
    raw_train = causal_panel[
        (causal_panel["timestamp"] >= history_start_ms) & (causal_panel["timestamp"] < train_end_ms)
    ][["timestamp", "symbol", "open", "high", "low", "close", "volume"]]
    train_rows_sha256 = _frame_sha(raw_train)

    origin_id = f"origin_{origin_index + 1:02d}"
    proposal = deploy.base_schedule[["execution_timestamp", "symbol"]].copy()
    proposal["proposal"] = deploy.proposal.astype(int)
    costs = {
        "normal": float(contract["accounting"]["normal_round_trip_bps"]),
        "stress": float(contract["accounting"]["stress_round_trip_bps"]),
    }
    evaluations: dict[str, dict[str, Any]] = {}
    for cost_label, cost_bps in costs.items():
        candidates = {}
        for candidate_id, genome in GENOMES.items():
            simulation = deploy.simulate(genome, scaler, cost_bps, details=True)
            candidates[candidate_id] = _simulation_payload(
                simulation, origin_id=origin_id, candidate_id=candidate_id, proposal=proposal,
                data_snapshot_sha256=manifest["snapshot_sha256"],
                train_rows_sha256=train_rows_sha256, scaler=scaler,
            )
        overlay = np.asarray(candidates["GA_overlay_f7a4ae11"]["daily_returns"], dtype=float)
        ema = np.asarray(candidates["EMA_TrendConsensus"]["daily_returns"], dtype=float)
        no_trade = np.asarray(candidates["NoTrade"]["daily_returns"], dtype=float)
        if not (len(overlay) == len(ema) == len(no_trade) == 3):
            raise ValueError("a three-day origin must have exactly three daily returns")
        evaluations[cost_label] = {
            "candidates": candidates,
            "paired_overlay_minus_ema": (overlay - ema).tolist(),
            "paired_overlay_minus_no_trade": (overlay - no_trade).tolist(),
        }

    return {
        "schema_version": "exia.genetic_primus.prospective_origin_result/1",
        "origin_id": origin_id,
        "origin_index": origin_index,
        "origin_start": start_ms,
        "origin_end_exclusive": end_ms,
        "train_start": train_start_ms,
        "train_end_exclusive": train_end_ms,
        "purge_bars": int(contract["training"]["purge_bars"]),
        "data_snapshot_sha256": manifest["snapshot_sha256"],
        "train_rows_sha256": train_rows_sha256,
        "scaler": scaler.mapping(),
        "evaluations": evaluations,
        "search_evaluations": 0,
        "safety": dict(contract["safety"]),
    }
