"""Offline continuous accounting for a chain of committed three-day origins.

Each origin refits its scaler using only purged training history. Decisions are
then replayed together, so a retrain boundary does not liquidate a position.
Historical V1 results are never rewritten by this evaluator.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .accounting_v2 import simulate_cash_targets
from .measurement_lab_v2 import (
    ALLOW_ALL, NO_TRADE, LabScaler, PreparedWindow, build_lab_features,
    daily_returns,
)
from .prospective_contract_v2 import SCHEMA as CONTRACT_SCHEMA, validate_contract
from .prospective_evaluator_v1 import _frame_sha
from .prospective_manifest_v2 import SCHEMA as MANIFEST_SCHEMA


def _ms(iso_utc: str) -> int:
    return int(datetime.fromisoformat(iso_utc.replace("Z", "+00:00")).timestamp() * 1000)


def validate_continuous_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError("a newly registered future V2 contract is required")
    validate_contract(contract)
    if contract["data_commitment"].get("manifest_version") != 2:
        raise ValueError("V2 origin commitments required")
    if any(contract["safety"].values()):
        raise ValueError("offline safety contract violated")
    training = contract["training"]
    selection = contract.get("model_selection", {})
    if training["window"] != "expanding" or selection.get("window") != training["window"]:
        raise ValueError("selection and deployment must use the same expanding window")
    if selection.get("scaler") != training["scaler"]:
        raise ValueError("selection and deployment scaler rules differ")
    if contract["accounting"].get("terminal_exit") != "completed_prefix_final_candle_close":
        raise ValueError("continuous terminal exit rule required")


def evaluate_committed_chain(
    contract: dict[str, Any],
    origin_inputs: Sequence[tuple[dict[str, Any], pd.DataFrame]],
) -> dict[str, Any]:
    """Evaluate verified V2 manifests in order; caller verifies their file bytes.

    The chain may be a completed prefix. Only the end of that prefix is
    liquidated for reporting; a later prefix is recomputed from origin zero.
    A verified append-only final panel may be reused for every origin; feature
    construction is clipped at each origin's end before fitting or decisions.
    """
    validate_continuous_contract(contract)
    origins = validate_contract(contract)
    if not origin_inputs or len(origin_inputs) > len(origins):
        raise ValueError("a nonempty registered origin prefix is required")
    timeframe_ms = int(contract["training"]["timeframe_minutes"]) * 60_000
    history_start_ms = _ms(contract["training"]["initial_history_start_utc"])
    train_start_ms = history_start_ms + timeframe_ms
    purge_ms = int(contract["training"]["purge_bars"]) * timeframe_ms
    symbols = tuple(contract["universe"])
    overlay_id = contract["candidates"][2]["candidate_id"]
    genomes = {"NoTrade": NO_TRADE, "EMA_TrendConsensus": ALLOW_ALL,
               overlay_id: tuple(contract["candidates"][2]["genome"])}
    schedules: dict[str, list[pd.DataFrame]] = {name: [] for name in genomes}
    bars: list[pd.DataFrame] = []
    lineage: list[pd.DataFrame] = []
    origin_metadata: list[dict[str, Any]] = []

    for index, (manifest, panel) in enumerate(origin_inputs):
        start_ms, end_ms = (int(bound.timestamp() * 1000) for bound in origins[index])
        if manifest.get("schema_version") != MANIFEST_SCHEMA or manifest.get("origin_index") != index:
            raise ValueError("origin manifest order or schema mismatch")
        if manifest.get("metadata", {}).get("coverage_end_exclusive") != end_ms:
            raise ValueError("origin snapshot ends at wrong boundary")
        if manifest["metadata"]["coverage_start"] > history_start_ms:
            raise ValueError("origin snapshot lacks training history")
        if panel["timestamp"].min() > history_start_ms:
            raise ValueError("panel lacks training history")
        train_end_ms = start_ms - purge_ms
        if train_end_ms <= train_start_ms:
            raise ValueError("insufficient purged training history")
        causal_panel = panel.loc[panel["timestamp"] < end_ms].copy()
        if causal_panel.empty or causal_panel["timestamp"].max() != end_ms - timeframe_ms:
            raise ValueError("origin is missing its terminal candle")
        features = build_lab_features(causal_panel)
        train = PreparedWindow.from_frame(
            features, symbols, train_start_ms, train_end_ms, timeframe_ms)
        deploy = PreparedWindow.from_frame(
            features, symbols, start_ms, end_ms, timeframe_ms)
        scaler = LabScaler.fit(train.values)
        raw_train = causal_panel.loc[
            causal_panel["timestamp"].ge(history_start_ms)
            & causal_panel["timestamp"].lt(train_end_ms),
            ["timestamp", "symbol", "open", "high", "low", "close", "volume"],
        ]
        train_sha = _frame_sha(raw_train)
        bars.append(deploy.bars)
        origin_id = f"origin_{index + 1:02d}"
        identity = deploy.base_schedule[["execution_timestamp", "symbol"]].copy()
        identity["origin_id"] = origin_id
        identity["proposal"] = deploy.proposal.astype(int)
        identity["data_snapshot_sha256"] = manifest["snapshot_sha256"]
        identity["train_rows_sha256"] = train_sha
        identity["scaler_fit_values_sha256"] = scaler.fit_values_sha256
        lineage.append(identity)
        origin_metadata.append({
            "origin_id": origin_id, "origin_index": index,
            "origin_start": start_ms, "origin_end_exclusive": end_ms,
            "train_start": train_start_ms, "train_end_exclusive": train_end_ms,
            "purge_bars": int(contract["training"]["purge_bars"]),
            "data_snapshot_sha256": manifest["snapshot_sha256"],
            "train_rows_sha256": train_sha, "scaler": scaler.mapping(),
        })
        for name, genome in genomes.items():
            schedules[name].append(deploy.schedule(genome, scaler))

    first_ms = origin_metadata[0]["origin_start"]
    final_ms = origin_metadata[-1]["origin_end_exclusive"]
    market = pd.concat(bars, ignore_index=True)
    identity = pd.concat(lineage, ignore_index=True)
    costs = {"normal": float(contract["accounting"]["normal_round_trip_bps"]),
             "stress": float(contract["accounting"]["stress_round_trip_bps"])}
    evaluations: dict[str, Any] = {}
    for label, cost_bps in costs.items():
        candidates: dict[str, Any] = {}
        for name, chunks in schedules.items():
            schedule = pd.concat(chunks, ignore_index=True)
            simulation = simulate_cash_targets(
                market, schedule, symbols=symbols, start_timestamp=first_ms,
                end_timestamp=final_ms, timeframe_ms=timeframe_ms,
                round_trip_cost_bps=cost_bps, record_details=True)
            decisions = simulation.decisions.merge(
                identity, on=["execution_timestamp", "symbol"],
                how="left", validate="one_to_one")
            if decisions["origin_id"].isna().any():
                raise ValueError("decision lineage mismatch")
            decisions["candidate_id"] = name
            decisions["action"] = decisions["desired_position"].astype(int)
            ledger_columns = (
                "origin_id", "signal_timestamp", "execution_timestamp", "symbol",
                "candidate_id", "proposal", "action", "position_before",
                "quantity_after", "model_sha256", "data_snapshot_sha256",
                "train_rows_sha256", "scaler_fit_values_sha256")
            ledger = decisions[list(ledger_columns)]
            returns = daily_returns(simulation)
            if len(returns) != 3 * len(origin_inputs) or not np.isfinite(returns).all():
                raise ValueError("continuous daily return count mismatch")
            candidates[name] = {
                "metrics": simulation.metrics,
                "daily_returns": returns.tolist(),
                "origin_daily_returns": [part.tolist() for part in returns.reshape(-1, 3)],
                "portfolio": simulation.portfolio.to_dict("records"),
                "trades": simulation.trades.to_dict("records"),
                "ledger": ledger.to_dict("records"),
                "ledger_sha256": _frame_sha(ledger),
                "model_sha256_by_origin": [
                    str(ledger.loc[ledger["origin_id"] == item["origin_id"],
                                   "model_sha256"].iloc[0]) for item in origin_metadata],
            }
        ga = np.asarray(candidates[overlay_id]["daily_returns"])
        ema = np.asarray(candidates["EMA_TrendConsensus"]["daily_returns"])
        flat = np.asarray(candidates["NoTrade"]["daily_returns"])
        evaluations[label] = {
            "candidates": candidates,
            "paired_overlay_minus_ema": (ga - ema).tolist(),
            "paired_overlay_minus_no_trade": (ga - flat).tolist(),
        }
    return {
        "schema_version": "exia.genetic_primus.prospective_continuous/2",
        "origin_count": len(origin_inputs), "origins": origin_metadata,
        "evaluations": evaluations, "search_evaluations": 0,
        "terminal_exit_only_at_prefix_end": True,
        "safety": dict(contract["safety"]),
    }
