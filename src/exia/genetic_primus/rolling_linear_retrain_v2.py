"""Causal gross-return fit with position-aware three-day decisions.

The model predicts the next origin's gross return. Execution costs belong to
the transition between actual positions, not to every training label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .measurement_lab_v2 import object_sha
from .rolling_linear_retrain_v1 import (
    DT, FEATURES, HORIZON_MS, _iso, fit_at_origin,
)


def fit_gross_at_origin(examples: pd.DataFrame, deploy_start_ms: int,
                        variant: str, *, purge_bars: int = 7,
                        ridge_alpha: float = 0.5,
                        round_trip_cost_bps: float = 16.0,
                        minimum_opportunities: int = 64,
                        minimum_ess: float = 40.0) -> dict:
    """Refit from completed gross labels, including every losing outcome."""
    if (not np.isfinite(round_trip_cost_bps)
            or not 0 <= round_trip_cost_bps < 20_000):
        raise ValueError("invalid execution cost")
    model = fit_at_origin(
        examples, deploy_start_ms, variant, purge_bars=purge_bars,
        ridge_alpha=ridge_alpha, normal_round_trip_bps=0.0,
        minimum_opportunities=minimum_opportunities,
        minimum_ess=minimum_ess,
    )
    model.pop("model_sha256")
    model["schema"] = "exia.genetic_primus.rolling_linear_model/2"
    model["prediction_target"] = "next_complete_3d_origin_gross_open_to_last_close"
    model["decision_round_trip_cost_bps"] = float(round_trip_cost_bps)
    model["decision_rule"] = "flat_enter_if_gross_gt_full_round_trip_cost;held_continue_if_gross_gt_zero"
    model["model_sha256"] = object_sha(model)
    return model


def schedule_with_position(examples: pd.DataFrame, symbols: tuple[str, ...],
                           deploy_start_ms: int, model: dict,
                           previous_targets: dict[str, int]) -> tuple[pd.DataFrame, dict]:
    """One decision per symbol, copied through the three-day origin."""
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("invalid fixed universe")
    if set(previous_targets) != set(symbols) or any(
            type(value) is not int or value not in (0, 1)
            for value in previous_targets.values()):
        raise ValueError("explicit prior long/flat position required for every symbol")
    if (model.get("schema") != "exia.genetic_primus.rolling_linear_model/2"
            or model.get("deploy_start_utc") != _iso(deploy_start_ms)
            or model.get("feature_names") != FEATURES
            or model.get("model_sha256") != object_sha({
                key: value for key, value in model.items() if key != "model_sha256"})):
        raise ValueError("wrong, mutated or stale model")
    cost_bps = model["decision_round_trip_cost_bps"]
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("invalid model execution cost")
    deploy = examples.loc[examples.origin_start.eq(deploy_start_ms)].set_index("symbol")
    if set(deploy.index) != set(symbols) or len(deploy) != len(symbols):
        raise ValueError("incomplete deploy feature universe")
    deploy = deploy.reindex(symbols)
    decisions = {}
    targets = {}
    for symbol, row in deploy.iterrows():
        prior = previous_targets[str(symbol)]
        gross = None
        if model["coefficients"] is not None and bool(row.proposal_long):
            values = row[list(FEATURES)].to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError("non-finite deploy feature")
            normalized = np.clip(
                (values - np.asarray(model["scaler"]["median"], float))
                / np.asarray(model["scaler"]["iqr"], float), -8.0, 8.0)
            beta = np.asarray(model["coefficients"], float)
            gross = float(beta[0] + normalized @ beta[1:])
            if not np.isfinite(gross):
                raise ValueError("non-finite gross prediction")
        fee_rate = cost_bps / 20_000.0
        if fee_rate >= 1.0:
            raise ValueError("execution fee exceeds capital")
        # A flat sleeve buys with cash/(1+fee), then pays fee on the exit.
        # Break-even: (1+gross)*(1-fee)/(1+fee) > 1.
        threshold = 0.0 if prior else (1.0 + fee_rate) / (1.0 - fee_rate) - 1.0
        target = int(gross is not None and gross > threshold)
        targets[str(symbol)] = target
        decisions[str(symbol)] = {
            "position_before": prior, "target": target,
            "predicted_gross_3d": gross,
            "incremental_threshold": threshold,
            "expected_incremental_vs_cash": None if gross is None else gross - threshold,
        }
    times = np.arange(deploy_start_ms, deploy_start_ms + HORIZON_MS, DT)
    schedule = pd.MultiIndex.from_product(
        [times, symbols], names=["execution_timestamp", "symbol"]
    ).to_frame(index=False)
    schedule["signal_timestamp"] = deploy_start_ms
    schedule["target"] = schedule.symbol.map(targets).astype(int)
    schedule["model_sha256"] = model["model_sha256"]
    return schedule, {"targets": targets, "decisions": decisions,
                      "model_sha256": model["model_sha256"]}


def build_rolling_schedule(examples: pd.DataFrame, symbols: tuple[str, ...],
                           first_origin_ms: int, end_exclusive_ms: int,
                           variant: str, *, purge_bars: int = 7,
                           ridge_alpha: float = 0.5,
                           round_trip_cost_bps: float = 16.0,
                           minimum_opportunities: int = 64,
                           minimum_ess: float = 40.0) -> tuple[pd.DataFrame, list[dict]]:
    """Refit every origin and carry the last target into its successor."""
    if end_exclusive_ms <= first_origin_ms or (end_exclusive_ms - first_origin_ms) % HORIZON_MS:
        raise ValueError("complete nonempty three-day origins required")
    if examples.duplicated(["origin_start", "symbol"]).any():
        raise ValueError("duplicate origin example")
    previous = {symbol: 0 for symbol in symbols}
    schedules = []
    ledger = []
    for origin in range(first_origin_ms, end_exclusive_ms, HORIZON_MS):
        model = fit_gross_at_origin(
            examples, origin, variant, purge_bars=purge_bars,
            ridge_alpha=ridge_alpha,
            round_trip_cost_bps=round_trip_cost_bps,
            minimum_opportunities=minimum_opportunities,
            minimum_ess=minimum_ess,
        )
        schedule, decision = schedule_with_position(
            examples, symbols, origin, model, previous)
        ledger.append({"origin_start": origin, "model": model, "decision": decision})
        schedules.append(schedule)
        previous = decision["targets"]
    return pd.concat(schedules, ignore_index=True), ledger
