from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .fitness_v4 import daily_portfolio
from .measurement import BootstrapBank
from .simulator import PrimusSimulation


@dataclass(frozen=True)
class PromotionProtocol:
    source_path: Path
    payload: Mapping[str, Any]
    sha256: str


@dataclass(frozen=True)
class PromotionDecision:
    passed: bool
    failed_gates: tuple[str, ...]
    metrics: Mapping[str, Any]


def load_promotion_protocol(path: Path) -> PromotionProtocol:
    source = path.resolve()
    raw = source.read_bytes()
    payload = json.loads(raw)
    expected = {
        "schema_version": "exia.genetic_primus.promotion_gate/1",
        "gate_id": "genetic_primus_challenger_gate_v1",
        "bootstrap_alpha": 0.05,
        "minimum_positive_outer_slice_fraction": 0.65,
        "stress_round_trip_cost_bps": 24.0,
        "max_drawdown_degradation": 0.0,
        "max_turnover_fills_per_day": 12.0,
        "max_direction_share": 0.9,
        "max_state_share": 0.8,
        "max_symbol_share": 0.5,
        "max_action_total_variation": 0.25,
        "max_normalized_model_drift": 0.75,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("promotion gate settings are not sealed")
    safety = payload.get("safety", {})
    expected_safety = {
        "credentials_allowed",
        "paper_allowed",
        "live_allowed",
        "orders_enabled",
        "promotion_authority",
    }
    if set(safety) != expected_safety or any(safety[key] is not False for key in expected_safety):
        raise PermissionError("promotion protocol must remain fail-closed")
    return PromotionProtocol(source, payload, hashlib.sha256(raw).hexdigest())


def paired_daily_returns(
    challenger: PrimusSimulation,
    incumbent: PrimusSimulation,
) -> pd.DataFrame:
    left = daily_portfolio(challenger.portfolio).rename(columns={"net_bps": "challenger_net_bps"})
    right = daily_portfolio(incumbent.portfolio).rename(columns={"net_bps": "incumbent_net_bps"})
    if not np.array_equal(left["timestamp"].to_numpy(np.int64), right["timestamp"].to_numpy(np.int64)):
        raise ValueError("paired promotion simulations must have identical daily timestamps")
    return pd.DataFrame(
        {
            "timestamp": left["timestamp"].to_numpy(np.int64),
            "challenger_net_bps": left["challenger_net_bps"].to_numpy(float),
            "incumbent_net_bps": right["incumbent_net_bps"].to_numpy(float),
        }
    )


def paired_lcb(
    paired: pd.DataFrame,
    *,
    bank: BootstrapBank,
    alpha: float,
) -> float:
    timestamps = paired["timestamp"].to_numpy(np.int64)
    difference = (
        paired["challenger_net_bps"].to_numpy(float)
        - paired["incumbent_net_bps"].to_numpy(float)
    )
    indices = bank.indices_for("overall", timestamps)
    return float(np.quantile(difference[indices].mean(axis=1), float(alpha)))


def action_total_variation(
    challenger: Mapping[str, int],
    incumbent: Mapping[str, int],
) -> float:
    keys = ("SHORT", "FLAT", "LONG")
    left = np.asarray([challenger.get(key, 0) for key in keys], dtype=float)
    right = np.asarray([incumbent.get(key, 0) for key in keys], dtype=float)
    left /= max(1.0, left.sum())
    right /= max(1.0, right.sum())
    return float(0.5 * np.abs(left - right).sum())


def ensemble_centroid(genomes: Sequence[np.ndarray]) -> np.ndarray:
    if not genomes:
        raise ValueError("model drift requires at least one genome")
    values = [np.asarray(genome, dtype=np.float64).reshape(-1) for genome in genomes]
    if len({len(value) for value in values}) != 1:
        raise ValueError("model drift genomes must have identical shapes")
    return np.mean(values, axis=0)


def normalized_model_drift(challenger: np.ndarray, incumbent: np.ndarray) -> float:
    left = np.asarray(challenger, dtype=np.float64).reshape(-1)
    right = np.asarray(incumbent, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("model drift vectors must have identical shapes")
    scale = max(float(np.linalg.norm(right)), np.sqrt(len(right)) * 0.05)
    return float(np.linalg.norm(left - right) / scale)


def _drawdown_gate_failures(
    *,
    challenger_drawdown: float,
    incumbent_drawdown: float,
    incumbent_turnover_fills: float,
    config: Mapping[str, Any],
) -> tuple[list[str], bool, str]:
    """Use an absolute budget; compare only against an incumbent that traded."""
    failures: list[str] = []
    incumbent_is_trading = incumbent_turnover_fills >= float(
        config.get("incumbent_minimum_turnover_fills", 1.0)
    )
    if "max_absolute_drawdown" not in config:
        if challenger_drawdown > incumbent_drawdown + float(config["max_drawdown_degradation"]):
            failures.append("drawdown_vs_incumbent")
        return failures, incumbent_is_trading, "legacy_relative"
    if challenger_drawdown > float(config["max_absolute_drawdown"]):
        failures.append("absolute_drawdown_budget")
    if incumbent_is_trading and challenger_drawdown > incumbent_drawdown + float(
        config["max_drawdown_degradation"]
    ):
        failures.append("drawdown_vs_incumbent")
    mode = "absolute_and_relative" if incumbent_is_trading else "absolute_no_trade_transition"
    return failures, incumbent_is_trading, mode


def evaluate_promotion_gate(
    *,
    paired: pd.DataFrame,
    stress_paired: pd.DataFrame,
    outer_slices: Sequence[Mapping[str, str]],
    challenger_metrics: Mapping[str, Any],
    incumbent_metrics: Mapping[str, Any],
    challenger_actions: Mapping[str, int],
    incumbent_actions: Mapping[str, int],
    challenger_centroid: np.ndarray,
    incumbent_centroid: np.ndarray,
    selection_bank: BootstrapBank,
    audit_bank: BootstrapBank,
    protocol: PromotionProtocol,
) -> PromotionDecision:
    if selection_bank.seed == audit_bank.seed or selection_bank.content_sha256 == audit_bank.content_sha256:
        raise ValueError("promotion audit bank must be independent from selection")
    alpha = float(protocol.payload["bootstrap_alpha"])
    selection_lcb = paired_lcb(paired, bank=selection_bank, alpha=alpha)
    audit_lcb = paired_lcb(paired, bank=audit_bank, alpha=alpha)
    stress_lcb = paired_lcb(stress_paired, bank=selection_bank, alpha=alpha)
    difference = paired["challenger_net_bps"] - paired["incumbent_net_bps"]
    positive_slices = 0
    valid_slices = 0
    for outer_slice in outer_slices:
        start = int(pd.Timestamp(outer_slice["start"]).timestamp() * 1000)
        end = int(pd.Timestamp(outer_slice["end"]).timestamp() * 1000)
        local = difference.loc[paired["timestamp"].ge(start) & paired["timestamp"].lt(end)]
        if len(local):
            valid_slices += 1
            positive_slices += int(float(local.mean()) > 0.0)
    positive_fraction = float(positive_slices / valid_slices) if valid_slices else 0.0
    action_drift = action_total_variation(challenger_actions, incumbent_actions)
    model_drift = normalized_model_drift(challenger_centroid, incumbent_centroid)
    challenger_drawdown = float(challenger_metrics["max_drawdown"])
    incumbent_drawdown = float(incumbent_metrics["max_drawdown"])
    failures: list[str] = []
    if selection_lcb <= 0.0:
        failures.append("paired_selection_lcb")
    if positive_fraction < float(protocol.payload["minimum_positive_outer_slice_fraction"]):
        failures.append("positive_outer_slices")
    if stress_lcb <= 0.0:
        failures.append("stress_cost_lcb")
    drawdown_failures, incumbent_is_trading, drawdown_gate_mode = _drawdown_gate_failures(
        challenger_drawdown=challenger_drawdown,
        incumbent_drawdown=incumbent_drawdown,
        incumbent_turnover_fills=float(incumbent_metrics.get("turnover_fills", 0.0)),
        config=protocol.payload,
    )
    failures.extend(drawdown_failures)
    for metric, gate, failure in (
        ("fills_per_day", "max_turnover_fills_per_day", "turnover"),
        ("max_direction_share", "max_direction_share", "direction_concentration"),
        ("max_state_share", "max_state_share", "state_concentration"),
        ("max_symbol_share", "max_symbol_share", "symbol_concentration"),
    ):
        if float(challenger_metrics[metric]) > float(protocol.payload[gate]):
            failures.append(failure)
    if audit_lcb <= 0.0:
        failures.append("paired_audit_lcb")
    action_drift_limit = float(
        protocol.payload["max_functional_action_drift"]
        if "max_functional_action_drift" in protocol.payload
        else protocol.payload["max_action_total_variation"]
    )
    model_drift_limit = float(
        protocol.payload["max_functional_logit_drift"]
        if "max_functional_logit_drift" in protocol.payload
        else protocol.payload["max_normalized_model_drift"]
    )
    if action_drift > action_drift_limit:
        failures.append("action_drift")
    if model_drift > model_drift_limit:
        failures.append("model_drift")
    metrics = {
        "paired_selection_lcb_bps": selection_lcb,
        "paired_audit_lcb_bps": audit_lcb,
        "stress_cost_lcb_bps": stress_lcb,
        "positive_outer_slices": positive_slices,
        "outer_slice_count": valid_slices,
        "positive_outer_slice_fraction": positive_fraction,
        "challenger_max_drawdown": challenger_drawdown,
        "incumbent_max_drawdown": incumbent_drawdown,
        "incumbent_is_trading": incumbent_is_trading,
        "drawdown_gate_mode": drawdown_gate_mode,
        "action_total_variation": action_drift,
        "normalized_model_drift": model_drift,
        "promotion_gate_passed": not failures,
        "failed_gates": failures,
    }
    return PromotionDecision(not failures, tuple(failures), metrics)


def _outer_block_lcb(
    values: np.ndarray,
    *,
    samples: int,
    block_origins: int,
    seed: int,
    alpha: float = 0.05,
) -> float:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(data) < block_origins:
        return -1.0e9
    rng = np.random.default_rng(seed)
    starts = np.arange(len(data) - block_origins + 1)
    blocks = int(np.ceil(len(data) / block_origins))
    selected = rng.choice(starts, size=(int(samples), blocks), replace=True)
    indices = (
        selected[:, :, None] + np.arange(block_origins)[None, None, :]
    ).reshape(int(samples), -1)[:, : len(data)]
    return float(np.quantile(data[indices].mean(axis=1), alpha))


def evaluate_outer_promotion_gate(
    *,
    benchmarks: pd.DataFrame,
    origins: pd.DataFrame,
    config: Mapping[str, Any],
    benchmark_portfolio: pd.DataFrame | None = None,
) -> PromotionDecision:
    """Evaluate promotion only from completed three-day deployment outcomes."""
    if (
        "challenger_deployable" in origins.columns
        and not bool(origins["challenger_deployable"].fillna(False).any())
    ):
        failures = (
            "no_eligible_challenger_evidence",
            "minimum_completed_outer_origins",
            "paired_selection_lcb",
            "paired_audit_lcb",
            "stress_cost_lcb",
            "positive_outer_slices",
            "temporal_predeployment_gate",
        )
        metrics = {
            "completed_outer_origins": 0,
            "paired_selection_lcb_bps": -1.0e9,
            "paired_audit_lcb_bps": -1.0e9,
            "stress_cost_lcb_bps": -1.0e9,
            "positive_outer_slice_fraction": 0.0,
            "challenger_max_drawdown": 0.0,
            "incumbent_max_drawdown": 0.0,
            "fills_per_day": 0.0,
            "max_action_total_variation": 0.0,
            "max_normalized_model_drift": 0.0,
            "max_functional_action_drift": 0.0,
            "max_functional_logit_drift": 0.0,
            "action_drift_semantics": "fixed_hash_anchor_action_disagreement",
            "model_drift_semantics": "fixed_hash_anchor_bounded_logit_distance",
            "predeployment_temporal_gates_passed": False,
            "no_eligible_challenger_evidence": True,
            "promotion_gate_passed": False,
            "promotion_approved": False,
            "promotion_authority": False,
            "failed_gates": list(failures),
        }
        return PromotionDecision(False, failures, metrics)
    outer = benchmarks.loc[benchmarks["evidence_role"].eq("outer_deployment")]
    pivot = outer.pivot(index="origin_index", columns="agent_id", values="net_bps")
    required = {
        "GeneticPrimusChallenger",
        "GeneticPrimusActive",
        "GeneticPrimusChallengerStress",
        "GeneticPrimusActiveStress",
    }
    missing = sorted(required - set(pivot.columns))
    if missing:
        raise ValueError(f"outer promotion evidence is missing agents: {missing}")
    paired = pivot.dropna(subset=sorted(required))
    normal_delta = (
        paired["GeneticPrimusChallenger"] - paired["GeneticPrimusActive"]
    ).to_numpy(float)
    stress_delta = (
        paired["GeneticPrimusChallengerStress"]
        - paired["GeneticPrimusActiveStress"]
    ).to_numpy(float)
    completed = int(len(normal_delta))
    positive_fraction = float(np.mean(normal_delta > 0.0)) if completed else 0.0
    samples = int(config["outer_bootstrap_samples"])
    block = int(config["outer_block_origins"])
    selection_lcb = _outer_block_lcb(
        normal_delta,
        samples=samples,
        block_origins=block,
        seed=int(config["selection_seed"]),
    )
    audit_lcb = _outer_block_lcb(
        normal_delta,
        samples=samples,
        block_origins=block,
        seed=int(config["audit_seed"]),
    )
    stress_lcb = _outer_block_lcb(
        stress_delta,
        samples=samples,
        block_origins=block,
        seed=int(config["audit_seed"]) + 1,
    )
    candidate = outer.loc[outer["agent_id"].eq("GeneticPrimusChallenger")]
    active = outer.loc[outer["agent_id"].eq("GeneticPrimusActive")]
    def aggregate_drawdown(agent_id: str, fallback: pd.DataFrame) -> float:
        if benchmark_portfolio is None or benchmark_portfolio.empty:
            return float(fallback["max_drawdown"].max()) if len(fallback) else 0.0
        local = benchmark_portfolio.loc[
            benchmark_portfolio["agent_id"].eq(agent_id)
        ].sort_values(["origin_index", "timestamp"], kind="stable")
        if local.empty:
            return 0.0
        equity = np.cumprod(1.0 + local["net_return"].to_numpy(float))
        peak = np.maximum.accumulate(equity)
        return float(np.max(1.0 - equity / peak))

    candidate_drawdown = aggregate_drawdown("GeneticPrimusChallenger", candidate)
    active_drawdown = aggregate_drawdown("GeneticPrimusActive", active)
    active_turnover_fills = float(active["turnover_fills"].sum())
    deployment_days = max(1, completed * 3)
    fills_per_day = float(candidate["turnover_fills"].sum() / deployment_days)
    action_drift = float(origins["challenger_retrain_action_drift"].max())
    model_drift = float(origins["challenger_retrain_model_drift"].max())
    predeployment_passed = bool(origins["predeployment_training_gate_passed"].all())
    failures: list[str] = []
    if completed < int(config["minimum_completed_outer_origins"]):
        failures.append("minimum_completed_outer_origins")
    if selection_lcb <= 0.0:
        failures.append("paired_selection_lcb")
    if audit_lcb <= 0.0:
        failures.append("paired_audit_lcb")
    if stress_lcb <= 0.0:
        failures.append("stress_cost_lcb")
    if positive_fraction < float(config["minimum_positive_outer_slice_fraction"]):
        failures.append("positive_outer_slices")
    drawdown_failures, incumbent_is_trading, drawdown_gate_mode = _drawdown_gate_failures(
        challenger_drawdown=candidate_drawdown,
        incumbent_drawdown=active_drawdown,
        incumbent_turnover_fills=active_turnover_fills,
        config=config,
    )
    failures.extend(drawdown_failures)
    if fills_per_day > float(config["max_turnover_fills_per_day"]):
        failures.append("turnover")
    for column, threshold, failure in (
        ("max_direction_share", "max_direction_share", "direction_concentration"),
        ("max_state_share", "max_state_share", "state_concentration"),
        ("max_symbol_share", "max_symbol_share", "symbol_concentration"),
    ):
        if float(candidate[column].max()) > float(config[threshold]):
            failures.append(failure)
    action_drift_limit = float(
        config["max_functional_action_drift"]
        if "max_functional_action_drift" in config
        else config["max_action_total_variation"]
    )
    model_drift_limit = float(
        config["max_functional_logit_drift"]
        if "max_functional_logit_drift" in config
        else config["max_normalized_model_drift"]
    )
    if action_drift > action_drift_limit:
        failures.append("action_drift")
    if model_drift > model_drift_limit:
        failures.append("model_drift")
    global_evidence = config.get("global_multiple_testing")
    dsr_probability = (
        float(global_evidence["deflated_sharpe_probability"])
        if global_evidence is not None
        and global_evidence.get("deflated_sharpe_probability") is not None
        else None
    )
    pbo = (
        float(global_evidence["pbo"])
        if global_evidence is not None and global_evidence.get("pbo") is not None
        else None
    )
    global_evidence_ready = bool(
        global_evidence is not None and global_evidence.get("promotion_eligible", False)
    )
    if bool(config.get("dsr_required", False) or config.get("pbo_required", False)) and not global_evidence_ready:
        failures.append("global_multiple_testing_incomplete")
    if bool(config.get("dsr_required", False)) and (
        dsr_probability is None
        or dsr_probability < float(config["minimum_dsr_probability"])
    ):
        failures.append("global_dsr")
    if bool(config.get("pbo_required", False)) and (
        pbo is None or pbo > float(config["maximum_pbo"])
    ):
        failures.append("global_pbo")
    if not predeployment_passed:
        failures.append("temporal_predeployment_gate")
    metrics = {
        "completed_outer_origins": completed,
        "paired_selection_lcb_bps": selection_lcb,
        "paired_audit_lcb_bps": audit_lcb,
        "stress_cost_lcb_bps": stress_lcb,
        "positive_outer_slice_fraction": positive_fraction,
        "challenger_max_drawdown": candidate_drawdown,
        "incumbent_max_drawdown": active_drawdown,
        "incumbent_turnover_fills": active_turnover_fills,
        "incumbent_is_trading": incumbent_is_trading,
        "drawdown_gate_mode": drawdown_gate_mode,
        "fills_per_day": fills_per_day,
        "max_action_total_variation": action_drift,
        "max_normalized_model_drift": model_drift,
        "max_functional_action_drift": action_drift,
        "max_functional_logit_drift": model_drift,
        "action_drift_semantics": "fixed_hash_anchor_action_disagreement",
        "model_drift_semantics": "fixed_hash_anchor_bounded_logit_distance",
        "global_deflated_sharpe_probability": dsr_probability,
        "global_probability_of_backtest_overfitting": pbo,
        "global_multiple_testing_evidence_ready": global_evidence_ready,
        "predeployment_temporal_gates_passed": predeployment_passed,
        "promotion_gate_passed": not failures,
        "promotion_approved": False,
        "promotion_authority": False,
        "failed_gates": failures,
    }
    return PromotionDecision(not failures, tuple(failures), metrics)
