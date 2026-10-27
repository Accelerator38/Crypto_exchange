from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from .benchmarks_v4 import (
    NoTradePolicy,
    mean_reversion_policy,
    momentum_policy,
    relative_momentum_policy,
    summarize_benchmark,
)
from .contracts import STATIC_FEATURE_NAMES
from .features import RobustScaler
from .measurement import canonical_sha256
from .policy_v4 import (
    EnsemblePolicyV4,
    GeneticPrimusPolicyV4,
    V4NetworkShape,
    functional_policy_drift_v4,
)
from .promotion_v4 import (
    action_total_variation,
)
from .simulator import (
    DECISION_COLUMNS,
    LEDGER_COLUMNS,
    overlay_opportunity_mask,
    simulate_policy,
)


@dataclass(frozen=True)
class RollingProtocol:
    source_path: Path
    payload: Mapping[str, Any]
    sha256: str

    @property
    def timeframe_ms(self) -> int:
        return int(self.payload["timeframe_minutes"]) * 60 * 1000

    @property
    def purge_ms(self) -> int:
        return int(self.payload["label_purge_bars"]) * self.timeframe_ms

    @property
    def deployment_ms(self) -> int:
        return int(pd.Timedelta(days=int(self.payload["deployment_horizon_days"])).total_seconds() * 1000)

    @property
    def cadence_ms(self) -> int:
        return int(pd.Timedelta(days=int(self.payload["retrain_cadence_days"])).total_seconds() * 1000)

    @property
    def evaluation_interval_ms(self) -> int:
        days = int(self.payload.get("evaluation_interval_days", self.payload["deployment_horizon_days"]))
        return int(pd.Timedelta(days=days).total_seconds() * 1000)


def _timestamp(value: str) -> int:
    return int(pd.Timestamp(value).timestamp() * 1000)


def fixed_baseline_scaler() -> RobustScaler:
    """Identity transform for fixed baselines defined on raw contract features."""
    size = len(STATIC_FEATURE_NAMES)
    return RobustScaler(median=np.zeros(size), scale=np.ones(size))


def fixed_baseline_sha256(policy_id: str) -> str:
    """Stable identity for a fixed policy; deliberately excludes train-window state."""
    if not policy_id:
        raise ValueError("fixed baseline policy id cannot be empty")
    return canonical_sha256(
        {
            "schema_version": "exia.genetic_primus.benchmark_policy/1",
            "policy_id": policy_id,
            "feature_space": "raw_static_contract_v1",
        }
    )


def validate_rolling_output_directory(output_dir: Path, *, resume_partial: bool) -> None:
    """Reject accidental overwrite and resume of completed rolling outputs."""
    if not resume_partial:
        if output_dir.exists():
            raise FileExistsError(f"rolling output directory already exists: {output_dir}")
        return
    if not output_dir.is_dir():
        raise FileNotFoundError(f"partial rolling output directory does not exist: {output_dir}")
    if (output_dir / "manifest.json").exists() or any(output_dir.glob("*.parquet")):
        raise FileExistsError("resume-partial refuses completed or root-artifact output")


def load_rolling_protocol(path: Path) -> RollingProtocol:
    source = path.resolve()
    raw = source.read_bytes()
    payload = json.loads(raw)
    schema = payload.get("schema_version")
    overlay_v1 = schema == "exia.genetic_primus.rolling.overlay_3d/1"
    overlay_v2 = schema == "exia.genetic_primus.rolling.overlay_3d/2"
    overlay_v3 = schema == "exia.genetic_primus.rolling.overlay_3d/3"
    overlay_v4 = schema == "exia.genetic_primus.rolling.overlay_3d/4"
    overlay_v5 = schema == "exia.genetic_primus.rolling.overlay_3d/5"
    overlay_v6 = schema == "exia.genetic_primus.rolling.overlay_3d/6"
    overlay_v4_contract = overlay_v4 or overlay_v5 or overlay_v6
    overlay_temporal = overlay_v3 or overlay_v4_contract
    overlay_modern = overlay_v2 or overlay_temporal
    overlay_schema = overlay_v1 or overlay_modern
    if schema not in {
        "exia.genetic_primus.rolling.3d/1",
        "exia.genetic_primus.rolling.overlay_3d/1",
        "exia.genetic_primus.rolling.overlay_3d/2",
        "exia.genetic_primus.rolling.overlay_3d/3",
        "exia.genetic_primus.rolling.overlay_3d/4",
        "exia.genetic_primus.rolling.overlay_3d/5",
        "exia.genetic_primus.rolling.overlay_3d/6",
    }:
        raise ValueError("unexpected rolling protocol schema")
    expected_protocol_id = (
        "genetic_primus_overlay_3d_v6"
        if overlay_v6
        else "genetic_primus_overlay_3d_v5"
        if overlay_v5
        else "genetic_primus_overlay_3d_v4"
        if overlay_v4
        else "genetic_primus_overlay_3d_v3"
        if overlay_v3
        else "genetic_primus_overlay_3d_v2"
        if overlay_v2
        else "genetic_primus_overlay_3d_v1"
        if overlay_v1
        else "genetic_primus_rolling_3d_v1"
    )
    if payload.get("protocol_id") != expected_protocol_id:
        raise ValueError("unexpected rolling protocol id")
    if int(payload.get("timeframe_minutes", 0)) != 240:
        raise ValueError("rolling protocol requires 4h bars")
    if int(payload.get("deployment_horizon_days", 0)) != 3:
        raise ValueError("deployment horizon must be three days")
    if int(payload.get("retrain_cadence_days", 0)) != 3:
        raise ValueError("retrain cadence must be three days")
    if int(payload.get("label_purge_bars", 0)) != 7:
        raise ValueError("rolling labels require a seven-bar purge")
    if payload.get("position_boundary_policy") != "force_flat_at_deployment_end":
        raise ValueError("rolling v1 requires explicit flat deployment boundaries")
    training = payload.get("genetic_training", {})
    expected_training = {
        "architecture_id": "linear",
        "seed_base": 303000,
        "seed_stride": 1009,
        "workers": 1,
        "island_count": 5,
        "population_size": 4,
        "generations": 5,
        "elite_count": 1,
        "tournament_size": 2,
        "crossover_rate": 0.7,
        "mutation_rate": 0.15,
        "mutation_sigma": 0.08,
        "random_immigrant_fraction": 0.25,
        "ensemble_sizes": [3, 5],
    }
    if overlay_modern:
        expected_training.update(
            {
                "output_mode": "binary_allow",
                "maximum_active_features": 8,
                "ensemble_sizes": [1, 3],
            }
        )
    if overlay_v4_contract:
        expected_training["maximum_active_features"] = 6
    if overlay_v6:
        expected_training["maximum_active_features"] = 4
    if training != expected_training:
        raise ValueError("rolling genetic training settings are not sealed")
    recency = payload.get("recency_fitness", {})
    expected_recency = {
        "half_life_days": [30, 90, 180],
        "uniform_anchor": 0.2,
        "rolling_fold_count": 12,
        "recent_lcb_weight": 0.7,
        "worst_window_lcb_weight": 0.3,
        "minimum_ess_days": {"30": 60.0, "90": 120.0, "180": 180.0},
        "drawdown_penalty_bps_per_pct": 1.0,
        "turnover_penalty_bps_per_fill_per_day": 0.5,
        "complexity_penalty_bps_per_100_parameters": 1.0,
        "active_weight_threshold": 0.05,
        "active_parameter_penalty_bps_per_100": 0.5,
        "active_feature_penalty_bps": 0.1,
        "action_instability_penalty_bps": 20.0,
    }
    if overlay_modern:
        expected_recency.update(
            {
                "weighting_mode": "uniform",
                "half_life_days": [180],
                "uniform_anchor": 1.0,
                "minimum_ess_days": {"180": 180.0},
            }
        )
    if overlay_v3:
        expected_recency["minimum_ess_days"] = {"180": 30.0}
    if overlay_v4_contract:
        expected_recency.update(
            {
                "weighting_mode": "half_life",
                "half_life_days": [90],
                "uniform_anchor": 0.2,
                "minimum_ess_days": {"90": 120.0},
                "turnover_penalty_bps_per_fill_per_day": 2.0,
                "action_instability_penalty_bps": 40.0,
            }
        )
    if overlay_v6:
        expected_recency["minimum_ess_fraction_of_observed_days"] = 0.75
    if recency != expected_recency:
        raise ValueError("rolling recency fitness settings are not sealed")
    if overlay_schema:
        expected_overlay = {
            "mode": "momentum_abstention",
            "baseline_policy_id": "Momentum_return_1",
            "baseline_feature": "return_1",
            "minimum_baseline_move_bps": 16.0,
            "maximum_allowed_fraction": 0.35,
            "excess_coverage_penalty_bps": 40.0,
        }
        if overlay_modern:
            expected_overlay = {
                "mode": "momentum_abstention_binary",
                "baseline_policy_id": "Momentum_return_1",
                "baseline_feature": "return_1",
                "minimum_baseline_move_bps": 16.0,
                "minimum_allowed_fraction": 0.05,
                "maximum_allowed_fraction": 0.35,
                "coverage_penalty_bps": 40.0,
            }
        if overlay_v4_contract:
            expected_overlay.update(
                {
                    "minimum_baseline_move_bps": 24.0,
                    "minimum_allowed_fraction": 0.03,
                    "maximum_allowed_fraction": 0.25,
                    "coverage_penalty_bps": 80.0,
                }
            )
        if overlay_v6:
            expected_overlay = {
                "mode": "momentum_abstention_binary",
                "baseline_policy_id": "RelativeMomentumRank_return_12",
                "baseline_feature": "relative_momentum_rank",
                "baseline_threshold_mode": "absolute_feature",
                "minimum_baseline_abs_value": 0.5,
                "maximum_allowed_fraction": 0.20,
                "coverage_penalty_bps": 80.0,
            }
        if payload.get("policy_overlay") != expected_overlay:
            raise ValueError("overlay policy settings are not sealed")
        expected_paired = {
            "references": ["NoTrade", "Momentum_return_1", "Incumbent"],
            "minimum_lcb_bps": 0.0,
            "negative_lcb_penalty_multiplier": 1.0,
        }
        if overlay_modern:
            expected_paired = {
                "references": ["NoTrade", "Momentum_return_1", "Incumbent"],
                "absolute_reference": "NoTrade",
                "baseline_reference": "Momentum_return_1",
                "minimum_lcb_bps": 0.0,
            }
        if overlay_v6:
            expected_paired = {
                "references": [
                    "NoTrade",
                    "RelativeMomentumRank_return_12",
                    "Incumbent",
                ],
                "absolute_reference": "NoTrade",
                "baseline_reference": "RelativeMomentumRank_return_12",
                "minimum_lcb_bps": 0.0,
            }
        if payload.get("paired_incremental_fitness") != expected_paired:
            raise ValueError("paired incremental fitness settings are not sealed")
        expected_multiple_testing = {
            "max_hypotheses_per_origin": 100,
            "audit_finalists": 5,
            "selection_bias_penalty": "gaussian_maximum",
            "selection_bias_penalty_multiplier": 1.0,
        }
        if overlay_modern:
            expected_multiple_testing = {
                "max_hypotheses_per_origin": 107,
                "audit_finalists": 5,
                "selection_bias_penalty": "gaussian_maximum_paired",
                "selection_bias_penalty_multiplier": 1.0,
            }
        if overlay_temporal:
            expected_multiple_testing.update(
                {
                    "selection_candidate_upper_bound": 100,
                    "ensemble_hypotheses": 2,
                }
            )
        if payload.get("adaptive_multiple_testing") != expected_multiple_testing:
            raise ValueError("adaptive multiple-testing settings are not sealed")
    if overlay_modern:
        if int(payload.get("training_window_days", 0)) != 730:
            raise ValueError("overlay v2 requires a rolling 730-day training window")
        expected_evidence = {
            "training_folds_are_outer_slices": False,
            "minimum_completed_outer_origins": 60,
            "minimum_positive_outer_slice_fraction": 0.65,
            "per_origin_training_gate_can_promote": False,
        }
        if overlay_temporal:
            expected_evidence.update(
                {
                    "outer_bootstrap_samples": 2000,
                    "outer_block_origins": 5,
                    "selection_seed": 930001,
                    "audit_seed": 940001,
                    "stress_round_trip_cost_bps": 24.0,
                    "max_drawdown_degradation": 0.0,
                    "max_turnover_fills_per_day": 12.0,
                    "max_direction_share": 0.9,
                    "max_state_share": 0.8,
                    "max_symbol_share": 0.5,
                    "max_action_total_variation": 0.25,
                    "max_normalized_model_drift": 0.75,
                }
            )
        if overlay_v5 or overlay_v6:
            expected_evidence.pop("max_action_total_variation")
            expected_evidence.pop("max_normalized_model_drift")
            expected_evidence.update(
                {
                    "max_absolute_drawdown": 0.10,
                    "incumbent_minimum_turnover_fills": 1,
                    "max_functional_action_drift": 0.20,
                    "max_functional_logit_drift": 0.15,
                }
            )
        if payload.get("promotion_evidence") != expected_evidence:
            raise ValueError("overlay v2 promotion evidence settings are not sealed")
    if overlay_temporal:
        expected_diversity = {
            "anchor_rows": 512,
            "minimum_action_disagreement": 0.10,
            "selection_method": "greedy_maximin_action_hamming",
        }
        if payload.get("search_diversity") != expected_diversity:
            raise ValueError("overlay v3 search diversity settings are not sealed")
        expected_temporal = {
            "validation_days": 60,
            "audit_days": 60,
            "boundary_purge_bars": 7,
            "inner_fold_count": 12,
            "validation_fold_count": 6,
            "audit_fold_count": 6,
            "scaler_fit_scope": "inner_train_only",
            "generation_selection_scope": "inner_train",
            "ensemble_selection_scope": "temporal_validation",
            "audit_used_for_selection": False,
        }
        if payload.get("temporal_validation") != expected_temporal:
            raise ValueError("overlay v3 temporal validation settings are not sealed")
    if overlay_v5 or overlay_v6:
        if payload.get("stability_selection") != {
            "minimum_active_frequency": 0.60,
            "minimum_sign_agreement": 0.80,
            "active_weight_threshold": 0.05,
            "incumbent_sign_control": True,
        }:
            raise ValueError("overlay v5 stability selection settings are not sealed")
        expected_abstention_evidence = {
            "minimum_opportunities": 180,
            "minimum_trades": 12,
            "minimum_trade_days": 12,
        }
        if overlay_v6:
            expected_abstention_evidence.update(
                {"minimum_trades": 0, "minimum_trade_days": 0}
            )
        if payload.get("abstention_evidence") != expected_abstention_evidence:
            raise ValueError("overlay v5 abstention evidence settings are not sealed")
        if payload.get("opportunity_normalized_concentration") != {
            "maximum_allocation_lift": 2.0,
            "minimum_bucket_opportunities": 12,
            "penalty_bps_per_excess_lift": 25.0,
        }:
            raise ValueError("overlay v5 concentration settings are not sealed")
        if payload.get("execution_controls") != {
            "entry_logit_threshold": 0.15,
            "exit_logit_threshold": -0.05,
            "cooldown_bars": 2,
        }:
            raise ValueError("overlay v5 hysteresis settings are not sealed")
        expected_registry = {
            "registry_path": "configs/genetic_primus_objective_registry_v2.json",
            "objective_id": "overlay_3d_v5_sealed",
            "dsr_required": True,
            "pbo_required": True,
            "minimum_dsr_probability": 0.95,
            "maximum_pbo": 0.50,
            "evidence_path": "Reports/Exia/Genetic_Primus/global_multiple_testing_v2.json",
        }
        if overlay_v6:
            expected_registry.update(
                {
                    "registry_path": "configs/genetic_primus_objective_registry_v3.json",
                    "objective_id": "overlay_3d_v6_relative_opportunity_sealed",
                    "evidence_path": "Reports/Exia/Genetic_Primus/global_multiple_testing_v3.json",
                }
            )
        if payload.get("objective_registry") != expected_registry:
            raise ValueError("overlay v5 objective registry settings are not sealed")
    if not (
        _timestamp(payload["training_start"])
        < _timestamp(payload["validation_start"])
        < _timestamp(payload["validation_end"])
    ):
        raise ValueError("rolling timestamps are not chronological")
    safety = payload.get("safety", {})
    expected = {
        "credentials_allowed",
        "paper_allowed",
        "live_allowed",
        "orders_enabled",
        "promotion_authority",
    }
    if set(safety) != expected or any(safety[key] is not False for key in expected):
        raise PermissionError("rolling evaluator must remain fail-closed")
    return RollingProtocol(source, payload, hashlib.sha256(raw).hexdigest())


def genetic_simulation_kwargs(protocol: RollingProtocol) -> dict[str, object]:
    overlay = protocol.payload.get("policy_overlay")
    if overlay is None:
        return {}
    controls = protocol.payload.get("execution_controls", {})
    threshold_mode = str(overlay.get("baseline_threshold_mode", "return_bps"))
    threshold_value = float(
        overlay.get(
            "minimum_baseline_abs_value",
            overlay.get("minimum_baseline_move_bps", 0.0),
        )
    )
    return {
        "policy_mode": str(overlay["mode"]),
        "overlay_baseline_feature": str(overlay["baseline_feature"]),
        "overlay_minimum_move_bps": float(overlay.get("minimum_baseline_move_bps", 0.0)),
        "overlay_threshold_mode": threshold_mode,
        "overlay_threshold_value": threshold_value,
        "overlay_entry_logit_threshold": float(
            controls.get("entry_logit_threshold", 0.0)
        ),
        "overlay_exit_logit_threshold": float(
            controls.get("exit_logit_threshold", 0.0)
        ),
        "overlay_cooldown_bars": int(controls.get("cooldown_bars", 0)),
    }


@dataclass(frozen=True)
class RollingOrigin:
    index: int
    train_start: int
    train_end: int
    purge_cutoff: int
    deployment_start: int
    deployment_end: int

    def to_mapping(self) -> dict[str, int]:
        return {
            "origin_index": self.index,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "purge_cutoff": self.purge_cutoff,
            "deployment_start": self.deployment_start,
            "deployment_end": self.deployment_end,
        }


def build_rolling_origins(protocol: RollingProtocol) -> tuple[RollingOrigin, ...]:
    initial_train_start = _timestamp(protocol.payload["training_start"])
    training_window_days = protocol.payload.get("training_window_days")
    training_window_ms = (
        int(pd.Timedelta(days=int(training_window_days)).total_seconds() * 1000)
        if training_window_days is not None
        else None
    )
    origin = _timestamp(protocol.payload["validation_start"])
    validation_end = _timestamp(protocol.payload["validation_end"])
    rows: list[RollingOrigin] = []
    while origin + protocol.evaluation_interval_ms <= validation_end:
        train_start = (
            max(initial_train_start, origin - training_window_ms)
            if training_window_ms is not None
            else initial_train_start
        )
        rows.append(
            RollingOrigin(
                index=len(rows),
                train_start=train_start,
                train_end=origin,
                purge_cutoff=origin - protocol.purge_ms,
                deployment_start=origin,
                deployment_end=origin + protocol.evaluation_interval_ms,
            )
        )
        origin += protocol.cadence_ms
    if not rows:
        raise ValueError("rolling protocol produces no complete deployment windows")
    for first, second in zip(rows[:-1], rows[1:], strict=True):
        if first.deployment_end != second.deployment_start:
            raise ValueError("rolling v1 deployment windows must be contiguous and non-overlapping")
    return tuple(rows)


@dataclass(frozen=True)
class OriginTrainingContext:
    origin: RollingOrigin
    scaler: RobustScaler
    supervised_features: np.ndarray
    supervised_targets: np.ndarray
    feature_frame: pd.DataFrame
    protocol_sha256: str
    output_dir: Path | None
    label_frame: pd.DataFrame | None = None
    overlay_supervised_features: np.ndarray | None = None
    overlay_supervised_targets: np.ndarray | None = None
    incumbent_genomes: tuple[np.ndarray, ...] = ()
    incumbent_model_sha256: str | None = None
    incumbent_scaler: RobustScaler | None = None


@dataclass(frozen=True)
class TrainedOriginModel:
    genome: np.ndarray
    shape: V4NetworkShape
    trainer_id: str
    metadata: Mapping[str, Any]
    ensemble_genomes: tuple[np.ndarray, ...] = ()
    promotion_approved: bool = True
    trained_scaler: RobustScaler | None = None


OriginTrainer = Callable[[OriginTrainingContext], TrainedOriginModel]


@dataclass(frozen=True)
class RollingEvaluation:
    origins: pd.DataFrame
    models: pd.DataFrame
    decisions: pd.DataFrame
    trades: pd.DataFrame
    portfolio: pd.DataFrame
    benchmarks: pd.DataFrame
    benchmark_portfolio: pd.DataFrame


def summarize_outer_deployment_evidence(
    benchmarks: pd.DataFrame,
    *,
    minimum_completed_origins: int,
    minimum_positive_fraction: float,
) -> dict[str, Any]:
    """Summarize only completed deployment windows; training folds never enter here."""
    required = {
        "origin_index",
        "agent_id",
        "net_bps",
        "evidence_role",
    }
    if not required <= set(benchmarks.columns):
        raise ValueError("outer evidence requires deployment benchmark columns")
    outer = benchmarks.loc[benchmarks["evidence_role"].eq("outer_deployment")]
    pivot = outer.pivot(index="origin_index", columns="agent_id", values="net_bps")
    needed = {"GeneticPrimusChallenger", "GeneticPrimusActive"}
    if not needed <= set(pivot.columns):
        return {
            "schema_version": "exia.genetic_primus.outer_deployment_evidence/1",
            "evidence_role": "outer_deployment",
            "completed_outer_origins": 0,
            "minimum_completed_outer_origins": int(minimum_completed_origins),
            "positive_outer_slice_fraction": 0.0,
            "minimum_positive_outer_slice_fraction": float(minimum_positive_fraction),
            "evidence_complete": False,
            "positive_fraction_gate_passed": False,
            "no_eligible_challenger_evidence": True,
            "promotion_approved": False,
            "promotion_authority": False,
        }
    paired = pivot.dropna(subset=sorted(needed))
    delta = paired["GeneticPrimusChallenger"] - paired["GeneticPrimusActive"]
    completed = int(len(delta))
    positive_fraction = float((delta > 0.0).mean()) if completed else 0.0
    evidence_complete = completed >= int(minimum_completed_origins)
    return {
        "schema_version": "exia.genetic_primus.outer_deployment_evidence/1",
        "evidence_role": "outer_deployment",
        "completed_outer_origins": completed,
        "minimum_completed_outer_origins": int(minimum_completed_origins),
        "positive_outer_slice_fraction": positive_fraction,
        "minimum_positive_outer_slice_fraction": float(minimum_positive_fraction),
        "evidence_complete": evidence_complete,
        "positive_fraction_gate_passed": bool(
            evidence_complete and positive_fraction >= minimum_positive_fraction
        ),
        "promotion_approved": False,
        "promotion_authority": False,
    }


def _array_sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def build_functional_drift_anchor(
    frame: pd.DataFrame,
    *,
    start_timestamp: int,
    cutoff_timestamp: int,
    maximum_rows: int = 512,
) -> tuple[pd.DataFrame, str]:
    """Select immutable pre-deployment rows by row-key SHA rather than frame order."""
    if maximum_rows <= 0:
        raise ValueError("functional drift anchor size must be positive")
    mask = frame["timestamp"].between(start_timestamp, cutoff_timestamp, inclusive="both")
    local = frame.loc[mask].copy()
    if local.empty:
        raise ValueError("functional drift anchor contains no rows")
    local["_anchor_key"] = [
        f"{int(timestamp)}:{symbol}"
        for timestamp, symbol in zip(local["timestamp"], local["symbol"], strict=True)
    ]
    if local["_anchor_key"].duplicated().any():
        raise ValueError("functional drift anchor row keys must be unique")
    local["_anchor_digest"] = local["_anchor_key"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    selected = local.sort_values(
        ["_anchor_digest", "_anchor_key"], kind="stable"
    ).head(maximum_rows)
    row_keys = selected["_anchor_key"].tolist()
    anchor_sha256 = hashlib.sha256(
        json.dumps(row_keys, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return selected.drop(columns=["_anchor_key", "_anchor_digest"]).reset_index(drop=True), anchor_sha256


def scaled_functional_drift_anchor(
    anchor: pd.DataFrame,
    *,
    scaler: RobustScaler,
) -> np.ndarray:
    static = scaler.transform(anchor)
    return np.concatenate(
        [static, np.zeros((len(static), 5), dtype=np.float32)],
        axis=1,
    )


def model_sha256(
    model: TrainedOriginModel,
    scaler: RobustScaler,
    origin: RollingOrigin,
    protocol_sha256: str,
) -> str:
    identity = {
        "schema_version": "exia.genetic_primus.rolling_model/1",
        "trainer_id": model.trainer_id,
        "promotion_approved": bool(model.promotion_approved),
        "shape": {
            "inputs": model.shape.inputs,
            "hidden": list(model.shape.hidden),
            "outputs": model.shape.outputs,
            "maximum_active_features": model.shape.maximum_active_features,
        },
        "origin": origin.to_mapping(),
        "protocol_sha256": protocol_sha256,
        "genome_sha256": _array_sha(np.asarray(model.genome, dtype=np.float32)),
        "ensemble_genome_sha256": [
            _array_sha(np.asarray(genome, dtype=np.float32)) for genome in model.ensemble_genomes
        ],
        "scaler_median_sha256": _array_sha(np.asarray(scaler.median, dtype=np.float64)),
        "scaler_scale_sha256": _array_sha(np.asarray(scaler.scale, dtype=np.float64)),
        "trainer_metadata_sha256": canonical_sha256(dict(model.metadata)),
    }
    return canonical_sha256(identity)


def _policy_for_model(model: TrainedOriginModel, genome: np.ndarray):
    if model.ensemble_genomes:
        return EnsemblePolicyV4(model.ensemble_genomes, shape=model.shape)
    return GeneticPrimusPolicyV4(genome, shape=model.shape)


def _flat_bootstrap_model(shape: V4NetworkShape, challenger_sha256: str) -> TrainedOriginModel:
    genome = np.zeros(shape.genome_size, dtype=np.float32)
    if shape.outputs == 1:
        genome[-1] = -1.0
    else:
        genome[-shape.outputs + 1] = 1.0
    return TrainedOriginModel(
        genome=genome,
        shape=shape,
        trainer_id="deterministic_no_trade_bootstrap/1",
        metadata={
            "status": "PAIRWISE_BOOTSTRAP_REJECTED_USE_NOTRADE",
            "rejected_challenger_model_sha256": challenger_sha256,
        },
        promotion_approved=False,
    )


def _origin_training_data(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    origin: RollingOrigin,
) -> tuple[RobustScaler, np.ndarray, np.ndarray, int]:
    scaler_mask = features["timestamp"].ge(origin.train_start) & features["timestamp"].lt(origin.train_end)
    scaler_frame = features.loc[scaler_mask]
    if scaler_frame.empty:
        raise ValueError(f"origin {origin.index} has no scaler data")
    scaler = RobustScaler.fit(scaler_frame)
    supervised_mask = (
        features["timestamp"].ge(origin.train_start)
        & features["timestamp"].lt(origin.purge_cutoff)
        & features["feature_supported"]
        & labels["label_supported"]
    )
    if not supervised_mask.any():
        raise ValueError(f"origin {origin.index} has no purged supervised rows")
    static = scaler.transform(features.loc[supervised_mask])
    dynamic = np.zeros((len(static), 5), dtype=np.float32)
    x = np.concatenate([static, dynamic], axis=1)
    y = labels.loc[supervised_mask, "target_position"].to_numpy(np.int8)
    return scaler, x, y, int(scaler_mask.sum())


def _overlay_supervised_data(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    origin: RollingOrigin,
    scaler: RobustScaler,
    *,
    baseline_feature: str,
    threshold_mode: str,
    threshold_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (
        features["timestamp"].ge(origin.train_start)
        & features["timestamp"].lt(origin.purge_cutoff)
        & features["feature_supported"]
        & labels["label_supported"]
    )
    baseline = features.loc[mask, baseline_feature].to_numpy(np.float64)
    opportunity = overlay_opportunity_mask(
        baseline,
        threshold_mode=threshold_mode,
        threshold_value=threshold_value,
    )
    if not opportunity.any():
        raise ValueError(f"origin {origin.index} has no overlay opportunities")
    static = scaler.transform(features.loc[mask])
    dynamic = np.zeros((len(static), 5), dtype=np.float32)
    x = np.concatenate([static, dynamic], axis=1)[opportunity]
    target_position = labels.loc[mask, "target_position"].to_numpy(np.int8)[opportunity]
    baseline_target = np.sign(baseline[opportunity]).astype(np.int8)
    allow = ((target_position != 0) & (target_position == baseline_target)).astype(np.int8)
    return x, allow


def evaluate_rolling(
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    protocol: RollingProtocol,
    trainer: OriginTrainer,
    round_trip_cost_bps: float,
    stress_round_trip_cost_bps: float | None = None,
    output_dir: Path | None = None,
    max_origins: int | None = None,
    resume_partial: bool = False,
) -> RollingEvaluation:
    if not features[["timestamp", "symbol"]].equals(labels[["timestamp", "symbol"]]):
        raise ValueError("rolling feature/label rows are not aligned")
    if round_trip_cost_bps <= 0:
        raise ValueError("rolling evaluation requires positive costs")
    if stress_round_trip_cost_bps is not None and stress_round_trip_cost_bps <= round_trip_cost_bps:
        raise ValueError("stress costs must exceed base rolling costs")
    origins = build_rolling_origins(protocol)
    if max_origins is not None:
        if max_origins <= 0:
            raise ValueError("max_origins must be positive")
        origins = origins[:max_origins]
    origin_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    decision_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    portfolio_frames: list[pd.DataFrame] = []
    benchmark_rows: list[dict[str, Any]] = []
    benchmark_portfolio_frames: list[pd.DataFrame] = []
    simulation_kwargs = genetic_simulation_kwargs(protocol)
    if not origins:
        raise ValueError("rolling evaluation has no deployment origins")
    functional_anchor, functional_anchor_sha256 = build_functional_drift_anchor(
        features,
        start_timestamp=origins[0].train_start,
        cutoff_timestamp=origins[0].purge_cutoff,
    )
    previous_model: tuple[TrainedOriginModel, np.ndarray, RobustScaler, str] | None = None
    previous_challenger_function: (
        tuple[tuple[np.ndarray, ...], V4NetworkShape, RobustScaler] | None
    ) = None
    previous_challenger_deployment_actions: Mapping[str, int] | None = None
    for origin in origins:
        incumbent_before = previous_model
        scaler, x, y, scaler_rows = _origin_training_data(features, labels, origin)
        origin_dir = output_dir / f"origin_{origin.index:04d}" if output_dir is not None else None
        if origin_dir is not None:
            origin_dir.mkdir(parents=True, exist_ok=resume_partial)
        overlay_x: np.ndarray | None = None
        overlay_y: np.ndarray | None = None
        overlay = protocol.payload.get("policy_overlay")
        if overlay is not None and overlay.get("mode") == "momentum_abstention_binary":
            overlay_x, overlay_y = _overlay_supervised_data(
                features,
                labels,
                origin,
                scaler,
                baseline_feature=str(overlay["baseline_feature"]),
                threshold_mode=str(overlay.get("baseline_threshold_mode", "return_bps")),
                threshold_value=float(
                    overlay.get(
                        "minimum_baseline_abs_value",
                        overlay.get("minimum_baseline_move_bps", 0.0),
                    )
                ),
            )
        context = OriginTrainingContext(
            origin=origin,
            scaler=scaler,
            supervised_features=x,
            supervised_targets=y,
            feature_frame=features,
            protocol_sha256=protocol.sha256,
            output_dir=origin_dir,
            label_frame=labels,
            overlay_supervised_features=overlay_x,
            overlay_supervised_targets=overlay_y,
            incumbent_genomes=(
                incumbent_before[0].ensemble_genomes
                or (incumbent_before[1].copy(),)
                if incumbent_before is not None
                else ()
            ),
            incumbent_model_sha256=incumbent_before[3] if incumbent_before is not None else None,
            incumbent_scaler=incumbent_before[2] if incumbent_before is not None else None,
        )
        model = trainer(context)
        genome = np.asarray(model.genome, dtype=np.float32).reshape(-1)
        if len(genome) != model.shape.genome_size:
            raise ValueError(f"origin {origin.index} trainer returned an invalid genome")
        if any(
            len(np.asarray(member, dtype=np.float32).reshape(-1)) != model.shape.genome_size
            for member in model.ensemble_genomes
        ):
            raise ValueError(f"origin {origin.index} trainer returned an invalid ensemble")
        challenger_scaler = model.trained_scaler or scaler
        challenger_id = model_sha256(model, challenger_scaler, origin, protocol.sha256)
        promotion_approved = bool(
            model.promotion_approved
            or (incumbent_before is None and not simulation_kwargs)
        )
        if promotion_approved:
            active_model, active_genome, active_scaler, active_id = (
                model,
                genome.copy(),
                challenger_scaler,
                challenger_id,
            )
            previous_model = (active_model, active_genome.copy(), active_scaler, active_id)
        elif incumbent_before is not None:
            assert incumbent_before is not None
            active_model, active_genome, active_scaler, active_id = incumbent_before
        else:
            active_model = _flat_bootstrap_model(model.shape, challenger_id)
            active_genome = np.asarray(active_model.genome, dtype=np.float32).copy()
            active_scaler = scaler
            active_id = model_sha256(active_model, active_scaler, origin, protocol.sha256)
            previous_model = (active_model, active_genome.copy(), active_scaler, active_id)
        challenger_members = tuple(
            np.asarray(member, dtype=np.float32).copy()
            for member in (model.ensemble_genomes or (genome,))
        )
        challenger_anchor_features = scaled_functional_drift_anchor(
            functional_anchor,
            scaler=challenger_scaler,
        )
        if previous_challenger_function is None:
            retrain_model_drift = 0.0
            retrain_action_drift = 0.0
        else:
            previous_members, previous_shape, previous_scaler = previous_challenger_function
            if previous_shape != model.shape:
                raise ValueError("functional drift requires a stable model shape")
            functional_drift = functional_policy_drift_v4(
                challenger_members,
                previous_members,
                shape=model.shape,
                challenger_features=challenger_anchor_features,
                incumbent_features=scaled_functional_drift_anchor(
                    functional_anchor,
                    scaler=previous_scaler,
                ),
            )
            retrain_model_drift = float(functional_drift["functional_logit_drift"])
            retrain_action_drift = float(functional_drift["functional_action_drift"])
        previous_challenger_function = (
            challenger_members,
            model.shape,
            challenger_scaler,
        )
        if origin_dir is not None:
            np.savez_compressed(
                origin_dir / "champion.npz",
                genome=active_genome,
                ensemble_genomes=(
                    np.stack(active_model.ensemble_genomes)
                    if active_model.ensemble_genomes
                    else np.empty((0, len(active_genome)))
                ),
            )
            np.savez_compressed(
                origin_dir / "challenger.npz",
                genome=genome,
                ensemble_genomes=(
                    np.stack(model.ensemble_genomes)
                    if model.ensemble_genomes
                    else np.empty((0, len(genome)))
                ),
            )
            (origin_dir / "scaler.json").write_text(
                json.dumps(active_scaler.to_mapping(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (origin_dir / "challenger_scaler.json").write_text(
                json.dumps(challenger_scaler.to_mapping(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            origin_manifest = {
                "schema_version": "exia.genetic_primus.rolling_origin/1",
                "status": "OFFLINE_ROLLING_ORIGIN_COMPLETE",
                "model_sha256": active_id,
                "challenger_model_sha256": challenger_id,
                "promotion_approved": promotion_approved,
                "challenger_status": model.metadata.get(
                    "candidate_status", "LEGACY_PROMOTION_EVALUATION"
                ),
                "challenger_deployable": bool(
                    model.metadata.get("candidate_deployable", promotion_approved)
                ),
                "challenger_retrain_functional_action_drift": retrain_action_drift,
                "challenger_retrain_functional_logit_drift": retrain_model_drift,
                "functional_drift_anchor_sha256": functional_anchor_sha256,
                "functional_drift_anchor_rows": int(len(functional_anchor)),
                "genome_sha256": _array_sha(active_genome),
                "challenger_genome_sha256": _array_sha(genome),
                "ensemble_genome_sha256": [
                    _array_sha(np.asarray(member, dtype=np.float32))
                    for member in active_model.ensemble_genomes
                ],
                "challenger_ensemble_genome_sha256": [
                    _array_sha(np.asarray(member, dtype=np.float32))
                    for member in model.ensemble_genomes
                ],
                "scaler_sha256": canonical_sha256(active_scaler.to_mapping()),
                "challenger_scaler_sha256": canonical_sha256(
                    challenger_scaler.to_mapping()
                ),
                "rolling_protocol_sha256": protocol.sha256,
                "trainer_id": active_model.trainer_id,
                "trainer_metadata": dict(active_model.metadata),
                "challenger_trainer_id": model.trainer_id,
                "challenger_trainer_metadata": dict(model.metadata),
                **origin.to_mapping(),
                "credentials_allowed": False,
                "paper_allowed": False,
                "live_allowed": False,
                "orders_enabled": False,
                "promotion_authority": False,
            }
            (origin_dir / "model_manifest.json").write_text(
                json.dumps(origin_manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        simulation = simulate_policy(
            features,
            policy=_policy_for_model(active_model, active_genome),
            scaler=active_scaler,
            start_timestamp=origin.deployment_start,
            end_timestamp=origin.deployment_end,
            round_trip_cost_bps=round_trip_cost_bps,
            record_decisions=True,
            model_sha256=active_id,
            origin_timestamp=origin.deployment_start,
            **simulation_kwargs,
        )
        challenger_simulation = (
            simulation
            if promotion_approved
            else simulate_policy(
                features,
                policy=_policy_for_model(model, genome),
                scaler=challenger_scaler,
                start_timestamp=origin.deployment_start,
                end_timestamp=origin.deployment_end,
                round_trip_cost_bps=round_trip_cost_bps,
                **simulation_kwargs,
            )
        )
        retrain_deployment_action_mix_drift = (
            action_total_variation(
                challenger_simulation.action_counts,
                previous_challenger_deployment_actions,
            )
            if previous_challenger_deployment_actions is not None
            else 0.0
        )
        previous_challenger_deployment_actions = dict(challenger_simulation.action_counts)
        challenger_deployable = bool(
            model.metadata.get("candidate_deployable", promotion_approved)
        )
        challenger_evidence_role = (
            "outer_deployment"
            if challenger_deployable
            else "diagnostic_outer_deployment"
        )
        def record_benchmark(
            *,
            agent_id: str,
            benchmark_kind: str,
            policy_sha256: str,
            benchmark_simulation,
            benchmark_cost_bps: float = round_trip_cost_bps,
            evidence_role: str = "outer_deployment",
        ) -> None:
            benchmark_rows.append(
                {
                    "origin_index": origin.index,
                    "origin_timestamp": origin.deployment_start,
                    "deployment_end": origin.deployment_end,
                    "agent_id": agent_id,
                    "benchmark_kind": benchmark_kind,
                    "evidence_role": evidence_role,
                    "policy_sha256": policy_sha256,
                    "round_trip_cost_bps": float(benchmark_cost_bps),
                    **summarize_benchmark(benchmark_simulation),
                }
            )
            local = benchmark_simulation.portfolio.copy()
            local["origin_index"] = origin.index
            local["origin_timestamp"] = origin.deployment_start
            local["agent_id"] = agent_id
            local["benchmark_kind"] = benchmark_kind
            local["evidence_role"] = evidence_role
            local["policy_sha256"] = policy_sha256
            benchmark_portfolio_frames.append(local)

        record_benchmark(
            agent_id="GeneticPrimusChallenger",
            benchmark_kind="candidate",
            policy_sha256=challenger_id,
            benchmark_simulation=challenger_simulation,
            evidence_role=challenger_evidence_role,
        )
        record_benchmark(
            agent_id="GeneticPrimusActive",
            benchmark_kind="active_model",
            policy_sha256=active_id,
            benchmark_simulation=simulation,
        )
        if stress_round_trip_cost_bps is not None:
            challenger_stress = simulate_policy(
                features,
                policy=_policy_for_model(model, genome),
                scaler=challenger_scaler,
                start_timestamp=origin.deployment_start,
                end_timestamp=origin.deployment_end,
                round_trip_cost_bps=stress_round_trip_cost_bps,
                **simulation_kwargs,
            )
            active_stress = simulate_policy(
                features,
                policy=_policy_for_model(active_model, active_genome),
                scaler=active_scaler,
                start_timestamp=origin.deployment_start,
                end_timestamp=origin.deployment_end,
                round_trip_cost_bps=stress_round_trip_cost_bps,
                **simulation_kwargs,
            )
            record_benchmark(
                agent_id="GeneticPrimusChallengerStress",
                benchmark_kind="candidate_stress",
                policy_sha256=challenger_id,
                benchmark_simulation=challenger_stress,
                benchmark_cost_bps=stress_round_trip_cost_bps,
                evidence_role=challenger_evidence_role,
            )
            record_benchmark(
                agent_id="GeneticPrimusActiveStress",
                benchmark_kind="active_model_stress",
                policy_sha256=active_id,
                benchmark_simulation=active_stress,
                benchmark_cost_bps=stress_round_trip_cost_bps,
            )
        baseline_policies = [NoTradePolicy(), momentum_policy(), mean_reversion_policy()]
        if protocol.payload.get("policy_overlay", {}).get(
            "baseline_policy_id"
        ) == "RelativeMomentumRank_return_12":
            baseline_policies.append(relative_momentum_policy())
        for baseline_policy in baseline_policies:
            baseline_sha = fixed_baseline_sha256(baseline_policy.policy_id)
            baseline_simulation = simulate_policy(
                features,
                policy=baseline_policy,
                scaler=fixed_baseline_scaler(),
                start_timestamp=origin.deployment_start,
                end_timestamp=origin.deployment_end,
                round_trip_cost_bps=round_trip_cost_bps,
            )
            record_benchmark(
                agent_id=baseline_policy.policy_id,
                benchmark_kind="fixed_baseline",
                policy_sha256=baseline_sha,
                benchmark_simulation=baseline_simulation,
            )
        if incumbent_before is not None:
            incumbent_model, incumbent_genome, incumbent_scaler, incumbent_sha = incumbent_before
            incumbent_simulation = simulate_policy(
                features,
                policy=_policy_for_model(incumbent_model, incumbent_genome),
                scaler=incumbent_scaler,
                start_timestamp=origin.deployment_start,
                end_timestamp=origin.deployment_end,
                round_trip_cost_bps=round_trip_cost_bps,
                **simulation_kwargs,
            )
            record_benchmark(
                agent_id="GeneticPrimusIncumbent",
                benchmark_kind="previous_origin_model",
                policy_sha256=incumbent_sha,
                benchmark_simulation=incumbent_simulation,
            )
        decisions = simulation.decisions.copy()
        decisions["train_end"] = origin.train_end
        decisions["purge_cutoff"] = origin.purge_cutoff
        decisions["deployment_end"] = origin.deployment_end
        decision_frames.append(decisions)
        trades = simulation.ledger.copy()
        trades["model_sha256"] = active_id
        trades["origin_timestamp"] = origin.deployment_start
        trade_frames.append(trades)
        portfolio = simulation.portfolio.copy()
        portfolio["model_sha256"] = active_id
        portfolio["origin_timestamp"] = origin.deployment_start
        portfolio_frames.append(portfolio)
        model_rows.append(
            {
                "origin_index": origin.index,
                "model_sha256": active_id,
                "challenger_model_sha256": challenger_id,
                "promotion_approved": promotion_approved,
                "predeployment_training_gate_passed": bool(
                    model.metadata.get("promotion_metrics", {}).get(
                        "predeployment_training_gate_passed", False
                    )
                ),
                "challenger_retrain_action_drift": retrain_action_drift,
                "challenger_retrain_model_drift": retrain_model_drift,
                "challenger_retrain_functional_action_drift": retrain_action_drift,
                "challenger_retrain_functional_logit_drift": retrain_model_drift,
                "challenger_retrain_deployment_action_mix_drift": (
                    retrain_deployment_action_mix_drift
                ),
                "functional_drift_anchor_sha256": functional_anchor_sha256,
                "functional_drift_anchor_rows": int(len(functional_anchor)),
                "challenger_status": model.metadata.get(
                    "candidate_status", "LEGACY_PROMOTION_EVALUATION"
                ),
                "challenger_deployable": bool(
                    model.metadata.get("candidate_deployable", promotion_approved)
                ),
                "genome_sha256": _array_sha(active_genome),
                "challenger_genome_sha256": _array_sha(genome),
                "ensemble_size": int(len(active_model.ensemble_genomes) or 1),
                "challenger_ensemble_size": int(len(model.ensemble_genomes) or 1),
                "scaler_sha256": canonical_sha256(active_scaler.to_mapping()),
                "challenger_scaler_sha256": canonical_sha256(
                    challenger_scaler.to_mapping()
                ),
                "trainer_id": active_model.trainer_id,
                "trainer_metadata_sha256": canonical_sha256(dict(active_model.metadata)),
                "challenger_trainer_id": model.trainer_id,
                "challenger_trainer_metadata_sha256": canonical_sha256(dict(model.metadata)),
                "supervised_rows": int(len(y)),
                "overlay_supervised_rows": int(len(overlay_y)) if overlay_y is not None else 0,
                "scaler_rows": scaler_rows,
                **origin.to_mapping(),
            }
        )
        origin_rows.append(
            {
                **origin.to_mapping(),
                "model_sha256": active_id,
                "challenger_model_sha256": challenger_id,
                "promotion_approved": promotion_approved,
                "challenger_status": model.metadata.get(
                    "candidate_status", "LEGACY_PROMOTION_EVALUATION"
                ),
                "challenger_deployable": bool(
                    model.metadata.get("candidate_deployable", promotion_approved)
                ),
                "evidence_role": "outer_deployment",
                "decisions": int(len(decisions)),
                "trades": int(len(trades)),
                "turnover_fills": int(simulation.turnover_fills),
            }
        )
    decisions = pd.concat(decision_frames, ignore_index=True) if decision_frames else pd.DataFrame(columns=DECISION_COLUMNS)
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame(columns=LEDGER_COLUMNS)
    portfolio = pd.concat(portfolio_frames, ignore_index=True) if portfolio_frames else pd.DataFrame()
    benchmark_portfolio = (
        pd.concat(benchmark_portfolio_frames, ignore_index=True)
        if benchmark_portfolio_frames
        else pd.DataFrame()
    )
    benchmarks = pd.DataFrame(benchmark_rows)
    if not benchmarks.empty:
        candidate_rows = benchmarks.loc[
            benchmarks["benchmark_kind"].isin({"candidate", "candidate_stress"})
        ]
        candidate_bps = {
            (int(row.origin_index), float(row.round_trip_cost_bps)): float(row.net_bps)
            for row in candidate_rows.itertuples(index=False)
        }
        benchmarks["candidate_delta_bps"] = [
            candidate_bps[(int(row.origin_index), float(row.round_trip_cost_bps))]
            - float(row.net_bps)
            for row in benchmarks.itertuples(index=False)
        ]
        benchmarks["candidate_outperformed"] = benchmarks["candidate_delta_bps"] > 0.0
        benchmarks["return_rank"] = (
            benchmarks.groupby(
                ["origin_index", "round_trip_cost_bps"], sort=False
            )["net_bps"]
            .rank(method="min", ascending=False)
            .astype("int64")
        )
    if not decisions.empty and decisions["model_sha256"].isna().any():
        raise RuntimeError("rolling decision ledger contains an unpinned decision")
    return RollingEvaluation(
        origins=pd.DataFrame(origin_rows),
        models=pd.DataFrame(model_rows),
        decisions=decisions,
        trades=trades,
        portfolio=portfolio,
        benchmarks=benchmarks,
        benchmark_portfolio=benchmark_portfolio,
    )
