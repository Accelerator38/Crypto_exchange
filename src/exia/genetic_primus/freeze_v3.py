from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Mapping

from .fitness import FitnessResult


def v3_freeze_failures(score: FitnessResult) -> tuple[str, ...]:
    failures = list(score.failed_gates)
    if not bool(score.metrics["temporal_gate_passed"]):
        failures.append("nonpositive_temporal_gate")
    return tuple(failures)


def v3_freeze_status(failures: tuple[str, ...]) -> str:
    return (
        "DEVELOPMENT_V3_PASSED_STRICT_GATES"
        if not failures
        else "DEVELOPMENT_V3_REJECTED_STRICT_GATES"
    )


def build_v3_freeze_manifest(
    *,
    experiment_manifest: Mapping[str, Any],
    experiment_manifest_sha256: str,
    champion_file_sha256: str,
    champion_genome_sha256: str,
    scaler_sha256: str,
    selected_generation: int,
    configured_generations: int,
    score: FitnessResult,
) -> dict[str, Any]:
    failures = v3_freeze_failures(score)
    status = v3_freeze_status(failures)
    periods = score.periods.loc[score.periods["supported"]]
    return {
        "schema_version": "exia.genetic_primus_freeze.v3",
        "agent_id": "Genetic_Primus",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": status,
        "selection_contract": experiment_manifest["selection_contract"],
        "experiment_manifest_sha256": experiment_manifest_sha256,
        "config_sha256": experiment_manifest["config_sha256"],
        "code_sha256": experiment_manifest["code_sha256"],
        "dataset_sha256": experiment_manifest["dataset_sha256"],
        "champion_file_sha256": champion_file_sha256,
        "champion_genome_sha256": champion_genome_sha256,
        "scaler_sha256": scaler_sha256,
        "selected_generation": int(selected_generation),
        "configured_generations": int(configured_generations),
        "split": "development",
        "cost_profile": "stress_round_trip",
        "metrics": dict(score.metrics),
        "temporal_evidence": {
            "supported_periods": int(len(periods)),
            "positive_period_scores": int((periods["period_score"] > 0.0).sum()),
            "nonpositive_period_scores": int((periods["period_score"] <= 0.0).sum()),
            "positive_period_lcb": int((periods["lcb_bps"] > 0.0).sum()),
            "nonpositive_period_lcb": int((periods["lcb_bps"] <= 0.0).sum()),
            "median_period_score": float(score.fitness),
        },
        "failed_gates": list(failures),
        "validation_allowed": status == "DEVELOPMENT_V3_PASSED_STRICT_GATES",
        "validation_opened": False,
        "oos_opened": False,
        "sanity_opened": False,
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }
