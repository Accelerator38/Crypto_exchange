"""Promotion acceptance checks for genetic-agent evaluation reports."""

from __future__ import annotations

from typing import Any


def _mode_by_name(genome_report: dict[str, Any], mode_name: str) -> dict[str, Any]:
    for mode in genome_report.get("modes", []):
        if mode.get("mode") == mode_name:
            return mode
    raise ValueError(f"mode not found: {mode_name}")


def evaluate_promotion_gate(
    report: dict[str, Any],
    *,
    baseline_index: int = 0,
    candidate_index: int = -1,
    mode_name: str = "fee_fixed_nextbar",
    min_robust_utility_delta: float = 0.0,
    max_turnover_rate: float | None = None,
    max_saturation_rate: float | None = None,
    max_invalid_open_logit_pressure: float | None = None,
    require_candidate_robust_pass: bool = True,
) -> dict[str, Any]:
    genomes = report.get("genomes", [])
    if not genomes:
        raise ValueError("report has no genomes")

    baseline = genomes[baseline_index]
    candidate = genomes[candidate_index]
    baseline_mode = _mode_by_name(baseline, mode_name)
    candidate_mode = _mode_by_name(candidate, mode_name)
    baseline_robust = baseline_mode.get("robust_score", {})
    candidate_robust = candidate_mode.get("robust_score", {})
    baseline_contract = baseline_mode.get("contract_metrics", {})
    candidate_contract = candidate_mode.get("contract_metrics", {})

    baseline_utility = float(baseline_robust.get("robust_utility", 0.0))
    candidate_utility = float(candidate_robust.get("robust_utility", 0.0))
    utility_delta = candidate_utility - baseline_utility
    baseline_turnover = float(baseline_contract.get("mean_turnover_rate", 0.0))
    candidate_turnover = float(candidate_contract.get("mean_turnover_rate", 0.0))
    baseline_saturation = float(baseline_contract.get("mean_saturation_rate", 0.0))
    candidate_saturation = float(candidate_contract.get("mean_saturation_rate", 0.0))
    baseline_invalid_open_pressure = float(
        baseline_contract.get("mean_invalid_open_logit_pressure", 0.0)
    )
    candidate_invalid_open_pressure = float(
        candidate_contract.get("mean_invalid_open_logit_pressure", 0.0)
    )
    baseline_max_invalid_open_pressure = float(
        baseline_contract.get(
            "max_invalid_open_logit_pressure",
            baseline_invalid_open_pressure,
        )
    )
    candidate_max_invalid_open_pressure = float(
        candidate_contract.get(
            "max_invalid_open_logit_pressure",
            candidate_invalid_open_pressure,
        )
    )

    failed_checks: list[str] = []
    if require_candidate_robust_pass and not candidate_robust.get("passes_default_gates", False):
        failed_checks.append("candidate_robust_gates")
    if utility_delta < min_robust_utility_delta:
        failed_checks.append("robust_utility_delta")
    if max_turnover_rate is not None and candidate_turnover > float(max_turnover_rate):
        failed_checks.append("max_turnover_rate")
    if max_saturation_rate is not None and candidate_saturation > float(max_saturation_rate):
        failed_checks.append("max_position_saturation_rate")
    if (
        max_invalid_open_logit_pressure is not None
        and candidate_max_invalid_open_pressure > float(max_invalid_open_logit_pressure)
    ):
        failed_checks.append("max_invalid_open_logit_pressure")

    return {
        "accepted": not failed_checks,
        "failed_checks": failed_checks,
        "mode": mode_name,
        "baseline_path": baseline.get("path"),
        "candidate_path": candidate.get("path"),
        "baseline_fitness": float(baseline_mode.get("fitness", 0.0)),
        "candidate_fitness": float(candidate_mode.get("fitness", 0.0)),
        "baseline_robust_utility": baseline_utility,
        "candidate_robust_utility": candidate_utility,
        "robust_utility_delta": utility_delta,
        "baseline_mean_turnover_rate": baseline_turnover,
        "candidate_mean_turnover_rate": candidate_turnover,
        "turnover_rate_delta": candidate_turnover - baseline_turnover,
        "baseline_mean_saturation_rate": baseline_saturation,
        "candidate_mean_saturation_rate": candidate_saturation,
        "saturation_rate_delta": candidate_saturation - baseline_saturation,
        "baseline_mean_invalid_open_logit_pressure": baseline_invalid_open_pressure,
        "candidate_mean_invalid_open_logit_pressure": candidate_invalid_open_pressure,
        "baseline_max_invalid_open_logit_pressure": baseline_max_invalid_open_pressure,
        "candidate_max_invalid_open_logit_pressure": candidate_max_invalid_open_pressure,
        "invalid_open_logit_pressure_delta": (
            candidate_invalid_open_pressure - baseline_invalid_open_pressure
        ),
        "candidate_failed_gates": candidate_robust.get("failed_gates", []),
    }
