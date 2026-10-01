"""Pure, fail-closed admission of a development selection to future audit.

Admission is not promotion or trading authority. It only permits registering
the exact validated genome in a separately committed offline outer interval.
"""

from __future__ import annotations

import math
from typing import Any


PASSED_POLICY = "selected_sparse_overlay_for_future_separate_audit_only"
SELECTION_SCHEMA = "exia.genetic_primus.development_selection_result/1"
REQUIRED_VALIDATION_CHECKS = frozenset({
    "normal_net_positive", "stress_net_positive",
    "normal_lcb_vs_NoTrade_positive", "stress_lcb_vs_NoTrade_positive",
    "normal_paired_lcb_vs_EMA_long_only_positive",
    "stress_paired_lcb_vs_EMA_long_only_positive",
    "drawdown_below_absolute_limit",
})
REQUIRED_SAFETY_FLAGS = frozenset({
    "credentials_allowed", "live_allowed", "network_allowed", "orders_enabled",
    "paper_allowed", "promotion_authority", "runtime_authority",
})


def admission_reasons(report: dict[str, Any], genome: object) -> list[str]:
    """Return stable rejection codes; an empty list means audit-registration ready."""
    reasons: list[str] = []
    if report.get("schema_version") != SELECTION_SCHEMA:
        reasons.append("UNSUPPORTED_SELECTION_SCHEMA")
    if report.get("evidence_class") != "revealed_historical_development_only":
        reasons.append("INVALID_DEVELOPMENT_EVIDENCE_CLASS")
    if report.get("final_policy") != PASSED_POLICY:
        reasons.append("NO_VALIDATED_CHALLENGER")
    selected = report.get("selected_genome")
    if not isinstance(selected, list) or selected != genome:
        reasons.append("GENOME_NOT_SELECTED")
    budget = report.get("candidate_budget")
    evaluations = report.get("search_evaluations")
    records = report.get("candidate_records")
    if (type(budget) is not int or type(evaluations) is not int
            or not 1 <= evaluations <= budget
            or not isinstance(records, list) or len(records) != evaluations):
        reasons.append("INVALID_SEARCH_BUDGET")
    elif isinstance(selected, list):
        matching = [record for record in records
                    if isinstance(record, dict) and record.get("genome") == selected]
        if (len(matching) != 1
                or not isinstance(matching[0].get("training"), dict)
                or type(matching[0]["training"].get("rank_score")) not in (int, float)
                or not math.isfinite(matching[0]["training"]["rank_score"])
                or matching[0]["training"]["rank_score"] <= 0):
            reasons.append("TRAIN_SELECTION_NOT_POSITIVE")
    validation = report.get("validation_result")
    if (not isinstance(validation, dict)
            or validation.get("genome") != selected
            or not isinstance(validation.get("gate"), dict)):
        reasons.append("VALIDATION_MISSING_OR_MISMATCHED")
    else:
        gate = validation["gate"]
        checks = gate.get("checks")
        if (gate.get("passed") is not True
                or not isinstance(checks, dict)
                or not REQUIRED_VALIDATION_CHECKS.issubset(checks)
                or not all(value is True for value in checks.values())):
            reasons.append("VALIDATION_GATE_FAILED")
    if report.get("automatic_promotion") is not False:
        reasons.append("AUTOMATIC_PROMOTION_NOT_DISABLED")
    safety = report.get("safety")
    if (not isinstance(safety, dict)
            or not REQUIRED_SAFETY_FLAGS.issubset(safety)
            or any(value is not False for value in safety.values())):
        reasons.append("OFFLINE_SAFETY_NOT_PROVEN")
    return reasons
