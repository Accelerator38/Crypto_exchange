"""Fail-closed progression from disclosed research to a future audit candidate."""

from __future__ import annotations

REQUIRED_CHECKS = frozenset({
    "normal_net_positive", "stress_net_positive",
    "normal_lcb_vs_NoTrade_positive", "stress_lcb_vs_NoTrade_positive",
    "normal_paired_lcb_vs_EMA_long_only_positive",
    "stress_paired_lcb_vs_EMA_long_only_positive",
    "drawdown_below_absolute_limit",
})
REQUIRED_SAFETY = frozenset({
    "credentials_allowed", "orders_enabled", "paper_allowed", "live_allowed",
    "promotion_authority", "runtime_authority", "network_allowed",
})


def stage_gates(episode: dict, linear: dict) -> dict:
    selected = linear.get("selected_variant")
    records = linear.get("records")
    matching = ([row for row in records if row.get("variant") == selected]
                if isinstance(records, list) and isinstance(selected, str) else [])
    validation = linear.get("validation_result")
    gate = validation.get("gate", {}) if isinstance(validation, dict) else {}
    checks = gate.get("checks")
    linear_ready = (
        linear.get("schema_version") == "exia.genetic_primus.rolling_linear_development_result/1"
        and linear.get("final_policy") == "linear_overlay_for_future_separate_audit_only"
        and len(matching) == 1
        and matching[0].get("train_score", {}).get("rank_score", 0) > 0
        and validation.get("variant") == selected
        and gate.get("passed") is True
        and isinstance(checks, dict)
        and REQUIRED_CHECKS.issubset(checks)
        and all(value is True for value in checks.values())
        and linear.get("automatic_promotion") is False
        and isinstance(linear.get("safety"), dict)
        and REQUIRED_SAFETY.issubset(linear["safety"])
        and all(value is False for value in linear["safety"].values())
    ) if isinstance(validation, dict) else False
    episode_ready = (
        episode.get("schema_version") == "exia.genetic_primus.episode_attribution_result/1"
        and episode.get("next_search_gate_passed") is True
        and isinstance(episode.get("paired_lcb"), dict)
        and all(episode["paired_lcb"].get(label, 0) > 0 for label in ("normal", "stress"))
    )
    return {
        "linear_inner_validation_passed": bool(linear_ready),
        "episode_attribution_passed": bool(episode_ready),
        "ga_search_allowed_under_current_plan": bool(linear_ready and episode_ready),
        "linear_candidate_can_be_considered_for_future_contract": bool(linear_ready),
        "prospective_contract_registered": False,
        "promotion_authority": False,
    }
