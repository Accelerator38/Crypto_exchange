"""GeneticCore v2 experiment helpers.

The helpers in this module keep the deployment shape as a singleton: they may
score a regime-specialist map during research, but the returned summary is a
single `GeneticsCore` candidate with probation-only live policy.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from panteon_v2.analysis.genetics_validation import (
    FITNESS_V4_REGIMES,
    fitness_v4_robust_score,
)


def _float_values(rows: Sequence[Mapping[str, Any]], name: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row.get(name, 0.0) or 0.0))
        except (TypeError, ValueError):
            values.append(0.0)
    return values


def _mode_payload(genome_payload: Mapping[str, Any], mode: str) -> dict[str, Any]:
    modes = genome_payload.get("modes", [])
    if not isinstance(modes, list):
        return {}
    for payload in modes:
        if isinstance(payload, dict) and str(payload.get("mode", "")) == mode:
            return dict(payload)
    return dict(modes[0]) if modes and isinstance(modes[0], dict) else {}


def _report_modes_by_path(report: Mapping[str, Any], *, mode: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    genomes = report.get("genomes", [])
    if not isinstance(genomes, list):
        return out
    for genome in genomes:
        if not isinstance(genome, dict):
            continue
        path = str(genome.get("path", "")).strip()
        if not path:
            continue
        out[path] = _mode_payload(genome, mode)
    return out


def _period_rows(mode_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    contract = mode_payload.get("contract_metrics", {})
    if not isinstance(contract, dict):
        return []
    periods = contract.get("periods", [])
    if not isinstance(periods, list):
        return []
    return [dict(row) for row in periods if isinstance(row, dict)]


def _period_returns(mode_payload: Mapping[str, Any]) -> list[float]:
    returns = mode_payload.get("period_rets", [])
    if not isinstance(returns, list):
        return []
    out: list[float] = []
    for value in returns:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def _compound_return_pct(returns: Sequence[float]) -> float:
    compound = 1.0
    for ret in returns:
        compound *= 1.0 + float(ret) / 100.0
    return float((compound - 1.0) * 100.0)


def candidate_metrics_from_mode(mode_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Compute v2 metrics from one evaluated genome mode payload."""

    period_rets = _period_returns(mode_payload)
    rows = _period_rows(mode_payload)
    regimes = [str(row.get("regime", "unknown")) for row in rows]
    fitness_v4 = fitness_v4_robust_score(
        period_rets,
        period_lcbs_pct=_float_values(rows, "min_per_symbol_lcb"),
        period_max_drawdowns_pct=_float_values(rows, "max_drawdown_pct"),
        period_turnover_rates=_float_values(rows, "turnover_rate"),
        period_effective_turnover_rates=_float_values(rows, "effective_turnover_rate"),
        period_saturation_rates=_float_values(rows, "saturation_rate"),
        period_invalid_open_pressures=_float_values(rows, "invalid_open_logit_pressure"),
        period_costs_pct=_float_values(rows, "cost_pct"),
        period_slippage_pct=_float_values(rows, "slippage_pct"),
        period_min_notional_pressures=_float_values(rows, "min_notional_pressure"),
        period_precision_costs_pct=_float_values(rows, "precision_cost_pct"),
        period_bad_signal_key_rates=_float_values(rows, "bad_signal_key_rate"),
        period_long_slot_rates=_float_values(rows, "mean_long_slot_rate"),
        period_short_slot_rates=_float_values(rows, "mean_short_slot_rate"),
        period_net_direction_biases=_float_values(rows, "mean_net_direction_bias"),
        period_regimes=regimes,
    )
    finite_rets = [float(ret) for ret in period_rets if math.isfinite(float(ret))]
    if finite_rets:
        mean_ret = float(sum(finite_rets) / len(finite_rets))
        min_ret = float(min(finite_rets))
        positive_period_pct = float(
            sum(1 for ret in finite_rets if ret > 0.0) / len(finite_rets) * 100.0
        )
    else:
        mean_ret = 0.0
        min_ret = 0.0
        positive_period_pct = 0.0
    return {
        "n_periods": len(finite_rets),
        "mean_ret": mean_ret,
        "compound_return_pct": _compound_return_pct(finite_rets),
        "min_ret": min_ret,
        "positive_period_pct": positive_period_pct,
        "period_rets": finite_rets,
        "fitness_v4": fitness_v4,
        "fitness_v4_robust": float(fitness_v4.get("fitness_v4_robust", float("-inf"))),
        "fitness_v4_passes_default_gates": bool(
            fitness_v4.get("passes_default_gates", False)
        ),
        "fitness_v4_failed_gates": list(fitness_v4.get("failed_gates", [])),
    }


def _mean_for_regime(mode_payload: Mapping[str, Any], regime: str) -> float | None:
    rows = _period_rows(mode_payload)
    returns = _period_returns(mode_payload)
    selected: list[float] = []
    for idx, row in enumerate(rows):
        if idx >= len(returns):
            continue
        if str(row.get("regime", "unknown")) == regime:
            selected.append(float(returns[idx]))
    if not selected:
        return None
    return float(sum(selected) / len(selected))


def _select_regime_map(
    report: Mapping[str, Any],
    *,
    mode: str,
    baseline_path: str,
    regimes: Sequence[str],
) -> dict[str, str]:
    by_path = _report_modes_by_path(report, mode=mode)
    if baseline_path not in by_path:
        raise ValueError(f"baseline path is missing from validation report: {baseline_path}")

    selected: dict[str, str] = {}
    for regime in regimes:
        best_path = baseline_path
        best_score = _mean_for_regime(by_path[baseline_path], regime)
        if best_score is None:
            best_score = float("-inf")
        for path, payload in by_path.items():
            score = _mean_for_regime(payload, regime)
            if score is None:
                continue
            if score > best_score:
                best_score = score
                best_path = path
        selected[str(regime)] = best_path
    return selected


def _synthetic_mode_for_regime_map(
    report: Mapping[str, Any],
    *,
    mode: str,
    regime_map: Mapping[str, str],
    fallback_path: str,
) -> dict[str, Any]:
    by_path = _report_modes_by_path(report, mode=mode)
    if fallback_path not in by_path:
        raise ValueError(f"fallback path is missing from report: {fallback_path}")
    baseline_rows = _period_rows(by_path[fallback_path])
    selected_rets: list[float] = []
    selected_rows: list[dict[str, Any]] = []
    for idx, baseline_row in enumerate(baseline_rows):
        regime = str(baseline_row.get("regime", "unknown"))
        path = str(regime_map.get(regime, fallback_path))
        payload = by_path.get(path, by_path[fallback_path])
        returns = _period_returns(payload)
        rows = _period_rows(payload)
        if idx >= len(returns):
            raise ValueError(f"period index {idx} missing for genome {path}")
        selected_rets.append(float(returns[idx]))
        row = dict(rows[idx]) if idx < len(rows) else dict(baseline_row)
        row["path"] = path
        selected_rows.append(row)
    return {
        "mode": mode,
        "period_rets": selected_rets,
        "contract_metrics": {"periods": selected_rows},
    }


def score_specialist_router(
    *,
    validation_report: Mapping[str, Any],
    oos_reports: Sequence[Mapping[str, Any]],
    baseline_path: str | None = None,
    mode: str = "fee_fixed_nextbar",
    regimes: Sequence[str] = FITNESS_V4_REGIMES,
    min_unique_selected_genomes: int = 2,
) -> dict[str, Any]:
    """Select validation specialists and evaluate the map as one singleton policy."""

    validation_by_path = _report_modes_by_path(validation_report, mode=mode)
    if not validation_by_path:
        raise ValueError("validation report does not contain genome modes")
    resolved_baseline = str(baseline_path or next(iter(validation_by_path)))
    selected_map = _select_regime_map(
        validation_report,
        mode=mode,
        baseline_path=resolved_baseline,
        regimes=regimes,
    )
    validation_mode = _synthetic_mode_for_regime_map(
        validation_report,
        mode=mode,
        regime_map=selected_map,
        fallback_path=resolved_baseline,
    )
    baseline_validation = candidate_metrics_from_mode(validation_by_path[resolved_baseline])
    selected_validation = candidate_metrics_from_mode(validation_mode)
    unique_selected_genomes = len({str(path) for path in selected_map.values()})

    failures: list[str] = []
    if selected_map == {str(regime): resolved_baseline for regime in regimes}:
        failures.append("baseline_selected")
    if unique_selected_genomes < int(min_unique_selected_genomes):
        failures.append("router_collapse")
    if selected_validation["mean_ret"] <= baseline_validation["mean_ret"]:
        failures.append("validation_mean_ret_not_improved")
    if not selected_validation["fitness_v4_passes_default_gates"]:
        failures.append("validation_fitness_v4_gates")

    oos_payloads: list[dict[str, Any]] = []
    for idx, report in enumerate(oos_reports):
        by_path = _report_modes_by_path(report, mode=mode)
        if resolved_baseline not in by_path:
            failures.append(f"oos_{idx}_baseline_missing")
            continue
        selected_mode = _synthetic_mode_for_regime_map(
            report,
            mode=mode,
            regime_map=selected_map,
            fallback_path=resolved_baseline,
        )
        selected = candidate_metrics_from_mode(selected_mode)
        baseline = candidate_metrics_from_mode(by_path[resolved_baseline])
        split_failures: list[str] = []
        if selected["mean_ret"] <= 0.0:
            split_failures.append(f"oos_{idx}_mean_ret_not_positive")
        if selected["mean_ret"] < baseline["mean_ret"]:
            split_failures.append(f"oos_{idx}_mean_ret_degraded")
        if not selected["fitness_v4_passes_default_gates"]:
            split_failures.append(f"oos_{idx}_fitness_v4_gates")
        failures.extend(split_failures)
        oos_payloads.append({
            "index": idx,
            "baseline": baseline,
            "selected": selected,
            "failures": split_failures,
        })

    failures = list(dict.fromkeys(failures))
    return {
        "schema_version": 1,
        "mode": mode,
        "deployment_shape": "singleton",
        "live_policy": "probation_only",
        "uses_live_ensemble": False,
        "baseline_path": resolved_baseline,
        "selected_regime_map": selected_map,
        "unique_selected_genomes": unique_selected_genomes,
        "min_unique_selected_genomes": int(min_unique_selected_genomes),
        "validation": {
            "baseline": baseline_validation,
            "selected": selected_validation,
        },
        "oos": oos_payloads,
        "promotion_failures": failures,
        "promotion_eligible": not failures,
    }


def summarize_singleton_candidate(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact live-safe summary for a specialist-router experiment."""

    return {
        "deployment_shape": "singleton",
        "live_policy": "probation_only",
        "uses_live_ensemble": False,
        "promotion_eligible": bool(result.get("promotion_eligible", False)),
        "baseline_path": result.get("baseline_path"),
        "selected_regime_map": dict(result.get("selected_regime_map") or {}),
        "unique_selected_genomes": result.get("unique_selected_genomes"),
        "min_unique_selected_genomes": result.get("min_unique_selected_genomes"),
        "promotion_failures": list(result.get("promotion_failures") or []),
    }
