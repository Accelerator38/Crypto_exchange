from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from panteon_v2.analysis.genetics_validation import (
    fitness_v3_robust_score,
    fitness_v4_robust_score,
)

ROUTER_REGIMES = ("crash", "bearish", "neutral", "bullish")
SPECIALIST_LABEL_BY_REGIME = {
    "crash": "GeneticsCrash",
    "bearish": "GeneticsBearish",
    "neutral": "GeneticsNeutral",
    "bullish": "GeneticsBullish",
}


def _normalize_exchange(value: Any) -> str:
    return str(value or "").strip().upper()


def _report_exchange(report: Dict[str, Any]) -> str:
    for key in ("exchange", "exchange_id", "domain_exchange"):
        value = _normalize_exchange(report.get(key))
        if value:
            return value
    metadata = report.get("metadata")
    if isinstance(metadata, dict):
        for key in ("exchange", "exchange_id", "domain_exchange"):
            value = _normalize_exchange(metadata.get(key))
            if value:
                return value
    return ""


def _strict_exchange_failures(
    reports: Sequence[tuple[str, Dict[str, Any]]],
    *,
    expected_exchange: str,
) -> List[str]:
    expected = _normalize_exchange(expected_exchange)
    failures: List[str] = []
    if not expected:
        return ["exchange_domain_missing"]
    for role, report in reports:
        actual = _report_exchange(report)
        if not actual:
            failures.append(f"{role}_exchange_missing")
        elif actual != expected:
            failures.append(f"{role}_exchange_mismatch:{actual}!={expected}")
    return failures


def apply_strict_promotion_contracts(
    selection: Dict[str, Any],
    *,
    reports: Sequence[tuple[str, Dict[str, Any]]] = (),
    expected_exchange: str = "",
    baseline_kind: str = "",
    require_flash_baseline: bool = True,
    require_oos_gate: bool = True,
    min_per_symbol_lcb: float | None = 0.0,
    max_oos_drawdown_pct: float | None = 12.0,
) -> Dict[str, Any]:
    """Mark selection ineligible unless it came from the full exchange/OOS chain."""

    failures = list(dict.fromkeys(str(item) for item in selection.get("promotion_failures", [])))
    failures.extend(
        item
        for item in _strict_exchange_failures(
            reports,
            expected_exchange=expected_exchange,
        )
        if item not in failures
    )
    if require_flash_baseline and str(baseline_kind or "").strip().lower() != "flash":
        failures.append("requires_flash_baseline")
    if require_oos_gate and "final_holdout_gate" not in selection:
        failures.append("requires_oos_holdout_gate")
    final_gate = selection.get("final_holdout_gate")
    if require_oos_gate and isinstance(final_gate, dict):
        if not bool(final_gate.get("promotion_eligible", False)):
            failures.append("oos_holdout_gate")
        for item in final_gate.get("holdouts", []):
            if not isinstance(item, dict):
                continue
            idx = item.get("index", 0)
            try:
                mean_delta = float(item.get("mean_ret_delta", 0.0) or 0.0)
            except (TypeError, ValueError):
                mean_delta = 0.0
            if mean_delta <= 0.0:
                failures.append(f"oos_{idx}_mean_ret_not_positive")
            candidate = item.get("candidate")
            if not isinstance(candidate, dict):
                continue
            if min_per_symbol_lcb is not None:
                lcb = candidate.get("min_per_symbol_lcb")
                if lcb is None:
                    failures.append(f"oos_{idx}_per_symbol_lcb_missing")
                else:
                    try:
                        lcb_value = float(lcb)
                    except (TypeError, ValueError):
                        lcb_value = float("-inf")
                    if lcb_value <= float(min_per_symbol_lcb):
                        failures.append(f"oos_{idx}_per_symbol_lcb")
            if max_oos_drawdown_pct is not None:
                dd = candidate.get("max_drawdown_pct")
                if dd is None:
                    failures.append(f"oos_{idx}_max_drawdown_missing")
                else:
                    try:
                        dd_value = float(dd)
                    except (TypeError, ValueError):
                        dd_value = float("inf")
                    if dd_value > float(max_oos_drawdown_pct):
                        failures.append(f"oos_{idx}_max_drawdown")

    failures = list(dict.fromkeys(failures))
    selection["exchange"] = _normalize_exchange(expected_exchange)
    selection["baseline_kind"] = str(baseline_kind or "").strip().lower()
    selection["strict_promotion_contracts"] = {
        "require_flash_baseline": bool(require_flash_baseline),
        "require_oos_gate": bool(require_oos_gate),
        "min_per_symbol_lcb": min_per_symbol_lcb,
        "max_oos_drawdown_pct": max_oos_drawdown_pct,
        "report_exchanges": {
            role: _report_exchange(report)
            for role, report in reports
        },
    }
    selection["promotion_failures"] = failures
    selection["promotion_eligible"] = (
        bool(selection.get("promotion_eligible", False))
        and not failures
    )
    return selection


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_existing_under_neiro_genetics(
    value: str | Path,
    *,
    results_root: Path,
    purpose: str,
) -> str:
    root = results_root.resolve()
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"{purpose} must stay under Results/neiro_genetics: {resolved}"
        ) from exc
    if not resolved.exists():
        raise ValueError(f"{purpose} does not exist: {resolved}")
    return str(resolved)


def _single_baseline_path(baseline_map: Dict[str, str]) -> str:
    for regime in ("neutral", "bearish", "bullish", "crash"):
        path = baseline_map.get(regime)
        if path:
            return str(path)
    for path in baseline_map.values():
        if path:
            return str(path)
    raise ValueError("baseline_regime_map must contain at least one genome path")


def build_specialists_manifest(
    selection: Dict[str, Any],
    *,
    results_root: Path,
    risk_off_crash_genome: str | Path,
    selection_source: str | Path | None = None,
    created_at: str | None = None,
    baseline_control: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    selected_raw = {
        str(regime): str(path)
        for regime, path in dict(selection.get("selected_regime_map") or {}).items()
        if path
    }
    baseline_raw = {
        str(regime): str(path)
        for regime, path in dict(selection.get("baseline_regime_map") or {}).items()
        if path
    }
    if not selected_raw:
        raise ValueError("selection must contain selected_regime_map")
    if not baseline_raw:
        raise ValueError("selection must contain baseline_regime_map")

    baseline_path = _single_baseline_path(baseline_raw)
    safe_baseline_path = _resolve_existing_under_neiro_genetics(
        baseline_path,
        results_root=results_root,
        purpose="baseline genome path",
    )
    safe_risk_off_path = _resolve_existing_under_neiro_genetics(
        risk_off_crash_genome,
        results_root=results_root,
        purpose="risk-off crash genome path",
    )

    failures = list(dict.fromkeys(str(item) for item in selection.get("promotion_failures", [])))
    selected_regime_map: Dict[str, str] = {}
    baseline_regime_map: Dict[str, str] = {}
    for regime in ROUTER_REGIMES:
        raw_baseline = baseline_raw.get(regime, baseline_path)
        baseline_regime_map[regime] = _resolve_existing_under_neiro_genetics(
            raw_baseline,
            results_root=results_root,
            purpose=f"{regime} baseline genome path",
        )

    crash_candidate = selected_raw.get("crash")
    crash_oos_eligible = bool(selection.get("crash_specialist_eligible", False))
    if crash_candidate and crash_oos_eligible:
        selected_regime_map["crash"] = _resolve_existing_under_neiro_genetics(
            crash_candidate,
            results_root=results_root,
            purpose="crash selected genome path",
        )
    else:
        selected_regime_map["crash"] = safe_risk_off_path
        if crash_candidate:
            failures.append("crash_specialist_not_oos_eligible")
        else:
            failures.append("crash_risk_off_fallback")

    for regime in ("bearish", "neutral", "bullish"):
        raw_selected = selected_raw.get(regime, baseline_path)
        selected_regime_map[regime] = _resolve_existing_under_neiro_genetics(
            raw_selected,
            results_root=results_root,
            purpose=f"{regime} selected genome path",
        )

    failures.extend(
        [
            "requires_panteon_flash_shadow_matrix",
            "requires_panteon_flash_executable_simulation",
            "requires_cost_aware_oos_strict_win",
        ]
    )
    failures = list(dict.fromkeys(failures))

    specialist_genome_map = {
        "GeneticsBest": safe_baseline_path,
        "GeneticsCrash": selected_regime_map["crash"],
        "GeneticsBullish": selected_regime_map["bullish"],
        "GeneticsBearish": selected_regime_map["bearish"],
        "GeneticsNeutral": selected_regime_map["neutral"],
    }
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "created_at": created_at or date.today().isoformat(),
        "baseline_control": baseline_control or {},
        "selection_source": None,
        "selected_is_baseline": bool(selection.get("selected_is_baseline", False)),
        "selection_promotion_eligible": bool(selection.get("promotion_eligible", False)),
        "selection_paper_trading_eligible": bool(selection.get("paper_trading_eligible", False)),
        "promotion_eligible": False,
        "paper_trading_eligible": False,
        "live_trading_eligible": False,
        "promotion_failures": failures,
        "specialist_genome_map": specialist_genome_map,
        "selected_regime_map": selected_regime_map,
        "baseline_regime_map": baseline_regime_map,
        "notes": {
            "crash": (
                "risk-off hold genome until a crash-OOS specialist strictly beats baseline"
                if selected_regime_map["crash"] == safe_risk_off_path
                else "explicit crash specialist gated by crash_specialist_eligible"
            ),
            "bearish": "fallback to overall baseline unless selection provided a gated bearish specialist",
            "neutral": "selected by validation/OOS router selection; not live-promoted by manifest alone",
            "bullish": "fallback to overall baseline unless selection provided a gated bullish specialist",
        },
    }
    if selection_source is not None:
        manifest["selection_source"] = _resolve_existing_under_neiro_genetics(
            selection_source,
            results_root=results_root,
            purpose="selection source path",
        )
    return manifest


def _mode_payload(genome_payload: Dict[str, Any], mode: str) -> Dict[str, Any]:
    for item in genome_payload.get("modes", []):
        if item.get("mode") == mode:
            return item
    raise KeyError(f"mode {mode!r} not found for {genome_payload.get('path')}")


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _contract_min_per_symbol_lcb(contract: Dict[str, Any]) -> float | None:
    for key in (
        "min_per_symbol_lcb",
        "min_symbol_pnl_lcb",
        "min_symbol_pnl_lcb_pct",
        "min_per_symbol_pnl_lcb_pct",
    ):
        value = _float_or_none(contract.get(key))
        if value is not None:
            return value
    for key in ("per_symbol_lcb", "per_symbol_pnl_lcb", "symbol_lcb"):
        payload = contract.get(key)
        if isinstance(payload, dict):
            values = [_float_or_none(value) for value in payload.values()]
            clean = [value for value in values if value is not None]
            if clean:
                return min(clean)
        if isinstance(payload, list):
            values = [
                _float_or_none(item.get("lcb") if isinstance(item, dict) else item)
                for item in payload
            ]
            clean = [value for value in values if value is not None]
            if clean:
                return min(clean)
    return None


def _contract_max_drawdown_pct(contract: Dict[str, Any], robust: Dict[str, Any]) -> float | None:
    for key in ("max_drawdown_pct", "max_dd_pct", "max_drawdown"):
        value = _float_or_none(contract.get(key))
        if value is not None:
            return value
    periods = contract.get("periods")
    if isinstance(periods, list):
        values = [
            _float_or_none(row.get("max_drawdown_pct"))
            for row in periods
            if isinstance(row, dict)
        ]
        clean = [value for value in values if value is not None]
        if clean:
            return max(clean)
    return _float_or_none(robust.get("max_drawdown_pct"))


def _summaries(report: Dict[str, Any], *, mode: str) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for genome_payload in report.get("genomes", []):
        mode_payload = _mode_payload(genome_payload, mode)
        stats = mode_payload.get("period_stats", {})
        robust = mode_payload.get("robust_score", {})
        contract = mode_payload.get("contract_metrics", {})
        if not isinstance(contract, dict):
            contract = {}
        fitness_v3 = _fitness_v3_from_mode(mode_payload)
        fitness_v4 = _fitness_v4_from_mode(mode_payload)
        min_per_symbol_lcb = _contract_min_per_symbol_lcb(contract)
        max_drawdown_pct = _contract_max_drawdown_pct(contract, robust)
        result.append({
            "path": str(genome_payload.get("path", "")),
            "position_state_features_enabled": bool(
                genome_payload.get("position_state_features_enabled", False)
            ),
            "fitness": float(mode_payload.get("fitness", 0.0)),
            "mean_ret": float(stats.get("mean_ret", 0.0)),
            "min_ret": float(stats.get("min_ret", 0.0)),
            "max_ret": float(stats.get("max_ret", 0.0)),
            "positive_period_pct": float(stats.get("positive_period_pct", 0.0)),
            "max_positive_contribution_pct": float(
                robust.get("max_positive_contribution_pct", 0.0)
            ),
            "passes_default_gates": bool(robust.get("passes_default_gates", False)),
            "fitness_v3_robust": float(fitness_v3.get("fitness_v3_robust", 0.0)),
            "fitness_v3_passes_default_gates": bool(
                fitness_v3.get("passes_default_gates", False)
            ),
            "fitness_v3_failed_gates": list(fitness_v3.get("failed_gates", [])),
            "fitness_v3": fitness_v3,
            "fitness_v4_robust": float(fitness_v4.get("fitness_v4_robust", 0.0)),
            "fitness_v4_passes_default_gates": bool(
                fitness_v4.get("passes_default_gates", False)
            ),
            "fitness_v4_failed_gates": list(fitness_v4.get("failed_gates", [])),
            "fitness_v4": fitness_v4,
            "mean_turnover_rate": float(contract.get("mean_turnover_rate", 0.0)),
            "max_turnover_rate": float(
                contract.get("max_turnover_rate", contract.get("mean_turnover_rate", 0.0))
            ),
            "mean_saturation_rate": float(contract.get("mean_saturation_rate", 0.0)),
            "max_saturation_rate": float(
                contract.get(
                    "max_saturation_rate",
                    contract.get("mean_saturation_rate", 0.0),
                )
            ),
            "mean_invalid_open_logit_pressure": float(
                contract.get("mean_invalid_open_logit_pressure", 0.0)
            ),
            "max_invalid_open_logit_pressure": float(
                contract.get(
                    "max_invalid_open_logit_pressure",
                    contract.get("mean_invalid_open_logit_pressure", 0.0),
                )
            ),
            "mean_long_slot_rate": float(contract.get("mean_long_slot_rate", 0.0)),
            "mean_short_slot_rate": float(contract.get("mean_short_slot_rate", 0.0)),
            "mean_net_direction_bias": float(contract.get("mean_net_direction_bias", 0.0)),
            "min_per_symbol_lcb": min_per_symbol_lcb,
            "max_drawdown_pct": max_drawdown_pct,
            "period_rets": list(mode_payload.get("period_rets", [])),
        })
    return result


def _fitness_v3_from_mode(mode_payload: Dict[str, Any]) -> Dict[str, Any]:
    period_rets = list(mode_payload.get("period_rets", []))
    contract = mode_payload.get("contract_metrics", {})
    periods = contract.get("periods", []) if isinstance(contract, dict) else []
    if not isinstance(periods, list):
        periods = []

    def values(name: str) -> list[float]:
        out: list[float] = []
        for row in periods:
            if not isinstance(row, dict):
                out.append(0.0)
                continue
            try:
                out.append(float(row.get(name, 0.0) or 0.0))
            except (TypeError, ValueError):
                out.append(0.0)
        return out

    regimes = [
        str(row.get("regime", "unknown") if isinstance(row, dict) else "unknown")
        for row in periods
    ]
    return fitness_v3_robust_score(
        period_rets,
        period_max_drawdowns_pct=values("max_drawdown_pct"),
        period_turnover_rates=values("turnover_rate"),
        period_effective_turnover_rates=values("effective_turnover_rate"),
        period_saturation_rates=values("saturation_rate"),
        period_invalid_open_pressures=values("invalid_open_logit_pressure"),
        period_costs_pct=values("cost_pct"),
        period_slippage_pct=values("slippage_pct"),
        period_regimes=regimes,
    )


def _fitness_v4_from_mode(mode_payload: Dict[str, Any]) -> Dict[str, Any]:
    period_rets = list(mode_payload.get("period_rets", []))
    contract = mode_payload.get("contract_metrics", {})
    periods = contract.get("periods", []) if isinstance(contract, dict) else []
    if not isinstance(periods, list):
        periods = []

    def values(name: str) -> list[float]:
        out: list[float] = []
        for row in periods:
            if not isinstance(row, dict):
                out.append(0.0)
                continue
            try:
                out.append(float(row.get(name, 0.0) or 0.0))
            except (TypeError, ValueError):
                out.append(0.0)
        return out

    regimes = [
        str(row.get("regime", "unknown") if isinstance(row, dict) else "unknown")
        for row in periods
    ]
    return fitness_v4_robust_score(
        period_rets,
        period_lcbs_pct=values("min_per_symbol_lcb"),
        period_max_drawdowns_pct=values("max_drawdown_pct"),
        period_turnover_rates=values("turnover_rate"),
        period_effective_turnover_rates=values("effective_turnover_rate"),
        period_saturation_rates=values("saturation_rate"),
        period_invalid_open_pressures=values("invalid_open_logit_pressure"),
        period_costs_pct=values("cost_pct"),
        period_slippage_pct=values("slippage_pct"),
        period_min_notional_pressures=values("min_notional_pressure"),
        period_precision_costs_pct=values("precision_cost_pct"),
        period_bad_signal_key_rates=values("bad_signal_key_rate"),
        period_long_slot_rates=values("mean_long_slot_rate"),
        period_short_slot_rates=values("mean_short_slot_rate"),
        period_net_direction_biases=values("mean_net_direction_bias"),
        period_regimes=regimes,
    )


def _stability_delta_score(candidate: Dict[str, Any], baseline: Dict[str, Any]) -> float:
    return float(
        (candidate["mean_ret"] - baseline["mean_ret"]) * 10.0
        + (candidate["min_ret"] - baseline["min_ret"]) * 3.0
        + (candidate["positive_period_pct"] - baseline["positive_period_pct"]) * 0.02
    )


def _period_summary_for_map(
    report: Dict[str, Any],
    regime_map: Dict[str, str],
    *,
    mode: str,
    fallback_path: str,
) -> Dict[str, Any]:
    by_path: Dict[str, Dict[str, Any]] = {}
    for genome_payload in report.get("genomes", []):
        path = str(genome_payload.get("path", ""))
        mode_payload = _mode_payload(genome_payload, mode)
        contract = mode_payload.get("contract_metrics", {})
        periods = contract.get("periods", [])
        rets = list(mode_payload.get("period_rets", []))
        by_path[path] = {"periods": periods, "rets": rets}

    if fallback_path not in by_path:
        raise ValueError(f"fallback genome not found in report: {fallback_path}")

    baseline_periods = by_path[fallback_path]["periods"]
    selected_rets: List[float] = []
    selected_rows: List[Dict[str, Any]] = []
    turnover_rates: List[float] = []
    saturation_rates: List[float] = []
    invalid_open_pressures: List[float] = []
    drawdowns: List[float] = []
    long_slot_rates: List[float] = []
    short_slot_rates: List[float] = []
    net_direction_biases: List[float] = []
    for idx, row in enumerate(baseline_periods):
        regime = str(row.get("regime", "neutral"))
        path = regime_map.get(regime, fallback_path)
        item = by_path.get(path) or by_path[fallback_path]
        rets = item["rets"]
        if idx >= len(rets):
            raise ValueError(f"period index {idx} missing for genome {path}")
        selected_period = item["periods"][idx] if idx < len(item["periods"]) else {}
        turnover_rate = float(selected_period.get("turnover_rate", 0.0))
        saturation_rate = float(selected_period.get("saturation_rate", 0.0))
        invalid_open_pressure = float(selected_period.get("invalid_open_logit_pressure", 0.0))
        max_drawdown_pct = float(selected_period.get("max_drawdown_pct", 0.0) or 0.0)
        long_slot_rate = float(selected_period.get("mean_long_slot_rate", 0.0) or 0.0)
        short_slot_rate = float(selected_period.get("mean_short_slot_rate", 0.0) or 0.0)
        net_direction_bias = float(selected_period.get("mean_net_direction_bias", 0.0) or 0.0)
        selected_rets.append(float(rets[idx]))
        turnover_rates.append(turnover_rate)
        saturation_rates.append(saturation_rate)
        invalid_open_pressures.append(invalid_open_pressure)
        drawdowns.append(max_drawdown_pct)
        long_slot_rates.append(long_slot_rate)
        short_slot_rates.append(short_slot_rate)
        net_direction_biases.append(net_direction_bias)
        selected_rows.append({
            "period": row.get("period", str(idx)),
            "regime": regime,
            "path": path,
            "ret": float(rets[idx]),
            "turnover_rate": turnover_rate,
            "saturation_rate": saturation_rate,
            "invalid_open_logit_pressure": invalid_open_pressure,
            "max_drawdown_pct": max_drawdown_pct,
            "mean_long_slot_rate": long_slot_rate,
            "mean_short_slot_rate": short_slot_rate,
            "mean_net_direction_bias": net_direction_bias,
        })

    if selected_rets:
        mean_ret = sum(selected_rets) / len(selected_rets)
        min_ret = min(selected_rets)
        max_ret = max(selected_rets)
        positive_period_pct = (
            sum(1 for value in selected_rets if value > 0.0)
            / len(selected_rets)
            * 100.0
        )
        mean_turnover_rate = sum(turnover_rates) / len(turnover_rates)
        max_turnover_rate = max(turnover_rates)
        mean_saturation_rate = sum(saturation_rates) / len(saturation_rates)
        max_saturation_rate = max(saturation_rates)
        mean_invalid_open_logit_pressure = (
            sum(invalid_open_pressures) / len(invalid_open_pressures)
        )
        max_invalid_open_logit_pressure = max(invalid_open_pressures)
        max_drawdown_pct = max(drawdowns)
        mean_long_slot_rate = sum(long_slot_rates) / len(long_slot_rates)
        mean_short_slot_rate = sum(short_slot_rates) / len(short_slot_rates)
        mean_abs_net_direction_bias = (
            sum(abs(value) for value in net_direction_biases)
            / len(net_direction_biases)
        )
    else:
        mean_ret = min_ret = max_ret = positive_period_pct = 0.0
        mean_turnover_rate = max_turnover_rate = 0.0
        mean_saturation_rate = max_saturation_rate = 0.0
        mean_invalid_open_logit_pressure = max_invalid_open_logit_pressure = 0.0
        max_drawdown_pct = 0.0
        mean_long_slot_rate = mean_short_slot_rate = mean_abs_net_direction_bias = 0.0

    return {
        "mean_ret": float(mean_ret),
        "min_ret": float(min_ret),
        "max_ret": float(max_ret),
        "positive_period_pct": float(positive_period_pct),
        "mean_turnover_rate": float(mean_turnover_rate),
        "max_turnover_rate": float(max_turnover_rate),
        "mean_saturation_rate": float(mean_saturation_rate),
        "max_saturation_rate": float(max_saturation_rate),
        "mean_invalid_open_logit_pressure": float(mean_invalid_open_logit_pressure),
        "max_invalid_open_logit_pressure": float(max_invalid_open_logit_pressure),
        "max_drawdown_pct": float(max_drawdown_pct),
        "mean_long_slot_rate": float(mean_long_slot_rate),
        "mean_short_slot_rate": float(mean_short_slot_rate),
        "mean_abs_net_direction_bias": float(mean_abs_net_direction_bias),
        "min_per_symbol_lcb": None,
        "period_rets": selected_rets,
        "periods": selected_rows,
    }


def select_regime_router(
    train_report: Dict[str, Any],
    validation_report: Dict[str, Any],
    *,
    mode: str = "fee_fixed_nextbar",
    guard_reports: List[Dict[str, Any]] | None = None,
    allowed_candidate_regimes: List[str] | None = None,
    min_train_mean_delta: float = 0.0,
    min_validation_mean_delta: float = 0.0,
    min_validation_min_ret_delta: float = 0.0,
    min_guard_mean_delta: float = 0.0,
    min_guard_min_ret_delta: float = 0.0,
    min_positive_period_pct: float = 0.0,
    min_validation_periods: int = 2,
    min_validation_regime_periods: int = 1,
    max_turnover_rate: float = 0.10,
    max_saturation_rate: float = 0.10,
    max_invalid_open_pressure: float = 0.05,
) -> Dict[str, Any]:
    validation_items = _summaries(validation_report, mode=mode)
    if not validation_items:
        raise ValueError("validation report contains no genomes")

    baseline_path = validation_items[0]["path"]
    candidate_paths = [item["path"] for item in validation_items]
    regimes = ROUTER_REGIMES
    allowed_regimes = set(allowed_candidate_regimes or regimes)
    baseline_map = {regime: baseline_path for regime in regimes}
    baseline_train = _period_summary_for_map(
        train_report,
        baseline_map,
        mode=mode,
        fallback_path=baseline_path,
    )
    baseline_validation = _period_summary_for_map(
        validation_report,
        baseline_map,
        mode=mode,
        fallback_path=baseline_path,
    )
    guard_reports = list(guard_reports or [])
    baseline_guards = [
        _period_summary_for_map(
            report,
            baseline_map,
            mode=mode,
            fallback_path=baseline_path,
        )
        for report in guard_reports
    ]

    best_map = baseline_map
    best_train = baseline_train
    best_validation = baseline_validation
    best_guards = baseline_guards
    best_score = float("-inf")
    candidates: List[Dict[str, Any]] = []

    for combo in itertools.product(candidate_paths, repeat=len(regimes)):
        regime_map = dict(zip(regimes, combo))
        if regime_map == baseline_map:
            continue
        if any(
            regime not in allowed_regimes and path != baseline_path
            for regime, path in regime_map.items()
        ):
            continue

        train = _period_summary_for_map(
            train_report,
            regime_map,
            mode=mode,
            fallback_path=baseline_path,
        )
        validation = _period_summary_for_map(
            validation_report,
            regime_map,
            mode=mode,
            fallback_path=baseline_path,
        )
        guards = [
            _period_summary_for_map(
                report,
                regime_map,
                mode=mode,
                fallback_path=baseline_path,
            )
            for report in guard_reports
        ]
        train_mean_delta = train["mean_ret"] - baseline_train["mean_ret"]
        validation_mean_delta = validation["mean_ret"] - baseline_validation["mean_ret"]
        validation_min_delta = validation["min_ret"] - baseline_validation["min_ret"]

        failures: List[str] = []
        if train_mean_delta <= min_train_mean_delta:
            failures.append("train_mean_ret")
        if validation_mean_delta < min_validation_mean_delta:
            failures.append("validation_mean_ret")
        if validation_min_delta < min_validation_min_ret_delta:
            failures.append("validation_min_ret")
        if validation["positive_period_pct"] < min_positive_period_pct:
            failures.append("validation_positive_period_pct")
        if validation["max_turnover_rate"] > max_turnover_rate:
            failures.append("validation_turnover")
        if validation["max_saturation_rate"] > max_saturation_rate:
            failures.append("validation_saturation")
        if validation["max_invalid_open_logit_pressure"] > max_invalid_open_pressure:
            failures.append("validation_invalid_open_pressure")
        for guard_idx, (guard, baseline_guard) in enumerate(zip(guards, baseline_guards)):
            guard_mean_delta = guard["mean_ret"] - baseline_guard["mean_ret"]
            guard_min_delta = guard["min_ret"] - baseline_guard["min_ret"]
            guard_positive_delta = (
                guard["positive_period_pct"] - baseline_guard["positive_period_pct"]
            )
            if guard_mean_delta < min_guard_mean_delta:
                failures.append(f"guard_{guard_idx}_mean_ret")
            if guard_min_delta < min_guard_min_ret_delta:
                failures.append(f"guard_{guard_idx}_min_ret")
            if guard_positive_delta < 0.0:
                failures.append(f"guard_{guard_idx}_positive_period_pct")
            if guard["max_turnover_rate"] > max_turnover_rate:
                failures.append(f"guard_{guard_idx}_turnover")
            if guard["max_saturation_rate"] > max_saturation_rate:
                failures.append(f"guard_{guard_idx}_saturation")
            if guard["max_invalid_open_logit_pressure"] > max_invalid_open_pressure:
                failures.append(f"guard_{guard_idx}_invalid_open_pressure")

        guard_stability_score = sum(
            _stability_delta_score(guard, baseline_guard)
            for guard, baseline_guard in zip(guards, baseline_guards)
        )
        score = (
            train_mean_delta
            + validation_mean_delta * 10.0
            + validation_min_delta * 3.0
            + guard_stability_score
        )
        item = {
            "accepted": not failures,
            "failures": failures,
            "score": float(score),
            "guard_stability_score": float(guard_stability_score),
            "regime_map": regime_map,
            "train": train,
            "validation": validation,
            "guards": guards,
        }
        candidates.append(item)
        if not failures and score > best_score:
            best_score = score
            best_map = regime_map
            best_train = train
            best_validation = validation
            best_guards = guards

    selected_is_baseline = best_map == baseline_map
    promotion_failures: List[str] = []
    if selected_is_baseline:
        promotion_failures.append("baseline_selected")
    validation_mean_delta = best_validation["mean_ret"] - baseline_validation["mean_ret"]
    validation_min_delta = best_validation["min_ret"] - baseline_validation["min_ret"]
    validation_positive_delta = (
        best_validation["positive_period_pct"]
        - baseline_validation["positive_period_pct"]
    )
    if validation_mean_delta <= 0.0:
        promotion_failures.append("validation_tie" if validation_mean_delta == 0.0 else "validation_mean_ret")
    if validation_min_delta < 0.0:
        promotion_failures.append("validation_min_ret")
    if validation_positive_delta < 0.0:
        promotion_failures.append("validation_positive_period_pct")
    if len(best_validation.get("period_rets", [])) < min_validation_periods:
        promotion_failures.append("validation_period_count")
    validation_regime_counts: Dict[str, int] = {}
    for row in best_validation.get("periods", []):
        regime = str(row.get("regime", "neutral"))
        validation_regime_counts[regime] = validation_regime_counts.get(regime, 0) + 1
    for regime, selected_path in best_map.items():
        if selected_path == baseline_map.get(regime):
            continue
        if validation_regime_counts.get(regime, 0) < min_validation_regime_periods:
            promotion_failures.append(f"validation_regime_coverage:{regime}")
    return {
        "mode": mode,
        "selected_is_baseline": selected_is_baseline,
        "promotion_eligible": not promotion_failures,
        "promotion_failures": promotion_failures,
        "selected_regime_map": best_map,
        "baseline_regime_map": baseline_map,
        "selection_score": None if selected_is_baseline else float(best_score),
        "train": best_train,
        "validation": best_validation,
        "guards": best_guards,
        "baseline_train": baseline_train,
        "baseline_validation": baseline_validation,
        "baseline_guards": baseline_guards,
        "thresholds": {
            "min_train_mean_delta": min_train_mean_delta,
            "min_validation_mean_delta": min_validation_mean_delta,
            "min_validation_min_ret_delta": min_validation_min_ret_delta,
            "min_guard_mean_delta": min_guard_mean_delta,
            "min_guard_min_ret_delta": min_guard_min_ret_delta,
            "min_positive_period_pct": min_positive_period_pct,
            "min_validation_periods": min_validation_periods,
            "min_validation_regime_periods": min_validation_regime_periods,
            "allowed_candidate_regimes": sorted(allowed_regimes),
            "max_turnover_rate": max_turnover_rate,
            "max_saturation_rate": max_saturation_rate,
            "max_invalid_open_pressure": max_invalid_open_pressure,
        },
        "candidates": candidates,
    }


def evaluate_regime_router_multisplit_gate(
    selection: Dict[str, Any],
    holdout_reports: List[Dict[str, Any]],
    *,
    mode: str = "fee_fixed_nextbar",
    min_oos_mean_delta: float = 0.0,
    min_oos_min_ret_delta: float = 0.0,
    min_oos_regime_periods: int = 1,
    max_turnover_rate: float = 0.10,
    max_saturation_rate: float = 0.10,
    max_invalid_open_pressure: float = 0.05,
) -> Dict[str, Any]:
    selected_map = dict(selection.get("selected_regime_map") or {})
    baseline_map = dict(selection.get("baseline_regime_map") or {})
    if not selected_map or not baseline_map:
        raise ValueError("selection must contain selected_regime_map and baseline_regime_map")

    baseline_paths = {str(path) for path in baseline_map.values()}
    if len(baseline_paths) != 1:
        raise ValueError("baseline_regime_map must point to one baseline genome")
    fallback_path = next(iter(baseline_paths))

    failures: List[str] = []
    if not bool(selection.get("promotion_eligible", False)):
        failures.append("selection_promotion_gate")

    holdouts: List[Dict[str, Any]] = []
    oos_regime_counts: Dict[str, int] = {}
    for idx, report in enumerate(holdout_reports):
        baseline = _period_summary_for_map(
            report,
            baseline_map,
            mode=mode,
            fallback_path=fallback_path,
        )
        candidate = _period_summary_for_map(
            report,
            selected_map,
            mode=mode,
            fallback_path=fallback_path,
        )
        mean_delta = candidate["mean_ret"] - baseline["mean_ret"]
        min_delta = candidate["min_ret"] - baseline["min_ret"]
        positive_delta = candidate["positive_period_pct"] - baseline["positive_period_pct"]
        split_failures: List[str] = []
        if mean_delta <= min_oos_mean_delta:
            failure = (
                f"oos_{idx}_mean_ret_tie"
                if abs(mean_delta - min_oos_mean_delta) <= 1e-12
                else f"oos_{idx}_mean_ret"
            )
            split_failures.append(failure)
        if min_delta < min_oos_min_ret_delta:
            split_failures.append(f"oos_{idx}_min_ret")
        if positive_delta < 0.0:
            split_failures.append(f"oos_{idx}_positive_period_pct")
        if candidate["mean_turnover_rate"] > max_turnover_rate:
            split_failures.append(f"oos_{idx}_turnover")
        if candidate["max_turnover_rate"] > max_turnover_rate:
            split_failures.append(f"oos_{idx}_max_turnover")
        if candidate["mean_saturation_rate"] > max_saturation_rate:
            split_failures.append(f"oos_{idx}_saturation")
        if candidate["max_saturation_rate"] > max_saturation_rate:
            split_failures.append(f"oos_{idx}_max_saturation")
        if candidate["mean_invalid_open_logit_pressure"] > max_invalid_open_pressure:
            split_failures.append(f"oos_{idx}_invalid_open_pressure")
        failures.extend(split_failures)
        for row in candidate.get("periods", []):
            regime = str(row.get("regime", "neutral"))
            oos_regime_counts[regime] = oos_regime_counts.get(regime, 0) + 1
        holdouts.append({
            "index": idx,
            "accepted": not split_failures,
            "failures": split_failures,
            "baseline": baseline,
            "candidate": candidate,
            "mean_ret_delta": float(mean_delta),
            "min_ret_delta": float(min_delta),
            "positive_period_pct_delta": float(positive_delta),
        })

    for regime, selected_path in selected_map.items():
        if selected_path == baseline_map.get(regime):
            continue
        if oos_regime_counts.get(regime, 0) < min_oos_regime_periods:
            failures.append(f"oos_regime_coverage:{regime}")

    return {
        "promotion_eligible": not failures,
        "promotion_failures": failures,
        "holdouts": holdouts,
        "thresholds": {
            "min_oos_mean_delta": min_oos_mean_delta,
            "min_oos_min_ret_delta": min_oos_min_ret_delta,
            "min_oos_regime_periods": min_oos_regime_periods,
            "max_turnover_rate": max_turnover_rate,
            "max_saturation_rate": max_saturation_rate,
            "max_invalid_open_pressure": max_invalid_open_pressure,
        },
    }


def _paper_failure_name(name: str) -> str:
    if name.startswith("oos_"):
        return "paper_" + name[len("oos_"):]
    return name


def evaluate_regime_router_paper_gate(
    selection: Dict[str, Any],
    paper_reports: List[Dict[str, Any]],
    *,
    mode: str = "fee_fixed_nextbar",
    min_paper_mean_delta: float = 0.0,
    min_paper_min_ret_delta: float = 0.0,
    min_paper_regime_periods: int = 1,
    max_turnover_rate: float = 0.10,
    max_saturation_rate: float = 0.10,
    max_invalid_open_pressure: float = 0.05,
) -> Dict[str, Any]:
    gate = evaluate_regime_router_multisplit_gate(
        selection,
        paper_reports,
        mode=mode,
        min_oos_mean_delta=min_paper_mean_delta,
        min_oos_min_ret_delta=min_paper_min_ret_delta,
        min_oos_regime_periods=min_paper_regime_periods,
        max_turnover_rate=max_turnover_rate,
        max_saturation_rate=max_saturation_rate,
        max_invalid_open_pressure=max_invalid_open_pressure,
    )
    failures = [_paper_failure_name(str(item)) for item in gate.get("promotion_failures", [])]
    paper_reports_payload: List[Dict[str, Any]] = []
    for item in gate.get("holdouts", []):
        payload = dict(item)
        payload["failures"] = [_paper_failure_name(str(value)) for value in item.get("failures", [])]
        payload["accepted"] = not payload["failures"]
        paper_reports_payload.append(payload)

    return {
        "paper_trading_eligible": not failures,
        "live_trading_eligible": False,
        "paper_failures": failures,
        "paper_reports": paper_reports_payload,
        "thresholds": {
            "min_paper_mean_delta": min_paper_mean_delta,
            "min_paper_min_ret_delta": min_paper_min_ret_delta,
            "min_paper_regime_periods": min_paper_regime_periods,
            "max_turnover_rate": max_turnover_rate,
            "max_saturation_rate": max_saturation_rate,
            "max_invalid_open_pressure": max_invalid_open_pressure,
        },
    }


def select_candidate(
    train_report: Dict[str, Any],
    validation_report: Dict[str, Any],
    *,
    mode: str = "fee_fixed_nextbar",
    guard_reports: List[Dict[str, Any]] | None = None,
    min_validation_mean_delta: float = 0.0,
    min_validation_min_ret_delta: float = 0.0,
    min_guard_mean_delta: float = 0.0,
    min_guard_min_ret_delta: float = 0.0,
    min_positive_period_pct: float = 0.0,
    max_turnover_rate: float = 0.10,
    max_saturation_rate: float = 0.10,
    max_invalid_open_pressure: float = 0.05,
    use_fitness_v3_robust: bool = False,
    use_fitness_v4_robust: bool = False,
) -> Dict[str, Any]:
    train_items = {item["path"]: item for item in _summaries(train_report, mode=mode)}
    validation_items = _summaries(validation_report, mode=mode)
    if not validation_items:
        raise ValueError("validation report contains no genomes")

    baseline = validation_items[0]
    best = baseline
    best_score = float("-inf")
    candidates: List[Dict[str, Any]] = []
    guard_reports = list(guard_reports or [])
    guard_items_by_report: List[tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]] = []
    for guard_report in guard_reports:
        guard_items = {item["path"]: item for item in _summaries(guard_report, mode=mode)}
        if baseline["path"] not in guard_items:
            raise ValueError(f"baseline genome not found in guard report: {baseline['path']}")
        guard_items_by_report.append((guard_items[baseline["path"]], guard_items))

    for candidate in validation_items[1:]:
        failures: List[str] = []
        if candidate["mean_ret"] < baseline["mean_ret"] + min_validation_mean_delta:
            failures.append("validation_mean_ret")
        if candidate["min_ret"] < baseline["min_ret"] + min_validation_min_ret_delta:
            failures.append("validation_min_ret")
        if candidate["positive_period_pct"] < min_positive_period_pct:
            failures.append("validation_positive_period_pct")
        if candidate["mean_turnover_rate"] > max_turnover_rate:
            failures.append("turnover")
        if candidate["mean_saturation_rate"] > max_saturation_rate:
            failures.append("saturation")
        if candidate["mean_invalid_open_logit_pressure"] > max_invalid_open_pressure:
            failures.append("invalid_open_pressure")
        if use_fitness_v3_robust:
            if not candidate["fitness_v3_passes_default_gates"]:
                failures.append("fitness_v3_gates")
            if candidate["fitness_v3_robust"] <= baseline["fitness_v3_robust"]:
                failures.append("validation_fitness_v3")
        if use_fitness_v4_robust:
            if not candidate["fitness_v4_passes_default_gates"]:
                failures.append("fitness_v4_gates")
            if candidate["fitness_v4_robust"] <= baseline["fitness_v4_robust"]:
                failures.append("validation_fitness_v4")

        train_item = train_items.get(candidate["path"])
        train_mean_delta = (
            float(train_item["mean_ret"] - train_items.get(baseline["path"], baseline)["mean_ret"])
            if train_item is not None and baseline["path"] in train_items
            else 0.0
        )
        validation_mean_delta = candidate["mean_ret"] - baseline["mean_ret"]
        validation_min_delta = candidate["min_ret"] - baseline["min_ret"]
        if validation_mean_delta <= 0.0 and validation_min_delta <= 0.0:
            failures.append("validation_tie")
        guards: List[Dict[str, Any]] = []
        guard_stability_score = 0.0
        for guard_idx, (baseline_guard, guard_items) in enumerate(guard_items_by_report):
            guard_item = guard_items.get(candidate["path"])
            if guard_item is None:
                failures.append(f"guard_{guard_idx}_missing")
                continue
            guard_mean_delta = guard_item["mean_ret"] - baseline_guard["mean_ret"]
            guard_min_delta = guard_item["min_ret"] - baseline_guard["min_ret"]
            guard_positive_delta = (
                guard_item["positive_period_pct"] - baseline_guard["positive_period_pct"]
            )
            if guard_mean_delta < min_guard_mean_delta:
                failures.append(f"guard_{guard_idx}_mean_ret")
            if guard_min_delta < min_guard_min_ret_delta:
                failures.append(f"guard_{guard_idx}_min_ret")
            if guard_positive_delta < 0.0:
                failures.append(f"guard_{guard_idx}_positive_period_pct")
            if guard_item["mean_turnover_rate"] > max_turnover_rate:
                failures.append(f"guard_{guard_idx}_turnover")
            if guard_item["mean_saturation_rate"] > max_saturation_rate:
                failures.append(f"guard_{guard_idx}_saturation")
            if guard_item["mean_invalid_open_logit_pressure"] > max_invalid_open_pressure:
                failures.append(f"guard_{guard_idx}_invalid_open_pressure")
            guard_stability_score += _stability_delta_score(guard_item, baseline_guard)
            guards.append({
                "index": guard_idx,
                "baseline": baseline_guard,
                "candidate": guard_item,
                "mean_ret_delta": float(guard_mean_delta),
                "min_ret_delta": float(guard_min_delta),
                "positive_period_pct_delta": float(guard_positive_delta),
            })
        score = (
            validation_mean_delta * 10.0
            + validation_min_delta * 3.0
            + train_mean_delta
            + (candidate["positive_period_pct"] - baseline["positive_period_pct"]) * 0.02
            + guard_stability_score
        )
        if use_fitness_v3_robust:
            score += (
                candidate["fitness_v3_robust"] - baseline["fitness_v3_robust"]
            ) * 5.0
        if use_fitness_v4_robust:
            score += (
                candidate["fitness_v4_robust"] - baseline["fitness_v4_robust"]
            ) * 5.0
        item = {
            "path": candidate["path"],
            "accepted": not failures,
            "failures": failures,
            "score": score,
            "guard_stability_score": float(guard_stability_score),
            "train": train_item,
            "validation": candidate,
            "guards": guards,
        }
        candidates.append(item)
        if not failures and score > best_score:
            best = candidate
            best_score = score

    selected_is_baseline = best["path"] == baseline["path"]
    promotion_failures: List[str] = []
    if selected_is_baseline:
        promotion_failures.append("baseline_selected")
    return {
        "mode": mode,
        "selected_path": best["path"],
        "selected_is_baseline": selected_is_baseline,
        "promotion_eligible": not promotion_failures,
        "promotion_failures": promotion_failures,
        "baseline_path": baseline["path"],
        "baseline_validation": baseline,
        "baseline_guards": [
            baseline_guard
            for baseline_guard, _items in guard_items_by_report
        ],
        "selection_score": None if selected_is_baseline else best_score,
        "thresholds": {
            "min_validation_mean_delta": min_validation_mean_delta,
            "min_validation_min_ret_delta": min_validation_min_ret_delta,
            "min_guard_mean_delta": min_guard_mean_delta,
            "min_guard_min_ret_delta": min_guard_min_ret_delta,
            "min_positive_period_pct": min_positive_period_pct,
            "max_turnover_rate": max_turnover_rate,
            "max_saturation_rate": max_saturation_rate,
            "max_invalid_open_pressure": max_invalid_open_pressure,
            "use_fitness_v3_robust": use_fitness_v3_robust,
            "use_fitness_v4_robust": use_fitness_v4_robust,
        },
        "candidates": candidates,
    }


def evaluate_single_candidate_multisplit_gate(
    selection: Dict[str, Any],
    holdout_reports: List[Dict[str, Any]],
    *,
    mode: str = "fee_fixed_nextbar",
    max_turnover_rate: float = 0.10,
    max_saturation_rate: float = 0.10,
    max_invalid_open_pressure: float = 0.05,
    use_fitness_v3_robust: bool = False,
    use_fitness_v4_robust: bool = False,
) -> Dict[str, Any]:
    selected_path = str(selection.get("selected_path") or "")
    baseline_path = str(selection.get("baseline_path") or "")
    if not selected_path or not baseline_path:
        raise ValueError("selection must contain selected_path and baseline_path")

    failures: List[str] = []
    if not bool(selection.get("promotion_eligible", False)):
        failures.append("selection_promotion_gate")

    holdouts: List[Dict[str, Any]] = []
    for idx, report in enumerate(holdout_reports):
        items = {item["path"]: item for item in _summaries(report, mode=mode)}
        if baseline_path not in items:
            raise ValueError(f"baseline genome not found in holdout report: {baseline_path}")
        if selected_path not in items:
            raise ValueError(f"selected genome not found in holdout report: {selected_path}")
        baseline = items[baseline_path]
        candidate = items[selected_path]
        mean_delta = candidate["mean_ret"] - baseline["mean_ret"]
        min_delta = candidate["min_ret"] - baseline["min_ret"]
        positive_delta = candidate["positive_period_pct"] - baseline["positive_period_pct"]
        crash_delta = (
            float(candidate["fitness_v3"].get("crash_floor_ret", 0.0))
            - float(baseline["fitness_v3"].get("crash_floor_ret", 0.0))
        )
        fitness_v3_delta = candidate["fitness_v3_robust"] - baseline["fitness_v3_robust"]
        fitness_v4_delta = candidate["fitness_v4_robust"] - baseline["fitness_v4_robust"]
        split_failures: List[str] = []
        if mean_delta < 0.0:
            split_failures.append(f"oos_{idx}_mean_ret")
        if min_delta < 0.0:
            split_failures.append(f"oos_{idx}_min_ret")
        if positive_delta < 0.0:
            split_failures.append(f"oos_{idx}_positive_period_pct")
        if crash_delta < 0.0:
            split_failures.append(f"oos_{idx}_crash_floor")
        if candidate["max_turnover_rate"] > max_turnover_rate:
            split_failures.append(f"oos_{idx}_max_turnover")
        if candidate["max_saturation_rate"] > max_saturation_rate:
            split_failures.append(f"oos_{idx}_max_saturation")
        if candidate["max_invalid_open_logit_pressure"] > max_invalid_open_pressure:
            split_failures.append(f"oos_{idx}_max_invalid_open_pressure")
        if use_fitness_v3_robust:
            if not candidate["fitness_v3_passes_default_gates"]:
                split_failures.append(f"oos_{idx}_fitness_v3_gates")
            if fitness_v3_delta < 0.0:
                split_failures.append(f"oos_{idx}_fitness_v3")
        if use_fitness_v4_robust:
            if not candidate["fitness_v4_passes_default_gates"]:
                split_failures.append(f"oos_{idx}_fitness_v4_gates")
            if fitness_v4_delta < 0.0:
                split_failures.append(f"oos_{idx}_fitness_v4")
        failures.extend(split_failures)
        holdouts.append({
            "index": idx,
            "accepted": not split_failures,
            "failures": split_failures,
            "baseline": baseline,
            "candidate": candidate,
            "mean_ret_delta": float(mean_delta),
            "min_ret_delta": float(min_delta),
            "positive_period_pct_delta": float(positive_delta),
            "crash_floor_delta": float(crash_delta),
            "fitness_v3_delta": float(fitness_v3_delta),
            "fitness_v4_delta": float(fitness_v4_delta),
        })

    return {
        "promotion_eligible": not failures,
        "promotion_failures": list(dict.fromkeys(failures)),
        "holdouts": holdouts,
        "thresholds": {
            "max_turnover_rate": max_turnover_rate,
            "max_saturation_rate": max_saturation_rate,
            "max_invalid_open_pressure": max_invalid_open_pressure,
            "use_fitness_v3_robust": use_fitness_v3_robust,
            "use_fitness_v4_robust": use_fitness_v4_robust,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select a genetics candidate using train and validation contract reports."
    )
    parser.add_argument("--train-report", required=True)
    parser.add_argument("--validation-report", required=True)
    parser.add_argument(
        "--final-report",
        action="append",
        default=None,
        help="Optional final/OOS contract report. Can be repeated for multi-split gating.",
    )
    parser.add_argument(
        "--guard-report",
        action="append",
        default=None,
        help="Optional robustness report used to filter regime-router maps before final/OOS gating.",
    )
    parser.add_argument(
        "--paper-report",
        action="append",
        default=None,
        help="Optional paper/shadow contract report used to set paper_trading_eligible only.",
    )
    parser.add_argument(
        "--allowed-candidate-regime",
        action="append",
        choices=ROUTER_REGIMES,
        default=None,
        help="Restrict non-baseline regime-router assignments to these regimes. Repeatable.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", default="fee_fixed_nextbar")
    parser.add_argument(
        "--exchange",
        default="",
        help="Required exchange/domain for promotion eligibility, for example MEXC or BITGET.",
    )
    parser.add_argument(
        "--baseline-kind",
        default="genome",
        choices=("genome", "flash"),
        help="Promotion baseline kind. Real promotion requires flash.",
    )
    parser.add_argument("--min-per-symbol-lcb", type=float, default=0.0)
    parser.add_argument("--max-oos-drawdown-pct", type=float, default=12.0)
    parser.add_argument(
        "--allow-non-flash-baseline",
        action="store_true",
        help="Keep promotion eligible without a Flash baseline. Intended for diagnostics only.",
    )
    parser.add_argument(
        "--allow-missing-oos-gate",
        action="store_true",
        help="Keep promotion eligible without final/OOS reports. Intended for diagnostics only.",
    )
    parser.add_argument(
        "--regime-router",
        action="store_true",
        help="Select a validation-safe per-regime genome map instead of one global genome.",
    )
    parser.add_argument("--min-train-mean-delta", type=float, default=0.0)
    parser.add_argument("--min-validation-mean-delta", type=float, default=0.0)
    parser.add_argument("--min-validation-min-ret-delta", type=float, default=0.0)
    parser.add_argument("--min-guard-mean-delta", type=float, default=0.0)
    parser.add_argument("--min-guard-min-ret-delta", type=float, default=0.0)
    parser.add_argument("--min-paper-mean-delta", type=float, default=0.0)
    parser.add_argument("--min-paper-min-ret-delta", type=float, default=0.0)
    parser.add_argument("--min-validation-periods", type=int, default=2)
    parser.add_argument("--min-validation-regime-periods", type=int, default=1)
    parser.add_argument("--min-oos-regime-periods", type=int, default=1)
    parser.add_argument("--min-paper-regime-periods", type=int, default=1)
    parser.add_argument("--min-positive-period-pct", type=float, default=0.0)
    parser.add_argument("--max-turnover-rate", type=float, default=0.10)
    parser.add_argument("--max-saturation-rate", type=float, default=0.10)
    parser.add_argument("--max-invalid-open-pressure", type=float, default=0.05)
    parser.add_argument("--use-fitness-v3-robust", action="store_true")
    parser.add_argument("--use-fitness-v4-robust", action="store_true")
    args = parser.parse_args(argv)

    train_report_path = Path(args.train_report)
    validation_report_path = Path(args.validation_report)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_report = _load_json(train_report_path)
    validation_report = _load_json(validation_report_path)
    reports_for_strict: List[tuple[str, Dict[str, Any]]] = [
        ("train", train_report),
        ("validation", validation_report),
    ]
    guard_reports = [_load_json(Path(path)) for path in args.guard_report or []]
    reports_for_strict.extend(
        (f"guard_{idx}", report)
        for idx, report in enumerate(guard_reports)
    )
    if args.regime_router:
        selection = select_regime_router(
            train_report,
            validation_report,
            mode=args.mode,
            guard_reports=guard_reports,
            allowed_candidate_regimes=args.allowed_candidate_regime,
            min_train_mean_delta=args.min_train_mean_delta,
            min_validation_mean_delta=args.min_validation_mean_delta,
            min_validation_min_ret_delta=args.min_validation_min_ret_delta,
            min_guard_mean_delta=args.min_guard_mean_delta,
            min_guard_min_ret_delta=args.min_guard_min_ret_delta,
            min_positive_period_pct=args.min_positive_period_pct,
            min_validation_periods=args.min_validation_periods,
            min_validation_regime_periods=args.min_validation_regime_periods,
            max_turnover_rate=args.max_turnover_rate,
            max_saturation_rate=args.max_saturation_rate,
            max_invalid_open_pressure=args.max_invalid_open_pressure,
        )
        if args.final_report:
            final_reports = [_load_json(Path(path)) for path in args.final_report]
            reports_for_strict.extend(
                (f"final_{idx}", report)
                for idx, report in enumerate(final_reports)
            )
            final_gate = evaluate_regime_router_multisplit_gate(
                selection,
                final_reports,
                mode=args.mode,
                min_oos_regime_periods=args.min_oos_regime_periods,
                max_turnover_rate=args.max_turnover_rate,
                max_saturation_rate=args.max_saturation_rate,
                max_invalid_open_pressure=args.max_invalid_open_pressure,
            )
            selection["final_holdout_gate"] = final_gate
            selection["promotion_eligible"] = (
                bool(selection.get("promotion_eligible", False))
                and bool(final_gate.get("promotion_eligible", False))
            )
            merged_failures = list(selection.get("promotion_failures", []))
            merged_failures.extend(final_gate.get("promotion_failures", []))
            selection["promotion_failures"] = list(dict.fromkeys(merged_failures))
        if args.paper_report:
            paper_reports = [_load_json(Path(path)) for path in args.paper_report]
            reports_for_strict.extend(
                (f"paper_{idx}", report)
                for idx, report in enumerate(paper_reports)
            )
            paper_gate = evaluate_regime_router_paper_gate(
                selection,
                paper_reports,
                mode=args.mode,
                min_paper_mean_delta=args.min_paper_mean_delta,
                min_paper_min_ret_delta=args.min_paper_min_ret_delta,
                min_paper_regime_periods=args.min_paper_regime_periods,
                max_turnover_rate=args.max_turnover_rate,
                max_saturation_rate=args.max_saturation_rate,
                max_invalid_open_pressure=args.max_invalid_open_pressure,
            )
            selection["paper_gate"] = paper_gate
            selection["paper_trading_eligible"] = bool(
                paper_gate.get("paper_trading_eligible", False)
            )
            selection["paper_failures"] = list(paper_gate.get("paper_failures", []))
            selection["live_trading_eligible"] = False
    else:
        selection = select_candidate(
            train_report,
            validation_report,
            mode=args.mode,
            guard_reports=guard_reports,
            min_validation_mean_delta=args.min_validation_mean_delta,
            min_validation_min_ret_delta=args.min_validation_min_ret_delta,
            min_guard_mean_delta=args.min_guard_mean_delta,
            min_guard_min_ret_delta=args.min_guard_min_ret_delta,
            min_positive_period_pct=args.min_positive_period_pct,
            max_turnover_rate=args.max_turnover_rate,
            max_saturation_rate=args.max_saturation_rate,
            max_invalid_open_pressure=args.max_invalid_open_pressure,
            use_fitness_v3_robust=args.use_fitness_v3_robust,
            use_fitness_v4_robust=args.use_fitness_v4_robust,
        )
        if args.final_report:
            final_reports = [_load_json(Path(path)) for path in args.final_report]
            reports_for_strict.extend(
                (f"final_{idx}", report)
                for idx, report in enumerate(final_reports)
            )
            final_gate = evaluate_single_candidate_multisplit_gate(
                selection,
                final_reports,
                mode=args.mode,
                max_turnover_rate=args.max_turnover_rate,
                max_saturation_rate=args.max_saturation_rate,
                max_invalid_open_pressure=args.max_invalid_open_pressure,
                use_fitness_v3_robust=args.use_fitness_v3_robust,
                use_fitness_v4_robust=args.use_fitness_v4_robust,
            )
            selection["final_holdout_gate"] = final_gate
            selection["promotion_eligible"] = (
                bool(selection.get("promotion_eligible", False))
                and bool(final_gate.get("promotion_eligible", False))
            )
            merged_failures = list(selection.get("promotion_failures", []))
            merged_failures.extend(final_gate.get("promotion_failures", []))
            selection["promotion_failures"] = list(dict.fromkeys(merged_failures))
    apply_strict_promotion_contracts(
        selection,
        reports=reports_for_strict,
        expected_exchange=args.exchange,
        baseline_kind=args.baseline_kind,
        require_flash_baseline=not args.allow_non_flash_baseline,
        require_oos_gate=not args.allow_missing_oos_gate,
        min_per_symbol_lcb=args.min_per_symbol_lcb,
        max_oos_drawdown_pct=args.max_oos_drawdown_pct,
    )
    out_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(selection, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
