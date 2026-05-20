from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mode_payload(genome_payload: Dict[str, Any], mode: str) -> Dict[str, Any]:
    for item in genome_payload.get("modes", []):
        if item.get("mode") == mode:
            return item
    raise KeyError(f"mode {mode!r} not found for {genome_payload.get('path')}")


def _summaries(report: Dict[str, Any], *, mode: str) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for genome_payload in report.get("genomes", []):
        mode_payload = _mode_payload(genome_payload, mode)
        stats = mode_payload.get("period_stats", {})
        robust = mode_payload.get("robust_score", {})
        contract = mode_payload.get("contract_metrics", {})
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
            "mean_turnover_rate": float(contract.get("mean_turnover_rate", 0.0)),
            "mean_saturation_rate": float(contract.get("mean_saturation_rate", 0.0)),
            "mean_invalid_open_logit_pressure": float(
                contract.get("mean_invalid_open_logit_pressure", 0.0)
            ),
            "period_rets": list(mode_payload.get("period_rets", [])),
        })
    return result


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
        selected_rets.append(float(rets[idx]))
        turnover_rates.append(turnover_rate)
        saturation_rates.append(saturation_rate)
        invalid_open_pressures.append(invalid_open_pressure)
        selected_rows.append({
            "period": row.get("period", str(idx)),
            "regime": regime,
            "path": path,
            "ret": float(rets[idx]),
            "turnover_rate": turnover_rate,
            "saturation_rate": saturation_rate,
            "invalid_open_logit_pressure": invalid_open_pressure,
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
    else:
        mean_ret = min_ret = max_ret = positive_period_pct = 0.0
        mean_turnover_rate = max_turnover_rate = 0.0
        mean_saturation_rate = max_saturation_rate = 0.0
        mean_invalid_open_logit_pressure = max_invalid_open_logit_pressure = 0.0

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
) -> Dict[str, Any]:
    validation_items = _summaries(validation_report, mode=mode)
    if not validation_items:
        raise ValueError("validation report contains no genomes")

    baseline_path = validation_items[0]["path"]
    candidate_paths = [item["path"] for item in validation_items]
    regimes = ("bearish", "neutral", "bullish")
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

        score = (
            train_mean_delta
            + validation_mean_delta * 10.0
            + validation_min_delta * 3.0
        )
        item = {
            "accepted": not failures,
            "failures": failures,
            "score": float(score),
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
        if mean_delta < min_oos_mean_delta:
            split_failures.append(f"oos_{idx}_mean_ret")
        if min_delta < min_oos_min_ret_delta:
            split_failures.append(f"oos_{idx}_min_ret")
        if positive_delta < 0.0:
            split_failures.append(f"oos_{idx}_positive_period_pct")
        if candidate["mean_turnover_rate"] > max_turnover_rate:
            split_failures.append(f"oos_{idx}_turnover")
        if candidate["mean_saturation_rate"] > max_saturation_rate:
            split_failures.append(f"oos_{idx}_saturation")
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
    min_validation_mean_delta: float = 0.0,
    min_validation_min_ret_delta: float = 0.0,
    min_positive_period_pct: float = 0.0,
    max_turnover_rate: float = 0.10,
    max_saturation_rate: float = 0.10,
    max_invalid_open_pressure: float = 0.05,
) -> Dict[str, Any]:
    train_items = {item["path"]: item for item in _summaries(train_report, mode=mode)}
    validation_items = _summaries(validation_report, mode=mode)
    if not validation_items:
        raise ValueError("validation report contains no genomes")

    baseline = validation_items[0]
    best = baseline
    best_score = float("-inf")
    candidates: List[Dict[str, Any]] = []

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
        score = (
            validation_mean_delta * 10.0
            + validation_min_delta * 3.0
            + train_mean_delta
            + (candidate["positive_period_pct"] - baseline["positive_period_pct"]) * 0.02
        )
        item = {
            "path": candidate["path"],
            "accepted": not failures,
            "failures": failures,
            "score": score,
            "train": train_item,
            "validation": candidate,
        }
        candidates.append(item)
        if not failures and score > best_score:
            best = candidate
            best_score = score

    selected_is_baseline = best["path"] == baseline["path"]
    return {
        "mode": mode,
        "selected_path": best["path"],
        "selected_is_baseline": selected_is_baseline,
        "baseline_path": baseline["path"],
        "baseline_validation": baseline,
        "selection_score": None if selected_is_baseline else best_score,
        "thresholds": {
            "min_validation_mean_delta": min_validation_mean_delta,
            "min_validation_min_ret_delta": min_validation_min_ret_delta,
            "min_positive_period_pct": min_positive_period_pct,
            "max_turnover_rate": max_turnover_rate,
            "max_saturation_rate": max_saturation_rate,
            "max_invalid_open_pressure": max_invalid_open_pressure,
        },
        "candidates": candidates,
    }


def main() -> int:
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
        choices=("bearish", "neutral", "bullish"),
        default=None,
        help="Restrict non-baseline regime-router assignments to these regimes. Repeatable.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", default="fee_fixed_nextbar")
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
    args = parser.parse_args()

    train_report_path = Path(args.train_report)
    validation_report_path = Path(args.validation_report)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_report = _load_json(train_report_path)
    validation_report = _load_json(validation_report_path)
    if args.regime_router:
        guard_reports = [_load_json(Path(path)) for path in args.guard_report or []]
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
        )
        if args.final_report:
            final_reports = [_load_json(Path(path)) for path in args.final_report]
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
            min_validation_mean_delta=args.min_validation_mean_delta,
            min_validation_min_ret_delta=args.min_validation_min_ret_delta,
            min_positive_period_pct=args.min_positive_period_pct,
            max_turnover_rate=args.max_turnover_rate,
            max_saturation_rate=args.max_saturation_rate,
            max_invalid_open_pressure=args.max_invalid_open_pressure,
        )
    out_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(selection, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
