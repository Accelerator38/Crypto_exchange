from __future__ import annotations

import argparse
import fnmatch
import json
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]


def _mode_payload(genome_payload: dict[str, Any], mode: str) -> dict[str, Any]:
    for item in genome_payload.get("modes", []):
        if item.get("mode") == mode:
            return item
    raise KeyError(f"mode {mode!r} not found for {genome_payload.get('path')}")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _metrics(genome_payload: dict[str, Any], *, mode: str) -> dict[str, Any]:
    mode_payload = _mode_payload(genome_payload, mode)
    stats = mode_payload.get("period_stats") or {}
    robust = mode_payload.get("robust_score") or {}
    contract = mode_payload.get("contract_metrics") or {}
    period_rets = [float(value) for value in mode_payload.get("period_rets", [])]
    return {
        "path": str(genome_payload.get("path", "")),
        "position_state_features_enabled": bool(
            genome_payload.get("position_state_features_enabled", False)
        ),
        "fitness": _float(mode_payload.get("fitness")),
        "mean_ret": _float(stats.get("mean_ret")),
        "min_ret": _float(stats.get("min_ret")),
        "max_ret": _float(stats.get("max_ret")),
        "positive_period_pct": _float(stats.get("positive_period_pct")),
        "passes_default_gates": bool(robust.get("passes_default_gates", False)),
        "failed_gates": list(robust.get("failed_gates") or []),
        "mean_turnover_rate": _float(contract.get("mean_turnover_rate")),
        "max_turnover_rate": _float(
            contract.get("max_turnover_rate", contract.get("mean_turnover_rate"))
        ),
        "mean_saturation_rate": _float(contract.get("mean_saturation_rate")),
        "max_saturation_rate": _float(
            contract.get("max_saturation_rate", contract.get("mean_saturation_rate"))
        ),
        "mean_invalid_open_logit_pressure": _float(
            contract.get("mean_invalid_open_logit_pressure")
        ),
        "max_invalid_open_logit_pressure": _float(
            contract.get(
                "max_invalid_open_logit_pressure",
                contract.get("mean_invalid_open_logit_pressure"),
            )
        ),
        "period_count": len(period_rets),
        "period_rets": period_rets,
    }


def _promotion_failures(
    candidate: dict[str, Any],
    *,
    min_oos_mean_delta: float,
    min_oos_min_ret_delta: float,
    min_positive_period_pct_delta: float,
    min_oos_periods: int,
    max_turnover_rate: float,
    max_saturation_rate: float,
    max_invalid_open_pressure: float,
    require_robust_pass: bool,
) -> list[str]:
    failures: list[str] = []
    if candidate["period_count"] < min_oos_periods:
        failures.append("oos_period_count")
    if candidate["mean_ret_delta"] <= min_oos_mean_delta:
        failures.append(
            "oos_mean_ret_tie"
            if abs(candidate["mean_ret_delta"] - min_oos_mean_delta) <= 1e-12
            else "oos_mean_ret"
        )
    if candidate["min_ret_delta"] < min_oos_min_ret_delta:
        failures.append("oos_min_ret")
    if candidate["positive_period_pct_delta"] < min_positive_period_pct_delta:
        failures.append("oos_positive_period_pct")
    if candidate["max_turnover_rate"] > max_turnover_rate:
        failures.append("oos_max_turnover")
    if candidate["max_saturation_rate"] > max_saturation_rate:
        failures.append("oos_max_saturation")
    if candidate["max_invalid_open_logit_pressure"] > max_invalid_open_pressure:
        failures.append("oos_invalid_open_pressure")
    if require_robust_pass and not candidate["passes_default_gates"]:
        failures.append("oos_robust_gate")
    return list(dict.fromkeys(failures))


def _candidate_by_path(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(genome.get("path", "")): genome
        for genome in list(report.get("genomes") or [])[1:]
    }


def _candidate_slots(report: dict[str, Any]) -> list[dict[str, Any]]:
    return list(report.get("genomes") or [])[1:]


def _candidate_paths(reports: Iterable[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for report in reports:
        for path in _candidate_by_path(report):
            if path and path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def _parse_strategy_specs(
    raw_specs: Sequence[str] | None,
    *,
    fold_count: int,
) -> list[dict[str, Any]] | None:
    if raw_specs is None:
        return None
    groups: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_specs:
        text = str(raw or "").strip()
        if "=" not in text:
            raise ValueError("strategy specs must use strategy_id=selector0,selector1,...")
        strategy_id, selector_text = text.split("=", 1)
        strategy_id = strategy_id.strip()
        selectors = [item.strip() for item in selector_text.split(",")]
        if not strategy_id:
            raise ValueError("strategy id must be non-empty")
        if strategy_id in seen:
            raise ValueError(f"duplicate strategy id: {strategy_id}")
        if any(not item for item in selectors):
            raise ValueError(f"strategy {strategy_id} contains an empty selector")
        if len(selectors) != fold_count:
            raise ValueError(
                f"strategy {strategy_id} has {len(selectors)} selectors, "
                f"but {fold_count} reports were provided"
            )
        seen.add(strategy_id)
        groups.append(
            {
                "strategy_id": strategy_id,
                "summary_path": strategy_id,
                "mode": "strategy_spec",
                "selectors": selectors,
                "selection_contract": {
                    "type": "fold_selectors",
                    "selectors": selectors,
                },
            }
        )
    return groups


def _candidate_matches_selector(candidate: dict[str, Any], selector: str) -> bool:
    path_text = str(candidate.get("path", ""))
    path = Path(path_text)
    choices = [
        path_text,
        path.name,
        path.stem,
    ]
    return any(fnmatch.fnmatchcase(choice, selector) for choice in choices)


def _candidate_by_selector(report: dict[str, Any], selector: str) -> dict[str, Any] | None:
    matches = [
        candidate
        for candidate in _candidate_slots(report)
        if _candidate_matches_selector(candidate, selector)
    ]
    if len(matches) > 1:
        paths = ", ".join(str(item.get("path", "")) for item in matches)
        raise ValueError(f"selector {selector!r} matched multiple candidates: {paths}")
    return matches[0] if matches else None


def _candidate_groups(
    reports: Sequence[dict[str, Any]],
    candidate_labels: Sequence[str] | None,
    strategy_specs: Sequence[str] | None,
) -> list[dict[str, Any]]:
    parsed_specs = _parse_strategy_specs(strategy_specs, fold_count=len(reports))
    if parsed_specs is not None:
        if candidate_labels is not None:
            raise ValueError("candidate labels and strategy specs are mutually exclusive")
        return parsed_specs

    if candidate_labels is None:
        return [
            {
                "strategy_id": path,
                "summary_path": path,
                "mode": "path",
                "path": path,
            }
            for path in _candidate_paths(reports)
        ]

    labels = [str(label).strip() for label in candidate_labels]
    if any(not label for label in labels):
        raise ValueError("candidate labels must be non-empty")
    if len(set(labels)) != len(labels):
        raise ValueError("candidate labels must be unique")
    for fold_index, report in enumerate(reports):
        candidate_count = len(_candidate_slots(report))
        if candidate_count != len(labels):
            raise ValueError(
                f"fold {fold_index} has {candidate_count} candidates, "
                f"but {len(labels)} candidate labels were provided"
            )
    return [
        {
            "strategy_id": label,
            "summary_path": label,
            "mode": "label",
            "candidate_index": index,
        }
        for index, label in enumerate(labels)
    ]


def _fold_candidate_summary(
    report: dict[str, Any],
    candidate_genome: dict[str, Any],
    *,
    fold_index: int,
    mode: str,
    min_oos_mean_delta: float,
    min_oos_min_ret_delta: float,
    min_positive_period_pct_delta: float,
    min_oos_periods: int,
    max_turnover_rate: float,
    max_saturation_rate: float,
    max_invalid_open_pressure: float,
    require_robust_pass: bool,
) -> dict[str, Any]:
    genomes = list(report.get("genomes") or [])
    if len(genomes) < 2:
        raise ValueError("each OOS report must contain baseline and candidate genomes")
    baseline = _metrics(genomes[0], mode=mode)
    candidate = _metrics(candidate_genome, mode=mode)
    candidate.update(
        {
            "fold_index": int(fold_index),
            "oos_start_date": report.get("start_date"),
            "oos_end_date": report.get("end_date"),
            "baseline_path": baseline["path"],
            "mean_ret_delta": candidate["mean_ret"] - baseline["mean_ret"],
            "min_ret_delta": candidate["min_ret"] - baseline["min_ret"],
            "positive_period_pct_delta": candidate["positive_period_pct"]
            - baseline["positive_period_pct"],
        }
    )
    failures = _promotion_failures(
        candidate,
        min_oos_mean_delta=min_oos_mean_delta,
        min_oos_min_ret_delta=min_oos_min_ret_delta,
        min_positive_period_pct_delta=min_positive_period_pct_delta,
        min_oos_periods=min_oos_periods,
        max_turnover_rate=max_turnover_rate,
        max_saturation_rate=max_saturation_rate,
        max_invalid_open_pressure=max_invalid_open_pressure,
        require_robust_pass=require_robust_pass,
    )
    candidate["promotion_failures"] = failures
    candidate["promotion_eligible"] = not failures
    return candidate


def summarize_walk_forward_reports(
    reports: Sequence[dict[str, Any]],
    *,
    candidate_labels: Sequence[str] | None = None,
    strategy_specs: Sequence[str] | None = None,
    mode: str = "fee_fixed_nextbar",
    min_oos_mean_delta: float = 0.0,
    min_oos_min_ret_delta: float = 0.0,
    min_positive_period_pct_delta: float = 0.0,
    min_oos_periods: int = 2,
    max_turnover_rate: float = 0.10,
    max_saturation_rate: float = 0.10,
    max_invalid_open_pressure: float = 0.05,
    require_robust_pass: bool = False,
) -> dict[str, Any]:
    """Summarize multiple OOS reports and require a candidate to pass every fold."""

    report_list = list(reports)
    if not report_list:
        raise ValueError("at least one OOS report is required")
    baselines = []
    for report in report_list:
        genomes = list(report.get("genomes") or [])
        if len(genomes) < 2:
            raise ValueError("each OOS report must contain baseline and candidate genomes")
        baseline = _metrics(genomes[0], mode=mode)
        baseline.update(
            {
                "oos_start_date": report.get("start_date"),
                "oos_end_date": report.get("end_date"),
            }
        )
        baselines.append(baseline)

    candidates: list[dict[str, Any]] = []
    for group in _candidate_groups(report_list, candidate_labels, strategy_specs):
        folds: list[dict[str, Any]] = []
        missing_fold_count = 0
        for fold_index, report in enumerate(report_list):
            if group["mode"] == "label":
                candidate_genome = _candidate_slots(report)[int(group["candidate_index"])]
            elif group["mode"] == "strategy_spec":
                selector = str(group["selectors"][fold_index])
                candidate_genome = _candidate_by_selector(report, selector)
            else:
                candidate_genome = _candidate_by_path(report).get(str(group["path"]))
            if candidate_genome is None:
                missing_fold_count += 1
                folds.append(
                    {
                        "fold_index": int(fold_index),
                        "oos_start_date": report.get("start_date"),
                        "oos_end_date": report.get("end_date"),
                        "path": str(group["summary_path"]),
                        "strategy_id": str(group["strategy_id"]),
                        "promotion_eligible": False,
                        "promotion_failures": ["candidate_missing_from_fold"],
                    }
                )
                continue
            folds.append(
                _fold_candidate_summary(
                    report,
                    candidate_genome,
                    fold_index=fold_index,
                    mode=mode,
                    min_oos_mean_delta=min_oos_mean_delta,
                    min_oos_min_ret_delta=min_oos_min_ret_delta,
                    min_positive_period_pct_delta=min_positive_period_pct_delta,
                    min_oos_periods=min_oos_periods,
                    max_turnover_rate=max_turnover_rate,
                    max_saturation_rate=max_saturation_rate,
                    max_invalid_open_pressure=max_invalid_open_pressure,
                    require_robust_pass=require_robust_pass,
                )
            )

        valid_folds = [fold for fold in folds if "mean_ret_delta" in fold]
        all_failures = [
            failure
            for fold in folds
            for failure in list(fold.get("promotion_failures") or [])
        ]
        promotion_eligible = bool(folds) and all(
            bool(fold.get("promotion_eligible")) for fold in folds
        )
        selected_fold_genome = ""
        for fold in reversed(folds):
            if fold.get("path") and "mean_ret_delta" in fold:
                selected_fold_genome = str(fold["path"])
                break
        candidates.append(
            {
                "path": str(group["summary_path"]),
                "strategy_id": str(group["strategy_id"]),
                "selection_contract": group.get(
                    "selection_contract",
                    {"type": str(group["mode"])},
                ),
                "selected_fold_genome": selected_fold_genome,
                "promotion_eligible": promotion_eligible,
                "promotion_failures": list(dict.fromkeys(all_failures)),
                "folds": folds,
                "fold_count": len(folds),
                "missing_fold_count": int(missing_fold_count),
                "mean_ret_delta_avg": (
                    sum(float(fold["mean_ret_delta"]) for fold in valid_folds)
                    / len(valid_folds)
                    if valid_folds
                    else float("-inf")
                ),
                "min_ret_delta_min": (
                    min(float(fold["min_ret_delta"]) for fold in valid_folds)
                    if valid_folds
                    else float("-inf")
                ),
                "positive_period_pct_delta_min": (
                    min(float(fold["positive_period_pct_delta"]) for fold in valid_folds)
                    if valid_folds
                    else float("-inf")
                ),
                "max_turnover_rate_max": (
                    max(float(fold["max_turnover_rate"]) for fold in valid_folds)
                    if valid_folds
                    else float("inf")
                ),
                "max_saturation_rate_max": (
                    max(float(fold["max_saturation_rate"]) for fold in valid_folds)
                    if valid_folds
                    else float("inf")
                ),
                "max_invalid_open_logit_pressure_max": (
                    max(
                        float(fold["max_invalid_open_logit_pressure"])
                        for fold in valid_folds
                    )
                    if valid_folds
                    else float("inf")
                ),
            }
        )

    eligible = [candidate for candidate in candidates if candidate["promotion_eligible"]]
    selected = (
        max(
            eligible,
            key=lambda item: (
                item["mean_ret_delta_avg"],
                item["min_ret_delta_min"],
                item["positive_period_pct_delta_min"],
            ),
        )
        if eligible
        else None
    )
    selected_is_baseline = selected is None
    baseline_path = baselines[0]["path"]
    return {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "exchange": str(report_list[0].get("exchange") or "").upper(),
        "mode": mode,
        "fold_count": len(report_list),
        "candidate_count": len(candidates),
        "selected_is_baseline": selected_is_baseline,
        "promotion_eligible": not selected_is_baseline,
        "promotion_failures": []
        if selected is not None
        else ["no_candidate_passed_all_folds"],
        "selected_genome": selected["selected_fold_genome"] if selected is not None else baseline_path,
        "selected_strategy_id": selected["strategy_id"] if selected is not None else "baseline",
        "baseline": {
            "path": baseline_path,
            "folds": baselines,
        },
        "candidates": candidates,
        "thresholds": {
            "min_oos_mean_delta": float(min_oos_mean_delta),
            "min_oos_min_ret_delta": float(min_oos_min_ret_delta),
            "min_positive_period_pct_delta": float(min_positive_period_pct_delta),
            "min_oos_periods": int(min_oos_periods),
            "max_turnover_rate": float(max_turnover_rate),
            "max_saturation_rate": float(max_saturation_rate),
            "max_invalid_open_pressure": float(max_invalid_open_pressure),
            "require_robust_pass": bool(require_robust_pass),
        },
    }


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_report(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve_path(path).read_text(encoding="utf-8-sig"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize multiple genetics OOS reports into a walk-forward promotion gate."
    )
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--candidate-label",
        action="append",
        default=None,
        help=(
            "Optional strategy label for candidates after the baseline, in report order. "
            "Repeat once per candidate to group different fold genomes by strategy."
        ),
    )
    parser.add_argument(
        "--strategy-spec",
        action="append",
        default=None,
        help=(
            "Explicit per-fold strategy contract in the form "
            "strategy_id=selector0,selector1,... . Each selector is matched against "
            "candidate path, filename, or stem in the corresponding report."
        ),
    )
    parser.add_argument("--mode", default="fee_fixed_nextbar")
    parser.add_argument("--min-oos-mean-delta", type=float, default=0.0)
    parser.add_argument("--min-oos-min-ret-delta", type=float, default=0.0)
    parser.add_argument("--min-positive-period-pct-delta", type=float, default=0.0)
    parser.add_argument("--min-oos-periods", type=int, default=2)
    parser.add_argument("--max-turnover-rate", type=float, default=0.10)
    parser.add_argument("--max-saturation-rate", type=float, default=0.10)
    parser.add_argument("--max-invalid-open-pressure", type=float, default=0.05)
    parser.add_argument("--require-robust-pass", action="store_true")
    args = parser.parse_args(argv)

    reports = [_load_report(path) for path in args.report]
    summary = summarize_walk_forward_reports(
        reports,
        candidate_labels=args.candidate_label,
        strategy_specs=args.strategy_spec,
        mode=args.mode,
        min_oos_mean_delta=args.min_oos_mean_delta,
        min_oos_min_ret_delta=args.min_oos_min_ret_delta,
        min_positive_period_pct_delta=args.min_positive_period_pct_delta,
        min_oos_periods=args.min_oos_periods,
        max_turnover_rate=args.max_turnover_rate,
        max_saturation_rate=args.max_saturation_rate,
        max_invalid_open_pressure=args.max_invalid_open_pressure,
        require_robust_pass=args.require_robust_pass,
    )
    summary["source_reports"] = [str(_resolve_path(path)) for path in args.report]
    out_path = _resolve_path(args.out)
    _write_json(out_path, summary)
    print(f"walk_forward_summary={out_path}")
    print(f"promotion_eligible={str(summary['promotion_eligible']).lower()}")
    print(f"selected_genome={summary['selected_genome']}")
    if summary["promotion_failures"]:
        print("promotion_failures=" + ",".join(summary["promotion_failures"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
