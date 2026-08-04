from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FAMILY = (
    ROOT / "freqtrade_pilot" / "candidates" / "exia_trend_family_v1.json"
)
DEFAULT_EXPERIMENTS = ROOT / "Reports" / "Exia" / "experiments"
DEFAULT_OUTPUT = ROOT / "Reports" / "Exia" / "trend_family_v1"
DEFAULT_LOOKAHEAD = DEFAULT_OUTPUT / "lookahead_analysis.csv"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _latest_development_experiment(
    experiments_root: Path,
    candidate_id: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    matches: list[tuple[str, Path, dict[str, Any], dict[str, Any]]] = []
    for metrics_path in experiments_root.glob(f"{candidate_id}_*/metrics.json"):
        metrics = _load_json(metrics_path)
        manifest_path = metrics_path.with_name("manifest.json")
        if not manifest_path.is_file():
            continue
        manifest = _load_json(manifest_path)
        if metrics.get("candidate_id") != candidate_id:
            continue
        if metrics.get("status") != "DEVELOPMENT_SCORED":
            continue
        if set(manifest.get("timeranges", {})) != {"development"}:
            continue
        if any(
            payload.get(flag) is not False
            for payload in (metrics, manifest)
            for flag in ("orders_enabled", "promotion_authority")
        ):
            raise ValueError(f"unsafe experiment flags: {metrics_path.parent.name}")
        matches.append(
            (
                str(metrics.get("generated_at", "")),
                metrics_path.parent,
                manifest,
                metrics,
            )
        )
    if not matches:
        raise FileNotFoundError(
            f"no development-only experiment for {candidate_id}"
        )
    _, experiment_dir, manifest, metrics = max(matches, key=lambda item: item[0])
    return experiment_dir, manifest, metrics


def _row(
    metrics: dict[str, Any],
    *,
    scope: str,
    scope_value: str,
) -> dict[str, Any]:
    matches = [
        row
        for row in metrics["catalog_rows"]
        if row["split"] == "development"
        and row["cost_scenario"] == "stress"
        and row["scope"] == scope
        and row["scope_value"] == scope_value
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one stress development row for {scope}:{scope_value}"
        )
    return matches[0]


def _gate_checks(row: dict[str, Any], gate: dict[str, Any]) -> dict[str, bool]:
    mean = row.get("mean_net_bps")
    lcb = row.get("block_lcb_net_bps")
    baseline_lcb = row.get("baseline_differential_lcb_bps")
    drawdown = row.get("max_drawdown")
    return {
        "minimum_closed_trades": int(row.get("closed_trades", 0))
        >= int(gate["min_closed_trades"]),
        "positive_mean_after_stress_costs": mean is not None
        and float(mean) > float(gate["min_mean_net_bps"]),
        "positive_block_lcb": lcb is not None
        and float(lcb) > float(gate["min_block_lcb_net_bps"]),
        "positive_baseline_differential_lcb": baseline_lcb is not None
        and float(baseline_lcb) > 0.0,
        "drawdown_within_limit": drawdown is not None
        and float(drawdown) <= float(gate["max_drawdown"]),
        "catalog_scope_pass": row.get("status") == "PASS",
    }


def _lookahead_results(path: Path) -> tuple[dict[str, dict[str, Any]], str | None]:
    if not path.is_file():
        return {}, None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    results = {}
    for row in rows:
        strategy = str(row.get("strategy", ""))
        if not strategy or strategy in results:
            raise ValueError("lookahead results contain missing or duplicate strategies")
        results[strategy] = {
            "strategy": strategy,
            "has_bias": str(row.get("has_bias", "")).lower() == "true",
            "total_signals": int(row.get("total_signals", 0)),
            "biased_entry_signals": int(row.get("biased_entry_signals", 0)),
            "biased_exit_signals": int(row.get("biased_exit_signals", 0)),
            "biased_indicators": str(row.get("biased_indicators", "")),
        }
    return results, _sha256(path)


def _taxonomy_runtime_status(
    candidate_specs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    taxonomy_shas = {
        str(spec["taxonomy"]["source_sha256"])
        for spec in candidate_specs.values()
    }
    if len(taxonomy_shas) != 1:
        raise ValueError("candidate family must use one taxonomy SHA")
    taxonomy_sha = next(iter(taxonomy_shas))
    matches = []
    for path in (ROOT / "Reports" / "Exia").glob(
        "market_mode_foundation_*/distribution.json"
    ):
        report = _load_json(path)
        if report.get("source", {}).get("taxonomy_sha256") == taxonomy_sha:
            matches.append((str(report.get("generated_at", "")), path, report))
    if not matches:
        return {
            "status": "MISSING_FOUNDATION_REPORT",
            "passed": False,
            "taxonomy_sha256": taxonomy_sha,
            "foundation_report": None,
            "foundation_report_sha256": None,
        }
    _, path, report = max(matches, key=lambda item: item[0])
    checks = report.get("checks", {})
    passed = (
        report.get("status") == "FOUNDATION_READY_FOR_CANDIDATE_DESIGN"
        and checks.get("restart_parity") is True
        and checks.get("runtime_startup_within_bitget_limit") is True
    )
    return {
        "status": "PASS_RESTART_STABLE"
        if passed
        else "REJECTED_RESTART_DEPENDENT",
        "passed": passed,
        "taxonomy_sha256": taxonomy_sha,
        "foundation_report": str(path.relative_to(ROOT)).replace("\\", "/"),
        "foundation_report_sha256": _sha256(path),
    }


def evaluate(
    *,
    family_path: Path,
    experiments_root: Path,
    lookahead_path: Path | None = None,
) -> dict[str, Any]:
    family = _load_json(family_path)
    safety_flags = (
        "paper_allowed",
        "live_allowed",
        "orders_enabled",
        "promotion_authority",
    )
    if any(family.get(flag) is not False for flag in safety_flags):
        raise ValueError("candidate family safety flags must all be false")
    if family.get("selection_window") != "development":
        raise ValueError("this evaluator accepts development selection only")
    if family.get("selection_cost") != "stress":
        raise ValueError("this evaluator requires stress-cost selection")

    candidate_ids = list(family["candidate_ids"])
    if len(candidate_ids) != int(family["maximum_variants"]):
        raise ValueError("family variant count differs from preregistered maximum")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate_ids must be unique")

    candidate_specs = {
        candidate_id: _load_json(family_path.parent / f"{candidate_id}.json")
        for candidate_id in candidate_ids
    }
    taxonomy_runtime = _taxonomy_runtime_status(candidate_specs)

    lookahead, lookahead_sha = _lookahead_results(
        lookahead_path or DEFAULT_LOOKAHEAD
    )

    candidates = []
    exploratory_scopes = []
    for candidate_id in candidate_ids:
        candidate_spec = candidate_specs[candidate_id]
        strategy_class = str(candidate_spec["strategy"]["class_name"])
        lookahead_row = lookahead.get(strategy_class)
        experiment_dir, manifest, metrics = _latest_development_experiment(
            experiments_root,
            candidate_id,
        )
        overall = _row(metrics, scope="overall", scope_value="all")
        checks = _gate_checks(overall, family["selection_gate"])
        passes_alpha_gate = all(checks.values())
        checks["lookahead_no_bias"] = (
            lookahead_row is not None
            and lookahead_row["has_bias"] is False
            and lookahead_row["total_signals"] >= 20
            and lookahead_row["biased_entry_signals"] == 0
            and lookahead_row["biased_exit_signals"] == 0
            and not lookahead_row["biased_indicators"]
        )
        checks["taxonomy_restart_stable"] = bool(taxonomy_runtime["passed"])
        candidates.append(
            {
                "candidate_id": candidate_id,
                "experiment_id": metrics["experiment_id"],
                "manifest_sha256": _sha256(experiment_dir / "manifest.json"),
                "metrics_sha256": _sha256(experiment_dir / "metrics.json"),
                "strategy_sha256": manifest["strategy_sha256"],
                "strategy_class": strategy_class,
                "lookahead": lookahead_row,
                "lookahead_status": "PASS"
                if checks["lookahead_no_bias"]
                else (
                    "NOT_RUN_ALPHA_FAILED"
                    if lookahead_row is None and not passes_alpha_gate
                    else "FAIL_OR_MISSING"
                ),
                "closed_trades": overall["closed_trades"],
                "fills": overall["fills"],
                "mean_net_bps": overall["mean_net_bps"],
                "block_lcb_net_bps": overall["block_lcb_net_bps"],
                "baseline_differential_lcb_bps": overall[
                    "baseline_differential_lcb_bps"
                ],
                "max_drawdown": overall["max_drawdown"],
                "gate_checks": checks,
                "passes_alpha_gate": passes_alpha_gate,
                "passes_selection_gate": all(checks.values()),
            }
        )
        for row in metrics["catalog_rows"]:
            if (
                row["split"] == "development"
                and row["cost_scenario"] == "stress"
                and row["scope"] != "overall"
                and row["status"] == "PASS"
            ):
                exploratory_scopes.append(
                    {
                        "candidate_id": candidate_id,
                        "scope": row["scope"],
                        "scope_value": row["scope_value"],
                        "closed_trades": row["closed_trades"],
                        "fills": row["fills"],
                        "mean_net_bps": row["mean_net_bps"],
                        "block_lcb_net_bps": row["block_lcb_net_bps"],
                        "baseline_differential_lcb_bps": row[
                            "baseline_differential_lcb_bps"
                        ],
                        "selection_authority": False,
                    }
                )

    passed = [item for item in candidates if item["passes_selection_gate"]]
    passed.sort(
        key=lambda item: (
            float(item["block_lcb_net_bps"]),
            float(item["mean_net_bps"]),
            item["candidate_id"],
        ),
        reverse=True,
    )
    selected = passed[: int(family["maximum_candidates_opened_on_validation"])]
    status = (
        "SELECTED_FOR_VALIDATION"
        if selected
        else "NO_CANDIDATE_FOR_VALIDATION"
    )
    blockers = []
    if not any(item["passes_alpha_gate"] for item in candidates):
        blockers.append("NO_CANDIDATE_PASSED_ALPHA_GATE")
    elif not selected:
        blockers.append("NO_CANDIDATE_PASSED_ALL_DEVELOPMENT_GATES")
    if not taxonomy_runtime["passed"]:
        blockers.append(str(taxonomy_runtime["status"]))
    return {
        "schema_version": "exia.family_development_decision.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "family_id": family["family_id"],
        "family_sha256": _sha256(family_path),
        "status": status,
        "taxonomy_runtime": taxonomy_runtime,
        "validation_blockers": blockers,
        "selection_window": "development",
        "selection_cost": "stress",
        "selection_gate": family["selection_gate"],
        "lookahead_results_sha256": lookahead_sha,
        "candidates": candidates,
        "selected_candidate_ids": [item["candidate_id"] for item in selected],
        "exploratory_positive_scopes": exploratory_scopes,
        "exploratory_scope_warning": (
            "Subscope PASS rows were observed after evaluating the full family and "
            "cannot authorize validation without a new preregistered hypothesis."
            if exploratory_scopes
            else "No positive stress-cost subscopes were observed."
        ),
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['family_id']} - development decision",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "This is a development-only research decision. It cannot authorize paper or live trading.",
        "",
        f"Status: `{report['status']}`.",
        "",
        f"Taxonomy runtime status: `{report['taxonomy_runtime']['status']}`.",
        "",
        "## Full8 stress-cost results",
        "",
        "| Candidate | Trades | Fills | Mean net bps | Block LCB bps | Baseline diff LCB bps | Drawdown | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["candidates"]:
        lines.append(
            "| {candidate_id} | {closed_trades} | {fills} | {mean} | {lcb} | "
            "{baseline} | {drawdown} | {gate} |".format(
                candidate_id=item["candidate_id"],
                closed_trades=item["closed_trades"],
                fills=item["fills"],
                mean=_fmt(item["mean_net_bps"]),
                lcb=_fmt(item["block_lcb_net_bps"]),
                baseline=_fmt(item["baseline_differential_lcb_bps"]),
                drawdown=_fmt(item["max_drawdown"], 6),
                gate="PASS" if item["passes_selection_gate"] else "FAIL",
            )
        )

    lines.extend(
        [
            "",
        "## Exploratory positive subscopes",
            "",
            "These rows are diagnostic only because they were observed after comparing the full family.",
            "",
            "| Candidate | Scope | Value | Trades | Mean net bps | Block LCB bps | Baseline diff LCB bps |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    if report["exploratory_positive_scopes"]:
        for item in report["exploratory_positive_scopes"]:
            lines.append(
                "| {candidate_id} | {scope} | {scope_value} | {closed_trades} | "
                "{mean} | {lcb} | {baseline} |".format(
                    **item,
                    mean=_fmt(item["mean_net_bps"]),
                    lcb=_fmt(item["block_lcb_net_bps"]),
                    baseline=_fmt(item["baseline_differential_lcb_bps"]),
                )
            )
    else:
        lines.append("| none | - | - | 0 | n/a | n/a | n/a |")

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "No validation, OOS, paper, or live stage is opened unless a candidate passes every preregistered development gate.",
            "",
            f"Selected for validation: `{', '.join(report['selected_candidate_ids']) or 'none'}`.",
            "",
            "All order and promotion authority flags remain false.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate one preregistered Exia family on development only."
    )
    parser.add_argument("--family", type=Path, default=DEFAULT_FAMILY)
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--lookahead-results",
        type=Path,
        default=DEFAULT_LOOKAHEAD,
    )
    args = parser.parse_args()

    report = evaluate(
        family_path=args.family.resolve(),
        experiments_root=args.experiments_root.resolve(),
        lookahead_path=args.lookahead_results.resolve(),
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "development_report.json"
    markdown_path = output_dir / "development_report.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] in {
        "SELECTED_FOR_VALIDATION",
        "NO_CANDIDATE_FOR_VALIDATION",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
