"""Evaluate one fixed CarryFlow exit hypothesis without changing runtime policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASELINE_HOLD_BARS = 6
CANDIDATE_HOLD_BARS = 4


def evaluate_exit_hypothesis(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    selection_root: str,
) -> dict[str, Any]:
    if selection_root not in reports:
        raise ValueError(f"selection root is missing: {selection_root}")
    root_rows = {
        label: _root_comparison(report)
        for label, report in reports.items()
    }
    evaluation_roots = [
        label for label in reports if label != selection_root
    ]
    if not evaluation_roots:
        raise ValueError("at least one independent evaluation root is required")

    evaluation_candidate_values = [
        value
        for label in evaluation_roots
        for value in root_rows[label]["candidate_net_bps"]
    ]
    evaluation_baseline_values = [
        value
        for label in evaluation_roots
        for value in root_rows[label]["baseline_net_bps"]
    ]
    root_collapses = [
        label
        for label in evaluation_roots
        if root_rows[label]["candidate_mean_net_bps"] is None
        or root_rows[label]["candidate_mean_net_bps"] <= 0.0
    ]
    candidate_mean = _mean(evaluation_candidate_values)
    baseline_mean = _mean(evaluation_baseline_values)
    beats_baseline = bool(
        candidate_mean is not None
        and baseline_mean is not None
        and candidate_mean > baseline_mean
    )
    passed = not root_collapses and beats_baseline
    feature_screen = _feature_screen(reports)

    return {
        "schema_version": "panteon.carryflow_exit_hypothesis.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "promotion_authority": False,
        "orders_enabled": False,
        "strategy_changed": False,
        "multiple_hypotheses_tested": False,
        "hypothesis": {
            "id": "fixed_hold_4_vs_fixed_hold_6",
            "candidate_hold_bars": CANDIDATE_HOLD_BARS,
            "baseline_hold_bars": BASELINE_HOLD_BARS,
            "entry_set": "same_policy_allowed_entries_with_both_endpoints",
            "cost_basis": "net_bps_from_sealed_tape_close_and_modeled_costs",
        },
        "selection_root": selection_root,
        "independent_evaluation_roots": evaluation_roots,
        "roots": root_rows,
        "independent_evaluation": {
            "observations": len(evaluation_candidate_values),
            "candidate_mean_net_bps": candidate_mean,
            "baseline_mean_net_bps": baseline_mean,
            "candidate_beats_baseline": beats_baseline,
            "candidate_root_collapses": root_collapses,
        },
        "expected_move_feature_screen": feature_screen,
        "passed": passed,
        "verdict": "accepted_for_new_prospective_root" if passed else "rejected",
        "profile_registration_allowed": passed,
        "expected_move_model_registration_allowed": feature_screen[
            "registration_allowed"
        ],
        "reason": (
            "candidate_positive_on_every_independent_root_and_beats_baseline"
            if passed
            else "independent_root_inconsistency_or_no_baseline_improvement"
        ),
    }


def _root_comparison(report: Mapping[str, Any]) -> dict[str, Any]:
    details = (
        report.get("signal_calibration", {}).get("candidate_details", [])
    )
    candidate_values: list[float] = []
    baseline_values: list[float] = []
    included = []
    for row in details:
        if str(row.get("policy_outcome")) != "ALLOW_OPEN":
            continue
        horizons = row.get("horizons") or {}
        candidate = horizons.get(str(CANDIDATE_HOLD_BARS))
        baseline = horizons.get(str(BASELINE_HOLD_BARS))
        if not isinstance(candidate, Mapping) or not isinstance(
            baseline, Mapping
        ):
            continue
        candidate_net = float(candidate["net_bps"])
        baseline_net = float(baseline["net_bps"])
        candidate_values.append(candidate_net)
        baseline_values.append(baseline_net)
        included.append(
            {
                "symbol": str(row["symbol"]),
                "bar": int(row["bar"]),
                "candidate_net_bps": candidate_net,
                "baseline_net_bps": baseline_net,
            }
        )
    return {
        "observations": len(included),
        "candidate_mean_net_bps": _mean(candidate_values),
        "baseline_mean_net_bps": _mean(baseline_values),
        "candidate_net_bps": candidate_values,
        "baseline_net_bps": baseline_values,
        "included_entries": included,
    }


def _feature_screen(
    reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    feature_names = (
        "ranking_score",
        "price_dislocation_bps",
        "oi_excess_bps",
    )
    rows_by_root = {
        label: _calibration_rows(report)
        for label, report in reports.items()
    }
    features = {}
    for feature_name in feature_names:
        per_root = {
            label: {
                "observations": len(rows),
                "pearson_vs_gross_6h": _pearson(
                    [row[feature_name] for row in rows],
                    [row["gross_6h_bps"] for row in rows],
                ),
            }
            for label, rows in rows_by_root.items()
        }
        pooled = [
            row for rows in rows_by_root.values() for row in rows
        ]
        correlations = [
            row["pearson_vs_gross_6h"]
            for row in per_root.values()
        ]
        features[feature_name] = {
            "observations": len(pooled),
            "pooled_pearson_vs_gross_6h": _pearson(
                [row[feature_name] for row in pooled],
                [row["gross_6h_bps"] for row in pooled],
            ),
            "per_root": per_root,
            "sign_stable_positive": bool(
                correlations
                and all(value is not None and value > 0.0 for value in correlations)
            ),
        }
    sufficient_observations = sum(map(len, rows_by_root.values())) >= 20
    stable_features = [
        name
        for name, row in features.items()
        if row["sign_stable_positive"]
        and row["pooled_pearson_vs_gross_6h"] is not None
        and row["pooled_pearson_vs_gross_6h"] > 0.0
    ]
    return {
        "target": "gross_6h_bps",
        "observations": sum(map(len, rows_by_root.values())),
        "minimum_observations": 20,
        "sufficient_observations": sufficient_observations,
        "features": features,
        "stable_positive_features": stable_features,
        "registration_allowed": bool(
            sufficient_observations and stable_features
        ),
        "interpretation": (
            "No feature may be registered as expected move without sufficient "
            "observations and a positive correlation sign on every root."
        ),
    }


def _calibration_rows(report: Mapping[str, Any]) -> list[dict[str, float]]:
    rows = []
    details = (
        report.get("signal_calibration", {}).get("candidate_details", [])
    )
    for row in details:
        if str(row.get("policy_outcome")) != "ALLOW_OPEN":
            continue
        horizon = (row.get("horizons") or {}).get(str(BASELINE_HOLD_BARS))
        components = row.get("edge_components") or {}
        if not isinstance(horizon, Mapping):
            continue
        try:
            rows.append(
                {
                    "ranking_score": float(row["feature_value"]),
                    "price_dislocation_bps": (
                        float(components["price_divergence"]) * 100.0
                    ),
                    "oi_excess_bps": float(components["oi_excess"]) * 1000.0,
                    "gross_6h_bps": float(horizon["gross_bps"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def render_markdown(report: Mapping[str, Any]) -> str:
    evaluation = report["independent_evaluation"]
    lines = [
        "# CarryFlow fixed-exit hypothesis",
        "",
        "## Verdict",
        "",
        f"**{str(report['verdict']).upper()}**",
        "",
        (
            f"`hold=4` was selected on `{report['selection_root']}` and evaluated "
            "without retuning on the remaining independent roots."
        ),
        "",
        "| Root | Role | Observations | Hold 4 net, bps | Hold 6 net, bps |",
        "|---|---|---:|---:|---:|",
    ]
    for label, row in report["roots"].items():
        role = "selection" if label == report["selection_root"] else "evaluation"
        lines.append(
            f"| {label} | {role} | {row['observations']} | "
            f"{_fmt(row['candidate_mean_net_bps'])} | "
            f"{_fmt(row['baseline_mean_net_bps'])} |"
        )
    lines.extend(
        [
            "",
            "## Independent evaluation",
            "",
            f"- Observations: {evaluation['observations']}",
            (
                "- Candidate mean net: "
                f"{_fmt(evaluation['candidate_mean_net_bps'])} bps"
            ),
            (
                "- Baseline mean net: "
                f"{_fmt(evaluation['baseline_mean_net_bps'])} bps"
            ),
            (
                "- Root collapses: "
                + (
                    ", ".join(evaluation["candidate_root_collapses"])
                    if evaluation["candidate_root_collapses"]
                    else "none"
                )
            ),
            "",
            "## Expected-move feature screen",
            "",
            (
                f"- Observations: "
                f"{report['expected_move_feature_screen']['observations']} "
                f"(minimum "
                f"{report['expected_move_feature_screen']['minimum_observations']})"
            ),
            (
                "- Stable positive features: "
                + (
                    ", ".join(
                        report["expected_move_feature_screen"][
                            "stable_positive_features"
                        ]
                    )
                    if report["expected_move_feature_screen"][
                        "stable_positive_features"
                    ]
                    else "none"
                )
            ),
            "",
            "| Feature | Pooled correlation | Sign positive on every root |",
            "|---|---:|---:|",
        ]
    )
    for name, row in report["expected_move_feature_screen"]["features"].items():
        lines.append(
            f"| {name} | {_fmt(row['pooled_pearson_vs_gross_6h'])} | "
            f"{str(row['sign_stable_positive']).lower()} |"
        )
    lines.extend(
        [
            "",
            (
                "`hold=4` is not registered as a policy profile when this "
                "report is rejected. The existing six-bar profile is also not "
                "promoted by this comparison."
            ),
            "",
            "This report is R&D-only and cannot authorize paper or live orders.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2f}"


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left, right)
    )
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else None


def _parse_root(value: str) -> tuple[str, Path]:
    label, separator, raw_path = str(value).partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise ValueError("--root must use LABEL=PATH")
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return label.strip(), path


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the single fixed hold=4 CarryFlow exit hypothesis "
            "against hold=6 on independent signal-quality reports."
        )
    )
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--selection-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)

    parsed = [_parse_root(value) for value in args.root]
    if len({label for label, _ in parsed}) != len(parsed):
        raise ValueError("root labels must be unique")
    report = evaluate_exit_hypothesis(
        {label: _read_json(path) for label, path in parsed},
        selection_root=str(args.selection_root),
    )
    report["inputs"] = [
        {
            "root": label,
            "path": str(path),
            "sha256": _sha256_file(path),
        }
        for label, path in parsed
    ]
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "passed": report["passed"],
                "research_only": report["research_only"],
                "promotion_authority": report["promotion_authority"],
                "orders_enabled": report["orders_enabled"],
                "independent_evaluation": report["independent_evaluation"],
                "output_json": str(output_json),
                "output_md": str(output_md),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
