from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from panteon_v2.analysis.genetic_core_v2_experiments import (  # noqa: E402
    candidate_metrics_from_mode,
    score_specialist_router,
    summarize_singleton_candidate,
)


DEFAULT_REPORTS = {
    "train": "contract_train_2022_2023.json",
    "validation": "contract_validation_2024.json",
    "oos_2025": "contract_oos_2025.json",
    "sanity_2026_h1": "contract_final_sanity_2026_h1.json",
}
ROUTER_SELECTION_REPORT = "selection_router_fitness_v4.json"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _mode_payload(genome_payload: Mapping[str, Any], mode: str) -> dict[str, Any]:
    modes = genome_payload.get("modes", [])
    if not isinstance(modes, list):
        return {}
    for item in modes:
        if isinstance(item, dict) and str(item.get("mode")) == mode:
            return dict(item)
    return dict(modes[0]) if modes and isinstance(modes[0], dict) else {}


def _genome_metrics(report: Mapping[str, Any], *, mode: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    genomes = report.get("genomes", [])
    if not isinstance(genomes, list):
        return out
    for genome in genomes:
        if not isinstance(genome, dict):
            continue
        path = str(genome.get("path", ""))
        metrics = candidate_metrics_from_mode(_mode_payload(genome, mode))
        out.append({
            "path": path,
            "name": Path(path).name,
            "mean_ret": metrics["mean_ret"],
            "compound_return_pct": metrics["compound_return_pct"],
            "positive_period_pct": metrics["positive_period_pct"],
            "min_ret": metrics["min_ret"],
            "fitness_v4_robust": metrics["fitness_v4_robust"],
            "failed_gates": metrics["fitness_v4_failed_gates"],
            "direction_bias_penalty": metrics["fitness_v4"].get(
                "direction_bias_penalty", 0.0
            ),
            "persistent_direction_bias_penalty": metrics["fitness_v4"].get(
                "persistent_direction_bias_penalty", 0.0
            ),
            "persistent_direction_bias_abs": metrics["fitness_v4"].get(
                "persistent_direction_bias_abs", 0.0
            ),
            "zero_period_pct": metrics["fitness_v4"].get("zero_period_pct", 0.0),
            "zero_period_penalty": metrics["fitness_v4"].get(
                "zero_period_penalty", 0.0
            ),
            "regime_collapse_penalty": metrics["fitness_v4"].get(
                "regime_collapse_penalty", 0.0
            ),
            "mean_abs_net_direction_bias": metrics["fitness_v4"].get(
                "mean_abs_net_direction_bias", 0.0
            ),
            "regime_positive_rate": metrics["fitness_v4"].get(
                "regime_positive_rate", 0.0
            ),
        })
    return out


def _best_by_compound(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("compound_return_pct", 0.0)))


def _normalized_persisted_router(
    payload: Mapping[str, Any],
    *,
    mode: str,
    fallback_baseline_path: str,
) -> dict[str, Any] | None:
    selected_map_payload = payload.get("selected_regime_map")
    if not isinstance(selected_map_payload, Mapping) or not selected_map_payload:
        return None

    selected_map = {
        str(regime): str(path)
        for regime, path in selected_map_payload.items()
    }
    thresholds = payload.get("thresholds", {})
    if not isinstance(thresholds, Mapping):
        thresholds = {}

    router = dict(payload)
    router["schema_version"] = int(router.get("schema_version", 1) or 1)
    router["mode"] = str(router.get("mode") or mode)
    router["deployment_shape"] = "singleton"
    router["live_policy"] = "probation_only"
    router["uses_live_ensemble"] = False
    router["baseline_path"] = str(router.get("baseline_path") or fallback_baseline_path)
    router["selected_regime_map"] = selected_map
    router["unique_selected_genomes"] = len(set(selected_map.values()))
    router["min_unique_selected_genomes"] = int(
        router.get("min_unique_selected_genomes")
        or thresholds.get("min_unique_selected_genomes")
        or 2
    )
    router["promotion_failures"] = list(router.get("promotion_failures") or [])
    router["promotion_eligible"] = bool(router.get("promotion_eligible", False))
    return router


def _load_persisted_router(
    run_dir: Path,
    *,
    mode: str,
    fallback_baseline_path: str,
) -> dict[str, Any] | None:
    selection_path = run_dir / ROUTER_SELECTION_REPORT
    if not selection_path.exists():
        return None
    return _normalized_persisted_router(
        _load_json(selection_path),
        mode=mode,
        fallback_baseline_path=fallback_baseline_path,
    )


def _theory_verdicts(
    *,
    router: Mapping[str, Any],
    validation_rows: list[dict[str, Any]],
    oos_rows: list[dict[str, Any]],
    sanity_rows: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    selected_map = dict(router.get("selected_regime_map") or {})
    unique_selected = {str(path) for path in selected_map.values()}
    router_promoted = bool(router.get("promotion_eligible", False))
    promotion_failures = set(router.get("promotion_failures") or [])
    best_oos = _best_by_compound(oos_rows)
    best_sanity = _best_by_compound(sanity_rows)
    max_direction_penalty = max(
        [float(row.get("direction_bias_penalty", 0.0)) for row in validation_rows]
        or [0.0]
    )
    min_regime_rate = min(
        [float(row.get("regime_positive_rate", 1.0)) for row in oos_rows + sanity_rows]
        or [1.0]
    )
    return {
        "specialist_to_singleton": {
            "verdict": "accepted" if router_promoted else "rejected",
            "reason": (
                "specialist map passed validation and holdout gates"
                if router_promoted
                else "selected specialist map failed OOS/sanity promotion gates"
            ),
        },
        "combine_genetic_specialists": {
            "verdict": (
                "rejected"
                if "router_collapse" in promotion_failures
                else "tested"
                if len(unique_selected) > 1
                else "inconclusive"
            ),
            "reason": f"selected {len(unique_selected)} unique genome(s) across regimes",
        },
        "direction_bias_penalty": {
            "verdict": "accepted" if max_direction_penalty > 0.0 else "inconclusive",
            "reason": (
                "v2 scoring detected one-sided exposure pressure"
                if max_direction_penalty > 0.0
                else "reports did not show enough directional bias pressure"
            ),
        },
        "regime_collapse_penalty": {
            "verdict": "accepted" if min_regime_rate < 0.50 else "inconclusive",
            "reason": f"minimum holdout regime positive rate was {min_regime_rate:.2f}",
        },
        "teacher_features": {
            "verdict": "inconclusive",
            "reason": "current run used teacher BC seeds, but runtime teacher logits were not added",
        },
        "architecture_change": {
            "verdict": "deferred",
            "reason": "v2 replay tests penalties/router before changing neural topology",
        },
        "best_archive_candidate": {
            "verdict": "observed",
            "reason": (
                f"OOS best={best_oos['name']} {best_oos['compound_return_pct']:.2f}%, "
                f"sanity best={best_sanity['name']} {best_sanity['compound_return_pct']:.2f}%"
                if best_oos and best_sanity
                else "not enough rows"
            ),
        },
    }


def build_summary(run_dir: Path, *, mode: str) -> dict[str, Any]:
    reports = {
        key: _load_json(run_dir / filename)
        for key, filename in DEFAULT_REPORTS.items()
    }
    validation_rows = _genome_metrics(reports["validation"], mode=mode)
    if not validation_rows:
        raise ValueError("validation report contains no evaluated genomes")
    baseline_path = validation_rows[0]["path"]
    recomputed_router = score_specialist_router(
        validation_report=reports["validation"],
        oos_reports=[reports["oos_2025"], reports["sanity_2026_h1"]],
        baseline_path=baseline_path,
        mode=mode,
    )
    persisted_router = _load_persisted_router(
        run_dir,
        mode=mode,
        fallback_baseline_path=baseline_path,
    )
    router = persisted_router or recomputed_router
    router_source = ROUTER_SELECTION_REPORT if persisted_router else "score_specialist_router"
    train_rows = _genome_metrics(reports["train"], mode=mode)
    oos_rows = _genome_metrics(reports["oos_2025"], mode=mode)
    sanity_rows = _genome_metrics(reports["sanity_2026_h1"], mode=mode)
    return {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "mode": mode,
        "baseline_path": baseline_path,
        "specialist_router_source": router_source,
        "singleton_summary": summarize_singleton_candidate(router),
        "specialist_router": router,
        "best_by_split": {
            "train": _best_by_compound(train_rows),
            "validation": _best_by_compound(validation_rows),
            "oos_2025": _best_by_compound(oos_rows),
            "sanity_2026_h1": _best_by_compound(sanity_rows),
        },
        "theory_verdicts": _theory_verdicts(
            router=router,
            validation_rows=validation_rows,
            oos_rows=oos_rows,
            sanity_rows=sanity_rows,
        ),
        "splits": {
            "train": train_rows,
            "validation": validation_rows,
            "oos_2025": oos_rows,
            "sanity_2026_h1": sanity_rows,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--mode", default="fee_fixed_nextbar")
    parser.add_argument("--out")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    summary = build_summary(run_dir, mode=str(args.mode))
    out_path = Path(args.out) if args.out else run_dir / "genetic_core_v2_experiment_summary.json"
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
