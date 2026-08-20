from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RUNTIME = SRC / "panteon_runtime"
TOOLS = ROOT / "tools"
for path in (SRC, RUNTIME, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from exia.scoring import deterministic_seed, moving_block_lcb  # noqa: E402
from run_exia_4h_candidate_screen_v1 import (  # noqa: E402
    _load_panel,
    _window_map,
    load_json,
    sha256_file,
)
from run_exia_4h_sparse_event_screen_v1 import (  # noqa: E402
    CONTROL_ID,
    validate_event_family,
)
from simple_research.forward_labels import build_event_forward_labels  # noqa: E402
from simple_research.strategies import build_market_regime, build_signal  # noqa: E402


DEFAULT_SPEC = ROOT / "configs" / "exia_4h_forward_feasibility_v1.json"
DEFAULT_OUTPUT = ROOT / "Reports" / "Exia" / "four_hour_forward_feasibility_v1"


def validate_spec(
    spec: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if spec.get("schema_version") != "exia.4h_forward_feasibility.v1":
        raise ValueError("unexpected forward-feasibility schema")
    if spec.get("profile_id") != "exia_4h_v1" or int(spec.get("timeframe_minutes", 0)) != 240:
        raise ValueError("forward feasibility must be sealed to exia_4h_v1")
    if spec.get("horizons_bars") != [1, 3, 6, 12]:
        raise ValueError("forward feasibility horizons must remain [1, 3, 6, 12]")
    if set(spec.get("safety", {}).values()) != {False}:
        raise ValueError("forward feasibility must not grant trading authority")
    gate = spec.get("stage_gate", {})
    if int(gate.get("minimum_observations", 0)) < 20:
        raise ValueError("forward feasibility requires at least 20 observations")
    for name in (
        "require_positive_stress_mean",
        "require_positive_stress_median",
        "require_positive_stress_lcb",
        "require_positive_familywise_lcb",
    ):
        if gate.get(name) is not True:
            raise ValueError(f"forward feasibility must keep {name} enabled")
    for name in (
        "development_to_validation_limit",
        "validation_to_oos_limit",
        "oos_to_sanity_limit",
    ):
        if int(gate.get(name, 0)) != 1:
            raise ValueError("only one event/horizon combination may advance")
    source = spec["event_family_source"]
    event_path = (ROOT / str(source["family"])).resolve()
    if sha256_file(event_path) != str(source["family_sha256"]):
        raise ValueError("event-family SHA does not match feasibility preregistration")
    event_family = load_json(event_path)
    _, entry_family, _ = validate_event_family(event_family)
    return event_family, entry_family


def _combination_id(event_id: str, horizon: int) -> str:
    return f"{event_id}__H{horizon}"


def _summarize(
    labels: pd.DataFrame,
    *,
    seed_key: str,
    minimum_observations: int,
    bootstrap_samples: int,
    alpha: float,
    familywise_alpha: float,
) -> dict[str, Any]:
    count = len(labels)
    if not count:
        return {
            "observations": 0,
            "mean_gross_bps": None,
            "mean_stress_net_bps": None,
            "median_stress_net_bps": None,
            "stress_lcb_bps": None,
            "familywise_stress_lcb_bps": None,
            "positive_stress_rate": None,
            "p01_stress_net_bps": None,
            "median_mfe_bps": None,
            "median_mae_bps": None,
            "p95_mae_bps": None,
            "mfe_surplus_lcb_bps": None,
            "median_capture_ratio": None,
            "status": "INSUFFICIENT",
        }
    values = labels["stress_net_bps"].astype(float)
    timestamps = labels["event_timestamp"].astype("int64")
    lcb = moving_block_lcb(
        values,
        timestamps,
        samples=bootstrap_samples,
        alpha=alpha,
        seed=deterministic_seed(seed_key),
    )
    familywise_lcb = moving_block_lcb(
        values,
        timestamps,
        samples=bootstrap_samples,
        alpha=familywise_alpha,
        seed=deterministic_seed(f"{seed_key}:familywise"),
    )
    mfe_surplus_lcb = moving_block_lcb(
        labels["mfe_stress_surplus_bps"].astype(float),
        timestamps,
        samples=bootstrap_samples,
        alpha=alpha,
        seed=deterministic_seed(f"{seed_key}:mfe"),
    )
    mean_stress = float(values.mean())
    median_stress = float(values.median())
    if count < minimum_observations or lcb is None or familywise_lcb is None:
        status = "INSUFFICIENT"
    elif mean_stress > 0 and median_stress > 0 and lcb > 0 and familywise_lcb > 0:
        status = "PASS"
    else:
        status = "FAIL"
    capture = pd.to_numeric(labels["capture_ratio"], errors="coerce").dropna()
    return {
        "observations": int(count),
        "mean_gross_bps": float(labels["gross_bps"].mean()),
        "mean_stress_net_bps": mean_stress,
        "median_stress_net_bps": median_stress,
        "stress_lcb_bps": lcb,
        "familywise_stress_lcb_bps": familywise_lcb,
        "positive_stress_rate": float((values > 0).mean()),
        "p01_stress_net_bps": float(values.quantile(0.01)),
        "median_mfe_bps": float(labels["mfe_bps"].median()),
        "median_mae_bps": float(labels["mae_bps"].median()),
        "p95_mae_bps": float(labels["mae_bps"].quantile(0.95)),
        "mfe_surplus_lcb_bps": mfe_surplus_lcb,
        "median_capture_ratio": float(capture.median()) if len(capture) else None,
        "status": status,
    }


def _gate_pass(metric: dict[str, Any], gate: dict[str, Any]) -> bool:
    return bool(
        not metric["comparison_only"]
        and int(metric["observations"]) >= int(gate["minimum_observations"])
        and float(metric["mean_stress_net_bps"] or 0.0) > 0.0
        and float(metric["median_stress_net_bps"] or 0.0) > 0.0
        and metric["stress_lcb_bps"] is not None
        and float(metric["stress_lcb_bps"]) > 0.0
        and metric["familywise_stress_lcb_bps"] is not None
        and float(metric["familywise_stress_lcb_bps"]) > 0.0
    )


def _rank_passed(metrics: list[dict[str, Any]], limit: int) -> list[str]:
    passed = [row for row in metrics if row["gate_pass"]]
    passed.sort(
        key=lambda row: (
            -float(row["familywise_stress_lcb_bps"]),
            -float(row["median_stress_net_bps"]),
            str(row["combination_id"]),
        )
    )
    return [str(row["combination_id"]) for row in passed[:limit]]


def _slice_metrics(
    observations: pd.DataFrame,
    *,
    column: str,
    spec: dict[str, Any],
    slice_trials: int,
) -> pd.DataFrame:
    if observations.empty:
        return pd.DataFrame()
    bootstrap = spec["bootstrap"]
    rows: list[dict[str, Any]] = []
    for keys, subset in observations.groupby(
        ["combination_id", "window", column], sort=True
    ):
        rows.append(
            {
                "combination_id": str(keys[0]),
                "window": str(keys[1]),
                column: str(keys[2]),
                **_summarize(
                    subset,
                    seed_key=f"{spec['analysis_id']}:slice:{column}:{keys[0]}:{keys[1]}:{keys[2]}",
                    minimum_observations=int(bootstrap["minimum_observations"]),
                    bootstrap_samples=int(bootstrap["samples"]),
                    alpha=float(bootstrap["alpha"]),
                    familywise_alpha=float(bootstrap["alpha"]) / max(1, slice_trials),
                ),
            }
        )
    return pd.DataFrame(rows)


def _render_markdown(report: dict[str, Any], metrics: pd.DataFrame) -> str:
    lines = [
        "# Exia 4h sparse-event forward feasibility v1",
        "",
        f"Verdict: `{report['verdict']}`.",
        "",
        "This audit labels event information directly; it does not use a stop, trailing rule or position policy.",
        "It grants no paper/live authority.",
        "",
        "## Stage progression",
        "",
        f"- Development -> validation: {', '.join(report['stage_progression']['validation']) or 'none'}",
        f"- Validation -> OOS: {', '.join(report['stage_progression']['oos']) or 'none'}",
        f"- OOS -> sanity: {', '.join(report['stage_progression']['sanity']) or 'none'}",
        f"- Final passed: {', '.join(report['final_passed']) or 'none'}",
        "",
        "## Results",
        "",
        "| Event | H | Window | Events | Labels | Coverage | Mean | Median | LCB | Family LCB | MFE med | MAE med | Gate |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    def fmt(value: Any) -> str:
        return "n/a" if value is None or pd.isna(value) else f"{float(value):.2f}"

    for row in metrics.sort_values(["window", "event_id", "horizon_bars"], kind="stable").itertuples():
        lines.append(
            f"| {row.event_id} | {row.horizon_bars} | {row.window} | {row.candidate_events} | "
            f"{row.observations} | {fmt(row.label_coverage)} | {fmt(row.mean_stress_net_bps)} | "
            f"{fmt(row.median_stress_net_bps)} | {fmt(row.stress_lcb_bps)} | "
            f"{fmt(row.familywise_stress_lcb_bps)} | {fmt(row.median_mfe_bps)} | "
            f"{fmt(row.median_mae_bps)} | "
            f"{'PASS' if row.gate_pass else 'CONTROL' if row.comparison_only else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "Returns are next-open to open after H bars and include the sealed 16 bps stress round-trip cost.",
            "Overlapping labels are allowed, but the LCB resamples complete UTC-day clusters.",
            "All paper/live/orders/promotion flags remain false.",
            "",
        ]
    )
    return "\n".join(lines)


def run(spec_path: Path, output_dir: Path) -> dict[str, Any]:
    spec_path = spec_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    spec = load_json(spec_path)
    event_family, entry_family = validate_spec(spec)
    frame = _load_panel(entry_family)
    windows = _window_map(entry_family)
    events = {str(row["event_id"]): row for row in event_family["events"]}
    shared = dict(event_family["shared_signal_params"])
    signals: dict[str, pd.Series] = {}
    for event_id, event in events.items():
        signals[event_id] = build_signal(
            frame,
            {
                "kind": str(event["strategy_kind"]),
                "params": {**shared, **dict(event["params"])},
            },
        )
    regime = build_market_regime(frame, shared)
    base_cost = float(entry_family["costs_bps"]["base_round_trip"])
    stress_cost = float(entry_family["costs_bps"]["stress_round_trip"])
    horizons = [int(value) for value in spec["horizons_bars"]]
    trial_count = len(events) * len(horizons)
    gate = spec["stage_gate"]
    bootstrap = spec["bootstrap"]
    metric_rows: list[dict[str, Any]] = []
    observation_frames: list[pd.DataFrame] = []

    def evaluate(combinations: list[tuple[str, int]], window_name: str) -> list[dict[str, Any]]:
        start, end = windows[window_name]
        requested_horizons = sorted({horizon for _, horizon in combinations})
        ordered = [
            (CONTROL_ID, horizon) for horizon in requested_horizons
        ] + [item for item in combinations if item[0] != CONTROL_ID]
        ordered = list(dict.fromkeys(ordered))
        current: list[dict[str, Any]] = []
        for event_id, horizon in ordered:
            labels = build_event_forward_labels(
                frame,
                signals[event_id],
                horizons_bars=[horizon],
                start_timestamp=start,
                end_timestamp=end,
                base_cost_bps=base_cost,
                stress_cost_bps=stress_cost,
            )
            combination_id = _combination_id(event_id, horizon)
            if not labels.empty:
                labels["event_id"] = event_id
                labels["combination_id"] = combination_id
                labels["window"] = window_name
                labels["direction"] = labels["direction"].map({1: "LONG", -1: "SHORT"})
                labels["market_regime"] = (
                    labels["event_timestamp"]
                    .map(regime)
                    .fillna(0)
                    .map({1: "bullish", -1: "bearish", 0: "neutral"})
                )
                observation_frames.append(labels)
            event_mask = (
                (frame["timestamp"] >= start)
                & (frame["timestamp"] < end)
                & signals[event_id].ne(0)
            )
            candidate_events = int(event_mask.sum())
            summary = _summarize(
                labels,
                seed_key=f"{spec['analysis_id']}:{combination_id}:{window_name}",
                minimum_observations=int(bootstrap["minimum_observations"]),
                bootstrap_samples=int(bootstrap["samples"]),
                alpha=float(bootstrap["alpha"]),
                familywise_alpha=float(bootstrap["alpha"]) / trial_count,
            )
            metric = {
                "combination_id": combination_id,
                "event_id": event_id,
                "horizon_bars": horizon,
                "window": window_name,
                "comparison_only": bool(events[event_id]["comparison_only"]),
                "candidate_events": candidate_events,
                "label_coverage": (
                    float(summary["observations"] / candidate_events)
                    if candidate_events
                    else None
                ),
                **summary,
            }
            metric["gate_pass"] = _gate_pass(metric, gate)
            current.append(metric)
            metric_rows.append(metric)
        return current

    development_combinations = [
        (event_id, horizon) for event_id in events for horizon in horizons
    ]
    development = evaluate(development_combinations, "development")
    validation_ids = _rank_passed(development, int(gate["development_to_validation_limit"]))

    def decode(values: list[str]) -> list[tuple[str, int]]:
        return [(value.rsplit("__H", 1)[0], int(value.rsplit("__H", 1)[1])) for value in values]

    validation = evaluate(decode(validation_ids), "validation") if validation_ids else []
    oos_ids = _rank_passed(validation, int(gate["validation_to_oos_limit"]))
    oos = evaluate(decode(oos_ids), "oos") if oos_ids else []
    sanity_ids = _rank_passed(oos, int(gate["oos_to_sanity_limit"]))
    sanity = evaluate(decode(sanity_ids), "sanity") if sanity_ids else []
    final_passed = _rank_passed(sanity, 1)

    metrics = pd.DataFrame(metric_rows)
    observations = (
        pd.concat(observation_frames, ignore_index=True)
        if observation_frames
        else pd.DataFrame()
    )
    symbol_metrics = _slice_metrics(
        observations,
        column="symbol",
        spec=spec,
        slice_trials=trial_count * 8,
    )
    regime_metrics = _slice_metrics(
        observations,
        column="market_regime",
        spec=spec,
        slice_trials=trial_count * 3,
    )
    verdict = (
        "FORWARD_FEASIBILITY_PASSED_SANITY"
        if final_passed
        else "NO_EVENT_HORIZON_PASSED_STAGED_GATE"
    )
    report = {
        "schema_version": "exia.4h_forward_feasibility_report.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_id": spec["analysis_id"],
        "spec_sha256": sha256_file(spec_path),
        "event_family_sha256": spec["event_family_source"]["family_sha256"],
        "source_dataset_sha256": entry_family["source"]["dataset_sha256"],
        "profile_id": spec["profile_id"],
        "timeframe_minutes": 240,
        "horizons_bars": horizons,
        "stress_cost_bps": stress_cost,
        "trial_count": trial_count,
        "verdict": verdict,
        "stage_progression": {
            "validation": validation_ids,
            "oos": oos_ids,
            "sanity": sanity_ids,
        },
        "final_passed": final_passed,
        "evaluated_configurations": len(metrics),
        "safety": dict(spec["safety"]),
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_frames = {
        "metrics": metrics,
        "observations": observations,
        "symbol_metrics": symbol_metrics,
        "regime_metrics": regime_metrics,
    }
    report["artifacts"] = {}
    for name, artifact_frame in artifact_frames.items():
        path = output_dir / f"{name}.parquet"
        artifact_frame.to_parquet(path, index=False)
        report["artifacts"][name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
        }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        _render_markdown(report, metrics), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "stage_progression": report["stage_progression"],
                "final_passed": final_passed,
                "orders_enabled": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Exia 4h forward feasibility audit")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args.spec, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
