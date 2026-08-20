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

from run_exia_4h_candidate_screen_v1 import (  # noqa: E402
    _load_panel,
    _window_map,
    load_json,
    sha256_file,
)
from run_exia_4h_forward_feasibility_v1 import (  # noqa: E402
    _summarize,
    validate_spec as validate_feasibility_spec,
)
from simple_research.forward_labels import build_event_forward_labels  # noqa: E402
from simple_research.strategies import build_market_regime, build_signal  # noqa: E402


DEFAULT_SPEC = ROOT / "configs" / "exia_4h_sol_pullback_h12_replication_v1.json"
DEFAULT_OUTPUT = ROOT / "Reports" / "Exia" / "four_hour_sol_pullback_h12_replication_v1"


def validate_replication_spec(
    spec: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if spec.get("schema_version") != "exia.4h_single_slice_replication.v1":
        raise ValueError("unexpected single-slice replication schema")
    if spec.get("profile_id") != "exia_4h_v1" or int(spec.get("timeframe_minutes", 0)) != 240:
        raise ValueError("replication must be sealed to exia_4h_v1")
    if spec.get("evaluation_windows") != ["validation", "oos", "sanity"]:
        raise ValueError("replication windows must remain validation, OOS, sanity")
    if set(spec.get("safety", {}).values()) != {False}:
        raise ValueError("replication must not grant trading authority")
    candidate = spec.get("candidate", {})
    expected_candidate = {
        "candidate_id": "SOL_PULLBACK_H12_REPLICATION",
        "source_combination_id": "BULL_TREND_FAST_RECLAIM__H12",
        "event_id": "BULL_TREND_FAST_RECLAIM",
        "symbol": "SOL/USDT",
        "direction": "LONG",
        "horizon_bars": 12,
    }
    if candidate != expected_candidate:
        raise ValueError("replication candidate differs from the selected development slice")
    gate = spec.get("stage_gate", {})
    if int(gate.get("minimum_observations", 0)) < 20:
        raise ValueError("replication requires at least 20 observations")
    if float(gate.get("minimum_label_coverage", 0.0)) < 0.95:
        raise ValueError("replication label coverage cannot be below 95%")
    for name in (
        "require_positive_stress_mean",
        "require_positive_stress_median",
        "require_positive_stress_lcb",
        "require_positive_familywise_lcb",
        "require_positive_mfe_surplus_lcb",
    ):
        if gate.get(name) is not True:
            raise ValueError(f"replication must keep {name} enabled")

    source = spec["development_source"]
    pinned = (
        (ROOT / str(source["feasibility_spec"]), "feasibility_spec_sha256"),
        (ROOT / str(source["report"]), "report_sha256"),
        (ROOT / str(source["symbol_metrics"]), "symbol_metrics_sha256"),
    )
    for path, hash_key in pinned:
        if not path.is_file() or sha256_file(path) != str(source[hash_key]):
            raise ValueError(f"development source SHA mismatch: {hash_key}")
    feasibility_spec = load_json(pinned[0][0])
    event_family, entry_family = validate_feasibility_spec(feasibility_spec)
    report = load_json(pinned[1][0])
    if report.get("spec_sha256") != source["feasibility_spec_sha256"]:
        raise ValueError("development report references another feasibility spec")
    if report.get("verdict") != "NO_EVENT_HORIZON_PASSED_STAGED_GATE":
        raise ValueError("replication source must be the closed aggregate experiment")
    metrics = pd.read_parquet(pinned[2][0])
    selected = metrics.loc[
        (metrics["combination_id"] == candidate["source_combination_id"])
        & (metrics["window"] == "development")
        & (metrics["symbol"] == candidate["symbol"])
    ]
    if len(selected) != 1:
        raise ValueError("development source does not contain exactly one selected slice")
    row = selected.iloc[0]
    if (
        row["status"] != "PASS"
        or int(row["observations"]) < 20
        or float(row["mean_stress_net_bps"]) <= 0
        or float(row["median_stress_net_bps"]) <= 0
        or float(row["stress_lcb_bps"]) <= 0
        or float(row["familywise_stress_lcb_bps"]) <= 0
    ):
        raise ValueError("selected development slice does not support replication")
    return feasibility_spec, event_family, entry_family


def isolate_symbol_signal(
    frame: pd.DataFrame,
    signal: pd.Series,
    *,
    symbol: str,
) -> pd.Series:
    if len(frame) != len(signal):
        raise ValueError("frame and signal lengths differ")
    isolated = signal.where(frame["symbol"].astype(str).eq(symbol), 0).astype("int8")
    if int(isolated.loc[frame["symbol"].astype(str).ne(symbol)].abs().sum()) != 0:
        raise ValueError("symbol isolation failed")
    return isolated


def _gate_pass(metric: dict[str, Any], gate: dict[str, Any]) -> bool:
    return bool(
        int(metric["observations"]) >= int(gate["minimum_observations"])
        and float(metric["label_coverage"] or 0.0) >= float(gate["minimum_label_coverage"])
        and float(metric["mean_stress_net_bps"] or 0.0) > 0.0
        and float(metric["median_stress_net_bps"] or 0.0) > 0.0
        and metric["stress_lcb_bps"] is not None
        and float(metric["stress_lcb_bps"]) > 0.0
        and metric["familywise_stress_lcb_bps"] is not None
        and float(metric["familywise_stress_lcb_bps"]) > 0.0
        and metric["mfe_surplus_lcb_bps"] is not None
        and float(metric["mfe_surplus_lcb_bps"]) > 0.0
    )


def _render_markdown(report: dict[str, Any], metrics: pd.DataFrame) -> str:
    lines = [
        "# Exia 4h SOL pullback H12 replication v1",
        "",
        f"Verdict: `{report['verdict']}`.",
        f"Terminal decision: `{report['terminal_decision']}`.",
        "",
        "This is a fixed single-candidate replication. Development was not reselected or recalculated.",
        "It grants no paper/live authority.",
        "",
        "| Window | Events | Labels | Coverage | Mean | Median | LCB | MFE surplus LCB | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    def fmt(value: Any) -> str:
        return "n/a" if value is None or pd.isna(value) else f"{float(value):.2f}"

    for row in metrics.itertuples():
        lines.append(
            f"| {row.window} | {row.candidate_events} | {row.observations} | "
            f"{fmt(row.label_coverage)} | {fmt(row.mean_stress_net_bps)} | "
            f"{fmt(row.median_stress_net_bps)} | {fmt(row.stress_lcb_bps)} | "
            f"{fmt(row.mfe_surplus_lcb_bps)} | {'PASS' if row.gate_pass else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "OOS and sanity are evaluated only when every preceding window passes unchanged gates.",
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
    _, event_family, entry_family = validate_replication_spec(spec)
    frame = _load_panel(entry_family)
    windows = _window_map(entry_family)
    candidate = spec["candidate"]
    event = next(
        row for row in event_family["events"] if row["event_id"] == candidate["event_id"]
    )
    params = {**event_family["shared_signal_params"], **event["params"]}
    full_signal = build_signal(
        frame, {"kind": event["strategy_kind"], "params": params}
    )
    signal = isolate_symbol_signal(frame, full_signal, symbol=candidate["symbol"])
    regime = build_market_regime(frame, event_family["shared_signal_params"])
    base_cost = float(entry_family["costs_bps"]["base_round_trip"])
    stress_cost = float(entry_family["costs_bps"]["stress_round_trip"])
    bootstrap = spec["bootstrap"]
    gate = spec["stage_gate"]
    metric_rows: list[dict[str, Any]] = []
    observation_frames: list[pd.DataFrame] = []

    def evaluate(window_name: str) -> dict[str, Any]:
        start, end = windows[window_name]
        labels = build_event_forward_labels(
            frame,
            signal,
            horizons_bars=[int(candidate["horizon_bars"])],
            start_timestamp=start,
            end_timestamp=end,
            base_cost_bps=base_cost,
            stress_cost_bps=stress_cost,
        )
        event_mask = (
            (frame["timestamp"] >= start)
            & (frame["timestamp"] < end)
            & signal.ne(0)
        )
        candidate_events = int(event_mask.sum())
        if not labels.empty:
            if set(labels["symbol"].unique()) != {candidate["symbol"]}:
                raise ValueError("replication labels contain another symbol")
            labels["candidate_id"] = candidate["candidate_id"]
            labels["window"] = window_name
            labels["direction"] = labels["direction"].map({1: "LONG", -1: "SHORT"})
            labels["market_regime"] = (
                labels["event_timestamp"]
                .map(regime)
                .fillna(0)
                .map({1: "bullish", -1: "bearish", 0: "neutral"})
            )
            observation_frames.append(labels)
        summary = _summarize(
            labels,
            seed_key=f"{spec['analysis_id']}:{window_name}",
            minimum_observations=int(bootstrap["minimum_observations"]),
            bootstrap_samples=int(bootstrap["samples"]),
            alpha=float(bootstrap["alpha"]),
            familywise_alpha=float(bootstrap["alpha"]),
        )
        metric = {
            "candidate_id": candidate["candidate_id"],
            "window": window_name,
            "candidate_events": candidate_events,
            "label_coverage": (
                float(summary["observations"] / candidate_events)
                if candidate_events
                else None
            ),
            **summary,
        }
        metric["gate_pass"] = _gate_pass(metric, gate)
        metric_rows.append(metric)
        return metric

    validation = evaluate("validation")
    oos = evaluate("oos") if validation["gate_pass"] else None
    sanity = evaluate("sanity") if oos and oos["gate_pass"] else None
    if not validation["gate_pass"]:
        verdict = "VALIDATION_REPLICATION_FAILED"
    elif not oos or not oos["gate_pass"]:
        verdict = "OOS_REPLICATION_FAILED"
    elif not sanity or not sanity["gate_pass"]:
        verdict = "SANITY_REPLICATION_FAILED"
    else:
        verdict = "REPLICATION_PASSED_SANITY"
    success = verdict == "REPLICATION_PASSED_SANITY"
    terminal_decision = spec["terminal_policy"]["success" if success else "failure"]
    metrics = pd.DataFrame(metric_rows)
    observations = (
        pd.concat(observation_frames, ignore_index=True)
        if observation_frames
        else pd.DataFrame()
    )
    report = {
        "schema_version": "exia.4h_single_slice_replication_report.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_id": spec["analysis_id"],
        "spec_sha256": sha256_file(spec_path),
        "development_report_sha256": spec["development_source"]["report_sha256"],
        "development_symbol_metrics_sha256": spec["development_source"]["symbol_metrics_sha256"],
        "source_dataset_sha256": entry_family["source"]["dataset_sha256"],
        "candidate": dict(candidate),
        "verdict": verdict,
        "terminal_decision": terminal_decision,
        "evaluated_windows": metrics["window"].tolist(),
        "window_pass": {
            str(row.window): bool(row.gate_pass) for row in metrics.itertuples()
        },
        "safety": dict(spec["safety"]),
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report["artifacts"] = {}
    for name, artifact_frame in {"metrics": metrics, "observations": observations}.items():
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
                "terminal_decision": terminal_decision,
                "evaluated_windows": report["evaluated_windows"],
                "window_pass": report["window_pass"],
                "orders_enabled": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replicate fixed Exia SOL pullback H12 slice")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args.spec, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
