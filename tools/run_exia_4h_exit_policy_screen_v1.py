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
    validate_family,
)
from run_exia_timebase_audit_v1 import summarize_trades  # noqa: E402
from simple_research.position_policy import (  # noqa: E402
    simulate_ohlc_position_policy,
    validate_position_policy,
)
from simple_research.strategies import build_market_regime, build_signal  # noqa: E402


DEFAULT_FAMILY = ROOT / "configs" / "exia_4h_exit_policy_family_v1.json"
DEFAULT_OUTPUT = ROOT / "Reports" / "Exia" / "four_hour_exit_policy_screen_v1"
BAR_MS = 240 * 60 * 1000
CONTROL_ID = "CONTROL_SIGNAL_EXIT"


def _entry_family_path(exit_family: dict[str, Any]) -> Path:
    return (ROOT / str(exit_family["entry_source"]["family"])).resolve()


def validate_exit_family(exit_family: dict[str, Any]) -> dict[str, Any]:
    if exit_family.get("schema_version") != "exia.4h_exit_policy_family.v1":
        raise ValueError("unexpected exit-policy family schema")
    if exit_family.get("profile_id") != "exia_4h_v1":
        raise ValueError("exit-policy family must use exia_4h_v1")
    if int(exit_family.get("timeframe_minutes", 0)) != 240:
        raise ValueError("exit-policy family must be sealed to 4h")
    if set(exit_family.get("safety", {}).values()) != {False}:
        raise ValueError("exit-policy family must not grant trading authority")
    policies = exit_family.get("policies")
    if not isinstance(policies, list) or len(policies) != 4:
        raise ValueError("exit-policy family must contain exactly four policies")
    identifiers = [str(item.get("policy_id")) for item in policies]
    if len(identifiers) != len(set(identifiers)) or identifiers.count(CONTROL_ID) != 1:
        raise ValueError("exit-policy identifiers must be unique and include one control")
    control = next(item for item in policies if item["policy_id"] == CONTROL_ID)
    if control.get("comparison_only") is not True:
        raise ValueError("control must be comparison-only")
    for policy in policies:
        validate_position_policy(dict(policy["params"]))
    gate = exit_family.get("stage_gate", {})
    required_true = (
        "require_positive_stress_mean",
        "require_positive_stress_median",
        "require_positive_stress_lcb",
        "require_positive_familywise_lcb",
    )
    if int(gate.get("minimum_closed_trades", 0)) < 20:
        raise ValueError("exit-policy gate cannot use fewer than 20 trades")
    if any(gate.get(name) is not True for name in required_true):
        raise ValueError("all preregistered positive-distribution gates must remain enabled")
    if float(gate.get("minimum_p01_improvement_vs_control_bps", 0.0)) <= 0:
        raise ValueError("P01 improvement versus control must be positive")
    if any(int(gate.get(name, 0)) != 1 for name in (
        "development_to_validation_limit",
        "validation_to_oos_limit",
        "oos_to_sanity_limit",
    )):
        raise ValueError("only one exit policy may advance at each stage")

    entry_path = _entry_family_path(exit_family)
    if sha256_file(entry_path) != str(exit_family["entry_source"]["family_sha256"]):
        raise ValueError("entry family SHA does not match preregistration")
    entry_family = load_json(entry_path)
    validate_family(entry_family)
    candidate_id = str(exit_family["entry_source"]["candidate_id"])
    if candidate_id not in {str(row["candidate_id"]) for row in entry_family["candidates"]}:
        raise ValueError("fixed entry candidate is missing from the sealed family")
    if exit_family["costs_bps"] != entry_family["costs_bps"]:
        raise ValueError("exit-policy costs must match the entry-family costs")
    return entry_family


def _gate_pass(
    metrics: dict[str, Any], gate: dict[str, Any], *, control_p01_bps: float
) -> bool:
    return bool(
        not metrics.get("comparison_only", False)
        and int(metrics["closed_trades"]) >= int(gate["minimum_closed_trades"])
        and float(metrics["mean_stress_net_bps"] or 0.0) > 0.0
        and float(metrics["median_stress_net_bps"] or 0.0) > 0.0
        and metrics["stress_lcb_bps"] is not None
        and float(metrics["stress_lcb_bps"]) > 0.0
        and metrics["familywise_stress_lcb_bps"] is not None
        and float(metrics["familywise_stress_lcb_bps"]) > 0.0
        and float(metrics["p01_stress_net_bps"] - control_p01_bps)
        >= float(gate["minimum_p01_improvement_vs_control_bps"])
    )


def _simulate_window(
    frame: pd.DataFrame,
    signal: pd.Series,
    regime: pd.Series,
    *,
    policy: dict[str, Any],
    window_name: str,
    start_timestamp: int,
    end_timestamp: int,
    exit_family: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    policy_id = str(policy["policy_id"])
    ledger = simulate_ohlc_position_policy(
        frame,
        signal,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        policy=dict(policy["params"]),
    )
    base_cost = float(exit_family["costs_bps"]["base_round_trip"])
    stress_cost = float(exit_family["costs_bps"]["stress_round_trip"])
    if not ledger.empty:
        ledger["policy_id"] = policy_id
        ledger["window"] = window_name
        ledger["timeframe_minutes"] = 240
        ledger["direction"] = ledger["direction"].map({1: "LONG", -1: "SHORT"})
        ledger["signal_timestamp"] = ledger["entry_timestamp"].astype("int64") - BAR_MS
        ledger["market_regime"] = (
            ledger["signal_timestamp"]
            .map(regime)
            .fillna(0)
            .map({1: "bullish", -1: "bearish", 0: "neutral"})
        )
        ledger["base_net_bps"] = ledger["gross_bps"].astype(float) - base_cost
        ledger["stress_net_bps"] = ledger["gross_bps"].astype(float) - stress_cost
    bootstrap = exit_family["bootstrap"]
    summary = summarize_trades(
        ledger,
        seed_key=f"{exit_family['family_id']}:{policy_id}:{window_name}",
        minimum_trades=int(bootstrap["minimum_trades"]),
        bootstrap_samples=int(bootstrap["samples"]),
        alpha=float(bootstrap["alpha"]),
        familywise_alpha=float(bootstrap["alpha"]) / len(exit_family["policies"]),
    )
    values = ledger["stress_net_bps"].astype(float) if not ledger.empty else pd.Series(dtype=float)
    metric = {
        "policy_id": policy_id,
        "window": window_name,
        "comparison_only": bool(policy["comparison_only"]),
        **summary,
        "median_stress_net_bps": float(values.median()) if len(values) else None,
        "p01_stress_net_bps": float(values.quantile(0.01)) if len(values) else None,
        "worst_stress_net_bps": float(values.min()) if len(values) else None,
        "best_stress_net_bps": float(values.max()) if len(values) else None,
        "mean_holding_bars": float(ledger["holding_bars"].mean()) if len(ledger) else None,
    }
    slices: list[dict[str, Any]] = []
    if not ledger.empty:
        for (market_regime, exit_reason), subset in ledger.groupby(
            ["market_regime", "exit_reason"], sort=True
        ):
            slice_values = subset["stress_net_bps"].astype(float)
            slices.append(
                {
                    "policy_id": policy_id,
                    "window": window_name,
                    "market_regime": str(market_regime),
                    "exit_reason": str(exit_reason),
                    "closed_trades": int(len(subset)),
                    "mean_stress_net_bps": float(slice_values.mean()),
                    "median_stress_net_bps": float(slice_values.median()),
                    "p01_stress_net_bps": float(slice_values.quantile(0.01)),
                }
            )
    return ledger, metric, slices


def _rank_passed(metrics: list[dict[str, Any]], *, limit: int) -> list[str]:
    passed = [row for row in metrics if row["gate_pass"]]
    passed.sort(
        key=lambda row: (
            -float(row["familywise_stress_lcb_bps"]),
            -float(row["median_stress_net_bps"]),
            str(row["policy_id"]),
        )
    )
    return [str(row["policy_id"]) for row in passed[:limit]]


def _symbol_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "policy_id",
        "window",
        "symbol",
        "closed_trades",
        "mean_stress_net_bps",
        "median_stress_net_bps",
        "p01_stress_net_bps",
    )
    if trades.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for keys, subset in trades.groupby(["policy_id", "window", "symbol"], sort=True):
        values = subset["stress_net_bps"].astype(float)
        rows.append(
            {
                "policy_id": str(keys[0]),
                "window": str(keys[1]),
                "symbol": str(keys[2]),
                "closed_trades": int(len(subset)),
                "mean_stress_net_bps": float(values.mean()),
                "median_stress_net_bps": float(values.median()),
                "p01_stress_net_bps": float(values.quantile(0.01)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _render_markdown(report: dict[str, Any], metrics: pd.DataFrame) -> str:
    lines = [
        "# Exia 4h fixed-entry exit-policy screen v1",
        "",
        f"Verdict: `{report['verdict']}`.",
        "",
        "Entry and regime are sealed to `MQ_EMA12_48_LONG`; only position exits differ.",
        "This is staged retrospective research and grants no paper/live authority.",
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
        "| Policy | Window | Trades | Mean | Median | LCB | Family LCB | P01 | P01 vs control | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in metrics.sort_values(["window", "policy_id"], kind="stable").itertuples():
        def fmt(value: Any) -> str:
            return "n/a" if value is None or pd.isna(value) else f"{float(value):.2f}"

        lines.append(
            f"| {row.policy_id} | {row.window} | {row.closed_trades} | "
            f"{fmt(row.mean_stress_net_bps)} | {fmt(row.median_stress_net_bps)} | "
            f"{fmt(row.stress_lcb_bps)} | {fmt(row.familywise_stress_lcb_bps)} | "
            f"{fmt(row.p01_stress_net_bps)} | {fmt(row.p01_improvement_vs_control_bps)} | "
            f"{'PASS' if row.gate_pass else 'CONTROL' if row.comparison_only else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "Gate requires >=20 trades, positive stress mean, median, ordinary and family-wise LCB, plus >=100 bps P01 improvement versus the same-window control.",
            "All fills use completed-bar ATR, next-open signal execution, gap-aware stops, prior-extrema trailing and stop-first ambiguity handling.",
            "Detailed trades, symbol slices and exit-reason slices are separate Parquet tables.",
            "All paper/live/orders/promotion flags remain false.",
            "",
        ]
    )
    return "\n".join(lines)


def run(family_path: Path, output_dir: Path) -> dict[str, Any]:
    family_path = family_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    exit_family = load_json(family_path)
    entry_family = validate_exit_family(exit_family)
    frame = _load_panel(entry_family)
    windows = _window_map(entry_family)
    fixed_candidate_id = str(exit_family["entry_source"]["candidate_id"])
    candidate = next(
        row for row in entry_family["candidates"] if row["candidate_id"] == fixed_candidate_id
    )
    strategy = {"kind": candidate["strategy_kind"], "params": dict(candidate["params"])}
    signal = build_signal(frame, strategy)
    regime = build_market_regime(frame, strategy["params"])
    policies = {str(row["policy_id"]): row for row in exit_family["policies"]}
    gate = exit_family["stage_gate"]
    metric_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []
    ledgers: list[pd.DataFrame] = []

    def evaluate(policy_ids: list[str], window_name: str) -> list[dict[str, Any]]:
        start, end = windows[window_name]
        current: list[dict[str, Any]] = []
        ordered_ids = [CONTROL_ID] + [item for item in policy_ids if item != CONTROL_ID]
        temporary: list[tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]] = []
        for policy_id in ordered_ids:
            temporary.append(
                _simulate_window(
                    frame,
                    signal,
                    regime,
                    policy=policies[policy_id],
                    window_name=window_name,
                    start_timestamp=start,
                    end_timestamp=end,
                    exit_family=exit_family,
                )
            )
        control_metric = next(item[1] for item in temporary if item[1]["policy_id"] == CONTROL_ID)
        control_p01 = float(control_metric["p01_stress_net_bps"])
        for ledger, metric, slices in temporary:
            metric["p01_improvement_vs_control_bps"] = (
                float(metric["p01_stress_net_bps"]) - control_p01
            )
            metric["gate_pass"] = _gate_pass(metric, gate, control_p01_bps=control_p01)
            current.append(metric)
            metric_rows.append(metric)
            slice_rows.extend(slices)
            if not ledger.empty:
                ledgers.append(ledger)
        return current

    alternatives = [item for item in policies if item != CONTROL_ID]
    development = evaluate(alternatives, "development")
    validation_ids = _rank_passed(
        development, limit=int(gate["development_to_validation_limit"])
    )
    validation = evaluate(validation_ids, "validation") if validation_ids else []
    oos_ids = _rank_passed(validation, limit=int(gate["validation_to_oos_limit"]))
    oos = evaluate(oos_ids, "oos") if oos_ids else []
    sanity_ids = _rank_passed(oos, limit=int(gate["oos_to_sanity_limit"]))
    sanity = evaluate(sanity_ids, "sanity") if sanity_ids else []
    final_passed = _rank_passed(sanity, limit=1)

    metrics = pd.DataFrame(metric_rows)
    trades = pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()
    exit_slices = pd.DataFrame(slice_rows)
    symbol_metrics = _symbol_metrics(trades)
    verdict = (
        "EXIT_POLICY_PASSED_SANITY"
        if final_passed
        else "NO_EXIT_POLICY_PASSED_STAGED_GATE"
    )
    report = {
        "schema_version": "exia.4h_exit_policy_screen_report.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "family_id": exit_family["family_id"],
        "family_sha256": sha256_file(family_path),
        "entry_family_sha256": exit_family["entry_source"]["family_sha256"],
        "source_dataset_sha256": entry_family["source"]["dataset_sha256"],
        "fixed_entry_candidate": fixed_candidate_id,
        "profile_id": exit_family["profile_id"],
        "timeframe_minutes": 240,
        "verdict": verdict,
        "policy_count": len(policies),
        "stage_progression": {
            "validation": validation_ids,
            "oos": oos_ids,
            "sanity": sanity_ids,
        },
        "final_passed": final_passed,
        "evaluated_configurations": len(metrics),
        "safety": dict(exit_family["safety"]),
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_frames = {
        "metrics": metrics,
        "trades": trades,
        "exit_reason_metrics": exit_slices,
        "symbol_metrics": symbol_metrics,
    }
    artifacts: dict[str, dict[str, str]] = {}
    for name, artifact_frame in artifact_frames.items():
        path = output_dir / f"{name}.parquet"
        artifact_frame.to_parquet(path, index=False)
        artifacts[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
        }
    report["artifacts"] = artifacts
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
    parser = argparse.ArgumentParser(description="Run staged Exia 4h exit-policy screen")
    parser.add_argument("--family", type=Path, default=DEFAULT_FAMILY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args.family, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
