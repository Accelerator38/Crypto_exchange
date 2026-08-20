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
from run_exia_4h_exit_policy_screen_v1 import validate_exit_family  # noqa: E402
from run_exia_timebase_audit_v1 import summarize_trades  # noqa: E402
from simple_research.position_policy import simulate_ohlc_position_policy  # noqa: E402
from simple_research.strategies import build_market_regime, build_signal  # noqa: E402


DEFAULT_FAMILY = ROOT / "configs" / "exia_4h_sparse_event_family_v1.json"
DEFAULT_OUTPUT = ROOT / "Reports" / "Exia" / "four_hour_sparse_event_screen_v1"
CONTROL_ID = "CONTROL_MQ_FRESH_ACTIVATION"
BAR_MS = 240 * 60 * 1000


def validate_event_family(
    family: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if family.get("schema_version") != "exia.4h_sparse_event_family.v1":
        raise ValueError("unexpected sparse-event family schema")
    if family.get("profile_id") != "exia_4h_v1" or int(family.get("timeframe_minutes", 0)) != 240:
        raise ValueError("sparse-event family must be sealed to exia_4h_v1")
    if set(family.get("safety", {}).values()) != {False}:
        raise ValueError("sparse-event family must not grant trading authority")
    events = family.get("events")
    if not isinstance(events, list) or len(events) != 4:
        raise ValueError("sparse-event family must contain one control and three alternatives")
    identifiers = [str(item.get("event_id")) for item in events]
    if len(identifiers) != len(set(identifiers)) or identifiers.count(CONTROL_ID) != 1:
        raise ValueError("event identifiers must be unique and include the fixed control")
    control = next(item for item in events if item["event_id"] == CONTROL_ID)
    if control.get("comparison_only") is not True:
        raise ValueError("event control must be comparison-only")
    expected_kinds = {
        "ema_activation_event",
        "market_regime_transition_event",
        "cross_sectional_breakout_event",
        "trend_pullback_continuation_event",
    }
    if {str(item.get("strategy_kind")) for item in events} != expected_kinds:
        raise ValueError("sparse-event family kinds differ from preregistration")
    shared_keys = set(family.get("shared_signal_params", {}))
    if any(shared_keys & set(item.get("params", {})) for item in events):
        raise ValueError("event-specific params cannot override shared signal settings")
    gate = family.get("stage_gate", {})
    if int(gate.get("minimum_closed_trades", 0)) < 20:
        raise ValueError("sparse-event gate cannot use fewer than 20 trades")
    for name in (
        "require_positive_stress_mean",
        "require_positive_stress_median",
        "require_positive_stress_lcb",
        "require_positive_familywise_lcb",
    ):
        if gate.get(name) is not True:
            raise ValueError(f"sparse-event gate must keep {name} enabled")
    if float(gate.get("minimum_p01_improvement_vs_control_bps", 0.0)) <= 0:
        raise ValueError("P01 improvement versus control must be positive")
    for name in (
        "development_to_validation_limit",
        "validation_to_oos_limit",
        "oos_to_sanity_limit",
    ):
        if int(gate.get(name, 0)) != 1:
            raise ValueError("only one sparse event may advance at each stage")

    policy_source = family["position_policy_source"]
    if policy_source.get("entry_mode") != "event":
        raise ValueError("sparse-event positions must use event entry mode")
    exit_path = (ROOT / str(policy_source["family"])).resolve()
    if sha256_file(exit_path) != str(policy_source["family_sha256"]):
        raise ValueError("exit-policy family SHA does not match preregistration")
    exit_family = load_json(exit_path)
    entry_family = validate_exit_family(exit_family)
    policy_id = str(policy_source["policy_id"])
    policies = {str(item["policy_id"]): item for item in exit_family["policies"]}
    if policy_id not in policies or policies[policy_id].get("comparison_only") is True:
        raise ValueError("fixed event position policy must be a preregistered alternative")
    return exit_family, entry_family, policies[policy_id]


def _gate_pass(
    metric: dict[str, Any], gate: dict[str, Any], *, control_p01_bps: float
) -> bool:
    return bool(
        not metric["comparison_only"]
        and int(metric["closed_trades"]) >= int(gate["minimum_closed_trades"])
        and float(metric["mean_stress_net_bps"] or 0.0) > 0.0
        and float(metric["median_stress_net_bps"] or 0.0) > 0.0
        and metric["stress_lcb_bps"] is not None
        and float(metric["stress_lcb_bps"]) > 0.0
        and metric["familywise_stress_lcb_bps"] is not None
        and float(metric["familywise_stress_lcb_bps"]) > 0.0
        and float(metric["p01_stress_net_bps"] - control_p01_bps)
        >= float(gate["minimum_p01_improvement_vs_control_bps"])
    )


def _simulate_window(
    frame: pd.DataFrame,
    signal: pd.Series,
    regime: pd.Series,
    *,
    event: dict[str, Any],
    position_policy: dict[str, Any],
    window_name: str,
    start_timestamp: int,
    end_timestamp: int,
    event_family: dict[str, Any],
    entry_family: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    event_id = str(event["event_id"])
    ledger = simulate_ohlc_position_policy(
        frame,
        signal,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        policy=dict(position_policy["params"]),
        entry_mode="event",
    )
    decision_mask = (
        (frame["timestamp"] >= start_timestamp - BAR_MS)
        & (frame["timestamp"] < end_timestamp - BAR_MS)
        & signal.ne(0)
    )
    event_count = int(decision_mask.sum())
    base_cost = float(entry_family["costs_bps"]["base_round_trip"])
    stress_cost = float(entry_family["costs_bps"]["stress_round_trip"])
    if not ledger.empty:
        ledger["event_id"] = event_id
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
    bootstrap = event_family["bootstrap"]
    summary = summarize_trades(
        ledger,
        seed_key=f"{event_family['family_id']}:{event_id}:{window_name}",
        minimum_trades=int(bootstrap["minimum_trades"]),
        bootstrap_samples=int(bootstrap["samples"]),
        alpha=float(bootstrap["alpha"]),
        familywise_alpha=float(bootstrap["alpha"]) / len(event_family["events"]),
    )
    values = ledger["stress_net_bps"].astype(float) if not ledger.empty else pd.Series(dtype=float)
    metric = {
        "event_id": event_id,
        "window": window_name,
        "comparison_only": bool(event["comparison_only"]),
        "candidate_events": event_count,
        "event_to_trade_rate": float(len(ledger) / event_count) if event_count else None,
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
            values = subset["stress_net_bps"].astype(float)
            slices.append(
                {
                    "event_id": event_id,
                    "window": window_name,
                    "market_regime": str(market_regime),
                    "exit_reason": str(exit_reason),
                    "closed_trades": int(len(subset)),
                    "mean_stress_net_bps": float(values.mean()),
                    "median_stress_net_bps": float(values.median()),
                    "p01_stress_net_bps": float(values.quantile(0.01)),
                }
            )
    return ledger, metric, slices


def _rank_passed(metrics: list[dict[str, Any]], limit: int) -> list[str]:
    passed = [row for row in metrics if row["gate_pass"]]
    passed.sort(
        key=lambda row: (
            -float(row["familywise_stress_lcb_bps"]),
            -float(row["median_stress_net_bps"]),
            str(row["event_id"]),
        )
    )
    return [str(row["event_id"]) for row in passed[:limit]]


def _group_metrics(
    trades: pd.DataFrame,
    column: str,
    *,
    event_family: dict[str, Any],
) -> pd.DataFrame:
    columns = (
        "event_id",
        "window",
        column,
        "closed_trades",
        "mean_stress_net_bps",
        "median_stress_net_bps",
        "p01_stress_net_bps",
        "stress_lcb_bps",
        "familywise_stress_lcb_bps",
        "status",
    )
    if trades.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    bootstrap = event_family["bootstrap"]
    slice_count = len(event_family["events"]) * int(trades[column].nunique())
    for keys, subset in trades.groupby(["event_id", "window", column], sort=True):
        values = subset["stress_net_bps"].astype(float)
        summary = summarize_trades(
            subset,
            seed_key=f"{event_family['family_id']}:slice:{column}:{keys[0]}:{keys[1]}:{keys[2]}",
            minimum_trades=int(bootstrap["minimum_trades"]),
            bootstrap_samples=int(bootstrap["samples"]),
            alpha=float(bootstrap["alpha"]),
            familywise_alpha=float(bootstrap["alpha"]) / max(1, slice_count),
        )
        rows.append(
            {
                "event_id": str(keys[0]),
                "window": str(keys[1]),
                column: str(keys[2]),
                "closed_trades": int(len(subset)),
                "mean_stress_net_bps": float(values.mean()),
                "median_stress_net_bps": float(values.median()),
                "p01_stress_net_bps": float(values.quantile(0.01)),
                "stress_lcb_bps": summary["stress_lcb_bps"],
                "familywise_stress_lcb_bps": summary["familywise_stress_lcb_bps"],
                "status": summary["status"],
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _render_markdown(report: dict[str, Any], metrics: pd.DataFrame) -> str:
    lines = [
        "# Exia 4h sparse-event entry screen v1",
        "",
        f"Verdict: `{report['verdict']}`.",
        "",
        "All events use one sealed 4h ATR/progress position lifecycle. A zero event does not close a position.",
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
        "| Event | Window | Events | Trades | Mean | Median | LCB | Family LCB | P01 | P01 vs control | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    def fmt(value: Any) -> str:
        return "n/a" if value is None or pd.isna(value) else f"{float(value):.2f}"

    for row in metrics.sort_values(["window", "event_id"], kind="stable").itertuples():
        lines.append(
            f"| {row.event_id} | {row.window} | {row.candidate_events} | {row.closed_trades} | "
            f"{fmt(row.mean_stress_net_bps)} | {fmt(row.median_stress_net_bps)} | "
            f"{fmt(row.stress_lcb_bps)} | {fmt(row.familywise_stress_lcb_bps)} | "
            f"{fmt(row.p01_stress_net_bps)} | {fmt(row.p01_improvement_vs_control_bps)} | "
            f"{'PASS' if row.gate_pass else 'CONTROL' if row.comparison_only else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "Gate: >=20 trades, positive stress mean/median/ordinary LCB/family-wise LCB and >=100 bps P01 improvement versus the event control.",
            "Detailed trades, symbols, exit reasons and market regimes are stored in separate Parquet tables.",
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
    family = load_json(family_path)
    exit_family, entry_family, position_policy = validate_event_family(family)
    frame = _load_panel(entry_family)
    windows = _window_map(entry_family)
    shared_params = dict(family["shared_signal_params"])
    events = {str(row["event_id"]): row for row in family["events"]}
    signals: dict[str, pd.Series] = {}
    for event_id, event in events.items():
        params = {**shared_params, **dict(event["params"])}
        signals[event_id] = build_signal(
            frame, {"kind": str(event["strategy_kind"]), "params": params}
        )
    regime = build_market_regime(frame, shared_params)
    gate = family["stage_gate"]
    metric_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []
    ledgers: list[pd.DataFrame] = []

    def evaluate(event_ids: list[str], window_name: str) -> list[dict[str, Any]]:
        start, end = windows[window_name]
        ordered_ids = [CONTROL_ID] + [item for item in event_ids if item != CONTROL_ID]
        temporary = [
            _simulate_window(
                frame,
                signals[event_id],
                regime,
                event=events[event_id],
                position_policy=position_policy,
                window_name=window_name,
                start_timestamp=start,
                end_timestamp=end,
                event_family=family,
                entry_family=entry_family,
            )
            for event_id in ordered_ids
        ]
        control = next(item[1] for item in temporary if item[1]["event_id"] == CONTROL_ID)
        control_p01 = float(control["p01_stress_net_bps"])
        current: list[dict[str, Any]] = []
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

    alternatives = [event_id for event_id in events if event_id != CONTROL_ID]
    development = evaluate(alternatives, "development")
    validation_ids = _rank_passed(development, int(gate["development_to_validation_limit"]))
    validation = evaluate(validation_ids, "validation") if validation_ids else []
    oos_ids = _rank_passed(validation, int(gate["validation_to_oos_limit"]))
    oos = evaluate(oos_ids, "oos") if oos_ids else []
    sanity_ids = _rank_passed(oos, int(gate["oos_to_sanity_limit"]))
    sanity = evaluate(sanity_ids, "sanity") if sanity_ids else []
    final_passed = _rank_passed(sanity, 1)

    metrics = pd.DataFrame(metric_rows)
    trades = pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()
    exit_reason_metrics = pd.DataFrame(slice_rows)
    symbol_metrics = _group_metrics(trades, "symbol", event_family=family)
    regime_metrics = _group_metrics(trades, "market_regime", event_family=family)
    verdict = "SPARSE_EVENT_PASSED_SANITY" if final_passed else "NO_SPARSE_EVENT_PASSED_STAGED_GATE"
    report = {
        "schema_version": "exia.4h_sparse_event_screen_report.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "family_id": family["family_id"],
        "family_sha256": sha256_file(family_path),
        "exit_family_sha256": family["position_policy_source"]["family_sha256"],
        "entry_family_sha256": exit_family["entry_source"]["family_sha256"],
        "source_dataset_sha256": entry_family["source"]["dataset_sha256"],
        "fixed_position_policy": family["position_policy_source"]["policy_id"],
        "entry_mode": "event",
        "profile_id": family["profile_id"],
        "timeframe_minutes": 240,
        "verdict": verdict,
        "event_count": len(events),
        "stage_progression": {
            "validation": validation_ids,
            "oos": oos_ids,
            "sanity": sanity_ids,
        },
        "final_passed": final_passed,
        "evaluated_configurations": len(metrics),
        "safety": dict(family["safety"]),
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_frames = {
        "metrics": metrics,
        "trades": trades,
        "exit_reason_metrics": exit_reason_metrics,
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
    parser = argparse.ArgumentParser(description="Run staged Exia 4h sparse-event screen")
    parser.add_argument("--family", type=Path, default=DEFAULT_FAMILY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args.family, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
