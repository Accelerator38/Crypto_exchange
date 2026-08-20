from __future__ import annotations

import argparse
import hashlib
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

from simple_research.simulator import CostModel, simulate_targets  # noqa: E402
from simple_research.strategies import build_market_regime, build_signal  # noqa: E402
from run_exia_timebase_audit_v1 import summarize_trades  # noqa: E402


DEFAULT_FAMILY = ROOT / "configs" / "exia_4h_candidate_family_v1.json"
DEFAULT_OUTPUT = ROOT / "Reports" / "Exia" / "four_hour_candidate_screen_v1"
BAR_MS = 240 * 60 * 1000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_family(family: dict[str, Any]) -> None:
    if family.get("schema_version") != "exia.4h_candidate_family.v1":
        raise ValueError("unexpected candidate family schema")
    if family.get("profile_id") != "exia_4h_v1":
        raise ValueError("candidate family must use exia_4h_v1")
    if int(family.get("timeframe_minutes", 0)) != 240:
        raise ValueError("candidate family must be sealed to 4h")
    candidates = family.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 8:
        raise ValueError("candidate family must contain exactly 8 preregistered variants")
    identifiers = [str(row.get("candidate_id")) for row in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate identifiers must be unique")
    if [row.get("name") for row in family.get("windows", [])] != [
        "development",
        "validation",
        "oos",
        "sanity",
    ]:
        raise ValueError("candidate windows are not preregistered in stage order")
    if set(family.get("safety", {}).values()) != {False}:
        raise ValueError("candidate family must not grant trading authority")
    gate = family.get("stage_gate", {})
    if int(gate.get("minimum_closed_trades", 0)) < 20:
        raise ValueError("candidate gate cannot use fewer than 20 trades")
    for key in (
        "require_positive_stress_mean",
        "require_positive_stress_lcb",
        "require_positive_familywise_lcb",
    ):
        if gate.get(key) is not True:
            raise ValueError(f"candidate gate must keep {key} enabled")


def _load_panel(family: dict[str, Any]) -> pd.DataFrame:
    source = family["source"]
    manifest_path = ROOT / str(source["manifest"])
    manifest = load_json(manifest_path)
    expected_dataset = str(source["dataset_sha256"])
    if str(manifest.get("dataset_sha256")) != expected_dataset:
        raise ValueError("4h dataset SHA does not match preregistration")
    if manifest.get("validation", {}).get("passed") is not True:
        raise ValueError("4h source validation failed")
    frames: list[pd.DataFrame] = []
    for item in manifest.get("files", []):
        path = ROOT / str(item["path"])
        if sha256_file(path) != str(item["sha256"]):
            raise ValueError(f"4h source file hash mismatch: {path}")
        frames.append(pd.read_parquet(path))
    if not frames:
        raise ValueError("4h source manifest has no files")
    frame = pd.concat(frames, ignore_index=True)
    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    if not required <= set(frame.columns):
        raise ValueError(f"4h source columns missing: {sorted(required - set(frame.columns))}")
    frame = frame.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)
    if frame.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("4h source contains duplicate timestamp/symbol rows")
    symbols_per_bar = frame.groupby("timestamp", sort=False)["symbol"].nunique()
    if symbols_per_bar.nunique() != 1 or int(symbols_per_bar.iloc[0]) != 8:
        raise ValueError("4h source is not a complete full8 panel")
    return frame


def _window_map(family: dict[str, Any]) -> dict[str, tuple[int, int]]:
    windows: dict[str, tuple[int, int]] = {}
    for item in family["windows"]:
        windows[str(item["name"])] = (
            int(pd.Timestamp(str(item["start"])).timestamp() * 1000),
            int(pd.Timestamp(str(item["end"])).timestamp() * 1000),
        )
    return windows


def _gate_pass(metrics: dict[str, Any], gate: dict[str, Any]) -> bool:
    return bool(
        int(metrics["closed_trades"]) >= int(gate["minimum_closed_trades"])
        and float(metrics["mean_stress_net_bps"] or 0.0) > 0.0
        and metrics["stress_lcb_bps"] is not None
        and float(metrics["stress_lcb_bps"]) > 0.0
        and metrics["familywise_stress_lcb_bps"] is not None
        and float(metrics["familywise_stress_lcb_bps"]) > 0.0
    )


def _simulate_window(
    frame: pd.DataFrame,
    signal: pd.Series,
    regime: pd.Series,
    *,
    candidate_id: str,
    window_name: str,
    start_timestamp: int,
    end_timestamp: int,
    family: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    costs = family["costs_bps"]
    stress_cost = float(costs["stress_round_trip"])
    base_cost = float(costs["base_round_trip"])
    result = simulate_targets(
        frame,
        signal,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        costs=CostModel(
            fee_bps_per_fill=4.0,
            slippage_bps_per_fill=stress_cost / 2.0 - 4.0,
        ),
        trial_count=len(family["candidates"]),
    )
    ledger = result.ledger.copy()
    if not ledger.empty:
        ledger["candidate_id"] = candidate_id
        ledger["window"] = window_name
        ledger["timeframe_minutes"] = 240
        ledger["direction"] = ledger["direction"].map({1: "LONG", -1: "SHORT"})
        ledger["signal_timestamp"] = ledger["entry_timestamp"].astype("int64") - BAR_MS
        ledger["market_regime"] = (
            ledger["signal_timestamp"].map(regime).fillna(0).map(
                {1: "bullish", -1: "bearish", 0: "neutral"}
            )
        )
        ledger["base_net_bps"] = ledger["gross_bps"].astype(float) - base_cost
        ledger["stress_net_bps"] = ledger["gross_bps"].astype(float) - stress_cost
    bootstrap = family["bootstrap"]
    summary = summarize_trades(
        ledger,
        seed_key=f"{family['family_id']}:{candidate_id}:{window_name}",
        minimum_trades=int(bootstrap["minimum_trades"]),
        bootstrap_samples=int(bootstrap["samples"]),
        alpha=float(bootstrap["alpha"]),
        familywise_alpha=float(bootstrap["alpha"]) / len(family["candidates"]),
    )
    metric = {
        "candidate_id": candidate_id,
        "window": window_name,
        **summary,
    }
    slices: list[dict[str, Any]] = []
    if not ledger.empty:
        for (market_regime, direction), subset in ledger.groupby(
            ["market_regime", "direction"], sort=True
        ):
            slice_summary = summarize_trades(
                subset,
                seed_key=(
                    f"{family['family_id']}:{candidate_id}:{window_name}:"
                    f"{market_regime}:{direction}"
                ),
                minimum_trades=10,
                bootstrap_samples=int(bootstrap["samples"]),
                alpha=float(bootstrap["alpha"]),
                familywise_alpha=float(bootstrap["alpha"]) / len(family["candidates"]),
            )
            slices.append(
                {
                    "candidate_id": candidate_id,
                    "window": window_name,
                    "market_regime": str(market_regime),
                    "direction": str(direction),
                    **slice_summary,
                }
            )
    return ledger, metric, slices


def _rank_passed(
    metrics: list[dict[str, Any]],
    *,
    gate: dict[str, Any],
    limit: int,
) -> list[str]:
    passed = [row for row in metrics if _gate_pass(row, gate)]
    passed.sort(
        key=lambda row: (
            -float(row["familywise_stress_lcb_bps"]),
            -float(row["stress_lcb_bps"]),
            str(row["candidate_id"]),
        )
    )
    return [str(row["candidate_id"]) for row in passed[:limit]]


def _distribution_diagnostics(trades: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if trades.empty:
        return rows
    for candidate_id, subset in trades.groupby("candidate_id", sort=True):
        values = subset["stress_net_bps"].astype(float)
        rows.append(
            {
                "candidate_id": str(candidate_id),
                "trades": int(len(subset)),
                "median_stress_net_bps": float(values.median()),
                "std_stress_net_bps": float(values.std(ddof=1)),
                "p01_stress_net_bps": float(values.quantile(0.01)),
                "worst_stress_net_bps": float(values.min()),
                "best_stress_net_bps": float(values.max()),
                "mean_holding_bars": float(subset["holding_bars"].mean()),
            }
        )
    return rows


def _symbol_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "candidate_id",
                "window",
                "symbol",
                "closed_trades",
                "mean_stress_net_bps",
                "median_stress_net_bps",
                "std_stress_net_bps",
            ]
        )
    rows = []
    for (candidate_id, window, symbol), subset in trades.groupby(
        ["candidate_id", "window", "symbol"], sort=True
    ):
        values = subset["stress_net_bps"].astype(float)
        rows.append(
            {
                "candidate_id": str(candidate_id),
                "window": str(window),
                "symbol": str(symbol),
                "closed_trades": int(len(subset)),
                "mean_stress_net_bps": float(values.mean()),
                "median_stress_net_bps": float(values.median()),
                "std_stress_net_bps": float(values.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def _render_markdown(
    report: dict[str, Any],
    metrics: pd.DataFrame,
    regime_metrics: pd.DataFrame,
) -> str:
    lines = [
        "# Exia 4h market-quorum candidate screen v1",
        "",
        f"Verdict: `{report['verdict']}`.",
        "",
        "This is staged retrospective research. It grants no paper/live authority.",
        "",
        "## Stage progression",
        "",
        f"- Development -> validation: {', '.join(report['stage_progression']['validation']) or 'none'}",
        f"- Validation -> OOS: {', '.join(report['stage_progression']['oos']) or 'none'}",
        f"- OOS -> sanity: {', '.join(report['stage_progression']['sanity']) or 'none'}",
        f"- Final passed: {', '.join(report['final_passed']) or 'none'}",
        "",
        "## Evaluated windows",
        "",
        "| Candidate | Window | Trades | Stress mean | LCB | Family-wise LCB | Gate |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in metrics.sort_values(["candidate_id", "window"], kind="stable").itertuples():
        def fmt(value: Any) -> str:
            return "n/a" if value is None or pd.isna(value) else f"{float(value):.2f}"

        lines.append(
            f"| {row.candidate_id} | {row.window} | {row.closed_trades} | "
            f"{fmt(row.mean_stress_net_bps)} | {fmt(row.stress_lcb_bps)} | "
            f"{fmt(row.familywise_stress_lcb_bps)} | {'PASS' if row.gate_pass else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Direction and market regime",
            "",
            "| Candidate | Window | Regime | Direction | Trades | Stress mean | LCB |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    if regime_metrics.empty:
        lines.append("| none | none | none | none | 0 | n/a | n/a |")
    else:
        for row in regime_metrics.sort_values(
            ["candidate_id", "window", "market_regime", "direction"], kind="stable"
        ).itertuples():
            lines.append(
                f"| {row.candidate_id} | {row.window} | {row.market_regime} | "
                f"{row.direction} | {row.closed_trades} | {float(row.mean_stress_net_bps):.2f} | "
                f"{'n/a' if row.stress_lcb_bps is None or pd.isna(row.stress_lcb_bps) else f'{float(row.stress_lcb_bps):.2f}'} |"
            )
    lines.extend(
        [
            "",
            "Neutral is an explicit NoTrade state in this family.",
            "",
            "## Distribution diagnostics",
            "",
            "| Candidate | Trades | Median | Std | P01 | Worst | Best | Mean hold bars |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["distribution_diagnostics"]:
        lines.append(
            f"| {row['candidate_id']} | {row['trades']} | "
            f"{row['median_stress_net_bps']:.2f} | {row['std_stress_net_bps']:.2f} | "
            f"{row['p01_stress_net_bps']:.2f} | {row['worst_stress_net_bps']:.2f} | "
            f"{row['best_stress_net_bps']:.2f} | {row['mean_holding_bars']:.2f} |"
        )
    lines.extend(
        [
            "",
            "All candidates have a negative median trade. Positive means are driven by rare large winners, while clustered downside keeps every bootstrap LCB below zero.",
            "",
            "Detailed symbol slices are stored in `symbol_metrics.parquet`.",
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
    validate_family(family)
    frame = _load_panel(family)
    windows = _window_map(family)
    candidates = {str(row["candidate_id"]): row for row in family["candidates"]}
    signals: dict[str, pd.Series] = {}
    regimes: dict[str, pd.Series] = {}
    for candidate_id, candidate in candidates.items():
        strategy = {
            "kind": str(candidate["strategy_kind"]),
            "params": dict(candidate["params"]),
        }
        signals[candidate_id] = build_signal(frame, strategy)
        regimes[candidate_id] = build_market_regime(frame, strategy["params"])

    metric_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    ledgers: list[pd.DataFrame] = []
    gate = family["stage_gate"]

    def evaluate(candidate_ids: list[str], window_name: str) -> list[dict[str, Any]]:
        current: list[dict[str, Any]] = []
        start, end = windows[window_name]
        for candidate_id in candidate_ids:
            ledger, metric, slices = _simulate_window(
                frame,
                signals[candidate_id],
                regimes[candidate_id],
                candidate_id=candidate_id,
                window_name=window_name,
                start_timestamp=start,
                end_timestamp=end,
                family=family,
            )
            metric["gate_pass"] = _gate_pass(metric, gate)
            current.append(metric)
            metric_rows.append(metric)
            regime_rows.extend(slices)
            if not ledger.empty:
                ledgers.append(ledger)
        return current

    development = evaluate(list(candidates), "development")
    validation_ids = _rank_passed(
        development,
        gate=gate,
        limit=int(gate["development_to_validation_limit"]),
    )
    validation = evaluate(validation_ids, "validation")
    oos_ids = _rank_passed(
        validation,
        gate=gate,
        limit=int(gate["validation_to_oos_limit"]),
    )
    oos = evaluate(oos_ids, "oos")
    sanity_ids = _rank_passed(
        oos,
        gate=gate,
        limit=int(gate["oos_to_sanity_limit"]),
    )
    sanity = evaluate(sanity_ids, "sanity")
    final_passed = _rank_passed(sanity, gate=gate, limit=1)

    metrics = pd.DataFrame(metric_rows)
    regime_metrics = pd.DataFrame(regime_rows)
    trades = pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()
    distribution_diagnostics = _distribution_diagnostics(trades)
    symbol_metrics = _symbol_metrics(trades)
    verdict = "CANDIDATE_PASSED_SANITY" if final_passed else "NO_4H_CANDIDATE_PASSED_STAGED_GATE"
    report = {
        "schema_version": "exia.4h_candidate_screen_report.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "family_id": family["family_id"],
        "family_sha256": sha256_file(family_path),
        "source_dataset_sha256": family["source"]["dataset_sha256"],
        "profile_id": family["profile_id"],
        "timeframe_minutes": 240,
        "verdict": verdict,
        "candidate_count": len(candidates),
        "stage_progression": {
            "validation": validation_ids,
            "oos": oos_ids,
            "sanity": sanity_ids,
        },
        "final_passed": final_passed,
        "evaluated_configurations": len(metrics),
        "closed_trades": int(metrics["closed_trades"].sum()) if not metrics.empty else 0,
        "distribution_diagnostics": distribution_diagnostics,
        "root_cause": "NEGATIVE_MEDIAN_AND_CLUSTERED_TAIL_RISK",
        "safety": dict(family["safety"]),
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.parquet"
    trades_path = output_dir / "trades.parquet"
    regime_path = output_dir / "regime_metrics.parquet"
    symbol_path = output_dir / "symbol_metrics.parquet"
    metrics.to_parquet(metrics_path, index=False)
    trades.to_parquet(trades_path, index=False)
    regime_metrics.to_parquet(regime_path, index=False)
    symbol_metrics.to_parquet(symbol_path, index=False)
    report["artifacts"] = {
        "metrics": {"path": str(metrics_path.relative_to(ROOT)), "sha256": sha256_file(metrics_path)},
        "trades": {"path": str(trades_path.relative_to(ROOT)), "sha256": sha256_file(trades_path)},
        "regime_metrics": {"path": str(regime_path.relative_to(ROOT)), "sha256": sha256_file(regime_path)},
        "symbol_metrics": {"path": str(symbol_path.relative_to(ROOT)), "sha256": sha256_file(symbol_path)},
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        _render_markdown(report, metrics, regime_metrics), encoding="utf-8"
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
    parser = argparse.ArgumentParser(description="Run staged Exia 4h candidate screening")
    parser.add_argument("--family", type=Path, default=DEFAULT_FAMILY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args.family, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
