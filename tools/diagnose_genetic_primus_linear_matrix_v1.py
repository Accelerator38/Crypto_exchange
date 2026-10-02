"""Descriptive attribution of the frozen June/July linear matrix only."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exia.genetic_primus.accounting_v2 import simulate_cash_targets
from exia.genetic_primus.measurement_lab_v2 import object_sha
from exia.genetic_primus.prospective_contract_v1 import file_sha256
from exia.genetic_primus.prospective_manifest_v1 import load_verified_snapshot
from exia.genetic_primus.rolling_linear_matrix_v1 import build_matrix_examples
from exia.genetic_primus.rolling_linear_retrain_v1 import DT, HORIZON_MS

RESULT = ROOT / "Reports/Exia/Genetic_Primus/rolling_linear_matrix_v1_20261002/result.json"
RESULT_SHA = "99418ebacb37ebc66ed139746543215809464612b169af8af6a33514b46a44ad"
CONFIG = ROOT / "configs/genetic_primus_rolling_linear_matrix_v1.json"
OUTPUT = ROOT / "Reports/Exia/Genetic_Primus/linear_matrix_diagnostic_v1_20261002/result.json"


def _ms(value: str) -> int:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(None):
        raise ValueError("UTC required")
    return int(timestamp.timestamp() * 1000)


def _f(value: float) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _mean(values: pd.Series) -> float | None:
    return _f(values.mean()) if len(values) else None


def _row_summary(rows: pd.DataFrame) -> dict:
    actual = rows.forward_gross_3d
    return {"count": int(len(rows)), "positive_count": int(actual.gt(0).sum()),
            "mean_gross_3d": _mean(actual), "median_gross_3d": _f(actual.median()) if len(rows) else None}


def _schedule(ledger: list[dict], symbols: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for item in ledger:
        origin = int(item["origin_start"])
        decisions = item["decision"]["decisions"]
        if set(decisions) != set(symbols):
            raise ValueError("incomplete frozen decision universe")
        if item["decision"]["model_sha256"] != item["model"]["model_sha256"]:
            raise ValueError("model/decision SHA mismatch")
        frame = pd.MultiIndex.from_product(
            [np.arange(origin, origin + HORIZON_MS, DT), symbols],
            names=["execution_timestamp", "symbol"]).to_frame(index=False)
        frame["signal_timestamp"] = origin
        frame["target"] = frame.symbol.map({s: int(decisions[s]["target"]) for s in symbols})
        frame["model_sha256"] = item["model"]["model_sha256"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _decision_rows(ledger: list[dict], opportunities: pd.DataFrame) -> pd.DataFrame:
    decisions = []
    for item in ledger:
        for symbol, decision in item["decision"]["decisions"].items():
            decisions.append({"origin_start": int(item["origin_start"]), "symbol": symbol,
                              "target": int(decision["target"]),
                              "position_before": int(decision["position_before"]),
                              "forecast": decision["predicted_gross_3d"],
                              "threshold": float(decision["incremental_threshold"])})
    rows = opportunities.merge(pd.DataFrame(decisions), on=["origin_start", "symbol"],
                               how="left", validate="one_to_one")
    if len(rows) != len(decisions) or rows.target.isna().any():
        raise ValueError("frozen decisions do not match origin labels")
    if not rows.forecast.notna().equals(rows.proposal_long.astype(bool)):
        raise ValueError("forecast availability differs from base proposal")
    return rows


def _forecast_summary(rows: pd.DataFrame) -> dict:
    proposal = rows.loc[rows.proposal_long].copy()
    if not len(proposal):
        raise ValueError("empty base opportunity set")
    proposal["error"] = proposal.forecast - proposal.forward_gross_3d
    forecast = proposal.forecast.to_numpy(float)
    actual = proposal.forward_gross_3d.to_numpy(float)
    correlation = (float(np.corrcoef(forecast, actual)[0, 1])
                   if len(proposal) > 2 and np.std(forecast) > 0 and np.std(actual) > 0 else None)
    threshold_margin = proposal.forecast - proposal.threshold
    active = proposal.loc[proposal.target.eq(1)]
    abstained = proposal.loc[proposal.target.eq(0)]
    sorted_proposals = proposal.sort_values(["forecast", "origin_start", "symbol"])
    thirds = np.array_split(np.arange(len(sorted_proposals)), 3)
    return {"all_proposals": _row_summary(proposal),
            "active": _row_summary(active), "abstained": _row_summary(abstained),
            "forecast_mean": _mean(proposal.forecast),
            "forecast_min": _f(proposal.forecast.min()),
            "forecast_max": _f(proposal.forecast.max()),
            "mean_absolute_forecast_error": _mean(proposal.error.abs()),
            "forecast_actual_pearson_descriptive_only": correlation,
            "mean_margin_over_incremental_threshold": _mean(threshold_margin),
            "forecast_rank_terciles": [
                {"rank": i + 1, "forecast_mean": _mean(sorted_proposals.iloc[indices].forecast),
                 **_row_summary(sorted_proposals.iloc[indices])}
                for i, indices in enumerate(thirds)],
            "held_target_count": int(rows.position_before.eq(1).sum()),
            "new_entry_count": int((rows.position_before.eq(0) & rows.target.eq(1)).sum()),
            "exit_count": int((rows.position_before.eq(1) & rows.target.eq(0)).sum())}


def _trade_summary(sim, symbols: tuple[str, ...], expected_net: float) -> dict:
    trades = sim.trades
    if abs(sim.metrics["net_return"] - expected_net) > 1e-10:
        raise AssertionError("replayed accounting differs from frozen result")
    if not trades.empty and abs(float(trades.net_pnl.sum()) - expected_net) > 1e-9:
        raise AssertionError("trade PnL does not reconcile to portfolio")
    by_symbol = {}
    for symbol in symbols:
        subset = trades.loc[trades.symbol.eq(symbol)] if not trades.empty else trades
        by_symbol[symbol] = {"closed_trades": int(len(subset)),
                             "net_pnl": float(subset.net_pnl.sum()) if len(subset) else 0.0,
                             "gross_pnl": float(subset.gross_pnl.sum()) if len(subset) else 0.0,
                             "fees": float((subset.entry_fee + subset.exit_fee).sum()) if len(subset) else 0.0,
                             "mean_holding_days": _f((subset.exit_timestamp - subset.entry_timestamp).mean() / 86_400_000) if len(subset) else None}
    durations = ((trades.exit_timestamp - trades.entry_timestamp) / 86_400_000
                 if not trades.empty else pd.Series(dtype=float))
    return {"net_return": float(sim.metrics["net_return"]),
            "closed_trades": int(len(trades)),
            "mean_holding_days": _mean(durations),
            "median_holding_days": _f(durations.median()) if len(durations) else None,
            "maximum_holding_days": _f(durations.max()) if len(durations) else None,
            "terminal_exit_count": int(trades.exit_reason.eq("interval_end").sum()) if len(trades) else 0,
            "by_symbol": by_symbol}


def _evaluate() -> dict:
    if Path(sys.executable).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve():
        raise ValueError("project Python required")
    if file_sha256(RESULT) != RESULT_SHA:
        raise ValueError("frozen result changed")
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    c = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (result["config_sha256"] != file_sha256(CONFIG)
            or result["selected_candidate"] is not None
            or result["validation_result"] is not None
            or result["final_policy"] != "NoTrade"):
        raise ValueError("diagnostic is scoped to failed frozen training result")
    manifest_path = ROOT / c["source_manifest_path"]
    if file_sha256(manifest_path) != c["source_manifest_sha256"]:
        raise ValueError("manifest changed")
    contract_path = ROOT / c["source_contract_path"]
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    manifest, panel = load_verified_snapshot(
        manifest_path, root=ROOT, contract_path=contract_path, contract=contract)
    if manifest["snapshot_sha256"] != result["source_snapshot_sha256"]:
        raise ValueError("snapshot changed")
    symbols = tuple(contract["universe"])
    first = _ms(c["development_start_utc"])
    end = first + c["training_origin_count"] * HORIZON_MS
    examples = build_matrix_examples(panel, symbols, _ms(c["history_start_utc"]), first, end)
    opportunities = examples.loc[examples.origin_start.ge(first)].copy()
    if (len(opportunities) != c["training_origin_count"] * len(symbols)
            or opportunities.origin_end_exclusive.max() != end):
        raise ValueError("training-only origin bounds changed")
    reference = {"all_decisions": int(len(opportunities)),
                 "ema_proposals": _row_summary(opportunities.loc[opportunities.proposal_long]),
                 "no_proposal": _row_summary(opportunities.loc[~opportunities.proposal_long])}
    atr_median = float(opportunities.atr_14_pct.median())
    reference["atr_median_from_train_decisions"] = atr_median
    reference["ema_proposals_low_atr"] = _row_summary(opportunities.loc[
        opportunities.proposal_long & opportunities.atr_14_pct.le(atr_median)])
    reference["ema_proposals_high_atr"] = _row_summary(opportunities.loc[
        opportunities.proposal_long & opportunities.atr_14_pct.gt(atr_median)])
    diagnostics = []
    for frozen in result["records"]:
        ledger = frozen["origin_models"]
        if len(ledger) != c["training_origin_count"]:
            raise ValueError("unexpected number of frozen origin models")
        rows = _decision_rows(ledger, opportunities)
        schedule = _schedule(ledger, symbols)
        sim = simulate_cash_targets(
            panel, schedule, symbols=symbols, start_timestamp=first,
            end_timestamp=end, timeframe_ms=DT,
            round_trip_cost_bps=c["normal_round_trip_bps"], record_details=True)
        expected = frozen["evaluations"]["normal"]
        if (int(sim.metrics["trades"]) != expected["trades"]
                or abs(sim.metrics["total_fees"] - expected["fees"]) > 1e-10):
            raise AssertionError("replayed trades/fees differ from frozen result")
        symbol_opportunities = {}
        for symbol in symbols:
            own = rows.loc[rows.symbol.eq(symbol)]
            symbol_opportunities[symbol] = {
                "proposals": int(own.proposal_long.sum()),
                "active_origins": int(own.target.sum()),
                "proposal_outcomes": _row_summary(own.loc[own.proposal_long]),
                "active_outcomes": _row_summary(own.loc[own.target.eq(1)])}
        diagnostics.append({"feature_set": frozen["feature_set"], "variant": frozen["variant"],
                            "decisions_sha256": object_sha(
                                rows[["origin_start", "symbol", "target", "forecast",
                                      "forward_gross_3d"]].astype(object)
                                .where(lambda frame: frame.notna(), None).to_dict("records")),
                            "forecast": _forecast_summary(rows),
                            "origin_target_exposure_fraction": float(rows.target.mean()),
                            "symbol_opportunities": symbol_opportunities,
                            "trades": _trade_summary(sim, symbols, expected["net_return"])})
    return {"schema_version": "exia.genetic_primus.linear_matrix_diagnostic/1",
            "evidence_class": "exploratory_attribution_of_revealed_train_only",
            "frozen_result_sha256": RESULT_SHA, "script_sha256": file_sha256(Path(__file__)),
            "source_snapshot_sha256": result["source_snapshot_sha256"],
            "train_start_ms": first, "train_end_exclusive_ms": end,
            "validation_labels_used": False, "new_model_fits": 0,
            "diagnostic_reference": reference, "diagnostics": diagnostics,
            "promotion_authority": False, "network_allowed": False}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("immutable diagnostic result already exists")
    result = _evaluate()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "FROZEN_TRAIN_DIAGNOSTIC_COMPLETE",
                      "proposals": result["diagnostic_reference"]["ema_proposals"]["count"],
                      "validation_labels_used": False, "new_model_fits": 0},
                     separators=(",", ":")))


if __name__ == "__main__":
    main()
