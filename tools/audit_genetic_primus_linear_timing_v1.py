"""Frozen train-only audit of EMA feature, label and cash-fill timestamps."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from exia.genetic_primus.accounting_v2 import simulate_cash_targets
from exia.genetic_primus.prospective_contract_v1 import file_sha256
from exia.genetic_primus.prospective_manifest_v1 import load_verified_snapshot
from exia.genetic_primus.rolling_linear_matrix_v1 import build_matrix_examples
from exia.genetic_primus.rolling_linear_retrain_v1 import DT, HORIZON_MS
from tools.diagnose_genetic_primus_linear_matrix_v1 import _schedule

SOURCE = ROOT / "Reports/Exia/Genetic_Primus/rolling_linear_matrix_v1_20261002/result.json"
SOURCE_SHA = "99418ebacb37ebc66ed139746543215809464612b169af8af6a33514b46a44ad"
CONFIG = ROOT / "configs/genetic_primus_rolling_linear_matrix_v1.json"
OUTPUT = ROOT / "Reports/Exia/Genetic_Primus/linear_timing_audit_v1_20261002/result.json"


def _ms(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(None):
        raise ValueError("UTC timestamp required")
    return int(parsed.timestamp() * 1000)


def _iso(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, timezone.utc).isoformat().replace("+00:00", "Z")


def _manual_features(group: pd.DataFrame) -> dict[int, dict]:
    """Separate simple recurrence, without the production feature builder."""
    ordered = group.sort_values("timestamp", kind="stable")
    times = ordered.timestamp.to_numpy(np.int64)
    closes = ordered.close.to_numpy(float)
    e12 = np.empty(len(closes), dtype=float)
    e48 = np.empty(len(closes), dtype=float)
    e12[0] = e48[0] = closes[0]
    for index in range(1, len(closes)):
        e12[index] = (2.0 / 13.0) * closes[index] + (11.0 / 13.0) * e12[index - 1]
        e48[index] = (2.0 / 49.0) * closes[index] + (47.0 / 49.0) * e48[index - 1]
    values = {}
    for index in range(12, len(closes)):
        path = np.abs(np.diff(closes[index - 12:index + 1])).sum()
        values[int(times[index])] = {
            "ema_gap": float(e12[index] / e48[index] - 1.0),
            "ema_slope": float(e12[index] / e12[index - 3] - 1.0),
            "return_12": float(closes[index] / closes[index - 12] - 1.0),
            "efficiency_12": float(abs(closes[index] - closes[index - 12]) / path)
            if path > 0 else float("nan"),
        }
    return values


def _require_close(left: float, right: float, name: str) -> float:
    delta = abs(float(left) - float(right))
    if not np.isclose(left, right, rtol=1e-10, atol=1e-12):
        raise AssertionError(f"{name} mismatch: {left} vs {right}")
    return delta


def _market_lookup(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("duplicate market candle")
    return panel.set_index(["timestamp", "symbol"]).sort_index()


def _check_fills(panel: pd.DataFrame, market: pd.DataFrame,
                 source: dict, symbols: tuple[str, ...], start: int, end: int) -> dict:
    checked = []
    for feature_set in ("two", "four"):
        frozen = next(item for item in source["records"]
                      if item["feature_set"] == feature_set and item["variant"] == "uniform")
        schedule = _schedule(frozen["origin_models"], symbols)
        sim = simulate_cash_targets(
            panel, schedule, symbols=symbols, start_timestamp=start,
            end_timestamp=end, timeframe_ms=DT, round_trip_cost_bps=16.0,
            record_details=True)
        if abs(sim.metrics["net_return"] - frozen["evaluations"]["normal"]["net_return"]) > 1e-10:
            raise AssertionError("cash replay does not match frozen PnL")
        exit_reasons = {}
        held_across_origins = 0
        for row in sim.trades.itertuples(index=False):
            entry = int(row.entry_timestamp)
            exit_time = int(row.exit_timestamp)
            if entry % HORIZON_MS != start % HORIZON_MS or row.signal_timestamp != entry:
                raise AssertionError("fill is not at signalled origin boundary")
            _require_close(row.entry_price, market.loc[(entry, row.symbol), "open"], "entry open")
            if row.exit_reason == "target_change":
                _require_close(row.exit_price, market.loc[(exit_time, row.symbol), "open"], "exit open")
            elif row.exit_reason == "interval_end":
                if exit_time != end:
                    raise AssertionError("unexpected terminal close")
                _require_close(row.exit_price, market.loc[(end - DT, row.symbol), "close"], "terminal close")
            else:
                raise AssertionError("unregistered exit reason")
            exit_reasons[row.exit_reason] = exit_reasons.get(row.exit_reason, 0) + 1
            held_across_origins += int(exit_time - entry > HORIZON_MS)
        checked.append({"feature_set": feature_set, "closed_trades": int(len(sim.trades)),
                        "entry_at_origin_open_count": int(len(sim.trades)),
                        "same_boundary_signal_and_entry_count": int(len(sim.trades)),
                        "exit_reasons": exit_reasons,
                        "held_longer_than_one_origin_count": held_across_origins,
                        "net_reconciled": True})
    return {item["feature_set"]: item for item in checked}


def _evaluate() -> dict:
    if Path(sys.executable).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve():
        raise ValueError("project Python required")
    if file_sha256(SOURCE) != SOURCE_SHA:
        raise ValueError("frozen source changed")
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    c = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (source["config_sha256"] != file_sha256(CONFIG)
            or source["validation_result"] is not None
            or source["selected_candidate"] is not None):
        raise ValueError("not the frozen failed train attempt")
    manifest_path = ROOT / c["source_manifest_path"]
    if file_sha256(manifest_path) != c["source_manifest_sha256"]:
        raise ValueError("manifest changed")
    contract_path = ROOT / c["source_contract_path"]
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    manifest, panel = load_verified_snapshot(
        manifest_path, root=ROOT, contract_path=contract_path, contract=contract)
    if manifest["snapshot_sha256"] != source["source_snapshot_sha256"]:
        raise ValueError("snapshot changed")
    symbols = tuple(contract["universe"])
    start = _ms(c["development_start_utc"])
    end = start + c["training_origin_count"] * HORIZON_MS
    examples = build_matrix_examples(panel, symbols, _ms(c["history_start_utc"]), start, end)
    origin_examples = examples.loc[examples.origin_start.ge(start)].set_index(["origin_start", "symbol"])
    frozen = next(item for item in source["records"]
                  if item["feature_set"] == "two" and item["variant"] == "uniform")
    proposal_origins = [int(item["origin_start"]) for item in frozen["origin_models"]
                        if any(d["predicted_gross_3d"] is not None
                               for d in item["decision"]["decisions"].values())]
    if len(proposal_origins) != 7 or max(proposal_origins) + HORIZON_MS >= end:
        raise ValueError("unexpected disclosed proposal origins")
    market = _market_lookup(panel.loc[panel.timestamp.lt(end)])
    manual = {symbol: _manual_features(panel.loc[
        panel.symbol.eq(symbol) & panel.timestamp.lt(end)]) for symbol in symbols}
    audit_rows = []
    discrepancies = {key: 0.0 for key in ("return_12", "efficiency_12", "gross_label")}
    for origin in proposal_origins:
        ledger = next(item for item in frozen["origin_models"] if item["origin_start"] == origin)
        last_label_end = int(ledger["model"]["max_train_label_end_exclusive"])
        if last_label_end > origin - c["purge_bars"] * DT:
            raise AssertionError("purge violated")
        for symbol in symbols:
            previous = origin - DT
            last = origin + HORIZON_MS - DT
            produced = origin_examples.loc[(origin, symbol)]
            calculated = manual[symbol][previous]
            proposal = (calculated["ema_gap"] > 0 and calculated["ema_slope"] > 0
                        and calculated["return_12"] > 0 and calculated["efficiency_12"] >= 0.30)
            if bool(produced.proposal_long) != proposal:
                raise AssertionError("EMA proposal differs from manual past-only calculation")
            for name in ("return_12", "efficiency_12"):
                discrepancies[name] = max(discrepancies[name],
                                          _require_close(produced[name], calculated[name], name))
            open_price = float(market.loc[(origin, symbol), "open"])
            last_close = float(market.loc[(last, symbol), "close"])
            gross = last_close / open_price - 1.0
            discrepancies["gross_label"] = max(discrepancies["gross_label"],
                _require_close(produced.forward_gross_3d, gross, "gross label"))
            next_open = float(market.loc[(origin + HORIZON_MS, symbol), "open"])
            before_close = float(market.loc[(previous, symbol), "close"])
            frozen_proposal = ledger["decision"]["decisions"][symbol]["predicted_gross_3d"] is not None
            if proposal != frozen_proposal:
                raise AssertionError("frozen decision did not use the audited proposal")
            audit_rows.append({"origin_utc": _iso(origin), "symbol": symbol,
                               "proposal": proposal, "prior_bar_timestamp_utc": _iso(previous),
                               "prior_bar_close_available_utc": _iso(origin),
                               "entry_open_timestamp_utc": _iso(origin),
                               "last_label_bar_timestamp_utc": _iso(last),
                               "label_available_utc": _iso(origin + HORIZON_MS),
                               "prior_close_to_entry_open_gap_bps":
                                   (open_price / before_close - 1.0) * 10_000.0,
                               "last_close_to_next_open_gap_bps":
                                   (next_open / last_close - 1.0) * 10_000.0,
                               "manual_forward_gross_3d": gross,
                               "latest_training_label_age_hours":
                                   (origin - last_label_end) / 3_600_000.0})
    selected = [row for row in audit_rows if row["proposal"]]
    if len(selected) != 27:
        raise AssertionError("unexpected number of disclosed base opportunities")
    entry_gaps = [abs(row["prior_close_to_entry_open_gap_bps"]) for row in selected]
    exit_gaps = [abs(row["last_close_to_next_open_gap_bps"]) for row in selected]
    fills = _check_fills(panel, market, source, symbols, start, end)
    return {"schema_version": "exia.genetic_primus.linear_timing_audit/1",
            "evidence_class": "train_only_engineering_measurement_audit",
            "source_result_sha256": SOURCE_SHA,
            "script_sha256": file_sha256(Path(__file__)),
            "source_snapshot_sha256": source["source_snapshot_sha256"],
            "train_end_exclusive_utc": _iso(end),
            "proposal_origin_count": len(proposal_origins),
            "proposal_count": len(selected), "audited_origin_symbol_rows": len(audit_rows),
            "manual_feature_and_label_max_abs_error": discrepancies,
            "max_abs_prior_close_to_entry_open_gap_bps_on_proposals": max(entry_gaps),
            "max_abs_last_close_to_next_open_gap_bps_on_proposals": max(exit_gaps),
            "minimum_latest_training_label_age_hours": min(row["latest_training_label_age_hours"] for row in audit_rows),
            "registered_purge_hours": c["purge_bars"] * DT / 3_600_000.0,
            "zero_latency_boundary_assumption": True,
            "fills": fills, "rows": audit_rows,
            "validation_labels_used": False, "new_model_fits": 0,
            "network_allowed": False, "promotion_authority": False}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("immutable timing audit already exists")
    result = _evaluate()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "TRAIN_TIMING_AUDIT_COMPLETE",
                      "origins": result["proposal_origin_count"],
                      "proposals": result["proposal_count"],
                      "max_entry_gap_bps": round(result["max_abs_prior_close_to_entry_open_gap_bps_on_proposals"], 5),
                      "max_exit_gap_bps": round(result["max_abs_last_close_to_next_open_gap_bps_on_proposals"], 5)},
                     separators=(",", ":")))


if __name__ == "__main__":
    main()
