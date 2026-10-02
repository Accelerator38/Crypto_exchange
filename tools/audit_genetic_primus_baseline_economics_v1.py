"""Immutable offline audit of the already disclosed EMA/GA cash simulations."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exia.genetic_primus.ablation_v1 import summarize_simulation, targets_for_policy
from exia.genetic_primus.accounting_v2 import simulate_cash_targets
from exia.genetic_primus.prospective_contract_v1 import file_sha256, load_and_validate
from exia.genetic_primus.prospective_evaluator_v1 import _frame_sha
from exia.genetic_primus.prospective_manifest_v1 import load_verified_snapshot
from run_genetic_primus_ablation_v1 import _archived_decisions, CONFIG as SOURCE_CONFIG

CONFIG = ROOT / "configs/genetic_primus_baseline_economics_audit_v1.json"
CONFIG_SHA = "a7f552eabb8988cf81820782209ec8ff9755b98b17bca39981fb71c8b1ed3984"
OUTPUT = ROOT / "Reports/Exia/Genetic_Primus/baseline_economics_audit_v1_20261002/result.json"
DT = 14_400_000
HORIZON = 3 * 86_400_000


def _path(relative: str) -> Path:
    path = (ROOT / relative.replace("\\", "/")).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("source path leaves project")
    return path


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _checked_config() -> dict:
    if Path(sys.executable).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve():
        raise ValueError("project Python required")
    if file_sha256(CONFIG) != CONFIG_SHA:
        raise ValueError("audit contract bytes changed")
    config = _json(CONFIG)
    if (config["schema_version"] != "exia.genetic_primus.baseline_economics_audit/1"
            or config["evidence_class"] != "revealed_historical_descriptive_audit_only"
            or any(config["safety"].values())):
        raise ValueError("invalid offline audit contract")
    for key in ("source_ablation_result", "source_manifest"):
        if file_sha256(_path(config[key + "_path"])) != config[key + "_sha256"]:
            raise ValueError(f"{key} bytes changed")
    return config


def _data_integrity(panel: pd.DataFrame, manifest: dict, symbols: tuple[str, ...]) -> dict:
    meta = manifest["metadata"]
    start, end = int(meta["coverage_start"]), int(meta["coverage_end_exclusive"])
    expected_times = np.arange(start, end, DT, dtype=np.int64)
    counts = panel.groupby("timestamp", sort=True).symbol.nunique()
    if (len(panel) != len(expected_times) * len(symbols)
            or not np.array_equal(counts.index.to_numpy(np.int64), expected_times)
            or not counts.eq(len(symbols)).all()
            or panel.duplicated(["timestamp", "symbol"]).any()
            or set(panel.symbol) != set(symbols)):
        raise ValueError("incomplete or duplicate full8 4h grid")
    values = panel[["open", "high", "low", "close", "volume"]].to_numpy(float)
    if (not np.isfinite(values).all() or (values[:, :4] <= 0).any()
            or (values[:, 4] < 0).any()
            or (values[:, 1] < np.maximum(values[:, 0], values[:, 3])).any()
            or (values[:, 2] > np.minimum(values[:, 0], values[:, 3])).any()):
        raise ValueError("invalid OHLCV values")
    return {
        "coverage_start_ms": start, "coverage_end_exclusive_ms": end,
        "timestamp_count": len(expected_times), "rows": len(panel),
        "symbols": list(symbols), "duplicates": 0, "missing_symbol_bars": 0,
        "off_grid_timestamps": int(np.count_nonzero(expected_times % DT)),
        "zero_volume_bars": int(np.count_nonzero(values[:, 4] == 0)),
        "finite_positive_consistent_ohlc": True,
    }


def _source_lineage(manifest: dict) -> dict:
    """Verify local derivative manifests and recorded request metadata."""
    early = []
    endpoint_categories = set()
    verified_chunks = 0
    verified_raw_pages = 0
    for index, item in enumerate(manifest["snapshot_files"]):
        part = _path(item["path"])
        if file_sha256(part) != item["sha256"]:
            raise ValueError("snapshot part changed")
        if index < 4:
            derived_manifest = _json(part.with_suffix(".manifest.json"))
            source_path = _path(derived_manifest["source_manifest"])
            source = _json(source_path)
            if (derived_manifest["source_dataset_sha256"] != source["dataset_sha256"]
                    or not any(_path(row["path"]) == part and row["sha256"] == item["sha256"]
                               for row in derived_manifest["files"])):
                raise ValueError("early derived/source lineage changed")
            early.append({"snapshot_part": item["path"],
                          "derived_manifest_sha256": file_sha256(part.with_suffix(".manifest.json")),
                          "source_manifest": derived_manifest["source_manifest"],
                          "source_manifest_sha256": file_sha256(source_path),
                          "exchange": source.get("exchange"),
                          "market_type": source.get("market_type"),
                          "source_timeframe": source.get("timeframe")})
            continue
        derived_path = part.with_name(part.name.replace("_240m.parquet", "_manifest.json"))
        derived = _json(derived_path)
        if not any(_path(row["path"]) == part and row["sha256"] == item["sha256"]
                   for row in derived["files"]):
            raise ValueError("September derived part lineage changed")
        for chunk_item in derived["chunks"]:
            chunk_path = _path(chunk_item["path"])
            if file_sha256(chunk_path) != chunk_item["sha256"]:
                raise ValueError("recorded chunk changed")
            chunk = _json(chunk_path)
            if chunk["plan_sha256"] != derived["plan_sha256"]:
                raise ValueError("chunk plan changed")
            verified_chunks += 1
            for page_item in chunk["raw_pages"]:
                page_path = _path(page_item["path"])
                if file_sha256(page_path) != page_item["sha256"]:
                    raise ValueError("raw page changed")
                page = _json(page_path)
                if (page["plan_sha256"] != derived["plan_sha256"]
                        or hashlib.sha256(page["response_utf8"].encode("utf-8")).hexdigest()
                        != page["response_sha256"]):
                    raise ValueError("raw response bytes changed")
                params = page["params"]
                endpoint_categories.add((page["endpoint"], params.get("category"),
                                         params.get("type"), params.get("interval")))
                verified_raw_pages += 1
    if (any(row["exchange"] != "BITGET" or row["market_type"] != "futures"
            or row["source_timeframe"] != "1m" for row in early)
            or endpoint_categories != {(
                "https://api.bitget.com/api/v3/market/candles",
                "USDT-FUTURES", "market", "1m")}):
        raise ValueError("mixed or unknown price source/product metadata")
    return {
        "early_1m_source_manifests": early,
        "verified_derived_parts": len(manifest["snapshot_files"]),
        "verified_recorded_chunks": verified_chunks,
        "verified_recorded_raw_pages": verified_raw_pages,
        "recorded_raw_endpoint_categories": [list(row) for row in sorted(endpoint_categories)],
        "top_manifest_declares_exchange_product": False,
        "funding_book_trades_or_fill_data_in_snapshot": False,
    }


def _economics(simulation, market: pd.DataFrame, symbols: tuple[str, ...],
               start: int, origin_ends: list[int]) -> dict:
    trades = simulation.trades
    decisions = simulation.decisions
    entry_notional = 0.0 if trades.empty else float(
        (trades.quantity.abs() * trades.entry_price).sum())
    fills_equity = simulation.events.loc[
        simulation.events.phase.eq("after_fills"), ["timestamp", "equity"]]
    exposure = decisions.merge(
        market.rename(columns={"timestamp": "execution_timestamp"}),
        on=["execution_timestamp", "symbol"], validate="one_to_one"
    ).merge(fills_equity.rename(columns={"timestamp": "execution_timestamp"}),
            on="execution_timestamp", validate="many_to_one")
    exposure["capital_fraction"] = (
        exposure.quantity_after.abs() * exposure.open / exposure.equity)
    if len(exposure) != len(decisions) or not np.isfinite(exposure.capital_fraction).all():
        raise ValueError("unreconciled exposure grid")
    by_symbol = {}
    episodes = []
    for symbol in symbols:
        group = trades.loc[trades.symbol.eq(symbol)] if not trades.empty else trades
        bars = exposure.loc[exposure.symbol.eq(symbol)]
        durations = ((group.exit_timestamp - group.entry_timestamp) / DT).tolist() if len(group) else []
        by_symbol[symbol] = {
            "trades": int(len(group)),
            "gross_pnl_pct_initial_capital": float(group.gross_pnl.sum() * 100) if len(group) else 0.0,
            "net_pnl_pct_initial_capital": float(group.net_pnl.sum() * 100) if len(group) else 0.0,
            "fees_pct_initial_capital": float((group.entry_fee + group.exit_fee).sum() * 100)
            if len(group) else 0.0,
            "entry_notional_turnover": float((group.quantity.abs() * group.entry_price).sum())
            if len(group) else 0.0,
            "active_bars": int(bars.quantity_after.ne(0).sum()),
            "mean_capital_exposure_fraction": float(bars.capital_fraction.mean()),
            "max_capital_exposure_fraction": float(bars.capital_fraction.max()),
            "median_episode_bars": float(np.median(durations)) if durations else None,
            "max_episode_bars": float(max(durations)) if durations else None,
            "profitable_episodes": int(group.net_pnl.gt(0).sum()) if len(group) else 0,
        }
    if len(trades):
        for index, row in enumerate(trades.itertuples(index=False)):
            episodes.append({
                "episode_index": index, "symbol": str(row.symbol),
                "entry_timestamp": int(row.entry_timestamp),
                "exit_timestamp": int(row.exit_timestamp),
                "duration_bars": float((row.exit_timestamp - row.entry_timestamp) / DT),
                "entry_notional": float(abs(row.quantity) * row.entry_price),
                "gross_pnl": float(row.gross_pnl),
                "entry_fee": float(row.entry_fee), "exit_fee": float(row.exit_fee),
                "net_pnl": float(row.net_pnl),
                "entry_model_sha256": str(row.entry_model_sha256),
                "exit_reason": str(row.exit_reason),
            })
    portfolio = simulation.portfolio.copy()
    portfolio["origin_index"] = ((portfolio.timestamp - 1 - start) // HORIZON).astype(int)
    origins = []
    for index, end in enumerate(origin_ends):
        block = portfolio.loc[portfolio.origin_index.eq(index)]
        if len(block) != HORIZON // DT:
            raise ValueError("incomplete origin return block")
        origins.append({"origin_index": index + 1,
                        "end_exclusive_timestamp": int(end),
                        "net_return_pct": float((np.prod(1.0 + block.net_return) - 1.0) * 100)})
    total_gross = sum(row["gross_pnl_pct_initial_capital"] for row in by_symbol.values())
    total_net = sum(row["net_pnl_pct_initial_capital"] for row in by_symbol.values())
    total_fees = sum(row["fees_pct_initial_capital"] for row in by_symbol.values())
    if (abs(total_net - simulation.metrics["net_return"] * 100) > 1e-8
            or abs(total_gross - total_fees - total_net) > 1e-8
            or abs(sum(row["entry_notional_turnover"] for row in by_symbol.values())
                   - entry_notional) > 1e-9):
        raise ArithmeticError("trade/portfolio decomposition failed")
    total_exposure = exposure.groupby("execution_timestamp").capital_fraction.sum()
    return {"by_symbol": by_symbol, "episodes": episodes,
            "by_origin": origins, "mean_portfolio_capital_exposure_fraction": float(total_exposure.mean()),
            "max_portfolio_capital_exposure_fraction": float(total_exposure.max()),
            "active_symbol_bars": int(decisions.quantity_after.ne(0).sum()),
            "gross_pct_initial_capital": total_gross,
            "fees_pct_initial_capital": total_fees,
            "net_pct_initial_capital": total_net,
            "entry_notional_turnover": entry_notional}


def build() -> dict:
    config = _checked_config()
    contract, _ = load_and_validate(SOURCE_CONFIG)
    source = _json(_path(config["source_ablation_result_path"]))
    if (config["fixed_universe"] != contract["universe"]
            or source["contract_sha256"] != file_sha256(SOURCE_CONFIG)
            or config["costs_round_trip_bps"]["normal"]
            != contract["accounting"]["normal_round_trip_bps"]
            or config["costs_round_trip_bps"]["stress"]
            != contract["accounting"]["stress_round_trip_bps"]):
        raise ValueError("fixed universe/accounting/source contract mismatch")
    manifest, panel = load_verified_snapshot(
        _path(config["source_manifest_path"]), root=ROOT,
        contract_path=SOURCE_CONFIG, contract=contract)
    symbols = tuple(config["fixed_universe"])
    data_integrity = _data_integrity(panel, manifest, symbols)
    source_lineage = _source_lineage(manifest)
    base, lineage, origin_ends = _archived_decisions(contract)
    start = int(base.execution_timestamp.min())
    end = int(origin_ends[-1])
    if (source["archive_lineage"] != lineage
            or source["interval_start"] != start
            or source["interval_end_exclusive"] != end
            or len(origin_ends) != 9):
        raise ValueError("archived nine-origin decisions changed")
    market = panel.loc[panel.timestamp.ge(start) & panel.timestamp.lt(end),
                       ["timestamp", "symbol", "open", "close"]].copy()
    if _frame_sha(market) != source["market_rows_sha256"] or len(market) != len(base):
        raise ValueError("disclosed market rows changed")
    result = {
        "schema_version": "exia.genetic_primus.baseline_economics_audit.result/1",
        "objective_id": config["objective_id"], "evidence_class": config["evidence_class"],
        "config_sha256": file_sha256(CONFIG), "script_sha256": file_sha256(Path(__file__)),
        "source_ablation_result_sha256": config["source_ablation_result_sha256"],
        "source_manifest_sha256": config["source_manifest_sha256"],
        "market_rows_sha256": source["market_rows_sha256"],
        "archive_lineage": lineage, "data_integrity": data_integrity,
        "source_lineage": source_lineage, "interval_start": start,
        "interval_end_exclusive": end, "policies": {},
        "promotion_authority": False, "safety": dict(config["safety"]),
    }
    for policy in config["policies"]:
        schedule = targets_for_policy(base, policy)
        archived = source["policies"][policy][0]
        if archived["seed"] is not None or _frame_sha(schedule) != archived["schedule_sha256"]:
            raise ValueError("fixed-policy schedule changed")
        evaluations = {}
        for label, bps in config["costs_round_trip_bps"].items():
            simulation = simulate_cash_targets(
                market, schedule, symbols=symbols, start_timestamp=start,
                end_timestamp=end, timeframe_ms=DT,
                round_trip_cost_bps=bps, record_details=True)
            summary = summarize_simulation(simulation, schedule, base,
                                           tuple(origin_ends), bps)
            if summary != archived["evaluations"][label]:
                raise ValueError(f"archived {policy}/{label} metrics changed")
            evaluations[label] = {
                "archived_metrics_replayed_exactly": True,
                "metrics": summary,
                "economics": _economics(simulation, market, symbols, start, origin_ends),
            }
        result["policies"][policy] = {
            "schedule_sha256": archived["schedule_sha256"],
            "evaluations": evaluations,
        }
    return result


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("immutable baseline audit already exists")
    result = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "status": "DISCLOSED_BASELINE_AUDIT_ONLY",
        "policies": len(result["policies"]),
        "snapshot_rows": result["data_integrity"]["rows"],
        "raw_pages_verified": result["source_lineage"]["verified_recorded_raw_pages"],
        "promotion": result["promotion_authority"],
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
