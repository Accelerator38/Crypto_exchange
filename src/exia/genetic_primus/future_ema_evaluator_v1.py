"""Pure market validation and fixed EMA viability evaluation; no I/O or network."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .accounting_v2 import simulate_cash_targets
from .latency_sensitivity_v1 import delay_target_schedule
from .measurement_lab_v2 import (
    array_sha, block_bank, build_lab_features, daily_returns, object_sha, paired_lcb,
)

DT = 14_400_000
DAY = 86_400_000
HORIZON = 3 * DAY
COLUMNS = ("timestamp", "symbol", "open", "high", "low", "close", "volume")


def utc_ms(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(None):
        raise ValueError("UTC timestamp required")
    return int(parsed.timestamp() * 1000)


def require_completed(now_utc: datetime, end_ms: int) -> None:
    if now_utc.tzinfo is None or now_utc.utcoffset() != timezone.utc.utcoffset(None):
        raise ValueError("UTC clock required")
    if int(now_utc.timestamp() * 1000) < end_ms:
        raise ValueError("registered future interval is incomplete")


def validate_market_grid(panel: pd.DataFrame, symbols: tuple[str, ...],
                         start_ms: int, end_ms: int) -> pd.DataFrame:
    """Require exact full-universe open-time grid and physically valid OHLCV."""
    if (not symbols or len(set(symbols)) != len(symbols) or end_ms <= start_ms
            or start_ms % DT or end_ms % DT or not set(COLUMNS).issubset(panel.columns)):
        raise ValueError("invalid market grid contract")
    result = panel[list(COLUMNS)].copy()
    if (result.empty or result.isna().any().any()
            or not result.symbol.map(lambda v: isinstance(v, str)).all()):
        raise ValueError("empty market or invalid symbol/value")
    timestamps = pd.to_numeric(result.timestamp, errors="raise").to_numpy(float)
    if (not np.isfinite(timestamps).all() or (timestamps != np.floor(timestamps)).any()
            or (timestamps < start_ms).any() or (timestamps >= end_ms).any()):
        raise ValueError("invalid or out-of-interval timestamp")
    result["timestamp"] = timestamps.astype(np.int64)
    numeric = list(COLUMNS[2:])
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="raise")
    values = result[numeric].to_numpy(float)
    if (not np.isfinite(values).all() or (result[list(COLUMNS[2:6])] <= 0).any().any()
            or (result.volume < 0).any()
            or (result.high < result[["open", "close", "low"]].max(axis=1)).any()
            or (result.low > result[["open", "close", "high"]].min(axis=1)).any()):
        raise ValueError("invalid OHLC or volume")
    expected_rows = ((end_ms - start_ms) // DT) * len(symbols)
    if (len(result) != expected_rows or result.duplicated(["timestamp", "symbol"]).any()
            or set(result.symbol) != set(symbols)):
        raise ValueError("duplicate, missing or foreign market row")
    expected = pd.MultiIndex.from_product(
        [np.arange(start_ms, end_ms, DT), sorted(symbols)], names=["timestamp", "symbol"])
    result = result.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)
    if not pd.MultiIndex.from_frame(result[["timestamp", "symbol"]]).equals(expected):
        raise ValueError("missing or misaligned full-universe candle")
    return result


def ema_schedule(panel: pd.DataFrame, symbols: tuple[str, ...], start_ms: int,
                 end_ms: int, policy_sha: str) -> tuple[pd.DataFrame, list[dict]]:
    if (end_ms <= start_ms or (end_ms - start_ms) % HORIZON
            or not isinstance(policy_sha, str) or not policy_sha):
        raise ValueError("complete origins and fixed policy SHA required")
    features = build_lab_features(panel.loc[panel.timestamp.lt(end_ms)]).set_index(
        ["timestamp", "symbol"])
    frames, ledger = [], []
    names = ("ema_gap_12_48", "ema_slope_12_3", "return_12", "efficiency_12")
    previous = {symbol: 0 for symbol in symbols}
    for origin in range(start_ms, end_ms, HORIZON):
        decisions = {}
        for symbol in symbols:
            row = features.loc[(origin - DT, symbol), list(names)]
            values = row.to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError("nonfinite previous-closed-bar feature")
            target = int(values[0] > 0 and values[1] > 0 and values[2] > 0 and values[3] >= 0.30)
            decisions[symbol] = {"proposal_long": bool(target), "target": target,
                                 "position_before": previous[symbol],
                                 "feature_bar_timestamp": origin - DT,
                                 "feature_available_timestamp": origin,
                                 "features": dict(zip(names, map(float, values))),
                                 "model_sha256": policy_sha}
            previous[symbol] = target
        frame = pd.MultiIndex.from_product(
            [np.arange(origin, origin + HORIZON, DT), symbols],
            names=["execution_timestamp", "symbol"]).to_frame(index=False)
        frame["signal_timestamp"] = origin
        frame["target"] = frame.symbol.map(previous).astype(int)
        frame["model_sha256"] = policy_sha
        frames.append(frame)
        ledger.append({"origin_start": origin, "origin_end_exclusive": origin + HORIZON,
                       "decisions": decisions})
    return pd.concat(frames, ignore_index=True), ledger


def viability_gate(scenarios: dict, signal_origins: int, primary: np.ndarray,
                   audit: np.ndarray, *, alpha: float, minimum_signal_origins: int,
                   drawdown_limit: float, concentration_limit: float) -> dict:
    conditions = {"distinct_signal_origins": signal_origins >= minimum_signal_origins}
    lcbs, origins, slices = {}, {}, {}
    for name in ("normal", "execution_stress"):
        item = scenarios[name]
        daily = np.asarray(item["daily_returns"], float)
        if (daily.shape != (180,) or not np.isfinite(daily).all()
                or (daily <= -1).any() or not np.isfinite(item["max_drawdown"])
                or not 0 <= item["max_drawdown"] <= 1):
            raise ValueError("invalid complete 180-day scenario")
        net = float(np.prod(1.0 + daily) - 1.0)
        if not np.isclose(net, item["net_return"], atol=1e-10, rtol=1e-10):
            raise ValueError("scenario daily PnL does not reconcile")
        conditions[f"{name}_net_positive"] = net > 0
        conditions[f"{name}_drawdown"] = item["max_drawdown"] <= drawdown_limit
        lcbs[name] = {}
        for label, bank in (("primary", primary), ("audit", audit)):
            value = paired_lcb(daily, np.zeros(180), bank, alpha)
            lcbs[name][label] = value
            conditions[f"{name}_{label}_lcb_positive"] = value > 0
        origins[name] = (np.prod(1 + daily.reshape(60, 3), axis=1) - 1).tolist()
        slices[name] = (np.prod(1 + daily.reshape(6, 30), axis=1) - 1).tolist()
    positive = np.maximum(np.asarray(origins["execution_stress"]), 0)
    concentration = float(positive.max() / positive.sum()) if positive.sum() > 0 else None
    conditions["stress_four_positive_outer_slices"] = sum(v > 0 for v in slices["execution_stress"]) >= 4
    conditions["stress_origin_concentration"] = concentration is not None and concentration <= concentration_limit
    return {"passed": all(conditions.values()), "conditions": conditions, "lcbs": lcbs,
            "origin_returns": origins, "outer_slice_returns": slices,
            "stress_positive_origin_concentration": concentration,
            "distinct_signal_origins": int(signal_origins)}


def evaluate_registered_panel(c: dict, history: pd.DataFrame, future: pd.DataFrame) -> dict:
    symbols = tuple(c["universe"])
    known_end = utc_ms(c["known_revealed_end_exclusive_utc"])
    start, end = utc_ms(c["future_start_utc"]), utc_ms(c["future_end_exclusive_utc"])
    history = validate_market_grid(history, symbols, int(history.timestamp.min()), known_end)
    future = validate_market_grid(future, symbols, known_end, end)
    if len(future) != c["required_new_rows"]:
        raise ValueError("wrong registered new segment size")
    combined = pd.concat([history, future], ignore_index=True)
    policy_sha = object_sha({"objective_id": c["objective_id"], "policy": c["policy"],
                             "frozen_code_sha256": c["frozen_code_sha256"]})
    schedule, ledger = ema_schedule(combined, symbols, start, end, policy_sha)
    signal_origins = sum(any(d["proposal_long"] for d in item["decisions"].values()) for item in ledger)
    scenarios = {}
    for scenario in c["evaluation_scenarios"]:
        delayed = delay_target_schedule(schedule, symbols, start, end, scenario["delay_bars"])
        sim = simulate_cash_targets(
            combined, delayed, symbols=symbols, start_timestamp=start, end_timestamp=end,
            timeframe_ms=DT, round_trip_cost_bps=scenario["round_trip_cost_bps"], record_details=False)
        scenarios[scenario["name"]] = {
            "net_return": float(sim.metrics["net_return"]),
            "max_drawdown": float(sim.metrics["max_drawdown"]),
            "closed_trades": int(sim.metrics["trades"]),
            "total_fees": float(sim.metrics["total_fees"]),
            "daily_returns": daily_returns(sim).tolist(),
            "schedule_sha256": object_sha(delayed.to_dict("records"))}
    b = c["bootstrap"]
    banks = {label: block_bank(b["daily_observations"], b["samples"], b["block_days"], b[f"{label}_seed"])
             for label in ("primary", "audit")}
    for label, bank in banks.items():
        if array_sha(bank) != b[f"{label}_bank_sha256"]:
            raise ValueError("registered resample bank differs")
    gate = viability_gate(
        scenarios, signal_origins, banks["primary"], banks["audit"], alpha=b["alpha"],
        minimum_signal_origins=c["minimum_distinct_signal_origins"],
        drawdown_limit=c["maximum_drawdown"],
        concentration_limit=c["single_origin_positive_pnl_concentration_limit"])
    status = ("INCONCLUSIVE" if signal_origins < c["minimum_distinct_signal_origins"] else
              "ELIGIBLE_FOR_SEPARATE_LATER_RESEARCH" if gate["passed"] else "EMA_BRANCH_FAILED")
    return {"objective_id": c["objective_id"], "scenarios": scenarios, "gate": gate,
            "origin_ledger": ledger, "policy_sha256": policy_sha,
            "reference_policy": {"name": "NoTrade", "net_return": 0.0},
            "research_status": status, "final_policy": "NoTrade", "new_model_fits": 0,
            "automatic_promotion": False, "promotion_authority": False}
