from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import RobustScaler
from .policy import GeneticPrimusPolicy


LEDGER_COLUMNS = (
    "symbol",
    "direction",
    "market_state",
    "signal_timestamp",
    "entry_timestamp",
    "exit_timestamp",
    "entry_price",
    "exit_price",
    "holding_bars",
    "gross_bps",
    "cost_bps",
    "net_bps",
    "fills",
    "exit_reason",
)

DECISION_COLUMNS = (
    "model_sha256",
    "origin_timestamp",
    "signal_timestamp",
    "execution_timestamp",
    "symbol",
    "market_state",
    "feature_supported",
    "position_before",
    "desired_position",
    "target_changed",
)


@dataclass(frozen=True)
class PrimusSimulation:
    ledger: pd.DataFrame
    portfolio: pd.DataFrame
    action_counts: dict[str, int]
    turnover_fills: int
    decisions: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=DECISION_COLUMNS)
    )
    diagnostics: dict[str, float | int] = field(default_factory=dict)


def _dynamic_features(
    *,
    position: int,
    entry_price: float,
    close: float,
    holding_bars: int,
    mfe_bps: float,
    mae_bps: float,
    round_trip_cost_bps: float,
) -> np.ndarray:
    unrealized = (
        position * (close / entry_price - 1.0) * 10_000.0 - round_trip_cost_bps
        if position and entry_price > 0.0
        else 0.0
    )
    return np.asarray(
        [
            float(position),
            np.clip(unrealized / 500.0, -8.0, 8.0),
            np.clip(holding_bars / 24.0, 0.0, 8.0),
            np.clip(mfe_bps / 500.0, 0.0, 8.0),
            np.clip(mae_bps / 500.0, -8.0, 0.0),
        ],
        dtype=np.float32,
    )


def overlay_permission_step(
    *,
    baseline_target: int,
    score: float,
    permission: bool,
    permission_direction: int,
    cooldown_remaining: int,
    entry_threshold: float,
    exit_threshold: float,
    cooldown_bars: int,
) -> tuple[bool, int, int]:
    """Pure hysteresis/cooldown transition for deterministic offline tests."""
    if baseline_target == 0:
        cooldown = cooldown_bars if permission else max(0, cooldown_remaining - 1)
        return False, 0, cooldown
    if permission and permission_direction != baseline_target:
        return False, 0, cooldown_bars
    if cooldown_remaining > 0:
        return False, 0, cooldown_remaining - 1
    threshold = exit_threshold if permission else entry_threshold
    next_permission = bool(score > threshold)
    if next_permission:
        return True, baseline_target, 0
    return False, 0, cooldown_bars if permission else 0


def overlay_opportunity_mask(
    values: np.ndarray,
    *,
    threshold_mode: str,
    threshold_value: float,
) -> np.ndarray:
    signals = np.asarray(values, dtype=np.float64)
    if threshold_value < 0.0:
        raise ValueError("overlay opportunity threshold cannot be negative")
    if threshold_mode == "return_bps":
        magnitude = np.abs(signals) * 10_000.0
    elif threshold_mode == "absolute_feature":
        magnitude = np.abs(signals)
    else:
        raise ValueError("unsupported overlay threshold mode")
    return np.isfinite(signals) & (magnitude > float(threshold_value))


def simulate_policy(
    frame: pd.DataFrame,
    *,
    policy: GeneticPrimusPolicy,
    scaler: RobustScaler,
    start_timestamp: int,
    end_timestamp: int,
    round_trip_cost_bps: float,
    record_decisions: bool = False,
    model_sha256: str | None = None,
    origin_timestamp: int | None = None,
    policy_mode: str = "direct",
    overlay_baseline_feature: str = "return_1",
    overlay_minimum_move_bps: float = 0.0,
    overlay_threshold_mode: str = "return_bps",
    overlay_threshold_value: float | None = None,
    overlay_entry_logit_threshold: float = 0.0,
    overlay_exit_logit_threshold: float = 0.0,
    overlay_cooldown_bars: int = 0,
) -> PrimusSimulation:
    """Run one policy with close-time decisions and next-open execution."""
    if round_trip_cost_bps <= 0:
        raise ValueError("simulation costs must be positive")
    if record_decisions and (not model_sha256 or origin_timestamp is None):
        raise ValueError("decision recording requires model SHA and origin timestamp")
    if policy_mode not in {
        "direct",
        "momentum_abstention",
        "momentum_abstention_binary",
    }:
        raise ValueError("unsupported simulation policy mode")
    if overlay_minimum_move_bps < 0.0:
        raise ValueError("overlay minimum move cannot be negative")
    threshold_value = (
        float(overlay_minimum_move_bps)
        if overlay_threshold_value is None
        else float(overlay_threshold_value)
    )
    overlay_opportunity_mask(
        np.asarray([0.0]),
        threshold_mode=overlay_threshold_mode,
        threshold_value=threshold_value,
    )
    if overlay_entry_logit_threshold < overlay_exit_logit_threshold:
        raise ValueError("overlay entry threshold must be >= exit threshold")
    if overlay_cooldown_bars < 0:
        raise ValueError("overlay cooldown cannot be negative")
    if policy_mode.startswith("momentum_abstention") and overlay_baseline_feature not in frame:
        raise ValueError("overlay baseline feature is absent from the simulation frame")
    static = scaler.transform(frame)
    source = frame.reset_index(drop=True).copy()
    source["_row_id"] = np.arange(len(source), dtype=np.int64)
    ledgers: list[dict[str, object]] = []
    return_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    action_counts = {"SHORT": 0, "FLAT": 0, "LONG": 0}
    overlay_opportunities = 0
    overlay_allowed = 0
    overlay_abstained = 0
    overlay_opportunities_by_direction: dict[str, int] = {}
    overlay_opportunities_by_state: dict[str, int] = {}
    overlay_opportunities_by_symbol: dict[str, int] = {}
    overlay_allowed_by_direction: dict[str, int] = {}
    overlay_allowed_by_state: dict[str, int] = {}
    overlay_allowed_by_symbol: dict[str, int] = {}
    turnover_fills = 0
    per_fill_cost = float(round_trip_cost_bps) / 2.0

    for symbol, raw_group in source.groupby("symbol", sort=True, observed=True):
        group = raw_group.sort_values("timestamp", kind="stable").reset_index(drop=True)
        group = group.loc[
            (group["timestamp"] >= int(start_timestamp))
            & (group["timestamp"] < int(end_timestamp))
        ].reset_index(drop=True)
        if len(group) < 2:
            continue
        row_ids = group["_row_id"].to_numpy(dtype=np.int64, copy=True)
        timestamps = group["timestamp"].to_numpy(dtype=np.int64, copy=True)
        opens = group["open"].to_numpy(dtype=np.float64, copy=True)
        highs = group["high"].to_numpy(dtype=np.float64, copy=True)
        lows = group["low"].to_numpy(dtype=np.float64, copy=True)
        closes = group["close"].to_numpy(dtype=np.float64, copy=True)
        feature_supported = group["feature_supported"].to_numpy(dtype=bool, copy=True)
        market_states = group["market_state"].to_numpy(copy=True)
        baseline_values = (
            group[overlay_baseline_feature].to_numpy(dtype=np.float64, copy=True)
            if policy_mode.startswith("momentum_abstention")
            else None
        )
        position = 0
        entry_price = 0.0
        entry_timestamp = 0
        entry_signal_timestamp = 0
        entry_market_state = "unknown"
        overlay_permission = False
        overlay_permission_direction = 0
        overlay_cooldown_remaining = 0
        holding_bars = 0
        mfe_bps = 0.0
        mae_bps = 0.0

        for index in range(len(group) - 1):
            row_timestamp = int(timestamps[index])
            next_timestamp = int(timestamps[index + 1])
            row_open = float(opens[index])
            next_open = float(opens[index + 1])
            if position:
                high_pnl = position * (
                    (float(highs[index]) if position > 0 else float(lows[index])) / entry_price
                    - 1.0
                ) * 10_000.0
                low_pnl = position * (
                    (float(lows[index]) if position > 0 else float(highs[index])) / entry_price
                    - 1.0
                ) * 10_000.0
                mfe_bps = max(mfe_bps, high_pnl)
                mae_bps = min(mae_bps, low_pnl)
            dynamic = _dynamic_features(
                position=position,
                entry_price=entry_price,
                close=float(closes[index]),
                holding_bars=holding_bars,
                mfe_bps=mfe_bps,
                mae_bps=mae_bps,
                round_trip_cost_bps=round_trip_cost_bps,
            )
            row_static = static[int(row_ids[index])]
            if bool(feature_supported[index]):
                policy_features = np.concatenate([row_static, dynamic])
                if policy_mode.startswith("momentum_abstention"):
                    baseline_value = float(baseline_values[index])
                    baseline_target = (
                        int(np.sign(baseline_value))
                        if overlay_opportunity_mask(
                            np.asarray([baseline_value]),
                            threshold_mode=overlay_threshold_mode,
                            threshold_value=threshold_value,
                        )[0]
                        else 0
                    )
                    if baseline_target:
                        overlay_opportunities += 1
                        direction_key = "LONG" if baseline_target > 0 else "SHORT"
                        state_key = str(market_states[index])
                        for counts, key in (
                            (overlay_opportunities_by_direction, direction_key),
                            (overlay_opportunities_by_state, state_key),
                            (overlay_opportunities_by_symbol, str(symbol)),
                        ):
                            counts[key] = counts.get(key, 0) + 1
                    if policy_mode == "momentum_abstention_binary":
                        controls_active = bool(
                            overlay_entry_logit_threshold != 0.0
                            or overlay_exit_logit_threshold != 0.0
                            or overlay_cooldown_bars != 0
                        )
                        if not controls_active:
                            desired = baseline_target if policy.allow(policy_features) else 0
                        else:
                            score = (
                                float(policy.allow_score(policy_features))
                                if hasattr(policy, "allow_score")
                                else (1.0 if policy.allow(policy_features) else -1.0)
                            )
                            (
                                overlay_permission,
                                overlay_permission_direction,
                                overlay_cooldown_remaining,
                            ) = overlay_permission_step(
                                baseline_target=baseline_target,
                                score=score,
                                permission=overlay_permission,
                                permission_direction=overlay_permission_direction,
                                cooldown_remaining=overlay_cooldown_remaining,
                                entry_threshold=overlay_entry_logit_threshold,
                                exit_threshold=overlay_exit_logit_threshold,
                                cooldown_bars=int(overlay_cooldown_bars),
                            )
                            desired = baseline_target if overlay_permission else 0
                    else:
                        desired = (
                            baseline_target
                            if policy.target(policy_features) == baseline_target
                            else 0
                        )
                    if desired:
                        overlay_allowed += 1
                        for counts, key in (
                            (overlay_allowed_by_direction, direction_key),
                            (overlay_allowed_by_state, state_key),
                            (overlay_allowed_by_symbol, str(symbol)),
                        ):
                            counts[key] = counts.get(key, 0) + 1
                    elif baseline_target:
                        overlay_abstained += 1
                else:
                    desired = policy.target(policy_features)
            else:
                desired = position
            action_counts[{-1: "SHORT", 0: "FLAT", 1: "LONG"}[desired]] += 1
            if record_decisions:
                decision_rows.append(
                    {
                        "model_sha256": str(model_sha256),
                        "origin_timestamp": int(origin_timestamp),
                        "signal_timestamp": row_timestamp,
                        "execution_timestamp": next_timestamp,
                        "symbol": str(symbol),
                        "market_state": str(market_states[index]),
                        "feature_supported": bool(feature_supported[index]),
                        "position_before": int(position),
                        "desired_position": int(desired),
                        "target_changed": bool(desired != position),
                    }
                )

            bar_net_return = position * (next_open / row_open - 1.0)
            fills = 0
            if desired != position:
                fills = (1 if position else 0) + (1 if desired else 0)
                turnover_fills += fills
                bar_net_return -= fills * per_fill_cost / 10_000.0
                if position:
                    gross_bps = position * (next_open / entry_price - 1.0) * 10_000.0
                    ledgers.append(
                        {
                            "symbol": str(symbol),
                            "direction": "LONG" if position > 0 else "SHORT",
                            "market_state": entry_market_state,
                            "signal_timestamp": entry_signal_timestamp,
                            "entry_timestamp": entry_timestamp,
                            "exit_timestamp": next_timestamp,
                            "entry_price": entry_price,
                            "exit_price": next_open,
                            "holding_bars": holding_bars,
                            "gross_bps": gross_bps,
                            "cost_bps": float(round_trip_cost_bps),
                            "net_bps": gross_bps - float(round_trip_cost_bps),
                            "fills": 2,
                            "exit_reason": "target_change",
                        }
                    )
                position = desired
                holding_bars = 0
                mfe_bps = 0.0
                mae_bps = 0.0
                if position:
                    entry_price = next_open
                    entry_timestamp = next_timestamp
                    entry_signal_timestamp = row_timestamp
                    entry_market_state = str(market_states[index])
                else:
                    entry_price = 0.0
            elif position:
                holding_bars += 1
            return_rows.append(
                {
                    "timestamp": next_timestamp,
                    "symbol": str(symbol),
                    "net_return": float(bar_net_return),
                }
            )

        if position:
            final_timestamp = int(timestamps[-1])
            final_open = float(opens[-1])
            exit_price = float(closes[-1])
            gross_bps = position * (exit_price / entry_price - 1.0) * 10_000.0
            ledgers.append(
                {
                    "symbol": str(symbol),
                    "direction": "LONG" if position > 0 else "SHORT",
                    "market_state": entry_market_state,
                    "signal_timestamp": entry_signal_timestamp,
                    "entry_timestamp": entry_timestamp,
                    "exit_timestamp": final_timestamp,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "holding_bars": holding_bars + 1,
                    "gross_bps": gross_bps,
                    "cost_bps": float(round_trip_cost_bps),
                    "net_bps": gross_bps - float(round_trip_cost_bps),
                    "fills": 2,
                    "exit_reason": "split_end",
                }
            )
            turnover_fills += 1
            return_rows.append(
                {
                    "timestamp": final_timestamp,
                    "symbol": str(symbol),
                    "net_return": position * (exit_price / final_open - 1.0)
                    - per_fill_cost / 10_000.0,
                }
            )

    ledger = pd.DataFrame(ledgers, columns=LEDGER_COLUMNS)
    if return_rows:
        returns = pd.DataFrame(return_rows)
        portfolio_wide = returns.pivot_table(
            index="timestamp", columns="symbol", values="net_return", aggfunc="sum", fill_value=0.0
        ).sort_index()
        net = portfolio_wide.mean(axis=1)
        portfolio = pd.DataFrame({"timestamp": net.index.astype("int64"), "net_return": net.values})
        portfolio["equity"] = (1.0 + portfolio["net_return"]).cumprod()
        portfolio["drawdown"] = portfolio["equity"] / portfolio["equity"].cummax() - 1.0
    else:
        portfolio = pd.DataFrame(columns=["timestamp", "net_return", "equity", "drawdown"])
    return PrimusSimulation(
        ledger=ledger,
        portfolio=portfolio,
        action_counts=action_counts,
        turnover_fills=int(turnover_fills),
        decisions=pd.DataFrame(decision_rows, columns=DECISION_COLUMNS),
        diagnostics={
            "policy_mode": policy_mode,
            "overlay_entry_logit_threshold": float(overlay_entry_logit_threshold),
            "overlay_threshold_mode": overlay_threshold_mode,
            "overlay_threshold_value": threshold_value,
            "overlay_exit_logit_threshold": float(overlay_exit_logit_threshold),
            "overlay_cooldown_bars": int(overlay_cooldown_bars),
            "overlay_opportunities": int(overlay_opportunities),
            "overlay_allowed": int(overlay_allowed),
            "overlay_abstained": int(overlay_abstained),
            "overlay_opportunities_by_direction": overlay_opportunities_by_direction,
            "overlay_opportunities_by_state": overlay_opportunities_by_state,
            "overlay_opportunities_by_symbol": overlay_opportunities_by_symbol,
            "overlay_allowed_by_direction": overlay_allowed_by_direction,
            "overlay_allowed_by_state": overlay_allowed_by_state,
            "overlay_allowed_by_symbol": overlay_allowed_by_symbol,
            "overlay_allowed_fraction": (
                float(overlay_allowed / overlay_opportunities)
                if overlay_opportunities
                else 0.0
            ),
        },
    )
