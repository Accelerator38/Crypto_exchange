from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts_v7 import V7_EVENT_IDS
from .fitness_v4 import contiguous_block_lcb
from .feasibility_v6 import apply_cooldown


ACCEPTED_EVENT_STATUS = "V7_DATA_SOURCE_AND_EVENT_CAPACITY_ACCEPTED"


def _seed(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def _execution_columns(local: pd.DataFrame, holding_bars: int) -> pd.DataFrame:
    result = local.copy()
    result["entry_timestamp"] = result["timestamp"].shift(-1)
    result["entry_price"] = result["open"].shift(-1)
    result["exit_timestamp"] = result["timestamp"].shift(-(holding_bars + 1))
    result["exit_price"] = result["open"].shift(-(holding_bars + 1))
    return result


def _event_rows(
    local: pd.DataFrame,
    *,
    mask: pd.Series,
    direction: pd.Series,
    event_id: str,
    cooldown: int,
    cost_bps: float,
) -> pd.DataFrame:
    selected = apply_cooldown(mask, cooldown)
    event = local.loc[selected].copy()
    event["event_id"] = event_id
    event["direction_value"] = direction.loc[selected].astype("int8")
    event = event.loc[
        event[["entry_timestamp", "entry_price", "exit_timestamp", "exit_price"]].notna().all(axis=1)
        & event["direction_value"].ne(0)
    ].copy()
    event["gross_bps"] = event["direction_value"] * (event["exit_price"] / event["entry_price"] - 1.0) * 10_000.0
    event["net_bps"] = event["gross_bps"] - float(cost_bps)
    event["direction"] = np.where(event["direction_value"] > 0, "LONG", "SHORT")
    return event


def build_event_ledger(frame: pd.DataFrame, config: Mapping[str, Any], holding_bars: int) -> pd.DataFrame:
    window = int(config["trailing_window_bars"])
    minimum = int(config["minimum_trailing_bars"])
    cooldown = int(config["event_cooldown_bars"])
    cost = float(config["stress_round_trip_cost_bps"])
    ledgers: list[pd.DataFrame] = []

    btc = frame.loc[frame["symbol"] == "BTC/USDT"].sort_values("timestamp", kind="stable").reset_index(drop=True)
    btc = _execution_columns(btc, holding_bars)
    btc["cross_total_p90"] = (
        btc["cross_liquidation_total_usd"]
        .where(btc["usable_cross_market"])
        .rolling(window, min_periods=minimum)
        .quantile(float(config["cross_total_quantile"]))
        .shift(1)
    )
    btc["market_abs_return_p75"] = (
        btc["market_return_1"].abs().rolling(window, min_periods=minimum).quantile(float(config["cross_return_quantile"])).shift(1)
    )
    cross_total = btc["cross_liquidation_total_usd"].replace(0.0, np.nan)
    btc["cross_buy_share"] = btc["cross_buy_liquidation_usd"] / cross_total
    btc["cross_sell_share"] = btc["cross_sell_liquidation_usd"] / cross_total
    cross_direction = np.sign(btc["cross_buy_liquidation_usd"] - btc["cross_sell_liquidation_usd"])
    cross_dominance = btc[["cross_buy_share", "cross_sell_share"]].max(axis=1)
    cross_mask = (
        btc["usable_cross_market"]
        & (btc["cross_liquidation_total_usd"] >= btc["cross_total_p90"])
        & (cross_dominance >= float(config["cross_side_share_minimum"]))
        & (btc["market_return_1"].abs() >= btc["market_abs_return_p75"])
        & np.sign(btc["market_return_1"]).eq(cross_direction)
        & cross_direction.ne(0)
    )
    for event_id, direction in (
        ("cross_market_cascade_reversal", -cross_direction),
        ("cross_market_cascade_continuation", cross_direction),
    ):
        event = _event_rows(
            btc,
            mask=cross_mask,
            direction=direction,
            event_id=event_id,
            cooldown=cooldown,
            cost_bps=cost,
        )
        ledgers.append(event)

    for symbol, source in frame.groupby("symbol", sort=True, observed=True):
        local = source.sort_values("timestamp", kind="stable").reset_index(drop=True).copy()
        local = _execution_columns(local, holding_bars)
        local["symbol_total_p95"] = (
            local["liquidation_total_usd"]
            .where(local["archive_available"])
            .rolling(window, min_periods=minimum)
            .quantile(float(config["symbol_total_quantile"]))
            .shift(1)
        )
        local["symbol_abs_return_p75"] = (
            local["return_1"].abs().rolling(window, min_periods=minimum).quantile(float(config["symbol_return_quantile"])).shift(1)
        )
        symbol_total = local["liquidation_total_usd"].replace(0.0, np.nan)
        buy_share = local["buy_liquidation_usd"] / symbol_total
        sell_share = local["sell_liquidation_usd"] / symbol_total
        dominance = pd.concat([buy_share, sell_share], axis=1).max(axis=1)
        flow_direction = np.sign(local["buy_liquidation_usd"] - local["sell_liquidation_usd"])
        mask = (
            local["archive_available"]
            & (local["liquidation_total_usd"] >= local["symbol_total_p95"])
            & (dominance >= float(config["symbol_side_share_minimum"]))
            & (local["return_1"].abs() >= local["symbol_abs_return_p75"])
            & np.sign(local["return_1"]).eq(flow_direction)
            & flow_direction.ne(0)
        )
        ledgers.append(
            _event_rows(
                local,
                mask=mask,
                direction=-flow_direction,
                event_id="isolated_symbol_cascade_reversal",
                cooldown=cooldown,
                cost_bps=cost,
            )
        )
    ledger = pd.concat(ledgers, ignore_index=True)
    keep = [
        "event_id",
        "symbol",
        "market_state",
        "direction",
        "direction_value",
        "timestamp",
        "entry_timestamp",
        "exit_timestamp",
        "entry_price",
        "exit_price",
        "gross_bps",
        "net_bps",
        "return_1",
        "market_return_1",
        "liquidation_total_usd",
        "cross_liquidation_total_usd",
        "available_symbol_count",
    ]
    return ledger.loc[:, keep].sort_values(["timestamp", "event_id", "symbol"], kind="stable").reset_index(drop=True)


def cluster_lcb(values: pd.DataFrame, config: Mapping[str, Any], key: str) -> float | None:
    if values.empty:
        return None
    local = values.loc[:, ["timestamp", "net_bps"]].copy()
    local["day"] = pd.to_datetime(local["timestamp"], unit="ms", utc=True).dt.floor("1D")
    daily = local.groupby("day", sort=True)["net_bps"].mean().to_numpy(dtype=float)
    return contiguous_block_lcb(
        daily,
        samples=int(config["bootstrap_samples"]),
        alpha=float(config["bootstrap_alpha"]),
        block_size=int(config["bootstrap_block_days"]),
        seed=_seed(key),
    )


def score_events(
    ledger: pd.DataFrame,
    *,
    folds: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    minimum_events = int(config["minimum_events_per_fold"])
    for event_id in V7_EVENT_IDS:
        event_rows = ledger.loc[ledger["event_id"] == event_id]
        accepted_parts = []
        for fold in folds:
            start = int(pd.Timestamp(fold["evaluation_start"]).timestamp() * 1000)
            end = int(pd.Timestamp(fold["evaluation_end"]).timestamp() * 1000)
            subset = event_rows.loc[
                (event_rows["timestamp"] >= start)
                & (event_rows["timestamp"] < end)
                & (event_rows["exit_timestamp"] < end)
            ]
            accepted_parts.append(subset)
            fold_rows.append(
                {
                    "event_id": event_id,
                    "fold": fold["name"],
                    "events": int(len(subset)),
                    "mean_net_bps": float(subset["net_bps"].mean()) if len(subset) else None,
                    "median_net_bps": float(subset["net_bps"].median()) if len(subset) else None,
                    "lcb_net_bps": cluster_lcb(subset, config, f"v7:{event_id}:{fold['name']}"),
                    "incidence_passed": len(subset) >= minimum_events,
                }
            )
        accepted = pd.concat(accepted_parts, ignore_index=True) if accepted_parts else pd.DataFrame()
        event_folds = pd.DataFrame(fold_rows).loc[lambda value: value["event_id"] == event_id]
        fold_means = event_folds["mean_net_bps"].dropna().astype(float)
        fold_lcbs = event_folds["lcb_net_bps"].dropna().astype(float)
        mean = float(accepted["net_bps"].mean()) if len(accepted) else None
        median = float(accepted["net_bps"].median()) if len(accepted) else None
        lcb = cluster_lcb(accepted, config, f"v7:{event_id}:overall")
        positive_folds = int((fold_means > 0.0).sum())
        worst_fold_mean = float(fold_means.min()) if len(fold_means) else None
        median_fold_lcb = float(fold_lcbs.median()) if len(fold_lcbs) else None
        failed = []
        if not event_folds["incidence_passed"].all():
            failed.append("minimum_events_per_fold")
        if positive_folds < int(config["minimum_positive_folds"]):
            failed.append("minimum_positive_folds")
        for condition, name in (
            (mean is not None and mean > 0.0, "nonpositive_overall_mean"),
            (median is not None and median > 0.0, "nonpositive_overall_median"),
            (lcb is not None and lcb > 0.0, "nonpositive_overall_lcb"),
            (worst_fold_mean is not None and worst_fold_mean > 0.0, "nonpositive_worst_fold_mean"),
            (median_fold_lcb is not None and median_fold_lcb > 0.0, "nonpositive_median_fold_lcb"),
        ):
            if not condition:
                failed.append(name)
        summary_rows.append(
            {
                "event_id": event_id,
                "events": int(len(accepted)),
                "mean_net_bps": mean,
                "median_net_bps": median,
                "lcb_net_bps": lcb,
                "positive_folds": positive_folds,
                "worst_fold_mean_bps": worst_fold_mean,
                "median_fold_lcb_bps": median_fold_lcb,
                "passed": not failed,
                "failed_gates": failed,
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows)


def slice_metrics(ledger: pd.DataFrame, config: Mapping[str, Any], column: str) -> pd.DataFrame:
    rows = []
    for (event_id, value), subset in ledger.groupby(["event_id", column], sort=True, observed=True):
        rows.append(
            {
                "event_id": str(event_id),
                column: str(value),
                "events": int(len(subset)),
                "mean_net_bps": float(subset["net_bps"].mean()),
                "median_net_bps": float(subset["net_bps"].median()),
                "lcb_net_bps": cluster_lcb(subset, config, f"v7:{event_id}:{column}:{value}"),
            }
        )
    return pd.DataFrame(rows)


def assert_evolution_authority(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != ACCEPTED_EVENT_STATUS
        or payload.get("evolution_allowed") is not True
        or not payload.get("passing_event_ids")
    ):
        raise PermissionError("V7 evolution blocked: liquidation event capacity was not accepted")
    if any(
        payload.get(key) is not False
        for key in ("validation_opened", "oos_opened", "paper_allowed", "live_allowed", "orders_enabled")
    ):
        raise PermissionError("V7 evolution authority is not fail-closed")
    return payload
