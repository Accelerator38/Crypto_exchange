from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .fitness_v4 import contiguous_block_lcb


EVENT_IDS = (
    "oi_taker_price_continuation",
    "crowding_unwind_reversal",
    "oi_price_divergence_reversal",
)

ACCEPTED_EVENT_STATUS = "V6_DATA_SOURCE_AND_EVENT_CAPACITY_ACCEPTED"


def assert_evolution_authority(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != ACCEPTED_EVENT_STATUS
        or payload.get("evolution_allowed") is not True
        or not payload.get("passing_event_ids")
    ):
        raise PermissionError(
            "V6 evolution blocked: preregistered stress-costed event capacity was not accepted"
        )
    if any(
        payload.get(key) is not False
        for key in ("validation_opened", "oos_opened", "paper_allowed", "live_allowed", "orders_enabled")
    ):
        raise PermissionError("V6 evolution authority is not fail-closed")
    return payload


def _seed(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def apply_cooldown(mask: pd.Series, bars: int) -> pd.Series:
    selected = np.zeros(len(mask), dtype=bool)
    last = -10**9
    for index, active in enumerate(mask.fillna(False).to_numpy(dtype=bool)):
        if active and index - last > int(bars):
            selected[index] = True
            last = index
    return pd.Series(selected, index=mask.index)


def build_event_ledger(frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    window = int(config["trailing_window_bars"])
    minimum = int(config["minimum_trailing_bars"])
    cooldown = int(config["event_cooldown_bars"])
    cost = float(config["stress_round_trip_cost_bps"])
    ledgers: list[pd.DataFrame] = []
    for symbol, source in frame.groupby("symbol", sort=True, observed=True):
        local = source.sort_values("timestamp", kind="stable").reset_index(drop=True).copy()
        abs_oi = local["oi_change_4h"].abs()
        abs_flow = local["kline_taker_imbalance_4h"].abs()
        abs_return = local["return_3"].abs()
        local["oi_abs_p90"] = abs_oi.rolling(window, min_periods=minimum).quantile(0.90).shift(1)
        local["flow_abs_p75"] = abs_flow.rolling(window, min_periods=minimum).quantile(0.75).shift(1)
        local["oi_p10"] = local["oi_change_4h"].rolling(window, min_periods=minimum).quantile(0.10).shift(1)
        local["return_abs_p90"] = abs_return.rolling(window, min_periods=minimum).quantile(0.90).shift(1)
        account_mean = local["account_long_short_log"].rolling(window, min_periods=minimum).mean().shift(1)
        account_std = local["account_long_short_log"].rolling(window, min_periods=minimum).std(ddof=0).shift(1)
        local["account_long_short_z"] = (
            (local["account_long_short_log"] - account_mean) / account_std.replace(0.0, np.nan)
        )
        price_direction = np.sign(local["return_3"])
        flow_direction = np.sign(local["kline_taker_imbalance_4h"])
        crowded_direction = np.sign(local["account_long_short_log"])
        definitions = {
            "oi_taker_price_continuation": (
                (abs_oi >= local["oi_abs_p90"])
                & (abs_flow >= local["flow_abs_p75"])
                & (price_direction != 0)
                & (price_direction == flow_direction),
                price_direction,
            ),
            "crowding_unwind_reversal": (
                (local["account_long_short_z"].abs() >= 2.0)
                & (local["oi_change_4h"] <= local["oi_p10"])
                & (crowded_direction != 0)
                & (price_direction == crowded_direction),
                -crowded_direction,
            ),
            "oi_price_divergence_reversal": (
                (abs_return >= local["return_abs_p90"])
                & (price_direction != 0)
                & (np.sign(local["oi_change_4h"]) == -price_direction),
                -price_direction,
            ),
        }
        local["entry_timestamp"] = local["timestamp"].shift(-1)
        local["entry_price"] = local["open"].shift(-1)
        local["exit_timestamp"] = local["timestamp"].shift(-7)
        local["exit_price"] = local["open"].shift(-7)
        for event_id, (mask, direction) in definitions.items():
            selected = apply_cooldown(mask, cooldown)
            event = local.loc[selected].copy()
            event["event_id"] = event_id
            event["direction_value"] = direction.loc[selected].astype("int8")
            event = event.loc[
                event[["entry_timestamp", "entry_price", "exit_timestamp", "exit_price"]].notna().all(axis=1)
            ].copy()
            event["gross_bps"] = (
                event["direction_value"] * (event["exit_price"] / event["entry_price"] - 1.0) * 10_000.0
            )
            event["net_bps"] = event["gross_bps"] - cost
            event["direction"] = np.where(event["direction_value"] > 0, "LONG", "SHORT")
            event["symbol"] = str(symbol)
            ledgers.append(
                event.loc[
                    : ,
                    [
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
                        "oi_change_4h",
                        "account_long_short_z",
                        "kline_taker_imbalance_4h",
                        "return_3",
                    ],
                ]
            )
    if not ledgers:
        return pd.DataFrame()
    return pd.concat(ledgers, ignore_index=True).sort_values(["timestamp", "event_id", "symbol"], kind="stable")


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
    fold_rows = []
    summary_rows = []
    minimum_events = int(config["minimum_events_per_fold"])
    for event_id in EVENT_IDS:
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
                    "lcb_net_bps": cluster_lcb(subset, config, f"v6:{event_id}:{fold['name']}"),
                    "incidence_passed": len(subset) >= minimum_events,
                }
            )
        accepted = pd.concat(accepted_parts, ignore_index=True) if accepted_parts else pd.DataFrame()
        event_folds = pd.DataFrame(fold_rows).loc[lambda value: value["event_id"] == event_id]
        fold_means = event_folds["mean_net_bps"].dropna().astype(float)
        fold_lcbs = event_folds["lcb_net_bps"].dropna().astype(float)
        overall_mean = float(accepted["net_bps"].mean()) if len(accepted) else None
        overall_median = float(accepted["net_bps"].median()) if len(accepted) else None
        overall_lcb = cluster_lcb(accepted, config, f"v6:{event_id}:overall")
        positive_folds = int((fold_means > 0.0).sum())
        worst_fold_mean = float(fold_means.min()) if len(fold_means) else None
        median_fold_lcb = float(fold_lcbs.median()) if len(fold_lcbs) else None
        failed = []
        if not event_folds["incidence_passed"].all():
            failed.append("minimum_events_per_fold")
        if positive_folds < int(config["minimum_positive_folds"]):
            failed.append("minimum_positive_folds")
        if overall_mean is None or overall_mean <= 0.0:
            failed.append("nonpositive_overall_mean")
        if overall_median is None or overall_median <= 0.0:
            failed.append("nonpositive_overall_median")
        if overall_lcb is None or overall_lcb <= 0.0:
            failed.append("nonpositive_overall_lcb")
        if worst_fold_mean is None or worst_fold_mean <= 0.0:
            failed.append("nonpositive_worst_fold_mean")
        if median_fold_lcb is None or median_fold_lcb <= 0.0:
            failed.append("nonpositive_median_fold_lcb")
        summary_rows.append(
            {
                "event_id": event_id,
                "events": int(len(accepted)),
                "mean_net_bps": overall_mean,
                "median_net_bps": overall_median,
                "lcb_net_bps": overall_lcb,
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
                "lcb_net_bps": cluster_lcb(subset, config, f"v6:{event_id}:{column}:{value}"),
            }
        )
    return pd.DataFrame(rows)
