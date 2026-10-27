from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_SHORT = -1
TARGET_FLAT = 0
TARGET_LONG = 1


def _strict_row_median_three(values: np.ndarray) -> np.ndarray:
    """Median of exactly three horizons, propagating NaN without nanmedian."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("strict label median requires exactly three columns")
    first, second, third = array[:, 0], array[:, 1], array[:, 2]
    median = np.maximum(
        np.minimum(first, second),
        np.minimum(np.maximum(first, second), third),
    )
    missing = np.isnan(first) | np.isnan(second) | np.isnan(third)
    return np.where(missing, np.nan, median)


def build_forward_labels(
    panel: pd.DataFrame,
    *,
    horizons_bars: tuple[int, ...] = (1, 3, 6),
    round_trip_cost_bps: float,
    minimum_edge_bps: float,
    minimum_votes: int = 2,
) -> pd.DataFrame:
    """Create train-only targets from next-open costed returns."""
    if tuple(horizons_bars) != (1, 3, 6):
        raise ValueError("Genetic_Primus V1 label horizons must be 1/3/6")
    if round_trip_cost_bps <= 0:
        raise ValueError("label costs must be positive")
    rows: list[pd.DataFrame] = []
    for symbol, raw in panel.groupby("symbol", sort=True, observed=True):
        group = raw.sort_values("timestamp", kind="stable").copy()
        entry_open = group["open"].shift(-1)
        long_columns: list[str] = []
        short_columns: list[str] = []
        for horizon in horizons_bars:
            exit_open = group["open"].shift(-(1 + horizon))
            gross = (exit_open / entry_open - 1.0) * 10_000.0
            long_name = f"long_net_h{horizon}_bps"
            short_name = f"short_net_h{horizon}_bps"
            group[long_name] = gross - float(round_trip_cost_bps)
            group[short_name] = -gross - float(round_trip_cost_bps)
            long_columns.append(long_name)
            short_columns.append(short_name)
        long_array = group[long_columns].to_numpy(dtype=np.float64, copy=True)
        short_array = group[short_columns].to_numpy(dtype=np.float64, copy=True)
        group["long_utility_bps"] = _strict_row_median_three(long_array)
        group["short_utility_bps"] = _strict_row_median_three(short_array)
        edge = float(minimum_edge_bps)
        long_votes = (
            (long_array[:, 0] > edge).astype(np.int8)
            + (long_array[:, 1] > edge).astype(np.int8)
            + (long_array[:, 2] > edge).astype(np.int8)
        )
        short_votes = (
            (short_array[:, 0] > edge).astype(np.int8)
            + (short_array[:, 1] > edge).astype(np.int8)
            + (short_array[:, 2] > edge).astype(np.int8)
        )
        best_long = (group["long_utility_bps"] > group["short_utility_bps"]) & (
            long_votes >= int(minimum_votes)
        )
        best_short = (group["short_utility_bps"] > group["long_utility_bps"]) & (
            short_votes >= int(minimum_votes)
        )
        group["target_position"] = np.select(
            [best_short, best_long], [TARGET_SHORT, TARGET_LONG], default=TARGET_FLAT
        ).astype("int8")
        group["label_supported"] = ~(
            np.isnan(long_array).any(axis=1) | np.isnan(short_array).any(axis=1)
        )
        group.loc[~group["label_supported"], "target_position"] = TARGET_FLAT
        group["symbol"] = str(symbol)
        rows.append(group)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["timestamp", "symbol"], kind="stable"
    ).reset_index(drop=True)
