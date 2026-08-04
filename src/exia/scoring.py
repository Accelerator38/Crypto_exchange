from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd


MIN_SCOPE_TRADES = 10
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_ALPHA = 0.05


def deterministic_seed(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def moving_block_lcb(
    values: pd.Series,
    timestamps: pd.Series,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    alpha: float = BOOTSTRAP_ALPHA,
    seed: int = 0,
) -> float | None:
    """Bootstrap UTC-day clusters so simultaneous symbol trades stay together."""
    if len(values) < 2 or len(values) != len(timestamps):
        return None
    frame = pd.DataFrame(
        {
            "value": pd.to_numeric(values, errors="raise").astype("float64"),
            "timestamp": pd.to_numeric(timestamps, errors="raise").astype("int64"),
        }
    )
    frame["day"] = pd.to_datetime(
        frame["timestamp"], unit="ms", utc=True
    ).dt.floor("1D")
    blocks = [group["value"].to_numpy(dtype="float64") for _, group in frame.groupby("day", sort=True)]
    if len(blocks) < 2:
        return None
    rng = np.random.default_rng(seed)
    block_sums = np.asarray([block.sum() for block in blocks], dtype="float64")
    block_counts = np.asarray([len(block) for block in blocks], dtype="float64")
    selected = rng.integers(0, len(blocks), size=(samples, len(blocks)))
    means = block_sums[selected].sum(axis=1) / block_counts[selected].sum(axis=1)
    return float(np.quantile(means, alpha))


def summarize_scope(
    ledger: pd.DataFrame,
    *,
    seed_key: str,
    max_drawdown: float | None = None,
    baseline_mean_bps: float = 0.0,
) -> dict[str, Any]:
    closed_trades = len(ledger)
    fills = int(ledger["fills"].sum()) if closed_trades else 0
    if not closed_trades:
        return {
            "closed_trades": 0,
            "fills": 0,
            "mean_net_bps": None,
            "block_lcb_net_bps": None,
            "baseline_differential_lcb_bps": None,
            "max_drawdown": max_drawdown,
            "status": "INSUFFICIENT",
        }

    values = ledger["net_bps"].astype("float64")
    if closed_trades < MIN_SCOPE_TRADES:
        return {
            "closed_trades": closed_trades,
            "fills": fills,
            "mean_net_bps": float(values.mean()),
            "block_lcb_net_bps": None,
            "baseline_differential_lcb_bps": None,
            "max_drawdown": max_drawdown,
            "status": "INSUFFICIENT",
        }
    differential = values - float(baseline_mean_bps)
    lcb = moving_block_lcb(
        values,
        ledger["signal_timestamp"],
        seed=deterministic_seed(seed_key),
    )
    differential_lcb = moving_block_lcb(
        differential,
        ledger["signal_timestamp"],
        seed=deterministic_seed(f"{seed_key}:baseline"),
    )
    if lcb is None or differential_lcb is None:
        status = "INSUFFICIENT"
    elif float(values.mean()) > 0 and lcb > 0 and differential_lcb > 0:
        status = "PASS"
    else:
        status = "FAIL"
    return {
        "closed_trades": closed_trades,
        "fills": fills,
        "mean_net_bps": float(values.mean()),
        "block_lcb_net_bps": lcb,
        "baseline_differential_lcb_bps": differential_lcb,
        "max_drawdown": max_drawdown,
        "status": status,
    }


def catalog_rows_for_run(
    ledger: pd.DataFrame,
    *,
    identity: dict[str, str],
    split: str,
    cost_scenario: str,
    max_drawdown: float,
    states: tuple[str, ...],
    symbols: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def append(scope: str, value: str, subset: pd.DataFrame) -> None:
        key = (
            f"{identity['experiment_id']}:{split}:{cost_scenario}:"
            f"{scope}:{value}"
        )
        summary = summarize_scope(
            subset,
            seed_key=key,
            max_drawdown=max_drawdown if scope == "overall" else None,
        )
        rows.append(
            {
                **identity,
                "split": split,
                "cost_scenario": cost_scenario,
                "scope": scope,
                "scope_value": value,
                **summary,
            }
        )

    append("overall", "all", ledger)
    for state in states:
        append("state", state, ledger.loc[ledger["market_state"] == state])
    for direction in ("LONG", "SHORT"):
        append(
            "direction",
            direction,
            ledger.loc[ledger["direction"] == direction],
        )
    for symbol in symbols:
        append("symbol", symbol, ledger.loc[ledger["symbol"] == symbol])
    return rows
