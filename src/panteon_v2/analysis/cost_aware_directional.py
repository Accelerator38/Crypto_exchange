"""Anchored walk-forward cost-aware directional classifier for Bitget."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


REPORT_SCHEMA_VERSION = "panteon.bitget_cost_aware_directional.v1"
EXPECTED_SYMBOLS = (
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "LINK/USDT",
)
DECISION_CADENCE_MINUTES = 5
HORIZON_MINUTES = 60
GROSS_BARRIER_BPS = 20.0
MEASURED_MAKER_TAKER_COST_BPS = 8.0
MAX_DRAWDOWN_BPS = 2_000.0
MAX_SYMBOL_TRADE_SHARE = 0.50
MIN_DIRECTION_TRADE_SHARE = 0.10
MIN_TRAIN_DAYS = 14
VALIDATION_DAYS = 7
OOS_DAYS = 7
THRESHOLD_GRID = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
LCB_Z = 1.6448536269514722
FEATURE_NAMES = (
    "return_1m_bps",
    "return_5m_bps",
    "return_15m_bps",
    "return_30m_bps",
    "return_60m_bps",
    "realized_vol_5m_bps",
    "realized_vol_15m_bps",
    "realized_vol_60m_bps",
    "volume_ratio_5m",
    "volume_ratio_15m",
    "volume_ratio_60m",
    "range_bps",
    "body_bps",
    "close_location",
    "distance_high_60_bps",
    "distance_low_60_bps",
    "market_breadth_5m",
    "btc_return_5m_bps",
    "hour_sin",
    "hour_cos",
)


def evaluate_cost_aware_directional(dataset_dir: str | Path) -> dict[str, Any]:
    history, source = _load_history(dataset_dir)
    labeled, feature_names = _build_labeled_dataset(history)
    folds = _anchored_folds(labeled)
    if not folds:
        raise ValueError("not enough complete days for anchored walk-forward")

    fold_reports = []
    all_oos_trades: list[dict[str, Any]] = []
    all_baseline_trades: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(folds, start=1):
        train = fold["train"]
        validation = fold["validation"]
        oos = fold["oos"]
        scaler, model = _fit_classifier(train, feature_names)
        validation_scored = _score_rows(
            validation,
            feature_names=feature_names,
            scaler=scaler,
            model=model,
        )
        threshold_rows = []
        for threshold in THRESHOLD_GRID:
            trades = _select_model_trades(validation_scored, threshold)
            metrics = _metrics(trades)
            threshold_rows.append(
                {
                    "threshold": threshold,
                    "metrics": metrics,
                    "passed": _split_passed(metrics),
                }
            )
        selected_threshold = _select_validation_threshold(threshold_rows)
        oos_scored = _score_rows(
            oos,
            feature_names=feature_names,
            scaler=scaler,
            model=model,
        )
        validation_trades = _select_model_trades(
            validation_scored,
            float(selected_threshold["threshold"]),
        )
        oos_trades = _select_model_trades(
            oos_scored,
            float(selected_threshold["threshold"]),
        )
        baseline_trades = _select_momentum_baseline(oos)
        all_oos_trades.extend(oos_trades)
        all_baseline_trades.extend(baseline_trades)
        fold_reports.append(
            {
                "fold": fold_index,
                "train_dates": fold["train_dates"],
                "validation_dates": fold["validation_dates"],
                "oos_dates": fold["oos_dates"],
                "train_rows": len(train),
                "validation_rows": len(validation),
                "oos_rows": len(oos),
                "train_classes": dict(
                    sorted(
                        Counter(int(value) for value in train["label_class"]).items()
                    )
                ),
                "threshold_search": threshold_rows,
                "selected_threshold": selected_threshold["threshold"],
                "validation_gate_passed": bool(selected_threshold["passed"]),
                "validation_metrics": _metrics(validation_trades),
                "oos_metrics": _metrics(oos_trades),
                "baseline_oos_metrics": _metrics(baseline_trades),
                "top_coefficients": _top_coefficients(
                    model=model,
                    feature_names=feature_names,
                ),
            }
        )

    aggregate = _metrics(all_oos_trades)
    baseline = _metrics(all_baseline_trades)
    failures = []
    if aggregate["fills"] < 20:
        failures.append("oos_fills_below_20")
    if aggregate["closed_trades"] < 10:
        failures.append("oos_closed_trades_below_10")
    if aggregate["mean_net_bps"] is None or aggregate["mean_net_bps"] <= 0.0:
        failures.append("oos_nonpositive_costed_expectancy")
    if aggregate["lcb_95_net_bps"] is None or aggregate["lcb_95_net_bps"] <= 0.0:
        failures.append("oos_nonpositive_lcb")
    if aggregate["max_drawdown_bps"] > MAX_DRAWDOWN_BPS:
        failures.append("oos_drawdown_above_20pct")
    if aggregate["top_symbol_trade_share"] > MAX_SYMBOL_TRADE_SHARE:
        failures.append("oos_symbol_concentration_above_50pct")
    if (
        aggregate["closed_trades"]
        and aggregate["minimum_direction_trade_share"] < MIN_DIRECTION_TRADE_SHARE
    ):
        failures.append("oos_direction_collapse")
    if any(not row["validation_gate_passed"] for row in fold_reports):
        failures.append("validation_gate_collapse")
    collapsed_folds = [
        row["fold"]
        for row in fold_reports
        if row["oos_metrics"]["mean_net_bps"] is None
        or row["oos_metrics"]["mean_net_bps"] <= 0.0
    ]
    failures.extend(f"oos_fold_expectancy_collapse:{fold}" for fold in collapsed_folds)
    if (
        baseline["mean_net_bps"] is not None
        and aggregate["mean_net_bps"] is not None
        and aggregate["mean_net_bps"] <= baseline["mean_net_bps"]
    ):
        failures.append("does_not_beat_momentum_baseline")

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "fixed_contract": {
            "decision_cadence_minutes": DECISION_CADENCE_MINUTES,
            "horizon_minutes": HORIZON_MINUTES,
            "gross_barrier_bps": GROSS_BARRIER_BPS,
            "measured_maker_taker_cost_bps": MEASURED_MAKER_TAKER_COST_BPS,
            "max_drawdown_bps": MAX_DRAWDOWN_BPS,
            "max_symbol_trade_share": MAX_SYMBOL_TRADE_SHARE,
            "minimum_direction_trade_share": MIN_DIRECTION_TRADE_SHARE,
            "successful_barrier_net_bps": (
                GROSS_BARRIER_BPS - MEASURED_MAKER_TAKER_COST_BPS
            ),
            "failed_barrier_net_bps": -(
                GROSS_BARRIER_BPS + MEASURED_MAKER_TAKER_COST_BPS
            ),
            "anchored_days": {
                "minimum_train": MIN_TRAIN_DAYS,
                "validation": VALIDATION_DAYS,
                "oos": OOS_DAYS,
            },
            "purged_split_boundaries": True,
            "threshold_grid": list(THRESHOLD_GRID),
            "threshold_selected_on": "validation_only",
            "global_nonoverlapping_position": True,
            "model": "balanced_multinomial_logistic_regression",
            "features": feature_names,
        },
        "dataset": {
            "history_rows": len(history),
            "labeled_decisions": len(labeled),
            "first_timestamp_ms": int(labeled["timestamp"].min()),
            "last_timestamp_ms": int(labeled["timestamp"].max()),
            "symbols": dict(sorted(Counter(labeled["symbol"]).items())),
            "classes": dict(
                sorted(Counter(int(value) for value in labeled["label_class"]).items())
            ),
            "ambiguous_rows_removed": int(
                labeled.attrs.get("ambiguous_rows_removed", 0)
            ),
        },
        "folds": fold_reports,
        "aggregate_oos": aggregate,
        "baseline_oos": baseline,
        "failures": failures,
        "verdict": "eligible_for_prospective_microstructure_validation"
        if not failures
        else "rejected",
        "research_only": True,
        "runtime_profile_created": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }


def render_directional_markdown(report: Mapping[str, Any]) -> str:
    aggregate = report["aggregate_oos"]
    baseline = report["baseline_oos"]
    lines = [
        "# Bitget cost-aware directional classifier",
        "",
        "## Verdict",
        "",
        f"**{str(report['verdict']).upper()}**",
        "",
        (
            f"Anchored OOS trades: {aggregate['closed_trades']}; mean gross: "
            f"{_fmt(aggregate['mean_gross_bps'])} bps; fixed cost: "
            f"{aggregate['mean_cost_bps']:.3f} bps; mean net: "
            f"{_fmt(aggregate['mean_net_bps'])} bps; LCB: "
            f"{_fmt(aggregate['lcb_95_net_bps'])} bps. Momentum baseline: "
            f"{_fmt(baseline['mean_net_bps'])} bps."
        ),
        "",
        "| Fold | Train | Validation | OOS | Threshold | Val net | "
        "OOS trades | OOS net | OOS LCB |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for fold in report["folds"]:
        lines.append(
            f"| {fold['fold']} | {fold['train_dates'][0]}.."
            f"{fold['train_dates'][1]} | {fold['validation_dates'][0]}.."
            f"{fold['validation_dates'][1]} | {fold['oos_dates'][0]}.."
            f"{fold['oos_dates'][1]} | {fold['selected_threshold']:.2f} | "
            f"{_fmt(fold['validation_metrics']['mean_net_bps'])} | "
            f"{fold['oos_metrics']['closed_trades']} | "
            f"{_fmt(fold['oos_metrics']['mean_net_bps'])} | "
            f"{_fmt(fold['oos_metrics']['lcb_95_net_bps'])} |"
        )
    lines.extend(
        [
            "",
            "Failures: " + (", ".join(report["failures"]) or "none"),
            "",
            (
                "The model uses closed-candle features only. Validation chooses "
                "the probability threshold; OOS never changes the model."
            ),
            "",
            "No runtime policy, paper order or live order is created.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_history(dataset_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = Path(dataset_dir).resolve()
    manifest_path = source / "integrity_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("exchange") != "BITGET":
        raise ValueError("BITGET history is required")
    if manifest.get("timeframe") != "1m":
        raise ValueError("true 1m history is required")
    if not bool((manifest.get("validation") or {}).get("passed")):
        raise ValueError("history integrity validation did not pass")
    requested = tuple(str(item) for item in manifest.get("requested_symbols") or ())
    if set(requested) != set(EXPECTED_SYMBOLS):
        raise ValueError("history must contain exact Bitget full8 symbols")
    coverage = manifest.get("coverage") or {}
    if set(coverage) != set(EXPECTED_SYMBOLS):
        raise ValueError("history coverage must contain exact Bitget full8 symbols")
    incomplete = [
        symbol
        for symbol in EXPECTED_SYMBOLS
        if int((coverage[symbol] or {}).get("head_missing_bars_from_requested_start") or 0)
        or int((coverage[symbol] or {}).get("tail_missing_bars_to_requested_end") or 0)
        or int((coverage[symbol] or {}).get("missing_bars_inside_coverage") or 0)
    ]
    if incomplete:
        raise ValueError(f"history coverage is incomplete: {sorted(incomplete)}")
    frames = []
    file_hashes = []
    for item in manifest.get("files") or ():
        path = source / Path(str(item["path"])).name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = _sha256_file(path)
        if digest != str(item["sha256"]):
            raise ValueError(f"history file hash mismatch: {path.name}")
        file_hashes.append(digest)
        frames.append(pd.read_csv(path))
    dataset_sha = hashlib.sha256("\n".join(file_hashes).encode("ascii")).hexdigest()
    if dataset_sha != str(manifest.get("dataset_sha256")):
        raise ValueError("history dataset hash mismatch")
    history = pd.concat(frames, ignore_index=True)
    required = {"timestamp", "open", "high", "low", "close", "volume", "symbol"}
    if not required.issubset(history.columns):
        raise ValueError("history OHLCV columns are incomplete")
    history = history[list(required)].copy()
    history["timestamp"] = pd.to_numeric(history["timestamp"], errors="raise").astype("int64")
    for column in ("open", "high", "low", "close", "volume"):
        history[column] = pd.to_numeric(history[column], errors="coerce")
    history["symbol"] = history["symbol"].astype(str)
    history = history.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    if history.duplicated(["symbol", "timestamp"]).any():
        raise ValueError("history contains duplicate symbol timestamps")
    return history, {
        "dataset_dir": str(source),
        "manifest_sha256": _sha256_file(manifest_path),
        "dataset_sha256": dataset_sha,
        "requested_start_date": manifest["requested_start_date"],
        "requested_end_date": manifest["requested_end_date"],
        "timeframe": "1m",
        "exchange": "BITGET",
    }


def _build_labeled_dataset(history: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = history.copy()
    grouped = frame.groupby("symbol", sort=False)
    for minutes in (1, 5, 15, 30, 60):
        frame[f"return_{minutes}m_bps"] = (
            frame["close"] / grouped["close"].shift(minutes) - 1.0
        ) * 10_000.0
    returns_1m = frame["return_1m_bps"]
    for minutes in (5, 15, 60):
        frame[f"realized_vol_{minutes}m_bps"] = (
            returns_1m.groupby(frame["symbol"], sort=False)
            .rolling(minutes, min_periods=minutes)
            .std(ddof=0)
            .reset_index(level=0, drop=True)
        )
        prior_volume = grouped["volume"].shift(1)
        baseline = (
            prior_volume.groupby(frame["symbol"], sort=False)
            .rolling(minutes, min_periods=minutes)
            .mean()
            .reset_index(level=0, drop=True)
        )
        frame[f"volume_ratio_{minutes}m"] = frame["volume"] / baseline
    frame["range_bps"] = (frame["high"] - frame["low"]) / frame["close"] * 10_000.0
    frame["body_bps"] = (frame["close"] - frame["open"]) / frame["open"] * 10_000.0
    frame["close_location"] = np.where(
        frame["high"] > frame["low"],
        (frame["close"] - frame["low"]) / (frame["high"] - frame["low"]),
        0.5,
    )
    rolling_high = (
        grouped["high"].rolling(60, min_periods=60).max().reset_index(level=0, drop=True)
    )
    rolling_low = (
        grouped["low"].rolling(60, min_periods=60).min().reset_index(level=0, drop=True)
    )
    frame["distance_high_60_bps"] = (frame["close"] / rolling_high - 1.0) * 10_000.0
    frame["distance_low_60_bps"] = (frame["close"] / rolling_low - 1.0) * 10_000.0
    breadth = (
        (frame["return_5m_bps"] > 0.0)
        .groupby(frame["timestamp"])
        .mean()
        .rename("market_breadth_5m")
    )
    frame = frame.join(breadth, on="timestamp")
    btc = (
        frame.loc[frame["symbol"] == "BTC/USDT", ["timestamp", "return_5m_bps"]]
        .set_index("timestamp")["return_5m_bps"]
        .rename("btc_return_5m_bps")
    )
    frame = frame.join(btc, on="timestamp")
    utc = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    hour = utc.dt.hour + utc.dt.minute / 60.0
    frame["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    frame["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    frame["date"] = utc.dt.strftime("%Y-%m-%d")

    decision_mask = (
        frame["timestamp"] % (DECISION_CADENCE_MINUTES * 60_000) == 0
    )
    decisions = frame.loc[decision_mask].copy()
    labels = []
    ambiguous = 0
    by_symbol = {
        symbol: rows.reset_index(drop=True)
        for symbol, rows in frame.groupby("symbol", sort=False)
    }
    for symbol, rows in by_symbol.items():
        decision_indices = np.flatnonzero(
            rows["timestamp"].to_numpy(dtype=np.int64)
            % (DECISION_CADENCE_MINUTES * 60_000)
            == 0
        )
        highs = rows["high"].to_numpy(dtype=float)
        lows = rows["low"].to_numpy(dtype=float)
        closes = rows["close"].to_numpy(dtype=float)
        timestamps = rows["timestamp"].to_numpy(dtype=np.int64)
        for index in decision_indices:
            if index + HORIZON_MINUTES >= len(rows):
                continue
            if timestamps[index + HORIZON_MINUTES] - timestamps[index] != HORIZON_MINUTES * 60_000:
                continue
            entry = closes[index]
            upper = entry * (1.0 + GROSS_BARRIER_BPS / 10_000.0)
            lower = entry * (1.0 - GROSS_BARRIER_BPS / 10_000.0)
            label_class = 0
            exit_index = index + HORIZON_MINUTES
            is_ambiguous = False
            for future in range(index + 1, index + HORIZON_MINUTES + 1):
                hit_upper = highs[future] >= upper
                hit_lower = lows[future] <= lower
                if hit_upper and hit_lower:
                    is_ambiguous = True
                    break
                if hit_upper:
                    label_class = 1
                    exit_index = future
                    break
                if hit_lower:
                    label_class = -1
                    exit_index = future
                    break
            if is_ambiguous:
                ambiguous += 1
                continue
            if label_class == 1:
                long_net = GROSS_BARRIER_BPS - MEASURED_MAKER_TAKER_COST_BPS
                short_net = -(GROSS_BARRIER_BPS + MEASURED_MAKER_TAKER_COST_BPS)
            elif label_class == -1:
                long_net = -(GROSS_BARRIER_BPS + MEASURED_MAKER_TAKER_COST_BPS)
                short_net = GROSS_BARRIER_BPS - MEASURED_MAKER_TAKER_COST_BPS
            else:
                gross = (closes[exit_index] / entry - 1.0) * 10_000.0
                long_net = gross - MEASURED_MAKER_TAKER_COST_BPS
                short_net = -gross - MEASURED_MAKER_TAKER_COST_BPS
            labels.append(
                {
                    "timestamp": int(timestamps[index]),
                    "symbol": symbol,
                    "label_class": label_class,
                    "exit_timestamp_ms": int(timestamps[exit_index]),
                    "long_net_bps": long_net,
                    "short_net_bps": short_net,
                }
            )
    labels_frame = pd.DataFrame(labels)
    decisions = decisions.merge(labels_frame, on=["timestamp", "symbol"], how="inner")
    symbol_features = pd.get_dummies(
        decisions["symbol"],
        prefix="symbol",
        dtype=float,
    )
    decisions = pd.concat([decisions, symbol_features], axis=1)
    feature_names = list(FEATURE_NAMES) + list(symbol_features.columns)
    decisions = decisions.replace([np.inf, -np.inf], np.nan)
    decisions = decisions.dropna(subset=feature_names).reset_index(drop=True)
    decisions.attrs["ambiguous_rows_removed"] = ambiguous
    return decisions, feature_names


def _anchored_folds(labeled: pd.DataFrame) -> list[dict[str, Any]]:
    dates = sorted(str(value) for value in labeled["date"].unique())
    folds = []
    train_days = MIN_TRAIN_DAYS
    while train_days + VALIDATION_DAYS + OOS_DAYS <= len(dates):
        train_dates = dates[:train_days]
        validation_dates = dates[train_days : train_days + VALIDATION_DAYS]
        oos_dates = dates[
            train_days + VALIDATION_DAYS : train_days + VALIDATION_DAYS + OOS_DAYS
        ]
        validation_start = _date_start_ms(validation_dates[0])
        oos_start = _date_start_ms(oos_dates[0])
        oos_end = _date_start_ms(_next_date(oos_dates[-1]))
        train = labeled[
            labeled["date"].isin(train_dates)
            & (labeled["exit_timestamp_ms"] < validation_start)
        ].copy()
        validation = labeled[
            labeled["date"].isin(validation_dates)
            & (labeled["exit_timestamp_ms"] < oos_start)
        ].copy()
        oos = labeled[
            labeled["date"].isin(oos_dates)
            & (labeled["exit_timestamp_ms"] < oos_end)
        ].copy()
        if not train.empty and not validation.empty and not oos.empty:
            folds.append(
                {
                    "train": train,
                    "validation": validation,
                    "oos": oos,
                    "train_dates": [train_dates[0], train_dates[-1]],
                    "validation_dates": [validation_dates[0], validation_dates[-1]],
                    "oos_dates": [oos_dates[0], oos_dates[-1]],
                }
            )
        train_days += OOS_DAYS
    return folds


def _fit_classifier(
    train: pd.DataFrame,
    feature_names: Sequence[str],
) -> tuple[StandardScaler, LogisticRegression]:
    classes = set(int(value) for value in train["label_class"])
    if classes != {-1, 0, 1}:
        raise ValueError(f"all three training classes are required, got {classes}")
    scaler = StandardScaler()
    matrix = scaler.fit_transform(train[list(feature_names)].to_numpy(dtype=float))
    model = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        max_iter=300,
        solver="lbfgs",
        random_state=20260731,
    )
    model.fit(matrix, train["label_class"].to_numpy(dtype=int))
    return scaler, model


def _score_rows(
    rows: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    scaler: StandardScaler,
    model: LogisticRegression,
) -> pd.DataFrame:
    scored = rows.copy()
    matrix = scaler.transform(scored[list(feature_names)].to_numpy(dtype=float))
    probabilities = model.predict_proba(matrix)
    class_index = {int(value): index for index, value in enumerate(model.classes_)}
    scored["prob_short"] = probabilities[:, class_index[-1]]
    scored["prob_no_trade"] = probabilities[:, class_index[0]]
    scored["prob_long"] = probabilities[:, class_index[1]]
    return scored


def _select_validation_threshold(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    passed = [row for row in rows if bool(row["passed"])]
    pool = passed or [
        row for row in rows if int(row["metrics"]["closed_trades"]) >= 10
    ] or list(rows)
    return max(
        pool,
        key=lambda row: (
            row["metrics"]["lcb_95_net_bps"] is not None,
            row["metrics"]["lcb_95_net_bps"] or -math.inf,
            row["metrics"]["mean_net_bps"] or -math.inf,
            -float(row["threshold"]),
        ),
    )


def _select_model_trades(rows: pd.DataFrame, threshold: float) -> list[dict[str, Any]]:
    candidates = []
    for row in rows.to_dict("records"):
        if float(row["prob_long"]) >= float(row["prob_short"]):
            direction = "LONG"
            score = float(row["prob_long"])
            net = float(row["long_net_bps"])
        else:
            direction = "SHORT"
            score = float(row["prob_short"])
            net = float(row["short_net_bps"])
        if score < threshold or score <= float(row["prob_no_trade"]):
            continue
        candidates.append(
            {
                "timestamp": int(row["timestamp"]),
                "exit_timestamp_ms": int(row["exit_timestamp_ms"]),
                "symbol": str(row["symbol"]),
                "direction": direction,
                "score": score,
                "net_bps": net,
            }
        )
    return _one_global_position(candidates)


def _select_momentum_baseline(rows: pd.DataFrame) -> list[dict[str, Any]]:
    candidates = []
    for row in rows.to_dict("records"):
        trend = float(row["return_5m_bps"])
        move_budget = abs(trend) + 2.0 * float(row["realized_vol_5m_bps"])
        if abs(trend) < 5.0 or move_budget < GROSS_BARRIER_BPS:
            continue
        direction = "LONG" if trend > 0.0 else "SHORT"
        candidates.append(
            {
                "timestamp": int(row["timestamp"]),
                "exit_timestamp_ms": int(row["exit_timestamp_ms"]),
                "symbol": str(row["symbol"]),
                "direction": direction,
                "score": move_budget,
                "net_bps": float(
                    row["long_net_bps"] if direction == "LONG" else row["short_net_bps"]
                ),
            }
        )
    return _one_global_position(candidates)


def _one_global_position(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_timestamp: dict[int, list[Mapping[str, Any]]] = {}
    for row in candidates:
        by_timestamp.setdefault(int(row["timestamp"]), []).append(row)
    selected = []
    next_allowed = -1
    for timestamp in sorted(by_timestamp):
        if timestamp < next_allowed:
            continue
        best = max(
            by_timestamp[timestamp],
            key=lambda row: (float(row["score"]), str(row["symbol"])),
        )
        selected.append(dict(best))
        next_allowed = int(best["exit_timestamp_ms"])
    return selected


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [float(row["net_bps"]) for row in rows]
    mean = statistics.fmean(values) if values else None
    symbols = dict(sorted(Counter(str(row["symbol"]) for row in rows).items()))
    directions = dict(
        sorted(Counter(str(row["direction"]) for row in rows).items())
    )
    lcb = (
        mean - LCB_Z * statistics.stdev(values) / math.sqrt(len(values))
        if len(values) >= 2 and mean is not None
        else None
    )
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return {
        "fills": len(values) * 2,
        "closed_trades": len(values),
        "mean_gross_bps": (
            mean + MEASURED_MAKER_TAKER_COST_BPS if mean is not None else None
        ),
        "mean_cost_bps": MEASURED_MAKER_TAKER_COST_BPS,
        "mean_net_bps": mean,
        "lcb_95_net_bps": lcb,
        "net_sum_bps": sum(values),
        "win_rate": sum(value > 0.0 for value in values) / len(values) if values else 0.0,
        "max_drawdown_bps": drawdown,
        "symbols": symbols,
        "directions": directions,
        "top_symbol_trade_share": (
            max(symbols.values()) / len(values) if values else 0.0
        ),
        "minimum_direction_trade_share": (
            min(directions.get("LONG", 0), directions.get("SHORT", 0))
            / len(values)
            if values
            else 0.0
        ),
    }


def _split_passed(metrics: Mapping[str, Any]) -> bool:
    return (
        int(metrics["fills"]) >= 20
        and int(metrics["closed_trades"]) >= 10
        and metrics["mean_net_bps"] is not None
        and float(metrics["mean_net_bps"]) > 0.0
        and metrics["lcb_95_net_bps"] is not None
        and float(metrics["lcb_95_net_bps"]) > 0.0
        and float(metrics["max_drawdown_bps"]) <= MAX_DRAWDOWN_BPS
        and float(metrics["top_symbol_trade_share"]) <= MAX_SYMBOL_TRADE_SHARE
        and float(metrics["minimum_direction_trade_share"])
        >= MIN_DIRECTION_TRADE_SHARE
    )


def _top_coefficients(
    *,
    model: LogisticRegression,
    feature_names: Sequence[str],
) -> dict[str, list[dict[str, float | str]]]:
    result = {}
    for class_value, coefficients in zip(model.classes_, model.coef_):
        ranked = sorted(
            zip(feature_names, coefficients),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )[:8]
        result[str(int(class_value))] = [
            {"feature": name, "coefficient": float(value)}
            for name, value in ranked
        ]
    return result


def _date_start_ms(value: str) -> int:
    return int(pd.Timestamp(value, tz="UTC").timestamp() * 1000)


def _next_date(value: str) -> str:
    return str((pd.Timestamp(value) + pd.Timedelta(days=1)).date())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"
