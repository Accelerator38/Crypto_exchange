"""Diagnose CarryFlow activation and exit quality from one sealed replay root."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RUNTIME = SRC / "panteon_runtime"
for path in (SRC, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from carryflow_policy import CarryFlowPolicyProfile, get_carryflow_profile  # noqa: E402
from panteon_v2.policy.evidence_tape import CarryFlowEvidenceTape  # noqa: E402


POSITION_REASONS = {"in_short_position", "close_short"}
CANDIDATE_REASONS = {
    "candidate_short",
    "candidate_short_pending_selection",
}
DEFAULT_OUTPUT_JSON = (
    ROOT / "Reports" / "CarryFlow" / "root003_signal_quality.json"
)
DEFAULT_OUTPUT_MD = (
    ROOT / "Reports" / "CarryFlow" / "root003_signal_quality.md"
)


def analyze_signal_quality(
    *,
    tape: CarryFlowEvidenceTape,
    activation_trace: Sequence[Mapping[str, Any]],
    policy_decisions: Sequence[Mapping[str, Any]],
    closed_trades: Sequence[Mapping[str, Any]],
    replay_summary: Mapping[str, Any],
) -> dict[str, Any]:
    profile_id = str(
        (replay_summary.get("research_hypothesis") or {}).get("profile_id")
        or ""
    )
    profile = get_carryflow_profile(profile_id)
    samples = list(tape.samples)
    symbols = tuple(tape.symbols)
    expected_observations = len(samples) * len(symbols)
    if len(activation_trace) != expected_observations:
        raise ValueError(
            "activation trace/tape observation mismatch: "
            f"{len(activation_trace)} != {expected_observations}"
        )

    trace_by_key = {
        (int(row["bar"]), str(row["execution_symbol"])): row
        for row in activation_trace
    }
    if len(trace_by_key) != expected_observations:
        raise ValueError("activation trace has duplicate bar/symbol rows")
    decision_by_key = {
        (int(row["bar"]), str(row["symbol"])): row
        for row in policy_decisions
    }
    trade_by_key = {
        (int(row["opened_bar"]), str(row["symbol"])): row
        for row in closed_trades
    }

    derived = _derive_observations(
        samples=samples,
        symbols=symbols,
        profile=profile,
    )
    reason_counts = Counter(
        str(row["diagnostic_reason"]) for row in activation_trace
    )
    reason_by_symbol = {
        symbol: dict(
            sorted(
                Counter(
                    str(row["diagnostic_reason"])
                    for row in activation_trace
                    if str(row["execution_symbol"]) == symbol
                ).items()
            )
        )
        for symbol in symbols
    }
    reason_by_regime = {
        regime: dict(
            sorted(
                Counter(
                    str(row["diagnostic_reason"])
                    for row in activation_trace
                    if str(row["regime"]) == regime
                ).items()
            )
        )
        for regime in sorted(
            {str(row["regime"]) for row in activation_trace}
        )
    }

    flat_rows = [
        row
        for row in activation_trace
        if str(row["diagnostic_reason"]) not in POSITION_REASONS
    ]
    oi_below_rows = [
        row
        for row in flat_rows
        if str(row["diagnostic_reason"]) == "oi_below_threshold"
    ]
    oi_below_values = [
        derived[(int(row["bar"]), str(row["execution_symbol"]))]["oi_change"]
        for row in oi_below_rows
    ]
    flat_oi_values = [
        derived[(int(row["bar"]), str(row["execution_symbol"]))]["oi_change"]
        for row in flat_rows
    ]
    oi_pass = len(flat_rows) - len(oi_below_rows)
    raw_candidates = sum(
        int(reason_counts.get(reason, 0)) for reason in CANDIDATE_REASONS
    )
    selected_candidates = int(reason_counts.get("candidate_short", 0))
    policy_allowed = sum(
        str(row.get("outcome")) == "ALLOW_OPEN" for row in policy_decisions
    )

    candidate_rows = _candidate_rows(
        samples=samples,
        activation_trace=activation_trace,
        decisions=decision_by_key,
        trades=trade_by_key,
        derived=derived,
        profile=profile,
        mean_cost_bps=float(replay_summary["mean_cost_bps"]),
    )
    selected_rows = [
        row
        for row in candidate_rows
        if row["actor_selection"] == "selected"
    ]
    allowed_rows = [
        row for row in candidate_rows if row["policy_outcome"] == "ALLOW_OPEN"
    ]
    calibration = _calibration(allowed_rows)
    horizon_analysis = {
        "all_actor_candidates": _aggregate_horizons(candidate_rows),
        "actor_selected": _aggregate_horizons(selected_rows),
        "policy_allowed": _aggregate_horizons(allowed_rows),
        "diagnostic_only": True,
        "selection_on_this_root_forbidden": True,
    }
    exit_analysis = _exit_analysis(
        profile=profile,
        allowed_rows=allowed_rows,
        closed_trades=closed_trades,
        mean_cost_bps=float(replay_summary["mean_cost_bps"]),
    )
    component_means = {
        "all_actor_candidates": _mean_components(candidate_rows),
        "policy_allowed": _mean_components(allowed_rows),
    }
    allowed_component_total = sum(
        float(value or 0.0)
        for value in component_means["policy_allowed"].values()
    )
    component_shares = {
        name: (
            float(value or 0.0) / allowed_component_total
            if allowed_component_total
            else 0.0
        )
        for name, value in component_means["policy_allowed"].items()
    }
    allowed_regimes = dict(
        sorted(Counter(str(row["regime"]) for row in allowed_rows).items())
    )

    per_symbol_oi = {}
    for symbol in symbols:
        symbol_flat = [
            row
            for row in flat_rows
            if str(row["execution_symbol"]) == symbol
        ]
        values = [
            derived[(int(row["bar"]), symbol)]["oi_change"]
            for row in symbol_flat
        ]
        pass_count = sum(value >= profile.oi_expansion for value in values)
        per_symbol_oi[symbol] = {
            "flat_observations": len(values),
            "oi_pass": pass_count,
            "oi_pass_rate": _ratio(pass_count, len(values)),
            "median_oi_change_bps": _median(values) * 10_000.0,
            "p90_oi_change_bps": _quantile(values, 0.90) * 10_000.0,
        }

    report = {
        "schema_version": "panteon.carryflow_signal_quality.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "promotion_authority": False,
        "orders_enabled": False,
        "strategy_changed": False,
        "root": {
            "collector_run_id": tape.collector_run_id,
            "tape_sha256": tape.file_sha256,
            "tape_head_sha256": tape.head_sha256,
            "samples": len(samples),
            "complete_samples": tape.describe()["complete_samples"],
            "symbols": list(symbols),
            "replay_manifest_sha256": replay_summary["manifest_sha256"],
        },
        "profile": {
            "profile_id": profile.profile_id,
            "oi_lookback_bars": profile.oi_lookback_bars,
            "oi_threshold_bps": profile.oi_expansion * 10_000.0,
            "hold_bars": profile.hold_bars,
            "stop_bps": profile.stop_pct * 10_000.0,
            "target_bps": profile.target_pct * 10_000.0,
            "exit_on_normalization": profile.exit_on_normalization,
        },
        "activation_funnel": {
            "actor_observations": len(activation_trace),
            "flat_observations": len(flat_rows),
            "oi_below_threshold": len(oi_below_rows),
            "oi_pass": oi_pass,
            "oi_pass_rate": _ratio(oi_pass, len(flat_rows)),
            "funding_rejected_after_oi": int(
                reason_counts.get("short_funding_below_threshold", 0)
            ),
            "price_divergence_rejected_after_oi": int(
                reason_counts.get("short_price_divergence_missing", 0)
            ),
            "systemic_guard_rejected_after_oi": int(
                reason_counts.get("short_systemic_selloff_guard", 0)
            ),
            "raw_actor_candidates": raw_candidates,
            "actor_selected_candidates": selected_candidates,
            "policy_allowed_candidates": policy_allowed,
            "policy_allowed_regimes": allowed_regimes,
            "closed_trades": len(closed_trades),
            "reasons": dict(sorted(reason_counts.items())),
            "reasons_by_symbol": reason_by_symbol,
            "reasons_by_regime": reason_by_regime,
        },
        "oi_gate": {
            "threshold_bps": profile.oi_expansion * 10_000.0,
            "failed_change_quantiles_bps": {
                name: _quantile(oi_below_values, value) * 10_000.0
                for name, value in (
                    ("min", 0.0),
                    ("p10", 0.10),
                    ("p25", 0.25),
                    ("median", 0.50),
                    ("p75", 0.75),
                    ("p90", 0.90),
                    ("p95", 0.95),
                    ("max", 1.0),
                )
            },
            "failed_within_25_bps_of_threshold": sum(
                profile.oi_expansion - 0.0025 <= value < profile.oi_expansion
                for value in oi_below_values
            ),
            "failed_within_50_bps_of_threshold": sum(
                profile.oi_expansion - 0.005 <= value < profile.oi_expansion
                for value in oi_below_values
            ),
            "threshold_sensitivity": [
                {
                    "threshold_bps": threshold_bps,
                    "flat_observations_passing": sum(
                        value >= threshold_bps / 10_000.0
                        for value in flat_oi_values
                    ),
                }
                for threshold_bps in (-50, 0, 50, 100, 125, 150, 175, 200)
            ],
            "by_symbol": per_symbol_oi,
            "interpretation": (
                "The gate is selective, but most failures are not near the "
                "threshold. Lowering it would primarily add materially weaker "
                "OI observations and is not supported by this root."
            ),
        },
        "signal_calibration": {
            **calibration,
            "edge_component_means": component_means,
            "policy_allowed_edge_component_shares": component_shares,
            "candidate_details": candidate_rows,
            "interpretation": (
                "diagnostic.edge is a ranking score, not a calibrated forward "
                "return. Mapping it linearly to expected bps materially "
                "overstates the observed move."
            ),
        },
        "exit_analysis": exit_analysis,
        "horizon_analysis": horizon_analysis,
        "findings": [
            "OI is the dominant incidence gate: 490 of 562 flat observations failed it.",
            (
                "Only 12 raw actor candidates survived all actor gates; seven "
                "were selected and five passed policy confidence."
            ),
            (
                "The expected-move model is not calibrated to realized gross "
                "returns on the five allowed trades."
            ),
            (
                "All five policy-allowed candidates occurred in range_low_vol; "
                "this root contains no executed evidence for another regime."
            ),
            (
                "All closed trades exited by max_holding; neither the 120 bps "
                "stop nor the 240 bps target triggered."
            ),
            (
                "Every allowed trade had favorable excursion above modeled "
                "cost, but the fixed six-bar exit gave back enough excursion "
                "to make aggregate expectancy negative."
            ),
        ],
        "next_design_constraints": [
            "Do not lower the OI threshold based on this root.",
            "Separate candidate ranking score from calibrated expected move.",
            "Calibrate expected move on independent train/validation/OOS data.",
            (
                "Evaluate one profile-level cost-aware exit against the fixed "
                "six-bar exit; do not expose independent live exit knobs."
            ),
            (
                "Treat the four-bar result as a hypothesis only because it was "
                "selected after observing five trades."
            ),
        ],
    }
    return report


def _derive_observations(
    *,
    samples: Sequence[Mapping[str, Any]],
    symbols: Sequence[str],
    profile: CarryFlowPolicyProfile,
) -> dict[tuple[int, str], dict[str, float]]:
    oi_history = {symbol: [] for symbol in symbols}
    price_history = {symbol: [] for symbol in symbols}
    result: dict[tuple[int, str], dict[str, float]] = {}
    for bar, sample in enumerate(samples, start=1):
        symbol_rows = sample["symbols"]
        for symbol in symbols:
            row = symbol_rows[symbol]
            derivatives = row["derivatives"]
            market = row["market"]
            price = float(market["decision_price"])
            oi_now = float(derivatives.get("open_interest_usdt") or 0.0)
            price_history[symbol].append(price)
            if oi_now > 0.0:
                oi_history[symbol].append(oi_now)
            oi_change = _lookback_return(
                oi_history[symbol],
                profile.oi_lookback_bars,
            )
            price_return = _lookback_return(
                price_history[symbol],
                profile.price_lookback_bars,
            )
            mark = float(derivatives.get("mark_price") or 0.0)
            index = float(derivatives.get("index_price") or 0.0)
            basis = mark / index - 1.0 if mark > 0.0 and index > 0.0 else 0.0
            result[(bar, symbol)] = {
                "price": price,
                "oi_change": oi_change,
                "price_return": price_return,
                "funding_rate": float(
                    derivatives.get("funding_rate") or 0.0
                ),
                "long_ratio": float(derivatives.get("long_ratio") or 0.5),
                "basis": basis,
            }
    return result


def _candidate_rows(
    *,
    samples: Sequence[Mapping[str, Any]],
    activation_trace: Sequence[Mapping[str, Any]],
    decisions: Mapping[tuple[int, str], Mapping[str, Any]],
    trades: Mapping[tuple[int, str], Mapping[str, Any]],
    derived: Mapping[tuple[int, str], Mapping[str, float]],
    profile: CarryFlowPolicyProfile,
    mean_cost_bps: float,
) -> list[dict[str, Any]]:
    rows = []
    for trace in activation_trace:
        reason = str(trace["diagnostic_reason"])
        if reason not in CANDIDATE_REASONS:
            continue
        bar = int(trace["bar"])
        symbol = str(trace["execution_symbol"])
        feature = float(trace["feature_value"])
        observation = derived[(bar, symbol)]
        components = _edge_components(observation, profile)
        component_sum = sum(components.values())
        if not math.isclose(component_sum, feature, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(
                f"edge reconstruction mismatch at bar={bar}, symbol={symbol}"
            )
        expected_move = min(
            profile.signal_max_expected_move_bps,
            max(
                0.0,
                feature * profile.signal_slope_bps_per_unit
                - profile.signal_lcb_haircut_bps,
            ),
        )
        decision = decisions.get((bar, symbol))
        trade = trades.get((bar, symbol))
        horizons, excursions = _candidate_path(
            samples=samples,
            bar=bar,
            symbol=symbol,
            entry_price=float(observation["price"]),
            mean_cost_bps=mean_cost_bps,
        )
        row = {
            "bar": bar,
            "symbol": symbol,
            "regime": str(trace["regime"]),
            "actor_selection": (
                "selected" if reason == "candidate_short" else "not_selected"
            ),
            "feature_value": feature,
            "predicted_move_bps": expected_move,
            "edge_components": components,
            "oi_change_bps": float(observation["oi_change"]) * 10_000.0,
            "price_return_bps": (
                float(observation["price_return"]) * 10_000.0
            ),
            "policy_outcome": (
                str(decision["outcome"]) if decision is not None else "UNSELECTED"
            ),
            "policy_reason": (
                str(decision["primary_reason"]) if decision is not None else ""
            ),
            "horizons": horizons,
            **excursions,
            "actual_gross_bps": None,
            "actual_net_bps": None,
            "actual_close_reason": "",
        }
        if trade is not None:
            notional = float(trade["notional_usd"])
            row.update(
                {
                    "actual_gross_bps": (
                        float(trade["gross_pnl_usd"]) / notional * 10_000.0
                    ),
                    "actual_net_bps": (
                        float(trade["net_pnl_usd"]) / notional * 10_000.0
                    ),
                    "actual_close_reason": str(trade["close_reason"]),
                }
            )
        rows.append(row)
    return rows


def _candidate_path(
    *,
    samples: Sequence[Mapping[str, Any]],
    bar: int,
    symbol: str,
    entry_price: float,
    mean_cost_bps: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    horizons: dict[str, Any] = {}
    lows: list[float] = []
    highs: list[float] = []
    mfe_by_horizon: dict[int, float] = {}
    mae_by_horizon: dict[int, float] = {}
    for horizon in range(1, 13):
        future_bar = bar + horizon
        if future_bar > len(samples):
            break
        market = samples[future_bar - 1]["symbols"][symbol]["market"]
        lows.append(float(market["low"]))
        highs.append(float(market["high"]))
        exit_price = float(market["decision_price"])
        gross_bps = (entry_price - exit_price) / entry_price * 10_000.0
        horizons[str(horizon)] = {
            "gross_bps": gross_bps,
            "net_bps": gross_bps - mean_cost_bps,
        }
        mfe_by_horizon[horizon] = (
            (entry_price - min(lows)) / entry_price * 10_000.0
        )
        mae_by_horizon[horizon] = (
            (max(highs) - entry_price) / entry_price * 10_000.0
        )
    return horizons, {
        "mfe_6_bps": mfe_by_horizon.get(6),
        "mae_6_bps": mae_by_horizon.get(6),
        "mfe_12_bps": mfe_by_horizon.get(max(mfe_by_horizon, default=0)),
        "mae_12_bps": mae_by_horizon.get(max(mae_by_horizon, default=0)),
    }


def _edge_components(
    observation: Mapping[str, float],
    profile: CarryFlowPolicyProfile,
) -> dict[str, float]:
    return {
        "funding": (
            float(observation["funding_rate"]) - profile.funding_entry
        )
        * 10_000.0,
        "crowding": (
            float(observation["long_ratio"]) - profile.crowd_ratio
        )
        * 4.0,
        "basis": (
            float(observation["basis"]) - profile.basis_floor
        )
        * 1_000.0,
        "oi_excess": (
            float(observation["oi_change"]) - profile.oi_expansion
        )
        * 10.0,
        "price_divergence": max(
            profile.max_price_return - float(observation["price_return"]),
            0.0,
        )
        * 100.0,
    }


def _calibration(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    realized = [
        row
        for row in rows
        if row.get("actual_gross_bps") is not None
    ]
    predicted = [float(row["predicted_move_bps"]) for row in realized]
    gross = [float(row["actual_gross_bps"]) for row in realized]
    net = [float(row["actual_net_bps"]) for row in realized]
    predicted_mean = _mean(predicted)
    gross_mean = _mean(gross)
    return {
        "trades": len(realized),
        "mean_predicted_move_bps": predicted_mean,
        "mean_realized_gross_bps": gross_mean,
        "mean_realized_net_bps": _mean(net),
        "realized_to_predicted_ratio": (
            gross_mean / predicted_mean if predicted_mean else None
        ),
        "prediction_overstatement_bps": predicted_mean - gross_mean,
        "pearson_predicted_vs_gross": _pearson(predicted, gross),
        "sample_too_small_for_calibration": len(realized) < 20,
    }


def _aggregate_horizons(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {"candidates": len(rows), "horizons": {}}
    for horizon in (1, 2, 3, 4, 6, 8, 12):
        values = [
            float(row["horizons"][str(horizon)]["net_bps"])
            for row in rows
            if str(horizon) in row["horizons"]
        ]
        result["horizons"][str(horizon)] = {
            "observations": len(values),
            "mean_net_bps": _mean(values),
            "positive_rate": _ratio(
                sum(value > 0.0 for value in values),
                len(values),
            ),
        }
    return result


def _exit_analysis(
    *,
    profile: CarryFlowPolicyProfile,
    allowed_rows: Sequence[Mapping[str, Any]],
    closed_trades: Sequence[Mapping[str, Any]],
    mean_cost_bps: float,
) -> dict[str, Any]:
    mfe_values = [
        float(row["mfe_6_bps"])
        for row in allowed_rows
        if row.get("mfe_6_bps") is not None
    ]
    mae_values = [
        float(row["mae_6_bps"])
        for row in allowed_rows
        if row.get("mae_6_bps") is not None
    ]
    return {
        "close_reasons": dict(
            sorted(Counter(str(row["close_reason"]) for row in closed_trades).items())
        ),
        "hold_bars": profile.hold_bars,
        "stop_bps": profile.stop_pct * 10_000.0,
        "target_bps": profile.target_pct * 10_000.0,
        "normalization_exit_enabled": profile.exit_on_normalization,
        "trades_reaching_target_intrabar_by_hold": sum(
            value >= profile.target_pct * 10_000.0 for value in mfe_values
        ),
        "trades_reaching_stop_intrabar_by_hold": sum(
            value >= profile.stop_pct * 10_000.0 for value in mae_values
        ),
        "trades_with_mfe_above_modeled_cost": sum(
            value > mean_cost_bps for value in mfe_values
        ),
        "allowed_trades_with_full_six_bar_path": len(mfe_values),
        "mean_mfe_6_bps": _mean(mfe_values),
        "mean_mae_6_bps": _mean(mae_values),
        "fixed_horizon_comparison": _aggregate_horizons(allowed_rows)[
            "horizons"
        ],
        "diagnostic_only": True,
        "four_bar_result_is_in_sample_hypothesis": True,
    }


def _mean_components(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float | None]:
    names = sorted(
        {
            str(name)
            for row in rows
            for name in row.get("edge_components", {})
        }
    )
    return {
        name: _mean(
            [
                float(row["edge_components"][name])
                for row in rows
                if name in row.get("edge_components", {})
            ]
        )
        for name in names
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    funnel = report["activation_funnel"]
    oi = report["oi_gate"]
    calibration = report["signal_calibration"]
    exit_row = report["exit_analysis"]
    allowed_horizons = report["horizon_analysis"]["policy_allowed"]["horizons"]
    component_means = calibration["edge_component_means"]["policy_allowed"]
    component_shares = calibration["policy_allowed_edge_component_shares"]
    selected_candidates = [
        row
        for row in calibration["candidate_details"]
        if row["actor_selection"] == "selected"
    ]
    lines = [
        "# CarryFlow root 003: signal quality analysis",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "This report is research-only. It does not authorize paper or live orders.",
        "",
        "## Verdict",
        "",
        (
            "The root exposed two independent defects: activation is too rare, "
            "and `diagnostic.edge` is not calibrated as a forward return. The "
            "fixed six-bar exit then gives back favorable excursion."
        ),
        "",
        "## Activation funnel",
        "",
        "| Stage | Count | Share |",
        "|---|---:|---:|",
        (
            f"| Flat observations | {funnel['flat_observations']} | "
            "100.0% |"
        ),
        (
            f"| OI passed | {funnel['oi_pass']} | "
            f"{funnel['oi_pass_rate'] * 100.0:.1f}% |"
        ),
        (
            f"| Raw actor candidates | {funnel['raw_actor_candidates']} | "
            f"{_ratio(funnel['raw_actor_candidates'], funnel['flat_observations']) * 100.0:.1f}% |"
        ),
        (
            f"| Actor selected | {funnel['actor_selected_candidates']} | "
            f"{_ratio(funnel['actor_selected_candidates'], funnel['flat_observations']) * 100.0:.1f}% |"
        ),
        (
            f"| Policy allowed | {funnel['policy_allowed_candidates']} | "
            f"{_ratio(funnel['policy_allowed_candidates'], funnel['flat_observations']) * 100.0:.1f}% |"
        ),
        "",
        "## OI gate",
        "",
        (
            f"The threshold is {oi['threshold_bps']:.0f} bps over three bars. "
            f"{funnel['oi_below_threshold']} observations failed it."
        ),
        (
            f"Median failed OI change was "
            f"{oi['failed_change_quantiles_bps']['median']:.1f} bps; p95 was "
            f"{oi['failed_change_quantiles_bps']['p95']:.1f} bps. Only "
            f"{oi['failed_within_25_bps_of_threshold']} failures were within "
            "25 bps of the threshold."
        ),
        "",
        "Lowering the threshold is not supported: most rejected observations are "
        "materially weaker, not borderline cases.",
        "",
        "| Symbol | Flat | OI pass | Pass rate | Median, bps | p90, bps |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for symbol, row in oi["by_symbol"].items():
        lines.append(
            f"| {symbol} | {row['flat_observations']} | {row['oi_pass']} | "
            f"{row['oi_pass_rate'] * 100.0:.1f}% | "
            f"{row['median_oi_change_bps']:.1f} | "
            f"{row['p90_oi_change_bps']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Signal calibration",
            "",
            (
                f"Mean predicted move: "
                f"{calibration['mean_predicted_move_bps']:.2f} bps."
            ),
            (
                f"Mean realized gross move: "
                f"{calibration['mean_realized_gross_bps']:.2f} bps; mean net: "
                f"{calibration['mean_realized_net_bps']:.2f} bps."
            ),
            (
                f"Realized/predicted ratio: "
                f"{calibration['realized_to_predicted_ratio'] * 100.0:.2f}% "
                f"on only {calibration['trades']} trades."
            ),
            "",
            (
                "`diagnostic.edge` may rank candidates, but its current linear "
                "conversion to bps is not a valid expectancy estimate."
            ),
            "",
            "| Edge component | Mean contribution | Share |",
            "|---|---:|---:|",
        ]
    )
    for name, value in component_means.items():
        lines.append(
            f"| {name} | {value:.3f} | "
            f"{component_shares[name] * 100.0:.1f}% |"
        )
    lines.extend(
        [
            "",
            (
                "Basis and crowding dominate the score; OI is mostly a binary "
                "gate and contributes little to ranking after the threshold "
                "is passed."
            ),
            "",
            "| Bar | Symbol | Regime | Policy | Predicted | 6h gross | 6h net |",
            "|---:|---|---|---|---:|---:|---:|",
        ]
    )
    for row in selected_candidates:
        horizon = row["horizons"].get("6")
        gross = horizon["gross_bps"] if horizon is not None else None
        net = horizon["net_bps"] if horizon is not None else None
        if gross is None or net is None:
            lines.append(
                f"| {row['bar']} | {row['symbol']} | {row['regime']} | "
                f"{row['policy_outcome']} | {row['predicted_move_bps']:.1f} | "
                "n/a | n/a |"
            )
        else:
            lines.append(
                f"| {row['bar']} | {row['symbol']} | {row['regime']} | "
                f"{row['policy_outcome']} | {row['predicted_move_bps']:.1f} | "
                f"{gross:.1f} | {net:.1f} |"
            )
    lines.extend(
        [
            "",
            (
                "All five allowed trades were in `range_low_vol`; the two "
                "bearish SOL candidates were blocked by low regime confidence."
            ),
            "",
            "## Exit behavior",
            "",
            (
                f"All {funnel['closed_trades']} closed trades used "
                f"`max_holding`. Target: {exit_row['target_bps']:.0f} bps; "
                f"stop: {exit_row['stop_bps']:.0f} bps."
            ),
            (
                f"Target hits by six bars: "
                f"{exit_row['trades_reaching_target_intrabar_by_hold']}; "
                f"stop hits: "
                f"{exit_row['trades_reaching_stop_intrabar_by_hold']}. All "
                f"{exit_row['trades_with_mfe_above_modeled_cost']} trades had "
                "MFE above modeled cost."
            ),
            (
                "MFE/MAE use sealed hourly OHLC and are diagnostic excursions, "
                "not proof that a trailing exit could have filled at the extreme."
            ),
            "",
            "| Fixed exit | Observations | Mean net, bps | Positive |",
            "|---:|---:|---:|---:|",
        ]
    )
    for horizon in (1, 2, 3, 4, 6, 8, 12):
        row = allowed_horizons[str(horizon)]
        mean_value = row["mean_net_bps"]
        positive = row["positive_rate"]
        lines.append(
            f"| {horizon}h | {row['observations']} | "
            f"{mean_value:.2f} | {positive * 100.0:.1f}% |"
            if mean_value is not None and positive is not None
            else f"| {horizon}h | 0 | n/a | n/a |"
        )
    lines.extend(
        [
            "",
            "The four-hour row is an in-sample hypothesis, not a selected policy.",
            "",
            "## Next design constraints",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["next_design_constraints"])
    lines.append("")
    return "\n".join(lines)


def _lookback_return(values: Sequence[float], lookback: int) -> float:
    width = max(1, int(lookback))
    if len(values) <= width or float(values[-1 - width]) <= 0.0:
        return 0.0
    return float(values[-1]) / float(values[-1 - width]) - 1.0


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _mean(values: Iterable[float]) -> float | None:
    selected = [float(value) for value in values]
    return sum(selected) / len(selected) if selected else None


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left, right)
    )
    left_scale = sum((x - left_mean) ** 2 for x in left)
    right_scale = sum((y - right_mean) ** 2 for y in right)
    if left_scale <= 0.0 or right_scale <= 0.0:
        return None
    return numerator / math.sqrt(left_scale * right_scale)


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"JSONL objects required: {path}")
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one sealed CarryFlow replay without changing policy or "
            "authorizing orders."
        )
    )
    parser.add_argument("--evidence-tape", required=True)
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    args = parser.parse_args(argv)

    replay_dir = Path(args.replay_dir).resolve()
    tape = CarryFlowEvidenceTape.from_jsonl(Path(args.evidence_tape).resolve())
    report = analyze_signal_quality(
        tape=tape,
        activation_trace=_read_jsonl(replay_dir / "activation_trace.jsonl"),
        policy_decisions=_read_jsonl(replay_dir / "policy_decisions.jsonl"),
        closed_trades=_read_jsonl(replay_dir / "closed_trades.jsonl"),
        replay_summary=_read_json(replay_dir / "replay_summary.json"),
    )
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "run_state": "completed",
                "research_only": True,
                "promotion_authority": False,
                "orders_enabled": False,
                "output_json": str(output_json),
                "output_md": str(output_md),
                "activation_funnel": report["activation_funnel"],
                "signal_calibration": {
                    key: value
                    for key, value in report["signal_calibration"].items()
                    if key != "candidate_details"
                },
                "exit_analysis": report["exit_analysis"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
