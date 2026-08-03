from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from simple_research.dataset import load_feature_tape  # noqa: E402
from simple_research.regimes import (  # noqa: E402
    REGIME_CONTRACT,
    REGIME_ORDER,
    attribute_trade_regimes,
    build_market_regime_lookup,
)
from simple_research.simulator import CostModel, simulate_targets  # noqa: E402
from simple_research.strategies import build_signal  # noqa: E402


REGISTRY_PATH = ROOT / "configs" / "simple_research" / "preregistered_strategies_v1.json"
FREQTRADE_CANDIDATE = ROOT / "freqtrade_pilot" / "candidates" / "long_horizon_trend_v1.json"
TAPE_PATH = ROOT / "Retrodate" / "simple_research_reset_v1" / "bitget_full8_1h.parquet"
TAPE_MANIFEST_PATH = TAPE_PATH.with_suffix(".manifest.json")
REPORT_PATH = ROOT / "Reports" / "ModelRegimeAudit" / "current_models_by_regime_v1.json"
MARKDOWN_PATH = ROOT / "Reports" / "ModelRegimeAudit" / "current_models_by_regime_v1.md"
POST_DEVELOPMENT = ("validation", "oos", "sanity")
LONG_HORIZON_REPORT = ROOT / "Reports" / "FreqtradePilot" / "long_horizon_trend_v1.json"
STRATEGY_LAB_REPORT = (
    ROOT
    / "Reports"
    / "StrategyLab"
    / "p2_historical_oos_20260727"
    / "strategy_lab_summary.json"
)
CARRYFLOW_RANGE_REPORT = (
    ROOT / "Reports" / "CarryFlow" / "range_transition_breakout_20260727.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp(value: str) -> int:
    return int(pd.Timestamp(value).timestamp() * 1000)


def _cost_model(row: dict[str, Any]) -> CostModel:
    return CostModel(
        fee_bps_per_fill=float(row["fee_bps_per_fill"]),
        slippage_bps_per_fill=float(row["slippage_bps_per_fill"]),
    )


def _metrics(ledger: pd.DataFrame, *, trial_count: int) -> dict[str, Any]:
    values = [float(value) for value in ledger["net_bps"]] if not ledger.empty else []
    count = len(values)
    mean = statistics.mean(values) if values else None
    median = statistics.median(values) if values else None
    std = statistics.stdev(values) if count > 1 else None
    standard_error = std / math.sqrt(count) if std is not None else None
    lcb = mean - 1.96 * standard_error if standard_error is not None else None
    adjusted_z = NormalDist().inv_cdf(1.0 - 0.05 / max(1, trial_count))
    adjusted_lcb = (
        mean - adjusted_z * standard_error if standard_error is not None else None
    )
    wins = sum(value > 0 for value in values)
    directions: dict[str, Any] = {}
    for value, label in ((1, "LONG"), (-1, "SHORT")):
        subset = ledger.loc[ledger["direction"] == value]
        directions[label] = _metrics_without_direction(subset, trial_count=trial_count)
    return {
        "closed_trades": count,
        "fills": int(ledger["fills"].sum()) if count else 0,
        "wins": wins,
        "win_rate": wins / count if count else None,
        "mean_net_bps": mean,
        "median_net_bps": median,
        "lcb_95_net_bps": lcb,
        "multiple_testing_lcb_95_net_bps": adjusted_lcb,
        "directions": directions,
    }


def _metrics_without_direction(
    ledger: pd.DataFrame,
    *,
    trial_count: int,
) -> dict[str, Any]:
    values = [float(value) for value in ledger["net_bps"]] if not ledger.empty else []
    count = len(values)
    mean = statistics.mean(values) if values else None
    std = statistics.stdev(values) if count > 1 else None
    standard_error = std / math.sqrt(count) if std is not None else None
    adjusted_z = NormalDist().inv_cdf(1.0 - 0.05 / max(1, trial_count))
    return {
        "closed_trades": count,
        "mean_net_bps": mean,
        "lcb_95_net_bps": mean - 1.96 * standard_error
        if standard_error is not None
        else None,
        "multiple_testing_lcb_95_net_bps": mean - adjusted_z * standard_error
        if standard_error is not None
        else None,
    }


def _regime_metrics(
    ledger: pd.DataFrame,
    *,
    trial_count: int,
) -> dict[str, Any]:
    return {
        regime: _metrics(
            ledger.loc[ledger["market_regime"] == regime],
            trial_count=trial_count,
        )
        for regime in REGIME_ORDER
    }


def _specialization_status(
    aggregate: dict[str, Any],
    windows: dict[str, dict[str, Any]],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if aggregate["closed_trades"] < 50:
        reasons.append("holdout_trades_below_50")
    if aggregate["mean_net_bps"] is None or aggregate["mean_net_bps"] <= 0:
        reasons.append("nonpositive_stress_mean")
    if aggregate["lcb_95_net_bps"] is None or aggregate["lcb_95_net_bps"] <= 0:
        reasons.append("nonpositive_stress_lcb")
    if (
        aggregate["multiple_testing_lcb_95_net_bps"] is None
        or aggregate["multiple_testing_lcb_95_net_bps"] <= 0
    ):
        reasons.append("nonpositive_multiple_testing_lcb")
    eligible_windows = [
        value for value in windows.values() if value["closed_trades"] >= 10
    ]
    positive_windows = [
        value
        for value in eligible_windows
        if value["mean_net_bps"] is not None and value["mean_net_bps"] > 0
    ]
    if len(eligible_windows) < 2:
        reasons.append("fewer_than_two_windows_with_10_trades")
    if len(positive_windows) != len(eligible_windows):
        reasons.append("window_expectancy_collapse")

    if not reasons:
        return "CONFIRMED", []
    if (
        aggregate["closed_trades"] >= 30
        and aggregate["lcb_95_net_bps"] is not None
        and aggregate["lcb_95_net_bps"] > 0
    ):
        return "PROMISING_UNCONFIRMED", reasons
    if (
        aggregate["closed_trades"] >= 10
        and aggregate["mean_net_bps"] is not None
        and aggregate["mean_net_bps"] > 0
    ):
        return "POSITIVE_MEAN_ONLY", reasons
    return "FAILED_OR_INSUFFICIENT", reasons


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _retained_nonuniform_evidence() -> list[dict[str, Any]]:
    """Extract regime observations that cannot enter the uniform ranking."""
    rows: list[dict[str, Any]] = []

    long_horizon = json.loads(LONG_HORIZON_REPORT.read_text(encoding="utf-8"))
    sanity_bullish = long_horizon["evaluations"]["sanity"]["stress"][
        "market_regimes"
    ]["bullish"]
    rows.append(
        {
            "artifact_id": "LongHorizonTrendStrategyV1",
            "slice": "bullish / sanity",
            "cost_contract": "Freqtrade stress, 16 bps round trip",
            "closed_trades": sanity_bullish["closed_trades"],
            "mean_net_bps": sanity_bullish["mean_net_bps"],
            "lcb_95_net_bps": sanity_bullish["lcb_95_net_bps"],
            "verdict": "TERMINAL_REJECTED_RETROSPECTIVE",
            "note": "Bullish mean was positive in earlier windows but collapsed in sanity.",
            "source": str(LONG_HORIZON_REPORT.relative_to(ROOT)).replace("\\", "/"),
        }
    )

    strategy_lab = json.loads(STRATEGY_LAB_REPORT.read_text(encoding="utf-8"))
    candidates = {
        candidate["candidate_id"]: candidate
        for candidate in strategy_lab["candidates"]
    }
    pullback = candidates["regime_pullback_hourly_v1"]["per_regime"]["bearish"]
    rows.append(
        {
            "artifact_id": "regime_pullback_hourly_v1",
            "slice": "bearish",
            "cost_contract": "StrategyLab base, 12 bps round trip",
            "closed_trades": pullback["trades"],
            "mean_net_bps": pullback["mean_net_bps"],
            "lcb_95_net_bps": pullback["lcb_95_net_bps"],
            "verdict": "TERMINAL_REJECTED",
            "note": "Best of its two directional regimes was still negative.",
            "source": str(STRATEGY_LAB_REPORT.relative_to(ROOT)).replace("\\", "/"),
        }
    )
    compression = candidates["ohlcv_compression_transition_hourly_v1"][
        "per_regime"
    ]["compression_transition"]
    rows.append(
        {
            "artifact_id": "ohlcv_compression_transition_hourly_v1",
            "slice": "compression_transition",
            "cost_contract": "StrategyLab base, 12 bps round trip",
            "closed_trades": compression["trades"],
            "mean_net_bps": compression["mean_net_bps"],
            "lcb_95_net_bps": compression["lcb_95_net_bps"],
            "verdict": "TERMINAL_REJECTED",
            "note": "Short direction had positive mean but negative LCB and stress failed.",
            "source": str(STRATEGY_LAB_REPORT.relative_to(ROOT)).replace("\\", "/"),
        }
    )

    carryflow = json.loads(CARRYFLOW_RANGE_REPORT.read_text(encoding="utf-8"))
    carryflow_aggregate = carryflow["evaluation"]["aggregate"]
    rows.append(
        {
            "artifact_id": "CarryFlow range-transition breakout",
            "slice": "range_low_vol transition",
            "cost_contract": "observed cost, approximately 12 bps round trip",
            "closed_trades": carryflow_aggregate["trades"],
            "mean_net_bps": carryflow_aggregate["mean_net_bps"],
            "lcb_95_net_bps": carryflow_aggregate["lcb_95_net_bps"],
            "verdict": "ACTIVATION_OR_EXPECTANCY_FAILURE",
            "note": "Only eight trades; one root inactive and one root collapsed.",
            "source": str(CARRYFLOW_RANGE_REPORT.relative_to(ROOT)).replace("\\", "/"),
        }
    )
    return rows


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Current model regime audit",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        (
            "All rankings use the common stress cost of 16 bps round trip. "
            "Regimes are attributed from the last closed BTC 1h candle before "
            "entry and never alter signals."
        ),
        "",
        "## Scope and regime contract",
        "",
        (
            f"- Uniformly ranked models: "
            f"`{report['model_inventory']['uniformly_ranked']}` of "
            f"`{report['model_inventory']['evaluated']}` "
            "(flat and long-only are reference baselines)."
        ),
        "- `bullish`: BTC close > EMA72 > EMA336 and seven-day momentum > 0.",
        "- `bearish`: BTC close < EMA72 < EMA336 and seven-day momentum < 0.",
        (
            "- `volatile_mixed`: no directional trend and ATR14/close is at or "
            "above the trailing 180-day 75th percentile."
        ),
        (
            "- `range_low_vol`: no directional trend and ATR14/close is at or "
            "below the trailing 180-day 25th percentile."
        ),
        "- `neutral`: the remaining non-trending observations.",
        "- Attribution completeness: every closed trade has exactly one known regime.",
        "",
        "| Window | Timerange | Role |",
        "|---|---|---|",
        (
            f"| development | {report['uniform_splits']['development']} | "
            "hypothesis construction only |"
        ),
        f"| validation | {report['uniform_splits']['validation']} | first held-out check |",
        f"| oos | {report['uniform_splits']['oos']} | independent out-of-sample check |",
        (
            f"| sanity | {report['uniform_splits']['sanity']} | "
            "most recent temporal stability check |"
        ),
        "",
        "## Verdict",
        "",
        f"- Confirmed specializations: `{len(report['confirmed_specializations'])}`.",
        f"- Promising but unconfirmed: `{len(report['promising_unconfirmed'])}`.",
        "- Positive mean without robust LCB is not treated as edge.",
        "- Orders enabled: `false`; promotion authority: `false`.",
        "",
        "## Best available model per regime",
        "",
        (
            "| Regime | Model | Holdout trades | Stress mean bps | LCB bps | "
            "Adjusted LCB bps | Positive windows | Status |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for regime in REGIME_ORDER[:-1]:
        row = report["best_by_regime"].get(regime)
        if row is None:
            lines.append(
                f"| {regime} | none | 0 | n/a | n/a | n/a | 0/0 | "
                "FAILED_OR_INSUFFICIENT |"
            )
            continue
        lines.append(
            f"| {regime} | {row['model_id']} | {row['closed_trades']} | "
            f"{_fmt(row['mean_net_bps'])} | {_fmt(row['lcb_95_net_bps'])} | "
            f"{_fmt(row['multiple_testing_lcb_95_net_bps'])} | "
            f"{row['positive_windows']}/{row['eligible_windows']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Positive-mean slices",
            "",
            "| Model | Regime | Trades | Stress mean bps | LCB bps | Adjusted LCB bps | Status |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["positive_mean_slices"]:
        lines.append(
            f"| {row['model_id']} | {row['regime']} | {row['closed_trades']} | "
            f"{_fmt(row['mean_net_bps'])} | {_fmt(row['lcb_95_net_bps'])} | "
            f"{_fmt(row['multiple_testing_lcb_95_net_bps'])} | {row['status']} |"
        )

    nearest = report["models"]["ema_trend_12_48_v1"]["specializations"]["neutral"]
    lines.extend(
        [
            "",
            "## Nearest candidate: EMA 12/48 in neutral",
            "",
            (
                "The aggregate holdout LCB is positive before multiple-testing "
                "correction, but performance decays across time and the latest "
                "window is negative."
            ),
            "",
            "| Window | Trades | Stress mean bps | LCB bps | Adjusted LCB bps |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for split in POST_DEVELOPMENT:
        metrics = nearest["window_metrics"][split]
        lines.append(
            f"| {split} | {metrics['closed_trades']} | {_fmt(metrics['mean_net_bps'])} | "
            f"{_fmt(metrics['lcb_95_net_bps'])} | "
            f"{_fmt(metrics['multiple_testing_lcb_95_net_bps'])} |"
        )
    lines.append(
        f"| combined holdout | {nearest['closed_trades']} | {_fmt(nearest['mean_net_bps'])} | "
        f"{_fmt(nearest['lcb_95_net_bps'])} | "
        f"{_fmt(nearest['multiple_testing_lcb_95_net_bps'])} |"
    )

    overfit = report["models"]["mean_reversion_24_v1"]
    lines.extend(
        [
            "",
            "## Development-only false positive",
            "",
            (
                "`mean_reversion_24_v1 / volatile_mixed` passed even the adjusted "
                "LCB on development, then failed every later LCB and became "
                "negative in sanity. It is retained as a concrete overfitting "
                "warning."
            ),
            "",
            "| Window | Trades | Stress mean bps | LCB bps | Adjusted LCB bps |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for split in report["uniform_splits"]:
        metrics = overfit["windows"][split]["stress"]["regimes"]["volatile_mixed"]
        lines.append(
            f"| {split} | {metrics['closed_trades']} | {_fmt(metrics['mean_net_bps'])} | "
            f"{_fmt(metrics['lcb_95_net_bps'])} | "
            f"{_fmt(metrics['multiple_testing_lcb_95_net_bps'])} |"
        )

    lines.extend(
        [
            "",
            "## Retained non-uniform evidence",
            "",
            (
                "These rows are diagnostic only. Their data, cost contracts, and "
                "regime definitions are not uniform enough to merge into the "
                "ranking above."
            ),
            "",
            "| Artifact | Slice | Trades | Mean bps | LCB bps | Verdict |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in report["retained_nonuniform_evidence"]:
        lines.append(
            f"| {row['artifact_id']} | {row['slice']} | {row['closed_trades']} | "
            f"{_fmt(row['mean_net_bps'])} | {_fmt(row['lcb_95_net_bps'])} | "
            f"{row['verdict']} |"
        )
    lines.extend(
        [
            "",
            "## Evidence limitations",
            "",
            (
                "- Named Pantheon/Genetics agents have no retained promotable "
                "checkpoints or common costed regime ledgers."
            ),
            (
                "- CarryFlow and funding candidates have insufficient or zero "
                "trades and are excluded from ranking."
            ),
            (
                "- Trade-level LCB assumes independent observations; overlapping "
                "symbols can make it optimistic."
            ),
            "- This is retrospective model selection and cannot authorize paper/live trading.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    candidate = json.loads(FREQTRADE_CANDIDATE.read_text(encoding="utf-8"))
    tape_manifest = json.loads(TAPE_MANIFEST_PATH.read_text(encoding="utf-8"))
    expected_sha = registry["dataset"]["source_manifest_sha256"]
    actual_sha = tape_manifest["source_integrity"]["dataset_sha256"]
    if actual_sha != expected_sha:
        raise RuntimeError("Feature tape source SHA mismatch")
    frame = load_feature_tape(TAPE_PATH)
    regime_lookup = build_market_regime_lookup(frame)
    splits = candidate["splits"]
    costs = {
        name: _cost_model(row) for name, row in registry["costs"].items()
    }
    trial_count = len(registry["strategies"]) * (len(REGIME_ORDER) - 1)

    models: dict[str, Any] = {}
    for strategy in registry["strategies"]:
        model_id = str(strategy["id"])
        signal = build_signal(frame, strategy)
        windows: dict[str, Any] = {}
        holdout_ledgers: dict[str, list[pd.DataFrame]] = {
            "base": [],
            "stress": [],
        }
        for split, timerange in splits.items():
            start, end = timerange.split("-")
            windows[split] = {}
            for cost_name, cost in costs.items():
                result = simulate_targets(
                    frame,
                    signal,
                    start_timestamp=_timestamp(start),
                    end_timestamp=_timestamp(end),
                    costs=cost,
                    trial_count=trial_count,
                )
                ledger = attribute_trade_regimes(result.ledger, regime_lookup)
                unknown_trades = int((ledger["market_regime"] == "unknown").sum())
                if unknown_trades:
                    raise RuntimeError(
                        f"Unknown regime attribution: {model_id}/{split}/{cost_name}="
                        f"{unknown_trades}"
                    )
                windows[split][cost_name] = {
                    "overall": _metrics(ledger, trial_count=trial_count),
                    "regimes": _regime_metrics(ledger, trial_count=trial_count),
                }
                if split in POST_DEVELOPMENT:
                    holdout_ledgers[cost_name].append(ledger)

        holdout: dict[str, Any] = {}
        for cost_name, ledgers in holdout_ledgers.items():
            combined = pd.concat(ledgers, ignore_index=True)
            holdout[cost_name] = {
                "overall": _metrics(combined, trial_count=trial_count),
                "regimes": _regime_metrics(combined, trial_count=trial_count),
            }
        specializations: dict[str, Any] = {}
        for regime in REGIME_ORDER[:-1]:
            stress_windows = {
                split: windows[split]["stress"]["regimes"][regime]
                for split in POST_DEVELOPMENT
            }
            aggregate = holdout["stress"]["regimes"][regime]
            status, reasons = _specialization_status(aggregate, stress_windows)
            eligible = sum(
                value["closed_trades"] >= 10 for value in stress_windows.values()
            )
            positive = sum(
                value["closed_trades"] >= 10
                and value["mean_net_bps"] is not None
                and value["mean_net_bps"] > 0
                for value in stress_windows.values()
            )
            specializations[regime] = {
                **aggregate,
                "status": status,
                "reasons": reasons,
                "eligible_windows": eligible,
                "positive_windows": positive,
                "window_metrics": stress_windows,
            }
        models[model_id] = {
            "family": strategy["family"],
            "kind": strategy["kind"],
            "params": strategy["params"],
            "windows": windows,
            "holdout": holdout,
            "specializations": specializations,
        }

    ranked_rows: list[dict[str, Any]] = []
    for model_id, model in models.items():
        if model_id in {"flat_v1", "long_only_v1"}:
            continue
        for regime, metrics in model["specializations"].items():
            ranked_rows.append({"model_id": model_id, "regime": regime, **metrics})
    confirmed = [row for row in ranked_rows if row["status"] == "CONFIRMED"]
    promising = [
        row for row in ranked_rows if row["status"] == "PROMISING_UNCONFIRMED"
    ]
    positive = sorted(
        [
            row
            for row in ranked_rows
            if row["closed_trades"] >= 10
            and row["mean_net_bps"] is not None
            and row["mean_net_bps"] > 0
        ],
        key=lambda row: row["mean_net_bps"],
        reverse=True,
    )
    best_by_regime: dict[str, Any] = {}
    for regime in REGIME_ORDER[:-1]:
        candidates = [
            row
            for row in ranked_rows
            if row["regime"] == regime and row["closed_trades"] >= 10
        ]
        if candidates:
            best_by_regime[regime] = max(
                candidates,
                key=lambda row: (
                    row["multiple_testing_lcb_95_net_bps"]
                    if row["multiple_testing_lcb_95_net_bps"] is not None
                    else float("-inf")
                ),
            )

    report = {
        "schema_version": "panteon.current_model_regime_audit.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "source": {
            "registry": str(REGISTRY_PATH.relative_to(ROOT)).replace("\\", "/"),
            "registry_sha256": _sha256(REGISTRY_PATH),
            "feature_tape": str(TAPE_PATH.relative_to(ROOT)).replace("\\", "/"),
            "feature_tape_sha256": _sha256(TAPE_PATH),
            "dataset_sha256": actual_sha,
        },
        "regime_contract": REGIME_CONTRACT,
        "uniform_splits": splits,
        "costs": {
            name: {
                "fee_bps_per_fill": cost.fee_bps_per_fill,
                "slippage_bps_per_fill": cost.slippage_bps_per_fill,
                "round_trip_bps": cost.round_trip_bps,
            }
            for name, cost in costs.items()
        },
        "multiple_testing_trials": trial_count,
        "model_inventory": {
            "evaluated": len(registry["strategies"]),
            "uniformly_ranked": len(registry["strategies"]) - 2,
            "reference_baselines": ["flat_v1", "long_only_v1"],
            "named_pantheon_agents": "archived source only; no common retained costed ledgers",
        },
        "models": models,
        "best_by_regime": best_by_regime,
        "confirmed_specializations": confirmed,
        "promising_unconfirmed": promising,
        "positive_mean_slices": positive,
        "retained_nonuniform_evidence": _retained_nonuniform_evidence(),
        "excluded_current_artifacts": {
            "LongHorizonTrendStrategyV1": (
                "separate Freqtrade report: no regime passed base+stress LCB and "
                "candidate is terminal rejected"
            ),
            "UpstreamSampleDryRunStrategy": (
                "engineering sample; negative costed expectancy on only 60 days"
            ),
            "Pantheon_named_agents": (
                "archived source only; no retained promotable checkpoints or common "
                "costed regime ledgers"
            ),
            "CarryFlowAgentV2": (
                "insufficient prospective trades and terminal-rejected evidence roots"
            ),
            "funding_carry_hourly_v1": "zero activation; projected carry below cost floor",
        },
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    MARKDOWN_PATH.write_text(_render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "confirmed_specializations": len(confirmed),
                "promising_unconfirmed": len(promising),
                "positive_mean_slices": len(positive),
                "best_by_regime": {
                    regime: {
                        "model_id": row["model_id"],
                        "trades": row["closed_trades"],
                        "stress_mean_bps": row["mean_net_bps"],
                        "lcb_bps": row["lcb_95_net_bps"],
                        "adjusted_lcb_bps": row[
                            "multiple_testing_lcb_95_net_bps"
                        ],
                        "status": row["status"],
                    }
                    for regime, row in best_by_regime.items()
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
