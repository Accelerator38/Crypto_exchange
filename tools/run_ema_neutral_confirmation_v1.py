from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from simple_research.confirmation import (  # noqa: E402
    aggregate_complete_minutes_to_hour,
    apply_regime_allowlist,
    apply_regime_entry_gate,
    moving_block_bootstrap,
)
from simple_research.dataset import REQUIRED_COLUMNS, validate_ohlcv  # noqa: E402
from simple_research.regimes import (  # noqa: E402
    REGIME_CONTRACT,
    build_market_regime_lookup,
)
from simple_research.simulator import CostModel, simulate_targets  # noqa: E402
from simple_research.strategies import build_signal  # noqa: E402


DEFAULT_CONTRACT_PATH = (
    ROOT / "configs" / "simple_research" / "ema_neutral_confirmation_v1.json"
)
HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_ms(value: str) -> int:
    return int(pd.Timestamp(value).timestamp() * 1000)


def _cost_model(row: dict[str, Any]) -> CostModel:
    return CostModel(
        fee_bps_per_fill=float(row["fee_bps_per_fill"]),
        slippage_bps_per_fill=float(row["slippage_bps_per_fill"]),
    )


def _load_and_verify_sources(
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dataset = contract["confirmation_dataset"]
    warmup_path = ROOT / dataset["warmup_tape"]
    extension_path = ROOT / dataset["extension_source"]
    integrity_path = ROOT / dataset["extension_integrity_manifest"]
    expected_hashes = {
        warmup_path: dataset["warmup_tape_sha256"],
        extension_path: dataset["extension_source_sha256"],
        integrity_path: dataset["extension_integrity_manifest_sha256"],
    }
    actual_hashes = {path: _sha256(path) for path in expected_hashes}
    mismatches = [
        str(path)
        for path, expected in expected_hashes.items()
        if actual_hashes[path].lower() != str(expected).lower()
    ]
    if mismatches:
        raise RuntimeError(f"sealed source SHA mismatch: {mismatches}")
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if integrity.get("validation", {}).get("passed") is not True:
        raise RuntimeError("extension source integrity manifest is not PASS")

    warmup = pd.read_parquet(warmup_path, columns=list(REQUIRED_COLUMNS))
    minute = pd.read_csv(extension_path, usecols=list(REQUIRED_COLUMNS))
    hourly = aggregate_complete_minutes_to_hour(minute)
    source = {
        "warmup_tape": str(warmup_path.relative_to(ROOT)).replace("\\", "/"),
        "warmup_tape_sha256": actual_hashes[warmup_path],
        "extension_source": str(extension_path.relative_to(ROOT)).replace(
            "\\", "/"
        ),
        "extension_source_sha256": actual_hashes[extension_path],
        "extension_integrity_manifest": str(
            integrity_path.relative_to(ROOT)
        ).replace("\\", "/"),
        "extension_integrity_manifest_sha256": actual_hashes[integrity_path],
        "extension_integrity_passed": True,
    }
    return warmup, hourly, source


def _verify_overlap_parity(
    warmup: pd.DataFrame,
    extension: pd.DataFrame,
) -> dict[str, Any]:
    last_warmup = int(warmup["timestamp"].max())
    overlap = extension.loc[extension["timestamp"] <= last_warmup]
    merged = overlap.merge(
        warmup,
        on=["symbol", "timestamp"],
        how="left",
        suffixes=("_extension", "_warmup"),
        validate="one_to_one",
    )
    missing = int(merged["close_warmup"].isna().sum())
    max_differences: dict[str, float] = {}
    mismatch_rows = pd.Series(False, index=merged.index)
    for column in ("open", "high", "low", "close"):
        difference = (
            merged[f"{column}_extension"] - merged[f"{column}_warmup"]
        ).abs()
        max_differences[column] = float(difference.max()) if len(difference) else 0.0
        mismatch_rows |= ~np.isclose(
            merged[f"{column}_extension"],
            merged[f"{column}_warmup"],
            rtol=0.0,
            atol=1e-12,
        )
    mismatch_count = int(mismatch_rows.sum())
    if missing or mismatch_count:
        raise RuntimeError(
            "1m-to-1h extension does not match sealed hourly OHLC overlap: "
            f"missing={missing}, mismatches={mismatch_count}"
        )
    return {
        "passed": True,
        "overlap_rows": int(len(merged)),
        "missing_warmup_rows": missing,
        "ohlc_mismatch_rows": mismatch_count,
        "maximum_absolute_difference": max_differences,
    }


def _combine_history(
    warmup: pd.DataFrame,
    extension: pd.DataFrame,
    symbols: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    last_warmup = int(warmup["timestamp"].max())
    post = extension.loc[extension["timestamp"] > last_warmup].copy()
    combined = pd.concat([warmup, post], ignore_index=True)
    combined = combined.sort_values(
        ["symbol", "timestamp"], kind="stable", ignore_index=True
    )
    validation = validate_ohlcv(
        combined,
        timeframe_ms=HOUR_MS,
        expected_symbols=symbols,
    )
    return combined, {
        "post_warmup_rows": int(len(post)),
        "first_post_warmup_timestamp": int(post["timestamp"].min()),
        "last_post_warmup_timestamp": int(post["timestamp"].max()),
        "combined_validation": validation,
    }


def _slice_metrics(ledger: pd.DataFrame, *, trial_count: int) -> dict[str, Any]:
    values = [float(value) for value in ledger["net_bps"]]
    count = len(values)
    mean = statistics.mean(values) if values else None
    std = statistics.stdev(values) if count > 1 else None
    standard_error = std / math.sqrt(count) if std is not None else None
    ordinary_z = NormalDist().inv_cdf(0.95)
    adjusted_z = NormalDist().inv_cdf(1.0 - 0.05 / trial_count)
    return {
        "closed_trades": count,
        "fills": int(ledger["fills"].sum()) if count else 0,
        "mean_net_bps": mean,
        "lcb_95_net_bps": (
            mean - ordinary_z * standard_error
            if mean is not None and standard_error is not None
            else None
        ),
        "selection_adjusted_lcb_95_net_bps": (
            mean - adjusted_z * standard_error
            if mean is not None and standard_error is not None
            else None
        ),
        "win_rate": (
            sum(value > 0 for value in values) / count if count else None
        ),
    }


def _per_symbol(ledger: pd.DataFrame, *, trial_count: int) -> dict[str, Any]:
    return {
        str(symbol): _slice_metrics(group, trial_count=trial_count)
        for symbol, group in ledger.groupby("symbol", sort=True)
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _render_markdown(report: dict[str, Any]) -> str:
    base = report["candidate_evaluation"]["base"]["metrics"]
    stress = report["candidate_evaluation"]["stress"]["metrics"]
    baseline = report["unfiltered_ema_baseline"]["stress"]["metrics"]
    bootstrap = report["candidate_evaluation"]["stress"]["block_bootstrap"]
    lines = [
        f"# {report['candidate_id']} evaluation",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Verdict",
        "",
        f"- Status: `{report['status']}`.",
        f"- Passed: `{str(report['passed']).lower()}`.",
        f"- Evidence class: `{report['window']['classification']}`.",
        "- Paper/live/orders/promotion authority remain `false`.",
        "",
        "## Data integrity",
        "",
        f"- Confirmation window: `{report['window']['start']}` to "
        f"`{report['window']['end']}` (end exclusive).",
        f"- Calendar days: `{report['window']['calendar_days']}`; required: "
        f"`{report['window']['minimum_calendar_days']}`.",
        f"- 1m-to-1h overlap rows: "
        f"`{report['data_integrity']['overlap']['overlap_rows']}`; OHLC "
        "mismatches: `0`.",
        "",
        "## Aggregate metrics",
        "",
        "| Profile | Trades | Fills | Mean net bps | LCB bps | Adjusted LCB bps | Drawdown |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| candidate base | {base['closed_trades']} | {base['fills']} | "
        f"{_fmt(base['mean_net_bps'])} | {_fmt(base['lcb_95_net_bps'])} | "
        f"{_fmt(base['multiple_testing_lcb_95_net_bps'])} | "
        f"{_fmt(base['max_drawdown'], 4)} |",
        f"| candidate stress | {stress['closed_trades']} | {stress['fills']} | "
        f"{_fmt(stress['mean_net_bps'])} | {_fmt(stress['lcb_95_net_bps'])} | "
        f"{_fmt(stress['multiple_testing_lcb_95_net_bps'])} | "
        f"{_fmt(stress['max_drawdown'], 4)} |",
        f"| unfiltered EMA stress | {baseline['closed_trades']} | "
        f"{baseline['fills']} | {_fmt(baseline['mean_net_bps'])} | "
        f"{_fmt(baseline['lcb_95_net_bps'])} | "
        f"{_fmt(baseline['multiple_testing_lcb_95_net_bps'])} | "
        f"{_fmt(baseline['max_drawdown'], 4)} |",
        "",
        "## Time stability",
        "",
        "| Half | Trades | Stress mean bps | LCB bps |",
        "|---|---:|---:|---:|",
    ]
    for half, metrics in report["candidate_evaluation"]["stress"][
        "time_halves"
    ].items():
        lines.append(
            f"| {half} | {metrics['closed_trades']} | "
            f"{_fmt(metrics['mean_net_bps'])} | "
            f"{_fmt(metrics['lcb_95_net_bps'])} |"
        )
    lines.extend(
        [
            "",
            "## Stress metrics by symbol",
            "",
            "| Symbol | Trades | Mean net bps | LCB bps |",
            "|---|---:|---:|---:|",
        ]
    )
    for symbol, metrics in report["candidate_evaluation"]["stress"][
        "per_symbol"
    ].items():
        lines.append(
            f"| {symbol} | {metrics['closed_trades']} | "
            f"{_fmt(metrics['mean_net_bps'])} | "
            f"{_fmt(metrics['lcb_95_net_bps'])} |"
        )
    lines.extend(
        [
            "",
            "## 24-hour moving-block bootstrap",
            "",
            f"- Observed stress total return: "
            f"`{_fmt(bootstrap['observed_total_return'] * 100, 4)}%`.",
            f"- 95% total-return LCB: "
            f"`{_fmt(bootstrap['total_return_lcb'] * 100, 4)}%`.",
            f"- Observations/nominal blocks: `{bootstrap['observations']}` / "
            f"`{bootstrap['nominal_blocks']}`.",
            "",
            "## Gate",
            "",
            "| Check | Passed |",
            "|---|---|",
        ]
    )
    for name, passed in report["gate_checks"].items():
        lines.append(f"| {name} | {str(passed).lower()} |")
    lines.extend(
        [
            "",
            "No threshold may be changed under this candidate ID after this run.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a fixed EMA-neutral profile without order authority."
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT_PATH,
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (
            ROOT
            / "Reports"
            / "SimpleResearch"
            / str(contract["candidate_id"])
        )
    )
    report_path = output_dir / "confirmation_report.json"
    markdown_path = output_dir / "confirmation_report.md"
    trial_count = int(contract["selection_origin"]["selection_trials_debt"])
    symbols = tuple(contract["confirmation_dataset"]["symbols"])
    warmup, extension, source = _load_and_verify_sources(contract)
    overlap = _verify_overlap_parity(warmup, extension)
    frame, combination = _combine_history(warmup, extension, symbols)

    start = _utc_ms(contract["confirmation_dataset"]["start_inclusive"])
    end = _utc_ms(contract["confirmation_dataset"]["end_exclusive"])
    expected_last = end - HOUR_MS
    if int(frame["timestamp"].max()) < expected_last:
        raise RuntimeError("combined source does not cover confirmation end")
    calendar_days = (end - start) // DAY_MS

    regime_lookup = build_market_regime_lookup(frame)
    base_signal = build_signal(frame, contract["candidate"])
    gate_semantics = contract["candidate"].get(
        "regime_gate_semantics",
        "target_mask_flat_outside_allowed_regime",
    )
    if gate_semantics == "target_mask_flat_outside_allowed_regime":
        candidate_signal, regimes = apply_regime_allowlist(
            frame,
            base_signal,
            regime_lookup,
            allowed_regimes=contract["candidate"]["allowed_market_regimes"],
        )
    elif gate_semantics == "entry_only_follow_base_target_after_entry":
        candidate_signal, regimes = apply_regime_entry_gate(
            frame,
            base_signal,
            regime_lookup,
            allowed_entry_regimes=contract["candidate"][
                "allowed_market_regimes"
            ],
        )
    else:
        raise RuntimeError(f"unknown regime gate semantics: {gate_semantics}")
    confirmation_mask = (frame["timestamp"] >= start - HOUR_MS) & (
        frame["timestamp"] < end
    )
    unknown_decisions = int((regimes.loc[confirmation_mask] == "unknown").sum())
    if unknown_decisions:
        raise RuntimeError(
            f"confirmation window has {unknown_decisions} unknown regime decisions"
        )

    costs = {
        name: _cost_model(row) for name, row in contract["costs"].items()
    }
    candidate_results = {
        name: simulate_targets(
            frame,
            candidate_signal,
            start_timestamp=start,
            end_timestamp=end,
            costs=cost,
            trial_count=trial_count,
        )
        for name, cost in costs.items()
    }
    baseline_stress = simulate_targets(
        frame,
        base_signal,
        start_timestamp=start,
        end_timestamp=end,
        costs=costs["stress"],
        trial_count=trial_count,
    )
    midpoint = start + (end - start) // 2
    half_ranges = {
        "first": (start, midpoint),
        "second": (midpoint, end),
    }
    half_metrics = {
        name: simulate_targets(
            frame,
            candidate_signal,
            start_timestamp=half_start,
            end_timestamp=half_end,
            costs=costs["stress"],
            trial_count=trial_count,
        ).metrics
        for name, (half_start, half_end) in half_ranges.items()
    }
    bootstrap_config = contract["confirmation_gate"]["block_bootstrap"]
    bootstrap = moving_block_bootstrap(
        candidate_results["stress"].portfolio["net_return"],
        block_size=int(bootstrap_config["block_hours"]),
        replicates=int(bootstrap_config["replicates"]),
        seed=int(bootstrap_config["seed"]),
        confidence=float(bootstrap_config["confidence"]),
    )

    stress = candidate_results["stress"].metrics
    gate = contract["confirmation_gate"]
    traded_symbols = len(stress["symbol_trade_counts"])
    checks = {
        "minimum_calendar_days": (
            calendar_days
            >= int(
                contract["confirmation_dataset"][
                    "minimum_calendar_days_for_confirmation"
                ]
            )
        ),
        "minimum_closed_trades": (
            stress["closed_trades"] >= int(gate["minimum_closed_trades"])
        ),
        "minimum_fills": stress["fills"] >= int(gate["minimum_fills"]),
        "minimum_symbols_with_trades": (
            traded_symbols >= int(gate["minimum_symbols_with_trades"])
        ),
        "maximum_symbol_trade_share": (
            stress["max_symbol_trade_share"]
            <= float(gate["maximum_symbol_trade_share"])
        ),
        "positive_stress_mean_net_bps": stress["mean_net_bps"] > 0.0,
        "positive_stress_lcb_95": (
            stress["lcb_95_net_bps"] is not None
            and stress["lcb_95_net_bps"] > 0.0
        ),
        "positive_selection_adjusted_lcb_95": (
            stress["multiple_testing_lcb_95_net_bps"] is not None
            and stress["multiple_testing_lcb_95_net_bps"] > 0.0
        ),
        "positive_stress_mean_in_both_time_halves": all(
            metrics["closed_trades"] > 0 and metrics["mean_net_bps"] > 0.0
            for metrics in half_metrics.values()
        ),
        "maximum_drawdown": stress["max_drawdown"] <= float(
            gate["maximum_drawdown"]
        ),
        "positive_block_bootstrap_total_return_lcb": (
            bootstrap["total_return_lcb"] > 0.0
        ),
    }
    gate_passed = all(checks.values())
    sufficiency_checks = (
        "minimum_closed_trades",
        "minimum_fills",
        "minimum_symbols_with_trades",
    )
    performance_checks = (
        "positive_stress_mean_net_bps",
        "positive_stress_lcb_95",
        "positive_selection_adjusted_lcb_95",
        "positive_stress_mean_in_both_time_halves",
        "positive_block_bootstrap_total_return_lcb",
    )
    classification = contract["confirmation_dataset"]["classification"]
    independent_authority = bool(
        contract.get("independent_confirmation_authority", True)
    )
    passed = gate_passed and independent_authority
    count_sufficient = all(checks[name] for name in sufficiency_checks)
    performance_failed = any(not checks[name] for name in performance_checks)
    if passed:
        status = "PASSED_SEALED_CONFIRMATION"
    elif count_sufficient and performance_failed:
        status = (
            "FAILED_DIAGNOSTIC_REPLAY"
            if not independent_authority
            else "FAILED_SEALED_CONFIRMATION_EARLY"
        )
    elif not count_sufficient:
        status = "SEALED_HOLDBACK_INSUFFICIENT"
    elif gate_passed and not independent_authority:
        status = "POSITIVE_DIAGNOSTIC_REQUIRES_INDEPENDENT_CONFIRMATION"
    else:
        status = "SEALED_HOLDBACK_INSUFFICIENT_DURATION"

    window_regimes = regimes.loc[
        (frame["timestamp"] >= start) & (frame["timestamp"] < end)
    ]
    regime_counts = {
        str(regime): int(count)
        for regime, count in window_regimes.value_counts().sort_index().items()
    }
    report = {
        "schema_version": "simple_research.ema_neutral_confirmation.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "candidate_id": contract["candidate_id"],
        "contract_path": str(contract_path.relative_to(ROOT)).replace("\\", "/"),
        "contract_sha256": _sha256(contract_path),
        "status": status,
        "passed": passed,
        "gate_passed": gate_passed,
        "independent_confirmation_authority": independent_authority,
        "window": {
            "classification": classification,
            "start": contract["confirmation_dataset"]["start_inclusive"],
            "end": contract["confirmation_dataset"]["end_exclusive"],
            "calendar_days": int(calendar_days),
            "minimum_calendar_days": int(
                contract["confirmation_dataset"][
                    "minimum_calendar_days_for_confirmation"
                ]
            ),
        },
        "source": source,
        "data_integrity": {
            "overlap": overlap,
            "combination": combination,
            "unknown_regime_decisions": unknown_decisions,
        },
        "regime_contract": REGIME_CONTRACT,
        "regime_observation_counts": regime_counts,
        "signal_diagnostics": {
            "regime_gate_semantics": gate_semantics,
            "unfiltered_nonzero_bar_targets": int(
                (base_signal.loc[confirmation_mask] != 0).sum()
            ),
            "allowed_nonzero_bar_targets": int(
                (candidate_signal.loc[confirmation_mask] != 0).sum()
            ),
        },
        "candidate_evaluation": {
            name: {
                "metrics": result.metrics,
                "per_symbol": _per_symbol(
                    result.ledger,
                    trial_count=trial_count,
                ),
                **(
                    {
                        "time_halves": half_metrics,
                        "block_bootstrap": bootstrap,
                    }
                    if name == "stress"
                    else {}
                ),
            }
            for name, result in candidate_results.items()
        },
        "unfiltered_ema_baseline": {
            "stress": {
                "metrics": baseline_stress.metrics,
                "per_symbol": _per_symbol(
                    baseline_stress.ledger,
                    trial_count=trial_count,
                ),
            }
        },
        "gate_checks": checks,
        "failures": [name for name, value in checks.items() if not value],
        "selection_trials_debt": trial_count,
        "orders_enabled": False,
        "paper_allowed": False,
        "live_allowed": False,
        "promotion_authority": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "passed": passed,
                "calendar_days": calendar_days,
                "stress_metrics": stress,
                "bootstrap_total_return_lcb": bootstrap["total_return_lcb"],
                "failures": report["failures"],
                "report": str(report_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
