from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RUNTIME = SRC / "panteon_runtime"
for path in (SRC, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import panteon_agents  # noqa: E402
from carryflow_policy import (  # noqa: E402
    carryflow_profile_ids,
    get_carryflow_profile,
)
from panteon_agents import CarryFlowAgentV2  # noqa: E402
from panteon_v2.analysis.retrodate_market_runner import (  # noqa: E402
    RetrodateSnapshotState,
    load_retrodate_year_snapshots,
)
from panteon_v2.domain.types import MarketSnapshot  # noqa: E402
from panteon_v2.policy import (  # noqa: E402
    CarryFlowEvidenceTape,
    CarryFlowWarmupSeed,
    HistoricalDerivativesContext,
    PolicyReplayRunner,
    PolicyTarget,
    compute_runtime_fingerprint,
    load_policy_manifest,
    seal_manifest_payload,
)
from panteon_v2.policy.replay_runner import ReplayTradeOutcome  # noqa: E402
from panteon_v2.shadow.adapters import V1AgentAdapter  # noqa: E402


DEFAULT_DATA = ROOT / "Retrodate" / "bitget_futures_current_20260709" / "crypto_1m_2026_all_symbols.csv"
DEFAULT_SYMBOLS = ("BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK")


def _git_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    revision = result.stdout.strip().lower()
    if result.returncode != 0 or len(revision) < 7:
        raise RuntimeError("git revision unavailable")
    return revision


def _actor_config(profile_id: str = "screened_short_v1") -> dict[str, object]:
    profile = get_carryflow_profile(profile_id)
    return {"PROFILE_ID": profile.profile_id}


def _parse_symbols(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    symbols = tuple(
        dict.fromkeys(
            str(item).strip().upper()
            for item in str(value).split(",")
            if str(item).strip()
        )
    )
    if not symbols:
        raise ValueError("symbols must be non-empty")
    return symbols


def build_replay_manifest_payload(
    *,
    policy_id: str,
    symbols: Sequence[str],
    created_at: datetime,
    source_revision: str,
    runtime_fingerprint_sha256: str,
    round_trip_fee_bps: float,
    slippage_bps: float,
    safety_buffer_bps: float,
    capital_fraction: float,
    max_notional_usd: float,
    max_daily_loss_usd: float,
    stride_minutes: int,
    profile_id: str = "screened_short_v1",
) -> dict[str, Any]:
    profile = get_carryflow_profile(profile_id)
    config = _actor_config(profile.profile_id)
    required_move = round_trip_fee_bps + slippage_bps + safety_buffer_bps
    max_holding_minutes = profile.hold_bars * int(stride_minutes)
    rules = [
        {
            "symbol": str(symbol).upper(),
            "regime": str(regime).lower(),
            "direction": "SHORT",
            "min_expected_move_bps": required_move,
            "max_spread_bps": profile.max_spread_bps,
            "max_slippage_bps": profile.max_slippage_bps,
            "min_regime_confidence": profile.min_regime_confidence,
            "risk_mult": profile.risk_mult,
        }
        for symbol in symbols
        for regime in profile.allowed_regimes
    ]
    return {
        "schema_version": "panteon.policy.v1",
        "policy_id": policy_id,
        "target": "replay",
        "exchange": "BITGET",
        "actor": "CarryFlowAgentV2",
        "actor_config": config,
        "signal_model": {
            "feature": profile.signal_feature,
            "intercept_bps": 0.0,
            "slope_bps_per_unit": profile.signal_slope_bps_per_unit,
            "lcb_haircut_bps": profile.signal_lcb_haircut_bps,
            "min_feature_value": profile.signal_min_feature,
            "max_expected_move_bps": profile.signal_max_expected_move_bps,
        },
        "data": {
            "bar_interval_seconds": int(stride_minutes) * 60,
            "cadence_tolerance_seconds": 0,
            "max_bar_close_lag_seconds": 120,
            "max_derivatives_age_seconds": profile.max_data_age_seconds,
            "required_context_coverage_pct": 95.0,
        },
        "created_at": created_at.isoformat(),
        "expires_at": (created_at + timedelta(hours=24)).isoformat(),
        "source_revision": source_revision,
        "runtime_fingerprint_sha256": runtime_fingerprint_sha256,
        "costs": {
            "round_trip_fee_bps": round_trip_fee_bps,
            "slippage_bps": slippage_bps,
            "safety_buffer_bps": safety_buffer_bps,
        },
        "risk": {
            "capital_fraction": capital_fraction,
            "max_notional_usd": max_notional_usd,
            "max_open_positions": profile.max_positions,
            "max_daily_loss_usd": max_daily_loss_usd,
            "stop_loss_pct": profile.stop_pct * 100.0,
            "max_holding_minutes": max_holding_minutes,
            "max_signal_age_seconds": profile.max_signal_age_seconds,
        },
        "rules": rules,
        "evidence": [],
    }


def _filter_snapshot(snapshot: MarketSnapshot, symbols: set[str]) -> MarketSnapshot | None:
    selected = {
        symbol: price
        for symbol, price in snapshot.prices.items()
        if _base_symbol(symbol) in symbols
    }
    if not selected:
        return None
    keys = set(selected)
    return replace(
        snapshot,
        prices=selected,
        volumes={symbol: snapshot.volumes.get(symbol, 0.0) for symbol in keys},
        funding={symbol: snapshot.funding.get(symbol, 0.0) for symbol in keys},
        fees_bps_by_symbol={
            symbol: snapshot.fees_bps_by_symbol.get(symbol, 0.0) for symbol in keys
        },
        lookback_returns_pct={
            symbol: dict(snapshot.lookback_returns_pct.get(symbol, {})) for symbol in keys
        },
        lookback_volatility_pct={
            symbol: dict(snapshot.lookback_volatility_pct.get(symbol, {})) for symbol in keys
        },
        technicals_by_symbol={
            symbol: snapshot.technicals_by_symbol[symbol]
            for symbol in keys
            if symbol in snapshot.technicals_by_symbol
        },
        regimes_by_symbol={
            symbol: snapshot.regime_for_symbol(symbol) for symbol in keys
        },
        regime_features_by_symbol={
            symbol: snapshot.regime_features_for_symbol(symbol) for symbol in keys
        },
        bar_opens={
            symbol: snapshot.bar_opens[symbol]
            for symbol in keys
            if symbol in snapshot.bar_opens
        },
        bar_highs={
            symbol: snapshot.bar_highs[symbol]
            for symbol in keys
            if symbol in snapshot.bar_highs
        },
        bar_lows={
            symbol: snapshot.bar_lows[symbol]
            for symbol in keys
            if symbol in snapshot.bar_lows
        },
        bar_closes={
            symbol: snapshot.bar_closes[symbol]
            for symbol in keys
            if symbol in snapshot.bar_closes
        },
    )


def _base_symbol(value: str) -> str:
    clean = str(value or "").upper().replace("/", "")
    return clean[:-4] if clean.endswith("USDT") else clean


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _slice_metrics(outcomes: Sequence[ReplayTradeOutcome]) -> dict[str, Any]:
    values = [row.net_pnl_usd for row in outcomes]
    closed = len(values)
    net = sum(values)
    expectancy = net / closed if closed else 0.0
    if closed >= 2:
        lcb = expectancy - 1.6448536 * statistics.stdev(values) / math.sqrt(closed)
    else:
        lcb = None
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    costs = sum(row.costs_usd for row in outcomes)
    notionals = [row.notional_usd for row in outcomes if row.notional_usd > 0.0]
    mean_cost_bps = (
        statistics.fmean(
            row.costs_usd / row.notional_usd * 10_000.0
            for row in outcomes
            if row.notional_usd > 0.0
        )
        if notionals
        else 0.0
    )
    positive = sum(value for value in values if value > 0.0)
    negative = abs(sum(value for value in values if value < 0.0))
    failures: list[str] = []
    if closed * 2 < 20:
        failures.append("fills_below_20")
    if closed < 10:
        failures.append("closed_trades_below_10")
    if expectancy <= 0.0:
        failures.append("nonpositive_expectancy")
    if lcb is None or lcb <= 0.0:
        failures.append("nonpositive_lcb")
    if mean_cost_bps <= 0.0 and closed > 0:
        failures.append("zero_cost_attribution")
    return {
        "filled_orders": closed * 2,
        "closed_trades": closed,
        "gross_pnl_usd": sum(row.gross_pnl_usd for row in outcomes),
        "total_costs_usd": costs,
        "net_pnl_usd": net,
        "expectancy_after_costs_usd": expectancy,
        "expectancy_lcb_usd": lcb,
        "win_rate": (sum(value > 0.0 for value in values) / closed if closed else 0.0),
        "profit_factor": positive / negative if negative > 0.0 else None,
        "max_drawdown_usd": drawdown,
        "mean_cost_bps": mean_cost_bps,
        "passed": not failures,
        "failures": failures,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc)
    if not 0.0 < float(args.oos_fraction) < 1.0:
        raise ValueError("oos_fraction must be in (0, 1)")
    tape_path_raw = getattr(args, "evidence_tape", None)
    warmup_seed_path_raw = getattr(args, "warmup_seed", None)
    context_path_raw = getattr(args, "derivatives_context_csv", None)
    allow_legacy = bool(getattr(args, "allow_legacy_split_input", False))
    if tape_path_raw and context_path_raw:
        raise ValueError(
            "--evidence-tape cannot be combined with --derivatives-context-csv"
        )
    if warmup_seed_path_raw and not tape_path_raw:
        raise ValueError("--warmup-seed requires --evidence-tape")

    requested_symbols = _parse_symbols(getattr(args, "symbols", None))
    requested_stride = getattr(args, "stride_minutes", None)
    tape = None
    if tape_path_raw:
        tape = CarryFlowEvidenceTape.from_jsonl(
            tape_path_raw,
            expected_symbols=requested_symbols or None,
        )
        symbols = tape.symbols
        if tape.bar_interval_seconds % 60:
            raise ValueError("evidence tape interval must be whole minutes")
        stride_minutes = tape.bar_interval_seconds // 60
        if (
            requested_stride is not None
            and int(requested_stride) != stride_minutes
        ):
            raise ValueError(
                "--stride-minutes does not match the evidence tape contract"
            )
        input_settings_source = "evidence_tape_contract"
    else:
        symbols = requested_symbols or DEFAULT_SYMBOLS
        stride_minutes = (
            60 if requested_stride is None else int(requested_stride)
        )
        if stride_minutes <= 0:
            raise ValueError("stride_minutes must be positive")
        input_settings_source = "legacy_diagnostic_arguments"

    profile = get_carryflow_profile(
        str(getattr(args, "profile", "screened_short_v1"))
    )
    regimes = profile.allowed_regimes
    flatten_end = not bool(getattr(args, "no_flatten_end", False))
    policy_id = args.policy_id or (
        f"carryflow-{profile.profile_id}-"
        f"{'-'.join(symbols).lower()}-{created_at:%Y%m%d%H%M%S}"
    )
    out_dir = Path(args.out_dir) if args.out_dir else (
        ROOT / "Reports" / "PolicyReplayV1" / f"{created_at:%Y%m%d_%H%M%S}_{policy_id}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_payload = seal_manifest_payload(
        build_replay_manifest_payload(
            policy_id=policy_id,
            symbols=symbols,
            created_at=created_at,
            source_revision=_git_revision(ROOT),
            runtime_fingerprint_sha256=compute_runtime_fingerprint(ROOT),
            round_trip_fee_bps=args.round_trip_fee_bps,
            slippage_bps=args.slippage_bps,
            safety_buffer_bps=args.safety_buffer_bps,
            capital_fraction=args.capital_fraction,
            max_notional_usd=args.max_notional_usd,
            max_daily_loss_usd=args.max_daily_loss_usd,
            stride_minutes=stride_minutes,
            profile_id=profile.profile_id,
        )
    )
    manifest_path = out_dir / "replay_manifest.json"
    _write_json(manifest_path, manifest_payload)
    loaded = load_policy_manifest(
        manifest_path,
        project_root=ROOT,
        expected_sha256=manifest_payload["manifest_sha256"],
        now=created_at,
        required_target=PolicyTarget.REPLAY,
        required_exchange="BITGET",
    )
    warmup_seed = None
    if tape_path_raw:
        assert tape is not None
        if warmup_seed_path_raw:
            warmup_seed = CarryFlowWarmupSeed.from_json(
                warmup_seed_path_raw,
                expected_symbols=symbols,
            )
            warmup_seed.validate_for_tape(tape)
        bundle = tape.replay_bundle(
            loaded.manifest.data,
            warmup_seed=warmup_seed,
        )
        filtered = list(bundle.snapshots)
        context_provider = bundle.derivatives_context
        input_contract = "unified_hash_chained_tape"
    else:
        if not allow_legacy:
            raise ValueError(
                "authoritative replay requires --evidence-tape; use "
                "--allow-legacy-split-input only for diagnostics"
            )
        state = RetrodateSnapshotState()
        snapshots = load_retrodate_year_snapshots(
            args.data,
            stride_minutes=stride_minutes,
            state=state,
            max_snapshots=args.max_snapshots,
            use_live_regime_detector=args.use_live_regime_detector,
        )
        selected_symbols = set(symbols)
        filtered = [
            row
            for snapshot in snapshots
            if (row := _filter_snapshot(snapshot, selected_symbols)) is not None
        ]
        if not filtered:
            raise ValueError("no replay snapshots remain after symbol filtering")
        context_provider = None
        if context_path_raw:
            context_provider = HistoricalDerivativesContext.from_csv(
                context_path_raw,
                symbols=symbols,
            )
        input_contract = "legacy_split_diagnostic_only"
    panteon_agents.set_fetcher(None)
    actor = V1AgentAdapter(label="CarryFlowAgentV2", v1_agent=CarryFlowAgentV2())
    warmup_bar_count = 0
    try:
        runner = PolicyReplayRunner.create(
            loaded_manifest=loaded,
            agent=actor,
            initial_capital_usd=args.initial_capital_usd,
            assumed_spread_bps=args.assumed_spread_bps,
            exchange_min_notional_usd=args.exchange_min_notional_usd,
            derivatives_context_provider=context_provider,
        )
        if warmup_seed is not None:
            warmup_bar_count = runner.actor_adapter.warmup(
                warmup_seed.snapshots()
            )
        panteon_agents.set_fetcher(context_provider)
        report = runner.run(filtered, flatten_end=flatten_end)
    finally:
        panteon_agents.set_fetcher(None)

    summary = report.as_dict()
    split_index = min(
        max(1, int(len(filtered) * (1.0 - args.oos_fraction))),
        max(1, len(filtered) - 1),
    ) if len(filtered) >= 2 else 0
    split_bar = filtered[split_index].bar if split_index else 0
    train_outcomes = [
        row for row in runner.trade_outcomes if split_bar and row.closed_bar < split_bar
    ]
    oos_outcomes = [
        row for row in runner.trade_outcomes if split_bar and row.opened_bar >= split_bar
    ]
    boundary_outcomes = [
        row
        for row in runner.trade_outcomes
        if row not in train_outcomes and row not in oos_outcomes
    ]
    train_metrics = _slice_metrics(train_outcomes)
    oos_metrics = _slice_metrics(oos_outcomes)
    robust_failures: list[str] = []
    if not report.evidence_eligible:
        robust_failures.extend(f"full:{reason}" for reason in report.evidence_failures)
    if not oos_metrics["passed"]:
        robust_failures.extend(f"oos:{reason}" for reason in oos_metrics["failures"])
    if tape is None:
        robust_failures.append("input:legacy_split_data_not_promotion_eligible")
    summary["sequential_split"] = {
        "oos_fraction": args.oos_fraction,
        "split_bar": split_bar,
        "split_timestamp": (
            filtered[split_index].timestamp.isoformat() if split_index else ""
        ),
        "train": train_metrics,
        "oos": oos_metrics,
        "boundary_trade_count": len(boundary_outcomes),
        "robust_passed_before_cost_stress": not robust_failures,
        "robust_failures": robust_failures,
    }
    summary["data_path"] = str(
        Path(tape_path_raw or args.data).resolve()
    )
    summary["output_dir"] = str(out_dir.resolve())
    summary["symbols"] = list(symbols)
    summary["bar_interval_minutes"] = stride_minutes
    summary["input_settings_source"] = input_settings_source
    summary["regimes"] = list(regimes)
    summary["research_only"] = True
    summary["promotion_authority"] = False
    summary["input_contract"] = input_contract
    summary["robust_input_contract"] = tape is not None
    summary["policy_source_revision"] = loaded.manifest.source_revision
    summary["evidence_source_revision"] = (
        tape.source_revision if tape is not None else ""
    )
    summary["cross_revision_replay"] = bool(
        tape is not None
        and tape.source_revision != loaded.manifest.source_revision
    )
    summary["evidence_tape"] = tape.describe() if tape is not None else None
    summary["warmup_seed"] = (
        {
            **warmup_seed.describe(),
            "prospective_for_tape": warmup_seed.is_prospective_for_tape(tape),
        }
        if warmup_seed is not None and tape is not None
        else None
    )
    summary["warmup_bars"] = warmup_bar_count
    summary["warmup_symbol_observations"] = warmup_bar_count * len(symbols)
    summary["continuous_warmup_bars_avoided"] = warmup_bar_count
    summary["flatten_end"] = flatten_end
    summary["end_positions_censored"] = (
        report.remaining_open_positions if not flatten_end else 0
    )
    summary["research_hypothesis"] = {
        "profile_id": profile.profile_id,
        "flow_model": profile.flow_model,
        "allowed_regimes": list(profile.allowed_regimes),
        "hold_bars": profile.hold_bars,
        "oi_lookback_bars": profile.oi_lookback_bars,
        "price_lookback_bars": profile.price_lookback_bars,
        "runtime_knob_count": 1,
    }
    summary["derivatives_context"] = (
        {
            "mode": "embedded_in_evidence_tape",
            "path": str(Path(tape_path_raw).resolve()),
            **context_provider.describe(),
        }
        if tape is not None
        else
        {
            "mode": "historical_point_in_time",
            "path": str(Path(context_path_raw).resolve()),
            **context_provider.describe(),
        }
        if context_provider is not None
        else {
            "mode": "missing_hard_block",
            "path": "",
            "rows": 0,
            "symbols": [],
        }
    )
    if tape is None:
        failures = list(summary["evidence_failures"])
        failures.append("legacy_split_data_not_promotion_eligible")
        summary["evidence_failures"] = list(dict.fromkeys(failures))
        summary["evidence_eligible"] = False
    _write_json(out_dir / "replay_summary.json", summary)
    _write_jsonl(
        out_dir / "activation_trace.jsonl",
        (trace.as_dict() for trace in runner.activation_traces),
    )
    _write_jsonl(
        out_dir / "policy_decisions.jsonl",
        (trace.as_dict() for trace in runner.policy_traces),
    )
    _write_jsonl(
        out_dir / "execution_audit.jsonl",
        (row.as_dict() for row in runner.execution_audit),
    )
    _write_jsonl(
        out_dir / "closed_trades.jsonl",
        (row.as_dict() for row in runner.trade_outcomes),
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exact research-only CarryFlow policy replay on Bitget data."
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument(
        "--evidence-tape",
        help="Unified hash-chained Bitget market+derivatives JSONL tape.",
    )
    parser.add_argument(
        "--warmup-seed",
        help=(
            "Immutable closed-OHLCV seed created before the first tape bar. "
            "It warms indicators but is never counted as trade evidence."
        ),
    )
    parser.add_argument(
        "--allow-legacy-split-input",
        action="store_true",
        help="Diagnostic only; output is never promotion-eligible.",
    )
    parser.add_argument(
        "--derivatives-context-csv",
        help=(
            "Point-in-time funding/OI/long-short CSV. Without it CarryFlow "
            "opens remain hard-blocked."
        ),
    )
    parser.add_argument("--out-dir")
    parser.add_argument("--policy-id")
    parser.add_argument(
        "--symbols",
        help=(
            "Optional exact tape-contract assertion. Authoritative replay "
            "derives the symbol set from the evidence tape."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=carryflow_profile_ids(),
        default="screened_short_v1",
        help="One immutable policy profile; scalar actor overrides are disabled.",
    )
    parser.add_argument(
        "--stride-minutes",
        type=int,
        help=(
            "Legacy diagnostic input cadence. Authoritative replay derives "
            "it from the evidence tape; an explicit value is only an assertion."
        ),
    )
    parser.add_argument("--max-snapshots", type=int)
    parser.add_argument("--oos-fraction", type=float, default=0.35)
    parser.add_argument("--use-live-regime-detector", action="store_true")
    parser.add_argument("--initial-capital-usd", type=float, default=1000.0)
    parser.add_argument("--capital-fraction", type=float, default=0.01)
    parser.add_argument("--max-notional-usd", type=float, default=10.0)
    parser.add_argument("--max-daily-loss-usd", type=float, default=5.0)
    parser.add_argument("--exchange-min-notional-usd", type=float, default=5.0)
    parser.add_argument("--round-trip-fee-bps", type=float, default=8.0)
    parser.add_argument("--slippage-bps", type=float, default=4.0)
    parser.add_argument("--safety-buffer-bps", type=float, default=4.0)
    parser.add_argument("--assumed-spread-bps", type=float, default=2.0)
    parser.add_argument(
        "--no-flatten-end",
        action="store_true",
        help=(
            "Do not manufacture a close at the final evidence bar; report "
            "remaining positions as right-censored."
        ),
    )
    args = parser.parse_args(argv)

    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["parity_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
