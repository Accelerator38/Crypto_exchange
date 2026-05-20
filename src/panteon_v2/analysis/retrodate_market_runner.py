"""Run Panteon v2 live-like benchmarks on Retrodate OHLCV CSV data."""

from __future__ import annotations

import argparse
import csv
import json
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Iterable, Optional, Sequence

from ..app.agent_bootstrap import register_all_v1_agents
from ..app.bootstrap import LiveExecutionConfig, build_production_pipeline
from ..app.main_loop import StepResult, main_loop
from ..app.output_writer import OutputWriter, OutputWriterConfig
from ..app.shadow_tournament import ProductionShadowTournament
from ..attribution import CandidateRejected, CandidateScored, EventLog, ShadowActorUpdated
from ..domain.types import MarketSnapshot, Regime
from ..execution import FakeExchange
from ..selection import AgentRegistry, StrategistConfig
from ..shadow.feed import ReplayFeed
from .soft_allocator import (
    shadow_pnl_events_from_shadow_updates,
    simulate_perfect_monthly_panteon,
    simulate_soft_allocator_policies,
    write_shadow_pnl_events,
    write_perfect_panteon_report,
    write_soft_allocator_report,
)
from .retrodate_validator import (
    RetrodateDirReport,
    RetrodateFileReport,
    RetrodateValidationError,
    validate_retrodate_dir,
)
from .allocation_diagnostics import analyze_trading_log, write_allocation_diagnostics


DEFAULT_YEARS = (2022, 2023, 2024, 2025, 2026)
DEFAULT_FIXED_AGENT_PLAYER_SETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Fixed_BearDefense",
        ("LiveCrashHunter", "FundingArb", "BearReliefFadeAgent", "CrashPanicShortAgent"),
    ),
    (
        "Fixed_BullBreakout",
        ("VolBreakoutHunter", "MomentumScalper", "RichardDennis", "LiveOIBreakout"),
    ),
    (
        "Fixed_NeutralValidator",
        (
            "ResearchValidatorAgent",
            "NeutralLiquiditySweep",
            "NeutralRangeScalper",
            "AnchorFlowMomentum",
        ),
    ),
    (
        "Fixed_RegimePullback",
        ("LiveRegimePullback", "LiveMeanRev", "LiveVolCompress"),
    ),
    (
        "Fixed_RotationFlow",
        ("BullRotationAgent", "BearReliefFadeAgent", "AnchorFlowMomentum", "LiveTrendFollow"),
    ),
    (
        "Fixed_CoreDiversified",
        (
            "LiveCrashHunter",
            "VolBreakoutHunter",
            "ResearchValidatorAgent",
            "NeutralLiquiditySweep",
            "AnchorFlowMomentum",
        ),
    ),
    (
        "Fixed_TrendRecovery",
        ("LiveTrendFollow", "RichardDennis", "LiveRegimePullback"),
    ),
)
DEFAULT_PROBATION_LOSS_LABEL_PREFIXES: tuple[str, ...] = (
    "Solo_",
    "Fixed_",
    "Antonius_",
    "Optimal_",
)
DEFAULT_RETRO_HARD_POLICY_DENY_LABELS: tuple[str, ...] = (
    "Solo_PlayerFunding",
    "Solo_LiveAfterShock",
    "Solo_CarryFlowAgentV2",
    "Solo_LiveRegimePullback",
    "Solo_LiveTrendFollow",
    "DefaultEnsemble",
    "DefensiveResearch",
    "TrendResearch",
    "NeutralEdgeResearch",
    "MeanRevResearch",
    "Fixed_BearDefense",
    "Fixed_BullBreakout",
    "Fixed_CoreDiversified",
    "Fixed_NeutralValidator",
    "Fixed_RegimePullback",
    "Fixed_RotationFlow",
    "Fixed_TrendRecovery",
    "Perfect_CrashSwitch",
    "Perfect_GeneticsBearCrash",
    "Perfect_MeanRev",
    "Perfect_NeutralValidator",
    "Perfect_OIBreakout",
    "Solo_BullRotationAgent",
    "Solo_GeneticsCore",
    "Solo_LiveMeanRev",
    "Solo_LiveOIBreakout",
    "Solo_RichardDennis",
)


@dataclass(frozen=True)
class RetrodateMarketConfig:
    """Configuration for a Retrodate market replay benchmark."""

    data_dir: Path | str = Path("Retrodate")
    results_root: Path | str = Path("Results") / "RetrodateMarket"
    years: tuple[int, ...] = DEFAULT_YEARS
    timeframe: str = "1m"
    stride_minutes: int = 60
    initial_capital: float = 1000.0
    include_optional_agents: bool = False
    optional_agent_labels: tuple[str, ...] = ()
    invalid_policy: str = "exclude"
    write_every_bars: int = 2000
    full_snapshot_every: int = 2000
    recompute_quarantine_every: int = 24
    progress_every_bars: int = 1000
    max_bars: Optional[int] = None
    solo_agent_candidate_limit: int = 3
    fixed_agent_players_enabled: bool = False
    fixed_agent_player_sets: tuple[tuple[str, tuple[str, ...]], ...] = ()
    actionable_fallback_enabled: bool = False
    actionable_fallback_min_score: Optional[float] = None
    actionable_fallback_require_has_data: bool = False
    current_actionable_candidate_layer_enabled: bool = False
    real_promotion_gate_enabled: bool = False
    real_promotion_min_closed_trades: int = 20
    real_promotion_min_pnl_pct: float = 0.0
    real_promotion_max_drawdown_pct: float = 25.0
    real_promotion_loss_budget_pct: float = -1.0
    real_promotion_probation_min_score: float = 0.0
    use_v3_rolling_score: bool = False
    use_v3_shadow_rolling_score: bool = False
    use_v3_soft_shadow_score: bool = False
    use_v3_executable_soft_top1_score: bool = False
    use_v3_executable_soft_confirmed_score: bool = False
    use_v3_entry_causal_score: bool = False
    v3_shadow_position_gate_enabled: bool = True
    v3_shadow_flat_handoff_enabled: bool = False
    v3_shadow_fresh_handoff_enabled: bool = False
    v3_shadow_fresh_handoff_max_age_bars: int = 1
    v3_shadow_rolling_window_bars: int = 24
    v3_shadow_rolling_min_closed_trades: int = 50
    v3_entry_causal_min_filled: int = 3
    v3_entry_causal_actionability_weight: float = 1.0
    v3_real_loss_rescue_enabled: bool = False
    v3_real_loss_rescue_min_virtual_pnl_pct: float = 10.0
    v3_real_loss_rescue_max_virtual_dd_pct: float = 50.0
    v3_real_loss_rescue_min_actionable_share: float = 0.05
    v3_real_loss_rescue_min_recent_filled: int = 1
    v3_real_loss_rescue_max_real_loss_pct: float = -3.0
    v3_real_loss_rescue_allow_genetics: bool = False
    v3_probation_shadow_rescue_enabled: bool = False
    v3_probation_shadow_rescue_min_virtual_pnl_pct: float = 10.0
    v3_probation_shadow_rescue_max_virtual_dd_pct: float = 50.0
    v3_probation_shadow_rescue_min_actionable_share: float = 0.05
    v3_probation_shadow_rescue_min_recent_filled: int = 1
    v3_probation_shadow_rescue_min_recent_pnl_usd: float = 0.0
    v3_probation_shadow_rescue_allow_genetics: bool = False
    v3_persistent_loss_kill_min_closed_trades: int = 0
    v3_persistent_loss_kill_pnl_pct: float = -2.0
    v3_persistent_loss_kill_win_rate_pct: float = 0.0
    v3_persistent_loss_requires_virtual_weakness: bool = True
    v3_persistent_loss_virtual_max_pnl_pct: float = 0.0
    v3_persistent_loss_virtual_min_dd_pct: float = 25.0
    v3_probation_loss_kill_min_closed_trades: int = 0
    v3_probation_loss_kill_pnl_pct: float = -0.15
    v3_probation_loss_kill_win_rate_pct: float = 50.0
    v3_probation_loss_kill_label_prefixes: tuple[str, ...] = (
        DEFAULT_PROBATION_LOSS_LABEL_PREFIXES
    )
    genetics_probation_execution_enabled: bool = False
    genetics_probation_risk_mult: float = 0.25
    genetics_probation_max_real_trades: int = 20
    genetics_probation_require_shadow_confirmation: bool = True
    v3_realized_profit_lock_min_closed_trades: int = 0
    v3_realized_profit_lock_min_peak_pnl_pct: float = 0.75
    v3_realized_profit_lock_max_giveback_pct: float = 0.55
    v3_realized_profit_lock_floor_pnl_pct: float = 0.25
    v3_panteon_equity_guard_enabled: bool = False
    v3_panteon_equity_guard_min_peak_pnl_pct: float = 2.0
    v3_panteon_equity_guard_max_giveback_pct: float = 1.0
    v3_panteon_equity_guard_floor_pnl_pct: float = 2.0
    v3_panteon_equity_guard_cooldown_bars: int = 720
    hard_policy_deny_labels: tuple[str, ...] = DEFAULT_RETRO_HARD_POLICY_DENY_LABELS

    def __post_init__(self) -> None:
        if self.stride_minutes <= 0:
            raise ValueError("stride_minutes must be > 0")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")
        if self.solo_agent_candidate_limit < 1:
            raise ValueError("solo_agent_candidate_limit must be >= 1")
        if self.invalid_policy not in {"exclude", "fail"}:
            raise ValueError("invalid_policy must be 'exclude' or 'fail'")
        object.__setattr__(
            self,
            "optional_agent_labels",
            tuple(str(label) for label in self.optional_agent_labels),
        )
        object.__setattr__(
            self,
            "hard_policy_deny_labels",
            tuple(str(label) for label in self.hard_policy_deny_labels),
        )
        fixed_sets = _normalize_fixed_agent_player_sets(self.fixed_agent_player_sets)
        if self.fixed_agent_players_enabled and not fixed_sets:
            fixed_sets = DEFAULT_FIXED_AGENT_PLAYER_SETS
        object.__setattr__(self, "fixed_agent_player_sets", fixed_sets)
        if self.real_promotion_min_closed_trades < 0:
            raise ValueError("real_promotion_min_closed_trades must be >= 0")
        if self.real_promotion_max_drawdown_pct < 0:
            raise ValueError("real_promotion_max_drawdown_pct must be >= 0")
        if (
            self.actionable_fallback_min_score is not None
            and not isinstance(self.actionable_fallback_min_score, (int, float))
        ):
            raise ValueError("actionable_fallback_min_score must be numeric or None")
        if self.v3_persistent_loss_kill_min_closed_trades < 0:
            raise ValueError("v3_persistent_loss_kill_min_closed_trades must be >= 0")
        if self.v3_probation_loss_kill_min_closed_trades < 0:
            raise ValueError("v3_probation_loss_kill_min_closed_trades must be >= 0")
        if not (0.0 < self.genetics_probation_risk_mult <= 1.0):
            raise ValueError("genetics_probation_risk_mult must be in (0, 1]")
        if self.genetics_probation_max_real_trades < 0:
            raise ValueError("genetics_probation_max_real_trades must be >= 0")
        if self.v3_realized_profit_lock_min_closed_trades < 0:
            raise ValueError("v3_realized_profit_lock_min_closed_trades must be >= 0")
        if self.v3_shadow_rolling_window_bars < 1:
            raise ValueError("v3_shadow_rolling_window_bars must be >= 1")
        if self.v3_shadow_rolling_min_closed_trades < 0:
            raise ValueError("v3_shadow_rolling_min_closed_trades must be >= 0")
        if self.v3_entry_causal_min_filled < 0:
            raise ValueError("v3_entry_causal_min_filled must be >= 0")
        if self.v3_entry_causal_actionability_weight < 0:
            raise ValueError("v3_entry_causal_actionability_weight must be >= 0")
        if self.v3_shadow_fresh_handoff_max_age_bars < 0:
            raise ValueError("v3_shadow_fresh_handoff_max_age_bars must be >= 0")
        if not 0.0 <= self.v3_real_loss_rescue_min_actionable_share <= 1.0:
            raise ValueError("v3_real_loss_rescue_min_actionable_share must be in [0, 1]")
        if self.v3_real_loss_rescue_min_recent_filled < 0:
            raise ValueError("v3_real_loss_rescue_min_recent_filled must be >= 0")
        if self.v3_real_loss_rescue_max_virtual_dd_pct < 0:
            raise ValueError("v3_real_loss_rescue_max_virtual_dd_pct must be >= 0")
        if not 0.0 <= self.v3_probation_shadow_rescue_min_actionable_share <= 1.0:
            raise ValueError("v3_probation_shadow_rescue_min_actionable_share must be in [0, 1]")
        if self.v3_probation_shadow_rescue_min_recent_filled < 0:
            raise ValueError("v3_probation_shadow_rescue_min_recent_filled must be >= 0")
        if self.v3_probation_shadow_rescue_max_virtual_dd_pct < 0:
            raise ValueError("v3_probation_shadow_rescue_max_virtual_dd_pct must be >= 0")
        if not 0.0 <= self.v3_persistent_loss_kill_win_rate_pct <= 100.0:
            raise ValueError("v3_persistent_loss_kill_win_rate_pct must be in [0, 100]")
        if self.v3_persistent_loss_virtual_min_dd_pct < 0:
            raise ValueError("v3_persistent_loss_virtual_min_dd_pct must be >= 0")
        if not 0.0 <= self.v3_probation_loss_kill_win_rate_pct <= 100.0:
            raise ValueError("v3_probation_loss_kill_win_rate_pct must be in [0, 100]")
        if self.v3_realized_profit_lock_min_peak_pnl_pct < 0:
            raise ValueError("v3_realized_profit_lock_min_peak_pnl_pct must be >= 0")
        if self.v3_realized_profit_lock_max_giveback_pct < 0:
            raise ValueError("v3_realized_profit_lock_max_giveback_pct must be >= 0")
        if self.v3_panteon_equity_guard_min_peak_pnl_pct < 0:
            raise ValueError("v3_panteon_equity_guard_min_peak_pnl_pct must be >= 0")
        if self.v3_panteon_equity_guard_max_giveback_pct < 0:
            raise ValueError("v3_panteon_equity_guard_max_giveback_pct must be >= 0")
        if self.v3_panteon_equity_guard_cooldown_bars < 0:
            raise ValueError("v3_panteon_equity_guard_cooldown_bars must be >= 0")
        object.__setattr__(
            self,
            "v3_probation_loss_kill_label_prefixes",
            tuple(
                str(prefix)
                for prefix in self.v3_probation_loss_kill_label_prefixes
                if str(prefix)
            ),
        )


@dataclass(frozen=True)
class RetrodateFileSelection:
    """Validated Retrodate files selected for a benchmark run."""

    report: RetrodateDirReport
    valid_reports: tuple[RetrodateFileReport, ...]
    excluded_files: tuple[RetrodateFileReport, ...]
    missing_years: tuple[int, ...]

    @property
    def valid_files(self) -> tuple[Path, ...]:
        return tuple(item.path for item in self.valid_reports)

    @property
    def executed_years(self) -> tuple[int, ...]:
        return tuple(
            int(item.expected_year)
            for item in self.valid_reports
            if item.expected_year is not None
        )


@dataclass
class RetrodateSnapshotState:
    """Mutable state preserved while loading multiple yearly files."""

    next_bar: int = 1
    btc_closes: Deque[float] = field(default_factory=lambda: deque(maxlen=24))


@dataclass(frozen=True)
class RetrodateRunSummary:
    """Summary returned after a benchmark run finishes."""

    output_dir: Path
    report_path: Path
    summary_path: Path
    requested_years: tuple[int, ...]
    executed_years: tuple[int, ...]
    excluded_files: tuple[str, ...]
    missing_years: tuple[int, ...]
    bars_processed: int
    registered_agents: tuple[str, ...]
    player_profile_count: int
    first_timestamp: str
    last_timestamp: str
    stride_minutes: int
    max_bars: Optional[int]
    step_errors: tuple[str, ...]


def select_retrodate_files(config: RetrodateMarketConfig) -> RetrodateFileSelection:
    """Validate and select requested Retrodate CSV files."""
    root = Path(config.data_dir)
    report = validate_retrodate_dir(root)
    if report.issues:
        raise RetrodateValidationError(_format_directory_issues(report))

    requested = tuple(int(year) for year in config.years)
    selected = [
        item
        for item in report.files
        if item.expected_year in requested and item.timeframe == config.timeframe
    ]
    excluded = tuple(item for item in selected if not item.is_valid)
    if excluded and config.invalid_policy == "fail":
        raise RetrodateValidationError(_format_selection_errors(excluded))

    valid_reports = tuple(
        sorted(
            (item for item in selected if item.is_valid),
            key=lambda item: (item.expected_year or 0, item.path.name),
        )
    )
    seen_years = {item.expected_year for item in valid_reports}
    missing_years = tuple(year for year in requested if year not in seen_years)
    if not valid_reports:
        raise RetrodateValidationError(
            "No valid Retrodate files selected for requested years: "
            + ", ".join(str(year) for year in requested)
        )
    return RetrodateFileSelection(
        report=report,
        valid_reports=valid_reports,
        excluded_files=excluded,
        missing_years=missing_years,
    )


def load_retrodate_year_snapshots(
    csv_path: str | Path,
    *,
    stride_minutes: int,
    state: RetrodateSnapshotState,
) -> list[MarketSnapshot]:
    """Load one yearly CSV into chronological multi-symbol market snapshots."""
    if stride_minutes <= 0:
        raise ValueError("stride_minutes must be > 0")
    stride_ms = int(stride_minutes) * 60 * 1000
    prices_by_ts: dict[int, dict[str, float]] = {}
    volumes_by_ts: dict[int, dict[str, float]] = {}

    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                timestamp = int(str(row.get("timestamp") or "").strip())
            except ValueError:
                continue
            if timestamp % stride_ms != 0:
                continue
            symbol = str(row.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            close = _coerce_float(row.get("close"))
            if close <= 0:
                continue
            prices_by_ts.setdefault(timestamp, {})[symbol] = close
            volumes_by_ts.setdefault(timestamp, {})[symbol] = max(0.0, _coerce_float(row.get("volume")))

    snapshots: list[MarketSnapshot] = []
    for timestamp in sorted(prices_by_ts):
        prices = dict(sorted(prices_by_ts[timestamp].items()))
        volumes = {
            symbol: volumes_by_ts.get(timestamp, {}).get(symbol, 0.0)
            for symbol in prices
        }
        regime, confidence = _classify_regime(prices, state)
        dt = datetime.fromtimestamp(timestamp / 1000.0, timezone.utc)
        snapshots.append(
            MarketSnapshot(
                bar=state.next_bar,
                timestamp=dt,
                regime=regime,
                regime_confidence=confidence,
                prices=prices,
                volumes=volumes,
                month=dt.month,
            )
        )
        state.next_bar += 1
    return snapshots


def run_retrodate_market_benchmark(config: RetrodateMarketConfig) -> RetrodateRunSummary:
    """Run the selected Retrodate files through the live-like Panteon pipeline."""
    selection = select_retrodate_files(config)
    output_dir = _new_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=False)

    registry = AgentRegistry()
    registered_agents = tuple(
        register_all_v1_agents(
            registry,
            include_optional=config.include_optional_agents,
            optional_agent_labels=(
                config.optional_agent_labels
                if config.optional_agent_labels
                else None
            ),
            skip_on_error=True,
        )
    )
    exchange = FakeExchange(name="RETRODATE_MARKET")
    strategist_config = _build_strategist_config(config)
    pipeline = build_production_pipeline(
        registry=registry,
        exchange=exchange,
        initial_capital=config.initial_capital,
        strategist_config=strategist_config,
        live_execution_config=_build_live_execution_config(config),
    )
    pipeline.mode = "retrodate_market"
    pipeline.timeframe = f"{config.stride_minutes}m-from-{config.timeframe}"
    pipeline.session_id = output_dir.name
    pipeline.run_id = output_dir.name
    pipeline.actionable_fallback_enabled = bool(config.actionable_fallback_enabled)
    pipeline.actionable_fallback_min_score = config.actionable_fallback_min_score
    pipeline.actionable_fallback_require_has_data = bool(
        config.actionable_fallback_require_has_data
    )
    pipeline.current_actionable_candidate_layer_enabled = bool(
        config.current_actionable_candidate_layer_enabled
        or config.use_v3_executable_soft_confirmed_score
    )
    pipeline.solo_agent_candidate_limit = int(config.solo_agent_candidate_limit)
    pipeline.fixed_agent_player_sets = (
        tuple(config.fixed_agent_player_sets)
        if config.fixed_agent_players_enabled
        else ()
    )
    shadow_event_log = EventLog()
    pipeline.shadow_tournament = ProductionShadowTournament(
        registry=pipeline.registry,
        perf=pipeline.virtual_perf,
        risk_config=pipeline.risk_config,
        event_log=shadow_event_log,
    )

    writer = OutputWriter(
        pipeline,
        OutputWriterConfig(
            output_dir=str(output_dir),
            write_every_bars=max(1, int(config.write_every_bars)),
            full_snapshot_every=max(1, int(config.full_snapshot_every)),
            latest_dir=str(Path(config.results_root).resolve()),
        ),
    )

    state = RetrodateSnapshotState()
    bars_processed = 0
    first_timestamp = ""
    last_timestamp = ""
    step_errors: list[str] = []
    processed_years: list[int] = []

    try:
        for file_report in selection.valid_reports:
            remaining = None
            if config.max_bars is not None:
                remaining = max(0, int(config.max_bars) - bars_processed)
                if remaining <= 0:
                    break
            snapshots = load_retrodate_year_snapshots(
                file_report.path,
                stride_minutes=config.stride_minutes,
                state=state,
            )
            if remaining is not None:
                snapshots = snapshots[:remaining]
            if not snapshots:
                continue
            if not first_timestamp:
                first_timestamp = snapshots[0].timestamp.isoformat()
            last_timestamp = snapshots[-1].timestamp.isoformat()
            processed_years.append(int(file_report.expected_year or 0))
            feed = ReplayFeed(snapshots=snapshots)
            steps = main_loop(
                pipeline,
                feed,
                recompute_quarantine_every=max(1, int(config.recompute_quarantine_every)),
                on_step=_make_on_step(writer, config),
            )
            bars_processed += len(steps)
            step_errors.extend(str(step.error) for step in steps if step.error)
    finally:
        writer.close()

    try:
        trading_log = output_dir / "trading.log"
        if trading_log.exists():
            write_allocation_diagnostics(
                output_dir,
                analyze_trading_log(trading_log),
            )
    except Exception as exc:
        step_errors.append(
            f"allocation_diagnostics_failed: {type(exc).__name__}: {exc}"
        )

    try:
        write_candidate_diagnostics(
            output_dir,
            candidate_events=tuple(
                pipeline.event_log.query(event_types=[CandidateScored])
            ),
            rejection_events=tuple(
                pipeline.event_log.query(event_types=[CandidateRejected])
            ),
        )
    except Exception as exc:
        step_errors.append(
            f"candidate_diagnostics_failed: {type(exc).__name__}: {exc}"
        )

    try:
        shadow_updates = tuple(shadow_event_log.query(event_types=[ShadowActorUpdated]))
        shadow_pnl_events = shadow_pnl_events_from_shadow_updates(shadow_updates)
        shadow_agent_pnl_events = shadow_pnl_events_from_shadow_updates(
            shadow_updates,
            actor_type="agent",
        )
        soft_report = simulate_soft_allocator_policies(
            shadow_pnl_events,
            initial_capital=config.initial_capital,
        )
        perfect_report = simulate_perfect_monthly_panteon(
            shadow_pnl_events,
            initial_capital=config.initial_capital,
        )
        write_soft_allocator_report(output_dir, soft_report)
        write_perfect_panteon_report(output_dir, perfect_report)
        write_shadow_pnl_events(output_dir, shadow_pnl_events)
        write_shadow_pnl_events(
            output_dir,
            shadow_agent_pnl_events,
            filename="shadow_agent_pnl_events.jsonl",
        )
        write_oracle_mismatch_report(
            output_dir,
            output_dir / "trading.log",
            shadow_updates=shadow_updates,
            candidate_rejections=tuple(
                pipeline.event_log.query(event_types=[CandidateRejected])
            ),
        )
    except Exception as exc:
        step_errors.append(f"soft_allocator_report_failed: {type(exc).__name__}: {exc}")

    report_path = output_dir / "analysis_report.md"
    summary_path = output_dir / "run_summary.json"
    summary = RetrodateRunSummary(
        output_dir=output_dir,
        report_path=report_path,
        summary_path=summary_path,
        requested_years=tuple(config.years),
        executed_years=tuple(year for year in processed_years if year),
        excluded_files=tuple(item.path.name for item in selection.excluded_files),
        missing_years=selection.missing_years,
        bars_processed=bars_processed,
        registered_agents=registered_agents,
        player_profile_count=len(pipeline.profiles),
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        stride_minutes=config.stride_minutes,
        max_bars=config.max_bars,
        step_errors=tuple(step_errors[:50]),
    )
    _write_run_summary(summary_path, summary, selection, config)
    _write_analysis_report(report_path, summary, selection)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = _parse_cli_config(argv)
    summary = run_retrodate_market_benchmark(config)
    print(f"output_dir={summary.output_dir}", flush=True)
    print(f"analysis_report={summary.report_path}", flush=True)
    print(f"run_summary={summary.summary_path}", flush=True)
    return 0


def _parse_cli_config(argv: Optional[Sequence[str]] = None) -> RetrodateMarketConfig:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    return RetrodateMarketConfig(
        data_dir=Path(args.data_dir),
        results_root=Path(args.results_root),
        years=_parse_years(args.years),
        timeframe=args.timeframe,
        stride_minutes=args.stride_minutes,
        initial_capital=args.initial_capital,
        include_optional_agents=args.include_optional_agents,
        optional_agent_labels=_parse_optional_agent_labels(args.optional_agent_labels),
        invalid_policy=args.invalid_policy,
        write_every_bars=args.write_every_bars,
        full_snapshot_every=args.full_snapshot_every,
        recompute_quarantine_every=args.recompute_quarantine_every,
        progress_every_bars=args.progress_every_bars,
        max_bars=args.max_bars,
        solo_agent_candidate_limit=args.solo_agent_candidate_limit,
        fixed_agent_players_enabled=(
            args.enable_fixed_agent_players or bool(args.fixed_agent_player_set)
        ),
        fixed_agent_player_sets=_parse_fixed_agent_player_sets(args.fixed_agent_player_set),
        actionable_fallback_enabled=args.enable_actionable_fallback,
        actionable_fallback_min_score=args.actionable_fallback_min_score,
        actionable_fallback_require_has_data=args.actionable_fallback_require_has_data,
        current_actionable_candidate_layer_enabled=(
            args.enable_current_actionable_candidate_layer
        ),
        real_promotion_gate_enabled=args.real_promotion_gate,
        real_promotion_min_closed_trades=args.real_promotion_min_closed_trades,
        real_promotion_min_pnl_pct=args.real_promotion_min_pnl_pct,
        real_promotion_max_drawdown_pct=args.real_promotion_max_dd_pct,
        real_promotion_loss_budget_pct=args.real_promotion_loss_budget_pct,
        real_promotion_probation_min_score=args.real_promotion_probation_min_score,
        use_v3_rolling_score=args.use_v3_rolling_score,
        use_v3_shadow_rolling_score=args.use_v3_shadow_rolling_score,
        use_v3_soft_shadow_score=args.use_v3_soft_shadow_score,
        use_v3_executable_soft_top1_score=(
            args.use_v3_executable_soft_top1_score
        ),
        use_v3_executable_soft_confirmed_score=(
            args.use_v3_executable_soft_confirmed_score
        ),
        use_v3_entry_causal_score=args.use_v3_entry_causal_score,
        v3_shadow_position_gate_enabled=not args.disable_v3_shadow_position_gate,
        v3_shadow_flat_handoff_enabled=args.enable_v3_shadow_flat_handoff,
        v3_shadow_fresh_handoff_enabled=args.enable_v3_shadow_fresh_handoff,
        v3_shadow_fresh_handoff_max_age_bars=args.v3_shadow_fresh_handoff_max_age_bars,
        v3_shadow_rolling_window_bars=args.v3_shadow_rolling_window_bars,
        v3_shadow_rolling_min_closed_trades=args.v3_shadow_rolling_min_closed_trades,
        v3_entry_causal_min_filled=args.v3_entry_causal_min_filled,
        v3_entry_causal_actionability_weight=args.v3_entry_causal_actionability_weight,
        v3_real_loss_rescue_enabled=args.enable_v3_real_loss_rescue,
        v3_real_loss_rescue_min_virtual_pnl_pct=(
            args.v3_real_loss_rescue_min_virtual_pnl_pct
        ),
        v3_real_loss_rescue_max_virtual_dd_pct=(
            args.v3_real_loss_rescue_max_virtual_dd_pct
        ),
        v3_real_loss_rescue_min_actionable_share=(
            args.v3_real_loss_rescue_min_actionable_share
        ),
        v3_real_loss_rescue_min_recent_filled=(
            args.v3_real_loss_rescue_min_recent_filled
        ),
        v3_real_loss_rescue_max_real_loss_pct=(
            args.v3_real_loss_rescue_max_real_loss_pct
        ),
        v3_real_loss_rescue_allow_genetics=args.allow_genetics_real_loss_rescue,
        v3_probation_shadow_rescue_enabled=args.enable_v3_probation_shadow_rescue,
        v3_probation_shadow_rescue_min_virtual_pnl_pct=(
            args.v3_probation_shadow_rescue_min_virtual_pnl_pct
        ),
        v3_probation_shadow_rescue_max_virtual_dd_pct=(
            args.v3_probation_shadow_rescue_max_virtual_dd_pct
        ),
        v3_probation_shadow_rescue_min_actionable_share=(
            args.v3_probation_shadow_rescue_min_actionable_share
        ),
        v3_probation_shadow_rescue_min_recent_filled=(
            args.v3_probation_shadow_rescue_min_recent_filled
        ),
        v3_probation_shadow_rescue_min_recent_pnl_usd=(
            args.v3_probation_shadow_rescue_min_recent_pnl_usd
        ),
        v3_probation_shadow_rescue_allow_genetics=(
            args.allow_genetics_probation_shadow_rescue
        ),
        v3_persistent_loss_kill_min_closed_trades=(
            args.v3_persistent_loss_kill_min_closed_trades
        ),
        v3_persistent_loss_kill_pnl_pct=args.v3_persistent_loss_kill_pnl_pct,
        v3_persistent_loss_kill_win_rate_pct=args.v3_persistent_loss_kill_win_rate_pct,
        v3_persistent_loss_requires_virtual_weakness=(
            not args.v3_persistent_loss_ignore_virtual_quality
        ),
        v3_persistent_loss_virtual_max_pnl_pct=(
            args.v3_persistent_loss_virtual_max_pnl_pct
        ),
        v3_persistent_loss_virtual_min_dd_pct=(
            args.v3_persistent_loss_virtual_min_dd_pct
        ),
        v3_probation_loss_kill_min_closed_trades=(
            args.v3_probation_loss_kill_min_closed_trades
        ),
        v3_probation_loss_kill_pnl_pct=args.v3_probation_loss_kill_pnl_pct,
        v3_probation_loss_kill_win_rate_pct=(
            args.v3_probation_loss_kill_win_rate_pct
        ),
        v3_probation_loss_kill_label_prefixes=(
            _parse_probation_loss_label_prefixes(args)
        ),
        genetics_probation_execution_enabled=(
            args.enable_genetics_probation_execution
        ),
        genetics_probation_risk_mult=args.genetics_probation_risk_mult,
        genetics_probation_max_real_trades=args.genetics_probation_max_real_trades,
        genetics_probation_require_shadow_confirmation=(
            not args.disable_genetics_probation_shadow_confirmation
        ),
        v3_realized_profit_lock_min_closed_trades=(
            args.v3_realized_profit_lock_min_closed_trades
        ),
        v3_realized_profit_lock_min_peak_pnl_pct=(
            args.v3_realized_profit_lock_min_peak_pnl_pct
        ),
        v3_realized_profit_lock_max_giveback_pct=(
            args.v3_realized_profit_lock_max_giveback_pct
        ),
        v3_realized_profit_lock_floor_pnl_pct=(
            args.v3_realized_profit_lock_floor_pnl_pct
        ),
        v3_panteon_equity_guard_enabled=args.enable_v3_panteon_equity_guard,
        v3_panteon_equity_guard_min_peak_pnl_pct=(
            args.v3_panteon_equity_guard_min_peak_pnl_pct
        ),
        v3_panteon_equity_guard_max_giveback_pct=(
            args.v3_panteon_equity_guard_max_giveback_pct
        ),
        v3_panteon_equity_guard_floor_pnl_pct=(
            args.v3_panteon_equity_guard_floor_pnl_pct
        ),
        v3_panteon_equity_guard_cooldown_bars=(
            args.v3_panteon_equity_guard_cooldown_bars
        ),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="Retrodate")
    parser.add_argument("--results-root", default=str(Path("Results") / "RetrodateMarket"))
    parser.add_argument("--years", default=",".join(str(year) for year in DEFAULT_YEARS))
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--stride-minutes", type=int, default=60)
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--include-optional-agents", action="store_true")
    parser.add_argument("--optional-agent-labels", default="")
    parser.add_argument("--invalid-policy", choices=("exclude", "fail"), default="exclude")
    parser.add_argument("--write-every-bars", type=int, default=2000)
    parser.add_argument("--full-snapshot-every", type=int, default=2000)
    parser.add_argument("--recompute-quarantine-every", type=int, default=24)
    parser.add_argument("--progress-every-bars", type=int, default=1000)
    parser.add_argument("--max-bars", type=int)
    parser.add_argument("--solo-agent-candidate-limit", type=int, default=3)
    parser.add_argument("--enable-fixed-agent-players", action="store_true")
    parser.add_argument(
        "--fixed-agent-player-set",
        action="append",
        default=[],
        help="Add fixed player as Label=AgentA,AgentB. Can be repeated.",
    )
    parser.add_argument("--enable-actionable-fallback", action="store_true")
    parser.add_argument("--actionable-fallback-min-score", type=float)
    parser.add_argument("--actionable-fallback-require-has-data", action="store_true")
    parser.add_argument("--enable-current-actionable-candidate-layer", action="store_true")
    parser.add_argument("--real-promotion-gate", action="store_true")
    parser.add_argument("--real-promotion-min-closed-trades", type=int, default=20)
    parser.add_argument("--real-promotion-min-pnl-pct", type=float, default=0.0)
    parser.add_argument("--real-promotion-max-dd-pct", type=float, default=25.0)
    parser.add_argument("--real-promotion-loss-budget-pct", type=float, default=-1.0)
    parser.add_argument("--real-promotion-probation-min-score", type=float, default=0.0)
    parser.add_argument("--use-v3-rolling-score", action="store_true")
    parser.add_argument("--use-v3-shadow-rolling-score", action="store_true")
    parser.add_argument("--use-v3-soft-shadow-score", action="store_true")
    parser.add_argument("--use-v3-executable-soft-top1-score", action="store_true")
    parser.add_argument("--use-v3-executable-soft-confirmed-score", action="store_true")
    parser.add_argument("--use-v3-entry-causal-score", action="store_true")
    parser.add_argument("--disable-v3-shadow-position-gate", action="store_true")
    parser.add_argument("--enable-v3-shadow-flat-handoff", action="store_true")
    parser.add_argument("--enable-v3-shadow-fresh-handoff", action="store_true")
    parser.add_argument("--v3-shadow-fresh-handoff-max-age-bars", type=int, default=1)
    parser.add_argument("--v3-shadow-rolling-window-bars", type=int, default=24)
    parser.add_argument("--v3-shadow-rolling-min-closed-trades", type=int, default=50)
    parser.add_argument("--v3-entry-causal-min-filled", type=int, default=3)
    parser.add_argument("--v3-entry-causal-actionability-weight", type=float, default=1.0)
    parser.add_argument("--enable-v3-real-loss-rescue", action="store_true")
    parser.add_argument("--v3-real-loss-rescue-min-virtual-pnl-pct", type=float, default=10.0)
    parser.add_argument("--v3-real-loss-rescue-max-virtual-dd-pct", type=float, default=50.0)
    parser.add_argument("--v3-real-loss-rescue-min-actionable-share", type=float, default=0.05)
    parser.add_argument("--v3-real-loss-rescue-min-recent-filled", type=int, default=1)
    parser.add_argument("--v3-real-loss-rescue-max-real-loss-pct", type=float, default=-3.0)
    parser.add_argument("--allow-genetics-real-loss-rescue", action="store_true")
    parser.add_argument("--enable-v3-probation-shadow-rescue", action="store_true")
    parser.add_argument("--v3-probation-shadow-rescue-min-virtual-pnl-pct", type=float, default=10.0)
    parser.add_argument("--v3-probation-shadow-rescue-max-virtual-dd-pct", type=float, default=50.0)
    parser.add_argument("--v3-probation-shadow-rescue-min-actionable-share", type=float, default=0.05)
    parser.add_argument("--v3-probation-shadow-rescue-min-recent-filled", type=int, default=1)
    parser.add_argument("--v3-probation-shadow-rescue-min-recent-pnl-usd", type=float, default=0.0)
    parser.add_argument("--allow-genetics-probation-shadow-rescue", action="store_true")
    parser.add_argument("--v3-persistent-loss-kill-min-closed-trades", type=int, default=0)
    parser.add_argument("--v3-persistent-loss-kill-pnl-pct", type=float, default=-2.0)
    parser.add_argument("--v3-persistent-loss-kill-win-rate-pct", type=float, default=0.0)
    parser.add_argument("--v3-persistent-loss-ignore-virtual-quality", action="store_true")
    parser.add_argument("--v3-persistent-loss-virtual-max-pnl-pct", type=float, default=0.0)
    parser.add_argument("--v3-persistent-loss-virtual-min-dd-pct", type=float, default=25.0)
    parser.add_argument("--v3-probation-loss-kill-min-closed-trades", type=int, default=0)
    parser.add_argument("--v3-probation-loss-kill-pnl-pct", type=float, default=-0.15)
    parser.add_argument("--v3-probation-loss-kill-win-rate-pct", type=float, default=50.0)
    parser.add_argument(
        "--v3-probation-loss-kill-label-prefix",
        action="append",
        default=None,
        help=(
            "Restrict probation loss-kill to labels with this prefix. "
            "Can be repeated. Defaults to Solo_, Fixed_, Antonius_, and Optimal_."
        ),
    )
    parser.add_argument(
        "--v3-probation-loss-kill-all-labels",
        action="store_true",
        help="Apply probation loss-kill to every candidate label.",
    )
    parser.add_argument("--enable-genetics-probation-execution", action="store_true")
    parser.add_argument("--genetics-probation-risk-mult", type=float, default=0.25)
    parser.add_argument("--genetics-probation-max-real-trades", type=int, default=20)
    parser.add_argument(
        "--disable-genetics-probation-shadow-confirmation",
        action="store_true",
    )
    parser.add_argument("--v3-realized-profit-lock-min-closed-trades", type=int, default=0)
    parser.add_argument("--v3-realized-profit-lock-min-peak-pnl-pct", type=float, default=0.75)
    parser.add_argument("--v3-realized-profit-lock-max-giveback-pct", type=float, default=0.55)
    parser.add_argument("--v3-realized-profit-lock-floor-pnl-pct", type=float, default=0.25)
    parser.add_argument("--enable-v3-panteon-equity-guard", action="store_true")
    parser.add_argument("--v3-panteon-equity-guard-min-peak-pnl-pct", type=float, default=2.0)
    parser.add_argument("--v3-panteon-equity-guard-max-giveback-pct", type=float, default=1.0)
    parser.add_argument("--v3-panteon-equity-guard-floor-pnl-pct", type=float, default=2.0)
    parser.add_argument("--v3-panteon-equity-guard-cooldown-bars", type=int, default=720)
    return parser


def _build_live_execution_config(config: RetrodateMarketConfig) -> LiveExecutionConfig:
    return LiveExecutionConfig(
        genetics_probation_execution_enabled=(
            config.genetics_probation_execution_enabled
        ),
        genetics_probation_risk_mult=config.genetics_probation_risk_mult,
        genetics_probation_max_real_trades=config.genetics_probation_max_real_trades,
        genetics_probation_require_shadow_confirmation=(
            config.genetics_probation_require_shadow_confirmation
        ),
    )


def _build_strategist_config(config: RetrodateMarketConfig) -> StrategistConfig:
    executable_soft_top1 = bool(config.use_v3_executable_soft_top1_score)
    executable_soft_confirmed = bool(config.use_v3_executable_soft_confirmed_score)
    executable_soft = executable_soft_top1 or executable_soft_confirmed
    rolling_window = (
        24 if executable_soft else config.v3_shadow_rolling_window_bars
    )
    min_closed_trades = (
        20 if executable_soft_top1
        else max(50, int(config.v3_shadow_rolling_min_closed_trades))
        if executable_soft_confirmed
        else config.v3_shadow_rolling_min_closed_trades
    )
    return StrategistConfig(
        hard_policy_deny_labels=config.hard_policy_deny_labels,
        real_promotion_gate_enabled=config.real_promotion_gate_enabled,
        real_promotion_min_closed_trades=config.real_promotion_min_closed_trades,
        real_promotion_min_pnl_pct=config.real_promotion_min_pnl_pct,
        real_promotion_max_drawdown_pct=config.real_promotion_max_drawdown_pct,
        real_promotion_loss_budget_pct=config.real_promotion_loss_budget_pct,
        real_promotion_probation_min_score=config.real_promotion_probation_min_score,
        use_v3_rolling_score=config.use_v3_rolling_score or executable_soft,
        use_v3_entry_causal_score=config.use_v3_entry_causal_score,
        use_v3_shadow_rolling_score=config.use_v3_shadow_rolling_score,
        use_v3_soft_shadow_score=config.use_v3_soft_shadow_score or executable_soft,
        v3_current_actionable_gate_enabled=(
            config.current_actionable_candidate_layer_enabled or executable_soft
        ),
        v3_shadow_position_gate_enabled=config.v3_shadow_position_gate_enabled,
        v3_shadow_flat_handoff_enabled=config.v3_shadow_flat_handoff_enabled,
        v3_shadow_fresh_handoff_enabled=config.v3_shadow_fresh_handoff_enabled,
        v3_shadow_fresh_handoff_max_age_bars=config.v3_shadow_fresh_handoff_max_age_bars,
        v3_shadow_rolling_window_bars=rolling_window,
        v3_shadow_rolling_min_closed_trades=min_closed_trades,
        v3_entry_causal_min_filled=config.v3_entry_causal_min_filled,
        v3_entry_causal_actionability_weight=config.v3_entry_causal_actionability_weight,
        v3_real_loss_rescue_enabled=config.v3_real_loss_rescue_enabled,
        v3_real_loss_rescue_min_virtual_pnl_pct=(
            config.v3_real_loss_rescue_min_virtual_pnl_pct
        ),
        v3_real_loss_rescue_max_virtual_dd_pct=(
            config.v3_real_loss_rescue_max_virtual_dd_pct
        ),
        v3_real_loss_rescue_min_actionable_share=(
            config.v3_real_loss_rescue_min_actionable_share
        ),
        v3_real_loss_rescue_min_recent_filled=(
            config.v3_real_loss_rescue_min_recent_filled
        ),
        v3_real_loss_rescue_max_real_loss_pct=(
            config.v3_real_loss_rescue_max_real_loss_pct
        ),
        v3_real_loss_rescue_allow_genetics=(
            config.v3_real_loss_rescue_allow_genetics
        ),
        v3_probation_shadow_rescue_enabled=(
            config.v3_probation_shadow_rescue_enabled
        ),
        v3_probation_shadow_rescue_min_virtual_pnl_pct=(
            config.v3_probation_shadow_rescue_min_virtual_pnl_pct
        ),
        v3_probation_shadow_rescue_max_virtual_dd_pct=(
            config.v3_probation_shadow_rescue_max_virtual_dd_pct
        ),
        v3_probation_shadow_rescue_min_actionable_share=(
            config.v3_probation_shadow_rescue_min_actionable_share
        ),
        v3_probation_shadow_rescue_min_recent_filled=(
            config.v3_probation_shadow_rescue_min_recent_filled
        ),
        v3_probation_shadow_rescue_min_recent_pnl_usd=(
            config.v3_probation_shadow_rescue_min_recent_pnl_usd
        ),
        v3_probation_shadow_rescue_allow_genetics=(
            config.v3_probation_shadow_rescue_allow_genetics
        ),
        v3_persistent_loss_kill_min_closed_trades=(
            config.v3_persistent_loss_kill_min_closed_trades
        ),
        v3_persistent_loss_kill_pnl_pct=config.v3_persistent_loss_kill_pnl_pct,
        v3_persistent_loss_kill_win_rate_pct=(
            config.v3_persistent_loss_kill_win_rate_pct
        ),
        v3_persistent_loss_requires_virtual_weakness=(
            config.v3_persistent_loss_requires_virtual_weakness
        ),
        v3_persistent_loss_virtual_max_pnl_pct=(
            config.v3_persistent_loss_virtual_max_pnl_pct
        ),
        v3_persistent_loss_virtual_min_dd_pct=(
            config.v3_persistent_loss_virtual_min_dd_pct
        ),
        v3_probation_loss_kill_min_closed_trades=(
            config.v3_probation_loss_kill_min_closed_trades
        ),
        v3_probation_loss_kill_pnl_pct=config.v3_probation_loss_kill_pnl_pct,
        v3_probation_loss_kill_win_rate_pct=(
            config.v3_probation_loss_kill_win_rate_pct
        ),
        v3_probation_loss_kill_label_prefixes=(
            config.v3_probation_loss_kill_label_prefixes
        ),
        v3_realized_profit_lock_min_closed_trades=(
            config.v3_realized_profit_lock_min_closed_trades
        ),
        v3_realized_profit_lock_min_peak_pnl_pct=(
            config.v3_realized_profit_lock_min_peak_pnl_pct
        ),
        v3_realized_profit_lock_max_giveback_pct=(
            config.v3_realized_profit_lock_max_giveback_pct
        ),
        v3_realized_profit_lock_floor_pnl_pct=(
            config.v3_realized_profit_lock_floor_pnl_pct
        ),
        v3_panteon_equity_guard_enabled=config.v3_panteon_equity_guard_enabled,
        v3_panteon_equity_guard_min_peak_pnl_pct=(
            config.v3_panteon_equity_guard_min_peak_pnl_pct
        ),
        v3_panteon_equity_guard_max_giveback_pct=(
            config.v3_panteon_equity_guard_max_giveback_pct
        ),
        v3_panteon_equity_guard_floor_pnl_pct=(
            config.v3_panteon_equity_guard_floor_pnl_pct
        ),
        v3_panteon_equity_guard_cooldown_bars=(
            config.v3_panteon_equity_guard_cooldown_bars
        ),
        hard_policy_experimental_min_bar=0,
    )


def _make_on_step(configured_writer: OutputWriter, config: RetrodateMarketConfig):
    def _on_step(step: StepResult) -> None:
        configured_writer.write(step)
        progress_every = int(config.progress_every_bars or 0)
        if progress_every > 0 and step.bar % progress_every == 0:
            print(
                "retrodate progress "
                f"bar={step.bar} regime={step.regime.label} "
                f"leader={step.leader or '-'} "
                f"shadow_filled={step.n_shadow_filled}",
                flush=True,
            )

    return _on_step


def _classify_regime(
    prices: dict[str, float],
    state: RetrodateSnapshotState,
) -> tuple[Regime, float]:
    btc = prices.get("BTC/USDT") or prices.get("BTCUSDT")
    if btc is None:
        return Regime.NEUTRAL, 0.50
    if len(state.btc_closes) < state.btc_closes.maxlen:
        state.btc_closes.append(float(btc))
        return Regime.NEUTRAL, 0.55

    anchor = float(state.btc_closes[0])
    ret = (float(btc) / anchor - 1.0) if anchor > 0 else 0.0
    state.btc_closes.append(float(btc))
    confidence = min(0.95, 0.55 + abs(ret) * 4.0)
    if ret <= -0.08:
        return Regime.CRASH, confidence
    if ret <= -0.025:
        return Regime.BEARISH, confidence
    if ret >= 0.025:
        return Regime.BULLISH, confidence
    return Regime.NEUTRAL, max(0.55, 0.70 - abs(ret) * 3.0)


def _coerce_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _new_output_dir(config: RetrodateMarketConfig) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    return (
        Path(config.results_root).resolve()
        / "RETRODATE_MARKET"
        / f"{ts}_retrodate_market_v2"
    )


def _parse_years(raw: str) -> tuple[int, ...]:
    years = tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())
    if not years:
        raise ValueError("years must not be empty")
    return years


def _parse_optional_agent_labels(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(raw or "").split(",") if part.strip())


def _parse_probation_loss_label_prefixes(args: argparse.Namespace) -> tuple[str, ...]:
    if bool(getattr(args, "v3_probation_loss_kill_all_labels", False)):
        return ()
    raw_prefixes = getattr(args, "v3_probation_loss_kill_label_prefix", None)
    if raw_prefixes is None:
        return DEFAULT_PROBATION_LOSS_LABEL_PREFIXES
    return tuple(str(prefix).strip() for prefix in raw_prefixes if str(prefix).strip())


def write_candidate_diagnostics(
    output_dir: Path | str,
    *,
    candidate_events: Sequence[object],
    rejection_events: Sequence[object] = (),
    filename: str = "candidate_diagnostics.json",
) -> Path:
    buckets: dict[str, dict[str, Any]] = {}
    selected_rows = 0
    for event in candidate_events or ():
        label = str(_event_value(event, "player_label") or "").strip()
        if not label:
            continue
        row = buckets.setdefault(label, _new_candidate_diagnostic_bucket(label))
        row["bars_seen"] += 1
        selected = bool(_event_value(event, "selected_by_pantheon") or False)
        row["selected_bars"] += int(selected)
        selected_rows += int(selected)
        row["score_sum"] += _safe_float(_event_value(event, "score"))
        row["has_data_rows"] += int(bool(_event_value(event, "has_data") or False))
        row["closed_trades_sum"] += _safe_float(_event_value(event, "closed_trades"))
        row["signals_sum"] += _safe_float(_event_value(event, "signals"))
        row["recent_bars_sum"] += _safe_float(_event_value(event, "recent_bars"))
        row["recent_actionable_bars_sum"] += _safe_float(
            _event_value(event, "recent_actionable_bars")
        )
        row["actionable_share_sum"] += _safe_float(
            _event_value(event, "actionable_share")
        )
        row["recent_filled_sum"] += _safe_float(_event_value(event, "recent_filled"))
        row["recent_pnl_usd_sum"] += _safe_float(_event_value(event, "recent_pnl_usd"))

    rejections: dict[str, dict[str, Any]] = {}
    for event in rejection_events or ():
        label = str(_event_value(event, "player_label") or "").strip()
        if not label:
            continue
        reason = str(_event_value(event, "reason") or "")
        row = rejections.setdefault(label, {"count": 0, "last_reason": ""})
        row["count"] += 1
        row["last_reason"] = reason

    candidate_payload = {
        label: _candidate_diagnostic_payload(row)
        for label, row in sorted(
            buckets.items(),
            key=lambda item: (-item[1]["selected_bars"], item[0]),
        )
    }
    data = {
        "candidate_rows": sum(row["bars_seen"] for row in buckets.values()),
        "selected_rows": selected_rows,
        "rejection_rows": sum(row["count"] for row in rejections.values()),
        "group_summary": _candidate_group_summary(candidate_payload),
        "candidates": candidate_payload,
        "rejections": dict(sorted(rejections.items())),
    }
    path = Path(output_dir) / filename
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def _new_candidate_diagnostic_bucket(label: str) -> dict[str, Any]:
    return {
        "label": label,
        "group_type": _candidate_group_type(label),
        "bars_seen": 0,
        "selected_bars": 0,
        "score_sum": 0.0,
        "has_data_rows": 0,
        "closed_trades_sum": 0.0,
        "signals_sum": 0.0,
        "recent_bars_sum": 0.0,
        "recent_actionable_bars_sum": 0.0,
        "actionable_share_sum": 0.0,
        "recent_filled_sum": 0.0,
        "recent_pnl_usd_sum": 0.0,
    }


def _candidate_diagnostic_payload(row: dict[str, Any]) -> dict[str, Any]:
    bars = max(1, int(row["bars_seen"]))
    return {
        "group_type": row["group_type"],
        "bars_seen": int(row["bars_seen"]),
        "selected_bars": int(row["selected_bars"]),
        "selected_share_pct": _share(row["selected_bars"], row["bars_seen"]),
        "avg_score": float(row["score_sum"]) / bars,
        "has_data_share_pct": _share(row["has_data_rows"], row["bars_seen"]),
        "avg_closed_trades": float(row["closed_trades_sum"]) / bars,
        "avg_signals": float(row["signals_sum"]) / bars,
        "avg_recent_bars": float(row["recent_bars_sum"]) / bars,
        "avg_recent_actionable_bars": (
            float(row["recent_actionable_bars_sum"]) / bars
        ),
        "avg_actionable_share": float(row["actionable_share_sum"]) / bars,
        "avg_recent_filled": float(row["recent_filled_sum"]) / bars,
        "avg_recent_pnl_usd": float(row["recent_pnl_usd_sum"]) / bars,
    }


def _candidate_group_summary(candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, dict[str, Any]] = {}
    for row in candidates.values():
        group = str(row.get("group_type") or "profile")
        bucket = summary.setdefault(group, {
            "candidate_count": 0,
            "bars_seen": 0,
            "selected_bars": 0,
        })
        bucket["candidate_count"] += 1
        bucket["bars_seen"] += int(row.get("bars_seen", 0) or 0)
        bucket["selected_bars"] += int(row.get("selected_bars", 0) or 0)
    for bucket in summary.values():
        bucket["selected_share_pct"] = _share(
            bucket["selected_bars"],
            bucket["bars_seen"],
        )
    return dict(sorted(summary.items()))


def _candidate_group_type(label: str) -> str:
    if label == "NoTrade":
        return "notrade"
    if label.startswith("Solo_"):
        return "solo"
    if label.startswith("Fixed_"):
        return "fixed"
    if label.startswith("Antonius_") or label.startswith("Perfect_"):
        return "regime_switch"
    if label.endswith("StaticRotator") or label.startswith("Optimal_"):
        return "rotating"
    return "profile"


def write_oracle_mismatch_report(
    output_dir: str | Path,
    trading_log: str | Path,
    *,
    shadow_updates: Sequence[object],
    candidate_rejections: Sequence[object],
    filename: str = "oracle_mismatch_report.json",
) -> Path:
    rows = _parse_trading_log_rows(Path(trading_log))
    player_updates = [
        event for event in shadow_updates or ()
        if str(_event_value(event, "actor_type") or "") == "player"
    ]
    current_shadow = _shadow_update_index(player_updates)
    perfect_by_bar = _perfect_monthly_leader_by_bar(player_updates)
    soft_by_bar = _soft_confirmed_leader_by_bar(player_updates)
    rejections = _rejection_index(candidate_rejections)
    out_rows: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    for row in rows:
        bar = int(row.get("bar", 0) or 0)
        if bar <= 0:
            continue
        real_leader = str(
            row.get("executed_leader")
            or row.get("selected_leader")
            or row.get("leader")
            or ""
        )
        perfect_leader = perfect_by_bar.get(bar, "")
        soft_leader = soft_by_bar.get(bar, "")
        target = perfect_leader if perfect_leader and perfect_leader != "CASH" else soft_leader
        if not target or target == real_leader:
            continue
        reason = _oracle_mismatch_reason(
            bar=bar,
            target_label=target,
            trading_row=row,
            current_shadow=current_shadow,
            rejections=rejections,
        )
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        out_rows.append({
            "bar": bar,
            "regime": row.get("regime", ""),
            "real_leader": real_leader,
            "selected_leader": row.get("selected_leader", ""),
            "executed_leader": row.get("executed_leader", ""),
            "perfect_leader": perfect_leader,
            "soft_leader": soft_leader,
            "reason": reason,
            "raw_signals": int(row.get("raw_signals", 0) or 0),
            "signals": int(row.get("signals", 0) or 0),
            "filled": int(row.get("filled", 0) or 0),
        })
    data = {
        "summary": {
            "trading_rows": len(rows),
            "mismatch_rows": len(out_rows),
            "reason_counts": dict(sorted(reason_counts.items())),
        },
        "rows": out_rows,
    }
    path = Path(output_dir) / filename
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def _parse_trading_log_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields: dict[str, Any] = {}
        for part in line.split("  "):
            part = part.strip()
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key in {
                "bar",
                "raw_signals",
                "signals",
                "filled",
                "rejected",
                "blocked",
            }:
                try:
                    fields[key] = int(value)
                except ValueError:
                    fields[key] = 0
            else:
                fields[key] = value
        if fields:
            rows.append(fields)
    return rows


def _shadow_update_index(
    updates: Sequence[object],
) -> dict[tuple[int, str], dict[str, float]]:
    index: dict[tuple[int, str], dict[str, float]] = {}
    for event in updates or ():
        bar = int(_event_value(event, "bar") or 0)
        label = str(_event_value(event, "actor_label") or "").strip()
        if bar <= 0 or not label:
            continue
        row = index.setdefault((bar, label), {
            "signals": 0.0,
            "filled": 0.0,
            "pnl_usd": 0.0,
            "closed_trades": 0.0,
        })
        row["signals"] += _safe_float(_event_value(event, "signals"))
        row["filled"] += _safe_float(_event_value(event, "filled"))
        row["pnl_usd"] += _safe_float(_event_value(event, "realized_pnl_usd"))
        row["closed_trades"] += _safe_float(_event_value(event, "closed_trades"))
    return index


def _perfect_monthly_leader_by_bar(updates: Sequence[object]) -> dict[int, str]:
    month_label_pnl: dict[str, dict[str, float]] = {}
    month_by_bar: dict[int, str] = {}
    for event in updates or ():
        month = _event_month_from_update(event)
        label = str(_event_value(event, "actor_label") or "").strip()
        bar = int(_event_value(event, "bar") or 0)
        if not month or not label or bar <= 0:
            continue
        month_by_bar[bar] = month
        bucket = month_label_pnl.setdefault(month, {})
        bucket[label] = bucket.get(label, 0.0) + _safe_float(
            _event_value(event, "realized_pnl_usd")
        )
    best_by_month: dict[str, str] = {}
    for month, pnl_by_label in month_label_pnl.items():
        if not pnl_by_label:
            continue
        best_label = max(pnl_by_label, key=lambda label: (pnl_by_label[label], label))
        best_by_month[month] = best_label if pnl_by_label[best_label] > 0.0 else "CASH"
    return {
        bar: best_by_month.get(month, "")
        for bar, month in month_by_bar.items()
    }


def _soft_confirmed_leader_by_bar(updates: Sequence[object]) -> dict[int, str]:
    grouped: dict[int, list[object]] = {}
    for event in updates or ():
        bar = int(_event_value(event, "bar") or 0)
        label = str(_event_value(event, "actor_label") or "").strip()
        if bar > 0 and label:
            grouped.setdefault(bar, []).append(event)
    state: dict[tuple[str, str], dict[str, Any]] = {}
    out: dict[int, str] = {}
    for bar in sorted(grouped):
        bar_events = grouped[bar]
        regime = str(_event_value(bar_events[0], "regime") or "all").strip().lower() or "all"
        out[bar] = _soft_confirmed_leader_for_regime(state, regime, current_bar=bar)
        for event in bar_events:
            label = str(_event_value(event, "actor_label") or "").strip()
            event_regime = str(_event_value(event, "regime") or "all").strip().lower() or "all"
            row = state.setdefault(
                (event_regime, label),
                {"trades": 0, "bars": [], "prefix": [0.0]},
            )
            row["trades"] += max(0, int(_event_value(event, "closed_trades") or 0))
            row["bars"].append(bar)
            row["prefix"].append(
                float(row["prefix"][-1])
                + _safe_float(_event_value(event, "realized_pnl_usd"))
            )
    return out


def _soft_confirmed_leader_for_regime(
    state: dict[tuple[str, str], dict[str, Any]],
    regime: str,
    *,
    current_bar: int,
) -> str:
    scored: list[tuple[str, float]] = []
    cutoff = int(current_bar) - 24
    for (item_regime, label), row in state.items():
        if item_regime != regime:
            continue
        if int(row.get("trades", 0) or 0) < 50:
            continue
        bars = row.get("bars", []) or []
        prefix = row.get("prefix", [0.0]) or [0.0]
        first_index = bisect_right(bars, cutoff)
        score = float(prefix[-1]) - float(prefix[first_index])
        if score > 0.0:
            scored.append((label, score))
    if not scored:
        return ""
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[0][0]


def _rejection_index(
    rejections: Sequence[object],
) -> dict[tuple[int, str], str]:
    out: dict[tuple[int, str], str] = {}
    for event in rejections or ():
        bar = int(_event_value(event, "bar") or 0)
        label = str(
            _event_value(event, "player_label")
            or _event_value(event, "label")
            or ""
        ).strip()
        reason = str(_event_value(event, "reason") or "")
        if bar > 0 and label:
            out[(bar, label)] = reason
    return out


def _oracle_mismatch_reason(
    *,
    bar: int,
    target_label: str,
    trading_row: dict[str, Any],
    current_shadow: dict[tuple[int, str], dict[str, float]],
    rejections: dict[tuple[int, str], str],
) -> str:
    rejection = str(rejections.get((bar, target_label), "") or "").lower()
    if "kill" in rejection:
        return "kill"
    if rejection:
        return "rejected by policy"
    shadow = current_shadow.get((bar, target_label), {})
    if _safe_float(shadow.get("signals", 0.0)) <= 0.0 and _safe_float(
        shadow.get("filled", 0.0)
    ) <= 0.0:
        return "no current signal"
    reason_text = " ".join(
        str(trading_row.get(key, "") or "").lower()
        for key in ("reason", "fallback_reason", "error")
    )
    if "cooldown" in reason_text:
        return "cooldown"
    if str(trading_row.get("executed_leader") or trading_row.get("leader") or "") in {
        "",
        "-",
        "NoTrade",
    }:
        return "no candidate"
    return "low score"


def _event_month_from_update(event: object) -> str:
    value = _event_value(event, "timestamp")
    if isinstance(value, datetime):
        return value.strftime("%Y-%m")
    text = str(value or "").strip()
    if len(text) >= 7:
        return text[:7]
    return ""


def _event_value(event: object, key: str) -> object:
    if isinstance(event, dict):
        return event.get(key)
    return getattr(event, key, None)


def _safe_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _share(part: object, total: object) -> float:
    denom = _safe_float(total)
    if denom <= 0:
        return 0.0
    return _safe_float(part) / denom * 100.0


def _parse_fixed_agent_player_sets(
    raw_items: Sequence[str],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    parsed: list[tuple[str, tuple[str, ...]]] = []
    for raw in raw_items or ():
        text = str(raw or "").strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError("fixed agent player set must use Label=AgentA,AgentB")
        label, raw_agents = text.split("=", 1)
        agents = tuple(
            part.strip() for part in raw_agents.split(",") if part.strip()
        )
        parsed.append((label.strip(), agents))
    return _normalize_fixed_agent_player_sets(parsed)


def _normalize_fixed_agent_player_sets(
    raw_sets: Sequence[tuple[str, Sequence[str]]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    normalized: list[tuple[str, tuple[str, ...]]] = []
    seen_labels = set()
    for raw_label, raw_agents in raw_sets or ():
        label = str(raw_label or "").strip()
        if not label:
            raise ValueError("fixed agent player label must not be empty")
        if label in seen_labels:
            raise ValueError(f"duplicate fixed agent player label: {label}")
        seen_labels.add(label)
        if isinstance(raw_agents, str):
            agents = tuple(
                part.strip() for part in raw_agents.split(",") if part.strip()
            )
        else:
            agents = tuple(
                str(agent or "").strip()
                for agent in raw_agents
                if str(agent or "").strip()
            )
        if not agents:
            raise ValueError(f"fixed agent player {label} must include at least one agent")
        if len(set(agents)) != len(agents):
            raise ValueError(f"fixed agent player {label} has duplicate agents")
        normalized.append((label, agents))
    return tuple(normalized)


def _format_directory_issues(report: RetrodateDirReport) -> str:
    lines = [f"Retrodate directory is not usable: {report.path}"]
    for issue in report.issues:
        lines.append(f"directory:{issue.code}: {issue.message}")
    return "\n".join(lines)


def _format_selection_errors(reports: Iterable[RetrodateFileReport]) -> str:
    lines = ["Requested Retrodate files failed validation"]
    for report in reports:
        for issue in report.issues:
            lines.append(f"{report.path.name}:{issue.code}: {issue.message}")
    return "\n".join(lines)


def _write_run_summary(
    path: Path,
    summary: RetrodateRunSummary,
    selection: RetrodateFileSelection,
    config: RetrodateMarketConfig,
) -> None:
    effective_strategist = _build_strategist_config(config)
    data = {
        "output_dir": str(summary.output_dir),
        "analysis_report": str(summary.report_path),
        "requested_years": summary.requested_years,
        "executed_years": summary.executed_years,
        "excluded_files": [_file_report_payload(item) for item in selection.excluded_files],
        "missing_years": summary.missing_years,
        "bars_processed": summary.bars_processed,
        "first_timestamp": summary.first_timestamp,
        "last_timestamp": summary.last_timestamp,
        "stride_minutes": summary.stride_minutes,
        "timeframe": config.timeframe,
        "initial_capital": config.initial_capital,
        "include_optional_agents": config.include_optional_agents,
        "optional_agent_labels": list(config.optional_agent_labels),
        "solo_agent_candidate_limit": config.solo_agent_candidate_limit,
        "fixed_agent_players_enabled": config.fixed_agent_players_enabled,
        "fixed_agent_player_count": (
            len(config.fixed_agent_player_sets)
            if config.fixed_agent_players_enabled
            else 0
        ),
        "fixed_agent_player_sets": [
            {"label": label, "agent_labels": list(agent_labels)}
            for label, agent_labels in (
                config.fixed_agent_player_sets
                if config.fixed_agent_players_enabled
                else ()
            )
        ],
        "actionable_fallback_enabled": config.actionable_fallback_enabled,
        "actionable_fallback_min_score": config.actionable_fallback_min_score,
        "actionable_fallback_require_has_data": (
            config.actionable_fallback_require_has_data
        ),
        "current_actionable_candidate_layer_enabled": (
            config.current_actionable_candidate_layer_enabled
            or config.use_v3_executable_soft_confirmed_score
        ),
        "real_promotion_gate_enabled": config.real_promotion_gate_enabled,
        "real_promotion_min_closed_trades": config.real_promotion_min_closed_trades,
        "real_promotion_min_pnl_pct": config.real_promotion_min_pnl_pct,
        "real_promotion_max_drawdown_pct": config.real_promotion_max_drawdown_pct,
        "real_promotion_loss_budget_pct": config.real_promotion_loss_budget_pct,
        "real_promotion_probation_min_score": config.real_promotion_probation_min_score,
        "use_v3_rolling_score": config.use_v3_rolling_score,
        "use_v3_entry_causal_score": config.use_v3_entry_causal_score,
        "use_v3_shadow_rolling_score": config.use_v3_shadow_rolling_score,
        "use_v3_soft_shadow_score": config.use_v3_soft_shadow_score,
        "use_v3_executable_soft_top1_score": (
            config.use_v3_executable_soft_top1_score
        ),
        "use_v3_executable_soft_confirmed_score": (
            config.use_v3_executable_soft_confirmed_score
        ),
        "v3_shadow_position_gate_enabled": config.v3_shadow_position_gate_enabled,
        "v3_shadow_flat_handoff_enabled": config.v3_shadow_flat_handoff_enabled,
        "v3_shadow_fresh_handoff_enabled": config.v3_shadow_fresh_handoff_enabled,
        "v3_shadow_fresh_handoff_max_age_bars": config.v3_shadow_fresh_handoff_max_age_bars,
        "v3_shadow_rolling_window_bars": config.v3_shadow_rolling_window_bars,
        "v3_shadow_rolling_min_closed_trades": config.v3_shadow_rolling_min_closed_trades,
        "effective_v3_current_actionable_gate_enabled": (
            effective_strategist.v3_current_actionable_gate_enabled
        ),
        "effective_v3_shadow_rolling_window_bars": (
            effective_strategist.v3_shadow_rolling_window_bars
        ),
        "effective_v3_shadow_rolling_min_closed_trades": (
            effective_strategist.v3_shadow_rolling_min_closed_trades
        ),
        "v3_entry_causal_min_filled": config.v3_entry_causal_min_filled,
        "v3_entry_causal_actionability_weight": (
            config.v3_entry_causal_actionability_weight
        ),
        "v3_real_loss_rescue_enabled": config.v3_real_loss_rescue_enabled,
        "v3_real_loss_rescue_min_virtual_pnl_pct": (
            config.v3_real_loss_rescue_min_virtual_pnl_pct
        ),
        "v3_real_loss_rescue_max_virtual_dd_pct": (
            config.v3_real_loss_rescue_max_virtual_dd_pct
        ),
        "v3_real_loss_rescue_min_actionable_share": (
            config.v3_real_loss_rescue_min_actionable_share
        ),
        "v3_real_loss_rescue_min_recent_filled": (
            config.v3_real_loss_rescue_min_recent_filled
        ),
        "v3_real_loss_rescue_max_real_loss_pct": (
            config.v3_real_loss_rescue_max_real_loss_pct
        ),
        "v3_real_loss_rescue_allow_genetics": (
            config.v3_real_loss_rescue_allow_genetics
        ),
        "v3_probation_shadow_rescue_enabled": (
            config.v3_probation_shadow_rescue_enabled
        ),
        "v3_probation_shadow_rescue_min_virtual_pnl_pct": (
            config.v3_probation_shadow_rescue_min_virtual_pnl_pct
        ),
        "v3_probation_shadow_rescue_max_virtual_dd_pct": (
            config.v3_probation_shadow_rescue_max_virtual_dd_pct
        ),
        "v3_probation_shadow_rescue_min_actionable_share": (
            config.v3_probation_shadow_rescue_min_actionable_share
        ),
        "v3_probation_shadow_rescue_min_recent_filled": (
            config.v3_probation_shadow_rescue_min_recent_filled
        ),
        "v3_probation_shadow_rescue_min_recent_pnl_usd": (
            config.v3_probation_shadow_rescue_min_recent_pnl_usd
        ),
        "v3_probation_shadow_rescue_allow_genetics": (
            config.v3_probation_shadow_rescue_allow_genetics
        ),
        "v3_persistent_loss_kill_min_closed_trades": (
            config.v3_persistent_loss_kill_min_closed_trades
        ),
        "v3_persistent_loss_kill_pnl_pct": config.v3_persistent_loss_kill_pnl_pct,
        "v3_persistent_loss_kill_win_rate_pct": (
            config.v3_persistent_loss_kill_win_rate_pct
        ),
        "v3_persistent_loss_requires_virtual_weakness": (
            config.v3_persistent_loss_requires_virtual_weakness
        ),
        "v3_persistent_loss_virtual_max_pnl_pct": (
            config.v3_persistent_loss_virtual_max_pnl_pct
        ),
        "v3_persistent_loss_virtual_min_dd_pct": (
            config.v3_persistent_loss_virtual_min_dd_pct
        ),
        "v3_probation_loss_kill_min_closed_trades": (
            config.v3_probation_loss_kill_min_closed_trades
        ),
        "v3_probation_loss_kill_pnl_pct": config.v3_probation_loss_kill_pnl_pct,
        "v3_probation_loss_kill_win_rate_pct": (
            config.v3_probation_loss_kill_win_rate_pct
        ),
        "v3_probation_loss_kill_label_prefixes": list(
            config.v3_probation_loss_kill_label_prefixes
        ),
        "hard_policy_deny_labels": list(config.hard_policy_deny_labels),
        "genetics_probation_execution_enabled": (
            config.genetics_probation_execution_enabled
        ),
        "genetics_probation_risk_mult": config.genetics_probation_risk_mult,
        "genetics_probation_max_real_trades": (
            config.genetics_probation_max_real_trades
        ),
        "genetics_probation_require_shadow_confirmation": (
            config.genetics_probation_require_shadow_confirmation
        ),
        "v3_realized_profit_lock_min_closed_trades": (
            config.v3_realized_profit_lock_min_closed_trades
        ),
        "v3_realized_profit_lock_min_peak_pnl_pct": (
            config.v3_realized_profit_lock_min_peak_pnl_pct
        ),
        "v3_realized_profit_lock_max_giveback_pct": (
            config.v3_realized_profit_lock_max_giveback_pct
        ),
        "v3_realized_profit_lock_floor_pnl_pct": (
            config.v3_realized_profit_lock_floor_pnl_pct
        ),
        "v3_panteon_equity_guard_enabled": (
            config.v3_panteon_equity_guard_enabled
        ),
        "v3_panteon_equity_guard_min_peak_pnl_pct": (
            config.v3_panteon_equity_guard_min_peak_pnl_pct
        ),
        "v3_panteon_equity_guard_max_giveback_pct": (
            config.v3_panteon_equity_guard_max_giveback_pct
        ),
        "v3_panteon_equity_guard_floor_pnl_pct": (
            config.v3_panteon_equity_guard_floor_pnl_pct
        ),
        "v3_panteon_equity_guard_cooldown_bars": (
            config.v3_panteon_equity_guard_cooldown_bars
        ),
        "registered_agents": list(summary.registered_agents),
        "player_profile_count": summary.player_profile_count,
        "max_bars": summary.max_bars,
        "step_errors": list(summary.step_errors),
        "validation_files": [_file_report_payload(item) for item in selection.report.files],
    }
    soft_allocator = _load_json(summary.output_dir / "soft_allocator_report.json")
    if soft_allocator:
        data["soft_allocator_report"] = {
            "best_single_label": soft_allocator.get("best_single_label", ""),
            "best_single_pnl_usd": soft_allocator.get("best_single_pnl_usd", 0.0),
            "best_policy_name": soft_allocator.get("best_policy_name", ""),
            "best_policy_pnl_usd": soft_allocator.get("best_policy_pnl_usd", 0.0),
            "best_policy_max_drawdown_pct": soft_allocator.get(
                "best_policy_max_drawdown_pct",
                0.0,
            ),
            "regret_vs_best_single_usd": soft_allocator.get(
                "regret_vs_best_single_usd",
                0.0,
            ),
            "beats_best_single": soft_allocator.get("beats_best_single", False),
        }
    perfect_panteon = _load_json(summary.output_dir / "perfect_panteon_report.json")
    if perfect_panteon:
        data["perfect_panteon_report"] = {
            "pnl_usd": perfect_panteon.get("pnl_usd", 0.0),
            "pnl_pct": perfect_panteon.get("pnl_pct", 0.0),
            "max_drawdown_pct": perfect_panteon.get("max_drawdown_pct", 0.0),
            "month_count": perfect_panteon.get("month_count", 0),
            "profitable_months": perfect_panteon.get("profitable_months", 0),
            "cash_months": perfect_panteon.get("cash_months", 0),
            "closed_trades": perfect_panteon.get("closed_trades", 0.0),
        }
    allocation_diagnostics = _load_json(
        summary.output_dir / "allocation_diagnostics.json"
    )
    if allocation_diagnostics:
        data["allocation_diagnostics"] = {
            "bars": allocation_diagnostics.get("bars", 0),
            "no_trade_bars": allocation_diagnostics.get("no_trade_bars", 0),
            "no_trade_share_pct": allocation_diagnostics.get("no_trade_share_pct", 0.0),
            "raw_zero_bars": allocation_diagnostics.get("raw_zero_bars", 0),
            "raw_zero_share_pct": allocation_diagnostics.get("raw_zero_share_pct", 0.0),
            "filled_zero_bars": allocation_diagnostics.get("filled_zero_bars", 0),
            "filled_zero_share_pct": allocation_diagnostics.get(
                "filled_zero_share_pct",
                0.0,
            ),
        }
    candidate_diagnostics = _load_json(
        summary.output_dir / "candidate_diagnostics.json"
    )
    if candidate_diagnostics:
        data["candidate_diagnostics"] = {
            "candidate_rows": candidate_diagnostics.get("candidate_rows", 0),
            "selected_rows": candidate_diagnostics.get("selected_rows", 0),
            "rejection_rows": candidate_diagnostics.get("rejection_rows", 0),
            "group_summary": candidate_diagnostics.get("group_summary", {}),
        }
    oracle_mismatch = _load_json(summary.output_dir / "oracle_mismatch_report.json")
    if oracle_mismatch:
        data["oracle_mismatch_report"] = oracle_mismatch.get("summary", {})
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _write_analysis_report(
    path: Path,
    summary: RetrodateRunSummary,
    selection: RetrodateFileSelection,
) -> None:
    status = _load_json(summary.output_dir / "status.json")
    agents_payload = _load_json(summary.output_dir / "leaderboard_agents.json")
    players_payload = _load_json(summary.output_dir / "leaderboard_players.json")
    agents = agents_payload.get("agents", {}) if isinstance(agents_payload, dict) else {}
    players = players_payload.get("players", {}) if isinstance(players_payload, dict) else {}
    live_session = status.get("live_session", {}) if isinstance(status, dict) else {}
    shadow = status.get("shadow", {}) if isinstance(status, dict) else {}
    soft_allocator = _load_json(summary.output_dir / "soft_allocator_report.json")
    perfect_panteon = _load_json(summary.output_dir / "perfect_panteon_report.json")
    allocation_diagnostics = _load_json(
        summary.output_dir / "allocation_diagnostics.json"
    )
    candidate_diagnostics = _load_json(
        summary.output_dir / "candidate_diagnostics.json"
    )
    oracle_mismatch = _load_json(summary.output_dir / "oracle_mismatch_report.json")

    excluded_lines = []
    for item in selection.excluded_files:
        issues = "; ".join(f"{issue.code}: {issue.message}" for issue in item.issues)
        excluded_lines.append(f"- `{item.path.name}`: {issues}")
    if not excluded_lines:
        excluded_lines.append("- none")

    dashboard_files = [
        "dashboard.html",
        "dashboard.txt",
        "dashboard_latest.png",
        "shadow_dashboard.png",
        "regime_dashboard.png",
        "memory_dashboard.png",
        "status.json",
        "leaderboard_agents.json",
        "leaderboard_players.json",
        "soft_allocator_report.md",
        "soft_allocator_report.json",
        "perfect_panteon_report.md",
        "perfect_panteon_report.json",
        "allocation_diagnostics.json",
        "candidate_diagnostics.json",
        "oracle_mismatch_report.json",
        "causal_entry_decisions.jsonl",
        "shadow_player_pnl_events.jsonl",
        "shadow_agent_pnl_events.jsonl",
    ]
    dashboard_lines = [
        f"- `{name}`"
        for name in dashboard_files
        if (summary.output_dir / name).exists()
    ]
    note_lines = [
        "- This run uses the existing live-like Panteon v2 pipeline, OutputWriter, leaderboards, and dashboard renderers.",
        "- Shadow tournament evaluates registered agents and composed player profiles on every replayed bar.",
        "- The default run is an hourly-stride benchmark; a full 1m replay is available with --stride-minutes 1 but is much heavier.",
    ]
    if any(
        int(report.expected_year or 0) == 2026 and not report.is_valid
        for report in selection.excluded_files
    ):
        note_lines.insert(
            2,
            "- The requested 2026 file was excluded by validation; check Data Exclusions above.",
        )

    lines = [
        "# Retrodate Market Benchmark",
        "",
        "## Scope",
        f"- Requested years: {', '.join(str(year) for year in summary.requested_years)}",
        f"- Executed years: {', '.join(str(year) for year in summary.executed_years) or 'none'}",
        f"- Period: {summary.first_timestamp or 'n/a'} to {summary.last_timestamp or 'n/a'}",
        f"- Bars processed: {summary.bars_processed}",
        f"- Replay stride: {summary.stride_minutes} minutes from 1m OHLCV data",
        f"- Registered agents: {len(summary.registered_agents)}",
        f"- Configured player profiles: {summary.player_profile_count}",
        f"- Player leaderboard rows: {len(players)}",
        "",
        "## Data Exclusions",
        *excluded_lines,
        "",
        "## Panteon Result",
        f"- Current leader: `{status.get('current_leader') or '-'}`",
        f"- Panteon owned PnL: {_fmt_pct(live_session.get('panteon_owned_pnl_pct'))}",
        f"- Panteon realized PnL USD: {_fmt_money(live_session.get('panteon_owned_realized_pnl_usd'))}",
        f"- Panteon max drawdown: {_fmt_pct(status.get('panteon_max_drawdown_pct'))}",
        f"- Panteon realized max drawdown: {_fmt_pct(status.get('panteon_realized_max_drawdown_pct'))}",
        f"- Real closed trades: {live_session.get('real_closed_trades', 0)}",
        f"- Open Panteon positions: {live_session.get('panteon_owned_positions_count', 0)}",
        f"- Last shadow actors: {shadow.get('actors', 0)}",
        "",
        *(
            _allocation_diagnostics_report_lines(allocation_diagnostics) + [""]
            if allocation_diagnostics else []
        ),
        *(
            _candidate_diagnostics_report_lines(candidate_diagnostics) + [""]
            if candidate_diagnostics else []
        ),
        *(
            _oracle_mismatch_report_lines(oracle_mismatch) + [""]
            if oracle_mismatch else []
        ),
        *(_soft_allocator_report_lines(soft_allocator) + [""] if soft_allocator else []),
        *(_perfect_panteon_report_lines(perfect_panteon) + [""] if perfect_panteon else []),
        "## Top Agents By Virtual PnL",
        _markdown_table(_top_rows(agents, limit=10)),
        "",
        "## Top Players By Virtual PnL",
        _markdown_table(_top_rows(players, limit=10)),
        "",
        "## Notes",
        *note_lines,
        "",
        "## Dashboard Artifacts",
        *(dashboard_lines or ["- none"]),
    ]
    if summary.step_errors:
        lines.extend([
            "",
            "## Step Errors",
            *[f"- {error}" for error in summary.step_errors[:20]],
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _allocation_diagnostics_report_lines(report: dict[str, Any]) -> list[str]:
    if not report:
        return []
    top = []
    leaders = report.get("leaders", {})
    if isinstance(leaders, dict):
        for label, payload in list(leaders.items())[:5]:
            if isinstance(payload, dict):
                top.append(
                    f"- `{label}`: bars {int(payload.get('bars', 0) or 0)}, "
                    f"raw-zero {_fmt_pct(payload.get('raw_zero_share_pct'))}, "
                    f"filled-zero {_fmt_pct(payload.get('filled_zero_share_pct'))}"
                )
    return [
        "## Allocation Diagnostics",
        f"- Bars analyzed: {int(report.get('bars', 0) or 0)}",
        f"- NoTrade share: {_fmt_pct(report.get('no_trade_share_pct'))}",
        f"- Raw-zero share: {_fmt_pct(report.get('raw_zero_share_pct'))}",
        f"- Filled-zero share: {_fmt_pct(report.get('filled_zero_share_pct'))}",
        f"- Details: `allocation_diagnostics.json`",
        *(["- Top leaders by bar share:"] + top if top else []),
    ]


def _candidate_diagnostics_report_lines(report: dict[str, Any]) -> list[str]:
    if not report:
        return []
    groups = report.get("group_summary", {})
    group_lines = []
    if isinstance(groups, dict):
        for group, payload in groups.items():
            if isinstance(payload, dict):
                group_lines.append(
                    f"- `{group}`: candidates {int(payload.get('candidate_count', 0) or 0)}, "
                    f"selected bars {int(payload.get('selected_bars', 0) or 0)}, "
                    f"selected share {_fmt_pct(payload.get('selected_share_pct'))}"
                )
    return [
        "## Candidate Diagnostics",
        f"- Candidate score rows: {int(report.get('candidate_rows', 0) or 0)}",
        f"- Selected rows: {int(report.get('selected_rows', 0) or 0)}",
        f"- Rejection rows: {int(report.get('rejection_rows', 0) or 0)}",
        f"- Details: `candidate_diagnostics.json`",
        *(["- Group summary:"] + group_lines if group_lines else []),
    ]


def _oracle_mismatch_report_lines(report: dict[str, Any]) -> list[str]:
    if not report:
        return []
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    reason_counts = summary.get("reason_counts", {}) if isinstance(summary, dict) else {}
    reason_lines = []
    if isinstance(reason_counts, dict):
        reason_lines = [
            f"- `{reason}`: {int(count or 0)}"
            for reason, count in sorted(reason_counts.items())
        ]
    return [
        "## Oracle Mismatch",
        f"- Trading rows: {int(summary.get('trading_rows', 0) or 0)}",
        f"- Mismatch rows: {int(summary.get('mismatch_rows', 0) or 0)}",
        f"- Details: `oracle_mismatch_report.json`",
        *(["- Reason counts:"] + reason_lines if reason_lines else []),
    ]


def _soft_allocator_report_lines(report: dict[str, Any]) -> list[str]:
    if not report:
        return []
    return [
        "## Soft Allocator Simulation",
        f"- Best single shadow player: `{report.get('best_single_label') or '-'}` "
        f"({_fmt_money(report.get('best_single_pnl_usd'))}, "
        f"{_fmt_pct(report.get('best_single_pnl_pct'))}, "
        f"DD {_fmt_pct(report.get('best_single_max_drawdown_pct'))})",
        f"- Best soft policy: `{report.get('best_policy_name') or '-'}` "
        f"({_fmt_money(report.get('best_policy_pnl_usd'))}, "
        f"{_fmt_pct(report.get('best_policy_pnl_pct'))}, "
        f"DD {_fmt_pct(report.get('best_policy_max_drawdown_pct'))})",
        f"- Regret vs best single: {_fmt_money(report.get('regret_vs_best_single_usd'))}",
        f"- Beats best single: {'yes' if report.get('beats_best_single') else 'no'}",
        f"- Details: `soft_allocator_report.md`, `soft_allocator_report.json`",
    ]


def _perfect_panteon_report_lines(report: dict[str, Any]) -> list[str]:
    if not report:
        return []
    return [
        "## Perfect Panteon Monthly Oracle",
        f"- Perfect monthly PnL: {_fmt_money(report.get('pnl_usd'))}, "
        f"{_fmt_pct(report.get('pnl_pct'))}",
        f"- Max drawdown: {_fmt_pct(report.get('max_drawdown_pct'))}",
        f"- Months observed: {int(report.get('month_count', 0) or 0)}",
        f"- Profitable months: {int(report.get('profitable_months', 0) or 0)}",
        f"- Cash months: {int(report.get('cash_months', 0) or 0)}",
        f"- Closed trades: {float(report.get('closed_trades', 0.0) or 0.0):.2f}",
        f"- Details: `perfect_panteon_report.md`, `perfect_panteon_report.json`",
    ]


def _file_report_payload(report: RetrodateFileReport) -> dict[str, Any]:
    return {
        "path": str(report.path),
        "expected_year": report.expected_year,
        "timeframe": report.timeframe,
        "rows": report.rows,
        "symbols": report.symbols,
        "first_datetime": report.first_datetime,
        "last_datetime": report.last_datetime,
        "found_years": report.found_years,
        "is_valid": report.is_valid,
        "issues": [
            {"code": issue.code, "message": issue.message}
            for issue in report.issues
        ],
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _top_rows(payload: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    rows = []
    for label, metrics in payload.items():
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "label": str(label),
                "pnl_pct": _coerce_float(metrics.get("pnl_pct")),
                "closed_trades": int(_coerce_float(metrics.get("closed_trades"))),
                "win_rate": _coerce_float(metrics.get("win_rate")),
                "signals": int(_coerce_float(metrics.get("signals"))),
                "max_drawdown_pct": _coerce_float(metrics.get("max_drawdown_pct")),
            }
        )
    return sorted(rows, key=lambda item: item["pnl_pct"], reverse=True)[:limit]


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No data._"
    lines = [
        "| Rank | Label | PnL % | Trades | Win rate % | Signals | Max DD % |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"| {index} | `{row['label']}` | "
            f"{row['pnl_pct']:.2f} | {row['closed_trades']} | "
            f"{row['win_rate']:.2f} | {row['signals']} | "
            f"{row['max_drawdown_pct']:.2f} |"
        )
    return "\n".join(lines)


def _fmt_pct(value: object) -> str:
    return f"{_coerce_float(value):.2f}%"


def _fmt_money(value: object) -> str:
    return f"${_coerce_float(value):.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
