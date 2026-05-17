"""Run Panteon v2 live-like benchmarks on Retrodate OHLCV CSV data."""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Iterable, Optional, Sequence

from ..app.agent_bootstrap import register_all_v1_agents
from ..app.bootstrap import build_production_pipeline
from ..app.main_loop import StepResult, main_loop
from ..app.output_writer import OutputWriter, OutputWriterConfig
from ..app.shadow_tournament import ProductionShadowTournament
from ..domain.types import MarketSnapshot, Regime
from ..execution import FakeExchange
from ..selection import AgentRegistry, StrategistConfig
from ..shadow.feed import ReplayFeed
from .retrodate_validator import (
    RetrodateDirReport,
    RetrodateFileReport,
    RetrodateValidationError,
    validate_retrodate_dir,
)


DEFAULT_YEARS = (2022, 2023, 2024, 2025, 2026)


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
    real_promotion_gate_enabled: bool = False
    real_promotion_min_closed_trades: int = 20
    real_promotion_min_pnl_pct: float = 0.0
    real_promotion_max_drawdown_pct: float = 25.0
    real_promotion_loss_budget_pct: float = -1.0
    real_promotion_probation_min_score: float = 0.0
    use_v3_rolling_score: bool = False

    def __post_init__(self) -> None:
        if self.stride_minutes <= 0:
            raise ValueError("stride_minutes must be > 0")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")
        if self.invalid_policy not in {"exclude", "fail"}:
            raise ValueError("invalid_policy must be 'exclude' or 'fail'")
        object.__setattr__(
            self,
            "optional_agent_labels",
            tuple(str(label) for label in self.optional_agent_labels),
        )
        if self.real_promotion_min_closed_trades < 0:
            raise ValueError("real_promotion_min_closed_trades must be >= 0")
        if self.real_promotion_max_drawdown_pct < 0:
            raise ValueError("real_promotion_max_drawdown_pct must be >= 0")


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
    seen_years = {item.expected_year for item in selected}
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
    )
    pipeline.mode = "retrodate_market"
    pipeline.timeframe = f"{config.stride_minutes}m-from-{config.timeframe}"
    pipeline.session_id = output_dir.name
    pipeline.run_id = output_dir.name
    pipeline.shadow_tournament = ProductionShadowTournament(
        registry=pipeline.registry,
        perf=pipeline.virtual_perf,
        risk_config=pipeline.risk_config,
        event_log=None,
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
        real_promotion_gate_enabled=args.real_promotion_gate,
        real_promotion_min_closed_trades=args.real_promotion_min_closed_trades,
        real_promotion_min_pnl_pct=args.real_promotion_min_pnl_pct,
        real_promotion_max_drawdown_pct=args.real_promotion_max_dd_pct,
        real_promotion_loss_budget_pct=args.real_promotion_loss_budget_pct,
        real_promotion_probation_min_score=args.real_promotion_probation_min_score,
        use_v3_rolling_score=args.use_v3_rolling_score,
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
    parser.add_argument("--real-promotion-gate", action="store_true")
    parser.add_argument("--real-promotion-min-closed-trades", type=int, default=20)
    parser.add_argument("--real-promotion-min-pnl-pct", type=float, default=0.0)
    parser.add_argument("--real-promotion-max-dd-pct", type=float, default=25.0)
    parser.add_argument("--real-promotion-loss-budget-pct", type=float, default=-1.0)
    parser.add_argument("--real-promotion-probation-min-score", type=float, default=0.0)
    parser.add_argument("--use-v3-rolling-score", action="store_true")
    return parser


def _build_strategist_config(config: RetrodateMarketConfig) -> StrategistConfig:
    return StrategistConfig(
        real_promotion_gate_enabled=config.real_promotion_gate_enabled,
        real_promotion_min_closed_trades=config.real_promotion_min_closed_trades,
        real_promotion_min_pnl_pct=config.real_promotion_min_pnl_pct,
        real_promotion_max_drawdown_pct=config.real_promotion_max_drawdown_pct,
        real_promotion_loss_budget_pct=config.real_promotion_loss_budget_pct,
        real_promotion_probation_min_score=config.real_promotion_probation_min_score,
        use_v3_rolling_score=config.use_v3_rolling_score,
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
        "real_promotion_gate_enabled": config.real_promotion_gate_enabled,
        "real_promotion_min_closed_trades": config.real_promotion_min_closed_trades,
        "real_promotion_min_pnl_pct": config.real_promotion_min_pnl_pct,
        "real_promotion_max_drawdown_pct": config.real_promotion_max_drawdown_pct,
        "real_promotion_loss_budget_pct": config.real_promotion_loss_budget_pct,
        "real_promotion_probation_min_score": config.real_promotion_probation_min_score,
        "use_v3_rolling_score": config.use_v3_rolling_score,
        "registered_agents": list(summary.registered_agents),
        "player_profile_count": summary.player_profile_count,
        "max_bars": summary.max_bars,
        "step_errors": list(summary.step_errors),
        "validation_files": [_file_report_payload(item) for item in selection.report.files],
    }
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
    ]
    dashboard_lines = [
        f"- `{name}`"
        for name in dashboard_files
        if (summary.output_dir / name).exists()
    ]

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
        f"- Real closed trades: {live_session.get('real_closed_trades', 0)}",
        f"- Open Panteon positions: {live_session.get('panteon_owned_positions_count', 0)}",
        f"- Last shadow actors: {shadow.get('actors', 0)}",
        "",
        "## Top Agents By Virtual PnL",
        _markdown_table(_top_rows(agents, limit=10)),
        "",
        "## Top Players By Virtual PnL",
        _markdown_table(_top_rows(players, limit=10)),
        "",
        "## Notes",
        "- This run uses the existing live-like Panteon v2 pipeline, OutputWriter, leaderboards, and dashboard renderers.",
        "- Shadow tournament evaluates registered agents and composed player profiles on every replayed bar.",
        "- Because the available 2026 file contains 2022-2023 timestamps, it is excluded by validation unless invalid_policy=fail is disabled explicitly.",
        "- The default run is an hourly-stride benchmark; a full 1m replay is available with --stride-minutes 1 but is much heavier.",
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
