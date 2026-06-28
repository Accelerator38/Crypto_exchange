"""Offline soft-allocation analysis over shadow actor realized PnL streams."""

from __future__ import annotations

import json
import math
from bisect import bisect_right
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ShadowPnLEvent:
    bar: int
    label: str
    timestamp: str = ""
    regime: str = "all"
    pnl_usd: float = 0.0
    closed_trades: int = 0
    wins: int = 0
    symbol_action_outcomes: tuple[tuple[str, str, float, int, int], ...] = ()


@dataclass(frozen=True)
class SoftAllocatorPolicy:
    name: str
    top_k: int = 3
    cash_reserve_weight: float = 0.15
    max_weight_per_leader: float = 0.45
    min_closed_trades: int = 20
    half_life_bars: int = 2160
    drawdown_penalty: float = 0.15
    persistent_loss_min_closed_trades: int = 30
    persistent_loss_pnl_usd: float = -20.0
    persistent_loss_max_weight: float = 0.05
    score_scope: str = "global"
    score_mode: str = "top"
    rolling_window_bars: int = 0


@dataclass(frozen=True)
class SoftAllocatorPolicyResult:
    policy_name: str
    pnl_usd: float
    pnl_pct: float
    final_equity_usd: float
    max_drawdown_pct: float
    weighted_closed_trades: float
    average_cash_weight_pct: float
    average_leader_count: float
    bars: int


@dataclass(frozen=True)
class SoftAllocatorReport:
    initial_capital: float
    event_count: int
    bar_count: int
    player_count: int
    best_single_label: str
    best_single_pnl_usd: float
    best_single_pnl_pct: float
    best_single_max_drawdown_pct: float
    best_policy_name: str
    best_policy_pnl_usd: float
    best_policy_pnl_pct: float
    best_policy_max_drawdown_pct: float
    regret_vs_best_single_usd: float
    beats_best_single: bool
    policy_results: tuple[SoftAllocatorPolicyResult, ...]


@dataclass(frozen=True)
class PerfectMonthSelection:
    month: str
    label: str
    pnl_usd: float
    pnl_pct: float
    closed_trades: float
    wins: float
    win_rate_pct: float


@dataclass(frozen=True)
class PerfectPanteonReport:
    initial_capital: float
    event_count: int
    month_count: int
    player_count: int
    pnl_usd: float
    pnl_pct: float
    final_equity_usd: float
    max_drawdown_pct: float
    closed_trades: float
    wins: float
    win_rate_pct: float
    profitable_months: int
    cash_months: int
    months: tuple[PerfectMonthSelection, ...]


@dataclass
class _ActorState:
    total_pnl_usd: float = 0.0
    total_trades: int = 0
    total_wins: int = 0
    decayed_pnl_usd: float = 0.0
    decayed_trades: float = 0.0
    decayed_wins: float = 0.0
    equity_usd: float = 0.0
    peak_equity_usd: float = 0.0
    max_drawdown_pct: float = 0.0
    recent_bars: list[int] = field(default_factory=list)
    recent_pnl_prefix: list[float] = field(default_factory=lambda: [0.0])


@dataclass
class OnlineSoftAllocator:
    """Causal online state for applying a soft allocation policy during replay/live.

    The state is advanced from already-completed shadow actor updates. Call
    ``weights_for_bar`` before adding updates from that same bar to keep the
    execution decision causal.
    """

    policy: SoftAllocatorPolicy
    initial_capital: float
    actor_type: str = "player"
    _states_by_scope: dict[str, dict[str, _ActorState]] = field(default_factory=dict, init=False)
    _last_bar: int = field(default=0, init=False)

    def weights_for_bar(self, regime: object, current_bar: int) -> dict[str, float]:
        bar = max(0, int(current_bar or 0))
        self._advance_to_bar(bar)
        scope = _scope_for_regime(regime, self.policy)
        return _weights_for_policy(
            self._states_by_scope.get(scope, {}),
            self.policy,
            current_bar=bar,
        )

    def update_from_shadow_updates(self, updates: Iterable[object]) -> None:
        events = shadow_pnl_events_from_shadow_updates(
            updates,
            actor_type=str(self.actor_type or "player"),
        )
        for event in sorted(events, key=lambda item: (item.bar, item.label)):
            if not event.label or int(event.bar) <= 0 or not math.isfinite(float(event.pnl_usd)):
                continue
            self._advance_to_bar(int(event.bar))
            scope = _scope_for_event(event, self.policy)
            states = self._states_by_scope.setdefault(scope, {})
            state = states.setdefault(
                event.label,
                _ActorState(
                    equity_usd=self.initial_capital,
                    peak_equity_usd=self.initial_capital,
                ),
            )
            _apply_actor_event(
                state,
                event,
                initial_capital=max(1e-9, float(self.initial_capital or 0.0)),
            )

    def _advance_to_bar(self, bar: int) -> None:
        if bar <= 0:
            return
        if self._last_bar and bar > self._last_bar:
            elapsed = bar - self._last_bar
            for states in self._states_by_scope.values():
                _decay_states(states.values(), elapsed_bars=elapsed, policy=self.policy)
        if bar > self._last_bar:
            self._last_bar = bar


def default_soft_allocator_policies() -> tuple[SoftAllocatorPolicy, ...]:
    return (
        SoftAllocatorPolicy(
            name="soft_regime_top1_24",
            top_k=1,
            cash_reserve_weight=0.0,
            max_weight_per_leader=1.0,
            min_closed_trades=20,
            half_life_bars=1_000_000,
            drawdown_penalty=0.0,
            persistent_loss_max_weight=0.0,
            score_scope="regime",
            score_mode="top",
            rolling_window_bars=24,
        ),
        SoftAllocatorPolicy(
            name="soft_regime_top1_24_confirmed",
            top_k=1,
            cash_reserve_weight=0.0,
            max_weight_per_leader=1.0,
            min_closed_trades=50,
            half_life_bars=1_000_000,
            drawdown_penalty=0.0,
            persistent_loss_max_weight=0.0,
            score_scope="regime",
            score_mode="top",
            rolling_window_bars=24,
        ),
        SoftAllocatorPolicy(
            name="soft_regime_contrarian_720",
            top_k=1,
            cash_reserve_weight=0.0,
            max_weight_per_leader=1.0,
            min_closed_trades=20,
            half_life_bars=1_000_000,
            drawdown_penalty=0.0,
            persistent_loss_max_weight=1.0,
            score_scope="regime",
            score_mode="bottom",
            rolling_window_bars=720,
        ),
        SoftAllocatorPolicy(
            name="soft_top1_alltime",
            top_k=1,
            cash_reserve_weight=0.0,
            max_weight_per_leader=1.0,
            min_closed_trades=20,
            half_life_bars=1_000_000,
            drawdown_penalty=0.0,
            persistent_loss_max_weight=0.0,
        ),
        SoftAllocatorPolicy(
            name="soft_top2_alltime_capped",
            top_k=2,
            cash_reserve_weight=0.0,
            max_weight_per_leader=0.70,
            min_closed_trades=20,
            half_life_bars=1_000_000,
            drawdown_penalty=0.0,
            persistent_loss_max_weight=0.0,
        ),
        SoftAllocatorPolicy(
            name="soft_top2_decayed",
            top_k=2,
            cash_reserve_weight=0.10,
            max_weight_per_leader=0.55,
            min_closed_trades=20,
            half_life_bars=1440,
            drawdown_penalty=0.10,
        ),
        SoftAllocatorPolicy(
            name="soft_top3_decayed",
            top_k=3,
            cash_reserve_weight=0.15,
            max_weight_per_leader=0.45,
            min_closed_trades=20,
            half_life_bars=2160,
            drawdown_penalty=0.15,
        ),
        SoftAllocatorPolicy(
            name="soft_top4_dd_capped",
            top_k=4,
            cash_reserve_weight=0.20,
            max_weight_per_leader=0.35,
            min_closed_trades=20,
            half_life_bars=2880,
            drawdown_penalty=0.30,
            persistent_loss_max_weight=0.03,
        ),
    )


def shadow_pnl_events_from_shadow_updates(
    events: Iterable[object],
    *,
    actor_type: str = "player",
) -> list[ShadowPnLEvent]:
    rows: list[ShadowPnLEvent] = []
    for event in events:
        if str(_event_value(event, "actor_type") or "") != actor_type:
            continue
        label = str(_event_value(event, "actor_label") or "").strip()
        if not label:
            continue
        rows.append(
            ShadowPnLEvent(
                bar=_int(_event_value(event, "bar")),
                label=label,
                timestamp=_timestamp_iso(_event_value(event, "timestamp")),
                regime=str(_event_value(event, "regime") or "all").strip() or "all",
                pnl_usd=_float(_event_value(event, "realized_pnl_usd")),
                closed_trades=_int(_event_value(event, "closed_trades")),
                wins=_int(_event_value(event, "winning_trades")),
                symbol_action_outcomes=_coerce_symbol_action_outcomes(
                    _event_value(event, "symbol_action_outcomes")
                ),
            )
        )
    return rows


def simulate_soft_allocator_policies(
    events: Sequence[ShadowPnLEvent],
    *,
    initial_capital: float,
    policies: Sequence[SoftAllocatorPolicy] | None = None,
) -> SoftAllocatorReport:
    clean_events = [
        event for event in events
        if event.label and int(event.bar) > 0 and math.isfinite(float(event.pnl_usd))
    ]
    clean_events.sort(key=lambda item: (item.bar, item.label))
    capital = max(1e-9, float(initial_capital or 0.0))
    grouped = _group_by_bar(clean_events)
    single = _single_actor_stats(clean_events, initial_capital=capital)
    policy_list = tuple(policies or default_soft_allocator_policies())
    policy_results = tuple(
        _simulate_policy(grouped, initial_capital=capital, policy=policy)
        for policy in policy_list
    )
    best_policy = max(
        policy_results,
        key=lambda item: (item.pnl_usd, -item.max_drawdown_pct, item.policy_name),
        default=SoftAllocatorPolicyResult(
            policy_name="",
            pnl_usd=0.0,
            pnl_pct=0.0,
            final_equity_usd=capital,
            max_drawdown_pct=0.0,
            weighted_closed_trades=0.0,
            average_cash_weight_pct=100.0,
            average_leader_count=0.0,
            bars=0,
        ),
    )
    best_single_label, best_single_pnl, best_single_dd = single
    return SoftAllocatorReport(
        initial_capital=capital,
        event_count=len(clean_events),
        bar_count=len(grouped),
        player_count=len({event.label for event in clean_events}),
        best_single_label=best_single_label,
        best_single_pnl_usd=best_single_pnl,
        best_single_pnl_pct=best_single_pnl / capital * 100.0,
        best_single_max_drawdown_pct=best_single_dd,
        best_policy_name=best_policy.policy_name,
        best_policy_pnl_usd=best_policy.pnl_usd,
        best_policy_pnl_pct=best_policy.pnl_pct,
        best_policy_max_drawdown_pct=best_policy.max_drawdown_pct,
        regret_vs_best_single_usd=max(0.0, best_single_pnl - best_policy.pnl_usd),
        beats_best_single=best_policy.pnl_usd > best_single_pnl,
        policy_results=policy_results,
    )


def simulate_perfect_monthly_panteon(
    events: Sequence[ShadowPnLEvent],
    *,
    initial_capital: float,
    cash_label: str = "CASH",
) -> PerfectPanteonReport:
    """Monthly hindsight oracle: choose the best actor for each month."""

    capital = max(1e-9, float(initial_capital or 0.0))
    clean_events = [
        event for event in events
        if (
            event.label
            and _event_month(event)
            and int(event.bar) > 0
            and math.isfinite(float(event.pnl_usd))
        )
    ]
    clean_events.sort(key=lambda item: (_event_month(item), item.bar, item.label))
    grouped = _group_by_month_and_label(clean_events)
    months: list[PerfectMonthSelection] = []
    equity = capital
    curve = [equity]
    closed_trades = 0.0
    wins = 0.0
    for month in sorted(grouped):
        by_label = grouped[month]
        totals = {
            label: sum(float(event.pnl_usd) for event in items)
            for label, items in by_label.items()
        }
        best_label = max(totals, key=lambda label: (totals[label], label))
        best_pnl = float(totals[best_label])
        if best_pnl <= 0.0:
            months.append(PerfectMonthSelection(
                month=month,
                label=cash_label,
                pnl_usd=0.0,
                pnl_pct=0.0,
                closed_trades=0.0,
                wins=0.0,
                win_rate_pct=0.0,
            ))
            curve.append(equity)
            continue
        month_trades = 0.0
        month_wins = 0.0
        for event in sorted(by_label[best_label], key=lambda item: (item.bar, item.label)):
            equity += float(event.pnl_usd)
            curve.append(equity)
            trades = max(0, int(event.closed_trades))
            event_wins = max(0, int(event.wins))
            month_trades += trades
            month_wins += event_wins
        closed_trades += month_trades
        wins += month_wins
        months.append(PerfectMonthSelection(
            month=month,
            label=best_label,
            pnl_usd=best_pnl,
            pnl_pct=best_pnl / capital * 100.0,
            closed_trades=month_trades,
            wins=month_wins,
            win_rate_pct=(month_wins / month_trades * 100.0) if month_trades else 0.0,
        ))
    pnl = equity - capital
    return PerfectPanteonReport(
        initial_capital=capital,
        event_count=len(clean_events),
        month_count=len(grouped),
        player_count=len({event.label for event in clean_events}),
        pnl_usd=pnl,
        pnl_pct=pnl / capital * 100.0,
        final_equity_usd=equity,
        max_drawdown_pct=_curve_max_drawdown_pct(curve),
        closed_trades=closed_trades,
        wins=wins,
        win_rate_pct=(wins / closed_trades * 100.0) if closed_trades else 0.0,
        profitable_months=sum(1 for item in months if item.pnl_usd > 0.0),
        cash_months=sum(1 for item in months if item.label == cash_label),
        months=tuple(months),
    )


def write_soft_allocator_report(
    output_dir: str | Path,
    report: SoftAllocatorReport,
) -> tuple[Path, Path]:
    root = Path(output_dir)
    json_path = root / "soft_allocator_report.json"
    md_path = root / "soft_allocator_report.md"
    json_path.write_text(
        json.dumps(soft_allocator_report_to_dict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md_path.write_text(render_soft_allocator_markdown(report), encoding="utf-8")
    return json_path, md_path


def write_perfect_panteon_report(
    output_dir: str | Path,
    report: PerfectPanteonReport,
) -> tuple[Path, Path]:
    root = Path(output_dir)
    json_path = root / "perfect_panteon_report.json"
    md_path = root / "perfect_panteon_report.md"
    json_path.write_text(
        json.dumps(perfect_panteon_report_to_dict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md_path.write_text(render_perfect_panteon_markdown(report), encoding="utf-8")
    return json_path, md_path


def write_shadow_pnl_events(
    output_dir: str | Path,
    events: Sequence[ShadowPnLEvent],
    *,
    filename: str = "shadow_player_pnl_events.jsonl",
) -> Path:
    path = Path(output_dir) / filename
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
    return path


def soft_allocator_report_to_dict(report: SoftAllocatorReport) -> dict:
    data = asdict(report)
    data["policy_results"] = [asdict(item) for item in report.policy_results]
    return data


def perfect_panteon_report_to_dict(report: PerfectPanteonReport) -> dict:
    data = asdict(report)
    data["months"] = [asdict(item) for item in report.months]
    return data


def render_soft_allocator_markdown(report: SoftAllocatorReport) -> str:
    lines = [
        "# Soft Allocator Simulation",
        "",
        "## Summary",
        f"- Shadow player PnL events: {report.event_count}",
        f"- Bars observed: {report.bar_count}",
        f"- Players observed: {report.player_count}",
        f"- Best single shadow player: `{report.best_single_label or '-'}` "
        f"({_fmt_money(report.best_single_pnl_usd)}, {_fmt_pct(report.best_single_pnl_pct)}, "
        f"DD {_fmt_pct(report.best_single_max_drawdown_pct)})",
        f"- Best soft policy: `{report.best_policy_name or '-'}` "
        f"({_fmt_money(report.best_policy_pnl_usd)}, {_fmt_pct(report.best_policy_pnl_pct)}, "
        f"DD {_fmt_pct(report.best_policy_max_drawdown_pct)})",
        f"- Regret vs best single: {_fmt_money(report.regret_vs_best_single_usd)}",
        f"- Beats best single: {'yes' if report.beats_best_single else 'no'}",
        "",
        "## Policy Results",
        "| Policy | PnL USD | PnL % | Max DD % | Weighted trades | Avg cash % | Avg leaders |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in sorted(report.policy_results, key=lambda row: row.pnl_usd, reverse=True):
        lines.append(
            f"| `{item.policy_name}` | {item.pnl_usd:.2f} | {item.pnl_pct:.2f} | "
            f"{item.max_drawdown_pct:.2f} | {item.weighted_closed_trades:.2f} | "
            f"{item.average_cash_weight_pct:.2f} | {item.average_leader_count:.2f} |"
        )
    return "\n".join(lines) + "\n"


def render_perfect_panteon_markdown(report: PerfectPanteonReport) -> str:
    lines = [
        "# Perfect Panteon Monthly Oracle",
        "",
        "## Summary",
        f"- Shadow player PnL events: {report.event_count}",
        f"- Months observed: {report.month_count}",
        f"- Players observed: {report.player_count}",
        f"- Perfect monthly PnL: {_fmt_money(report.pnl_usd)}, {_fmt_pct(report.pnl_pct)}",
        f"- Max drawdown: {_fmt_pct(report.max_drawdown_pct)}",
        f"- Closed trades: {report.closed_trades:.2f}",
        f"- Win rate: {_fmt_pct(report.win_rate_pct)}",
        f"- Profitable months: {report.profitable_months}",
        f"- Cash months: {report.cash_months}",
        "",
        "## Monthly Selection",
        "| Month | Selected | PnL USD | PnL % | Trades | Win rate % |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in report.months:
        lines.append(
            f"| `{item.month}` | `{item.label}` | {item.pnl_usd:.2f} | "
            f"{item.pnl_pct:.2f} | {item.closed_trades:.2f} | {item.win_rate_pct:.2f} |"
        )
    return "\n".join(lines) + "\n"


def _simulate_policy(
    grouped: Mapping[int, Sequence[ShadowPnLEvent]],
    *,
    initial_capital: float,
    policy: SoftAllocatorPolicy,
) -> SoftAllocatorPolicyResult:
    states_by_scope: dict[str, dict[str, _ActorState]] = {}
    equity = float(initial_capital)
    curve = [equity]
    weighted_trades = 0.0
    cash_weight_sum = 0.0
    leader_count_sum = 0.0
    last_bar = 0
    bars = 0
    for bar in sorted(grouped):
        if last_bar:
            for states in states_by_scope.values():
                _decay_states(states.values(), elapsed_bars=max(0, bar - last_bar), policy=policy)
        last_bar = bar
        bar_events = grouped[bar]
        weights_by_scope = {
            scope: _weights_for_policy(states, policy, current_bar=bar)
            for scope, states in states_by_scope.items()
        }
        bar_pnl = 0.0
        for event in bar_events:
            weight = weights_by_scope.get(_scope_for_event(event, policy), {}).get(event.label, 0.0)
            if weight:
                bar_pnl += weight * float(event.pnl_usd)
                weighted_trades += weight * max(0, int(event.closed_trades))
        equity += bar_pnl
        curve.append(equity)
        active_weights = _active_bar_weights(bar_events, weights_by_scope, policy)
        allocated = sum(active_weights.values())
        cash_weight_sum += max(0.0, 1.0 - allocated)
        leader_count_sum += sum(1 for value in active_weights.values() if value > 1e-9)
        bars += 1
        for event in bar_events:
            scope = _scope_for_event(event, policy)
            states = states_by_scope.setdefault(scope, {})
            state = states.setdefault(
                event.label,
                _ActorState(equity_usd=initial_capital, peak_equity_usd=initial_capital),
            )
            _apply_actor_event(state, event, initial_capital=initial_capital)
    pnl = equity - initial_capital
    return SoftAllocatorPolicyResult(
        policy_name=policy.name,
        pnl_usd=pnl,
        pnl_pct=pnl / initial_capital * 100.0,
        final_equity_usd=equity,
        max_drawdown_pct=_curve_max_drawdown_pct(curve),
        weighted_closed_trades=weighted_trades,
        average_cash_weight_pct=(cash_weight_sum / bars * 100.0) if bars else 100.0,
        average_leader_count=(leader_count_sum / bars) if bars else 0.0,
        bars=bars,
    )


def _weights_for_policy(
    states: Mapping[str, _ActorState],
    policy: SoftAllocatorPolicy,
    *,
    current_bar: int,
) -> dict[str, float]:
    target = max(0.0, min(1.0, 1.0 - float(policy.cash_reserve_weight)))
    if target <= 0.0:
        return {}
    scored: list[tuple[str, float, float]] = []
    for label, state in states.items():
        if state.total_trades < max(0, int(policy.min_closed_trades)):
            continue
        score = _score_for_state(state, policy, current_bar=current_bar)
        if score <= 0:
            continue
        cap = _cap_for_state(state, policy)
        if cap <= 0:
            continue
        scored.append((label, score, cap))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return _capped_score_weights(scored[: max(0, int(policy.top_k))], target)


def _score_for_state(
    state: _ActorState,
    policy: SoftAllocatorPolicy,
    *,
    current_bar: int,
) -> float:
    raw_score = _raw_policy_pnl_score(state, policy, current_bar=current_bar)
    mode = str(policy.score_mode or "top").strip().lower()
    if mode == "bottom":
        return max(0.0, -raw_score)
    return max(
        0.0,
        raw_score - max(0.0, float(policy.drawdown_penalty)) * _actor_drawdown_usd(state),
    )


def _raw_policy_pnl_score(
    state: _ActorState,
    policy: SoftAllocatorPolicy,
    *,
    current_bar: int,
) -> float:
    window = max(0, int(policy.rolling_window_bars))
    if window <= 0:
        return float(state.decayed_pnl_usd)
    cutoff = int(current_bar) - window
    first_index = bisect_right(state.recent_bars, cutoff)
    return state.recent_pnl_prefix[-1] - state.recent_pnl_prefix[first_index]


def _active_bar_weights(
    bar_events: Sequence[ShadowPnLEvent],
    weights_by_scope: Mapping[str, Mapping[str, float]],
    policy: SoftAllocatorPolicy,
) -> dict[str, float]:
    active: dict[str, float] = {}
    active_scopes = {_scope_for_event(event, policy) for event in bar_events}
    for scope in active_scopes:
        for label, weight in weights_by_scope.get(scope, {}).items():
            active[f"{scope}|{label}"] = float(weight)
    return active


def _scope_for_event(event: ShadowPnLEvent, policy: SoftAllocatorPolicy) -> str:
    if str(policy.score_scope or "global").strip().lower() == "regime":
        return str(event.regime or "all").strip().lower() or "all"
    return "all"


def _scope_for_regime(regime: object, policy: SoftAllocatorPolicy) -> str:
    if str(policy.score_scope or "global").strip().lower() != "regime":
        return "all"
    raw = getattr(regime, "label", regime)
    return str(raw or "all").strip().lower() or "all"


def _cap_for_state(state: _ActorState, policy: SoftAllocatorPolicy) -> float:
    base = max(0.0, min(1.0, float(policy.max_weight_per_leader)))
    if (
        state.total_trades >= max(0, int(policy.persistent_loss_min_closed_trades))
        and state.total_pnl_usd <= float(policy.persistent_loss_pnl_usd)
    ):
        return min(base, max(0.0, min(1.0, float(policy.persistent_loss_max_weight))))
    return base


def _capped_score_weights(
    scored: Sequence[tuple[str, float, float]],
    target: float,
) -> dict[str, float]:
    remaining = [(label, score, cap) for label, score, cap in scored if score > 0 and cap > 0]
    weights: dict[str, float] = {}
    remaining_target = float(target)
    while remaining and remaining_target > 1e-12:
        total_score = sum(score for _, score, _ in remaining)
        if total_score <= 0:
            break
        capped: list[tuple[str, float, float]] = []
        uncapped: list[tuple[str, float, float, float]] = []
        for label, score, cap in remaining:
            proposed = remaining_target * score / total_score
            if proposed >= cap:
                capped.append((label, score, cap))
            else:
                uncapped.append((label, score, cap, proposed))
        if not capped:
            for label, _, _, proposed in uncapped:
                weights[label] = weights.get(label, 0.0) + proposed
            break
        for label, _, cap in capped:
            allocation = min(cap, remaining_target)
            weights[label] = weights.get(label, 0.0) + allocation
            remaining_target -= allocation
        capped_labels = {label for label, _, _ in capped}
        remaining = [
            (label, score, max(0.0, cap - weights.get(label, 0.0)))
            for label, score, cap in remaining
            if label not in capped_labels and max(0.0, cap - weights.get(label, 0.0)) > 1e-12
        ]
    return {label: weight for label, weight in weights.items() if weight > 1e-12}


def _single_actor_stats(
    events: Sequence[ShadowPnLEvent],
    *,
    initial_capital: float,
) -> tuple[str, float, float]:
    curves: dict[str, list[float]] = {}
    totals: dict[str, float] = {}
    for event in events:
        label = event.label
        curve = curves.setdefault(label, [initial_capital])
        next_equity = curve[-1] + float(event.pnl_usd)
        curve.append(next_equity)
        totals[label] = totals.get(label, 0.0) + float(event.pnl_usd)
    if not totals:
        return "", 0.0, 0.0
    best_label = max(totals, key=lambda label: (totals[label], label))
    return best_label, totals[best_label], _curve_max_drawdown_pct(curves[best_label])


def _group_by_bar(events: Sequence[ShadowPnLEvent]) -> dict[int, list[ShadowPnLEvent]]:
    grouped: dict[int, list[ShadowPnLEvent]] = {}
    for event in events:
        grouped.setdefault(int(event.bar), []).append(event)
    return grouped


def _group_by_month_and_label(
    events: Sequence[ShadowPnLEvent],
) -> dict[str, dict[str, list[ShadowPnLEvent]]]:
    grouped: dict[str, dict[str, list[ShadowPnLEvent]]] = {}
    for event in events:
        month = _event_month(event)
        if not month:
            continue
        grouped.setdefault(month, {}).setdefault(event.label, []).append(event)
    return grouped


def _decay_states(
    states: Iterable[_ActorState],
    *,
    elapsed_bars: int,
    policy: SoftAllocatorPolicy,
) -> None:
    half_life = max(1, int(policy.half_life_bars))
    decay = 0.5 ** (max(0, int(elapsed_bars)) / half_life)
    for state in states:
        state.decayed_pnl_usd *= decay
        state.decayed_trades *= decay
        state.decayed_wins *= decay


def _apply_actor_event(
    state: _ActorState,
    event: ShadowPnLEvent,
    *,
    initial_capital: float,
) -> None:
    pnl = float(event.pnl_usd)
    trades = max(0, int(event.closed_trades))
    wins = max(0, int(event.wins))
    state.total_pnl_usd += pnl
    state.total_trades += trades
    state.total_wins += wins
    state.decayed_pnl_usd += pnl
    state.decayed_trades += trades
    state.decayed_wins += wins
    state.recent_bars.append(int(event.bar))
    state.recent_pnl_prefix.append(state.recent_pnl_prefix[-1] + pnl)
    if state.equity_usd <= 0:
        state.equity_usd = initial_capital
    if state.peak_equity_usd <= 0:
        state.peak_equity_usd = initial_capital
    state.equity_usd += pnl
    if state.equity_usd > state.peak_equity_usd:
        state.peak_equity_usd = state.equity_usd
    if state.peak_equity_usd > 0:
        state.max_drawdown_pct = max(
            state.max_drawdown_pct,
            (state.peak_equity_usd - state.equity_usd) / state.peak_equity_usd * 100.0,
        )


def _actor_drawdown_usd(state: _ActorState) -> float:
    return max(0.0, float(state.peak_equity_usd) - float(state.equity_usd))


def _curve_max_drawdown_pct(curve: Sequence[float]) -> float:
    peak = 0.0
    max_dd = 0.0
    for value in curve:
        current = float(value or 0.0)
        if current <= 0:
            continue
        peak = max(peak, current)
        if peak > 0:
            max_dd = max(max_dd, (peak - current) / peak * 100.0)
    return max_dd


def _event_value(event: object, key: str) -> object:
    if isinstance(event, Mapping):
        return event.get(key)
    return getattr(event, key, None)


def _event_month(event: ShadowPnLEvent) -> str:
    stamp = str(event.timestamp or "").strip()
    if len(stamp) >= 7 and stamp[4:5] == "-" and stamp[7:8] in ("", "-", "T", " "):
        return stamp[:7]
    return ""


def _timestamp_iso(value: object) -> str:
    if value is None:
        return ""
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        try:
            return str(iso())
        except Exception:
            return ""
    return str(value or "")


def _coerce_symbol_action_outcomes(
    value: object,
) -> tuple[tuple[str, str, float, int, int], ...]:
    rows: list[tuple[str, str, float, int, int]] = []
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        return ()
    for item in value:
        if not isinstance(item, Iterable) or isinstance(item, (str, bytes)):
            continue
        parts = tuple(item)
        if len(parts) < 5:
            continue
        symbol = str(parts[0] or "").strip().upper()
        action = str(parts[1] or "").strip().upper()
        if not symbol or not action:
            continue
        rows.append((
            symbol,
            action,
            _float(parts[2]),
            _int(parts[3]),
            _int(parts[4]),
        ))
    return tuple(rows)


def _int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _fmt_pct(value: float) -> str:
    return f"{float(value):.2f}%"


def _fmt_money(value: float) -> str:
    return f"${float(value):.2f}"
