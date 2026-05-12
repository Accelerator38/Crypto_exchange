"""Comparator — сравнение решений v1 vs v2 за окно времени.

Главные метрики:
  • Какого лидера выбрал v1 vs v2 на каждом баре
  • Перекрытие сигналов (v1 ∩ v2 / v1 ∪ v2)
  • Какие агенты попали в карантин в v1 vs v2
  • Σ realized PnL v1 vs v2 attribution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from ..attribution import AttributionLedger
from ..memory import QuarantineManager
from ..replay.v1_parser import V1Session
from .runner import StepResult


# ────────────────────────────────────────────────────────────────────
# Comparison report
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LeaderDiff:
    """Сравнение какого лидера выбрали v1 и v2 на каждом баре."""

    bar:        int
    v1_leader:  str
    v2_leader:  str

    @property
    def matches(self) -> bool:
        return self.v1_leader == self.v2_leader


@dataclass
class ComparisonReport:
    session_path:        str
    exchange:            str
    # Агрегаты
    n_bars_compared:     int
    n_leader_match:      int
    n_leader_diff:       int
    # Сигналы
    n_v1_signals:        int
    n_v2_signals:        int
    n_signals_overlap:   int          # одинаковые (sym, action) пары
    # PnL
    v1_pnl_pct:          float
    v2_attribution_pct:  float
    pnl_diff_pct:        float
    # Карантин
    v1_quarantined:      Set[str] = field(default_factory=set)
    v2_quarantined:      Set[str] = field(default_factory=set)
    # Подробности
    leader_diffs:        List[LeaderDiff] = field(default_factory=list)
    notes:               List[str] = field(default_factory=list)

    @property
    def leader_match_rate(self) -> float:
        if self.n_bars_compared <= 0:
            return 0.0
        return self.n_leader_match / self.n_bars_compared * 100.0

    @property
    def signal_overlap_rate(self) -> float:
        union = max(self.n_v1_signals + self.n_v2_signals - self.n_signals_overlap, 1)
        return self.n_signals_overlap / union * 100.0


# ────────────────────────────────────────────────────────────────────
# Comparison logic
# ────────────────────────────────────────────────────────────────────


def _normalize(label: str) -> str:
    return (label or "").replace("V_", "").strip()


def compare_v1_vs_v2(
    session:    V1Session,
    v2_steps:   List[StepResult],
    v2_qm:      QuarantineManager,
    v2_ledger:  AttributionLedger,
) -> ComparisonReport:
    """Главная функция сравнения.

    Сравниваем:
      • На каждом баре v1: кто был выбран в selected_player → сравниваем
        с v2 leader на том же или ближайшем баре.
      • Сигналы: пары (sym, action_code) — overlap в %.
      • Карантин: множества лейблов, симметрическая разность.
      • PnL: v1.pnl_pct vs Σ v2_ledger.player_pnl / initial_capital × 100.
    """
    notes: List[str] = []

    # ── v1 lider per bar ────────────────────────────────────────────
    v1_leaders_by_bar: Dict[int, str] = {}
    v1_signal_keys: Set[tuple] = set()
    for sig in session.real_signals:
        bar = sig.bar
        leader = _normalize(sig.selected_player)
        if leader and bar not in v1_leaders_by_bar:
            v1_leaders_by_bar[bar] = leader
        v1_signal_keys.add((sig.sym, sig.action_code))

    # ── v2 leader per bar ────────────────────────────────────────────
    v2_leaders_by_bar: Dict[int, str] = {}
    v2_signal_keys: Set[tuple] = set()
    for step in v2_steps:
        if step.leader:
            v2_leaders_by_bar[step.bar] = step.leader
        for sig in step.signals:
            v2_signal_keys.add((sig.sym, int(sig.action)))

    # ── Leader sequence diff ─────────────────────────────────────────
    common_bars = sorted(set(v1_leaders_by_bar) & set(v2_leaders_by_bar))
    leader_diffs: List[LeaderDiff] = []
    n_match = 0
    for bar in common_bars:
        d = LeaderDiff(
            bar=bar,
            v1_leader=v1_leaders_by_bar[bar],
            v2_leader=v2_leaders_by_bar[bar],
        )
        leader_diffs.append(d)
        if d.matches:
            n_match += 1

    if not common_bars:
        notes.append("Нет общих баров для сравнения лидера (v1 и v2 не пересекаются по time).")

    # ── Signals overlap ──────────────────────────────────────────────
    overlap = v1_signal_keys & v2_signal_keys

    # ── PnL ──────────────────────────────────────────────────────────
    v1_pnl_pct = session.pnl_pct
    if session.initial_capital > 0:
        v2_attr_pct = v2_ledger.total_realized_pnl / session.initial_capital * 100.0
    else:
        v2_attr_pct = 0.0
    pnl_diff = v2_attr_pct - v1_pnl_pct

    # ── Карантины ────────────────────────────────────────────────────
    v1_q = set(session.quarantined)
    v2_q = {_normalize(x) for x in v2_qm.all_quarantined()}

    # ── Notes на базе наблюдений ─────────────────────────────────────
    if v1_q != v2_q:
        added = v2_q - v1_q
        removed = v1_q - v2_q
        if added:
            notes.append(f"v2 добавил в карантин: {sorted(added)}")
        if removed:
            notes.append(f"v2 убрал из карантина: {sorted(removed)}")

    if abs(pnl_diff) > 1.0:
        notes.append(
            f"PnL diff = {pnl_diff:+.2f}% — выходит за норму ±1.00%"
        )

    return ComparisonReport(
        session_path=session.session_path,
        exchange=session.exchange,
        n_bars_compared=len(common_bars),
        n_leader_match=n_match,
        n_leader_diff=len(common_bars) - n_match,
        n_v1_signals=len(v1_signal_keys),
        n_v2_signals=len(v2_signal_keys),
        n_signals_overlap=len(overlap),
        v1_pnl_pct=v1_pnl_pct,
        v2_attribution_pct=v2_attr_pct,
        pnl_diff_pct=pnl_diff,
        v1_quarantined=v1_q,
        v2_quarantined=v2_q,
        leader_diffs=leader_diffs,
        notes=notes,
    )


# ────────────────────────────────────────────────────────────────────
# Pretty-print
# ────────────────────────────────────────────────────────────────────


def render_comparison(report: ComparisonReport) -> str:
    out = []
    line = "═" * 78
    out.append(line)
    out.append(f"  V1 vs V2 SHADOW comparison: {report.exchange}")
    out.append(f"  Session: {report.session_path}")
    out.append(line)
    out.append(f"  v1 PnL%:           {report.v1_pnl_pct:+.4f}%")
    out.append(f"  v2 attribution%:   {report.v2_attribution_pct:+.4f}%")
    out.append(f"  diff:              {report.pnl_diff_pct:+.4f}%")
    out.append("")
    out.append(f"  Bars compared:     {report.n_bars_compared}")
    if report.n_bars_compared > 0:
        out.append(
            f"  Leader match:      {report.n_leader_match}/{report.n_bars_compared} "
            f"({report.leader_match_rate:.1f}%)"
        )
    out.append("")
    out.append(f"  v1 unique signals: {report.n_v1_signals}")
    out.append(f"  v2 unique signals: {report.n_v2_signals}")
    out.append(
        f"  overlap:           {report.n_signals_overlap} "
        f"({report.signal_overlap_rate:.1f}% of union)"
    )
    out.append("")
    out.append(f"  v1 quarantined ({len(report.v1_quarantined)}):  "
               f"{', '.join(sorted(report.v1_quarantined)) or '(none)'}")
    out.append(f"  v2 quarantined ({len(report.v2_quarantined)}):  "
               f"{', '.join(sorted(report.v2_quarantined)) or '(none)'}")
    if report.notes:
        out.append("")
        out.append("  Notes:")
        for n in report.notes:
            out.append(f"    • {n}")

    # Distinct leader-diffs (топ 10)
    distinct = []
    seen = set()
    for d in report.leader_diffs:
        if d.matches:
            continue
        key = (d.v1_leader, d.v2_leader)
        if key in seen:
            continue
        seen.add(key)
        distinct.append(d)
        if len(distinct) >= 10:
            break
    if distinct:
        out.append("")
        out.append("  Distinct leader disagreements (sample):")
        for d in distinct:
            out.append(f"    bar={d.bar:<6d}  v1={d.v1_leader:<28s} v2={d.v2_leader}")
    return "\n".join(out)
