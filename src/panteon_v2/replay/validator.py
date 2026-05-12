"""Validator — Q-гарантии на исторических v1-данных.

Запускается после synthesize_v2_state. Проверяет:
  Q1: ни один лидер v1 не использовал агента, который v1 пометил
      карантинным в leaderboard
  Q2: Σ AttributionLedger.player_pnl ≈ realized_pnl_total
      (то есть сумма attribution ≈ нашему расчёту PnL trade'ов)
  Q4: ни один selected_player из real_signals не имеет в составе
      карантинного агента (через v1 contributors)
  Дополнительно:
    • ratio attributed/total signals
    • orphan-closes (close без open)
    • расхождение с v1 stats.pnl_pct
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .synthesizer import SynthState
from .v1_parser import V1Session


@dataclass(frozen=True)
class ValidationResult:
    """Один Q-чек."""

    name:        str
    passed:      bool
    detail:      str = ""
    severity:    str = "info"   # "info" | "warn" | "fail"


@dataclass
class ReplayValidationReport:
    """Полный отчёт по валидации одной сессии."""

    session_path:        str
    exchange:            str
    initial_capital:     float
    v1_pnl_pct:          float
    v2_attribution_pnl:  float
    v2_attribution_pct:  float       # как % от initial_capital
    n_real_signals:      int
    n_open:              int
    n_close:             int
    n_orphan_close:      int
    quarantined_count:   int
    failed_orders:       int
    results:             List[ValidationResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(
            r.passed for r in self.results if r.severity != "info"
        )

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.results if not r.passed and r.severity == "fail")

    @property
    def warn_count(self) -> int:
        return sum(1 for r in self.results if not r.passed and r.severity == "warn")


# ────────────────────────────────────────────────────────────────────


def _normalize(label: str) -> str:
    return label.replace("V_", "").strip()


def validate_session(
    session: V1Session,
    state:   SynthState,
    *,
    pnl_tolerance_pct: float = 0.5,    # допустимое расхождение в %
) -> ReplayValidationReport:
    """Главная функция. Возвращает отчёт со всеми Q-чеками."""
    results: List[ValidationResult] = []

    # ── Q1: ни один real-signal не должен исходить от карантинного игрока ──
    quarantined_players = {
        _normalize(label) for label, ent in session.players.items()
        if ent.status in ("quarantine", "shadow_only", "purgatory")
    }
    quarantined_agents = {
        _normalize(label) for label, ent in session.agents.items()
        if ent.status in ("quarantine", "shadow_only", "purgatory")
    }
    leaks_player: List[str] = []
    leaks_agent: List[str] = []
    for sig in session.real_signals:
        sp = _normalize(sig.selected_player)
        sa = _normalize(sig.selected_agents)
        if sp and sp in quarantined_players:
            leaks_player.append(sp)
        if sa and sa in quarantined_agents:
            leaks_agent.append(sa)
    leaks_player_set = sorted(set(leaks_player))
    leaks_agent_set = sorted(set(leaks_agent))
    results.append(ValidationResult(
        name="Q1: no quarantined player as leader",
        passed=not leaks_player_set,
        detail=(
            "PASS — ни один карантинный player не торговал" if not leaks_player_set
            else f"FAIL — карантинные players торговали: {leaks_player_set}"
        ),
        severity="fail",
    ))
    results.append(ValidationResult(
        name="Q4: no quarantined agent as initiator",
        passed=not leaks_agent_set,
        detail=(
            "PASS — карантинные agents не были инициаторами"
            if not leaks_agent_set
            else f"FAIL — карантинные agents инициировали: {leaks_agent_set}"
        ),
        severity="fail",
    ))

    # ── Q2: AttributionLedger.consistency ──
    consistent = state.ledger.consistency_check()
    results.append(ValidationResult(
        name="Q2: ledger consistency check",
        passed=consistent,
        detail=("PASS" if consistent
                else "FAIL — Σ by_player + unattributed != total_realized"),
        severity="fail",
    ))

    # ── Sanity: orphan closes (close без open) ──
    orphan_ratio = (
        state.n_orphan_close / max(state.n_close + state.n_orphan_close, 1)
    )
    results.append(ValidationResult(
        name="orphan close ratio",
        passed=orphan_ratio < 0.30,
        detail=(
            f"orphan_close/total = {state.n_orphan_close}/"
            f"{state.n_close + state.n_orphan_close} ({orphan_ratio:.0%})"
        ),
        severity="warn",
    ))

    # ── Sanity: расхождение v2 attribution vs v1 stats.pnl ──
    v2_attribution_pnl = state.ledger.total_realized_pnl
    v2_attribution_pct = (
        v2_attribution_pnl / session.initial_capital * 100.0
        if session.initial_capital > 0 else 0.0
    )
    diff = abs(v2_attribution_pct - session.pnl_pct)
    pnl_check_passed = diff < pnl_tolerance_pct
    # Замечание: точное равенство недостижимо т.к. fees/slippage в v2
    # синтетические. Главное чтобы порядок величины совпадал.
    results.append(ValidationResult(
        name=f"v2 attribution ≈ v1 stats.pnl_pct (±{pnl_tolerance_pct:.2f}%)",
        passed=pnl_check_passed,
        detail=(
            f"v1.pnl_pct={session.pnl_pct:+.4f}%, "
            f"v2.attribution_pct={v2_attribution_pct:+.4f}%, "
            f"diff={diff:.4f}%"
        ),
        severity="info",  # informational, не fail
    ))

    # ── Sanity: failed_orders v1 (показатель health-проблем) ──
    if session.failed_orders > 0:
        results.append(ValidationResult(
            name="v1 had failed_orders",
            passed=False,
            detail=(
                f"v1 had {session.failed_orders} failed orders — в v2 их бы "
                f"закрыл SymbolHealthMonitor через 3 неуспеха"
            ),
            severity="warn",
        ))

    # ── Информативная статистика ──
    results.append(ValidationResult(
        name="signals stats",
        passed=True,
        detail=(
            f"real_signals={len(session.real_signals)}, "
            f"open={state.n_open}, close={state.n_close}, "
            f"quarantined={len(session.quarantined)}"
        ),
        severity="info",
    ))

    return ReplayValidationReport(
        session_path=session.session_path,
        exchange=session.exchange,
        initial_capital=session.initial_capital,
        v1_pnl_pct=session.pnl_pct,
        v2_attribution_pnl=v2_attribution_pnl,
        v2_attribution_pct=v2_attribution_pct,
        n_real_signals=len(session.real_signals),
        n_open=state.n_open,
        n_close=state.n_close,
        n_orphan_close=state.n_orphan_close,
        quarantined_count=len(session.quarantined),
        failed_orders=session.failed_orders,
        results=results,
    )


# ────────────────────────────────────────────────────────────────────
# Pretty-print отчёт
# ────────────────────────────────────────────────────────────────────


def render_report(report: ReplayValidationReport) -> str:
    """ASCII-отчёт для CLI."""
    sym_status = {True: "✓", False: "✗"}
    sym_sev = {"fail": "[FAIL]", "warn": "[WARN]", "info": "[INFO]"}
    out = []
    line = "═" * 78
    out.append(line)
    out.append(f"  Replay validation: {report.exchange}")
    out.append(f"  Session: {report.session_path}")
    out.append(line)
    out.append(f"  Initial capital:    ${report.initial_capital:.4f}")
    out.append(f"  v1 PnL%:            {report.v1_pnl_pct:+.4f}%")
    out.append(f"  v2 attribution PnL: ${report.v2_attribution_pnl:+.4f} "
               f"({report.v2_attribution_pct:+.4f}%)")
    out.append(f"  Real signals:       {report.n_real_signals}")
    out.append(f"  Open / Close / Orphan: "
               f"{report.n_open} / {report.n_close} / {report.n_orphan_close}")
    out.append(f"  v1 quarantined:     {report.quarantined_count}")
    out.append(f"  v1 failed orders:   {report.failed_orders}")
    out.append("")
    out.append("  Checks:")
    for r in report.results:
        marker = sym_status[r.passed]
        sev = sym_sev.get(r.severity, "")
        out.append(f"    {marker} {sev:6s} {r.name}")
        if r.detail:
            out.append(f"            {r.detail}")
    out.append("")
    if report.fail_count == 0 and report.warn_count == 0:
        out.append("  Overall: ✓ ALL CHECKS PASSED")
    elif report.fail_count == 0:
        out.append(f"  Overall: ✓ no FAILS, {report.warn_count} warnings")
    else:
        out.append(f"  Overall: ✗ {report.fail_count} FAILS, "
                   f"{report.warn_count} warnings")
    return "\n".join(out)
