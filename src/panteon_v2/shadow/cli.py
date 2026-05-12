"""CLI: shadow-run на v1-сессии (replay-style).

Берёт уже завершённую v1-сессию (Results/.../), синтезирует
MarketSnapshot-ы из всех её сигналов и прогоняет через v2 ShadowRunner.
В конце сравнивает v1 решения с v2.

Использование:
    cd src
    python -m panteon_v2.shadow.cli --session /path/to/MEXC/2026-05-05_*/
    python -m panteon_v2.shadow.cli --latest MEXC BITGET
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

from ..domain.types import Action
from ..replay import parse_session
from ..replay.cli import _find_latest
from ..selection import AgentRegistry
from ..tests._helpers import FakeAgent
from .adapters import make_market_snapshot
from .comparator import compare_v1_vs_v2, render_comparison
from .feed import ReplayFeed
from .runner import ShadowRunner


def _build_registry_from_session(session) -> AgentRegistry:
    """Создаёт AgentRegistry с FakeAgent-обёртками всех label-ов из сессии.

    Так как мы не можем импортировать v1-агенты без зависимости, мы
    регистрируем псевдоагенты, которые на запросе vote() возвращают то,
    что v1 реально делал (best-effort из all_signals.csv).
    """
    registry = AgentRegistry()
    seen = set()
    # Из leaderboard agents
    for label in session.agents:
        clean = label.replace("V_", "")
        if clean in seen:
            continue
        seen.add(clean)
        registry.register(FakeAgent(clean))
    # Также добавим агентов из selected_agents
    for sig in session.real_signals:
        clean = (sig.selected_agents or "").replace("V_", "")
        if clean and clean not in seen:
            seen.add(clean)
            registry.register(FakeAgent(clean))
    return registry


def _build_feed_from_session(session) -> ReplayFeed:
    """Группирует сигналы по бару, для каждого бара собирает MarketSnapshot."""
    by_bar: Dict[int, Dict[str, float]] = {}
    regime_by_bar: Dict[int, str] = {}
    bars_order: List[int] = []
    for sig in session.real_signals:
        bar = sig.bar
        if bar not in by_bar:
            by_bar[bar] = {}
            bars_order.append(bar)
        if sig.price > 0:
            by_bar[bar][sig.sym] = sig.price
        if sig.regime:
            regime_by_bar[bar] = sig.regime
    bars_order.sort()
    feed = ReplayFeed()
    for bar in bars_order:
        prices = by_bar[bar]
        if not prices:
            continue
        feed.append(make_market_snapshot(
            bar=bar,
            prices=prices,
            regime=regime_by_bar.get(bar) or session.current_regime,
        ))
    return feed


def _run_one(session_path: str) -> int:
    """Загружаем v1-сессию, прогоняем через v2 ShadowRunner, сравниваем."""
    try:
        session = parse_session(session_path, prefer_csv=True)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    registry = _build_registry_from_session(session)
    feed = _build_feed_from_session(session)

    runner = ShadowRunner.with_defaults(
        registry=registry,
        feed=feed,
        seed_quarantine=session.quarantined,
        balance_usd=session.initial_capital or 1000.0,
    )

    # Прогоняем
    steps = runner.run_until_exhausted()

    # Сравнение
    report = compare_v1_vs_v2(
        session=session,
        v2_steps=steps,
        v2_qm=runner.qm,
        v2_ledger=runner.ledger,
    )
    print(render_comparison(report))
    return 0


def main(argv: List[str] = None) -> int:
    p = argparse.ArgumentParser(
        description="Phase 8 shadow-run: v2 dry-run на v1-сессии.",
    )
    p.add_argument("--session", action="append", default=[],
                   help="Путь к v1-сессии (можно несколько)")
    p.add_argument("--latest", nargs="*", metavar="EXCHANGE",
                   help="Найти самые свежие сессии (MEXC, BITGET)")
    args = p.parse_args(argv)

    paths = list(args.session)
    if args.latest:
        for ex in args.latest:
            try:
                paths.append(_find_latest(ex.upper()))
            except FileNotFoundError as exc:
                print(f"WARN: {exc}", file=sys.stderr)
    if not paths:
        p.print_help()
        return 1

    overall = 0
    for path in paths:
        rc = _run_one(path)
        overall = max(overall, rc)
        print()
    return overall


if __name__ == "__main__":
    sys.exit(main())
