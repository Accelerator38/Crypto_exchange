"""CLI: запуск Phase-7 replay-валидации на конкретной v1-сессии.

Использование:
    cd src
    python -m panteon_v2.replay.cli /path/to/Results/MEXC/2026-05-05_*/
    python -m panteon_v2.replay.cli /path/to/session1/ /path/to/session2/
    python -m panteon_v2.replay.cli --latest MEXC BITGET   # самые свежие
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List

from .synthesizer import synthesize_v2_state
from .v1_parser import parse_session
from .validator import render_report, validate_session


def _find_latest(exchange: str, results_root: str = None) -> str:
    """Найти самую свежую сессию для биржи."""
    candidates = []
    roots = [results_root] if results_root else [
        "/sessions/loving-lucid-wozniak/mnt/Crypto_exchange/Results",
        "Results",
    ]
    for root in roots:
        path = os.path.join(root, exchange)
        if not os.path.isdir(path):
            continue
        for entry in sorted(os.listdir(path)):
            full = os.path.join(path, entry)
            if os.path.isdir(full) and entry.startswith("2026-"):
                candidates.append(full)
    if not candidates:
        raise FileNotFoundError(f"No sessions found for {exchange}")
    return candidates[-1]


def _run_one(session_path: str, *, render_dashboard: bool = False) -> int:
    """Запускает replay+validation для одной сессии. Возвращает exit code."""
    try:
        session = parse_session(session_path, prefer_csv=True)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    state = synthesize_v2_state(session)
    report = validate_session(session, state)
    print(render_report(report))

    if render_dashboard:
        # Опционально — рендерим v2-dashboard на основании replay state
        try:
            from ..dashboards import DashboardRenderer, TextRenderer
            renderer = DashboardRenderer(
                ledger=state.ledger,
                perf=state.perf,
                qm=state.qm,
                event_log=state.event_log,
            )
            main = renderer.build_main(timeline_max=15)
            print()
            print(TextRenderer().render_main(main))
        except Exception as exc:
            print(f"\n[dashboard failed: {exc}]", file=sys.stderr)

    return 0 if report.fail_count == 0 else 1


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 7 replay-validation: прогон v1-сессии через v2.",
    )
    parser.add_argument(
        "sessions", nargs="*",
        help="Пути к v1-сессиям (директории с status.json и leaderboard_*.json)",
    )
    parser.add_argument(
        "--latest", nargs="*", metavar="EXCHANGE",
        help="Найти самую свежую сессию для каждой биржи (MEXC, BITGET)",
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Дополнительно отрендерить v2-dashboard",
    )
    args = parser.parse_args(argv)

    paths: List[str] = list(args.sessions or [])
    if args.latest:
        for exchange in args.latest:
            try:
                paths.append(_find_latest(exchange.upper()))
            except FileNotFoundError as exc:
                print(f"WARN: {exc}", file=sys.stderr)

    if not paths:
        parser.print_help()
        return 1

    overall = 0
    for path in paths:
        # Если path содержит wildcard, разворачиваем
        expanded = glob.glob(path) if any(c in path for c in "*?[") else [path]
        for one in expanded:
            rc = _run_one(one, render_dashboard=args.dashboard)
            overall = max(overall, rc)
            print()
    return overall


if __name__ == "__main__":
    sys.exit(main())
