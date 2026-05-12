"""Operator CLI for production:
    python -m panteon_v2.app.cli --dryrun
    python -m panteon_v2.app.cli --replay /path/to/v1/session
    python -m panteon_v2.app.cli --migrate-v1 /path/to/v1/memory.json --to /path/to/v2/snapshot.json
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List

from ..dashboards import TextRenderer
from ..memory import PerformanceMemory
from ..replay import parse_session
from ..selection import AgentRegistry
from ..shadow.cli import _build_feed_from_session, _build_registry_from_session
from .bootstrap import build_dryrun_pipeline
from .main_loop import main_loop
from .migration import (
    migrate_from_v1_memory_file,
    save_v2_snapshot,
)


log = logging.getLogger(__name__)


def cmd_dryrun() -> int:
    """Запустить dry-run pipeline на пустом feed (smoke check)."""
    registry = AgentRegistry()
    # Без v1-агентов registry пустой — bootstrap не сработает.
    # Это нормально для smoke: проверяем что pipeline собирается.
    print("DRY-RUN: build_dryrun_pipeline check (registry=empty → должна быть ValueError)")
    try:
        _ = build_dryrun_pipeline(registry=registry, initial_capital=1000.0)
        print("UNEXPECTED: empty registry прошёл валидацию")
        return 1
    except ValueError as exc:
        print(f"OK — empty registry rejected: {exc}")
        return 0


def cmd_replay(session_path: str) -> int:
    """Прогнать v1-сессию через production pipeline (dry-run на FakeExchange)."""
    if not os.path.isdir(session_path):
        print(f"ERROR: session not found: {session_path}", file=sys.stderr)
        return 2
    session = parse_session(session_path, prefer_csv=True)
    registry = _build_registry_from_session(session)
    feed = _build_feed_from_session(session)
    pipeline = build_dryrun_pipeline(
        registry=registry,
        initial_capital=session.initial_capital or 1000.0,
        seed_quarantine=session.quarantined,
    )
    steps = main_loop(pipeline, feed)
    pipeline.ledger.replay_from_event_log(pipeline.event_log)
    main_data = pipeline.renderer.build_main(timeline_max=15)
    print(TextRenderer().render_main(main_data))
    print()
    print(f"Total bars processed: {len(steps)}")
    return 0


def cmd_migrate(v1_path: str, v2_path: str) -> int:
    """Migrate v1 memory file → v2 snapshot."""
    perf = PerformanceMemory()
    report = migrate_from_v1_memory_file(v1_path, perf)
    print(f"Migrated {report.n_labels} labels, {report.n_regime_pairs} pairs")
    if report.warnings:
        print("Warnings:")
        for w in report.warnings:
            print(f"  • {w}")
    save_v2_snapshot(perf, v2_path)
    print(f"Saved v2 snapshot: {v2_path}")
    return 0 if report.n_labels > 0 else 1


def main(argv: List[str] = None) -> int:
    p = argparse.ArgumentParser(description="Panteon v2 operator CLI")
    sub = p.add_mutually_exclusive_group()
    sub.add_argument("--dryrun", action="store_true",
                     help="smoke check: pipeline builds")
    sub.add_argument("--replay", metavar="SESSION_PATH",
                     help="прогнать v1-сессию через production pipeline (dry)")
    sub.add_argument("--migrate-v1", metavar="V1_MEMORY_JSON",
                     help="мигрировать v1 _regime_memory")
    p.add_argument("--to", metavar="V2_SNAPSHOT_JSON",
                   help="куда сохранить v2 snapshot (для --migrate-v1)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    if args.dryrun:
        return cmd_dryrun()
    if args.replay:
        return cmd_replay(args.replay)
    if args.migrate_v1:
        if not args.to:
            print("ERROR: --migrate-v1 requires --to PATH", file=sys.stderr)
            return 1
        return cmd_migrate(args.migrate_v1, args.to)
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
