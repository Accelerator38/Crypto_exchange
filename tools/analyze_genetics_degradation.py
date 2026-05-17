from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from panteon_v2.analysis.genetics_degradation_monitoring import (  # noqa: E402
    build_genetics_degradation_report,
)


def _load_jsonl(path_value: str) -> list[dict]:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize genetics degradation/quarantine state from a v2 results directory.",
    )
    parser.add_argument("--results-dir", required=True, help="Directory containing status.json and leaderboard_agents.json.")
    parser.add_argument("--events-jsonl", default=None, help="Optional v2 event log JSONL for shadow blocked/rejected reasons.")
    parser.add_argument("--out", default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir
    status_path = results_dir / "status.json"
    leaderboard_path = results_dir / "leaderboard_agents.json"

    status = json.loads(status_path.read_text(encoding="utf-8"))
    leaderboard = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    shadow_events = _load_jsonl(args.events_jsonl) if args.events_jsonl else None
    report = build_genetics_degradation_report(
        status,
        leaderboard,
        shadow_events=shadow_events,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
