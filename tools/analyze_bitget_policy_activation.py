from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS = ROOT / "tools"
for path in (SRC, TOOLS):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from run_bitget_policy_canary import (  # noqa: E402
    DEFAULT_BLOCKED_REGIMES,
    DEFAULT_MANIFEST,
    DEFAULT_REPORTS_DIR,
    build_activation_gap_report,
    load_hypothesis,
    load_matrix_evidence,
    write_activation_gap_report,
)
from run_panteon3_single_component_canary import _parse_symbols  # noqa: E402


DEFAULT_MONITOR_REPORT = (
    DEFAULT_REPORTS_DIR / "latest_actionable_regime_monitor.json"
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build an activation-gap report from an existing Bitget policy monitor.",
    )
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--matrix-summary", required=True)
    parser.add_argument("--monitor-report", default=str(DEFAULT_MONITOR_REPORT))
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--symbols", default="")
    parser.add_argument(
        "--blocked-regime",
        action="append",
        default=[],
        help="Regime that blocks strict canary precheck; repeatable.",
    )
    args = parser.parse_args(argv)

    hypothesis = load_hypothesis(args.manifest, args.profile_id)
    profile_symbols = _parse_symbols(hypothesis.get("symbols"))  # type: ignore[arg-type]
    symbols = _parse_symbols(args.symbols) or profile_symbols
    matrix_evidence = load_matrix_evidence(
        args.profile_id,
        args.matrix_summary,
        require_pass=False,
    )
    monitor_payload = json.loads(Path(args.monitor_report).read_text(encoding="utf-8"))
    blocked_regimes = tuple(args.blocked_regime or DEFAULT_BLOCKED_REGIMES)
    report = build_activation_gap_report(
        profile_id=args.profile_id,
        hypothesis=hypothesis,
        symbols=symbols,
        matrix_evidence=matrix_evidence,
        monitor_payload=monitor_payload,
        blocked_regimes=blocked_regimes,
    )
    report_path = write_activation_gap_report(args.reports_dir, report)
    report["path"] = str(report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
