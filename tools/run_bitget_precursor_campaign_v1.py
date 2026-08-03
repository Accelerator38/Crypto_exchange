"""Run or update the immutable Bitget precursor evidence campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RUNTIME = SRC / "panteon_runtime"
for path in (ROOT, SRC, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from panteon_v2.data.bitget.precursor_campaign import (  # noqa: E402
    run_precursor_campaign,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze or update one multi-session Bitget precursor campaign. "
            "Existing sessions remain development; later sealed sessions are "
            "independent validation/OOS evidence."
        )
    )
    parser.add_argument("--research-root", required=True)
    parser.add_argument("--campaign-dir", required=True)
    args = parser.parse_args(argv)
    report = run_precursor_campaign(
        research_root=args.research_root,
        campaign_dir=args.campaign_dir,
    )
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "lock_sha256": report["campaign_lock"]["lock_sha256"],
                "contract_sha256": report["campaign_lock"]["contract_sha256"],
                "role_counts": report["evidence"]["role_counts"],
                "passed_candidates": report["passed_candidates"],
                "campaign_failures": report["campaign_failures"],
                "report_json": str(
                    Path(args.campaign_dir).resolve() / "latest_report.json"
                ),
                "report_md": str(
                    Path(args.campaign_dir).resolve() / "latest_report.md"
                ),
                "runtime_profile_created": False,
                "orders_enabled": False,
                "promotion_authority": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
