"""Evaluate the immutable Bitget precursor campaign under data contract v2."""

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
    render_precursor_campaign_markdown,
    run_precursor_campaign_v2,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--research-root",
        default=str(ROOT / "Retrodate" / "bitget_data_v2" / "research"),
    )
    parser.add_argument(
        "--campaign-dir",
        default=str(
            ROOT / "Reports" / "BitgetData" / "precursor_campaign_v2"
        ),
    )
    args = parser.parse_args(argv)
    report = run_precursor_campaign_v2(
        research_root=args.research_root,
        campaign_dir=args.campaign_dir,
    )
    campaign_dir = Path(args.campaign_dir)
    campaign_dir.mkdir(parents=True, exist_ok=True)
    (campaign_dir / "latest_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (campaign_dir / "latest_report.md").write_text(
        render_precursor_campaign_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
