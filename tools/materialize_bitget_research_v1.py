"""Build the fixed Bitget minute feature and executable-label dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from panteon_v2.data.bitget import materialize_research_dataset  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build fixed one-minute features and costed forward labels from "
            "an immutable Bitget frame_1s_v1 dataset."
        )
    )
    parser.add_argument("--frame-dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    summary = materialize_research_dataset(
        frame_dataset_dir=args.frame_dataset_dir,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "database": str(summary.database_path),
                "database_sha256": summary.database_sha256,
                "manifest": str(summary.manifest_path),
                "manifest_sha256": summary.manifest_sha256,
                "bars": summary.bars,
                "eligible_bars": summary.eligible_bars,
                "labels": summary.labels,
                "eligible_labels": summary.eligible_labels,
                "research_only": True,
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
