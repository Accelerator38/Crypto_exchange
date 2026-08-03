"""Build deterministic one-second research frames from one sealed session."""

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

from panteon_v2.data.bitget import materialize_frame_1s  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize frame_1s_v1 from one sealed Bitget session."
    )
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = materialize_frame_1s(
        session_dir=args.session_dir,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "database": str(summary.database_path),
                "manifest": str(summary.manifest_path),
                "database_sha256": summary.database_sha256,
                "manifest_sha256": summary.manifest_sha256,
                "frames": summary.frames,
                "complete_frames": summary.complete_frames,
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
