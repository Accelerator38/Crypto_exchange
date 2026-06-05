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

from panteon_v2.analysis.flash_live_admission_attribution import (  # noqa: E402
    DEFAULT_TARGET_ACTORS,
    write_flash_live_admission_attribution,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build live Flash admission attribution from causal_entry_decisions.jsonl."
    )
    parser.add_argument("run_dir")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--filename", default="flash_live_admission_attribution.json")
    parser.add_argument(
        "--markdown-filename",
        default="flash_live_admission_attribution.md",
    )
    parser.add_argument(
        "--target-actor",
        action="append",
        dest="target_actors",
        default=None,
    )
    parser.add_argument("--top-n", type=int, default=50)
    args = parser.parse_args(argv)

    json_path, md_path = write_flash_live_admission_attribution(
        args.run_dir,
        output_dir=args.output_dir,
        filename=args.filename,
        markdown_filename=args.markdown_filename,
        target_actors=tuple(args.target_actors or DEFAULT_TARGET_ACTORS),
        top_n=args.top_n,
    )
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
