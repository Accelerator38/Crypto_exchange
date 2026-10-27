from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exia.genetic_primus_v8_clean.research import run_retrospective_shadow  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an engineering-only V8 Clean shadow on rejected historical V7 events"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--acknowledge-retrospective-only", action="store_true")
    args = parser.parse_args()
    result = run_retrospective_shadow(
        root=ROOT,
        output_dir=args.output_dir,
        retrospective_acknowledged=args.acknowledge_retrospective_only,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_dir": str(args.output_dir.resolve()),
                "comparison_rows": len(result["comparison"]),
                "orders_enabled": result["orders_enabled"],
                "promotion_authority": result["promotion_authority"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
