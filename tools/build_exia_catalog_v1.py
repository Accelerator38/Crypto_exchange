from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from exia.catalog import build_catalog  # noqa: E402


def main() -> int:
    report = build_catalog(
        experiments_root=ROOT / "Reports" / "Exia" / "experiments",
        output_dir=ROOT / "Reports" / "Exia" / "catalog",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
