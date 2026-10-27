from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exia.genetic_primus_v8_clean.status import check_run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Check one Genetic Primus V8 clean evidence root")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = check_run(
        run_dir=args.run_dir,
        contract_path=ROOT / "configs/genetic_primus_v8_clean.json",
        root=ROOT,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["result"] != "SUCCESS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
