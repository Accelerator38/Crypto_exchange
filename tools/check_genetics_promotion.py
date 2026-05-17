from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from panteon_v2.analysis.genetics_promotion import evaluate_promotion_gate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether a genetics candidate passes promotion gates.")
    parser.add_argument("--report", required=True, help="Evaluation JSON with baseline and candidate genomes.")
    parser.add_argument("--out", default=None, help="Optional output JSON path.")
    parser.add_argument("--baseline-index", type=int, default=0)
    parser.add_argument("--candidate-index", type=int, default=-1)
    parser.add_argument("--mode", default="fee_fixed_nextbar")
    parser.add_argument("--min-robust-utility-delta", type=float, default=0.0)
    parser.add_argument("--max-turnover-rate", type=float, default=None)
    parser.add_argument("--max-saturation-rate", type=float, default=None)
    parser.add_argument("--max-invalid-open-logit-pressure", type=float, default=None)
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    report = json.loads(report_path.read_text(encoding="utf-8"))

    result = evaluate_promotion_gate(
        report,
        baseline_index=args.baseline_index,
        candidate_index=args.candidate_index,
        mode_name=args.mode,
        min_robust_utility_delta=args.min_robust_utility_delta,
        max_turnover_rate=args.max_turnover_rate,
        max_saturation_rate=args.max_saturation_rate,
        max_invalid_open_logit_pressure=args.max_invalid_open_logit_pressure,
    )

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
