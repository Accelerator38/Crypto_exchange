from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RUNTIME = SRC / "panteon_runtime"
for path in (SRC, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from panteon_v2.analysis.strategy_lab import (  # noqa: E402
    evaluate_market_neutral_portfolio_development,
    write_market_neutral_portfolio_development_report,
)
from panteon_v2.policy.development_candidate import (  # noqa: E402
    DevelopmentCandidateContract,
)
from panteon_v2.policy.strategy_candidate_registry import (  # noqa: E402
    StrategyCandidateRegistry,
)


CONTRACT_PATH = (
    ROOT / "configs" / "strategy_candidate_p5_market_neutral_portfolio_v1.json"
)
DATASET_REGISTRY_PATH = ROOT / "configs" / "strategy_candidates_v1.json"
REPORT_DIR = (
    ROOT
    / "Reports"
    / "StrategyLab"
    / "market_neutral_portfolio_development_20260729"
)


def main() -> int:
    contract = DevelopmentCandidateContract.from_json(CONTRACT_PATH)
    registry = StrategyCandidateRegistry.from_json(DATASET_REGISTRY_PATH)
    report = evaluate_market_neutral_portfolio_development(
        contract.payload,
        registry,
        repository_root=ROOT,
    )
    json_path, markdown_path = write_market_neutral_portfolio_development_report(
        report,
        output_dir=REPORT_DIR,
    )
    print(
        json.dumps(
            {
                "candidate_id": report["candidate_id"],
                "profile_sha256": report["profile_sha256"],
                "stage": report["stage"],
                "sealed_windows": report["sealed_windows"],
                "verdict": report["verdict"],
                "passed": report["passed"],
                "failures": report["failures"],
                "candidate": {
                    "signals": report["candidate"]["candidate_signals"],
                    "filled_orders": report["candidate"]["filled_orders"],
                    "closed_portfolios": report["candidate"][
                        "closed_portfolios"
                    ],
                    "mean_net_bps": report["candidate"]["metrics"][
                        "mean_net_bps"
                    ],
                    "lcb_95_net_bps": report["candidate"]["metrics"][
                        "lcb_95_net_bps"
                    ],
                },
                "baseline_comparison": report["baseline_comparison"],
                "report_sha256": _sha256_file(json_path),
                "json": str(json_path.resolve()),
                "markdown": str(markdown_path.resolve()),
                "orders_enabled": False,
                "promotion_authority": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
