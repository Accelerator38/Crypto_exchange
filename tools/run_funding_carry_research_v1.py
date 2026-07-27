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

from panteon_v2.analysis.bitget_funding_history import (  # noqa: E402
    build_or_verify_funding_snapshot,
)
from panteon_v2.analysis.strategy_lab import (  # noqa: E402
    evaluate_funding_recent_screen,
    write_funding_recent_screen_report,
)
from panteon_v2.policy.strategy_candidate_registry import (  # noqa: E402
    StrategyCandidateRegistry,
)


REGISTRY_PATH = ROOT / "configs" / "strategy_candidates_v1.json"
SNAPSHOT_DIR = (
    ROOT / "Retrodate" / "bitget_funding_history_v1_20260727"
)
REPORT_DIR = (
    ROOT / "Reports" / "StrategyLab" / "funding_carry_recent_screen_20260727"
)


def main() -> int:
    registry = StrategyCandidateRegistry.from_json(REGISTRY_PATH)
    dataset = registry.payload["dataset"]
    funding_manifest = build_or_verify_funding_snapshot(
        SNAPSHOT_DIR,
        symbols=dataset["symbols"],
        required_start_at=dataset["start_at"],
        required_end_at=dataset["end_at"],
    )
    report = evaluate_funding_recent_screen(
        registry,
        repository_root=ROOT,
        funding_snapshot_dir=SNAPSHOT_DIR,
    )
    json_path, markdown_path = write_funding_recent_screen_report(
        report,
        output_dir=REPORT_DIR,
    )
    payload = {
        "candidate_id": report["candidate_id"],
        "verdict": report["verdict"],
        "continuation_allowed": report["continuation_allowed"],
        "early_rejection": report["early_rejection"],
        "funding_dataset_sha256": funding_manifest["dataset_sha256"],
        "funding_manifest_sha256": funding_manifest["manifest_sha256"],
        "report_sha256": _sha256_file(json_path),
        "observed_common_window": funding_manifest["observed_common_window"],
        "registered_oos_coverage_complete": funding_manifest[
            "registered_oos_coverage_complete"
        ],
        "screen": {
            "candidate_signals": report["screen"]["candidate_signals"],
            "filled_orders": report["screen"]["filled_orders"],
            "closed_trades": report["screen"]["closed_trades"],
            "mean_net_bps": report["screen"]["metrics"]["mean_net_bps"],
            "lcb_95_net_bps": report["screen"]["metrics"]["lcb_95_net_bps"],
        },
        "activation_funnel": report["activation_funnel"],
        "json": str(json_path.resolve()),
        "markdown": str(markdown_path.resolve()),
        "orders_enabled": False,
        "promotion_authority": False,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
