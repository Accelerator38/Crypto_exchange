from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RUNTIME = SRC / "panteon_runtime"
for path in (SRC, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from panteon_v2.analysis.portfolio_failure_decomposition import (  # noqa: E402
    build_portfolio_failure_decomposition,
    verify_development_replay_parity,
    write_portfolio_failure_decomposition,
)
from panteon_v2.analysis.strategy_lab import (  # noqa: E402
    aggregate_market_frames,
    evaluate_market_neutral_portfolio_development,
    load_aligned_hourly_market,
)
from panteon_v2.policy.development_candidate import (  # noqa: E402
    DevelopmentCandidateContract,
)
from panteon_v2.policy.experiment_registry import (  # noqa: E402
    ExperimentRegistryError,
    StrategyExperimentRegistry,
    sha256_file,
)
from panteon_v2.policy.strategy_candidate_registry import (  # noqa: E402
    StrategyCandidateRegistry,
)


CONTRACT_PATH = (
    ROOT / "configs" / "strategy_candidate_p5_market_neutral_portfolio_v1.json"
)
DATASET_REGISTRY_PATH = ROOT / "configs" / "strategy_candidates_v1.json"
EXPERIMENT_REGISTRY_PATH = (
    ROOT / "configs" / "strategy_experiment_registry_v1.json"
)
EXPERIMENT_ID = "weekly_top2_bottom2_relative_momentum_4h_v1"
SOURCE_REPORT_PATH = (
    ROOT
    / "Reports"
    / "StrategyLab"
    / "market_neutral_portfolio_development_20260729"
    / "market_neutral_portfolio_development.json"
)
REPORT_DIR = SOURCE_REPORT_PATH.parent


def main() -> int:
    experiment_registry = StrategyExperimentRegistry.from_json(
        EXPERIMENT_REGISTRY_PATH
    )
    experiment = experiment_registry.experiment(EXPERIMENT_ID)
    source_artifact = next(
        (
            row
            for row in experiment["artifacts"]
            if row["kind"] == "development_screen"
        ),
        None,
    )
    if source_artifact is None:
        raise ExperimentRegistryError("development screen is not registered")
    if (ROOT / source_artifact["path"]).resolve() != SOURCE_REPORT_PATH.resolve():
        raise ExperimentRegistryError("registered development report path mismatch")
    source_sha = sha256_file(SOURCE_REPORT_PATH)
    if source_sha != source_artifact["sha256"]:
        raise ExperimentRegistryError("registered development report SHA mismatch")
    sealed_report = json.loads(SOURCE_REPORT_PATH.read_text(encoding="utf-8"))

    contract = DevelopmentCandidateContract.from_json(CONTRACT_PATH)
    dataset_registry = StrategyCandidateRegistry.from_json(
        DATASET_REGISTRY_PATH
    )
    replay = evaluate_market_neutral_portfolio_development(
        contract.payload,
        dataset_registry,
        repository_root=ROOT,
    )
    replay_parity = verify_development_replay_parity(
        sealed_report,
        replay,
    )
    hourly_frames, _ = load_aligned_hourly_market(
        dataset_registry,
        repository_root=ROOT,
    )
    frames = aggregate_market_frames(
        hourly_frames,
        bars_per_frame=int(
            contract.payload["candidate"]["event_contract"]["aggregate_bars"]
        ),
    )
    report = build_portfolio_failure_decomposition(
        replay,
        frames=frames,
        source_report_sha256=source_sha,
        replay_parity=replay_parity,
    )
    json_path, markdown_path = write_portfolio_failure_decomposition(
        report,
        output_dir=REPORT_DIR,
    )
    print(
        json.dumps(
            {
                "candidate_id": report["candidate_id"],
                "source_report_sha256": source_sha,
                "replay_parity": report["replay_parity"],
                "diagnostic_flags": report["diagnostic_flags"],
                "market_exposure": report["market_exposure"],
                "concentration": report["concentration"],
                "drawdown": report["drawdown"],
                "report_sha256": sha256_file(json_path),
                "json": str(json_path.resolve()),
                "markdown": str(markdown_path.resolve()),
                "validation_opened": False,
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
