from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_genetic_primus_v5_data_source import assert_evolution_authority


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATE = ROOT / "Reports/Exia/Genetic_Primus/v5_data_source_gate_20260824/report.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed Genetic_Primus V5 stage runner")
    parser.add_argument("--stage", choices=("development", "validation", "oos", "paper"), required=True)
    parser.add_argument("--gate-report", type=Path, default=DEFAULT_GATE)
    args = parser.parse_args()
    if args.stage == "development":
        assert_evolution_authority(args.gate_report.resolve())
        raise RuntimeError("V5 development implementation is unavailable until an accepted source manifest exists")
    raise PermissionError(f"V5 {args.stage} is blocked until development passes and a frozen stage authority exists")


if __name__ == "__main__":
    main()
