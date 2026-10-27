from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exia.genetic_primus.feasibility_v7 import assert_evolution_authority  # noqa: E402


DEFAULT_EVENT_MANIFEST = ROOT / "Reports/Exia/Genetic_Primus/v7_event_feasibility_20260824/manifest.json"


def authorize_stage(stage: str, event_manifest: Path) -> None:
    if stage == "development":
        assert_evolution_authority(event_manifest)
        raise RuntimeError("V7 model-screen implementation is unavailable until event capacity passes")
    raise PermissionError(f"V7 {stage} is blocked until development passes and a frozen authority exists")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed Genetic_Primus V7 stage runner")
    parser.add_argument("--stage", choices=("development", "validation", "oos", "paper"), required=True)
    parser.add_argument("--event-manifest", type=Path, default=DEFAULT_EVENT_MANIFEST)
    args = parser.parse_args()
    authorize_stage(args.stage, args.event_manifest.resolve())


if __name__ == "__main__":
    main()
