from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from simple_research.dataset import build_feature_tape


FULL8 = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT", "BNB/USDT", "LINK/USDT")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build canonical SimpleResearch tapes.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "Retrodate" / "simple_research_reset_v1",
    )
    args = parser.parse_args()
    definitions = (
        (
            PROJECT_ROOT / "Retrodate" / "bitget_futures_history_v3_2022_20260714",
            args.output_dir / "bitget_full8_1h.parquet",
            "1h",
            3_600_000,
        ),
        (
            PROJECT_ROOT / "Retrodate" / "bitget_1m_discovery_v1_20260615_20260728",
            args.output_dir / "bitget_full8_1m.parquet",
            "1m",
            60_000,
        ),
    )
    reports = [
        build_feature_tape(
            source_dir=source,
            output_path=output,
            timeframe=timeframe,
            timeframe_ms=timeframe_ms,
            expected_symbols=FULL8,
        )
        for source, output, timeframe, timeframe_ms in definitions
    ]
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
