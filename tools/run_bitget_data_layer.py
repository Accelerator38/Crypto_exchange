"""Run the public, read-only Bitget market data collector."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from panteon_v2.data.bitget import (  # noqa: E402
    BitgetSegmentStore,
    BitgetDataCollectorLock,
    load_bitget_data_profile,
    recover_unsealed_sessions,
    run_public_collection,
)
from panteon_v2.data.bitget.segment_store import (  # noqa: E402
    compute_collector_fingerprint,
    current_git_revision,
)


DEFAULT_DATA_DIR = ROOT / "Retrodate" / "bitget_data_v1"
PROFILE_DIR = ROOT / "configs" / "bitget_data_profiles"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect public Bitget trade/books5/ticker events into sealed SQLite."
    )
    parser.add_argument(
        "--profile-id",
        default="bitget_full8_microstructure_v1",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Optional bounded technical run; omit for continuous collection.",
    )
    args = parser.parse_args(argv)
    with BitgetDataCollectorLock(args.data_dir):
        profile_path = PROFILE_DIR / f"{args.profile_id}.json"
        profile = load_bitget_data_profile(profile_path)
        recovered = recover_unsealed_sessions(args.data_dir)
        collector_files = (
            Path(__file__).resolve(),
            SRC / "panteon_v2" / "data" / "bitget" / "profile.py",
            SRC / "panteon_v2" / "data" / "bitget" / "rest_reconciler.py",
            SRC / "panteon_v2" / "data" / "bitget" / "segment_store.py",
            SRC / "panteon_v2" / "data" / "bitget" / "websocket_collector.py",
        )
        store = BitgetSegmentStore(
            data_dir=args.data_dir,
            profile=profile,
            source_revision=current_git_revision(ROOT),
            collector_fingerprint_sha256=compute_collector_fingerprint(
                collector_files
            ),
        )
        summary = asyncio.run(
            run_public_collection(
                profile=profile,
                store=store,
                duration_seconds=args.duration_seconds,
            )
        )
    summary["recovered_sessions"] = [
        {
            "session_id": item.session_id,
            "recovered_segments": item.recovered_segments,
            "manifest_sha256": item.manifest_sha256,
        }
        for item in recovered
    ]
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
