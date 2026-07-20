"""Probe whether official Bitget history can form CarryFlow evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RUNTIME = SRC / "panteon_runtime"
for path in (SRC, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ccxt  # noqa: E402

from panteon_v2.policy.historical_evidence import (  # noqa: E402
    audit_bitget_historical_capabilities,
)


DEFAULT_OUTPUT = (
    ROOT / "Reports" / "CarryFlow" / "bitget_historical_capability_latest.json"
)


def _report_sha256(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("report_sha256", None)
    raw = json.dumps(
        canonical,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def run(args: argparse.Namespace, *, exchange: Any | None = None) -> dict[str, Any]:
    client = exchange or ccxt.bitget(
        {"enableRateLimit": True, "options": {"defaultType": "swap"}}
    )
    if exchange is None:
        client.load_markets()
    report = audit_bitget_historical_capabilities(
        client,
        symbol=str(args.symbol),
        timeframe=str(args.timeframe),
    )
    report["client"] = {
        "library": "ccxt",
        "version": str(getattr(ccxt, "__version__", "unknown")),
    }
    report["report_sha256"] = _report_sha256(report)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["output"] = str(output)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed audit of official Bitget historical sources required "
            "by CarryFlow. This tool never creates an evidence tape."
        )
    )
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["promotion_backfill_possible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
