"""Read-only Bitget Demo credential and account access check.

The command never creates, changes, or cancels an order.  Demo credentials are
kept separate from live credentials and every authenticated request carries
Bitget's mandatory ``paptrading: 1`` header.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
RUNTIME_DIR = SRC_DIR / "panteon_runtime"
for path in (SRC_DIR, RUNTIME_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load_env() -> None:
    try:
        from env_bootstrap import load_local_env

        load_local_env(PROJECT_ROOT)
    except ImportError:
        return


def run_check() -> dict[str, object]:
    _load_env()
    required = {
        "BITGET_DEMO_API_KEY": os.getenv("BITGET_DEMO_API_KEY", ""),
        "BITGET_DEMO_SECRET_KEY": os.getenv("BITGET_DEMO_SECRET_KEY", ""),
        "BITGET_DEMO_PASSPHRASE": os.getenv("BITGET_DEMO_PASSPHRASE", ""),
    }
    missing = tuple(name for name, value in required.items() if not value)
    if missing:
        return {
            "passed": False,
            "reason": "demo_credentials_missing",
            "missing": list(missing),
            "orders_sent": 0,
        }

    from bitget_api import BitgetDirectClient

    client = BitgetDirectClient(
        required["BITGET_DEMO_API_KEY"],
        required["BITGET_DEMO_SECRET_KEY"],
        required["BITGET_DEMO_PASSPHRASE"],
        demo=True,
    )
    demo_header = str(client.swap.headers.get("paptrading") or "") == "1"
    if not demo_header:
        return {
            "passed": False,
            "reason": "mandatory_demo_header_missing",
            "orders_sent": 0,
        }
    try:
        account = client.get_futures_account()
        positions = client.get_futures_positions()
    except Exception as exc:
        return {
            "passed": False,
            "reason": "demo_read_probe_failed",
            "error_type": type(exc).__name__,
            "orders_sent": 0,
        }
    return {
        "passed": True,
        "reason": "demo_read_access_confirmed",
        "demo_header": True,
        "equity": float(account.get("equity", 0.0) or 0.0),
        "available": float(account.get("available", 0.0) or 0.0),
        "open_positions": len(positions),
        "orders_sent": 0,
    }


def main() -> int:
    payload = run_check()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("passed") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
