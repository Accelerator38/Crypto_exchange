"""Thin entrypoint for running Panteon trading on MEXC."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = PROJECT_ROOT / "src" / "panteon_runtime"
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

from env_bootstrap import load_local_env


LAUNCH_MODE = "live_futures"  # paper | live_futures

load_local_env(PROJECT_ROOT)

os.environ["CRYPTO_EXCHANGE"] = "MEXC"
os.environ["MEXC_TRADING_MODE"] = LAUNCH_MODE

required = ("MEXC_API_KEY", "MEXC_SECRET_KEY")
mode = os.getenv("MEXC_TRADING_MODE", "live_futures").strip().lower()
mode_note = "real futures trading enabled" if mode == "live_futures" else "virtual portfolio; scoring enabled"
print(f"[START] MEXC launch mode: {mode.upper()} ({mode_note})")
missing = [name for name in required if not os.getenv(name)]
if mode != "paper" and missing:
    print(
        "MEXC live launch is missing credentials: "
        + ", ".join(missing)
        + f"\nPut them in {PROJECT_ROOT}\\.env.local or set LAUNCH_MODE='paper' in Start_MEXC.py.",
        file=sys.stderr,
    )
    sys.exit(2)

runpy.run_module("Panteon_Trade", run_name="__main__")
