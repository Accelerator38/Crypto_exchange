"""Generic entrypoint for running Panteon trading.

Uses CRYPTO_EXCHANGE from the environment, or MEXC when it is not set.
"""

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


DEFAULT_TRADING_MODES = {
    "BITGET": "live_futures",
    "MEXC": "live_futures",
}

load_local_env(PROJECT_ROOT)

exchange = (os.getenv("CRYPTO_EXCHANGE") or "MEXC").strip().upper()
mode = DEFAULT_TRADING_MODES.get(exchange, "live_futures").strip().lower()
os.environ["CRYPTO_EXCHANGE"] = exchange
os.environ[f"{exchange}_TRADING_MODE"] = mode

print(f"[START] DEFAULT launch exchange: {exchange} mode: {mode.upper()}")

runpy.run_module("Panteon_Trade", run_name="__main__")
