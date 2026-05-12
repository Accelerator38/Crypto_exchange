"""Generic v2 entrypoint, использует CRYPTO_EXCHANGE из env.

Drop-in замена Start_DEFAULT.py для v2.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
RUNTIME_DIR = SRC_DIR / "panteon_runtime"

for p in (SRC_DIR, RUNTIME_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from env_bootstrap import load_local_env  # type: ignore
    load_local_env(PROJECT_ROOT)
except ImportError:
    pass

DEFAULT_TRADING_MODES = {
    "BITGET": "live_futures",
    "MEXC":   "live_futures",
}

exchange = (os.getenv("CRYPTO_EXCHANGE") or "MEXC").strip().upper()
mode_env = (
    f"{exchange}_TRADING_MODE",
    "PANTEON_TRADING_MODE",
)
mode = None
for var in mode_env:
    if os.getenv(var):
        mode = os.getenv(var).strip().lower()
        break
if mode is None:
    mode = DEFAULT_TRADING_MODES.get(exchange, "live_futures").strip().lower()

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"v2_{exchange.lower()}_runtime.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logging.getLogger(__name__).info("Runtime log: %s", LOG_FILE)

from panteon_v2.app.startup import start_production

SNAPSHOT_PATH = str(PROJECT_ROOT / "panteon_v2_state" / f"{exchange.lower()}_snapshot.json")
EVENTS_JSONL  = str(PROJECT_ROOT / "logs" / f"v2_{exchange.lower()}_events.jsonl")
INITIAL_CAPITAL = float(os.getenv(f"{exchange}_INITIAL_CAPITAL", "100.0"))

sys.exit(start_production(
    exchange=exchange,
    mode=mode,
    initial_capital=INITIAL_CAPITAL,
    snapshot_path=SNAPSHOT_PATH,
    jsonl_event_log=EVENTS_JSONL,
    sleep_between_polls_sec=5.0,
))
