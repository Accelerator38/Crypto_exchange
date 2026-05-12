"""Thin entrypoint для запуска Panteon v2 на Bitget.

Drop-in замена для Start_BITGET.py — те же команды, но pipeline через
panteon_v2/. Для live_futures требуются Bitget API ключи; исполнение идёт
через реальный v2 adapter поверх v1 futures client.

Использование:
    python Start_BITGET_v2.py
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

LAUNCH_MODE = os.getenv("BITGET_TRADING_MODE", "live_futures").strip().lower()

os.environ["CRYPTO_EXCHANGE"] = "BITGET"
os.environ["BITGET_TRADING_MODE"] = LAUNCH_MODE

if (
    LAUNCH_MODE != "paper"
    and not os.getenv("PANTEON_V2_ALLOW_SYSTEM_PYTHON")
    and Path(sys.prefix).resolve() == Path(sys.base_prefix).resolve()
):
    print(
        "[START v2] BITGET live launch blocked: run it with a virtualenv Python, "
        "not the system interpreter. Set PANTEON_V2_ALLOW_SYSTEM_PYTHON=1 to override.",
        file=sys.stderr,
    )
    sys.exit(4)

from panteon_v2.app.single_instance import acquire_single_instance

_INSTANCE_LOCK = acquire_single_instance(
    "panteon_v2_bitget",
    lock_dir=PROJECT_ROOT / "state" / "locks",
)
if _INSTANCE_LOCK is None:
    print(
        "[START v2] BITGET is already running; second live instance blocked.",
        file=sys.stderr,
    )
    sys.exit(3)

required = ("BITGET_API_KEY", "BITGET_SECRET_KEY", "BITGET_PASSPHRASE")
missing = [name for name in required if not os.getenv(name)]
mode_note = (
    "real futures trading requested; adapter/feed check follows" if LAUNCH_MODE == "live_futures"
    else "virtual portfolio; scoring enabled"
)
print(f"[START v2] BITGET launch mode: {LAUNCH_MODE.upper()} ({mode_note})")
if LAUNCH_MODE != "paper" and missing:
    print(
        "Bitget live launch is missing credentials: "
        + ", ".join(missing)
        + f"\nPut them in {PROJECT_ROOT}\\.env.local "
        + "or set BITGET_TRADING_MODE='paper'.",
        file=sys.stderr,
    )
    sys.exit(2)

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "v2_bitget_runtime.log"

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

SNAPSHOT_PATH = str(PROJECT_ROOT / "panteon_v2_state" / "bitget_snapshot.json")
EVENTS_JSONL  = str(PROJECT_ROOT / "logs" / "v2_bitget_events.jsonl")
_INITIAL_CAPITAL_RAW = os.getenv("BITGET_INITIAL_CAPITAL")
INITIAL_CAPITAL = float(_INITIAL_CAPITAL_RAW) if _INITIAL_CAPITAL_RAW else None

sys.exit(start_production(
    exchange="BITGET",
    mode=LAUNCH_MODE,
    initial_capital=INITIAL_CAPITAL,
    snapshot_path=SNAPSHOT_PATH,
    jsonl_event_log=EVENTS_JSONL,
    sleep_between_polls_sec=5.0,
))
