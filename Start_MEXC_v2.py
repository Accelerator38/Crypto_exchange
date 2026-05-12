"""Thin entrypoint для запуска Panteon v2 на MEXC.

Drop-in замена для Start_MEXC.py — те же команды, но pipeline через
panteon_v2/. Для live_futures требуются MEXC API ключи; исполнение идёт
через реальный v2 adapter поверх v1 futures client.

Использование:
    python Start_MEXC_v2.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
RUNTIME_DIR = SRC_DIR / "panteon_runtime"

# v2 в src/, v1-агенты тоже в src/panteon_runtime/ (для register_all_v1_agents)
for p in (SRC_DIR, RUNTIME_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Загрузка .env.local (используем v1 env_bootstrap — он не требует
# самого Panteon v1, только парсит файл)
try:
    from env_bootstrap import load_local_env  # type: ignore
    load_local_env(PROJECT_ROOT)
except ImportError:
    pass

# Режим запуска (по аналогии с v1 Start_MEXC.py)
LAUNCH_MODE = os.getenv("MEXC_TRADING_MODE", "live_futures").strip().lower()

os.environ["CRYPTO_EXCHANGE"] = "MEXC"
os.environ["MEXC_TRADING_MODE"] = LAUNCH_MODE

# Sanity check ключей для live режима
required = ("MEXC_API_KEY", "MEXC_SECRET_KEY")
missing = [name for name in required if not os.getenv(name)]
mode_note = (
    "real futures trading requested; adapter/feed check follows" if LAUNCH_MODE == "live_futures"
    else "virtual portfolio; scoring enabled"
)
print(f"[START v2] MEXC launch mode: {LAUNCH_MODE.upper()} ({mode_note})")
if LAUNCH_MODE != "paper" and missing:
    print(
        "MEXC live launch is missing credentials: "
        + ", ".join(missing)
        + f"\nPut them in {PROJECT_ROOT}\\.env.local "
        + "or set MEXC_TRADING_MODE='paper'.",
        file=sys.stderr,
    )
    sys.exit(2)

# Логирование
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "v2_mexc_runtime.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logging.getLogger(__name__).info("Runtime log: %s", LOG_FILE)

# Запуск v2 production
from panteon_v2.app.startup import start_production

# Параметры запуска. Под каждый prod-deploy скорректируйте:
SNAPSHOT_PATH = str(PROJECT_ROOT / "panteon_v2_state" / "mexc_snapshot.json")
EVENTS_JSONL  = str(PROJECT_ROOT / "logs" / "v2_mexc_events.jsonl")
INITIAL_CAPITAL = float(os.getenv("MEXC_INITIAL_CAPITAL", "100.0"))

sys.exit(start_production(
    exchange="MEXC",
    mode=LAUNCH_MODE,
    initial_capital=INITIAL_CAPITAL,
    snapshot_path=SNAPSHOT_PATH,
    jsonl_event_log=EVENTS_JSONL,
    sleep_between_polls_sec=5.0,
))
