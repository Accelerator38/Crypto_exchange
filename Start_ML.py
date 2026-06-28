"""Universal launcher for the ML entry-edge executor.

Spawns one independent process per exchange in ML_EXCHANGES (default
"BITGET,MEXC"). Each process holds its own single-instance lock, trades its own
account, and writes its own dashboard (Results/ML/dashboard_ML_<EX>.png), so the
venues never interfere.

Per-exchange trading mode comes from <EX>_TRADING_MODE (default "paper"); set it
to "live_futures" for real money. Shared ML_* env vars (model, threshold, sizing,
symbols) apply to both. Once MEXC returns 6026, futures OPEN is locally disabled
for the configured cooldown; it still scores, reconciles and renders dashboards.

Run it once; it stays alive until you close it or a lock-guarded supervisor
relaunches it. Real money requires ML_EXECUTOR_DRY_RUN=0 too.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
RUNTIME_DIR = SRC_DIR / "panteon_runtime"


def _prepare_imports() -> None:
    for path in (SRC_DIR, RUNTIME_DIR):
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)


def _load_env() -> None:
    try:
        from env_bootstrap import load_local_env  # type: ignore
        load_local_env(PROJECT_ROOT)
    except ImportError:
        pass


def _worker(exchange: str) -> int:
    _prepare_imports()
    _load_env()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    from panteon_v2.ml.runner import run_executor
    return run_executor(exchange)


def main() -> int:
    _prepare_imports()
    _load_env()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    log = logging.getLogger("ml_launcher")

    exchanges = [e.strip().upper() for e in os.getenv("ML_EXCHANGES", "BITGET,MEXC").split(",") if e.strip()]
    if not exchanges:
        log.error("ML_EXCHANGES is empty.")
        return 2
    log.info("Launching ML executor for: %s", ", ".join(exchanges))

    procs = []
    for ex in exchanges:
        p = multiprocessing.Process(target=_worker, args=(ex,), name=f"ml-{ex.lower()}", daemon=False)
        p.start()
        log.info("  started %s worker pid=%s", ex, p.pid)
        procs.append(p)

    for p in procs:
        p.join()
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
