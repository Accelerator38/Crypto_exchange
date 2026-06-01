"""Thin entrypoint for launching Panteon Legend on Bitget."""

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
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _load_env() -> None:
    try:
        from env_bootstrap import load_local_env  # type: ignore

        load_local_env(PROJECT_ROOT)
    except ImportError:
        pass


def _configure_logging() -> None:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "legend_bitget_runtime.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def main() -> int:
    _prepare_imports()
    _load_env()

    launch_mode = os.getenv("BITGET_TRADING_MODE", "live_futures").strip().lower()
    os.environ["CRYPTO_EXCHANGE"] = "BITGET"
    os.environ["BITGET_TRADING_MODE"] = launch_mode
    os.environ["PANTEON_PROFILE"] = "panteon_legend_v2"

    required = ("BITGET_API_KEY", "BITGET_SECRET_KEY", "BITGET_PASSPHRASE")
    missing = [name for name in required if not os.getenv(name)]
    if launch_mode != "paper" and missing:
        print(
            "Bitget Legend live launch is missing credentials: "
            + ", ".join(missing)
            + f"\nPut them in {PROJECT_ROOT}\\.env.local "
            + "or set BITGET_TRADING_MODE='paper'.",
            file=sys.stderr,
        )
        return 2

    _configure_logging()

    from panteon_legend.startup import default_state_path, start_legend_production

    state_path = default_state_path(PROJECT_ROOT, "BITGET")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    events_jsonl = PROJECT_ROOT / "logs" / "legend_bitget_events.jsonl"
    initial_capital_raw = os.getenv("BITGET_INITIAL_CAPITAL")
    initial_capital = float(initial_capital_raw) if initial_capital_raw else None

    print(f"[START Legend] BITGET launch mode: {launch_mode.upper()}")
    return start_legend_production(
        exchange="BITGET",
        mode=launch_mode,
        initial_capital=initial_capital,
        snapshot_path=str(state_path),
        jsonl_event_log=str(events_jsonl),
        results_root=str(PROJECT_ROOT / "Results"),
        sleep_between_polls_sec=5.0,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
