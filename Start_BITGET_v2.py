"""Thin entrypoint for launching Panteon v2 on Bitget."""

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


def _block_system_python_for_live(mode: str) -> None:
    if (
        mode != "paper"
        and not os.getenv("PANTEON_V2_ALLOW_SYSTEM_PYTHON")
        and Path(sys.prefix).resolve() == Path(sys.base_prefix).resolve()
    ):
        print(
            "[START v2] BITGET live launch blocked: run it with a virtualenv Python, "
            "not the system interpreter. Set PANTEON_V2_ALLOW_SYSTEM_PYTHON=1 to override.",
            file=sys.stderr,
        )
        sys.exit(4)


def _lock_owner_hint(lock_path: Path) -> str:
    info_path = lock_path.with_name(lock_path.name + ".info")
    try:
        data = {}
        source = info_path if info_path.exists() else lock_path
        with source.open("rb") as fh:
            text = fh.read(4096).decode("utf-8", errors="replace")
        for raw in text.splitlines():
            key, sep, value = raw.partition("=")
            if sep:
                data[key.strip()] = value.strip()
        pid = data.get("pid", "?")
        parent_pid = data.get("parent_pid", "?")
        executable = data.get("sys_executable", "?")
        return f"lock owner pid={pid}, parent_pid={parent_pid}, executable={executable}"
    except OSError:
        return f"lock file: {lock_path}"


def _acquire_instance_lock() -> None:
    from panteon_v2.app.single_instance import acquire_single_instance

    lock_path = PROJECT_ROOT / "state" / "locks" / "panteon_v2_bitget.lock"
    instance_lock = acquire_single_instance(
        "panteon_v2_bitget",
        lock_dir=lock_path.parent,
    )
    if instance_lock is None:
        print(
            "[START v2] BITGET is already running; second live instance blocked "
            f"({_lock_owner_hint(lock_path)}).",
            file=sys.stderr,
        )
        sys.exit(3)


def _configure_logging() -> Path:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "v2_bitget_runtime.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logging.getLogger(__name__).info("Runtime log: %s", log_file)
    logging.getLogger(__name__).info(
        "Process: pid=%s ppid=%s executable=%s",
        os.getpid(),
        os.getppid(),
        sys.executable,
    )
    if os.name == "nt":
        logging.getLogger(__name__).info(
            "Windows note: a tiny venv launcher parent process can appear next "
            "to the lock-owning trading process; the lock owner is the active bot."
        )
    return log_file


def main() -> int:
    _prepare_imports()
    _load_env()

    launch_mode = os.getenv("BITGET_TRADING_MODE", "live_futures").strip().lower()
    os.environ["CRYPTO_EXCHANGE"] = "BITGET"
    os.environ["BITGET_TRADING_MODE"] = launch_mode

    _block_system_python_for_live(launch_mode)
    _acquire_instance_lock()

    required = ("BITGET_API_KEY", "BITGET_SECRET_KEY", "BITGET_PASSPHRASE")
    missing = [name for name in required if not os.getenv(name)]
    mode_note = (
        "real futures trading requested; adapter/feed check follows"
        if launch_mode == "live_futures"
        else "virtual portfolio; scoring enabled"
    )
    print(f"[START v2] BITGET launch mode: {launch_mode.upper()} ({mode_note})")
    if launch_mode != "paper" and missing:
        print(
            "Bitget live launch is missing credentials: "
            + ", ".join(missing)
            + f"\nPut them in {PROJECT_ROOT}\\.env.local "
            + "or set BITGET_TRADING_MODE='paper'.",
            file=sys.stderr,
        )
        return 2

    _configure_logging()

    from panteon_v2.app.startup import start_production

    snapshot_path = str(PROJECT_ROOT / "panteon_v2_state" / "bitget_snapshot.json")
    events_jsonl = str(PROJECT_ROOT / "logs" / "v2_bitget_events.jsonl")
    initial_capital_raw = os.getenv("BITGET_INITIAL_CAPITAL")
    initial_capital = float(initial_capital_raw) if initial_capital_raw else None

    return start_production(
        exchange="BITGET",
        mode=launch_mode,
        initial_capital=initial_capital,
        snapshot_path=snapshot_path,
        jsonl_event_log=events_jsonl,
        sleep_between_polls_sec=5.0,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
