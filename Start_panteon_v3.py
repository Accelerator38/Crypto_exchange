"""Unified Panteon v3 launcher.

Exchange switches live at the top of this file:
MEXC=ON
BITGET=ON

The parent process starts one hidden worker per exchange by re-invoking this
file with ``--worker EXCHANGE``. Workers call ``start_production`` directly;
the old ``Start_MEXC_v2.py`` and ``Start_BITGET_v2.py`` files are not used by
this launcher.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent

EXCHANGES: dict[str, str] = {
    "MEXC": "ON",
    "BITGET": "ON",
}

LOCK_NAMES: dict[str, str] = {
    "MEXC": "panteon_v2_mexc",
    "BITGET": "panteon_v2_bitget",
}

MODE_ENV: dict[str, str] = {
    "MEXC": "MEXC_TRADING_MODE",
    "BITGET": "BITGET_TRADING_MODE",
}

REQUIRED_CREDENTIALS: dict[str, tuple[str, ...]] = {
    "MEXC": ("MEXC_API_KEY", "MEXC_SECRET_KEY"),
    "BITGET": ("BITGET_API_KEY", "BITGET_SECRET_KEY", "BITGET_PASSPHRASE"),
}

INITIAL_CAPITAL_ENV: dict[str, str] = {
    "MEXC": "MEXC_INITIAL_CAPITAL",
    "BITGET": "BITGET_INITIAL_CAPITAL",
}

ON_VALUES = {"1", "ON", "TRUE", "YES", "Y"}
LOCK_OFFSET = 4096
SRC_DIR = PROJECT_ROOT / "src"
RUNTIME_DIR = SRC_DIR / "panteon_runtime"
GENETICS_SPECIALISTS_ENV = "PANTEON_V2_GENETICS_SPECIALISTS_MANIFEST"
DEFAULT_GENETICS_SPECIALISTS_MANIFEST = (
    PROJECT_ROOT
    / "Results"
    / "neiro_genetics"
    / "live_active_shadow_20260604"
    / "genetics_specialists_manifest.json"
)


@dataclass(frozen=True)
class LaunchResult:
    exchange: str
    status: str
    message: str
    pid: int | None = None


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


def _configure_default_genetics_manifests() -> None:
    if os.getenv(GENETICS_SPECIALISTS_ENV):
        return
    if DEFAULT_GENETICS_SPECIALISTS_MANIFEST.exists():
        os.environ[GENETICS_SPECIALISTS_ENV] = str(DEFAULT_GENETICS_SPECIALISTS_MANIFEST)


def _enabled_exchanges(config: Mapping[str, str] = EXCHANGES) -> tuple[str, ...]:
    enabled: list[str] = []
    for exchange, value in config.items():
        if str(value or "").strip().upper() in ON_VALUES:
            enabled.append(exchange)
    return tuple(enabled)


def _selected_exchanges(raw: str) -> tuple[str, ...]:
    if not raw:
        return _enabled_exchanges()
    requested = {
        part.strip().upper()
        for part in raw.replace(";", ",").split(",")
        if part.strip()
    }
    return tuple(exchange for exchange in _enabled_exchanges() if exchange in requested)


def _python_executable() -> str:
    if os.name == "nt":
        candidate = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = PROJECT_ROOT / ".venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def _read_pyvenv_base_executable(venv_root: Path) -> str | None:
    cfg_path = venv_root / "pyvenv.cfg"
    try:
        lines = cfg_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for raw in lines:
        key, sep, value = raw.partition("=")
        if sep and key.strip().lower() == "base-executable":
            candidate = Path(value.strip())
            if candidate.exists():
                return str(candidate)
    return None


def _windows_python_process_context(
    python_executable: str,
) -> tuple[str, dict[str, str]]:
    env: dict[str, str] = {}
    if os.name != "nt":
        return python_executable, env

    python_path = Path(python_executable)
    venv_root = python_path.parent.parent
    if python_path.parent.name.lower() != "scripts" or not (venv_root / "pyvenv.cfg").exists():
        return python_executable, env

    base_executable = _read_pyvenv_base_executable(venv_root)
    if not base_executable:
        return python_executable, env

    env["__PYVENV_LAUNCHER__"] = str(python_path)
    env["VIRTUAL_ENV"] = str(venv_root)
    env["PATH"] = str(python_path.parent) + os.pathsep + env.get("PATH", "")
    env.pop("PYTHONHOME", None)
    return base_executable, env


def _worker_args(exchange: str) -> tuple[str, str, str]:
    return ("Start_panteon_v3.py", "--worker", exchange)


def _lock_path(exchange: str) -> Path:
    lock_name = LOCK_NAMES[exchange]
    return PROJECT_ROOT / "state" / "locks" / f"{lock_name}.lock"


def _lock_is_held(path: Path) -> bool:
    if not path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+b")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(LOCK_OFFSET)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                return True
            handle.seek(LOCK_OFFSET)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            return False

        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False
    finally:
        handle.close()


def _lock_owner_hint(path: Path) -> str:
    info_path = path.with_name(path.name + ".info")
    source = info_path if info_path.exists() else path
    try:
        lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return str(path)
    data: dict[str, str] = {}
    for raw in lines:
        key, sep, value = raw.partition("=")
        if sep:
            data[key.strip()] = value.strip()
    pid = data.get("pid", "?")
    parent_pid = data.get("parent_pid", "?")
    executable = data.get("sys_executable", "?")
    return f"pid={pid}, parent_pid={parent_pid}, executable={executable}"


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _ps_array(values: Sequence[str]) -> str:
    return "@(" + ", ".join(_ps_quote(str(value)) for value in values) + ")"


def _spawn_child(exchange: str, python_executable: str) -> int:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"v3_{exchange.lower()}_launcher.out.log"
    err_path = log_dir / f"v3_{exchange.lower()}_launcher.err.log"
    worker_args = _worker_args(exchange)
    with out_path.open("a", encoding="utf-8") as out_handle:
        out_handle.write(
            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] launching {exchange} "
            f"via {' '.join(worker_args)}\n"
        )
    if os.name == "nt":
        return _spawn_child_windows(
            python_executable=python_executable,
            worker_args=worker_args,
            out_path=out_path,
            err_path=err_path,
        )

    out_handle = out_path.open("a", encoding="utf-8")
    err_handle = err_path.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [python_executable, *worker_args],
            cwd=str(PROJECT_ROOT),
            stdout=out_handle,
            stderr=err_handle,
        )
        return int(process.pid)
    finally:
        out_handle.close()
        err_handle.close()


def _spawn_child_windows(
    *,
    python_executable: str,
    worker_args: Sequence[str],
    out_path: Path,
    err_path: Path,
) -> int:
    launch_executable, launch_env = _windows_python_process_context(python_executable)
    try:
        err_path.touch(exist_ok=True)
    except OSError:
        pass
    env_script = ""
    if launch_env:
        env_script = "".join(
            f"$env:{name} = {_ps_quote(value)}; "
            for name, value in sorted(launch_env.items())
        )
        env_script += "Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue; "
    command = (
        env_script
        + "$p = Start-Process "
        f"-FilePath {_ps_quote(launch_executable)} "
        f"-ArgumentList {_ps_array(worker_args)} "
        f"-WorkingDirectory {_ps_quote(str(PROJECT_ROOT))} "
        "-WindowStyle Hidden -PassThru; "
        "Write-Output $p.Id"
    )
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Start-Process failed: "
            + (completed.stderr.strip() or completed.stdout.strip() or "unknown error")
        )
    for raw in reversed(completed.stdout.splitlines()):
        text = raw.strip()
        if text.isdigit():
            return int(text)
    raise RuntimeError("Start-Process did not return a child pid")


def _block_system_python_for_live(exchange: str, mode: str) -> None:
    if (
        mode != "paper"
        and not os.getenv("PANTEON_V2_ALLOW_SYSTEM_PYTHON")
        and Path(sys.prefix).resolve() == Path(sys.base_prefix).resolve()
    ):
        print(
            f"[START v3] {exchange} live launch blocked: run it with a virtualenv "
            "Python, not the system interpreter. Set "
            "PANTEON_V2_ALLOW_SYSTEM_PYTHON=1 to override.",
            file=sys.stderr,
        )
        sys.exit(4)


def _acquire_instance_lock(exchange: str) -> None:
    from panteon_v2.app.single_instance import acquire_single_instance

    lock_name = LOCK_NAMES[exchange]
    lock_path = _lock_path(exchange)
    instance_lock = acquire_single_instance(
        lock_name,
        lock_dir=lock_path.parent,
    )
    if instance_lock is None:
        print(
            f"[START v3] {exchange} is already running; second live instance "
            f"blocked ({_lock_owner_hint(lock_path)}).",
            file=sys.stderr,
        )
        sys.exit(3)


def _configure_logging(exchange: str) -> Path:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"v3_{exchange.lower()}_runtime.log"
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
            "Windows note: v3 starts the base interpreter with venv launch "
            "context to avoid a persistent venv launcher parent process."
        )
    return log_file


def _initial_capital(exchange: str) -> float | None:
    raw = os.getenv(INITIAL_CAPITAL_ENV[exchange])
    return float(raw) if raw else None


def _run_exchange_worker(exchange: str) -> int:
    exchange = exchange.upper()
    if exchange not in EXCHANGES:
        print(f"[START v3] unknown exchange worker: {exchange}", file=sys.stderr)
        return 2

    _prepare_imports()
    _load_env()
    _configure_default_genetics_manifests()

    mode_env = MODE_ENV[exchange]
    launch_mode = os.getenv(mode_env, "live_futures").strip().lower()
    os.environ["CRYPTO_EXCHANGE"] = exchange
    os.environ[mode_env] = launch_mode

    _block_system_python_for_live(exchange, launch_mode)
    _acquire_instance_lock(exchange)

    missing = [
        name for name in REQUIRED_CREDENTIALS[exchange] if not os.getenv(name)
    ]
    mode_note = (
        "real futures trading requested; adapter/feed check follows"
        if launch_mode == "live_futures"
        else "virtual portfolio; scoring enabled"
    )
    print(f"[START v3] {exchange} launch mode: {launch_mode.upper()} ({mode_note})")
    if launch_mode != "paper" and missing:
        print(
            f"{exchange} live launch is missing credentials: "
            + ", ".join(missing)
            + f"\nPut them in {PROJECT_ROOT}\\.env.local "
            + f"or set {mode_env}='paper'.",
            file=sys.stderr,
        )
        return 2

    _configure_logging(exchange)

    from panteon_v2.app.startup import start_production

    exchange_lower = exchange.lower()
    snapshot_path = str(
        PROJECT_ROOT / "panteon_v2_state" / f"{exchange_lower}_snapshot.json"
    )
    events_jsonl = str(PROJECT_ROOT / "logs" / f"v2_{exchange_lower}_events.jsonl")
    return start_production(
        exchange=exchange,
        mode=launch_mode,
        initial_capital=_initial_capital(exchange),
        snapshot_path=snapshot_path,
        jsonl_event_log=events_jsonl,
        sleep_between_polls_sec=5.0,
    )


def _launch_exchange(
    exchange: str,
    *,
    dry_run: bool = False,
    python_executable: str | None = None,
) -> LaunchResult:
    if exchange not in EXCHANGES:
        return LaunchResult(exchange, "disabled", "unknown exchange")

    lock_path = _lock_path(exchange)
    if _lock_is_held(lock_path):
        return LaunchResult(
            exchange,
            "already_running",
            f"{exchange} already running ({_lock_owner_hint(lock_path)})",
        )
    if dry_run:
        return LaunchResult(
            exchange,
            "dry_run",
            f"would launch {' '.join(_worker_args(exchange))}",
        )

    try:
        pid = _spawn_child(
            exchange,
            python_executable or _python_executable(),
        )
    except Exception as exc:
        return LaunchResult(exchange, "failed", str(exc))
    return LaunchResult(
        exchange,
        "launched",
        f"started {' '.join(_worker_args(exchange))}",
        pid,
    )


def _print_plan(selected: Sequence[str]) -> None:
    print("[START v3] exchange switches:")
    for exchange, value in EXCHANGES.items():
        marker = "selected" if exchange in selected else "off"
        print(f"[START v3] {exchange}={value} ({marker})")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start Panteon v3 trading processes.")
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated exchange filter, for example MEXC or MEXC,BITGET.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--worker",
        choices=tuple(EXCHANGES),
        default="",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)

    if args.worker:
        return _run_exchange_worker(str(args.worker).upper())

    selected = _selected_exchanges(args.only)
    _print_plan(selected)
    if not selected:
        print("[START v3] no exchanges selected")
        return 0

    failed = False
    for exchange in selected:
        result = _launch_exchange(exchange, dry_run=bool(args.dry_run))
        pid_text = f" pid={result.pid}" if result.pid is not None else ""
        print(f"[START v3] {exchange}: {result.status}{pid_text} - {result.message}")
        failed = failed or result.status == "failed"
    return 2 if failed else 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
