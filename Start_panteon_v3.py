"""Unified Panteon v3 launcher.

Exchange switches live at the top of this file:
MEXC=ON
BITGET=ON
"""

from __future__ import annotations

import argparse
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

START_SCRIPTS: dict[str, str] = {
    "MEXC": "Start_MEXC_v2.py",
    "BITGET": "Start_BITGET_v2.py",
}

LOCK_NAMES: dict[str, str] = {
    "MEXC": "panteon_v2_mexc",
    "BITGET": "panteon_v2_bitget",
}

ON_VALUES = {"1", "ON", "TRUE", "YES", "Y"}
LOCK_OFFSET = 4096


@dataclass(frozen=True)
class LaunchResult:
    exchange: str
    status: str
    message: str
    pid: int | None = None


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


def _spawn_child(exchange: str, script: Path, python_executable: str) -> int:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"v3_{exchange.lower()}_launcher.out.log"
    err_path = log_dir / f"v3_{exchange.lower()}_launcher.err.log"
    with out_path.open("a", encoding="utf-8") as out_handle:
        out_handle.write(
            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] launching {exchange} via {script.name}\n"
        )
    if os.name == "nt":
        return _spawn_child_windows(
            script=script,
            python_executable=python_executable,
            out_path=out_path,
            err_path=err_path,
        )

    out_handle = out_path.open("a", encoding="utf-8")
    err_handle = err_path.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [python_executable, str(script.relative_to(PROJECT_ROOT))],
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
    script: Path,
    python_executable: str,
    out_path: Path,
    err_path: Path,
) -> int:
    command = (
        "$p = Start-Process "
        f"-FilePath {_ps_quote(python_executable)} "
        f"-ArgumentList {_ps_quote(str(script.relative_to(PROJECT_ROOT)))} "
        f"-WorkingDirectory {_ps_quote(str(PROJECT_ROOT))} "
        f"-RedirectStandardOutput {_ps_quote(str(out_path))} "
        f"-RedirectStandardError {_ps_quote(str(err_path))} "
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


def _launch_exchange(
    exchange: str,
    *,
    dry_run: bool = False,
    python_executable: str | None = None,
) -> LaunchResult:
    script_name = START_SCRIPTS.get(exchange)
    if not script_name:
        return LaunchResult(exchange, "disabled", "no start script configured")
    script = PROJECT_ROOT / script_name
    if not script.exists():
        return LaunchResult(exchange, "failed", f"missing start script: {script}")

    lock_path = _lock_path(exchange)
    if _lock_is_held(lock_path):
        return LaunchResult(
            exchange,
            "already_running",
            f"{exchange} already running ({_lock_owner_hint(lock_path)})",
        )
    if dry_run:
        return LaunchResult(exchange, "dry_run", f"would launch {script.name}")

    pid = _spawn_child(
        exchange,
        script,
        python_executable or _python_executable(),
    )
    return LaunchResult(exchange, "launched", f"started {script.name}", pid)


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
    args = parser.parse_args(argv)

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
    raise SystemExit(main())
