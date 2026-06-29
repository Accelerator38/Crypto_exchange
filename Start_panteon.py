"""Unified Panteon launcher.

Edit these launch switches before running:

BITGET = "ON"
MEXC = "OFF"

trade_regime = "multi"
trade_regime = "singlton(GeneticsCore)"

``multi`` runs the normal Panteon/Flash actor selection. ``singlton(<actor>)``
allows only the named actor to send real orders; the rest of the runtime keeps
collecting shadow/statistical evidence where the production pipeline supports it.
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

BITGET = "ON"
MEXC = "OFF"
trade_regime = "multi"

EXCHANGES: dict[str, str] = {
    "BITGET": BITGET,
    "MEXC": MEXC,
}

LOCK_NAMES: dict[str, str] = {
    "BITGET": "panteon_v2_bitget",
    "MEXC": "panteon_v2_mexc",
}

MODE_ENV: dict[str, str] = {
    "BITGET": "BITGET_TRADING_MODE",
    "MEXC": "MEXC_TRADING_MODE",
}

REQUIRED_CREDENTIALS: dict[str, tuple[str, ...]] = {
    "BITGET": ("BITGET_API_KEY", "BITGET_SECRET_KEY", "BITGET_PASSPHRASE"),
    "MEXC": ("MEXC_API_KEY", "MEXC_SECRET_KEY"),
}

INITIAL_CAPITAL_ENV: dict[str, str] = {
    "BITGET": "BITGET_INITIAL_CAPITAL",
    "MEXC": "MEXC_INITIAL_CAPITAL",
}

ON_VALUES = {"1", "ON", "TRUE", "YES", "Y"}
VIRTUAL_TRADING_MODES = {"paper", "paper_live_feed", "shadow_live_feed"}
LOCK_OFFSET = 4096
SRC_DIR = PROJECT_ROOT / "src"
RUNTIME_DIR = SRC_DIR / "panteon_runtime"
ACTOR_ALIASES = {
    "carryflowagentv2": "CarryFlowAgentV2",
    "geneticbest": "GeneticsBest",
    "geneticcore": "GeneticsCore",
    "geneticrisktight": "GeneticsRiskTight",
    "geneticsbearish": "GeneticsBearish",
    "geneticsbest": "GeneticsBest",
    "geneticsbullish": "GeneticsBullish",
    "geneticscore": "GeneticsCore",
    "geneticsneutral": "GeneticsNeutral",
    "geneticsregimerouter": "GeneticsRegimeRouter",
    "geneticsrisktight": "GeneticsRiskTight",
    "livecrashhunter": "LiveCrashHunter",
    "liveoibreakout": "LiveOIBreakout",
    "livevolcompress": "LiveVolCompress",
    "momentumscalper": "MomentumScalper",
}


@dataclass(frozen=True)
class LaunchResult:
    exchange: str
    status: str
    message: str
    pid: int | None = None


@dataclass(frozen=True)
class TradeRegime:
    mode: str
    singleton_actor: str = ""


def _prepare_imports() -> None:
    for path in (SRC_DIR, RUNTIME_DIR):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _load_env() -> None:
    _prepare_imports()
    try:
        from env_bootstrap import load_local_env  # type: ignore

        load_local_env(PROJECT_ROOT)
    except ImportError:
        return


def _parse_trade_regime(raw: str | None = None) -> TradeRegime:
    text = str(trade_regime if raw is None else raw).strip()
    lower = text.lower()
    if lower == "multi":
        return TradeRegime("multi")
    if lower.startswith("singlton(") and text.endswith(")"):
        actor = text[text.find("(") + 1 : -1].strip()
        if not actor:
            raise ValueError("trade_regime singlton(...) requires an actor name")
        return TradeRegime("singlton", _canonical_actor_label(actor))
    raise ValueError("trade_regime must be 'multi' or 'singlton(<actor>)'")


def _normalization_key(text: str) -> str:
    return "".join(ch.lower() for ch in str(text or "") if ch.isalnum())


def _canonical_actor_label(actor_label: str) -> str:
    text = str(actor_label or "").strip()
    return ACTOR_ALIASES.get(_normalization_key(text), text)


def _enabled_exchanges(config: Mapping[str, str] = EXCHANGES) -> tuple[str, ...]:
    enabled: list[str] = []
    for exchange, value in config.items():
        if str(value or "").strip().upper() in ON_VALUES:
            enabled.append(str(exchange).upper())
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
    candidate = (
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
        if os.name == "nt"
        else PROJECT_ROOT / ".venv" / "bin" / "python"
    )
    return str(candidate) if candidate.exists() else sys.executable


def _worker_args(exchange: str) -> tuple[str, str, str]:
    return ("Start_panteon.py", "--worker", exchange.upper())


def _lock_path(exchange: str) -> Path:
    return PROJECT_ROOT / "state" / "locks" / f"{LOCK_NAMES[exchange.upper()]}.lock"


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
    return (
        f"pid={data.get('pid', '?')}, "
        f"parent_pid={data.get('parent_pid', '?')}, "
        f"executable={data.get('sys_executable', '?')}"
    )


def _spawn_child(exchange: str, python_executable: str) -> int:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"panteon_{exchange.lower()}_launcher.out.log"
    err_path = log_dir / f"panteon_{exchange.lower()}_launcher.err.log"
    with out_path.open("a", encoding="utf-8") as out_handle, err_path.open(
        "a",
        encoding="utf-8",
    ) as err_handle:
        out_handle.write(
            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] launching {exchange} "
            f"via {' '.join(_worker_args(exchange))}\n"
        )
        process = subprocess.Popen(
            [python_executable, *_worker_args(exchange)],
            cwd=str(PROJECT_ROOT),
            stdout=out_handle,
            stderr=err_handle,
        )
        return int(process.pid)


def _block_system_python_for_live(exchange: str, mode: str) -> None:
    if (
        mode not in VIRTUAL_TRADING_MODES
        and not os.getenv("PANTEON_V2_ALLOW_SYSTEM_PYTHON")
        and Path(sys.prefix).resolve() == Path(sys.base_prefix).resolve()
    ):
        print(
            f"[Start_panteon] {exchange} live launch blocked: use .venv Python "
            "or set PANTEON_V2_ALLOW_SYSTEM_PYTHON=1.",
            file=sys.stderr,
        )
        sys.exit(4)


def _acquire_instance_lock(exchange: str) -> None:
    from panteon_v2.app.single_instance import acquire_single_instance

    lock_path = _lock_path(exchange)
    instance_lock = acquire_single_instance(
        LOCK_NAMES[exchange],
        lock_dir=lock_path.parent,
    )
    if instance_lock is None:
        print(
            f"[Start_panteon] {exchange} is already running "
            f"({_lock_owner_hint(lock_path)}).",
            file=sys.stderr,
        )
        sys.exit(3)


def _configure_logging(exchange: str) -> None:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"panteon_{exchange.lower()}_runtime.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def _initial_capital(exchange: str) -> float | None:
    raw = os.getenv(INITIAL_CAPITAL_ENV[exchange])
    return float(raw) if raw else None


def _launch_mode(exchange: str) -> str:
    return os.getenv(MODE_ENV[exchange], "live_futures").strip().lower()


def _live_preflight_result(exchange: str, launch_mode: str):
    if launch_mode in VIRTUAL_TRADING_MODES:
        return None
    _prepare_imports()
    from panteon_v2.app.live_preflight import config_from_env, run_live_preflight

    return run_live_preflight(
        exchange,
        launch_mode,
        config=config_from_env(PROJECT_ROOT),
    )


def _base_actor_label(raw: str) -> str:
    text = str(raw or "").strip()
    if ":" in text:
        text = text.split(":", 1)[1].strip()
    changed = True
    while changed:
        changed = False
        for prefix in ("Solo_", "V_"):
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
                changed = True
    return text


def _single_actor_aliases(actor_label: str) -> tuple[str, ...]:
    base = _base_actor_label(actor_label)
    if not base:
        raise ValueError("singleton actor label is required")
    return tuple(dict.fromkeys((
        str(actor_label).strip(),
        base,
        f"agent:{base}",
        f"Solo_{base}",
        f"ensemble:Solo_{base}",
    )))


def _singleton_flash_config(exchange: str, actor_label: str):
    from dataclasses import replace
    from panteon_v2.app import startup

    base = startup._resolve_flash_allocator_config(exchange)
    aliases = _single_actor_aliases(actor_label)
    return replace(
        base,
        live_real_actor_whitelist=aliases,
        range_low_vol_real_actor_allowlist=aliases,
        promotion_derived_router_enabled=True,
        promotion_derived_actor_labels=(_base_actor_label(actor_label),),
        promotion_derived_dynamic_best_enabled=False,
        max_signals_per_actor=1,
    )


def _run_exchange_worker(exchange: str) -> int:
    exchange = exchange.upper()
    if exchange not in EXCHANGES:
        print(f"[Start_panteon] unknown exchange worker: {exchange}", file=sys.stderr)
        return 2

    _prepare_imports()
    _load_env()
    parsed_regime = _parse_trade_regime()

    launch_mode = _launch_mode(exchange)
    os.environ["CRYPTO_EXCHANGE"] = exchange
    os.environ[MODE_ENV[exchange]] = launch_mode

    _block_system_python_for_live(exchange, launch_mode)
    _acquire_instance_lock(exchange)

    missing = [
        name for name in REQUIRED_CREDENTIALS[exchange] if not os.getenv(name)
    ]
    if launch_mode not in VIRTUAL_TRADING_MODES and missing:
        print(
            f"{exchange} live launch is missing credentials: "
            + ", ".join(missing)
            + f"\nPut them in {PROJECT_ROOT}\\.env.local or set "
            + f"{MODE_ENV[exchange]}='paper'.",
            file=sys.stderr,
        )
        return 2

    _configure_logging(exchange)

    from panteon_v2.app.startup import start_production

    flash_config = None
    if parsed_regime.mode == "singlton":
        flash_config = _singleton_flash_config(exchange, parsed_regime.singleton_actor)

    exchange_lower = exchange.lower()
    print(
        f"[Start_panteon] {exchange} mode={launch_mode} "
        f"trade_regime={parsed_regime.mode}"
        + (
            f" actor={parsed_regime.singleton_actor}"
            if parsed_regime.singleton_actor
            else ""
        )
    )
    return start_production(
        exchange=exchange,
        mode=launch_mode,
        initial_capital=_initial_capital(exchange),
        snapshot_path=str(PROJECT_ROOT / "panteon_v2_state" / f"{exchange_lower}_snapshot.json"),
        jsonl_event_log=str(PROJECT_ROOT / "logs" / f"v2_{exchange_lower}_events.jsonl"),
        sleep_between_polls_sec=5.0,
        flash_allocator_config_override=flash_config,
    )


def _launch_exchange(
    exchange: str,
    *,
    dry_run: bool = False,
    python_executable: str | None = None,
) -> LaunchResult:
    exchange = exchange.upper()
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

    _load_env()
    launch_mode = _launch_mode(exchange)
    preflight = _live_preflight_result(exchange, launch_mode)
    if preflight is not None and not preflight.passed:
        reasons = ", ".join(preflight.reasons) or "unknown"
        return LaunchResult(
            exchange,
            "preflight_failed",
            f"live preflight failed: {reasons}",
        )

    try:
        pid = _spawn_child(exchange, python_executable or _python_executable())
    except Exception as exc:
        return LaunchResult(exchange, "failed", str(exc))
    return LaunchResult(
        exchange,
        "launched",
        f"started {' '.join(_worker_args(exchange))}",
        pid,
    )


def _print_plan(selected: Sequence[str]) -> None:
    parsed_regime = _parse_trade_regime()
    print("[Start_panteon] exchange switches:")
    for exchange, value in EXCHANGES.items():
        marker = "selected" if exchange in selected else "off"
        print(f"[Start_panteon] {exchange}={value} ({marker})")
    print(
        f"[Start_panteon] trade_regime={parsed_regime.mode}"
        + (
            f" actor={parsed_regime.singleton_actor}"
            if parsed_regime.singleton_actor
            else ""
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start Panteon trading processes.")
    parser.add_argument("--only", default="", help="Comma-separated exchange filter.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", choices=tuple(EXCHANGES), default="", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        return _run_exchange_worker(str(args.worker).upper())

    selected = _selected_exchanges(args.only)
    _print_plan(selected)
    if not selected:
        print("[Start_panteon] no exchanges selected")
        return 0

    failed = False
    for exchange in selected:
        result = _launch_exchange(exchange, dry_run=bool(args.dry_run))
        pid_text = f" pid={result.pid}" if result.pid is not None else ""
        print(f"[Start_panteon] {exchange}: {result.status}{pid_text} - {result.message}")
        failed = failed or result.status in {"failed", "preflight_failed"}
    return 2 if failed else 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
