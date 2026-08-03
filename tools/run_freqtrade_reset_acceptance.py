from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESET_ROOT = PROJECT_ROOT / "freqtrade_reset"
CONFIG = RESET_ROOT / "user_data" / "config.json"
DEFAULT_EXECUTABLE = PROJECT_ROOT / ".venv-freqtrade" / "Scripts" / "python.exe"
DEFAULT_REPORT = PROJECT_ROOT / "Reports" / "FreqtradeReset" / "acceptance_v1.json"


def _utf8_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def _freqtrade_command(executable: Path, *args: str) -> list[str]:
    return [str(executable), "-m", "freqtrade", *args]


def _run(command: list[str], timeout: float = 120.0) -> dict[str, object]:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=RESET_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=_utf8_environment(),
    )
    combined = completed.stdout + "\n" + completed.stderr
    fatal_markers = (
        "CRITICAL -",
        " ERROR -",
        "Configuration error:",
        "Traceback (most recent call last)",
    )
    passed = completed.returncode == 0 and not any(
        marker in combined for marker in fatal_markers
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "passed": passed,
    }


def _market_metadata_check(executable: Path) -> dict[str, object]:
    script = (
        "import ccxt,json;"
        "e=ccxt.bitget({'enableRateLimit':True,'options':{'defaultType':'swap'}});"
        "m=e.load_markets()['BTC/USDT:USDT'];"
        "print(json.dumps({'symbol':m.get('symbol'),'id':m.get('id'),"
        "'active':m.get('active'),'swap':m.get('swap'),'linear':m.get('linear'),"
        "'contract_size':m.get('contractSize'),'precision':m.get('precision'),"
        "'limits':m.get('limits')}))"
    )
    result = _run([str(executable), "-c", script], timeout=120.0)
    metadata: dict[str, object] = {}
    try:
        metadata = json.loads(str(result["stdout_tail"]).strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        result["passed"] = False
    amount_min = ((metadata.get("limits") or {}).get("amount") or {}).get("min")
    cost_min = ((metadata.get("limits") or {}).get("cost") or {}).get("min")
    precision = metadata.get("precision") or {}
    metadata_passed = (
        metadata.get("symbol") == "BTC/USDT:USDT"
        and metadata.get("active") is True
        and metadata.get("swap") is True
        and metadata.get("linear") is True
        and float(amount_min or 0.0) > 0.0
        and float(cost_min or 0.0) > 0.0
        and float(precision.get("amount") or 0.0) > 0.0
        and float(precision.get("price") or 0.0) > 0.0
    )
    result["market_metadata"] = metadata
    result["passed"] = bool(result["passed"] and metadata_passed)
    return result


def _pulse_contract_check(executable: Path) -> dict[str, object]:
    strategy_dir = str(RESET_ROOT / "user_data" / "strategies")
    script = (
        "import sys,json,pandas as pd;"
        f"sys.path.insert(0,{strategy_dir!r});"
        "from DeterministicPulseStrategy import DeterministicPulseStrategy;"
        "s=DeterministicPulseStrategy(config={'dry_run':True});s.bot_start();"
        "d=pd.DataFrame({'date':pd.to_datetime(["
        "'2026-08-03T00:00:00Z','2026-08-03T00:15:00Z'],utc=True),"
        "'volume':[1.0,1.0]});"
        "d=s.populate_entry_trend(d,{});d=s.populate_exit_trend(d,{});"
        "blocked=False;"
        "\ntry:\n DeterministicPulseStrategy(config={'dry_run':False}).bot_start()"
        "\nexcept Exception:\n blocked=True"
        "\nprint(json.dumps({'entries':int(d['enter_long'].sum()),"
        "'exits':int(d['exit_long'].sum()),'live_blocked':blocked}))"
    )
    result = _run([str(executable), "-c", script], timeout=30.0)
    contract: dict[str, object] = {}
    try:
        contract = json.loads(str(result["stdout_tail"]).strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        result["passed"] = False
    contract_passed = contract == {
        "entries": 1,
        "exits": 1,
        "live_blocked": True,
    }
    result["pulse_contract"] = contract
    result["passed"] = bool(result["passed"] and contract_passed)
    return result


def _trade_smoke(executable: Path, strategy: str, timeout_seconds: float) -> dict[str, object]:
    log_path = RESET_ROOT / "user_data" / "logs" / f"{strategy}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.unlink(missing_ok=True)
    command = _freqtrade_command(
        executable,
        "trade",
        "--config",
        str(CONFIG),
        "--userdir",
        str(RESET_ROOT / "user_data"),
        "--strategy",
        strategy,
        "--db-url",
        f"sqlite:///user_data/{strategy}.dryrun.sqlite",
        "--logfile",
        str(log_path),
    )
    process = subprocess.Popen(
        command,
        cwd=RESET_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=_utf8_environment(),
    )
    ready_markers = (
        "State changed to: RUNNING",
        "Changing state to: RUNNING",
        "Bot heartbeat",
    )
    fatal_markers = (
        "CRITICAL -",
        " ERROR -",
        "OperationalException:",
        "Traceback (most recent call last)",
    )
    deadline = time.monotonic() + timeout_seconds
    ready = False
    log_text = ""
    while time.monotonic() < deadline and process.poll() is None:
        time.sleep(1.0)
        if log_path.is_file():
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            if any(marker in log_text for marker in fatal_markers):
                break
            if any(marker in log_text for marker in ready_markers):
                ready = True
                break
    alive_until_stop = process.poll() is None
    if alive_until_stop:
        process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    if log_path.is_file():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    combined_tail = log_text[-6000:]
    full_output = log_text
    return {
        "command": command,
        "strategy": strategy,
        "alive_until_controlled_stop": alive_until_stop,
        "runtime_ready": ready,
        "server_side_stop_configured": "'stoploss_on_exchange': True" in log_text,
        "returncode_after_stop": process.returncode,
        "output_tail": combined_tail,
        "log_path": str(log_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "passed": (
            ready
            and "'stoploss_on_exchange': True" in log_text
            and not any(marker in full_output for marker in fatal_markers)
        ),
    }


def run_acceptance(
    *,
    executable: Path,
    report_path: Path,
    smoke_timeout_seconds: float,
) -> dict[str, object]:
    if not executable.is_file():
        raise FileNotFoundError(executable)
    checks = [
        _run(
            _freqtrade_command(
                executable,
                "list-strategies",
                "--config",
                str(CONFIG),
                "--userdir",
                str(RESET_ROOT / "user_data"),
            )
        ),
        _run(
            _freqtrade_command(
                executable,
                "show-config",
                "--config",
                str(CONFIG),
                "--userdir",
                str(RESET_ROOT / "user_data"),
            )
        ),
        _market_metadata_check(executable),
        _pulse_contract_check(executable),
    ]
    smokes = [
        _trade_smoke(executable, "NoTradeStrategy", smoke_timeout_seconds),
        _trade_smoke(
            executable,
            "DeterministicPulseStrategy",
            smoke_timeout_seconds,
        ),
    ]
    report: dict[str, object] = {
        "schema_version": "panteon.freqtrade_reset_acceptance.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "freqtrade_python": str(executable),
        "static_checks": checks,
        "trade_smokes": smokes,
        "passed": all(row["passed"] for row in checks + smokes),
        "orders_enabled": False,
        "promotion_authority": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Freqtrade dry-run acceptance.")
    parser.add_argument("--freqtrade-python", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--smoke-timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    report = run_acceptance(
        executable=args.freqtrade_python,
        report_path=args.report,
        smoke_timeout_seconds=args.smoke_timeout_seconds,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
