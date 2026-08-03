from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "freqtrade_pilot"
USER_DIR = PILOT / "user_data"
CONFIG_PATH = USER_DIR / "config.dryrun.json"
STRATEGY_DIR = USER_DIR / "strategies"
STRATEGY_NAME = "UpstreamSampleDryRunStrategy"
LOCK_PATH = PILOT / "upstream.lock.json"
BACKTEST_DIR = USER_DIR / "backtest_results"
REPORT_PATH = ROOT / "Reports" / "FreqtradePilot" / "pilot_report_v1.json"


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _freqtrade_command() -> list[str]:
    executable = ROOT / ".venv-freqtrade" / "Scripts" / "freqtrade.exe"
    if not executable.exists():
        raise RuntimeError(f"Freqtrade executable is missing: {executable}")
    return [str(executable)]


def _run(command: list[str], timeout: int = 600) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-30:])
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}\n{tail}")
    return result


def _base_args() -> list[str]:
    return [
        "--config",
        str(CONFIG_PATH),
        "--userdir",
        str(USER_DIR),
        "--strategy-path",
        str(STRATEGY_DIR),
        "--strategy",
        STRATEGY_NAME,
    ]


def _lookup_args() -> list[str]:
    return [
        "--config",
        str(CONFIG_PATH),
        "--userdir",
        str(USER_DIR),
        "--strategy-path",
        str(STRATEGY_DIR),
    ]


def _validate_upstream() -> dict[str, Any]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    actual_commit = _run(
        ["git", "-C", str(ROOT / lock["submodule_path"]), "rev-parse", "HEAD"]
    ).stdout.strip()
    version_text = _run(_freqtrade_command() + ["--version"]).stdout.strip()
    return {
        **lock,
        "actual_commit": actual_commit,
        "version_output": version_text,
        "commit_match": actual_commit == lock["commit"],
        "version_match": lock["package_version"] in version_text,
    }


def _validate_safety(config: dict[str, Any]) -> dict[str, Any]:
    exchange = config.get("exchange", {})
    checks = {
        "dry_run": config.get("dry_run") is True,
        "futures": config.get("trading_mode") == "futures",
        "isolated_margin": config.get("margin_mode") == "isolated",
        "force_entry_disabled": config.get("force_entry_enable") is False,
        "api_disabled": config.get("api_server", {}).get("enabled") is False,
        "telegram_disabled": config.get("telegram", {}).get("enabled") is False,
        "credentials_empty": not any(
            exchange.get(key) for key in ("key", "secret", "password")
        ),
        "bitget": exchange.get("name") == "bitget",
        "full8": len(exchange.get("pair_whitelist", [])) == 8,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "orders_enabled": False,
        "promotion_authority": False,
    }


def _download_data(days: int) -> None:
    _run(
        _freqtrade_command()
        + [
            "download-data",
            "--config",
            str(CONFIG_PATH),
            "--userdir",
            str(USER_DIR),
            "--days",
            str(days),
            "--timeframes",
            "5m",
            "--trading-mode",
            "futures",
            "--data-format-ohlcv",
            "feather",
        ],
        timeout=1800,
    )


def _audit_data(config: dict[str, Any]) -> dict[str, Any]:
    data_dir = USER_DIR / "data" / "bitget" / "futures"
    pairs: dict[str, Any] = {}
    for pair in config["exchange"]["pair_whitelist"]:
        symbol = pair.replace("/", "_").replace(":", "_")
        candle_path = data_dir / f"{symbol}-5m-futures.feather"
        funding_path = data_dir / f"{symbol}-1h-funding_rate.feather"
        if not candle_path.exists():
            pairs[pair] = {"present": False}
            continue

        candles = pd.read_feather(candle_path).sort_values("date")
        dates = pd.DatetimeIndex(candles["date"]).drop_duplicates()
        expected_rows = int((dates[-1] - dates[0]).total_seconds() // 300) + 1
        missing_rows = max(expected_rows - len(dates), 0)
        funding = pd.read_feather(funding_path) if funding_path.exists() else None
        pairs[pair] = {
            "present": True,
            "rows": len(candles),
            "unique_rows": len(dates),
            "start": dates[0].isoformat(),
            "end": dates[-1].isoformat(),
            "expected_rows": expected_rows,
            "missing_rows": missing_rows,
            "coverage": len(dates) / expected_rows if expected_rows else 0.0,
            "duplicate_rows": len(candles) - len(dates),
            "funding_rows": 0 if funding is None else len(funding),
            "funding_start": None
            if funding is None or funding.empty
            else pd.Timestamp(funding["date"].min()).isoformat(),
            "funding_end": None
            if funding is None or funding.empty
            else pd.Timestamp(funding["date"].max()).isoformat(),
        }

    present = [item for item in pairs.values() if item.get("present")]
    return {
        "pairs": pairs,
        "all_pairs_present": len(present) == len(pairs),
        "minimum_ohlcv_coverage": min((item["coverage"] for item in present), default=0),
        "funding_history_complete_for_60d": all(
            item.get("funding_rows", 0) >= 180 for item in present
        ),
    }


def _read_backtest(path: Path, fee: float) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        result_name = next(
            name
            for name in archive.namelist()
            if name.endswith(".json")
            and not name.endswith(".meta.json")
            and "config" not in name
        )
        payload = json.loads(archive.read(result_name))
    result = payload["strategy"][STRATEGY_NAME]
    fields = (
        "backtest_start",
        "backtest_end",
        "total_trades",
        "trade_count_long",
        "trade_count_short",
        "wins",
        "losses",
        "winrate",
        "profit_total_abs",
        "profit_total",
        "profit_mean",
        "profit_factor",
        "expectancy",
        "expectancy_ratio",
        "sharpe",
        "max_drawdown_abs",
        "max_drawdown_account",
        "rejected_signals",
    )
    summary = {field: result.get(field) for field in fields}
    summary.update(
        {
            "fee_per_fill": fee,
            "artifact": str(path.relative_to(ROOT)),
            "gate_pass": result["total_trades"] >= 20
            and result["expectancy"] > 0
            and result["profit_factor"] > 1
            and result["max_drawdown_account"] <= 0.20,
        }
    )
    return summary


def _run_backtest(fee: float) -> dict[str, Any]:
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    before = set(BACKTEST_DIR.glob("*.zip"))
    _run(
        _freqtrade_command()
        + [
            "backtesting",
            *_base_args(),
            "--timeframe",
            "5m",
            "--fee",
            str(fee),
            "--export",
            "trades",
            "--backtest-directory",
            str(BACKTEST_DIR),
        ],
        timeout=1800,
    )
    created = set(BACKTEST_DIR.glob("*.zip")) - before
    if not created:
        raise RuntimeError("Backtest completed without a new result archive")
    return _read_backtest(max(created, key=lambda item: item.stat().st_mtime), fee)


def _live_guard_probe() -> dict[str, Any]:
    sys.path.insert(0, str(STRATEGY_DIR))
    from UpstreamSampleDryRunStrategy import UpstreamSampleDryRunStrategy

    strategy = UpstreamSampleDryRunStrategy({"dry_run": False})
    try:
        strategy.bot_start()
    except RuntimeError as exc:
        return {"pass": True, "exception": str(exc)}
    return {"pass": False, "exception": None}


def _startup_smoke(timeout_seconds: int) -> dict[str, Any]:
    logs = USER_DIR / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    trade_log = logs / "pilot.log"
    stdout_path = logs / "pilot.stdout.log"
    stderr_path = logs / "pilot.stderr.log"
    database = USER_DIR / "pilot.dryrun.sqlite"
    leverage_cache = USER_DIR / "data" / "bitget" / "futures" / "leverage_tiers_USDT.json"
    for path in (trade_log, stdout_path, stderr_path, database):
        path.unlink(missing_ok=True)
    if leverage_cache.exists() and leverage_cache.stat().st_size == 0:
        leverage_cache.unlink()

    command = _freqtrade_command() + [
        "trade",
        *_base_args(),
        "--db-url",
        f"sqlite:///{database.as_posix()}",
        "--logfile",
        str(trade_log),
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=_environment(),
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            creationflags=creationflags,
        )
        ready = False
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline and process.poll() is None:
            time.sleep(1)
            content = trade_log.read_text(encoding="utf-8", errors="replace") if trade_log.exists() else ""
            if "Changing state to: RUNNING" in content:
                ready = True
                break
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    content = trade_log.read_text(encoding="utf-8", errors="replace") if trade_log.exists() else ""
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    return {
        "pass": ready,
        "dry_run_confirmed": "Dry run is enabled" in content
        and "Instance is running with dry_run enabled" in content,
        "running_state_reached": ready,
        "exit_code_after_controlled_stop": process.returncode,
        "error_tail": "\n".join(stderr.splitlines()[-20:]) if not ready else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the fail-closed Freqtrade Bitget pilot")
    parser.add_argument("--download-days", type=int, default=0)
    parser.add_argument("--startup-timeout", type=int, default=90)
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    upstream = _validate_upstream()
    safety = _validate_safety(config)
    if not upstream["commit_match"] or not upstream["version_match"]:
        raise RuntimeError("Pinned Freqtrade revision/version mismatch")
    if not safety["pass"]:
        raise RuntimeError("Pilot safety configuration failed closed")

    strategy_listing = _run(
        _freqtrade_command() + ["list-strategies", *_lookup_args(), "--one-column"]
    ).stdout.splitlines()
    if STRATEGY_NAME not in strategy_listing:
        raise RuntimeError(f"Strategy resolver did not find {STRATEGY_NAME}")
    if args.download_days > 0:
        _download_data(args.download_days)
    data_audit = _audit_data(config)
    base = _run_backtest(0.0004)
    stress = _run_backtest(0.0008)
    guard = _live_guard_probe()
    startup = _startup_smoke(args.startup_timeout)

    reasons = ["upstream_sample_strategy_is_not_a_production_strategy"]
    if base["expectancy"] <= 0:
        reasons.append("negative_base_costed_expectancy")
    if stress["expectancy"] <= 0:
        reasons.append("negative_fee_stress_expectancy")
    if base["profit_factor"] <= 1 or stress["profit_factor"] <= 1:
        reasons.append("profit_factor_not_above_one")
    if not data_audit["funding_history_complete_for_60d"]:
        reasons.append("public_funding_history_is_incomplete")
    if data_audit["minimum_ohlcv_coverage"] < 0.99:
        reasons.append("raw_ohlcv_has_material_gaps")
    if not startup["pass"]:
        reasons.append("dry_run_startup_smoke_failed")

    engine_pass = safety["pass"] and guard["pass"] and startup["pass"]
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "ENGINE_ACCEPTED_STRATEGY_REJECTED"
        if engine_pass
        else "TECHNICAL_ACCEPTANCE_FAILED",
        "engine": "Freqtrade",
        "upstream": upstream,
        "safety": safety,
        "data_audit": data_audit,
        "backtests": {
            "base_cost": base,
            "fee_stress_proxy": stress,
            "stress_note": "0.08% fee per fill is a conservative fee/slippage proxy, not order-book replay",
        },
        "live_guard_probe": guard,
        "startup_smoke": startup,
        "eligible_for_paper": False,
        "eligible_for_live": False,
        "rejection_reasons": reasons,
        "next_decision": "Keep the engine; replace and prospectively validate the sample strategy.",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if engine_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
