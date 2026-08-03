from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from simple_research.regimes import (  # noqa: E402
    REGIME_CONTRACT,
    REGIME_ORDER,
    build_market_regime_lookup,
)

PILOT = ROOT / "freqtrade_pilot"
USER_DIR = PILOT / "user_data"
BASE_CONFIG = USER_DIR / "config.dryrun.json"
HISTORICAL_CONFIG = USER_DIR / "config.historical.json"
STRATEGY_DIR = USER_DIR / "strategies"
CANDIDATE_PATH = PILOT / "candidates" / "long_horizon_trend_v1.json"
SOURCE_DIR = ROOT / "Retrodate" / "bitget_futures_history_v3_2022_20260714"
SOURCE_MANIFEST = SOURCE_DIR / "integrity_manifest.json"
DATA_ROOT = PILOT / "research_data" / "bitget"
DATA_DIR = DATA_ROOT / "futures"
RESULTS_ROOT = PILOT / "research_results" / "long_horizon_trend_v1"
REPORT_PATH = ROOT / "Reports" / "FreqtradePilot" / "long_horizon_trend_v1.json"
REGIME_REPORT_PATH = (
    ROOT / "Reports" / "FreqtradePilot" / "long_horizon_trend_v1_regimes.md"
)
STRATEGY_NAME = "LongHorizonTrendStrategyV1"
PAIR_WHITELIST = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))["exchange"][
    "pair_whitelist"
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _freqtrade() -> list[str]:
    executable = ROOT / ".venv-freqtrade" / "Scripts" / "freqtrade.exe"
    if not executable.exists():
        raise RuntimeError(f"Freqtrade executable is missing: {executable}")
    return [str(executable)]


def _run(command: list[str], timeout: int = 1800) -> subprocess.CompletedProcess[str]:
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
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-40:])
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}\n{tail}")
    return result


def _verify_source(candidate: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    expected_dataset_sha = candidate["dataset"]["dataset_sha256"]
    if manifest["dataset_sha256"] != expected_dataset_sha:
        raise RuntimeError("Canonical dataset SHA does not match candidate contract")
    failures: list[str] = []
    for item in manifest["files"]:
        path = ROOT / item["path"]
        if not path.exists() or _sha256(path) != item["sha256"]:
            failures.append(item["path"])
    if failures:
        raise RuntimeError(f"Canonical source integrity failed: {failures}")
    if not manifest["validation"]["passed"]:
        raise RuntimeError("Canonical source manifest is not validated")
    return {
        "dataset_sha256": manifest["dataset_sha256"],
        "manifest_sha256": _sha256(SOURCE_MANIFEST),
        "rows": sum(int(item["rows"]) for item in manifest["files"]),
        "symbols": manifest["requested_symbols"],
        "start": manifest["requested_start_date"],
        "end": manifest["requested_end_date"],
        "coverage_pass": True,
    }


def _conversion_current(source: dict[str, Any], symbols: list[str]) -> bool:
    manifest_path = DATA_DIR / "conversion_manifest.json"
    if not manifest_path.exists():
        return False
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("source_dataset_sha256") != source["dataset_sha256"]:
        return False
    return all(
        (DATA_DIR / f"{symbol}_USDT_USDT-1h-futures.feather").exists()
        for symbol in symbols
    )


def _build_freqtrade_data(
    source: dict[str, Any], symbols: list[str], *, rebuild: bool
) -> dict[str, Any]:
    if not rebuild and _conversion_current(source, symbols):
        return json.loads(
            (DATA_DIR / "conversion_manifest.json").read_text(encoding="utf-8")
        )

    frames = [
        pd.read_csv(
            path,
            usecols=["timestamp", "open", "high", "low", "close", "volume", "symbol"],
        )
        for path in sorted(SOURCE_DIR.glob("crypto_1m_*_all_symbols.csv"))
    ]
    combined = pd.concat(frames, ignore_index=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output: dict[str, Any] = {}
    for symbol in symbols:
        source_symbol = f"{symbol}/USDT"
        target = DATA_DIR / f"{symbol}_USDT_USDT-1h-futures.feather"
        frame = combined.loc[combined["symbol"] == source_symbol].copy()
        frame["date"] = pd.to_datetime(frame.pop("timestamp"), unit="ms", utc=True)
        frame = frame[["date", "open", "high", "low", "close", "volume"]]
        frame = frame.sort_values("date").drop_duplicates("date")
        deltas = frame["date"].diff().dropna().dt.total_seconds()
        if len(frame) != 39744 or not (deltas == 3600).all():
            raise RuntimeError(f"Converted cadence failed for {symbol}")
        frame.reset_index(drop=True).to_feather(target)
        output[symbol] = {
            "rows": len(frame),
            "start": frame["date"].iloc[0].isoformat(),
            "end": frame["date"].iloc[-1].isoformat(),
            "sha256": _sha256(target),
        }

    manifest = {
        "schema_version": "panteon.freqtrade_data_conversion.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "source_dataset_sha256": source["dataset_sha256"],
        "timeframe": "1h",
        "pairs": output,
    }
    (DATA_DIR / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _strategy_args() -> list[str]:
    return [
        "--config",
        str(BASE_CONFIG),
        "--config",
        str(HISTORICAL_CONFIG),
        "--userdir",
        str(USER_DIR),
        "--strategy-path",
        str(STRATEGY_DIR),
        "--strategy",
        STRATEGY_NAME,
    ]


def _market_regime_lookup() -> dict[int, str]:
    frame = pd.read_feather(DATA_DIR / "BTC_USDT_USDT-1h-futures.feather")
    frame["timestamp"] = frame["date"].map(
        lambda value: int(pd.Timestamp(value).timestamp() * 1000)
    )
    frame["symbol"] = "BTC/USDT"
    return build_market_regime_lookup(frame)


def _lcb(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.mean(values) - 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def _trade_group(trades: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(trade["profit_ratio"]) * 10_000.0 for trade in trades]
    positive = sum(value for value in values if value > 0)
    negative = abs(sum(value for value in values if value < 0))
    wins = sum(value > 0 for value in values)
    return {
        "closed_trades": len(trades),
        "wins": wins,
        "win_rate": wins / len(trades) if trades else None,
        "mean_net_bps": statistics.mean(values) if values else None,
        "median_net_bps": statistics.median(values) if values else None,
        "lcb_95_net_bps": _lcb(values),
        "profit_factor": positive / negative if negative > 0 else None,
        "net_profit_abs": sum(float(trade["profit_abs"]) for trade in trades),
    }


def _parse_backtest(
    path: Path,
    fee: float,
    gates: dict[str, Any],
    regime_lookup: dict[int, str],
) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        name = next(
            item
            for item in archive.namelist()
            if item.endswith(".json")
            and not item.endswith(".meta.json")
            and "config" not in item
        )
        result = json.loads(archive.read(name))["strategy"][STRATEGY_NAME]
    trades = [trade for trade in result["trades"] if not trade["is_open"]]
    aggregate = _trade_group(trades)
    directions = {
        "LONG": _trade_group([trade for trade in trades if not trade["is_short"]]),
        "SHORT": _trade_group([trade for trade in trades if trade["is_short"]]),
    }
    symbols = {
        pair: _trade_group([trade for trade in trades if trade["pair"] == pair])
        for pair in PAIR_WHITELIST
    }
    trades_by_regime: dict[str, list[dict[str, Any]]] = {
        regime: [] for regime in REGIME_ORDER
    }
    for trade in trades:
        decision_timestamp = int(trade["open_timestamp"]) - 3_600_000
        regime = regime_lookup.get(decision_timestamp, "unknown")
        trades_by_regime[regime].append(trade)
    market_regimes = {
        regime: _trade_group(trades_by_regime[regime]) for regime in REGIME_ORDER
    }
    promotable_symbols = sum(
        1
        for item in symbols.values()
        if item["closed_trades"] >= 5
        and item["lcb_95_net_bps"] is not None
        and item["lcb_95_net_bps"] > 0
    )
    failures: list[str] = []
    fills = sum(len(trade.get("orders", [])) for trade in trades)
    if fills < gates["min_fills_per_split"]:
        failures.append("fills_below_minimum")
    if len(trades) < gates["min_closed_trades_per_split"]:
        failures.append("closed_trades_below_minimum")
    if aggregate["mean_net_bps"] is None or aggregate["mean_net_bps"] <= 0:
        failures.append("nonpositive_expectancy")
    if aggregate["lcb_95_net_bps"] is None or aggregate["lcb_95_net_bps"] <= 0:
        failures.append("nonpositive_lcb")
    if result["profit_factor"] is None or result["profit_factor"] <= gates["min_profit_factor"]:
        failures.append("profit_factor_not_above_one")
    if result["max_drawdown_account"] > gates["max_drawdown_account"]:
        failures.append("max_drawdown_exceeded")
    for direction, item in directions.items():
        if (
            item["closed_trades"] < gates["min_direction_trades_per_split"]
            or item["lcb_95_net_bps"] is None
            or item["lcb_95_net_bps"] <= 0
        ):
            failures.append(f"direction_collapse:{direction}")
    if promotable_symbols < gates["min_promotable_symbols_per_split"]:
        failures.append("promotable_symbols_below_minimum")
    if result["profit_total_abs"] <= 0:
        failures.append("does_not_beat_no_trade_baseline")

    return {
        "fee_per_fill": fee,
        "fills": fills,
        **aggregate,
        "profit_total_abs": result["profit_total_abs"],
        "profit_total_ratio": result["profit_total"],
        "profit_factor": result["profit_factor"],
        "max_drawdown_abs": result["max_drawdown_abs"],
        "max_drawdown_account": result["max_drawdown_account"],
        "wins": result["wins"],
        "losses": result["losses"],
        "backtest_start": result["backtest_start"],
        "backtest_end": result["backtest_end"],
        "market_change": result["market_change"],
        "exit_reasons": result["exit_reason_summary"],
        "directions": directions,
        "symbols": symbols,
        "market_regimes": market_regimes,
        "promotable_symbols": promotable_symbols,
        "failures": failures,
        "gate_pass": not failures,
        "artifact": str(path.relative_to(ROOT)),
    }


def _run_backtest(
    split: str,
    timerange: str,
    fee: float,
    gates: dict[str, Any],
    run_dir: Path,
    regime_lookup: dict[int, str],
) -> dict[str, Any]:
    result_dir = run_dir / split / f"fee_{int(fee * 1_000_000)}"
    result_dir.mkdir(parents=True, exist_ok=True)
    before = set(result_dir.glob("*.zip"))
    _run(
        _freqtrade()
        + [
            "backtesting",
            *_strategy_args(),
            "--datadir",
            str(DATA_ROOT),
            "--timeframe",
            "1h",
            "--timerange",
            timerange,
            "--fee",
            str(fee),
            "--export",
            "trades",
            "--backtest-directory",
            str(result_dir),
        ]
    )
    created = set(result_dir.glob("*.zip")) - before
    if not created:
        raise RuntimeError(f"No backtest artifact created for {split} fee={fee}")
    return _parse_backtest(
        max(created, key=lambda path: path.stat().st_mtime),
        fee,
        gates,
        regime_lookup,
    )


def _number(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _write_regime_report(report: dict[str, Any]) -> None:
    lines = [
        "# LongHorizonTrendStrategyV1 regime attribution",
        "",
        "This is a post-trade diagnostic. Regimes did not filter or alter entries.",
        "Each trade is attributed using the last closed BTC 1h candle before entry.",
        "",
        "## Regime definition",
        "",
        "- `bullish`: BTC close > EMA72 > EMA336 and 7-day momentum > 0.",
        "- `bearish`: BTC close < EMA72 < EMA336 and 7-day momentum < 0.",
        "- `volatile_mixed`: no directional trend and ATR14/close is at or above its trailing 180-day 75th percentile.",
        "- `range_low_vol`: no directional trend and ATR14/close is at or below its trailing 180-day 25th percentile.",
        "- `neutral`: remaining observations.",
        "",
        "## Results",
        "",
        "| Window | Regime | Trades | Win % | Mean base bps | LCB base bps | Mean stress bps | LCB stress bps | Regime gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    minimum = 10
    for split, values in report["evaluations"].items():
        for regime in REGIME_ORDER:
            base = values["base"]["market_regimes"][regime]
            stress = values["stress"]["market_regimes"][regime]
            if base["closed_trades"] == 0 and stress["closed_trades"] == 0:
                continue
            passed = (
                base["closed_trades"] >= minimum
                and stress["closed_trades"] >= minimum
                and base["mean_net_bps"] is not None
                and base["mean_net_bps"] > 0
                and base["lcb_95_net_bps"] is not None
                and base["lcb_95_net_bps"] > 0
                and stress["mean_net_bps"] is not None
                and stress["mean_net_bps"] > 0
                and stress["lcb_95_net_bps"] is not None
                and stress["lcb_95_net_bps"] > 0
            )
            lines.append(
                f"| {split} | {regime} | {base['closed_trades']} | "
                f"{_number(None if base['win_rate'] is None else base['win_rate'] * 100, 1)} | "
                f"{_number(base['mean_net_bps'])} | {_number(base['lcb_95_net_bps'])} | "
                f"{_number(stress['mean_net_bps'])} | {_number(stress['lcb_95_net_bps'])} | "
                f"{'PASS' if passed else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            "A regime passes only with at least 10 trades and positive mean and LCB under both base and stress costs.",
            "No regime result grants paper/live authority.",
        ]
    )
    REGIME_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _live_guard_probe() -> dict[str, Any]:
    import sys

    sys.path.insert(0, str(STRATEGY_DIR))
    from LongHorizonTrendStrategyV1 import LongHorizonTrendStrategyV1

    try:
        LongHorizonTrendStrategyV1({"dry_run": False}).bot_start()
    except RuntimeError as exc:
        return {"pass": True, "exception": str(exc)}
    return {"pass": False, "exception": None}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate fixed long-horizon Freqtrade candidate")
    parser.add_argument("--rebuild-data", action="store_true")
    args = parser.parse_args()

    candidate = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))
    source = _verify_source(candidate)
    conversion = _build_freqtrade_data(
        source,
        list(candidate["dataset"]["symbols"]),
        rebuild=args.rebuild_data,
    )
    regime_lookup = _market_regime_lookup()
    listing = _run(
        _freqtrade()
        + [
            "list-strategies",
            "--config",
            str(BASE_CONFIG),
            "--config",
            str(HISTORICAL_CONFIG),
            "--userdir",
            str(USER_DIR),
            "--strategy-path",
            str(STRATEGY_DIR),
            "--one-column",
        ]
    ).stdout.splitlines()
    if STRATEGY_NAME not in listing:
        raise RuntimeError(f"Strategy resolver did not find {STRATEGY_NAME}")

    run_dir = RESULTS_ROOT / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    evaluations: dict[str, Any] = {}
    for split, timerange in candidate["splits"].items():
        evaluations[split] = {
            "timerange": timerange,
            "base": _run_backtest(
                split,
                timerange,
                candidate["costs"]["base_fee_per_fill"],
                candidate["gates"],
                run_dir,
                regime_lookup,
            ),
            "stress": _run_backtest(
                split,
                timerange,
                candidate["costs"]["stress_fee_per_fill"],
                candidate["gates"],
                run_dir,
                regime_lookup,
            ),
        }

    failed_windows = [
        f"{split}:{cost}"
        for split, values in evaluations.items()
        for cost in ("base", "stress")
        if not values[cost]["gate_pass"]
    ]
    retrospective_pass = not failed_windows
    report = {
        "schema_version": "panteon.freqtrade_candidate_report.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "RETROSPECTIVE_PASS_REQUIRES_PROSPECTIVE"
        if retrospective_pass
        else "TERMINAL_REJECTED_RETROSPECTIVE",
        "candidate_id": candidate["candidate_id"],
        "candidate_contract_sha256": _sha256(CANDIDATE_PATH),
        "source": source,
        "conversion": conversion,
        "regime_diagnostic": {**REGIME_CONTRACT, "source": "BTC/USDT:USDT"},
        "funding_assumption": {
            "rate": 0,
            "reason": "canonical dataset has no historical funding series",
            "limitation": "result cannot authorize paper or live trading",
        },
        "baseline": {"baseline_id": "no_trade_cash_v1", "net_pnl": 0},
        "evaluations": evaluations,
        "failed_windows": failed_windows,
        "live_guard_probe": _live_guard_probe(),
        "retrospective_pass": retrospective_pass,
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
        "archived_candidates_not_retried": {
            "funding_carry_hourly_v1": "max projected carry below cost floor and zero signals",
            "cross_sectional_trend_4h_v1": "negative LCB, direction collapse, and below baseline",
        },
        "next_decision": "Prospective dry-run is forbidden unless every retrospective base and stress window passes.",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_regime_report(report)
    compact = {
        "status": report["status"],
        "failed_windows": failed_windows,
        "splits": {
            split: {
                cost: {
                    "trades": values[cost]["closed_trades"],
                    "mean_bps": values[cost]["mean_net_bps"],
                    "lcb_bps": values[cost]["lcb_95_net_bps"],
                    "failures": values[cost]["failures"],
                }
                for cost in ("base", "stress")
            }
            for split, values in evaluations.items()
        },
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
