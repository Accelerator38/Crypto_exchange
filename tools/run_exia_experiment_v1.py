from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
STRATEGY_DIR = ROOT / "freqtrade_pilot" / "user_data" / "strategies"
for path in (SRC_ROOT, STRATEGY_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from exia.catalog import build_catalog  # noqa: E402
from exia.contracts import (  # noqa: E402
    ExperimentManifest,
    load_candidate_spec,
    sha256_file,
)
from exia.scoring import catalog_rows_for_run  # noqa: E402


PILOT = ROOT / "freqtrade_pilot"
USER_DIR = PILOT / "user_data"
BASE_CONFIG = USER_DIR / "config.dryrun.json"
HISTORICAL_CONFIG = USER_DIR / "config.historical.json"
DEFAULT_CANDIDATE = PILOT / "candidates" / "exia_market_mode_foundation_v1.json"
UPSTREAM_LOCK = PILOT / "upstream.lock.json"
DATA_ROOT = PILOT / "research_data" / "bitget"
DATA_DIR = DATA_ROOT / "futures"
CONVERSION_MANIFEST = DATA_DIR / "conversion_manifest.json"
FEATURE_TAPE = ROOT / "Retrodate" / "simple_research_reset_v1" / "bitget_full8_1h.parquet"
FEATURE_TAPE_MANIFEST = FEATURE_TAPE.with_suffix(".manifest.json")
RAW_RESULTS_ROOT = PILOT / "research_results" / "exia"
EXPERIMENTS_ROOT = ROOT / "Reports" / "Exia" / "experiments"
CATALOG_DIR = ROOT / "Reports" / "Exia" / "catalog"

LEDGER_COLUMNS = (
    "experiment_id",
    "candidate_id",
    "candidate_version",
    "family",
    "split",
    "cost_scenario",
    "symbol",
    "pair",
    "direction",
    "market_state",
    "signal_timestamp",
    "entry_timestamp",
    "exit_timestamp",
    "gross_bps",
    "recorded_fee_bps",
    "net_bps",
    "fills",
    "entry_tag",
    "exit_reason",
    "holding_minutes",
)

EXPERIMENT_CODE_PATHS = (
    Path("tools/run_exia_experiment_v1.py"),
    Path("src/exia/contracts.py"),
    Path("src/exia/scoring.py"),
    Path("src/exia/catalog.py"),
)


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _composite_sha256(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(paths, key=lambda item: item.as_posix()):
        path = ROOT / relative
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _freqtrade() -> Path:
    path = ROOT / ".venv-freqtrade" / "Scripts" / "freqtrade.exe"
    if not path.is_file():
        raise RuntimeError(f"Freqtrade executable is missing: {path}")
    return path


def _run(command: list[str], *, timeout: int = 1800) -> subprocess.CompletedProcess[str]:
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
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-50:])
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{tail}"
        )
    return result


def _verify_runtime(spec: Any) -> tuple[dict[str, Any], str]:
    config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    exchange = config["exchange"]
    checks = {
        "dry_run": config.get("dry_run") is True,
        "bitget": exchange.get("name") == "bitget",
        "credentials_empty": not any(
            exchange.get(key) for key in ("key", "secret", "password")
        ),
        "max_open_trades_match": int(config["max_open_trades"])
        == int(spec.risk["max_open_trades"]),
        "stake_match": float(config["stake_amount"])
        == float(spec.risk["stake_amount_usdt"]),
        "force_entry_disabled": config.get("force_entry_enable") is False,
        "api_disabled": config.get("api_server", {}).get("enabled") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Exia runtime config failed closed: {checks}")

    lock = json.loads(UPSTREAM_LOCK.read_text(encoding="utf-8"))
    revision = _run(
        ["git", "-C", str(ROOT / lock["submodule_path"]), "rev-parse", "HEAD"]
    ).stdout.strip()
    if revision != lock["commit"]:
        raise RuntimeError("Freqtrade revision does not match upstream lock")
    listing = _run(
        [
            str(_freqtrade()),
            "list-strategies",
            "--config",
            str(BASE_CONFIG),
            "--userdir",
            str(USER_DIR),
            "--strategy-path",
            str(STRATEGY_DIR),
            "--one-column",
        ]
    ).stdout.splitlines()
    if spec.strategy["class_name"] not in listing:
        raise RuntimeError(
            f"Freqtrade cannot resolve strategy {spec.strategy['class_name']}"
        )
    return checks, revision


def _verify_data(spec: Any) -> dict[str, Any]:
    source_manifest = json.loads(
        (ROOT / str(spec.dataset["manifest"])).read_text(encoding="utf-8")
    )
    if source_manifest.get("dataset_sha256") != spec.dataset["dataset_sha256"]:
        raise RuntimeError("source dataset SHA mismatch")
    if source_manifest.get("validation", {}).get("passed") is not True:
        raise RuntimeError("source dataset validation did not pass")
    conversion = json.loads(CONVERSION_MANIFEST.read_text(encoding="utf-8"))
    if conversion.get("source_dataset_sha256") != spec.dataset["dataset_sha256"]:
        raise RuntimeError("Freqtrade data conversion SHA mismatch")
    if conversion.get("timeframe") != spec.dataset["timeframe"]:
        raise RuntimeError("Freqtrade data conversion timeframe mismatch")
    failures: list[str] = []
    for symbol in spec.dataset["symbols"]:
        path = DATA_DIR / f"{symbol}_USDT_USDT-1h-futures.feather"
        expected = conversion.get("pairs", {}).get(symbol, {}).get("sha256")
        if not path.is_file() or not expected or sha256_file(path) != expected:
            failures.append(symbol)
    if failures:
        raise RuntimeError(f"Freqtrade canonical data integrity failed: {failures}")
    tape_manifest = json.loads(FEATURE_TAPE_MANIFEST.read_text(encoding="utf-8"))
    if (
        tape_manifest.get("source_integrity", {}).get("dataset_sha256")
        != spec.dataset["dataset_sha256"]
    ):
        raise RuntimeError("state attribution tape SHA mismatch")
    return {
        "dataset_sha256": spec.dataset["dataset_sha256"],
        "conversion_manifest_sha256": sha256_file(CONVERSION_MANIFEST),
        "feature_tape_sha256": sha256_file(FEATURE_TAPE),
        "symbols": list(spec.dataset["symbols"]),
    }


def _taxonomy_api(spec: Any) -> tuple[tuple[str, ...], Any]:
    source = ROOT / str(spec.taxonomy["source"])
    module_name = f"_exia_taxonomy_{str(spec.taxonomy['source_sha256'])[:12]}"
    module_spec = importlib.util.spec_from_file_location(module_name, source)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"cannot load taxonomy module: {source}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    market_mode_order = tuple(str(value) for value in module.MARKET_MODE_ORDER)
    if not set(spec.allowed_market_states) <= set(market_mode_order):
        raise RuntimeError("candidate states are not provided by pinned taxonomy")
    return market_mode_order, module.append_market_mode_columns


def _state_lookup(
    spec: Any,
) -> tuple[dict[tuple[str, int], str], tuple[str, ...]]:
    market_mode_order, classify = _taxonomy_api(spec)
    tape = pd.read_parquet(FEATURE_TAPE)
    lookup: dict[tuple[str, int], str] = {}
    for source_symbol, source in tape.groupby("symbol", sort=True):
        symbol = str(source_symbol).split("/")[0]
        frame = source.sort_values("timestamp", kind="stable").copy()
        frame["date"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        classified = classify(frame)
        lookup.update(
            {
                (symbol, int(timestamp)): str(mode)
                for timestamp, mode in zip(
                    classified["timestamp"], classified["exia_market_mode"]
                )
            }
        )
    return lookup, market_mode_order


def _strategy_args(class_name: str) -> list[str]:
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
        class_name,
    ]


def _read_backtest(path: Path, class_name: str) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        result_name = next(
            name
            for name in archive.namelist()
            if name.endswith(".json")
            and not name.endswith(".meta.json")
            and "config" not in name
        )
        payload = json.loads(archive.read(result_name))
    return payload["strategy"][class_name]


def _trade_ledger(
    trades: list[dict[str, Any]],
    *,
    experiment_id: str,
    spec: Any,
    split: str,
    cost_scenario: str,
    states: dict[tuple[str, int], str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade in trades:
        if trade.get("is_open"):
            raise RuntimeError("backtest export contains an open trade")
        pair = str(trade["pair"])
        symbol = pair.split("/")[0]
        entry_timestamp = int(trade["open_timestamp"])
        signal_timestamp = entry_timestamp - 3_600_000
        market_state = states.get((symbol, signal_timestamp))
        if market_state is None:
            raise RuntimeError(
                f"missing market state for {symbol} at {signal_timestamp}"
            )
        net_bps = float(trade["profit_ratio"]) * 10_000.0
        fee_bps = (
            float(trade.get("fee_open", 0.0))
            + float(trade.get("fee_close", 0.0))
        ) * 10_000.0
        rows.append(
            {
                "experiment_id": experiment_id,
                "candidate_id": spec.candidate_id,
                "candidate_version": spec.version,
                "family": spec.family,
                "split": split,
                "cost_scenario": cost_scenario,
                "symbol": symbol,
                "pair": pair,
                "direction": "SHORT" if trade["is_short"] else "LONG",
                "market_state": market_state,
                "signal_timestamp": signal_timestamp,
                "entry_timestamp": entry_timestamp,
                "exit_timestamp": int(trade["close_timestamp"]),
                "gross_bps": net_bps + fee_bps,
                "recorded_fee_bps": fee_bps,
                "net_bps": net_bps,
                "fills": len(trade.get("orders", [])),
                "entry_tag": trade.get("enter_tag"),
                "exit_reason": trade.get("exit_reason"),
                "holding_minutes": int(trade.get("trade_duration", 0)),
            }
        )
    return pd.DataFrame(rows, columns=LEDGER_COLUMNS)


def _run_backtest(
    *,
    experiment_id: str,
    spec: Any,
    split: str,
    timerange: str,
    cost_scenario: str,
    fee: float,
    states: dict[tuple[str, int], str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_dir = RAW_RESULTS_ROOT / experiment_id / split / cost_scenario
    raw_dir.mkdir(parents=True, exist_ok=False)
    command = [
        str(_freqtrade()),
        "backtesting",
        *_strategy_args(str(spec.strategy["class_name"])),
        "--datadir",
        str(DATA_ROOT),
        "--timeframe",
        str(spec.dataset["timeframe"]),
        "--timerange",
        timerange,
        "--fee",
        str(fee),
        "--cache",
        "none",
        "--export",
        "trades",
        "--backtest-directory",
        str(raw_dir),
    ]
    before = set(raw_dir.glob("*.zip"))
    _run(command)
    created = set(raw_dir.glob("*.zip")) - before
    if len(created) != 1:
        raise RuntimeError(
            f"expected one Freqtrade export for {split}/{cost_scenario}, got {len(created)}"
        )
    artifact = created.pop()
    result = _read_backtest(artifact, str(spec.strategy["class_name"]))
    trades = [trade for trade in result["trades"] if not trade.get("is_open")]
    ledger = _trade_ledger(
        trades,
        experiment_id=experiment_id,
        spec=spec,
        split=split,
        cost_scenario=cost_scenario,
        states=states,
    )
    summary = {
        "split": split,
        "cost_scenario": cost_scenario,
        "timerange": timerange,
        "fee_per_fill": fee,
        "closed_trades": len(ledger),
        "fills": int(ledger["fills"].sum()) if len(ledger) else 0,
        "max_drawdown": float(result["max_drawdown_account"]),
        "profit_total_abs": float(result["profit_total_abs"]),
        "artifact": str(artifact.relative_to(ROOT)).replace("\\", "/"),
        "artifact_sha256": sha256_file(artifact),
        "command": command,
    }
    return ledger, summary


def _render_report(metrics: dict[str, Any]) -> str:
    lines = [
        f"# Exia experiment: {metrics['experiment_id']}",
        "",
        f"Candidate: `{metrics['candidate_id']}` version `{metrics['candidate_version']}`.",
        f"Status: `{metrics['status']}`.",
        "",
        "This is an offline Freqtrade experiment. It cannot authorize paper/live.",
        "",
        "## Runs",
        "",
        "| Window | Cost | Trades | Fills | Mean bps | Block LCB | Drawdown |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    overall = {
        (row["split"], row["cost_scenario"]): row
        for row in metrics["catalog_rows"]
        if row["scope"] == "overall"
    }
    for run in metrics["runs"]:
        row = overall[(run["split"], run["cost_scenario"])]
        mean = "n/a" if row["mean_net_bps"] is None else f"{row['mean_net_bps']:.2f}"
        lcb = (
            "n/a"
            if row["block_lcb_net_bps"] is None
            else f"{row['block_lcb_net_bps']:.2f}"
        )
        lines.append(
            f"| {run['split']} | {run['cost_scenario']} | "
            f"{row['closed_trades']} | {row['fills']} | {mean} | {lcb} | "
            f"{run['max_drawdown']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- `orders_enabled=false`; `promotion_authority=false`.",
            "- No dynamic candidate selection or parameter mutation occurred.",
            "- Foundation zero trades are an engineering PASS, not alpha evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one immutable Exia Freqtrade experiment")
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--splits",
        default="all",
        help="Comma-separated candidate splits or 'all' (default).",
    )
    args = parser.parse_args()

    candidate_path = args.candidate.resolve()
    spec = load_candidate_spec(candidate_path, root=ROOT)
    if args.splits == "all":
        selected_splits = tuple(spec.splits)
    else:
        selected_splits = tuple(
            value.strip() for value in str(args.splits).split(",") if value.strip()
        )
        unknown_splits = sorted(set(selected_splits) - set(spec.splits))
        if not selected_splits or unknown_splits:
            raise ValueError(f"invalid selected splits: {unknown_splits}")
    runtime_checks, freqtrade_revision = _verify_runtime(spec)
    data = _verify_data(spec)
    states, market_mode_order = _state_lookup(spec)
    candidate_sha = sha256_file(candidate_path)
    started = datetime.now(UTC)
    experiment_id = (
        f"{spec.candidate_id}_{started.strftime('%Y%m%dt%H%M%Sz')}_"
        f"{candidate_sha[:8]}"
    )
    experiment_dir = EXPERIMENTS_ROOT / experiment_id
    if experiment_dir.exists():
        raise RuntimeError(f"immutable experiment already exists: {experiment_dir}")

    all_ledgers: list[pd.DataFrame] = []
    run_summaries: list[dict[str, Any]] = []
    catalog_rows: list[dict[str, Any]] = []
    identity = {
        "experiment_id": experiment_id,
        "candidate_id": spec.candidate_id,
        "candidate_version": spec.version,
        "family": spec.family,
        "dataset_sha256": str(spec.dataset["dataset_sha256"]),
        "taxonomy_sha256": str(spec.taxonomy["source_sha256"]),
    }
    costs = {
        "base": float(spec.costs["base_fee_per_fill"]),
        "stress": float(spec.costs["stress_fee_per_fill"]),
    }
    for split in selected_splits:
        timerange = spec.splits[split]
        for cost_scenario, fee in costs.items():
            ledger, summary = _run_backtest(
                experiment_id=experiment_id,
                spec=spec,
                split=split,
                timerange=timerange,
                cost_scenario=cost_scenario,
                fee=fee,
                states=states,
            )
            all_ledgers.append(ledger)
            run_summaries.append(summary)
            catalog_rows.extend(
                catalog_rows_for_run(
                    ledger,
                    identity=identity,
                    split=split,
                    cost_scenario=cost_scenario,
                    max_drawdown=summary["max_drawdown"],
                    states=market_mode_order,
                    symbols=tuple(str(value) for value in spec.dataset["symbols"]),
                )
            )

    ledger_all = pd.concat(all_ledgers, ignore_index=True)
    total_trades = len(ledger_all)
    if spec.family == "engineering_foundation":
        status = (
            "ENGINEERING_PASS_NO_TRADES"
            if total_trades == 0
            else "ENGINEERING_FOUNDATION_VIOLATION"
        )
    else:
        status = (
            "DEVELOPMENT_SCORED"
            if selected_splits == ("development",)
            else "SCORED_RETROSPECTIVE"
        )

    manifest_payload = {
        "schema_version": ExperimentManifest.SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "candidate_id": spec.candidate_id,
        "candidate_spec_sha256": candidate_sha,
        "strategy_sha256": str(spec.strategy["source_sha256"]),
        "taxonomy_sha256": str(spec.taxonomy["source_sha256"]),
        "dataset_sha256": str(spec.dataset["dataset_sha256"]),
        "freqtrade_revision": freqtrade_revision,
        "experiment_code_sha256": _composite_sha256(EXPERIMENT_CODE_PATHS),
        "runtime_config_sha256": sha256_file(BASE_CONFIG),
        "historical_config_sha256": sha256_file(HISTORICAL_CONFIG),
        "timeranges": {split: spec.splits[split] for split in selected_splits},
        "costs": costs,
        "command": [sys.executable, str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]],
        "orders_enabled": False,
        "promotion_authority": False,
    }
    ExperimentManifest.from_mapping(manifest_payload)
    metrics = {
        "schema_version": "exia.experiment_metrics.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "experiment_id": experiment_id,
        "candidate_id": spec.candidate_id,
        "candidate_version": spec.version,
        "family": spec.family,
        "status": status,
        "runtime_checks": runtime_checks,
        "data": data,
        "runs": run_summaries,
        "catalog_rows": catalog_rows,
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }
    experiment_dir.mkdir(parents=True, exist_ok=False)
    (experiment_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    ledger_all.to_parquet(
        experiment_dir / "trades.parquet", index=False, compression="zstd"
    )
    (experiment_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (experiment_dir / "report.md").write_text(
        _render_report(metrics), encoding="utf-8"
    )
    catalog = build_catalog(experiments_root=EXPERIMENTS_ROOT, output_dir=CATALOG_DIR)
    print(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "status": status,
                "total_trades_across_runs": total_trades,
                "catalog_experiments": catalog["experiment_count"],
                "experiment_dir": str(experiment_dir),
                "orders_enabled": False,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if status in {
        "ENGINEERING_PASS_NO_TRADES",
        "DEVELOPMENT_SCORED",
        "SCORED_RETROSPECTIVE",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
