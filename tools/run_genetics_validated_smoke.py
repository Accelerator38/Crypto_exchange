from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from panteon_runtime.exchange_symbols import normalize_base_symbol  # noqa: E402

DEFAULT_BASELINE = ROOT / "Genetics_DL_Agents" / "Agents" / "genetics" / "best_genome.npy"
DEFAULT_SETTINGS = ROOT / "settings.txt"


def _mode_payload(genome_payload: dict[str, Any], mode: str) -> dict[str, Any]:
    for item in genome_payload.get("modes", []):
        if item.get("mode") == mode:
            return item
    raise KeyError(f"mode {mode!r} not found for {genome_payload.get('path')}")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _metrics(genome_payload: dict[str, Any], *, mode: str) -> dict[str, Any]:
    mode_payload = _mode_payload(genome_payload, mode)
    stats = mode_payload.get("period_stats") or {}
    robust = mode_payload.get("robust_score") or {}
    contract = mode_payload.get("contract_metrics") or {}
    period_rets = [float(value) for value in mode_payload.get("period_rets", [])]
    return {
        "path": str(genome_payload.get("path", "")),
        "position_state_features_enabled": bool(
            genome_payload.get("position_state_features_enabled", False)
        ),
        "fitness": _float(mode_payload.get("fitness")),
        "mean_ret": _float(stats.get("mean_ret")),
        "min_ret": _float(stats.get("min_ret")),
        "max_ret": _float(stats.get("max_ret")),
        "positive_period_pct": _float(stats.get("positive_period_pct")),
        "passes_default_gates": bool(robust.get("passes_default_gates", False)),
        "failed_gates": list(robust.get("failed_gates") or []),
        "mean_turnover_rate": _float(contract.get("mean_turnover_rate")),
        "max_turnover_rate": _float(
            contract.get("max_turnover_rate", contract.get("mean_turnover_rate"))
        ),
        "mean_saturation_rate": _float(contract.get("mean_saturation_rate")),
        "max_saturation_rate": _float(
            contract.get("max_saturation_rate", contract.get("mean_saturation_rate"))
        ),
        "mean_invalid_open_logit_pressure": _float(
            contract.get("mean_invalid_open_logit_pressure")
        ),
        "max_invalid_open_logit_pressure": _float(
            contract.get(
                "max_invalid_open_logit_pressure",
                contract.get("mean_invalid_open_logit_pressure"),
            )
        ),
        "mean_raw_capacity_bar_rate": _float(
            contract.get("mean_raw_capacity_bar_rate", contract.get("capacity_bar_rate"))
        ),
        "max_raw_capacity_bar_rate": _float(
            contract.get(
                "max_raw_capacity_bar_rate",
                contract.get("mean_raw_capacity_bar_rate", contract.get("capacity_bar_rate")),
            )
        ),
        "mean_same_side_open_rate": _float(contract.get("mean_same_side_open_rate")),
        "max_same_side_open_rate": _float(
            contract.get("max_same_side_open_rate", contract.get("mean_same_side_open_rate"))
        ),
        "period_count": len(period_rets),
        "period_rets": period_rets,
    }


def summarize_single_candidate_oos_gate(
    oos_report: dict[str, Any],
    *,
    mode: str = "fee_fixed_nextbar",
    min_oos_mean_delta: float = 0.0,
    min_oos_min_ret_delta: float = 0.0,
    min_positive_period_pct_delta: float = 0.0,
    min_oos_periods: int = 2,
    max_turnover_rate: float = 0.10,
    max_saturation_rate: float = 0.10,
    max_invalid_open_pressure: float = 0.05,
    max_raw_capacity_bar_rate: float = 0.25,
    max_same_side_open_rate: float = 0.05,
    require_robust_pass: bool = False,
) -> dict[str, Any]:
    """Compare first genome (baseline) vs second genome (candidate) on OOS report."""

    genomes = list(oos_report.get("genomes") or [])
    if len(genomes) < 2:
        raise ValueError("OOS report must contain baseline and candidate genomes")
    baseline = _metrics(genomes[0], mode=mode)
    candidate = _metrics(genomes[1], mode=mode)

    mean_delta = candidate["mean_ret"] - baseline["mean_ret"]
    min_delta = candidate["min_ret"] - baseline["min_ret"]
    positive_delta = candidate["positive_period_pct"] - baseline["positive_period_pct"]
    candidate.update(
        {
            "mean_ret_delta": float(mean_delta),
            "min_ret_delta": float(min_delta),
            "positive_period_pct_delta": float(positive_delta),
        }
    )

    failures: list[str] = []
    if candidate["period_count"] < min_oos_periods:
        failures.append("oos_period_count")
    if mean_delta <= min_oos_mean_delta:
        failures.append(
            "oos_mean_ret_tie"
            if abs(mean_delta - min_oos_mean_delta) <= 1e-12
            else "oos_mean_ret"
        )
    if min_delta < min_oos_min_ret_delta:
        failures.append("oos_min_ret")
    if positive_delta < min_positive_period_pct_delta:
        failures.append("oos_positive_period_pct")
    if candidate["max_turnover_rate"] > max_turnover_rate:
        failures.append("oos_max_turnover")
    if candidate["max_saturation_rate"] > max_saturation_rate:
        failures.append("oos_max_saturation")
    if candidate["max_invalid_open_logit_pressure"] > max_invalid_open_pressure:
        failures.append("oos_invalid_open_pressure")
    if candidate["max_raw_capacity_bar_rate"] > max_raw_capacity_bar_rate:
        failures.append("oos_max_raw_capacity_bar")
    if candidate["max_same_side_open_rate"] > max_same_side_open_rate:
        failures.append("oos_max_same_side_open")
    if require_robust_pass and not candidate["passes_default_gates"]:
        failures.append("oos_robust_gate")

    failures = list(dict.fromkeys(failures))
    promotion_eligible = not failures
    return {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "exchange": str(oos_report.get("exchange") or "").upper(),
        "mode": mode,
        "oos_start_date": oos_report.get("start_date"),
        "oos_end_date": oos_report.get("end_date"),
        "selected_is_baseline": not promotion_eligible,
        "promotion_eligible": promotion_eligible,
        "promotion_failures": failures,
        "selected_genome": candidate["path"] if promotion_eligible else baseline["path"],
        "baseline": baseline,
        "candidate": candidate,
        "thresholds": {
            "min_oos_mean_delta": float(min_oos_mean_delta),
            "min_oos_min_ret_delta": float(min_oos_min_ret_delta),
            "min_positive_period_pct_delta": float(min_positive_period_pct_delta),
            "min_oos_periods": int(min_oos_periods),
            "max_turnover_rate": float(max_turnover_rate),
            "max_saturation_rate": float(max_saturation_rate),
            "max_invalid_open_pressure": float(max_invalid_open_pressure),
            "max_raw_capacity_bar_rate": float(max_raw_capacity_bar_rate),
            "max_same_side_open_rate": float(max_same_side_open_rate),
            "require_robust_pass": bool(require_robust_pass),
        },
    }


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def _run(args: Sequence[str], *, env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(
        list(args),
        cwd=str(ROOT),
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_plain_settings(path: str | Path = DEFAULT_SETTINGS) -> dict[str, str]:
    settings_path = Path(path)
    if not settings_path.exists():
        return {}
    out: dict[str, str] = {}
    for raw_line in settings_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.split("#", 1)[0].strip()
        key = key.strip().lower()
        if key and value:
            out[key] = value
    return out


def _parse_symbol_filter(raw_specs: Sequence[str] | None) -> tuple[str, ...]:
    if not raw_specs:
        return ()
    symbols: list[str] = []
    seen: set[str] = set()
    for raw_spec in raw_specs:
        for raw_item in str(raw_spec).replace(";", ",").split(","):
            item = raw_item.strip()
            if not item or item.lower() == "all":
                continue
            symbol = normalize_base_symbol(item)
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            symbols.append(symbol)
    return tuple(symbols)


def _resolve_symbol_filter(
    exchange: str,
    explicit_symbols: Sequence[str] | None,
    *,
    settings_path: str | Path = DEFAULT_SETTINGS,
) -> tuple[str, ...]:
    if explicit_symbols:
        return _parse_symbol_filter(explicit_symbols)

    cfg = _load_plain_settings(settings_path)
    exchange_id = str(exchange or "").strip().lower()
    keys = []
    if exchange_id:
        keys.extend((f"{exchange_id}_symbols", f"symbols_{exchange_id}"))
    keys.append("symbols")

    for key in keys:
        raw = cfg.get(key)
        if raw is None:
            continue
        if str(raw).strip().lower() == "all":
            return ()
        return _parse_symbol_filter([raw])
    return ()


def _append_symbol_filter_args(command: list[str], symbols: Sequence[str]) -> None:
    parsed = _parse_symbol_filter(list(symbols))
    if parsed:
        command.extend(["--symbols", ",".join(parsed)])


def _append_training_tuning_args(command: list[str], args: Any) -> None:
    command.extend(
        [
            "--feasible-best-seed-frac",
            str(float(args.feasible_best_seed_frac)),
            "--feasible-best-seed-sigma",
            str(float(args.feasible_best_seed_sigma)),
            "--feasible-open-logit-bias",
            str(float(args.feasible_open_logit_bias)),
            "--feasible-mutation-scope",
            str(args.feasible_mutation_scope),
        ]
    )


def _training_window_payload(run_dir: Path, args: Any) -> dict[str, Any]:
    smoke_summary = Path(run_dir) / "smoke_summary.json"
    if smoke_summary.exists():
        try:
            payload = json.loads(smoke_summary.read_text(encoding="utf-8"))
            return {
                "start_date": str(payload.get("start_date") or args.train_start_date),
                "end_date": str(payload.get("end_date") or args.train_end_date),
                "population": int(payload.get("population", args.population)),
                "generations": int(payload.get("generations", args.generations)),
            }
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return {
        "start_date": args.train_start_date,
        "end_date": args.train_end_date,
        "population": int(args.population),
        "generations": int(args.generations),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run or validate a genetics smoke candidate with an OOS baseline gate."
    )
    parser.add_argument("--exchange", default=os.getenv("CRYPTO_EXCHANGE", "BITGET"))
    parser.add_argument("--data-dir", default="Retrodate")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--existing-run-dir", default="")
    parser.add_argument("--baseline-genome", default=str(DEFAULT_BASELINE))
    parser.add_argument("--candidate-genome", default="")
    parser.add_argument(
        "--symbol",
        "--symbols",
        dest="symbol_filter",
        action="append",
        default=None,
        help=(
            "Restrict training/evaluation to these base symbols. "
            "Defaults to exchange-scoped settings symbols when configured."
        ),
    )
    parser.add_argument("--train-start-date", default="2025-10-01")
    parser.add_argument("--train-end-date", default="2026-01-31")
    parser.add_argument("--oos-start-date", default="2026-02-01")
    parser.add_argument("--oos-end-date", default="2026-03-31")
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--feasible-best-seed-frac", type=float, default=0.50)
    parser.add_argument("--feasible-best-seed-sigma", type=float, default=0.02)
    parser.add_argument("--feasible-open-logit-bias", type=float, default=1.50)
    parser.add_argument(
        "--feasible-mutation-scope",
        choices=("all", "output"),
        default="output",
    )
    parser.add_argument("--mode", default="fee_fixed_nextbar")
    parser.add_argument("--min-oos-mean-delta", type=float, default=0.0)
    parser.add_argument("--min-oos-min-ret-delta", type=float, default=0.0)
    parser.add_argument("--min-oos-periods", type=int, default=2)
    parser.add_argument("--max-turnover-rate", type=float, default=0.10)
    parser.add_argument("--max-saturation-rate", type=float, default=0.10)
    parser.add_argument("--max-invalid-open-pressure", type=float, default=0.05)
    parser.add_argument("--max-raw-capacity-bar-rate", type=float, default=0.25)
    parser.add_argument("--max-same-side-open-rate", type=float, default=0.05)
    parser.add_argument("--require-robust-pass", action="store_true")
    parser.add_argument("--summary-out", default="")
    args = parser.parse_args(argv)

    exchange = str(args.exchange or "").strip().upper()
    run_dir = _resolve_path(
        args.existing_run_dir
        or args.run_dir
        or (
            ROOT
            / "Results"
            / "neiro_genetics"
            / exchange
            / f"validated_smoke_{time.strftime('%Y%m%d_%H%M%S')}"
        )
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CRYPTO_EXCHANGE"] = exchange
    data_dir = _resolve_path(args.data_dir)
    env["GENETICS_DATA_DIR"] = str(data_dir)
    symbol_filter = _resolve_symbol_filter(exchange, args.symbol_filter)
    if symbol_filter:
        print("[validated-smoke] symbol_filter=" + ",".join(symbol_filter))

    if not args.existing_run_dir and not args.candidate_genome:
        smoke_args = [
            args.python,
            "tools\\run_genetics_smoke_training.py",
            "--population",
            str(args.population),
            "--generations",
            str(args.generations),
            "--start-date",
            args.train_start_date,
            "--end-date",
            args.train_end_date,
            "--out-dir",
            str(run_dir),
            "--evaluator",
            "cpu-numba",
            "--workers",
            str(args.workers),
            "--position-state-features",
            "source-meta",
            "--source-genome-mismatch",
            "transfer",
            "--staged-mutation-schedule",
        ]
        _append_training_tuning_args(smoke_args, args)
        _append_symbol_filter_args(smoke_args, symbol_filter)
        _run(smoke_args, env=env)

    candidate_genome = _resolve_path(args.candidate_genome or run_dir / "best_genome.npy")
    baseline_genome = _resolve_path(args.baseline_genome)
    if not candidate_genome.exists():
        raise FileNotFoundError(candidate_genome)
    if not baseline_genome.exists():
        raise FileNotFoundError(baseline_genome)

    oos_report_path = run_dir / f"contract_oos_{args.oos_start_date}_{args.oos_end_date}.json"
    eval_args = [
        args.python,
        "tools\\evaluate_genetics_contract.py",
        "--genome",
        str(baseline_genome),
        "--genome",
        str(candidate_genome),
        "--out",
        str(oos_report_path),
        "--start-date",
        args.oos_start_date,
        "--end-date",
        args.oos_end_date,
        "--data-dir",
        str(data_dir),
        "--exchange",
        exchange,
    ]
    _append_symbol_filter_args(eval_args, symbol_filter)
    _run(eval_args, env=env)

    oos_report = json.loads(oos_report_path.read_text(encoding="utf-8"))
    summary = summarize_single_candidate_oos_gate(
        oos_report,
        mode=args.mode,
        min_oos_mean_delta=args.min_oos_mean_delta,
        min_oos_min_ret_delta=args.min_oos_min_ret_delta,
        min_oos_periods=args.min_oos_periods,
        max_turnover_rate=args.max_turnover_rate,
        max_saturation_rate=args.max_saturation_rate,
        max_invalid_open_pressure=args.max_invalid_open_pressure,
        max_raw_capacity_bar_rate=args.max_raw_capacity_bar_rate,
        max_same_side_open_rate=args.max_same_side_open_rate,
        require_robust_pass=args.require_robust_pass,
    )
    summary.update(
        {
            "run_dir": str(run_dir),
            "baseline_genome": str(baseline_genome),
            "candidate_genome": str(candidate_genome),
            "oos_report": str(oos_report_path),
            "training_window": _training_window_payload(run_dir, args),
            "symbol_filter": list(symbol_filter),
            "training_tuning": {
                "feasible_best_seed_frac": float(args.feasible_best_seed_frac),
                "feasible_best_seed_sigma": float(args.feasible_best_seed_sigma),
                "feasible_open_logit_bias": float(args.feasible_open_logit_bias),
                "feasible_mutation_scope": str(args.feasible_mutation_scope),
            },
        }
    )
    summary_path = _resolve_path(args.summary_out) if args.summary_out else run_dir / "validated_smoke_summary.json"
    _write_json(summary_path, summary)
    print(f"validated_summary={summary_path}")
    print(f"promotion_eligible={str(summary['promotion_eligible']).lower()}")
    if summary["promotion_failures"]:
        print("promotion_failures=" + ",".join(summary["promotion_failures"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
