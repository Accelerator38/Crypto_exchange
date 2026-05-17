from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
GENETICS_DIR = ROOT / "Genetics_DL_Agents"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(GENETICS_DIR) not in sys.path:
    sys.path.insert(0, str(GENETICS_DIR))

import crypto_genetics as cg  # noqa: E402
from panteon_v2.analysis.genetics_validation import (  # noqa: E402
    build_rolling_year_folds,
    robust_period_score,
    split_precomp_by_periods,
)


def _torch_device():
    try:
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        return None


def _period_stats(period_rets: List[float]) -> Dict[str, Any]:
    arr = np.asarray(period_rets, dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "n_periods": int(arr.size),
        "mean_ret": float(arr.mean()),
        "median_ret": float(np.median(arr)),
        "std_ret": float(arr.std()),
        "min_ret": float(arr.min()),
        "max_ret": float(arr.max()),
        "positive_period_pct": float((arr > 0.0).mean() * 100.0),
        "best_period_idx": int(arr.argmax()),
        "worst_period_idx": int(arr.argmin()),
    }


def _evaluate_population(population: np.ndarray, precomp: list) -> tuple[np.ndarray, List[List[float]]]:
    device = _torch_device()
    if device is not None:
        return cg.GPUEvaluator(device).evaluate(population, precomp)
    ev = cg.CPUEvaluator(precomp, n_workers=1)
    try:
        return ev.evaluate(population)
    finally:
        shutdown = getattr(ev, "shutdown", None) or getattr(ev, "close", None)
        if shutdown is not None:
            shutdown()


def _infer_position_state_features_enabled(genome_path: Path) -> bool:
    """Infer whether a genome was trained with position-state feature injection."""
    path = Path(genome_path)
    if not path.exists():
        return False
    metadata_candidates = [
        path.with_name(f"{path.stem}_meta.json"),
        path.parent / "best_genome_meta.json",
    ]
    for meta_path in metadata_candidates:
        if not meta_path.exists():
            continue
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "position_state_features_enabled" in payload:
            return bool(payload["position_state_features_enabled"])
    return False


def _contract_metrics_for_genome(
    *,
    genome: np.ndarray,
    precomp: list,
    execution_lag_bars: int,
) -> Dict[str, Any]:
    population = genome[None].astype(np.float32)
    period_rows: list[dict[str, Any]] = []
    turnover_rates: list[float] = []
    saturation_rates: list[float] = []
    invalid_open_logit_pressures: list[float] = []

    for entry in precomp:
        feat, prices, syms, _month, period = entry[:5]
        regime = cg.map_regime_3(entry[6]) if len(entry) > 6 else "neutral"
        suppression_metrics = None
        if (
            cg.POSITION_STATE_FEATURES_ENABLED
            and (not cg.CURRENCY_SELECTION_ENABLED or cg.CURRENCY_LEARNING_ENABLED)
        ):
            actions, suppression_metrics = cg._batch_forward_position_aware_numpy(
                population,
                feat,
                prices,
                return_suppression_metrics=True,
            )
        else:
            actions = cg.GeneticTrainer._batch_forward_numpy(population, feat, prices=prices)
        if cg.CURRENCY_SELECTION_ENABLED:
            actions = cg._apply_currency_selection(
                actions.astype(np.int32),
                feat,
                regime,
                cg.TOP_CURRENCIES_N,
            ).astype(np.int8)
        metrics = cg._action_contract_metrics(actions.astype(np.int32))
        exec_actions = cg._apply_position_aware_action_policy(actions.astype(np.int32))
        exec_metrics = cg._action_contract_metrics(exec_actions.astype(np.int32))
        turnover_rate = float(metrics["turnover_rates"][0])
        saturation_rate = float(
            suppression_metrics["saturation_rates"][0]
            if suppression_metrics is not None
            else metrics["saturation_rates"][0]
        )
        invalid_open_logit_pressure = float(
            suppression_metrics["invalid_open_logit_pressures"][0]
            if suppression_metrics is not None
            else 0.0
        )
        effective_turnover_rate = float(exec_metrics["turnover_rates"][0])
        turnover_rates.append(turnover_rate)
        saturation_rates.append(saturation_rate)
        invalid_open_logit_pressures.append(invalid_open_logit_pressure)
        period_rows.append({
            "period": str(period),
            "regime": str(regime),
            "n_bars": int(feat.shape[0]),
            "n_symbols": int(len(syms)),
            "turnover_rate": turnover_rate,
            "effective_turnover_rate": effective_turnover_rate,
            "saturation_rate": saturation_rate,
            "invalid_open_logit_pressure": invalid_open_logit_pressure,
        })

    turnover_arr = np.asarray(turnover_rates, dtype=np.float64)
    saturation_arr = np.asarray(saturation_rates, dtype=np.float64)
    if turnover_arr.size == 0:
        return {
            "execution_lag_bars": int(execution_lag_bars),
            "turnover_target_rate": float(cg.TURNOVER_TARGET_RATE),
            "mean_turnover_rate": 0.0,
            "max_turnover_rate": 0.0,
            "mean_effective_turnover_rate": 0.0,
            "mean_saturation_rate": 0.0,
            "max_saturation_rate": 0.0,
            "mean_invalid_open_logit_pressure": 0.0,
            "max_invalid_open_logit_pressure": 0.0,
            "periods": [],
        }
    effective_turnover_arr = np.asarray(
        [float(row["effective_turnover_rate"]) for row in period_rows],
        dtype=np.float64,
    )
    invalid_open_pressure_arr = np.asarray(invalid_open_logit_pressures, dtype=np.float64)
    return {
        "execution_lag_bars": int(execution_lag_bars),
        "turnover_target_rate": float(cg.TURNOVER_TARGET_RATE),
        "mean_turnover_rate": float(turnover_arr.mean()),
        "max_turnover_rate": float(turnover_arr.max()),
        "mean_effective_turnover_rate": float(effective_turnover_arr.mean()),
        "mean_saturation_rate": float(saturation_arr.mean()),
        "max_saturation_rate": float(saturation_arr.max()),
        "mean_invalid_open_logit_pressure": float(invalid_open_pressure_arr.mean()),
        "max_invalid_open_logit_pressure": float(invalid_open_pressure_arr.max()),
        "periods": period_rows,
    }


def _evaluate_mode(
    *,
    genome: np.ndarray,
    precomp: list,
    mode_name: str,
    execution_lag_bars: int,
    futures_fee: float,
    position_state_features_enabled: bool,
) -> Dict[str, Any]:
    old_lag = cg.TRAIN_EXECUTION_LAG_BARS
    old_futures_fee = cg.TRAIN_FUTURES_FEE
    old_position_state = cg.POSITION_STATE_FEATURES_ENABLED
    try:
        cg.TRAIN_EXECUTION_LAG_BARS = int(execution_lag_bars)
        cg.TRAIN_FUTURES_FEE = float(futures_fee)
        cg.POSITION_STATE_FEATURES_ENABLED = bool(position_state_features_enabled)
        t0 = time.time()
        fits, period_ret_lists = _evaluate_population(genome[None].astype(np.float32), precomp)
        elapsed = time.time() - t0
        contract_metrics = _contract_metrics_for_genome(
            genome=genome,
            precomp=precomp,
            execution_lag_bars=execution_lag_bars,
        )
    finally:
        cg.TRAIN_EXECUTION_LAG_BARS = old_lag
        cg.TRAIN_FUTURES_FEE = old_futures_fee
        cg.POSITION_STATE_FEATURES_ENABLED = old_position_state

    period_rets = period_ret_lists[0] if period_ret_lists else []
    return {
        "mode": mode_name,
        "position_state_features_enabled": bool(position_state_features_enabled),
        "fitness": float(fits[0]),
        "execution_lag_bars": int(execution_lag_bars),
        "spot_fee": float(cg.TRAIN_FEE),
        "futures_fee": float(futures_fee),
        "slippage": float(cg.TRAIN_SLIPPAGE),
        "elapsed_sec": float(elapsed),
        "period_stats": _period_stats(period_rets),
        "robust_score": robust_period_score(period_rets),
        "contract_metrics": contract_metrics,
        "period_rets": [float(x) for x in period_rets],
    }


def _evaluate_walk_forward(
    *,
    genome: np.ndarray,
    precomp: list,
    folds: list[dict[str, Any]],
    execution_lag_bars: int,
    futures_fee: float,
    position_state_features_enabled: bool,
) -> Dict[str, Any]:
    fold_reports: list[dict[str, Any]] = []
    aggregate_period_rets: list[float] = []

    for fold in folds:
        validation_precomp = split_precomp_by_periods(precomp, fold["validation_periods"])
        if not validation_precomp:
            continue

        validation_report = _evaluate_mode(
            genome=genome,
            precomp=validation_precomp,
            mode_name="walk_forward_validation_fee_fixed_nextbar",
            execution_lag_bars=execution_lag_bars,
            futures_fee=futures_fee,
            position_state_features_enabled=position_state_features_enabled,
        )
        aggregate_period_rets.extend(validation_report["period_rets"])
        fold_reports.append({
            "fold_id": fold["fold_id"],
            "train_years": fold["train_years"],
            "validation_years": fold["validation_years"],
            "n_train_periods": len(fold["train_periods"]),
            "n_embargo_periods": len(fold["embargo_periods"]),
            "n_validation_periods": len(fold["validation_periods"]),
            "validation": validation_report,
        })

    return {
        "n_folds": len(fold_reports),
        "validation_aggregate_stats": _period_stats(aggregate_period_rets),
        "validation_aggregate_robust_score": robust_period_score(aggregate_period_rets),
        "folds": fold_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate saved genetic genomes under cost/timing contracts.")
    parser.add_argument(
        "--genome",
        action="append",
        default=None,
        help="Path to .npy genome. Can be repeated. Defaults to best_genome.npy.",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "Results" / "neiro_genetics" / "genetics_contract_eval.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional evaluation start date passed to the genetics data loader.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional evaluation end date passed to the genetics data loader.",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Also evaluate validation slices from rolling train/validation year folds.",
    )
    parser.add_argument(
        "--train-years",
        type=int,
        default=4,
        help="Number of training years per walk-forward fold.",
    )
    parser.add_argument(
        "--validation-years",
        type=int,
        default=1,
        help="Number of validation years per walk-forward fold.",
    )
    parser.add_argument(
        "--final-test-year",
        type=int,
        default=2025,
        help="Year reserved as final holdout and excluded from walk-forward validation.",
    )
    parser.add_argument(
        "--embargo-months",
        type=int,
        default=0,
        help="Number of final training months removed before each validation window.",
    )
    args = parser.parse_args()

    genome_paths = args.genome or [str(GENETICS_DIR / "Agents" / "genetics" / "best_genome.npy")]
    genomes = []
    for raw_path in genome_paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        genome = np.load(path).astype(np.float32)
        if genome.shape[0] != cg.GENOME_SIZE:
            raise ValueError(f"{path} genome size {genome.shape[0]} != expected {cg.GENOME_SIZE}")
        genomes.append((path, genome, _infer_position_state_features_enabled(path)))

    old_cfg_start = cg._cx._CFG.get("start_date")
    old_cfg_end = cg._cx._CFG.get("end_date")
    try:
        if args.start_date is not None:
            cg._cx._CFG["start_date"] = args.start_date
        if args.end_date is not None:
            cg._cx._CFG["end_date"] = args.end_date

        print("[eval] loading precomputed features...")
        precomp = cg._load_precomp()
    finally:
        if old_cfg_start is not None:
            cg._cx._CFG["start_date"] = old_cfg_start
        if old_cfg_end is not None:
            cg._cx._CFG["end_date"] = old_cfg_end

    if not precomp:
        raise RuntimeError("No precomputed periods loaded")
    print(f"[eval] periods={len(precomp)}")

    modes = [
        {
            "mode_name": "legacy_samebar_spot_fee_for_futures",
            "execution_lag_bars": 0,
            "futures_fee": float(cg.TRAIN_FEE),
        },
        {
            "mode_name": "fee_fixed_samebar",
            "execution_lag_bars": 0,
            "futures_fee": float(cg.TRAIN_FUTURES_FEE),
        },
        {
            "mode_name": "fee_fixed_nextbar",
            "execution_lag_bars": 1,
            "futures_fee": float(cg.TRAIN_FUTURES_FEE),
        },
    ]

    report: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "periods": len(precomp),
        "default_position_state_features_enabled": bool(cg.POSITION_STATE_FEATURES_ENABLED),
        "genomes": [],
    }
    folds: list[dict[str, Any]] = []
    if args.walk_forward:
        period_labels = [str(entry[4]) for entry in precomp if len(entry) > 4]
        folds = build_rolling_year_folds(
            period_labels,
            train_years=args.train_years,
            validation_years=args.validation_years,
            final_test_year=args.final_test_year,
            embargo_months=args.embargo_months,
        )
        report["walk_forward_config"] = {
            "train_years": args.train_years,
            "validation_years": args.validation_years,
            "final_test_year": args.final_test_year,
            "embargo_months": args.embargo_months,
            "n_folds": len(folds),
        }

    for path, genome, position_state_features_enabled in genomes:
        print(f"[eval] genome={path}")
        genome_report = {
            "path": str(path),
            "position_state_features_enabled": bool(position_state_features_enabled),
            "modes": [],
        }
        for mode in modes:
            print(f"[eval]   mode={mode['mode_name']}")
            genome_report["modes"].append(
                _evaluate_mode(
                    genome=genome,
                    precomp=precomp,
                    position_state_features_enabled=position_state_features_enabled,
                    **mode,
                )
            )
        if args.walk_forward:
            print(f"[eval]   walk_forward_folds={len(folds)}")
            genome_report["walk_forward"] = _evaluate_walk_forward(
                genome=genome,
                precomp=precomp,
                folds=folds,
                execution_lag_bars=1,
                futures_fee=float(cg.TRAIN_FUTURES_FEE),
                position_state_features_enabled=position_state_features_enabled,
            )
        report["genomes"].append(genome_report)

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
