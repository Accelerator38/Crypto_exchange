"""Evaluate one fixed CarryFlow entry model with leave-one-root-out labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RUNTIME = SRC / "panteon_runtime"
for path in (SRC, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from carryflow_policy import get_carryflow_profile  # noqa: E402
from panteon_v2.policy.evidence_tape import CarryFlowEvidenceTape  # noqa: E402


HORIZON_BARS = 6
RIDGE_ALPHA = 10.0
FEATURE_NAMES = (
    "oi_change_bps",
    "price_return_bps",
    "funding_bps",
    "crowding_pct_points",
    "basis_bps",
    "market_positive_breadth",
    "volume_ratio_6",
    "realized_vol_6_bps",
    "trend_12_bps",
    "bar_range_bps",
    "close_location",
    "is_range_low_vol",
)
MIN_TOTAL_TRADES = 10
LCB_Z = 1.6448536269514722


def build_labeled_rows(
    *,
    root_label: str,
    tape: CarryFlowEvidenceTape,
    activation_trace: Sequence[Mapping[str, Any]],
    profile_id: str,
    mean_cost_bps: float,
) -> list[dict[str, Any]]:
    profile = get_carryflow_profile(profile_id)
    samples = list(tape.samples)
    symbols = tuple(tape.symbols)
    expected = len(samples) * len(symbols)
    if len(activation_trace) != expected:
        raise ValueError(
            f"{root_label}: activation trace mismatch "
            f"{len(activation_trace)} != {expected}"
        )
    traces = {
        (int(row["bar"]), str(row["execution_symbol"])): row
        for row in activation_trace
    }
    if len(traces) != expected:
        raise ValueError(f"{root_label}: duplicate activation trace rows")

    oi_history = {symbol: [] for symbol in symbols}
    price_history = {symbol: [] for symbol in symbols}
    volume_history = {symbol: [] for symbol in symbols}
    return_history = {symbol: [] for symbol in symbols}
    rows = []
    for bar, sample in enumerate(samples, start=1):
        symbol_rows = sample["symbols"]
        current_returns: dict[str, float] = {}
        observations: dict[str, dict[str, float]] = {}
        for symbol in symbols:
            row = symbol_rows[symbol]
            derivatives = row["derivatives"]
            market = row["market"]
            price = float(market["decision_price"])
            volume = float(market["volume"])
            oi_now = float(derivatives.get("open_interest_usdt") or 0.0)
            previous_price = (
                float(price_history[symbol][-1])
                if price_history[symbol]
                else price
            )
            one_bar_return = (
                price / previous_price - 1.0 if previous_price > 0.0 else 0.0
            )
            volume_baseline = (
                statistics.fmean(volume_history[symbol][-6:])
                if volume_history[symbol]
                else volume
            )
            volume_ratio = (
                volume / volume_baseline if volume_baseline > 0.0 else 1.0
            )
            price_history[symbol].append(price)
            volume_history[symbol].append(volume)
            return_history[symbol].append(one_bar_return)
            if oi_now > 0.0:
                oi_history[symbol].append(oi_now)
            oi_change = _lookback_return(
                oi_history[symbol], profile.oi_lookback_bars
            )
            price_return = _lookback_return(
                price_history[symbol], profile.price_lookback_bars
            )
            current_returns[symbol] = price_return
            recent_returns = return_history[symbol][-6:]
            realized_vol = (
                statistics.pstdev(recent_returns)
                if len(recent_returns) >= 2
                else 0.0
            )
            trend_12 = _lookback_return(price_history[symbol], 12)
            high = float(market["high"])
            low = float(market["low"])
            bar_range = (
                (high - low) / price * 10_000.0 if price > 0.0 else 0.0
            )
            close_location = (
                (price - low) / (high - low) if high > low else 0.5
            )
            mark = float(derivatives.get("mark_price") or 0.0)
            index = float(derivatives.get("index_price") or 0.0)
            observations[symbol] = {
                "price": price,
                "oi_change_bps": oi_change * 10_000.0,
                "price_return_bps": price_return * 10_000.0,
                "funding_bps": (
                    float(derivatives.get("funding_rate") or 0.0) * 10_000.0
                ),
                "crowding_pct_points": (
                    float(derivatives.get("long_ratio") or 0.5) - 0.5
                )
                * 100.0,
                "basis_bps": (
                    (mark / index - 1.0) * 10_000.0
                    if mark > 0.0 and index > 0.0
                    else 0.0
                ),
                "volume_ratio_6": volume_ratio,
                "realized_vol_6_bps": realized_vol * 10_000.0,
                "trend_12_bps": trend_12 * 10_000.0,
                "bar_range_bps": bar_range,
                "close_location": close_location,
            }
        breadth = (
            sum(value >= 0.0 for value in current_returns.values())
            / len(current_returns)
            if current_returns
            else 0.0
        )
        for symbol in symbols:
            if not _row_eligible(
                bar=bar,
                symbol=symbol,
                samples=samples,
                sample=sample,
                trace=traces[(bar, symbol)],
                required_history=profile.required_history,
                allowed_regimes=profile.allowed_regimes,
            ):
                continue
            observation = observations[symbol]
            gross_bps, exit_reason = _forward_short_label(
                samples=samples,
                bar=bar,
                symbol=symbol,
                entry_price=observation["price"],
                stop_pct=profile.stop_pct,
                target_pct=profile.target_pct,
            )
            rows.append(
                {
                    "root": root_label,
                    "bar": bar,
                    "timestamp": str(sample["observed_at"]),
                    "symbol": symbol,
                    "regime": str(traces[(bar, symbol)]["regime"]),
                    **observation,
                    "market_positive_breadth": breadth,
                    "is_range_low_vol": float(
                        str(traces[(bar, symbol)]["regime"])
                        == "range_low_vol"
                    ),
                    "gross_6h_bps": gross_bps,
                    "cost_bps": float(mean_cost_bps),
                    "net_6h_bps": gross_bps - float(mean_cost_bps),
                    "label_exit_reason": exit_reason,
                }
            )
    return rows


def evaluate_leave_one_root_out(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    roots = tuple(sorted({str(row["root"]) for row in rows}))
    if len(roots) < 3:
        raise ValueError("at least three independent roots are required")
    fold_rows = []
    selected_all = []
    for held_out in roots:
        train = [row for row in rows if str(row["root"]) != held_out]
        test = [row for row in rows if str(row["root"]) == held_out]
        model = _fit_ridge(train)
        predictions = [
            {
                **dict(row),
                "predicted_net_6h_bps": _predict(model, row),
            }
            for row in test
        ]
        selected = _select_one_per_bar(predictions)
        selected_all.extend(selected)
        metrics = _metrics(
            [float(row["net_6h_bps"]) for row in selected]
        )
        fold_rows.append(
            {
                "held_out_root": held_out,
                "train_observations": len(train),
                "test_observations": len(test),
                "selected_trades": len(selected),
                "selected_symbols": dict(
                    sorted(Counter(str(row["symbol"]) for row in selected).items())
                ),
                "metrics": metrics,
                "model": {
                    "intercept_bps": model["intercept"],
                    "coefficients": dict(
                        zip(FEATURE_NAMES, model["coefficients"])
                    ),
                    "train_feature_means": dict(
                        zip(FEATURE_NAMES, model["means"])
                    ),
                    "train_feature_scales": dict(
                        zip(FEATURE_NAMES, model["scales"])
                    ),
                },
            }
        )

    aggregate = _metrics(
        [float(row["net_6h_bps"]) for row in selected_all]
    )
    root_collapses = [
        row["held_out_root"]
        for row in fold_rows
        if row["selected_trades"] == 0
        or row["metrics"]["mean_net_bps"] is None
        or row["metrics"]["mean_net_bps"] <= 0.0
    ]
    failures = []
    if aggregate["trades"] < MIN_TOTAL_TRADES:
        failures.append("closed_trades_below_10")
    if aggregate["mean_net_bps"] is None or aggregate["mean_net_bps"] <= 0.0:
        failures.append("nonpositive_costed_expectancy")
    if aggregate["lcb_95_net_bps"] is None or aggregate["lcb_95_net_bps"] <= 0.0:
        failures.append("nonpositive_lcb")
    failures.extend(f"root_expectancy_collapse:{root}" for root in root_collapses)
    return {
        "model_id": "ridge6_hybrid_market_flow_v1",
        "features": list(FEATURE_NAMES),
        "horizon_bars": HORIZON_BARS,
        "ridge_alpha": RIDGE_ALPHA,
        "prediction_gate": "predicted_net_6h_bps>0",
        "selection": "highest_prediction_per_nonoverlapping_decision_bar",
        "folds": fold_rows,
        "aggregate": aggregate,
        "root_expectancy_collapses": root_collapses,
        "passed": not failures,
        "failures": failures,
        "selected_rows": selected_all,
    }


def _fit_ridge(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("training rows are empty")
    matrix = np.asarray(
        [[float(row[name]) for name in FEATURE_NAMES] for row in rows],
        dtype=float,
    )
    target = np.asarray(
        [float(row["net_6h_bps"]) for row in rows],
        dtype=float,
    )
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    standardized = (matrix - means) / scales
    target_mean = float(target.mean())
    centered_target = target - target_mean
    gram = standardized.T @ standardized
    coefficients = np.linalg.solve(
        gram + RIDGE_ALPHA * np.eye(len(FEATURE_NAMES)),
        standardized.T @ centered_target,
    )
    return {
        "intercept": target_mean,
        "coefficients": coefficients.tolist(),
        "means": means.tolist(),
        "scales": scales.tolist(),
    }


def _predict(model: Mapping[str, Any], row: Mapping[str, Any]) -> float:
    values = np.asarray([float(row[name]) for name in FEATURE_NAMES])
    standardized = (
        values - np.asarray(model["means"])
    ) / np.asarray(model["scales"])
    return float(
        float(model["intercept"])
        + standardized @ np.asarray(model["coefficients"])
    )


def _select_one_per_bar(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected = []
    by_bar: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        if float(row["predicted_net_6h_bps"]) <= 0.0:
            continue
        by_bar.setdefault(int(row["bar"]), []).append(row)
    for bar in sorted(by_bar):
        best = max(
            by_bar[bar],
            key=lambda row: (
                float(row["predicted_net_6h_bps"]),
                str(row["symbol"]),
            ),
        )
        selected.append(dict(best))
    return selected


def _metrics(values: Sequence[float]) -> dict[str, Any]:
    selected = [float(value) for value in values]
    mean = statistics.fmean(selected) if selected else None
    lcb = (
        mean - LCB_Z * statistics.stdev(selected) / math.sqrt(len(selected))
        if len(selected) >= 2 and mean is not None
        else None
    )
    return {
        "trades": len(selected),
        "fills": len(selected) * 2,
        "mean_net_bps": mean,
        "lcb_95_net_bps": lcb,
        "positive_rate": (
            sum(value > 0.0 for value in selected) / len(selected)
            if selected
            else 0.0
        ),
        "net_sum_bps": sum(selected),
    }


def _row_eligible(
    *,
    bar: int,
    symbol: str,
    samples: Sequence[Mapping[str, Any]],
    sample: Mapping[str, Any],
    trace: Mapping[str, Any],
    required_history: int,
    allowed_regimes: Sequence[str],
) -> bool:
    model_history = max(int(required_history), 13)
    if bar < model_history:
        return False
    if (bar - model_history) % HORIZON_BARS != 0:
        return False
    if bar + HORIZON_BARS > len(samples):
        return False
    if str(trace["regime"]) not in set(allowed_regimes):
        return False
    if not bool(sample["symbols"][symbol]["complete"]):
        return False
    return all(
        bool(samples[index]["symbols"][symbol]["complete"])
        for index in range(bar, bar + HORIZON_BARS)
    )


def _forward_short_label(
    *,
    samples: Sequence[Mapping[str, Any]],
    bar: int,
    symbol: str,
    entry_price: float,
    stop_pct: float,
    target_pct: float,
) -> tuple[float, str]:
    stop_price = entry_price * (1.0 + float(stop_pct))
    target_price = entry_price * (1.0 - float(target_pct))
    for future_bar in range(bar + 1, bar + HORIZON_BARS + 1):
        market = samples[future_bar - 1]["symbols"][symbol]["market"]
        if float(market["high"]) >= stop_price:
            return -float(stop_pct) * 10_000.0, "stop_loss"
        if float(market["low"]) <= target_price:
            return float(target_pct) * 10_000.0, "take_profit"
    final_price = float(
        samples[bar + HORIZON_BARS - 1]["symbols"][symbol]["market"][
            "decision_price"
        ]
    )
    return (
        (entry_price - final_price) / entry_price * 10_000.0,
        "max_holding",
    )


def _lookback_return(values: Sequence[float], lookback: int) -> float:
    width = max(1, int(lookback))
    if len(values) <= width or float(values[-1 - width]) <= 0.0:
        return 0.0
    return float(values[-1]) / float(values[-1 - width]) - 1.0


def run(specs: Sequence[tuple[str, Path, Path]]) -> dict[str, Any]:
    all_rows = []
    inputs = []
    profile_ids = set()
    for label, tape_path, replay_dir in specs:
        tape = CarryFlowEvidenceTape.from_jsonl(tape_path)
        summary_path = replay_dir / "replay_summary.json"
        trace_path = replay_dir / "activation_trace.jsonl"
        summary = _read_json(summary_path)
        profile_id = str(
            (summary.get("research_hypothesis") or {}).get("profile_id") or ""
        )
        if not profile_id:
            raise ValueError(f"{label}: replay profile is missing")
        profile_ids.add(profile_id)
        rows = build_labeled_rows(
            root_label=label,
            tape=tape,
            activation_trace=_read_jsonl(trace_path),
            profile_id=profile_id,
            mean_cost_bps=float(summary["mean_cost_bps"]),
        )
        all_rows.extend(rows)
        inputs.append(
            {
                "root": label,
                "tape_path": str(tape_path),
                "tape_sha256": tape.file_sha256,
                "tape_head_sha256": tape.head_sha256,
                "replay_dir": str(replay_dir),
                "replay_summary_sha256": _sha256_file(summary_path),
                "activation_trace_sha256": _sha256_file(trace_path),
                "labeled_rows": len(rows),
            }
        )
    if len(profile_ids) != 1:
        raise ValueError("all roots must use the same source profile")
    evaluation = evaluate_leave_one_root_out(all_rows)
    return {
        "schema_version": "panteon.carryflow_forward_labels.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "promotion_authority": False,
        "orders_enabled": False,
        "runtime_profile_created": False,
        "source_profile_id": next(iter(profile_ids)),
        "label_contract": {
            "direction": "SHORT",
            "horizon_bars": HORIZON_BARS,
            "nonoverlapping_per_symbol": True,
            "stop_and_target_modeled": True,
            "costs": "per-root replay mean_cost_bps",
        },
        "inputs": inputs,
        "dataset": {
            "rows": len(all_rows),
            "roots": dict(
                sorted(Counter(str(row["root"]) for row in all_rows).items())
            ),
            "symbols": dict(
                sorted(Counter(str(row["symbol"]) for row in all_rows).items())
            ),
            "regimes": dict(
                sorted(Counter(str(row["regime"]) for row in all_rows).items())
            ),
        },
        "evaluation": evaluation,
        "verdict": (
            "eligible_for_new_prospective_root"
            if evaluation["passed"]
            else "rejected"
        ),
        "next_action": (
            "encode_one_immutable_profile_then_collect_new_prospective_root"
            if evaluation["passed"]
            else "do_not_modify_runtime_profile"
        ),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    evaluation = report["evaluation"]
    aggregate = evaluation["aggregate"]
    lines = [
        "# CarryFlow forward-label entry evaluation",
        "",
        "## Verdict",
        "",
        f"**{str(report['verdict']).upper()}**",
        "",
        (
            f"Rows: {report['dataset']['rows']}; selected held-out trades: "
            f"{aggregate['trades']}; mean net: "
            f"{_fmt(aggregate['mean_net_bps'])} bps; LCB: "
            f"{_fmt(aggregate['lcb_95_net_bps'])} bps."
        ),
        "",
        "| Held-out root | Test rows | Trades | Mean net, bps | LCB, bps |",
        "|---|---:|---:|---:|---:|",
    ]
    for fold in evaluation["folds"]:
        metrics = fold["metrics"]
        lines.append(
            f"| {fold['held_out_root']} | {fold['test_observations']} | "
            f"{metrics['trades']} | {_fmt(metrics['mean_net_bps'])} | "
            f"{_fmt(metrics['lcb_95_net_bps'])} |"
        )
    lines.extend(
        [
            "",
            "Failures: "
            + (
                ", ".join(evaluation["failures"])
                if evaluation["failures"]
                else "none"
            ),
            "",
            (
                "The model is trained only on other roots for every row. "
                "Labels are non-overlapping per symbol and include modeled "
                "costs plus the profile stop/target."
            ),
            "",
            "No runtime profile, paper order or live order is created.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2f}"


def _parse_spec(value: str) -> tuple[str, Path, Path]:
    label, separator, remainder = str(value).partition("=")
    tape_raw, pair_separator, replay_raw = remainder.partition("::")
    if (
        not separator
        or not pair_separator
        or not label.strip()
        or not tape_raw.strip()
        or not replay_raw.strip()
    ):
        raise ValueError("--root must use LABEL=TAPE::REPLAY_DIR")
    tape = Path(tape_raw).resolve()
    replay = Path(replay_raw).resolve()
    if not tape.is_file():
        raise FileNotFoundError(tape)
    if not replay.is_dir():
        raise FileNotFoundError(replay)
    return label.strip(), tape, replay


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"JSONL objects required: {path}")
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build fixed six-hour CarryFlow labels and evaluate one ridge "
            "entry model with leave-one-root-out validation."
        )
    )
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)
    specs = [_parse_spec(value) for value in args.root]
    if len({label for label, _, _ in specs}) != len(specs):
        raise ValueError("root labels must be unique")
    report = run(specs)
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "dataset": report["dataset"],
                "aggregate": report["evaluation"]["aggregate"],
                "failures": report["evaluation"]["failures"],
                "output_json": str(output_json),
                "output_md": str(output_md),
                "orders_enabled": False,
                "promotion_authority": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
