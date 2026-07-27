"""Evaluate one fixed range-transition breakout event on sealed Bitget roots."""

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


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RUNTIME = SRC / "panteon_runtime"
for path in (SRC, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from panteon_v2.policy.evidence_tape import CarryFlowEvidenceTape  # noqa: E402


RANGE_LOOKBACK_BARS = 12
MAX_RANGE_WIDTH = 0.03
MIN_VOLUME_RATIO = 1.30
OI_LOOKBACK_BARS = 3
MIN_OI_EXPANSION = 0.015
HOLD_BARS = 6
STOP_PCT = 0.012
TARGET_PCT = 0.024
EVENT_COOLDOWN_BARS = 6
MIN_CLOSED_TRADES = 10
LCB_Z = 1.6448536269514722
DOWN_REGIMES = frozenset({"bearish", "choppy_down", "crash"})
UP_REGIMES = frozenset({"bullish", "choppy_up"})
RANGE_REGIME = "range_low_vol"


def detect_transition_events(
    *,
    root_label: str,
    samples: Sequence[Mapping[str, Any]],
    symbols: Sequence[str],
    regimes_by_symbol_bar: Mapping[tuple[int, str], str],
    mean_cost_bps: float,
) -> dict[str, Any]:
    raw_events = []
    last_event: dict[tuple[str, str], int] = {}
    start = RANGE_LOOKBACK_BARS
    stop = len(samples) - HOLD_BARS
    for index in range(start, stop):
        bar = index + 1
        for symbol in symbols:
            regime = str(regimes_by_symbol_bar[(bar, str(symbol))])
            previous_regime = str(
                regimes_by_symbol_bar.get((bar - 1, str(symbol)), regime)
            )
            if previous_regime != RANGE_REGIME:
                continue
            if not _complete_window(samples, index=index, symbol=symbol):
                continue
            prior = [
                samples[prior_index]["symbols"][symbol]["market"]
                for prior_index in range(index - RANGE_LOOKBACK_BARS, index)
            ]
            current = samples[index]["symbols"][symbol]
            market = current["market"]
            derivatives = current["derivatives"]
            prior_high = max(float(row["high"]) for row in prior)
            prior_low = min(float(row["low"]) for row in prior)
            midpoint = (prior_high + prior_low) / 2.0
            range_width = (
                (prior_high - prior_low) / midpoint if midpoint > 0.0 else 1.0
            )
            price = float(market["decision_price"])
            direction = (
                "LONG"
                if price > prior_high
                else "SHORT"
                if price < prior_low
                else ""
            )
            if not direction or range_width > MAX_RANGE_WIDTH:
                continue
            if direction == "LONG" and regime in DOWN_REGIMES:
                continue
            if direction == "SHORT" and regime in UP_REGIMES:
                continue
            volume_baseline = statistics.median(
                float(row["volume"]) for row in prior
            )
            volume_ratio = (
                float(market["volume"]) / volume_baseline
                if volume_baseline > 0.0
                else 0.0
            )
            oi_now = float(derivatives.get("open_interest_usdt") or 0.0)
            oi_previous = float(
                samples[index - OI_LOOKBACK_BARS]["symbols"][symbol][
                    "derivatives"
                ].get("open_interest_usdt")
                or 0.0
            )
            oi_change = (
                oi_now / oi_previous - 1.0 if oi_previous > 0.0 else 0.0
            )
            if (
                volume_ratio < MIN_VOLUME_RATIO
                or oi_change < MIN_OI_EXPANSION
            ):
                continue
            key = (str(symbol), direction)
            if bar - last_event.get(key, -10_000) < EVENT_COOLDOWN_BARS:
                continue
            last_event[key] = bar
            boundary = prior_high if direction == "LONG" else prior_low
            breakout_distance_bps = abs(price / boundary - 1.0) * 10_000.0
            score = (
                breakout_distance_bps
                + max(volume_ratio - MIN_VOLUME_RATIO, 0.0) * 5.0
                + max(oi_change - MIN_OI_EXPANSION, 0.0) * 2_500.0
            )
            gross_bps, exit_reason = _forward_label(
                samples=samples,
                index=index,
                symbol=str(symbol),
                direction=direction,
                entry_price=price,
            )
            raw_events.append(
                {
                    "root": root_label,
                    "bar": bar,
                    "timestamp": str(samples[index]["observed_at"]),
                    "symbol": str(symbol),
                    "direction": direction,
                    "regime": regime,
                    "previous_regime": previous_regime,
                    "range_width": range_width,
                    "volume_ratio": volume_ratio,
                    "oi_change": oi_change,
                    "breakout_distance_bps": breakout_distance_bps,
                    "ranking_score": score,
                    "gross_bps": gross_bps,
                    "cost_bps": float(mean_cost_bps),
                    "net_bps": gross_bps - float(mean_cost_bps),
                    "exit_reason": exit_reason,
                }
            )
    selected = []
    for bar in sorted({int(row["bar"]) for row in raw_events}):
        candidates = [row for row in raw_events if int(row["bar"]) == bar]
        selected.append(
            dict(
                max(
                    candidates,
                    key=lambda row: (
                        float(row["ranking_score"]),
                        str(row["symbol"]),
                    ),
                )
            )
        )
    return {
        "raw_events": raw_events,
        "selected_events": selected,
    }


def evaluate_roots(root_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [
        dict(row)
        for result in root_results
        for row in result["selected_events"]
    ]
    aggregate = _metrics([float(row["net_bps"]) for row in selected])
    roots = sorted(str(result["root"]) for result in root_results)
    per_root = {}
    for root in roots:
        values = [
            float(row["net_bps"])
            for row in selected
            if str(row["root"]) == root
        ]
        per_root[root] = _metrics(values)
    per_direction = {}
    for direction in ("LONG", "SHORT"):
        values = [
            float(row["net_bps"])
            for row in selected
            if str(row["direction"]) == direction
        ]
        per_direction[direction] = _metrics(values)
    inactive_roots = [
        root
        for root, metrics in per_root.items()
        if metrics["trades"] == 0
    ]
    root_collapses = [
        root
        for root, metrics in per_root.items()
        if metrics["trades"] > 0
        and (
            metrics["mean_net_bps"] is None
            or metrics["mean_net_bps"] <= 0.0
        )
    ]
    direction_collapses = [
        direction
        for direction, metrics in per_direction.items()
        if metrics["trades"] > 0
        and (
            metrics["mean_net_bps"] is None
            or metrics["mean_net_bps"] <= 0.0
        )
    ]
    failures = []
    if aggregate["trades"] < MIN_CLOSED_TRADES:
        failures.append("closed_trades_below_10")
    if aggregate["mean_net_bps"] is None or aggregate["mean_net_bps"] <= 0.0:
        failures.append("nonpositive_costed_expectancy")
    if aggregate["lcb_95_net_bps"] is None or aggregate["lcb_95_net_bps"] <= 0.0:
        failures.append("nonpositive_lcb")
    failures.extend(f"inactive_root:{root}" for root in inactive_roots)
    failures.extend(
        f"root_expectancy_collapse:{root}" for root in root_collapses
    )
    failures.extend(
        f"direction_expectancy_collapse:{direction}"
        for direction in direction_collapses
    )
    return {
        "aggregate": aggregate,
        "per_root": per_root,
        "per_direction": per_direction,
        "inactive_roots": inactive_roots,
        "root_collapses": root_collapses,
        "direction_collapses": direction_collapses,
        "selected_events": selected,
        "passed": not failures,
        "failures": failures,
    }


def _forward_label(
    *,
    samples: Sequence[Mapping[str, Any]],
    index: int,
    symbol: str,
    direction: str,
    entry_price: float,
) -> tuple[float, str]:
    if direction == "LONG":
        stop_price = entry_price * (1.0 - STOP_PCT)
        target_price = entry_price * (1.0 + TARGET_PCT)
    else:
        stop_price = entry_price * (1.0 + STOP_PCT)
        target_price = entry_price * (1.0 - TARGET_PCT)
    for future_index in range(index + 1, index + HOLD_BARS + 1):
        market = samples[future_index]["symbols"][symbol]["market"]
        if direction == "LONG":
            if float(market["low"]) <= stop_price:
                return -STOP_PCT * 10_000.0, "stop_loss"
            if float(market["high"]) >= target_price:
                return TARGET_PCT * 10_000.0, "take_profit"
        else:
            if float(market["high"]) >= stop_price:
                return -STOP_PCT * 10_000.0, "stop_loss"
            if float(market["low"]) <= target_price:
                return TARGET_PCT * 10_000.0, "take_profit"
    exit_price = float(
        samples[index + HOLD_BARS]["symbols"][symbol]["market"][
            "decision_price"
        ]
    )
    gross = (
        (exit_price - entry_price) / entry_price
        if direction == "LONG"
        else (entry_price - exit_price) / entry_price
    )
    return gross * 10_000.0, "max_holding"


def _complete_window(
    samples: Sequence[Mapping[str, Any]],
    *,
    index: int,
    symbol: str,
) -> bool:
    return all(
        bool(samples[item]["symbols"][symbol]["complete"])
        for item in range(index - RANGE_LOOKBACK_BARS, index + HOLD_BARS + 1)
    )


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


def _regimes_by_symbol_bar(
    activation_trace: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str], str]:
    regimes: dict[tuple[int, str], str] = {}
    for row in activation_trace:
        bar = int(row["bar"])
        symbol = str(row["execution_symbol"])
        regime = str(row["regime"])
        key = (bar, symbol)
        previous = regimes.setdefault(key, regime)
        if previous != regime:
            raise ValueError(f"conflicting regimes at bar {bar}, symbol {symbol}")
    return regimes


def _symbol_transition_count(
    regimes_by_symbol_bar: Mapping[tuple[int, str], str],
) -> int:
    symbols = sorted(symbol for _, symbol in regimes_by_symbol_bar)
    total = 0
    for symbol in set(symbols):
        ordered = [
            regimes_by_symbol_bar[key]
            for key in sorted(regimes_by_symbol_bar)
            if key[1] == symbol
        ]
        total += sum(
            previous == RANGE_REGIME
            and current in DOWN_REGIMES.union(UP_REGIMES)
            for previous, current in zip(ordered, ordered[1:])
        )
    return total


def run(specs: Sequence[tuple[str, Path, Path]]) -> dict[str, Any]:
    root_results = []
    inputs = []
    for label, tape_path, replay_dir in specs:
        tape = CarryFlowEvidenceTape.from_jsonl(tape_path)
        summary_path = replay_dir / "replay_summary.json"
        trace_path = replay_dir / "activation_trace.jsonl"
        summary = _read_json(summary_path)
        trace = _read_jsonl(trace_path)
        regimes = _regimes_by_symbol_bar(trace)
        events = detect_transition_events(
            root_label=label,
            samples=tape.samples,
            symbols=tape.symbols,
            regimes_by_symbol_bar=regimes,
            mean_cost_bps=float(summary["mean_cost_bps"]),
        )
        root_results.append(
            {
                "root": label,
                "symbol_regime_transitions": _symbol_transition_count(regimes),
                "raw_events": events["raw_events"],
                "selected_events": events["selected_events"],
            }
        )
        inputs.append(
            {
                "root": label,
                "tape_path": str(tape_path),
                "tape_sha256": tape.file_sha256,
                "tape_head_sha256": tape.head_sha256,
                "replay_dir": str(replay_dir),
                "replay_summary_sha256": _sha256_file(summary_path),
                "activation_trace_sha256": _sha256_file(trace_path),
            }
        )
    evaluation = evaluate_roots(root_results)
    return {
        "schema_version": "panteon.range_transition_breakout.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "promotion_authority": False,
        "orders_enabled": False,
        "runtime_actor_created": False,
        "event_contract": {
            "range_lookback_bars": RANGE_LOOKBACK_BARS,
            "max_range_width": MAX_RANGE_WIDTH,
            "min_volume_ratio": MIN_VOLUME_RATIO,
            "oi_lookback_bars": OI_LOOKBACK_BARS,
            "min_oi_expansion": MIN_OI_EXPANSION,
            "hold_bars": HOLD_BARS,
            "stop_pct": STOP_PCT,
            "target_pct": TARGET_PCT,
            "event_cooldown_bars": EVENT_COOLDOWN_BARS,
            "max_portfolio_entries_per_bar": 1,
        },
        "inputs": inputs,
        "roots": [
            {
                "root": row["root"],
                "symbol_regime_transitions": row["symbol_regime_transitions"],
                "raw_events": len(row["raw_events"]),
                "selected_events": len(row["selected_events"]),
            }
            for row in root_results
        ],
        "evaluation": evaluation,
        "verdict": (
            "eligible_for_new_prospective_root"
            if evaluation["passed"]
            else "activation_or_expectancy_failure"
        ),
        "next_action": (
            "implement_immutable_research_actor"
            if evaluation["passed"]
            else "do_not_add_runtime_actor"
        ),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    evaluation = report["evaluation"]
    aggregate = evaluation["aggregate"]
    lines = [
        "# Range-transition breakout audit",
        "",
        "## Verdict",
        "",
        f"**{str(report['verdict']).upper()}**",
        "",
        "| Root | Symbol transitions | Raw events | Selected events | Mean net, bps |",
        "|---|---:|---:|---:|---:|",
    ]
    for root in report["roots"]:
        metrics = evaluation["per_root"][root["root"]]
        lines.append(
            f"| {root['root']} | {root['symbol_regime_transitions']} | "
            f"{root['raw_events']} | {root['selected_events']} | "
            f"{_fmt(metrics['mean_net_bps'])} |"
        )
    lines.extend(
        [
            "",
            (
                f"Aggregate: {aggregate['trades']} trades, "
                f"{_fmt(aggregate['mean_net_bps'])} mean net bps, "
                f"{_fmt(aggregate['lcb_95_net_bps'])} LCB bps."
            ),
            "",
            "Direction metrics:",
            "",
            "| Direction | Trades | Mean net, bps | LCB, bps |",
            "|---|---:|---:|---:|",
        ]
    )
    for direction, metrics in evaluation["per_direction"].items():
        lines.append(
            f"| {direction} | {metrics['trades']} | "
            f"{_fmt(metrics['mean_net_bps'])} | "
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
            "No runtime actor, paper order or live order is created.",
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
            "Evaluate one fixed range-transition breakout event on independent "
            "sealed roots."
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
                "roots": report["roots"],
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
