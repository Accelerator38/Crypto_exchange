from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from panteon_v2.analysis.genetic_core_contracts import exchange_cost_profile  # noqa: E402


DEFAULT_MODE = "fee_fixed_nextbar"


def reprice_report_costs(
    report: Mapping[str, Any],
    *,
    exchange: str | None = None,
    mode: str = DEFAULT_MODE,
    cost_model: str = "turnover",
) -> dict[str, Any]:
    """Reprice genetics report returns with the configured exchange cost profile."""

    exchange_name = str(exchange or report.get("exchange") or "").strip().upper()
    if not exchange_name:
        raise ValueError("exchange is required for genetics report repricing")
    if cost_model != "turnover":
        raise ValueError(f"unsupported cost model: {cost_model!r}")

    profile = exchange_cost_profile(exchange_name)
    fee_cost_pct = (
        float(profile.fees_bps)
        + float(profile.funding_bps)
        + float(profile.spread_bps)
    ) / 100.0
    slippage_pct = float(profile.slippage_bps) / 100.0
    repriced = copy.deepcopy(dict(report))
    repriced["exchange"] = exchange_name

    for genome in _mapping_items(repriced.get("genomes")):
        for mode_payload in _mapping_items(genome.get("modes")):
            if mode and str(mode_payload.get("mode") or "") != mode:
                continue
            _reprice_mode(
                mode_payload,
                fee_cost_pct=fee_cost_pct,
                slippage_pct=slippage_pct,
            )

    repriced["cost_reprice"] = {
        "exchange": exchange_name,
        "cost_model": cost_model,
        "fee_cost_pct_per_turnover": fee_cost_pct,
        "slippage_pct_per_turnover": slippage_pct,
        "fees_bps": float(profile.fees_bps),
        "funding_bps": float(profile.funding_bps),
        "spread_bps": float(profile.spread_bps),
        "slippage_bps": float(profile.slippage_bps),
    }
    return repriced


def _reprice_mode(
    mode_payload: dict[str, Any],
    *,
    fee_cost_pct: float,
    slippage_pct: float,
) -> None:
    period_rets = [_finite_float(value) for value in mode_payload.get("period_rets", [])]
    if not period_rets:
        return

    contract_metrics = mode_payload.get("contract_metrics")
    if not isinstance(contract_metrics, dict):
        contract_metrics = {}
        mode_payload["contract_metrics"] = contract_metrics
    periods = contract_metrics.get("periods")
    if not isinstance(periods, list):
        periods = [{} for _ in period_rets]
        contract_metrics["periods"] = periods
    while len(periods) < len(period_rets):
        periods.append({})

    repriced_rets: list[float] = []
    costs: list[float] = []
    slippages: list[float] = []
    for index, current_ret in enumerate(period_rets):
        period = periods[index]
        if not isinstance(period, dict):
            period = {}
            periods[index] = period

        turnover = _turnover_basis(period)
        previous_cost = _finite_float(period.get("cost_pct"))
        previous_slippage = _finite_float(period.get("slippage_pct"))
        new_cost = _round_metric(fee_cost_pct * turnover)
        new_slippage = _round_metric(slippage_pct * turnover)
        incremental_cost = (new_cost + new_slippage) - (
            previous_cost + previous_slippage
        )

        period["cost_pct"] = new_cost
        period["slippage_pct"] = new_slippage
        period["reprice_turnover_basis"] = _round_metric(turnover)
        repriced_rets.append(_round_metric(current_ret - incremental_cost))
        costs.append(new_cost)
        slippages.append(new_slippage)

    mode_payload["period_rets"] = repriced_rets
    mode_payload["period_stats"] = _period_stats(repriced_rets)
    contract_metrics["mean_cost_pct"] = _mean(costs)
    contract_metrics["mean_slippage_pct"] = _mean(slippages)
    contract_metrics["max_cost_pct"] = max(costs) if costs else 0.0
    contract_metrics["max_slippage_pct"] = max(slippages) if slippages else 0.0
    contract_metrics["cost_model"] = "turnover"


def _mapping_items(value: Any) -> Iterable[dict[str, Any]]:
    if not isinstance(value, list):
        return ()
    return (item for item in value if isinstance(item, dict))


def _turnover_basis(period: Mapping[str, Any]) -> float:
    candidates = (
        period.get("effective_turnover_rate"),
        period.get("turnover_rate"),
        period.get("turnover"),
    )
    values = [_finite_float(value) for value in candidates]
    positive = [value for value in values if value > 0.0]
    return max(positive) if positive else 1.0


def _period_stats(period_rets: list[float]) -> dict[str, Any]:
    if not period_rets:
        return {}
    mean = _mean(period_rets)
    sorted_rets = sorted(period_rets)
    mid = len(sorted_rets) // 2
    if len(sorted_rets) % 2:
        median = sorted_rets[mid]
    else:
        median = (sorted_rets[mid - 1] + sorted_rets[mid]) / 2.0
    variance = sum((value - mean) ** 2 for value in period_rets) / len(period_rets)
    return {
        "n_periods": len(period_rets),
        "mean_ret": mean,
        "median_ret": _round_metric(median),
        "std_ret": _round_metric(math.sqrt(variance)),
        "min_ret": min(period_rets),
        "max_ret": max(period_rets),
        "positive_period_pct": sum(1 for value in period_rets if value > 0.0)
        / len(period_rets)
        * 100.0,
        "best_period_idx": max(range(len(period_rets)), key=period_rets.__getitem__),
        "worst_period_idx": min(range(len(period_rets)), key=period_rets.__getitem__),
    }


def _mean(values: list[float]) -> float:
    return _round_metric(sum(values) / len(values)) if values else 0.0


def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _round_metric(value: float) -> float:
    return round(float(value), 12)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reprice GeneticCore report costs with real exchange fee/slippage."
    )
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--exchange", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mode", default=DEFAULT_MODE)
    args = parser.parse_args(argv)

    report = _load_json(args.report)
    repriced = reprice_report_costs(
        report,
        exchange=args.exchange,
        mode=args.mode,
    )
    _write_json(args.out, repriced)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
