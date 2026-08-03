from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

from .strategy_lab import MarketFrame


PORTFOLIO_FAILURE_DECOMPOSITION_SCHEMA_VERSION = (
    "panteon.portfolio_failure_decomposition.v1"
)


class PortfolioFailureDecompositionError(ValueError):
    pass


def verify_development_replay_parity(
    sealed_report: Mapping[str, Any],
    replay_report: Mapping[str, Any],
) -> dict[str, Any]:
    sealed_projection = _deterministic_projection(sealed_report)
    replay_projection = _deterministic_projection(replay_report)
    sealed_sha = _canonical_sha256(sealed_projection)
    replay_sha = _canonical_sha256(replay_projection)
    if sealed_sha != replay_sha:
        raise PortfolioFailureDecompositionError(
            "immutable development replay parity mismatch"
        )
    return {
        "passed": True,
        "sealed_projection_sha256": sealed_sha,
        "replay_projection_sha256": replay_sha,
    }


def build_portfolio_failure_decomposition(
    evaluation: Mapping[str, Any],
    *,
    frames: Sequence[MarketFrame],
    source_report_sha256: str,
    replay_parity: Mapping[str, Any],
) -> dict[str, Any]:
    if evaluation.get("candidate_id") != (
        "weekly_top2_bottom2_relative_momentum_4h_v1"
    ):
        raise PortfolioFailureDecompositionError(
            "unexpected portfolio candidate"
        )
    if evaluation.get("validation_opened") is not False:
        raise PortfolioFailureDecompositionError(
            "diagnostic cannot consume sealed windows"
        )
    candidate = evaluation["candidate"]
    trades = list(candidate.get("trades") or [])
    leg_trades = list(candidate.get("leg_trades") or [])
    if not trades or len(leg_trades) != len(trades) * 4:
        raise PortfolioFailureDecompositionError(
            "portfolio replay has incomplete raw ledgers"
        )

    benchmark_bps = [
        _equal_weight_market_return_bps(trade, frames=frames)
        for trade in trades
    ]
    portfolio_bps = [float(trade["net_bps"]) for trade in trades]
    beta = _linear_exposure(benchmark_bps, portfolio_bps)
    tail = _tail_decomposition(trades)
    drawdown = _drawdown_interval(trades)
    symbols = _contribution_rows(leg_trades, key="symbol")
    directions = _contribution_rows(leg_trades, key="direction")
    years = _year_rows(trades)
    candidate_metrics = candidate["metrics"]
    baseline_metrics = evaluation["baseline"]["metrics"]
    mean_gross_bps = float(candidate_metrics["mean_gross_bps"])
    real_cost_bps = float(candidate["real_cost_bps_per_leg"])

    symbol_abs_total = sum(
        abs(float(row["net_pnl_usd"])) for row in symbols.values()
    )
    largest_symbol_share = max(
        (
            abs(float(row["net_pnl_usd"])) / symbol_abs_total
            for row in symbols.values()
        ),
        default=0.0,
    )
    return {
        "schema_version": PORTFOLIO_FAILURE_DECOMPOSITION_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": evaluation["candidate_id"],
        "profile_sha256": evaluation["profile_sha256"],
        "source_report_sha256": str(source_report_sha256),
        "source_verdict": evaluation["verdict"],
        "replay_parity": dict(replay_parity),
        "observations": {
            "closed_portfolios": len(trades),
            "leg_trades": len(leg_trades),
            "first_entry_timestamp_ms": int(
                trades[0]["entry_timestamp_ms"]
            ),
            "last_exit_timestamp_ms": int(trades[-1]["exit_timestamp_ms"]),
        },
        "costs": {
            "mean_gross_bps": mean_gross_bps,
            "mean_net_bps": float(candidate_metrics["mean_net_bps"]),
            "round_trip_cost_bps_per_leg": real_cost_bps,
            "cost_share_of_abs_mean_gross": (
                real_cost_bps / abs(mean_gross_bps)
                if mean_gross_bps
                else None
            ),
        },
        "market_exposure": beta,
        "tail": tail,
        "drawdown": drawdown,
        "per_symbol": symbols,
        "per_direction": directions,
        "per_year": years,
        "concentration": {
            "largest_symbol_abs_pnl_share": largest_symbol_share,
            "worst_5_loss_share": tail["worst_5_loss_share"],
            "best_5_profit_share": tail["best_5_profit_share"],
        },
        "ranking_comparison": {
            "candidate_mean_net_bps": candidate_metrics["mean_net_bps"],
            "candidate_lcb_95_net_bps": candidate_metrics[
                "lcb_95_net_bps"
            ],
            "baseline_mean_net_bps": baseline_metrics["mean_net_bps"],
            "baseline_lcb_95_net_bps": baseline_metrics[
                "lcb_95_net_bps"
            ],
            "volatility_adjusted_rank_harmed_point_estimate": (
                float(candidate_metrics["mean_net_bps"])
                < float(baseline_metrics["mean_net_bps"])
            ),
        },
        "diagnostic_flags": {
            "cost_dominant": (
                real_cost_bps / abs(mean_gross_bps) >= 0.5
                if mean_gross_bps
                else True
            ),
            "market_beta_material": (
                abs(float(beta["beta"])) >= 0.2
                or abs(float(beta["correlation"])) >= 0.2
            ),
            "symbol_concentration_material": largest_symbol_share >= 0.25,
            "tail_loss_concentration_material": (
                float(tail["worst_5_loss_share"]) >= 0.5
            ),
            "ranking_transform_underperformed_baseline": (
                float(candidate_metrics["mean_net_bps"])
                < float(baseline_metrics["mean_net_bps"])
            ),
        },
        "continuation_allowed": False,
        "profile_registration_allowed": False,
        "runtime_actor_created": False,
        "validation_opened": False,
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }


def write_portfolio_failure_decomposition(
    report: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "portfolio_failure_decomposition.json"
    markdown_path = target / "portfolio_failure_decomposition.md"
    json_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        _render_markdown(report),
        encoding="utf-8",
    )
    return json_path, markdown_path


def _deterministic_projection(report: Mapping[str, Any]) -> Any:
    return _strip_nondeterministic(report)


def _strip_nondeterministic(value: Any, *, key: str = "") -> Any:
    if isinstance(value, Mapping):
        return {
            name: _strip_nondeterministic(item, key=name)
            for name, item in value.items()
            if name not in {"generated_at", "trades", "leg_trades"}
        }
    if isinstance(value, list):
        return [_strip_nondeterministic(item, key=key) for item in value]
    return value


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _equal_weight_market_return_bps(
    trade: Mapping[str, Any],
    *,
    frames: Sequence[MarketFrame],
) -> float:
    entry_index = int(trade["entry_index"])
    exit_index = int(trade["exit_index"])
    if not (0 <= entry_index < exit_index < len(frames)):
        raise PortfolioFailureDecompositionError(
            "portfolio trade frame indexes are invalid"
        )
    entry = frames[entry_index]
    exit_frame = frames[exit_index]
    if set(entry.bars) != set(exit_frame.bars):
        raise PortfolioFailureDecompositionError(
            "market benchmark frame symbols mismatch"
        )
    returns = [
        exit_frame.bars[symbol].open / entry.bars[symbol].open - 1.0
        for symbol in sorted(entry.bars)
    ]
    return mean(returns) * 10_000.0


def _linear_exposure(
    benchmark: Sequence[float],
    portfolio: Sequence[float],
) -> dict[str, float]:
    if len(benchmark) != len(portfolio) or len(benchmark) < 2:
        raise PortfolioFailureDecompositionError(
            "market exposure sample is invalid"
        )
    benchmark_mean = mean(benchmark)
    portfolio_mean = mean(portfolio)
    benchmark_delta = [value - benchmark_mean for value in benchmark]
    portfolio_delta = [value - portfolio_mean for value in portfolio]
    covariance = sum(
        left * right
        for left, right in zip(benchmark_delta, portfolio_delta)
    ) / (len(benchmark) - 1)
    benchmark_variance = sum(value * value for value in benchmark_delta) / (
        len(benchmark) - 1
    )
    portfolio_variance = sum(value * value for value in portfolio_delta) / (
        len(portfolio) - 1
    )
    beta = covariance / benchmark_variance if benchmark_variance > 0.0 else 0.0
    denominator = math.sqrt(benchmark_variance * portfolio_variance)
    correlation = covariance / denominator if denominator > 0.0 else 0.0
    alpha = portfolio_mean - beta * benchmark_mean
    return {
        "benchmark_mean_bps": benchmark_mean,
        "portfolio_mean_net_bps": portfolio_mean,
        "beta": beta,
        "correlation": correlation,
        "alpha_intercept_bps": alpha,
    }


def _tail_decomposition(
    trades: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(trades, key=lambda row: float(row["net_bps"]))
    losses = [float(row["net_pnl_usd"]) for row in trades if row["net_pnl_usd"] < 0]
    profits = [
        float(row["net_pnl_usd"]) for row in trades if row["net_pnl_usd"] > 0
    ]
    worst = ordered[:5]
    best = list(reversed(ordered[-5:]))
    total_loss = -sum(losses)
    total_profit = sum(profits)
    return {
        "negative_portfolios": len(losses),
        "positive_portfolios": len(profits),
        "total_loss_usd": total_loss,
        "total_profit_usd": total_profit,
        "worst_5_loss_share": (
            -sum(float(row["net_pnl_usd"]) for row in worst) / total_loss
            if total_loss > 0.0
            else 0.0
        ),
        "best_5_profit_share": (
            sum(float(row["net_pnl_usd"]) for row in best) / total_profit
            if total_profit > 0.0
            else 0.0
        ),
        "worst_5": [_trade_summary(row) for row in worst],
        "best_5": [_trade_summary(row) for row in best],
    }


def _drawdown_interval(
    trades: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    equity = peak = max_drawdown = 0.0
    peak_index = -1
    drawdown_start = drawdown_end = -1
    for index, trade in enumerate(trades):
        equity += float(trade["net_pnl_usd"])
        if equity > peak:
            peak = equity
            peak_index = index
        drawdown = peak - equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            drawdown_start = peak_index + 1
            drawdown_end = index
    interval = (
        trades[drawdown_start:drawdown_end + 1]
        if drawdown_start >= 0
        else []
    )
    return {
        "max_drawdown_usd": max_drawdown,
        "start_trade_index": drawdown_start,
        "end_trade_index": drawdown_end,
        "interval_portfolios": len(interval),
        "start_entry_timestamp_ms": (
            int(interval[0]["entry_timestamp_ms"]) if interval else None
        ),
        "end_exit_timestamp_ms": (
            int(interval[-1]["exit_timestamp_ms"]) if interval else None
        ),
        "interval_net_pnl_usd": sum(
            float(row["net_pnl_usd"]) for row in interval
        ),
    }


def _contribution_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    key: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return {
        label: {
            "trades": len(items),
            "mean_net_bps": mean(float(item["net_bps"]) for item in items),
            "net_pnl_usd": sum(
                float(item["net_pnl_usd"]) for item in items
            ),
            "positive_rate": sum(
                float(item["net_bps"]) > 0.0 for item in items
            )
            / len(items),
        }
        for label, items in sorted(grouped.items())
    }


def _year_rows(
    trades: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for trade in trades:
        year = str(
            datetime.fromtimestamp(
                int(trade["entry_timestamp_ms"]) / 1000.0,
                tz=timezone.utc,
            ).year
        )
        grouped.setdefault(year, []).append(trade)
    result: dict[str, dict[str, Any]] = {}
    for year, items in sorted(grouped.items()):
        values = [float(item["net_bps"]) for item in items]
        lcb = (
            mean(values) - 1.96 * stdev(values) / math.sqrt(len(values))
            if len(values) >= 2
            else None
        )
        result[year] = {
            "portfolios": len(items),
            "mean_net_bps": mean(values),
            "lcb_95_net_bps": lcb,
            "net_pnl_usd": sum(
                float(item["net_pnl_usd"]) for item in items
            ),
        }
    return result


def _trade_summary(trade: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "entry_timestamp_ms": int(trade["entry_timestamp_ms"]),
        "exit_timestamp_ms": int(trade["exit_timestamp_ms"]),
        "long_symbols": list(trade["long_symbols"]),
        "short_symbols": list(trade["short_symbols"]),
        "net_bps": float(trade["net_bps"]),
        "net_pnl_usd": float(trade["net_pnl_usd"]),
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    exposure = report["market_exposure"]
    tail = report["tail"]
    drawdown = report["drawdown"]
    ranking = report["ranking_comparison"]
    flags = report["diagnostic_flags"]
    lines = [
        "# P5 portfolio failure decomposition",
        "",
        "## Safety",
        "",
        "- Source verdict: `terminal_rejected_development`",
        "- Immutable replay parity: `passed`",
        "- Validation/OOS/sanity opened: `false`",
        "- Continuation/paper/live allowed: `false`",
        "",
        "## Root-cause indicators",
        "",
        f"- Market beta: `{exposure['beta']:.4f}`",
        f"- Market correlation: `{exposure['correlation']:.4f}`",
        f"- Alpha intercept: `{exposure['alpha_intercept_bps']:.4f} bps`",
        f"- Worst five loss share: `{tail['worst_5_loss_share']:.2%}`",
        f"- Best five profit share: `{tail['best_5_profit_share']:.2%}`",
        f"- Maximum drawdown: `${drawdown['max_drawdown_usd']:.4f}`",
        "- Candidate vs raw baseline mean: "
        f"`{ranking['candidate_mean_net_bps']:.4f}` vs "
        f"`{ranking['baseline_mean_net_bps']:.4f} bps`",
        "",
        "Flags: "
        + ", ".join(
            f"`{name}={str(value).lower()}`"
            for name, value in sorted(flags.items())
        ),
        "",
        "## Year stability",
        "",
        "| Year | Portfolios | Mean net bps | LCB bps | Net PnL USD |",
        "|---|---:|---:|---:|---:|",
    ]
    for year, row in report["per_year"].items():
        lines.append(
            f"| {year} | {row['portfolios']} | "
            f"{row['mean_net_bps']:.4f} | "
            f"{row['lcb_95_net_bps']:.4f} | "
            f"{row['net_pnl_usd']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Direction contribution",
            "",
            "| Direction | Legs | Mean net bps | Net PnL USD |",
            "|---|---:|---:|---:|",
        ]
    )
    for direction, row in report["per_direction"].items():
        lines.append(
            f"| {direction} | {row['trades']} | "
            f"{row['mean_net_bps']:.4f} | {row['net_pnl_usd']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Symbol contribution",
            "",
            "| Symbol | Legs | Mean net bps | Net PnL USD |",
            "|---|---:|---:|---:|",
        ]
    )
    for symbol, row in report["per_symbol"].items():
        lines.append(
            f"| {symbol} | {row['trades']} | "
            f"{row['mean_net_bps']:.4f} | {row['net_pnl_usd']:.4f} |"
        )
    return "\n".join(lines) + "\n"
