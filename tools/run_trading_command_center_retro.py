from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from panteon_v2.analysis.trading_command_center import run_trading_command_center_retro


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline TradingCommandCenter retro dry-run.")
    parser.add_argument("--data-dir", default="Retrodate")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-dir", default="")
    parser.add_argument("--years", default="2022,2023,2024,2025,2026")
    parser.add_argument("--stride-minutes", type=int, default=60)
    parser.add_argument("--min-signal-return-pct", type=float, default=4.0)
    parser.add_argument("--signal-lookback-bars", type=int, default=24)
    parser.add_argument("--take-profit-pct", type=float, default=2.0)
    parser.add_argument("--stop-loss-pct", type=float, default=1.0)
    parser.add_argument("--max-holding-bars", type=int, default=12)
    parser.add_argument("--min-actionable-confidence", type=float, default=0.65)
    args = parser.parse_args()

    years = tuple(int(part.strip()) for part in args.years.split(",") if part.strip())
    report = run_trading_command_center_retro(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        baseline_dir=Path(args.baseline_dir) if args.baseline_dir else None,
        years=years,
        stride_minutes=args.stride_minutes,
        config_overrides={
            "min_signal_return_pct": args.min_signal_return_pct,
            "signal_lookback_bars": args.signal_lookback_bars,
            "take_profit_pct": args.take_profit_pct,
            "stop_loss_pct": args.stop_loss_pct,
            "max_holding_bars": args.max_holding_bars,
            "min_actionable_confidence": args.min_actionable_confidence,
            "run_name": "trading_command_center_retro",
        },
    )
    print(json.dumps({
        "output_dir": str(Path(args.output_dir).resolve()),
        "pnl_usd": report.get("pnl_usd"),
        "pnl_pct": report.get("pnl_pct"),
        "max_drawdown_pct": report.get("max_drawdown_pct"),
        "risk_violations": report.get("risk_violations"),
        "daily_loss_violations": report.get("daily_loss_violations"),
        "live_executor_called": report.get("live_executor_called"),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
