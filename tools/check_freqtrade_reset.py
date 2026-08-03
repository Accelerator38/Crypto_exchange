from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "freqtrade_reset" / "user_data" / "config.json"
STRATEGY_DIR = PROJECT_ROOT / "freqtrade_reset" / "user_data" / "strategies"


def validate(config_path: Path = DEFAULT_CONFIG) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    exchange = config.get("exchange", {})
    no_trade_source = (STRATEGY_DIR / "NoTradeStrategy.py").read_text(
        encoding="utf-8"
    )
    pulse_source = (STRATEGY_DIR / "DeterministicPulseStrategy.py").read_text(
        encoding="utf-8"
    )
    checks = {
        "dry_run": config.get("dry_run") is True,
        "futures": config.get("trading_mode") == "futures",
        "isolated": config.get("margin_mode") == "isolated",
        "single_position": config.get("max_open_trades") == 1,
        "small_fixed_stake": float(config.get("stake_amount", 0.0)) <= 10.0,
        "bitget": exchange.get("name") == "bitget",
        "no_api_key": not str(exchange.get("key", "")).strip(),
        "no_api_secret": not str(exchange.get("secret", "")).strip(),
        "no_api_password": not str(exchange.get("password", "")).strip(),
        "force_entry_disabled": config.get("force_entry_enable") is False,
        "api_server_disabled": config.get("api_server", {}).get("enabled") is False,
        "no_trade_strategy_present": (STRATEGY_DIR / "NoTradeStrategy.py").is_file(),
        "pulse_strategy_present": (
            STRATEGY_DIR / "DeterministicPulseStrategy.py"
        ).is_file(),
        "server_side_stop_configured": (
            '"stoploss_on_exchange": True' in no_trade_source
            and '"stoploss_on_exchange": True' in pulse_source
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": "panteon.freqtrade_reset_check.v1",
        "passed": not failed,
        "checks": checks,
        "failed": failed,
        "orders_enabled": False,
        "promotion_authority": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate isolated Freqtrade reset safety.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = validate(args.config)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
