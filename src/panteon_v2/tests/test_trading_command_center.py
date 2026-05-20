from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from panteon_v2.analysis.trading_command_center import (
    CommandCenterConfig,
    CommandMarketBar,
    DryRunCommandExecutor,
    PortfolioSimulator,
    RetroRiskGate,
    TradingCommand,
    TradingCommandCenter,
    TradingIntent,
)


def _bar(
    index: int,
    price: float,
    *,
    symbol: str = "BTC/USDT",
    volume: float = 1000.0,
) -> CommandMarketBar:
    return CommandMarketBar(
        bar=index,
        timestamp=datetime(2022, 1, 1, tzinfo=timezone.utc) + timedelta(hours=index - 1),
        prices={symbol: price},
        volumes={symbol: volume},
        regime="neutral",
    )


def _command(
    bar: CommandMarketBar,
    *,
    side: str = "buy",
    notional: float = 2.0,
    symbol: str = "BTC/USDT",
    leader: str = "UnitLeader",
) -> TradingCommand:
    return TradingCommand(
        timestamp=bar.timestamp.isoformat(),
        mode="retro_dry_run",
        exchange="retro",
        symbol=symbol,
        side=side,
        order_type="market",
        qty=notional / bar.prices[symbol],
        notional_usd=notional,
        price=bar.prices[symbol],
        source="player",
        leader=leader,
        confidence=0.75,
        reason="unit",
        risk_checks={},
        dry_run=True,
    )


def test_risk_gate_enforces_budget_and_single_order_limits() -> None:
    cfg = CommandCenterConfig(
        initial_capital_usd=20.0,
        max_total_budget_usd=20.0,
        max_single_order_usd=2.0,
        max_open_positions=3,
    )
    portfolio = PortfolioSimulator(cfg)
    gate = RetroRiskGate(cfg)
    bar = _bar(1, 100.0)

    too_large = gate.evaluate(_command(bar, notional=2.01), portfolio, bar)
    assert too_large.allowed is False
    assert too_large.reason == "max_single_order_usd"
    assert too_large.checks["max_single_order_usd"]["allowed"] is False

    ok_command = _command(bar, notional=2.0)
    allowed = gate.evaluate(ok_command, portfolio, bar)
    assert allowed.allowed is True
    gate.record_accepted(ok_command, bar)

    duplicate = gate.evaluate(ok_command, portfolio, bar)
    assert duplicate.allowed is False
    assert duplicate.reason == "duplicate_order_on_same_bar"


def test_risk_gate_rejects_when_budget_is_exhausted() -> None:
    cfg = CommandCenterConfig(
        initial_capital_usd=20.0,
        max_total_budget_usd=20.0,
        max_single_order_usd=2.0,
        max_open_positions=20,
    )
    portfolio = PortfolioSimulator(cfg)
    gate = RetroRiskGate(cfg)

    for index in range(1, 11):
        symbol = f"SYM{index}/USDT"
        bar = _bar(index, 10.0, symbol=symbol)
        command = _command(bar, symbol=symbol, notional=2.0)
        assert gate.evaluate(command, portfolio, bar).allowed is True
        gate.record_accepted(command, bar)
        DryRunCommandExecutor(cfg).execute(command, portfolio, bar)

    extra_bar = _bar(11, 10.0, symbol="EXTRA/USDT")
    rejected = gate.evaluate(_command(extra_bar, symbol="EXTRA/USDT"), portfolio, extra_bar)
    assert rejected.allowed is False
    assert rejected.reason == "max_total_budget_usd"


def test_forced_close_is_allowed_after_daily_loss_kill_switch() -> None:
    cfg = CommandCenterConfig(
        initial_capital_usd=20.0,
        max_total_budget_usd=20.0,
        max_single_order_usd=2.0,
        max_daily_loss_usd=1.0,
        max_drawdown_pct=5.0,
    )
    portfolio = PortfolioSimulator(cfg)
    gate = RetroRiskGate(cfg)
    executor = DryRunCommandExecutor(cfg)
    open_bar = _bar(1, 100.0)
    open_command = _command(open_bar)

    assert gate.evaluate(open_command, portfolio, open_bar).allowed is True
    gate.record_accepted(open_command, open_bar)
    executor.execute(open_command, portfolio, open_bar)

    loss_bar = _bar(2, 40.0)
    portfolio.mark_to_market(loss_bar)
    close_commands = gate.forced_close_commands(portfolio, loss_bar)

    assert len(close_commands) == 1
    assert close_commands[0].side == "close"
    assert close_commands[0].source == "risk_exit"
    assert close_commands[0].dry_run is True
    allowed = gate.evaluate(close_commands[0], portfolio, loss_bar)
    assert allowed.allowed is True


def test_dry_run_executor_never_calls_injected_live_executor() -> None:
    class LiveExecutor:
        def __init__(self) -> None:
            self.called = False

        def send_order(self, *_args, **_kwargs):  # pragma: no cover - must not run
            self.called = True
            raise AssertionError("live executor was called")

    cfg = CommandCenterConfig(initial_capital_usd=20.0)
    live = LiveExecutor()
    executor = DryRunCommandExecutor(cfg, live_executor=live)
    bar = _bar(1, 100.0)
    result = executor.execute(_command(bar), PortfolioSimulator(cfg), bar)

    assert result.status == "filled"
    assert live.called is False


def test_command_center_exports_required_artifacts(tmp_path) -> None:
    class FixedAdapter:
        def decide(self, bar, registry, portfolio):
            if bar.bar == 1:
                return TradingIntent(
                    symbol="BTC/USDT",
                    side="buy",
                    source="player",
                    leader="FixedLeader",
                    confidence=0.9,
                    reason="unit open",
                )
            if bar.bar == 3:
                return TradingIntent(
                    symbol="BTC/USDT",
                    side="close",
                    source="player",
                    leader="FixedLeader",
                    confidence=0.9,
                    reason="unit close",
                )
            return TradingIntent.hold("FixedLeader", "unit hold")

    cfg = CommandCenterConfig(
        initial_capital_usd=20.0,
        max_single_order_usd=2.0,
        max_total_budget_usd=20.0,
        output_dir=tmp_path,
    )
    center = TradingCommandCenter(cfg, decision_adapter=FixedAdapter())
    report = center.run([_bar(1, 100.0), _bar(2, 101.0), _bar(3, 102.0)])

    assert report["risk_violations"] == 0
    assert report["live_executor_called"] is False
    for name in (
        "retro_trade_commands.jsonl",
        "retro_decisions.jsonl",
        "retro_risk_rejections.jsonl",
        "retro_position_snapshots.jsonl",
        "retro_equity_curve.csv",
        "retro_equity_curve.json",
        "retro_daily_summary.json",
        "retro_dashboard.html",
        "command_center_report.md",
        "command_center_report.json",
    ):
        assert (tmp_path / name).exists()

    commands = [
        json.loads(line)
        for line in (tmp_path / "retro_trade_commands.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert commands
    assert {row["dry_run"] for row in commands} == {True}
    assert commands[0]["mode"] == "retro_dry_run"
    assert commands[0]["exchange"] == "retro"
    assert commands[0]["notional_usd"] <= 2.0
