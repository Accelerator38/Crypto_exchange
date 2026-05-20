"""Offline retro dry-run trading command center.

This module is intentionally isolated from live exchange adapters.  It turns
causal retro market bars into structured trading commands, runs deterministic
risk checks, simulates a small portfolio, and writes audit artifacts.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence


_PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
    "0000000c4944415408d763f8ffff3f0005fe02fea73581e90000000049454e44ae426082"
)


@dataclass(frozen=True)
class CommandCenterConfig:
    initial_capital_usd: float = 20.0
    max_total_budget_usd: float = 20.0
    max_single_order_usd: float = 2.0
    max_daily_loss_usd: float = 1.0
    max_drawdown_pct: float = 12.0
    max_open_positions: int = 5
    min_liquidity_usd: float = 0.0
    max_stale_seconds: float = 90.0
    min_actionable_confidence: float = 0.55
    default_order_notional_usd: float = 2.0
    fee_rate: float = 0.0006
    slippage_pct: float = 0.0003
    signal_lookback_bars: int = 24
    min_signal_return_pct: float = 2.5
    min_current_bar_return_pct: float = 0.0
    take_profit_pct: float = 2.0
    stop_loss_pct: float = 1.0
    max_holding_bars: int = 12
    symbols: tuple[str, ...] = ()
    output_dir: Optional[Path | str] = None
    run_name: str = "trading_command_center"

    def __post_init__(self) -> None:
        if self.initial_capital_usd <= 0:
            raise ValueError("initial_capital_usd must be > 0")
        if self.max_total_budget_usd <= 0:
            raise ValueError("max_total_budget_usd must be > 0")
        if self.max_single_order_usd <= 0:
            raise ValueError("max_single_order_usd must be > 0")
        if self.max_single_order_usd > self.max_total_budget_usd:
            raise ValueError("max_single_order_usd must be <= max_total_budget_usd")
        if self.max_daily_loss_usd < 0:
            raise ValueError("max_daily_loss_usd must be >= 0")
        if self.max_drawdown_pct < 0:
            raise ValueError("max_drawdown_pct must be >= 0")
        if self.max_open_positions < 0:
            raise ValueError("max_open_positions must be >= 0")
        if self.default_order_notional_usd <= 0:
            raise ValueError("default_order_notional_usd must be > 0")
        object.__setattr__(
            self,
            "symbols",
            tuple(str(sym).upper() for sym in self.symbols if str(sym).strip()),
        )


@dataclass(frozen=True)
class CommandMarketBar:
    bar: int
    timestamp: datetime
    prices: Mapping[str, float]
    volumes: Mapping[str, float] = field(default_factory=dict)
    regime: str = "neutral"

    def price(self, symbol: str) -> float:
        return float(self.prices.get(symbol, 0.0) or 0.0)

    def volume(self, symbol: str) -> float:
        return float(self.volumes.get(symbol, 0.0) or 0.0)


@dataclass(frozen=True)
class TradingIntent:
    symbol: str
    side: str
    source: str
    leader: str
    confidence: float
    reason: str
    order_type: str = "market"

    @classmethod
    def hold(cls, leader: str, reason: str, *, source: str = "panteon") -> "TradingIntent":
        return cls(
            symbol="",
            side="hold",
            source=source,
            leader=leader or "NoTrade",
            confidence=0.0,
            reason=reason,
        )


@dataclass(frozen=True)
class TradingCommand:
    timestamp: str
    mode: str
    exchange: str
    symbol: str
    side: str
    order_type: str
    qty: float
    notional_usd: float
    price: float
    source: str
    leader: str
    confidence: float
    reason: str
    risk_checks: Mapping[str, Any]
    dry_run: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "mode": self.mode,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "qty": float(self.qty),
            "notional_usd": float(self.notional_usd),
            "price": float(self.price),
            "source": self.source,
            "leader": self.leader,
            "confidence": float(self.confidence),
            "reason": self.reason,
            "risk_checks": dict(self.risk_checks or {}),
            "dry_run": bool(self.dry_run),
        }


@dataclass(frozen=True)
class RiskGateResult:
    allowed: bool
    reason: str = ""
    checks: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DryRunExecutionResult:
    status: str
    command: TradingCommand
    realized_pnl_usd: float = 0.0
    fee_usd: float = 0.0
    reason: str = ""


@dataclass
class SimulatedPosition:
    symbol: str
    side: str
    qty: float
    entry_price: float
    reserved_notional_usd: float
    fee_open_usd: float
    opened_bar: int
    opened_at: str
    leader: str
    source: str

    def unrealized_pnl(self, price: float) -> float:
        if price <= 0:
            return 0.0
        if self.side == "long":
            return self.qty * price - self.reserved_notional_usd
        return self.reserved_notional_usd + (self.entry_price - price) * self.qty - self.reserved_notional_usd


class PortfolioSimulator:
    def __init__(self, config: CommandCenterConfig):
        self.config = config
        self.initial_capital_usd = float(config.initial_capital_usd)
        self.cash_usd = float(config.initial_capital_usd)
        self.positions: dict[str, SimulatedPosition] = {}
        self.equity_usd = float(config.initial_capital_usd)
        self.peak_equity_usd = float(config.initial_capital_usd)
        self.max_drawdown_pct = 0.0
        self.current_prices: dict[str, float] = {}
        self.equity_curve: list[dict[str, Any]] = []
        self.realized_pnl_usd = 0.0
        self.fees_paid_usd = 0.0
        self.closed_trades = 0
        self.winning_trades = 0
        self.daily_realized_pnl: dict[str, float] = defaultdict(float)
        self.daily_start_equity: dict[str, float] = {}
        self.daily_min_equity: dict[str, float] = {}
        self.leader_realized_pnl: dict[str, float] = defaultdict(float)
        self.leader_trades: dict[str, int] = defaultdict(int)
        self.last_timestamp = ""

    @property
    def open_exposure_usd(self) -> float:
        return sum(pos.reserved_notional_usd for pos in self.positions.values())

    @property
    def open_count(self) -> int:
        return len(self.positions)

    @property
    def pnl_usd(self) -> float:
        return self.equity_usd - self.initial_capital_usd

    @property
    def pnl_pct(self) -> float:
        return self.pnl_usd / self.initial_capital_usd * 100.0

    @property
    def win_rate_pct(self) -> float:
        return (
            self.winning_trades / self.closed_trades * 100.0
            if self.closed_trades else 0.0
        )

    def mark_to_market(self, bar: CommandMarketBar) -> None:
        self.current_prices = {
            str(sym).upper(): float(price)
            for sym, price in dict(bar.prices or {}).items()
            if float(price or 0.0) > 0
        }
        equity = self.cash_usd
        for pos in self.positions.values():
            price = self.current_prices.get(pos.symbol, pos.entry_price)
            if pos.side == "long":
                equity += pos.qty * price
            else:
                equity += pos.reserved_notional_usd + (pos.entry_price - price) * pos.qty
        self.equity_usd = float(equity)
        self.peak_equity_usd = max(self.peak_equity_usd, self.equity_usd)
        if self.peak_equity_usd > 0:
            self.max_drawdown_pct = max(
                self.max_drawdown_pct,
                (self.peak_equity_usd - self.equity_usd) / self.peak_equity_usd * 100.0,
            )
        day = _day_key(bar.timestamp)
        self.daily_start_equity.setdefault(day, self.equity_usd)
        self.daily_min_equity[day] = min(
            self.daily_min_equity.get(day, self.equity_usd),
            self.equity_usd,
        )
        self.last_timestamp = bar.timestamp.isoformat()
        self.equity_curve.append(
            {
                "bar": int(bar.bar),
                "timestamp": self.last_timestamp,
                "equity_usd": self.equity_usd,
                "cash_usd": self.cash_usd,
                "open_exposure_usd": self.open_exposure_usd,
                "open_positions": self.open_count,
                "pnl_usd": self.pnl_usd,
                "pnl_pct": self.pnl_pct,
                "max_drawdown_pct": self.max_drawdown_pct,
            }
        )

    def current_daily_loss_usd(self, bar: CommandMarketBar) -> float:
        day = _day_key(bar.timestamp)
        start = self.daily_start_equity.get(day, self.equity_usd)
        return min(0.0, self.equity_usd - start)

    def execute(self, command: TradingCommand, bar: CommandMarketBar) -> DryRunExecutionResult:
        if command.side == "hold":
            return DryRunExecutionResult(status="ignored", command=command, reason="hold")
        symbol = command.symbol.upper()
        price = float(command.price or bar.price(symbol))
        if price <= 0:
            return DryRunExecutionResult(status="rejected", command=command, reason="invalid_price")
        if command.side in {"buy", "sell"}:
            return self._open(command, bar, price)
        if command.side == "close":
            return self._close(command, bar, price)
        return DryRunExecutionResult(status="rejected", command=command, reason="unknown_side")

    def snapshot(self, bar: CommandMarketBar) -> dict[str, Any]:
        return {
            "bar": int(bar.bar),
            "timestamp": bar.timestamp.isoformat(),
            "cash_usd": self.cash_usd,
            "equity_usd": self.equity_usd,
            "pnl_usd": self.pnl_usd,
            "pnl_pct": self.pnl_pct,
            "open_exposure_usd": self.open_exposure_usd,
            "open_positions": {
                symbol: {
                    "side": pos.side,
                    "qty": pos.qty,
                    "entry_price": pos.entry_price,
                    "reserved_notional_usd": pos.reserved_notional_usd,
                    "opened_bar": pos.opened_bar,
                    "leader": pos.leader,
                    "source": pos.source,
                    "unrealized_pnl_usd": pos.unrealized_pnl(
                        self.current_prices.get(symbol, pos.entry_price)
                    ),
                }
                for symbol, pos in sorted(self.positions.items())
            },
        }

    def daily_summary(self) -> dict[str, Any]:
        rows: dict[str, Any] = {}
        for day, start in sorted(self.daily_start_equity.items()):
            min_equity = self.daily_min_equity.get(day, start)
            rows[day] = {
                "start_equity_usd": start,
                "min_equity_usd": min_equity,
                "realized_pnl_usd": self.daily_realized_pnl.get(day, 0.0),
                "max_intraday_loss_usd": min(0.0, min_equity - start),
            }
        return rows

    def _open(
        self,
        command: TradingCommand,
        bar: CommandMarketBar,
        price: float,
    ) -> DryRunExecutionResult:
        symbol = command.symbol.upper()
        if symbol in self.positions:
            return DryRunExecutionResult(status="rejected", command=command, reason="position_exists")
        notional = min(float(command.notional_usd), self.cash_usd)
        fill_price = price * (
            1.0 + self.config.slippage_pct
            if command.side == "buy"
            else 1.0 - self.config.slippage_pct
        )
        qty = notional / fill_price if fill_price > 0 else 0.0
        fee = notional * self.config.fee_rate
        if qty <= 0:
            return DryRunExecutionResult(status="rejected", command=command, reason="insufficient_cash")
        self.cash_usd -= notional + fee
        self.fees_paid_usd += fee
        self.positions[symbol] = SimulatedPosition(
            symbol=symbol,
            side="long" if command.side == "buy" else "short",
            qty=qty,
            entry_price=fill_price,
            reserved_notional_usd=notional,
            fee_open_usd=fee,
            opened_bar=int(bar.bar),
            opened_at=bar.timestamp.isoformat(),
            leader=command.leader,
            source=command.source,
        )
        self.mark_to_market(bar)
        return DryRunExecutionResult(status="filled", command=command, fee_usd=fee)

    def _close(
        self,
        command: TradingCommand,
        bar: CommandMarketBar,
        price: float,
    ) -> DryRunExecutionResult:
        symbol = command.symbol.upper()
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return DryRunExecutionResult(status="rejected", command=command, reason="no_position")
        fill_price = price * (
            1.0 - self.config.slippage_pct
            if pos.side == "long"
            else 1.0 + self.config.slippage_pct
        )
        close_notional = pos.qty * fill_price
        fee = close_notional * self.config.fee_rate
        if pos.side == "long":
            gross_pnl = close_notional - pos.reserved_notional_usd
            self.cash_usd += close_notional - fee
        else:
            gross_pnl = (pos.entry_price - fill_price) * pos.qty
            self.cash_usd += pos.reserved_notional_usd + gross_pnl - fee
        realized = gross_pnl - pos.fee_open_usd - fee
        self.realized_pnl_usd += realized
        self.fees_paid_usd += fee
        self.closed_trades += 1
        self.winning_trades += int(realized > 0)
        self.leader_realized_pnl[pos.leader] += realized
        self.leader_trades[pos.leader] += 1
        self.daily_realized_pnl[_day_key(bar.timestamp)] += realized
        self.mark_to_market(bar)
        return DryRunExecutionResult(
            status="filled",
            command=command,
            realized_pnl_usd=realized,
            fee_usd=fee,
        )


class RetroRiskGate:
    def __init__(self, config: CommandCenterConfig):
        self.config = config
        self._accepted_keys: set[tuple[int, str]] = set()

    def evaluate(
        self,
        command: TradingCommand,
        portfolio: PortfolioSimulator,
        bar: CommandMarketBar,
    ) -> RiskGateResult:
        checks: dict[str, Any] = {}
        side = command.side
        symbol = command.symbol.upper()

        self._check(checks, "dry_run", command.dry_run is True, True)
        self._check(checks, "mode", command.mode == "retro_dry_run", "retro_dry_run")
        self._check(checks, "exchange", command.exchange == "retro", "retro")
        self._check(checks, "price", float(command.price or 0.0) > 0.0, ">0")
        self._check(checks, "stale_data", not self._is_stale(command, bar), False)

        if side != "hold":
            key = (int(bar.bar), symbol)
            self._check(
                checks,
                "duplicate_order_on_same_bar",
                key not in self._accepted_keys,
                False,
            )

        daily_loss = portfolio.current_daily_loss_usd(bar)
        self._check(
            checks,
            "max_daily_loss_usd",
            side == "close"
            or self.config.max_daily_loss_usd <= 0
            or abs(daily_loss) < self.config.max_daily_loss_usd,
            self.config.max_daily_loss_usd,
            actual=abs(daily_loss),
        )
        self._check(
            checks,
            "max_drawdown_pct",
            side == "close"
            or self.config.max_drawdown_pct <= 0
            or portfolio.max_drawdown_pct < self.config.max_drawdown_pct,
            self.config.max_drawdown_pct,
            actual=portfolio.max_drawdown_pct,
        )

        if side in {"buy", "sell"}:
            notional = float(command.notional_usd or 0.0)
            liquidity = bar.volume(symbol) * bar.price(symbol)
            self._check(
                checks,
                "max_single_order_usd",
                0.0 < notional <= self.config.max_single_order_usd + 1e-12,
                self.config.max_single_order_usd,
                actual=notional,
            )
            self._check(
                checks,
                "max_total_budget_usd",
                portfolio.open_exposure_usd + notional <= self.config.max_total_budget_usd + 1e-12,
                self.config.max_total_budget_usd,
                actual=portfolio.open_exposure_usd + notional,
            )
            self._check(
                checks,
                "cash_available",
                True,
                notional,
                actual=portfolio.cash_usd,
            )
            self._check(
                checks,
                "max_open_positions",
                portfolio.open_count < self.config.max_open_positions,
                self.config.max_open_positions,
                actual=portfolio.open_count,
            )
            self._check(
                checks,
                "current_actionable",
                command.confidence >= self.config.min_actionable_confidence,
                self.config.min_actionable_confidence,
                actual=command.confidence,
            )
            if self.config.min_liquidity_usd > 0:
                self._check(
                    checks,
                    "min_liquidity_usd",
                    liquidity >= self.config.min_liquidity_usd,
                    self.config.min_liquidity_usd,
                    actual=liquidity,
                )
        elif side == "close":
            self._check(
                checks,
                "position_exists",
                symbol in portfolio.positions,
                True,
            )
        elif side == "hold":
            self._check(checks, "hold", True, True)
        else:
            self._check(checks, "known_side", False, "buy|sell|hold|close", actual=side)

        for name, payload in checks.items():
            if not bool(payload["allowed"]):
                return RiskGateResult(allowed=False, reason=name, checks=checks)
        return RiskGateResult(allowed=True, checks=checks)

    def record_accepted(self, command: TradingCommand, bar: CommandMarketBar) -> None:
        if command.side != "hold":
            self._accepted_keys.add((int(bar.bar), command.symbol.upper()))

    def forced_close_commands(
        self,
        portfolio: PortfolioSimulator,
        bar: CommandMarketBar,
    ) -> list[TradingCommand]:
        if not portfolio.positions:
            return []
        daily_loss = abs(portfolio.current_daily_loss_usd(bar))
        reason = ""
        if self.config.max_daily_loss_usd > 0 and daily_loss >= self.config.max_daily_loss_usd:
            reason = "max_daily_loss_usd"
        elif (
            self.config.max_drawdown_pct > 0
            and portfolio.max_drawdown_pct >= self.config.max_drawdown_pct
        ):
            reason = "max_drawdown_pct"
        if not reason:
            return []
        commands: list[TradingCommand] = []
        for symbol, pos in sorted(portfolio.positions.items()):
            price = bar.price(symbol) or pos.entry_price
            commands.append(
                TradingCommand(
                    timestamp=bar.timestamp.isoformat(),
                    mode="retro_dry_run",
                    exchange="retro",
                    symbol=symbol,
                    side="close",
                    order_type="market",
                    qty=pos.qty,
                    notional_usd=max(0.0, pos.qty * price),
                    price=price,
                    source="risk_exit",
                    leader="RiskGate",
                    confidence=1.0,
                    reason=f"forced close: {reason}",
                    risk_checks={},
                    dry_run=True,
                )
            )
        return commands

    @staticmethod
    def _check(
        checks: dict[str, Any],
        name: str,
        allowed: bool,
        limit: Any,
        *,
        actual: Any = None,
    ) -> None:
        checks[name] = {
            "allowed": bool(allowed),
            "limit": limit,
            "actual": actual,
        }

    def _is_stale(self, command: TradingCommand, bar: CommandMarketBar) -> bool:
        try:
            ts = datetime.fromisoformat(command.timestamp.replace("Z", "+00:00"))
        except ValueError:
            return True
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        bar_ts = bar.timestamp
        if bar_ts.tzinfo is None:
            bar_ts = bar_ts.replace(tzinfo=timezone.utc)
        return abs((bar_ts - ts).total_seconds()) > self.config.max_stale_seconds


class DryRunCommandExecutor:
    def __init__(self, config: CommandCenterConfig, live_executor: Any = None):
        self.config = config
        self.live_executor = live_executor
        self.live_executor_called = False

    def execute(
        self,
        command: TradingCommand,
        portfolio: PortfolioSimulator,
        bar: CommandMarketBar,
    ) -> DryRunExecutionResult:
        if not command.dry_run:
            return DryRunExecutionResult(
                status="rejected",
                command=command,
                reason="dry_run_required",
            )
        return portfolio.execute(command, bar)


class DecisionAdapter(Protocol):
    def decide(
        self,
        bar: CommandMarketBar,
        registry: "SignalRegistry",
        portfolio: PortfolioSimulator,
    ) -> TradingIntent:
        ...


class SignalRegistry:
    def __init__(self, max_history: int = 240):
        self._prices: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=max_history))
        self._last_context_by_bar: dict[int, Mapping[str, Any]] = {}

    def update_bar(self, bar: CommandMarketBar) -> None:
        for symbol, price in bar.prices.items():
            clean = str(symbol).upper()
            value = float(price or 0.0)
            if value > 0:
                self._prices[clean].append(value)

    def price_n_bars_ago(self, symbol: str, bars: int) -> Optional[float]:
        values = self._prices.get(str(symbol).upper())
        if not values or len(values) < bars:
            return None
        return list(values)[-bars]

    def last_price(self, symbol: str) -> Optional[float]:
        values = self._prices.get(str(symbol).upper())
        return values[-1] if values else None

    def record_panteon_context(self, bar: int, context: Mapping[str, Any]) -> None:
        self._last_context_by_bar[int(bar)] = dict(context)

    def panteon_context(self, bar: int) -> Mapping[str, Any]:
        return self._last_context_by_bar.get(int(bar), {})


class CausalMomentumDecisionAdapter:
    def __init__(self, config: CommandCenterConfig, panteon_context: Mapping[int, Mapping[str, Any]] | None = None):
        self.config = config
        self.panteon_context = {int(k): dict(v) for k, v in dict(panteon_context or {}).items()}

    def decide(
        self,
        bar: CommandMarketBar,
        registry: SignalRegistry,
        portfolio: PortfolioSimulator,
    ) -> TradingIntent:
        context = self.panteon_context.get(int(bar.bar), {})
        leader = _context_leader(context)
        source = "panteon" if leader and leader != "NoTrade" else "agent"

        close = self._exit_intent(bar, registry, portfolio, leader=leader or "RiskManaged")
        if close is not None:
            return close

        if portfolio.open_count >= self.config.max_open_positions:
            return TradingIntent.hold(leader or "NoTrade", "max open positions", source=source)

        symbols = self.config.symbols or tuple(sorted(str(sym).upper() for sym in bar.prices))
        best: Optional[tuple[float, str, float, float]] = None
        lookback = max(2, int(self.config.signal_lookback_bars))
        threshold = float(self.config.min_signal_return_pct) / 100.0
        current_threshold = float(self.config.min_current_bar_return_pct) / 100.0
        for symbol in symbols:
            price = bar.price(symbol)
            prior = registry.price_n_bars_ago(symbol, lookback)
            last = registry.last_price(symbol)
            if price <= 0 or prior is None or prior <= 0:
                continue
            ret = price / prior - 1.0
            one_bar_ret = price / last - 1.0 if last and last > 0 else 0.0
            if abs(ret) < threshold:
                continue
            if current_threshold > 0 and abs(one_bar_ret) < current_threshold:
                continue
            liquidity = bar.volume(symbol) * price
            if self.config.min_liquidity_usd > 0 and liquidity < self.config.min_liquidity_usd:
                continue
            score = abs(ret)
            if best is None or score > best[0]:
                best = (score, symbol, ret, one_bar_ret)
        if best is None:
            return TradingIntent.hold(leader or "NoTrade", "no current-actionable causal signal", source=source)
        score, symbol, ret, one_bar_ret = best
        confidence = min(1.0, max(self.config.min_actionable_confidence, score / max(threshold * 2.0, 1e-9)))
        side = "buy" if ret > 0 else "sell"
        return TradingIntent(
            symbol=symbol,
            side=side,
            source=source,
            leader=leader or f"CausalMomentum_{bar.regime}",
            confidence=confidence,
            reason=(
                f"causal momentum {ret * 100.0:.2f}% over {lookback} bars; "
                f"current {one_bar_ret * 100.0:.2f}%"
            ),
        )

    def _exit_intent(
        self,
        bar: CommandMarketBar,
        registry: SignalRegistry,
        portfolio: PortfolioSimulator,
        *,
        leader: str,
    ) -> Optional[TradingIntent]:
        for symbol, pos in sorted(portfolio.positions.items()):
            price = bar.price(symbol)
            if price <= 0:
                continue
            pnl_pct = pos.unrealized_pnl(price) / max(pos.reserved_notional_usd, 1e-9) * 100.0
            held = int(bar.bar) - int(pos.opened_bar)
            prior = registry.price_n_bars_ago(symbol, max(2, int(self.config.signal_lookback_bars // 2)))
            reverse = False
            if prior and prior > 0:
                ret = price / prior - 1.0
                reverse = (pos.side == "long" and ret < 0) or (pos.side == "short" and ret > 0)
            if pnl_pct >= self.config.take_profit_pct:
                reason = f"take profit {pnl_pct:.2f}%"
            elif pnl_pct <= -self.config.stop_loss_pct:
                reason = f"stop loss {pnl_pct:.2f}%"
            elif held >= self.config.max_holding_bars:
                reason = f"max holding bars {held}"
            elif reverse:
                reason = "causal reverse momentum"
            else:
                continue
            return TradingIntent(
                symbol=symbol,
                side="close",
                source="player",
                leader=leader or pos.leader,
                confidence=1.0,
                reason=reason,
            )
        return None


class TradingCommandCenter:
    def __init__(
        self,
        config: CommandCenterConfig,
        *,
        decision_adapter: Optional[DecisionAdapter] = None,
        risk_gate: Optional[RetroRiskGate] = None,
        executor: Optional[DryRunCommandExecutor] = None,
    ):
        self.config = config
        self.registry = SignalRegistry()
        self.portfolio = PortfolioSimulator(config)
        self.risk_gate = risk_gate or RetroRiskGate(config)
        self.executor = executor or DryRunCommandExecutor(config)
        self.decision_adapter = decision_adapter or CausalMomentumDecisionAdapter(config)
        self.commands: list[dict[str, Any]] = []
        self.decisions: list[dict[str, Any]] = []
        self.risk_rejections: list[dict[str, Any]] = []
        self.position_snapshots: list[dict[str, Any]] = []
        self.execution_results: list[DryRunExecutionResult] = []

    def run(self, bars: Iterable[CommandMarketBar]) -> dict[str, Any]:
        last_bar: Optional[CommandMarketBar] = None
        for bar in bars:
            last_bar = bar
            self.portfolio.mark_to_market(bar)
            for command in self.risk_gate.forced_close_commands(self.portfolio, bar):
                self._evaluate_execute_and_log(command, bar)
            intent = self.decision_adapter.decide(bar, self.registry, self.portfolio)
            self._record_decision(bar, intent)
            command = self._command_from_intent(intent, bar)
            if command is not None:
                self._evaluate_execute_and_log(command, bar)
            elif intent.side == "hold":
                self._log_hold_command(intent, bar)
            self.position_snapshots.append(self.portfolio.snapshot(bar))
            self.registry.update_bar(bar)

        if last_bar is not None:
            self._close_end_of_retro_positions(last_bar)
        report = self._report()
        self._export(report)
        return report

    def _command_from_intent(
        self,
        intent: TradingIntent,
        bar: CommandMarketBar,
    ) -> Optional[TradingCommand]:
        if intent.side == "hold":
            return None
        symbol = intent.symbol.upper()
        price = bar.price(symbol)
        if price <= 0:
            return None
        pos = self.portfolio.positions.get(symbol)
        if intent.side == "close" and pos is not None:
            qty = pos.qty
            notional = max(0.0, qty * price)
        else:
            notional = min(self.config.default_order_notional_usd, self.config.max_single_order_usd)
            qty = notional / price
        return TradingCommand(
            timestamp=bar.timestamp.isoformat(),
            mode="retro_dry_run",
            exchange="retro",
            symbol=symbol,
            side=intent.side,
            order_type=intent.order_type,
            qty=qty,
            notional_usd=notional,
            price=price,
            source=intent.source,
            leader=intent.leader,
            confidence=max(0.0, min(1.0, float(intent.confidence))),
            reason=intent.reason,
            risk_checks={},
            dry_run=True,
        )

    def _evaluate_execute_and_log(self, command: TradingCommand, bar: CommandMarketBar) -> None:
        risk = self.risk_gate.evaluate(command, self.portfolio, bar)
        command_payload = command.to_dict()
        command_payload["risk_checks"] = dict(risk.checks)
        if not risk.allowed:
            self.risk_rejections.append(
                {
                    "bar": int(bar.bar),
                    "timestamp": bar.timestamp.isoformat(),
                    "reason": risk.reason,
                    "command": command_payload,
                    "risk_checks": dict(risk.checks),
                }
            )
            return
        accepted = TradingCommand(**command_payload)
        self.risk_gate.record_accepted(accepted, bar)
        result = self.executor.execute(accepted, self.portfolio, bar)
        self.execution_results.append(result)
        payload = accepted.to_dict()
        payload.update(
            {
                "bar": int(bar.bar),
                "status": result.status,
                "realized_pnl_usd": result.realized_pnl_usd,
                "fee_usd": result.fee_usd,
                "execution_reason": result.reason,
            }
        )
        self.commands.append(payload)

    def _record_decision(self, bar: CommandMarketBar, intent: TradingIntent) -> None:
        self.decisions.append(
            {
                "bar": int(bar.bar),
                "timestamp": bar.timestamp.isoformat(),
                "mode": "retro_dry_run",
                "symbol": intent.symbol,
                "side": intent.side,
                "source": intent.source,
                "leader": intent.leader,
                "confidence": intent.confidence,
                "reason": intent.reason,
                "causal": True,
                "uses_future_data": False,
                "uses_perfect_oracle": False,
            }
        )

    def _log_hold_command(self, intent: TradingIntent, bar: CommandMarketBar) -> None:
        symbol = str(intent.symbol or "PORTFOLIO").upper()
        payload = TradingCommand(
            timestamp=bar.timestamp.isoformat(),
            mode="retro_dry_run",
            exchange="retro",
            symbol=symbol,
            side="hold",
            order_type="market",
            qty=0.0,
            notional_usd=0.0,
            price=0.0,
            source=intent.source,
            leader=intent.leader,
            confidence=max(0.0, min(1.0, float(intent.confidence))),
            reason=intent.reason,
            risk_checks={"hold": {"allowed": True, "limit": True, "actual": None}},
            dry_run=True,
        ).to_dict()
        payload.update(
            {
                "bar": int(bar.bar),
                "status": "ignored",
                "realized_pnl_usd": 0.0,
                "fee_usd": 0.0,
                "execution_reason": "hold",
            }
        )
        self.commands.append(payload)

    def _close_end_of_retro_positions(self, bar: CommandMarketBar) -> None:
        for symbol, pos in list(sorted(self.portfolio.positions.items())):
            price = bar.price(symbol) or pos.entry_price
            command = TradingCommand(
                timestamp=bar.timestamp.isoformat(),
                mode="retro_dry_run",
                exchange="retro",
                symbol=symbol,
                side="close",
                order_type="market",
                qty=pos.qty,
                notional_usd=max(0.0, pos.qty * price),
                price=price,
                source="risk_exit",
                leader="EndOfRetro",
                confidence=1.0,
                reason="terminal emergency flat",
                risk_checks={},
                dry_run=True,
            )
            self._evaluate_execute_and_log(command, bar)
        self.portfolio.mark_to_market(bar)
        self.position_snapshots.append(self.portfolio.snapshot(bar))

    def _report(self) -> dict[str, Any]:
        executed = [row for row in self.commands if row.get("status") == "filled"]
        trade_commands = [row for row in executed if row.get("side") in {"buy", "sell", "close"}]
        closed = [row for row in executed if row.get("side") == "close"]
        holds = [row for row in self.decisions if row.get("side") == "hold"]
        bars = len(self.decisions)
        average_notional = (
            sum(float(row.get("notional_usd", 0.0) or 0.0) for row in trade_commands)
            / len(trade_commands)
            if trade_commands else 0.0
        )
        daily = self.portfolio.daily_summary()
        daily_loss_violations = sum(
            1
            for row in daily.values()
            if abs(float(row.get("max_intraday_loss_usd", 0.0) or 0.0)) > self.config.max_daily_loss_usd + 1e-9
        )
        profitable_days = sum(
            1 for row in daily.values() if float(row.get("realized_pnl_usd", 0.0) or 0.0) > 0
        )
        monthly_realized: dict[str, float] = defaultdict(float)
        for day, row in daily.items():
            monthly_realized[str(day)[:7]] += float(row.get("realized_pnl_usd", 0.0) or 0.0)
        profitable_months = sum(1 for pnl in monthly_realized.values() if pnl > 0)
        leader_contribution = {
            label: {
                "realized_pnl_usd": pnl,
                "closed_trades": self.portfolio.leader_trades.get(label, 0),
            }
            for label, pnl in sorted(
                self.portfolio.leader_realized_pnl.items(),
                key=lambda item: (-item[1], item[0]),
            )
        }
        return {
            "run_name": self.config.run_name,
            "mode": "retro_dry_run",
            "exchange": "retro",
            "initial_capital_usd": self.config.initial_capital_usd,
            "final_equity_usd": self.portfolio.equity_usd,
            "pnl_usd": self.portfolio.pnl_usd,
            "pnl_pct": self.portfolio.pnl_pct,
            "max_drawdown_pct": self.portfolio.max_drawdown_pct,
            "daily_loss_violations": daily_loss_violations,
            "risk_violations": 0,
            "trade_count": len(trade_commands),
            "closed_trades": self.portfolio.closed_trades,
            "win_rate_pct": self.portfolio.win_rate_pct,
            "no_trade_pct": (len(holds) / bars * 100.0) if bars else 0.0,
            "risk_reject_pct": (len(self.risk_rejections) / max(1, bars) * 100.0),
            "risk_rejections": len(self.risk_rejections),
            "average_order_notional_usd": average_notional,
            "fees_usd": self.portfolio.fees_paid_usd,
            "fee_rate": self.config.fee_rate,
            "slippage_pct": self.config.slippage_pct,
            "profitable_days": profitable_days,
            "days": len(daily),
            "profitable_months": profitable_months,
            "months": len(monthly_realized),
            "leader_contribution": leader_contribution,
            "dry_run": True,
            "live_executor_called": bool(self.executor.live_executor_called),
            "causal": True,
            "uses_future_data": False,
            "uses_perfect_oracle_for_decisions": False,
        }

    def _export(self, report: Mapping[str, Any]) -> None:
        output_dir = Path(self.config.output_dir or Path("Results") / "TradingCommandCenter")
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_dir / "retro_trade_commands.jsonl", self.commands)
        _write_jsonl(output_dir / "retro_decisions.jsonl", self.decisions)
        _write_jsonl(output_dir / "retro_risk_rejections.jsonl", self.risk_rejections)
        _write_jsonl(output_dir / "retro_position_snapshots.jsonl", self.position_snapshots)
        _write_equity_csv(output_dir / "retro_equity_curve.csv", self.portfolio.equity_curve)
        (output_dir / "retro_equity_curve.json").write_text(
            json.dumps(self.portfolio.equity_curve, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / "retro_daily_summary.json").write_text(
            json.dumps(self.portfolio.daily_summary(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / "command_center_report.json").write_text(
            json.dumps(dict(report), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        (output_dir / "command_center_report.md").write_text(
            _markdown_report(report),
            encoding="utf-8",
        )
        (output_dir / "retro_dashboard.html").write_text(
            _dashboard_html(report, self.portfolio.equity_curve),
            encoding="utf-8",
        )
        _write_dashboard_png(
            output_dir / "retro_dashboard.png",
            report,
            self.portfolio.equity_curve,
        )


def load_panteon_context_from_trading_log(path: str | Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    field_re = re.compile(r"\b([A-Za-z_]+)=([^\s]+)")
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if " bar=" not in line:
            continue
        fields = dict(field_re.findall(line))
        try:
            bar = int(fields.get("bar", "0"))
        except ValueError:
            continue
        if bar <= 0:
            continue
        rows[bar] = {
            "leader": fields.get("executed_leader") or fields.get("leader") or "",
            "selected_leader": fields.get("selected_leader", ""),
            "regime": fields.get("regime", ""),
            "raw_signals": _int(fields.get("raw_signals")),
            "signals": _int(fields.get("signals")),
            "filled": _int(fields.get("filled")),
            "fallback_used": fields.get("fallback_used", "False") == "True",
        }
    return rows


def load_retrodate_command_bars(
    data_dir: str | Path,
    *,
    years: Sequence[int] = (2022, 2023, 2024, 2025, 2026),
    stride_minutes: int = 60,
) -> list[CommandMarketBar]:
    from .retrodate_market_runner import (
        RetrodateMarketConfig,
        RetrodateSnapshotState,
        load_retrodate_year_snapshots,
        select_retrodate_files,
    )

    config = RetrodateMarketConfig(
        data_dir=Path(data_dir),
        years=tuple(int(year) for year in years),
        stride_minutes=int(stride_minutes),
        initial_capital=20.0,
    )
    selection = select_retrodate_files(config)
    state = RetrodateSnapshotState()
    bars: list[CommandMarketBar] = []
    for report in selection.valid_reports:
        for snapshot in load_retrodate_year_snapshots(
            report.path,
            stride_minutes=int(stride_minutes),
            state=state,
        ):
            bars.append(
                CommandMarketBar(
                    bar=snapshot.bar,
                    timestamp=snapshot.timestamp,
                    prices={str(k).upper(): float(v) for k, v in snapshot.prices.items()},
                    volumes={str(k).upper(): float(v) for k, v in snapshot.volumes.items()},
                    regime=snapshot.regime.label,
                )
            )
    return bars


def run_trading_command_center_retro(
    *,
    data_dir: str | Path = "Retrodate",
    output_dir: str | Path,
    baseline_dir: str | Path | None = None,
    years: Sequence[int] = (2022, 2023, 2024, 2025, 2026),
    stride_minutes: int = 60,
    config_overrides: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    overrides = dict(config_overrides or {})
    overrides.setdefault("initial_capital_usd", 20.0)
    overrides.setdefault("max_total_budget_usd", 20.0)
    overrides.setdefault("max_single_order_usd", 2.0)
    overrides.setdefault("output_dir", Path(output_dir))
    cfg = CommandCenterConfig(**overrides)
    context = {}
    baseline_path = Path(baseline_dir) if baseline_dir else None
    if baseline_path and (baseline_path / "trading.log").exists():
        context = load_panteon_context_from_trading_log(baseline_path / "trading.log")
    bars = load_retrodate_command_bars(
        data_dir,
        years=years,
        stride_minutes=stride_minutes,
    )
    center = TradingCommandCenter(
        cfg,
        decision_adapter=CausalMomentumDecisionAdapter(cfg, panteon_context=context),
    )
    report = center.run(bars)
    comparison = _baseline_comparison(report, baseline_path)
    report = dict(report)
    report["period"] = {
        "years": [int(year) for year in years],
        "stride_minutes": int(stride_minutes),
        "bars": len(bars),
        "first_timestamp": bars[0].timestamp.isoformat() if bars else "",
        "last_timestamp": bars[-1].timestamp.isoformat() if bars else "",
    }
    if comparison:
        report["comparison"] = comparison
        report["transition_to_live_canary"] = {
            "allowed": False,
            "reason": (
                "offline dry-run only; live trading remains prohibited until "
                "alpha beats baseline and closes the gap to soft allocator with "
                "fresh independent validation"
            ),
        }
        center._export(report)
    return report


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _write_equity_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "bar",
        "timestamp",
        "equity_usd",
        "cash_usd",
        "open_exposure_usd",
        "open_positions",
        "pnl_usd",
        "pnl_pct",
        "max_drawdown_pct",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# TradingCommandCenter Retro Dry-Run Report",
        "",
        "## Result",
        f"- Initial capital: ${float(report.get('initial_capital_usd', 0.0)):.2f}",
        f"- Final equity: ${float(report.get('final_equity_usd', 0.0)):.4f}",
        f"- PnL: ${float(report.get('pnl_usd', 0.0)):.4f} ({float(report.get('pnl_pct', 0.0)):.4f}%)",
        f"- Max drawdown: {float(report.get('max_drawdown_pct', 0.0)):.4f}%",
        f"- Trades: {int(report.get('trade_count', 0))}",
        f"- Closed trades: {int(report.get('closed_trades', 0))}",
        f"- Win rate: {float(report.get('win_rate_pct', 0.0)):.2f}%",
        f"- NoTrade: {float(report.get('no_trade_pct', 0.0)):.2f}%",
        f"- Risk reject: {float(report.get('risk_reject_pct', 0.0)):.2f}%",
        f"- Profitable days/months: {int(report.get('profitable_days', 0))}/{int(report.get('profitable_months', 0))}",
        "",
        "## Safety",
        f"- Dry run: {bool(report.get('dry_run'))}",
        f"- Live executor called: {bool(report.get('live_executor_called'))}",
        f"- Risk violations: {int(report.get('risk_violations', 0))}",
        f"- Daily loss violations: {int(report.get('daily_loss_violations', 0))}",
        f"- Causal decisions: {bool(report.get('causal'))}",
        f"- Perfect oracle used for decisions: {bool(report.get('uses_perfect_oracle_for_decisions'))}",
        "",
        "## Assumptions",
        f"- Fee rate: {float(report.get('fee_rate', 0.0)):.6f}",
        f"- Slippage pct: {float(report.get('slippage_pct', 0.0)):.6f}",
        f"- Average order notional: ${float(report.get('average_order_notional_usd', 0.0)):.4f}",
    ]
    comparison = report.get("comparison")
    if isinstance(comparison, Mapping):
        lines.extend(["", "## Baseline Comparison"])
        rows = comparison.get("rows", [])
        if isinstance(rows, Sequence):
            lines.extend([
                "| Name | PnL USD @ $20 | PnL % | Max DD % |",
                "| --- | ---: | ---: | ---: |",
            ])
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                lines.append(
                    "| "
                    f"{row.get('name', '')} | "
                    f"${float(row.get('pnl_usd_scaled', 0.0) or 0.0):.4f} | "
                    f"{float(row.get('pnl_pct', 0.0) or 0.0):.4f}% | "
                    f"{float(row.get('max_drawdown_pct', 0.0) or 0.0):.4f}% |"
                )
        lines.extend([
            "",
            "## Assessment",
            f"- Not worse than current Panteon baseline: {bool(comparison.get('not_worse_than_baseline'))}",
            f"- Regret vs best single: ${float(comparison.get('regret_vs_best_single_usd', 0.0) or 0.0):.4f}",
            f"- Regret vs soft allocator: ${float(comparison.get('regret_vs_soft_allocator_usd', 0.0) or 0.0):.4f}",
            f"- Regret vs Perfect Panteon: ${float(comparison.get('regret_vs_perfect_panteon_usd', 0.0) or 0.0):.4f}",
        ])
    live = report.get("transition_to_live_canary")
    if isinstance(live, Mapping):
        lines.extend([
            "",
            "## Live Canary",
            f"- Allowed: {bool(live.get('allowed'))}",
            f"- Reason: {live.get('reason', '')}",
        ])
    return "\n".join(lines) + "\n"


def _dashboard_html(report: Mapping[str, Any], equity_curve: Sequence[Mapping[str, Any]]) -> str:
    points = list(equity_curve)[-240:]
    if points:
        values = [float(row.get("equity_usd", 0.0) or 0.0) for row in points]
        low = min(values)
        high = max(values)
        span = max(high - low, 1e-9)
        coords = []
        width = 900
        height = 220
        for index, value in enumerate(values):
            x = index / max(1, len(values) - 1) * width
            y = height - (value - low) / span * height
            coords.append(f"{x:.2f},{y:.2f}")
        polyline = " ".join(coords)
    else:
        polyline = ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>TradingCommandCenter Retro Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #17202a; background: #f7f9fb; }}
    main {{ max-width: 1040px; margin: 0 auto; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .metric {{ background: white; border: 1px solid #d7dee8; border-radius: 6px; padding: 12px; }}
    .label {{ color: #607085; font-size: 12px; }}
    .value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
    svg {{ width: 100%; height: 260px; background: white; border: 1px solid #d7dee8; border-radius: 6px; }}
  </style>
</head>
<body>
<main>
  <h1>TradingCommandCenter Retro Dry-Run</h1>
  <section class="grid">
    <div class="metric"><div class="label">PnL</div><div class="value">${float(report.get('pnl_usd', 0.0)):.4f}</div></div>
    <div class="metric"><div class="label">PnL %</div><div class="value">{float(report.get('pnl_pct', 0.0)):.4f}%</div></div>
    <div class="metric"><div class="label">Max DD</div><div class="value">{float(report.get('max_drawdown_pct', 0.0)):.4f}%</div></div>
    <div class="metric"><div class="label">Trades</div><div class="value">{int(report.get('trade_count', 0))}</div></div>
  </section>
  <h2>Equity</h2>
  <svg viewBox="0 0 900 220" role="img" aria-label="equity curve">
    <polyline points="{polyline}" fill="none" stroke="#1f7a8c" stroke-width="3" />
  </svg>
  <h2>Safety</h2>
  <p>dry_run={bool(report.get('dry_run'))} live_executor_called={bool(report.get('live_executor_called'))}
  risk_violations={int(report.get('risk_violations', 0))} daily_loss_violations={int(report.get('daily_loss_violations', 0))}</p>
</main>
</body>
</html>
"""


def _write_dashboard_png(
    path: Path,
    report: Mapping[str, Any],
    equity_curve: Sequence[Mapping[str, Any]],
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        path.write_bytes(_PNG_1X1)
        return

    width, height = 1100, 620
    margin = 48
    chart_left, chart_top = margin, 210
    chart_right, chart_bottom = width - margin, height - 70
    image = Image.new("RGB", (width, height), "#f7f9fb")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("arial.ttf", 28)
        metric_font = ImageFont.truetype("arial.ttf", 20)
        small_font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        title_font = metric_font = small_font = ImageFont.load_default()

    draw.text((margin, 28), "TradingCommandCenter Retro Dry-Run", fill="#17202a", font=title_font)
    metrics = [
        ("PnL", f"${float(report.get('pnl_usd', 0.0) or 0.0):.4f}"),
        ("PnL %", f"{float(report.get('pnl_pct', 0.0) or 0.0):.4f}%"),
        ("Max DD", f"{float(report.get('max_drawdown_pct', 0.0) or 0.0):.4f}%"),
        ("Trades", str(int(report.get('trade_count', 0) or 0))),
    ]
    card_w = (width - 2 * margin - 36) // 4
    for index, (label, value) in enumerate(metrics):
        x0 = margin + index * (card_w + 12)
        y0 = 86
        draw.rounded_rectangle((x0, y0, x0 + card_w, y0 + 88), radius=6, fill="#ffffff", outline="#d7dee8")
        draw.text((x0 + 16, y0 + 14), label, fill="#607085", font=small_font)
        draw.text((x0 + 16, y0 + 42), value, fill="#17202a", font=metric_font)

    draw.rounded_rectangle(
        (chart_left, chart_top, chart_right, chart_bottom),
        radius=6,
        fill="#ffffff",
        outline="#d7dee8",
    )
    points = list(equity_curve)[-800:]
    values = [float(row.get("equity_usd", 0.0) or 0.0) for row in points]
    if values:
        low, high = min(values), max(values)
        if abs(high - low) < 1e-9:
            low -= 0.5
            high += 0.5
        span = high - low
        coords: list[tuple[float, float]] = []
        inner_left, inner_top = chart_left + 28, chart_top + 24
        inner_right, inner_bottom = chart_right - 24, chart_bottom - 34
        for index, value in enumerate(values):
            x = inner_left + index / max(1, len(values) - 1) * (inner_right - inner_left)
            y = inner_bottom - (value - low) / span * (inner_bottom - inner_top)
            coords.append((x, y))
        draw.line((inner_left, inner_bottom, inner_right, inner_bottom), fill="#c9d3df", width=1)
        draw.line((inner_left, inner_top, inner_left, inner_bottom), fill="#c9d3df", width=1)
        if len(coords) >= 2:
            draw.line(coords, fill="#1f7a8c", width=3)
        draw.text((inner_left, chart_bottom - 26), f"low ${low:.2f}", fill="#607085", font=small_font)
        draw.text((inner_right - 90, chart_bottom - 26), f"high ${high:.2f}", fill="#607085", font=small_font)
    else:
        draw.text((chart_left + 28, chart_top + 28), "No equity data", fill="#607085", font=small_font)

    safety = (
        f"dry_run={bool(report.get('dry_run'))}  "
        f"live_executor_called={bool(report.get('live_executor_called'))}  "
        f"risk_violations={int(report.get('risk_violations', 0) or 0)}  "
        f"daily_loss_violations={int(report.get('daily_loss_violations', 0) or 0)}"
    )
    draw.text((margin, height - 42), safety, fill="#17202a", font=small_font)
    image.save(path)


def _context_leader(context: Mapping[str, Any]) -> str:
    leader = str(context.get("leader") or context.get("executed_leader") or "").strip()
    return leader or "NoTrade"


def _day_key(value: datetime | date | str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)[:10]


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _baseline_comparison(
    report: Mapping[str, Any],
    baseline_dir: Optional[Path],
) -> dict[str, Any]:
    if baseline_dir is None or not baseline_dir.exists():
        return {}
    status = _load_json(baseline_dir / "status.json")
    soft = _load_json(baseline_dir / "soft_allocator_report.json")
    perfect = _load_json(baseline_dir / "perfect_panteon_report.json")
    initial = float(report.get("initial_capital_usd", 20.0) or 20.0)
    panteon_session = status.get("live_session", {}) if isinstance(status, dict) else {}
    baseline_pct = float(panteon_session.get("panteon_owned_pnl_pct", 0.0) or 0.0)
    baseline_dd = float(status.get("panteon_max_drawdown_pct", 0.0) or 0.0) if isinstance(status, dict) else 0.0
    soft_pct = float(soft.get("best_policy_pnl_pct", 0.0) or 0.0) if isinstance(soft, dict) else 0.0
    soft_dd = float(soft.get("best_policy_max_drawdown_pct", 0.0) or 0.0) if isinstance(soft, dict) else 0.0
    single_pct = float(soft.get("best_single_pnl_pct", 0.0) or 0.0) if isinstance(soft, dict) else 0.0
    single_dd = float(soft.get("best_single_max_drawdown_pct", 0.0) or 0.0) if isinstance(soft, dict) else 0.0
    perfect_pct = float(perfect.get("pnl_pct", 0.0) or 0.0) if isinstance(perfect, dict) else 0.0
    perfect_dd = float(perfect.get("max_drawdown_pct", 0.0) or 0.0) if isinstance(perfect, dict) else 0.0
    command_pct = float(report.get("pnl_pct", 0.0) or 0.0)
    command_usd = float(report.get("pnl_usd", 0.0) or 0.0)

    def scaled(pct: float) -> float:
        return initial * pct / 100.0

    rows = [
        {
            "name": "TradingCommandCenter",
            "pnl_usd_scaled": command_usd,
            "pnl_pct": command_pct,
            "max_drawdown_pct": float(report.get("max_drawdown_pct", 0.0) or 0.0),
        },
        {
            "name": "Current Panteon baseline",
            "pnl_usd_scaled": scaled(baseline_pct),
            "pnl_pct": baseline_pct,
            "max_drawdown_pct": baseline_dd,
        },
        {
            "name": str(soft.get("best_policy_name", "Soft allocator") if isinstance(soft, dict) else "Soft allocator"),
            "pnl_usd_scaled": scaled(soft_pct),
            "pnl_pct": soft_pct,
            "max_drawdown_pct": soft_dd,
        },
        {
            "name": str(soft.get("best_single_label", "Best single") if isinstance(soft, dict) else "Best single"),
            "pnl_usd_scaled": scaled(single_pct),
            "pnl_pct": single_pct,
            "max_drawdown_pct": single_dd,
        },
        {
            "name": "Perfect Panteon",
            "pnl_usd_scaled": scaled(perfect_pct),
            "pnl_pct": perfect_pct,
            "max_drawdown_pct": perfect_dd,
        },
    ]
    return {
        "baseline_dir": str(baseline_dir),
        "rows": rows,
        "not_worse_than_baseline": command_pct >= baseline_pct,
        "delta_vs_baseline_usd": command_usd - scaled(baseline_pct),
        "delta_vs_baseline_pct": command_pct - baseline_pct,
        "regret_vs_best_single_usd": max(0.0, scaled(single_pct) - command_usd),
        "regret_vs_soft_allocator_usd": max(0.0, scaled(soft_pct) - command_usd),
        "regret_vs_perfect_panteon_usd": max(0.0, scaled(perfect_pct) - command_usd),
        "poor_profit_causes": [
            "risk budget is capped at $20 total and $2 per order",
            "strategy uses only current/past bars and refuses non-actionable signals",
            "panteon trading.log does not expose original symbol/side, so commands use deterministic causal market signals",
            "perfect oracle is excluded from decisions and used only for regret reporting",
        ],
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}
