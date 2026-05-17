"""RiskLimits — pre-flight проверки безопасности перед отправкой ордера.

Все проверки в одном месте, явные параметры. Заменяет в v1 разрозненные
проверки `min_notional`, `MAX_POS`, `if balance < threshold: return`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

from ..domain.types import Action, Signal


@dataclass(frozen=True)
class RiskLimitsConfig:
    """Параметры risk-проверок."""

    max_open_positions:    int   = 8     # абсолютный лимит open позиций
    max_per_symbol:        int   = 1     # одна позиция на символ
    min_notional_usd:      float = 5.0   # минимальный ордер в USDT
    max_notional_usd:      float = 1000.0  # верхняя граница чтобы не разогнаться
    max_leverage:          int   = 5
    floor_to_exchange_min_notional: bool = False
    max_min_notional_upscale: float = 4.0
    external_position_player_labels: Tuple[str, ...] = ("", "RecoveredExchangePosition")
    capital_fraction:      float = 0.10  # доля капитала на сделку (10%)

    def __post_init__(self) -> None:
        if self.max_open_positions < 0:
            raise ValueError("max_open_positions must be ≥ 0")
        if not 0 < self.capital_fraction <= 1.0:
            raise ValueError(f"capital_fraction must be in (0, 1], got {self.capital_fraction}")
        if self.min_notional_usd < 0:
            raise ValueError("min_notional_usd must be >= 0")
        if self.max_min_notional_upscale < 1.0:
            raise ValueError("max_min_notional_upscale must be >= 1")


DEFAULT_RISK = RiskLimitsConfig()


@dataclass(frozen=True)
class RiskCheckResult:
    """Результат pre-flight проверки."""

    allowed:   bool
    reason:    str = ""              # пусто если allowed
    qty:       float = 0.0           # рекомендуемое qty (0 если deny)
    notional:  float = 0.0           # ожидаемый notional


class RiskLimits:
    """Pre-flight проверки.

    Использование:
        rl = RiskLimits(config=RiskLimitsConfig(...))
        result = rl.evaluate(signal, balance_usd, current_positions, exchange)
        if result.allowed:
            send_order(signal, qty=result.qty)
        else:
            log.info("blocked: %s", result.reason)
    """

    def __init__(self, *, config: RiskLimitsConfig = DEFAULT_RISK):
        self._config = config

    def evaluate(
        self,
        signal:           Signal,
        *,
        balance_usd:      float,
        open_positions:   Dict[str, "object"],   # sym → any (только len + check by sym)
        min_notional_for_sym: float = 0.0,
    ) -> RiskCheckResult:
        """Проверить можно ли исполнять сигнал.

        Возвращает RiskCheckResult с qty (если allowed) или reason (если нет).
        """
        cfg = self._config

        # 1. Hold action — пропускаем без действий
        if signal.action.is_hold:
            return RiskCheckResult(
                allowed=False,
                reason="hold action",
            )

        # 2. Close-actions: не нужно проверять risk-лимиты, кроме наличия позиции
        if signal.action.is_close:
            position = open_positions.get(signal.sym)
            if position is None:
                return RiskCheckResult(
                    allowed=False,
                    reason="no position to close",
                )
            if self._is_external_position(position, set(cfg.external_position_player_labels)):
                return RiskCheckResult(
                    allowed=False,
                    reason=f"external recovered position on {signal.sym} is not panteon-owned",
                )
            return RiskCheckResult(
                allowed=True,
                qty=0.0,    # qty заполнит executor из позиции
                notional=0.0,
            )

        # 3. Open-actions
        # Лимит на открытых позициях
        if self._owned_open_count(open_positions) >= cfg.max_open_positions:
            return RiskCheckResult(
                allowed=False,
                reason=f"max_open_positions={cfg.max_open_positions} reached",
            )

        # Лимит на per-symbol
        if signal.sym in open_positions:
            return RiskCheckResult(
                allowed=False,
                reason=f"position already open on {signal.sym}",
            )

        # Notional-расчёт с учётом fraction × signal.action.fraction (half/full)
        size_mult = signal.action.fraction or 1.0  # 0.5 для half, 1.0 для full
        risk_mult = max(0.0, signal.risk_mult)
        notional = balance_usd * cfg.capital_fraction * size_mult * risk_mult

        # Min/max notional
        exchange_min = float(min_notional_for_sym or 0.0)
        effective_min = max(cfg.min_notional_usd, exchange_min)
        if notional < effective_min:
            can_floor = (
                cfg.floor_to_exchange_min_notional
                and exchange_min > 0
                and effective_min <= cfg.max_notional_usd
                and notional > 0
            )
            upscale = (effective_min / notional) if notional > 0 else float("inf")
            if can_floor and upscale <= cfg.max_min_notional_upscale:
                notional = effective_min
            else:
                detail = ""
                if can_floor:
                    detail = (
                        f" (min floor would require {upscale:.2f}x "
                        f"> {cfg.max_min_notional_upscale:.2f}x)"
                    )
                return RiskCheckResult(
                    allowed=False,
                    reason=f"notional ${notional:.2f} < min ${effective_min:.2f}{detail}",
                )
        if notional > cfg.max_notional_usd:
            # capping вместо отказа
            notional = cfg.max_notional_usd

        if signal.price <= 0:
            return RiskCheckResult(
                allowed=False,
                reason=f"invalid signal.price={signal.price}",
            )
        qty = notional / signal.price

        return RiskCheckResult(
            allowed=True,
            qty=qty,
            notional=notional,
        )

    @property
    def config(self) -> RiskLimitsConfig:
        return self._config

    def _owned_open_count(self, open_positions: Dict[str, "object"]) -> int:
        external_labels = set(self._config.external_position_player_labels)
        return sum(
            1
            for position in open_positions.values()
            if self._counts_towards_open_limit(position, external_labels)
        )

    @staticmethod
    def _counts_towards_open_limit(
        position: "object",
        external_labels: Set[str],
    ) -> bool:
        return not RiskLimits._is_external_position(position, external_labels)

    @staticmethod
    def _is_external_position(
        position: "object",
        external_labels: Set[str],
    ) -> bool:
        owner = getattr(position, "by_player", None)
        if owner is None:
            return False
        return str(owner or "") in external_labels
