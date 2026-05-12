"""Шаблон адаптера биржевого коннектора v1 под v2 Exchange Protocol.

Это **единственное место** в panteon_v2/, где допустим импорт из
panteon_runtime/. Это сознательная boundary — анти-corruption layer
(Эванс), отделяющий чистую архитектуру v2 от legacy connector-ов.

Реализуйте по этому образцу:
  • src/panteon_v2/app/bitget_adapter.py
  • src/panteon_v2/app/mexc_adapter.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..domain.types import Action, Signal, Trade
from ..execution.exchange import (
    Exchange,
    ExchangePosition,
    OrderResult,
    OrderStatus,
)


# ────────────────────────────────────────────────────────────────────
# TEMPLATE — copy & adapt for real connector
# ────────────────────────────────────────────────────────────────────


class V1ExchangeAdapter:
    """Шаблон-адаптер. Замените _client на реальный v1-коннектор.

    Контракт v2 Exchange Protocol:
      send_order(signal, *, qty) → OrderResult
      get_position(sym) → Optional[ExchangePosition]
      get_all_positions() → Dict[str, ExchangePosition]
      get_min_notional(sym) → float
    """

    name: str

    def __init__(
        self,
        v1_client: Any,           # экземпляр MexcDirectClient/BitgetDirectClient
        *,
        name: str = "EXCHANGE",
        default_min_notional: float = 5.0,
    ):
        self._client = v1_client
        self.name = name
        self._default_min_notional = float(default_min_notional)

    # ── Required by Exchange Protocol ───────────────────────────────

    def send_order(self, signal: Signal, *, qty: float) -> OrderResult:
        """Отправка ордера через v1-коннектор.

        Структура ответа v1 connector-а — словарь со столбцами
        success/code/data/msg. Здесь мы конвертируем в OrderResult.
        """
        try:
            # Пример: реальный v1-коннектор имеет разные методы для
            # buy/sell/short/close. Маппим из Action.
            method = self._select_v1_method(signal.action)
            raw = method(
                symbol=signal.sym,
                quantity=qty,
                price=signal.price,        # может игнорироваться для market-ордеров
                # ... другие kwargs ...
            )
        except Exception as exc:
            return OrderResult(
                status=OrderStatus.REJECTED,
                signal_id=signal.id,
                sym=signal.sym,
                message=f"adapter exception: {type(exc).__name__}: {exc}",
            )

        # Парсим ответ v1
        if not isinstance(raw, dict):
            return OrderResult(
                status=OrderStatus.REJECTED,
                signal_id=signal.id,
                sym=signal.sym,
                message=f"unexpected response type: {type(raw).__name__}",
            )

        success = bool(raw.get("success"))
        if not success:
            return OrderResult(
                status=OrderStatus.REJECTED,
                signal_id=signal.id,
                sym=signal.sym,
                message=str(raw.get("msg") or raw.get("error") or "unknown"),
            )

        order_id = str(raw.get("order_id") or raw.get("data") or "")
        # Цена fill — биржа возвращает её в `fillPrice` или подобном поле.
        # Если нет — используем signal.price (приблизительно).
        fill_price = float(raw.get("fillPrice") or signal.price)
        fee = float(raw.get("fee") or 0.0)

        # Side определяем из action или из позиции для close
        side = signal.action.side
        if not side and signal.action.is_close:
            existing = self.get_position(signal.sym)
            side = existing.side if existing else "long"

        trade = Trade(
            signal_id=signal.id,
            bar=signal.bar,
            sym=signal.sym,
            side=side or "long",
            qty=qty,
            fill_price=fill_price,
            fee=fee,
            funding=0.0,
            exchange_order_id=order_id,
            timestamp=datetime.now(timezone.utc),
        )
        return OrderResult(
            status=OrderStatus.FILLED,
            signal_id=signal.id,
            sym=signal.sym,
            exchange_order_id=order_id,
            trade=trade,
        )

    def get_position(self, sym: str) -> Optional[ExchangePosition]:
        """Текущая позиция по symbol. None если нет.

        Зависит от v1 connector API. Заменить на реальный вызов.
        """
        try:
            raw = self._client.get_position(sym)
            if not raw:
                return None
            return ExchangePosition(
                sym=str(raw.get("symbol") or sym).upper(),
                side=str(raw.get("side") or "long").lower(),
                qty=float(raw.get("qty") or 0.0),
                entry=float(raw.get("entry") or 0.0),
                leverage=int(raw.get("leverage") or 1),
                unrealized_pnl=float(raw.get("unrealized_pnl") or 0.0),
            )
        except Exception:
            return None

    def get_all_positions(self) -> Dict[str, ExchangePosition]:
        try:
            raw = self._client.get_all_positions()
        except Exception:
            return {}
        out: Dict[str, ExchangePosition] = {}
        for item in raw or []:
            try:
                pos = ExchangePosition(
                    sym=str(item.get("symbol") or "").upper(),
                    side=str(item.get("side") or "long").lower(),
                    qty=float(item.get("qty") or 0.0),
                    entry=float(item.get("entry") or 0.0),
                    leverage=int(item.get("leverage") or 1),
                    unrealized_pnl=float(item.get("unrealized_pnl") or 0.0),
                )
                if pos.sym:
                    out[pos.sym] = pos
            except (TypeError, ValueError):
                continue
        return out

    def get_min_notional(self, sym: str) -> float:
        """Минимальный размер позиции в USDT.

        v1 connector обычно знает это per-exchange. Если нет — fallback.
        """
        try:
            value = self._client.get_min_notional(sym)
            if value and value > 0:
                return float(value)
        except Exception:
            pass
        return self._default_min_notional

    # ── Internal helpers ────────────────────────────────────────────

    def _select_v1_method(self, action: Action):
        """Маппинг Action → конкретный метод v1-клиента.

        Заменить на реальные имена методов вашего connector-а.
        """
        if action.is_long_open:
            return self._client.create_long_order  # noqa: TODO real method
        if action.is_short_open:
            return self._client.create_short_order  # noqa: TODO
        if action.is_close:
            return self._client.close_position     # noqa: TODO
        raise ValueError(f"No method for action {action.name}")


# ────────────────────────────────────────────────────────────────────
# Static check — V1ExchangeAdapter must satisfy Exchange Protocol
# ────────────────────────────────────────────────────────────────────


def _validate_protocol() -> None:
    """Проверка что класс соответствует Protocol (вызывается при impor-те)."""
    # Этот вызов сработает при создании реального инстанса —
    # `isinstance(adapter, Exchange)` должен вернуть True.
    pass
