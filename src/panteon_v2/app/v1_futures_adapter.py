"""Shared v1 futures client adapter for the v2 Exchange protocol."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

from ..domain.types import Action, Signal, Trade
from ..execution.exchange import ExchangePosition, OrderResult, OrderStatus


log = logging.getLogger(__name__)


def load_runtime_leverage(default: int = 2) -> int:
    """Best-effort read of the existing v1 leverage setting."""
    try:
        from mexc_connector import _load_settings, _parse_settings  # type: ignore

        cfg = _parse_settings(_load_settings())
        return int(float(cfg.get("leverage", default) or default))
    except Exception:
        return int(default)


def load_runtime_trade_fraction(default: float = 0.10) -> float:
    """Best-effort read of the existing v1 trade_fraction setting."""
    try:
        from mexc_connector import _load_settings, _parse_settings  # type: ignore

        cfg = _parse_settings(_load_settings())
        value = float(cfg.get("trade_fraction", default) or default)
        return value if 0 < value <= 1.0 else float(default)
    except Exception:
        return float(default)


class V1FuturesExchangeAdapter:
    """Adapter from v1 futures clients to the v2 Exchange protocol.

    The adapter expects an order client with ``place_order(symbol, side, vol,
    leverage)``. Position reads are best-effort through direct clients,
    futures clients, or the legacy private ``_req`` hook.
    """

    name: str

    def __init__(
        self,
        *,
        name: str,
        order_client: Any,
        read_client: Optional[Any] = None,
        leverage: Optional[int] = None,
        default_min_notional: float = 5.0,
        default_fee_rate: float = 0.0006,
        close_via_place_order: bool = False,
    ) -> None:
        self.name = name
        self._order_client = order_client
        self._read_client = read_client or order_client
        self._leverage = int(leverage if leverage is not None else load_runtime_leverage())
        self._default_min_notional = float(default_min_notional)
        self._default_fee_rate = float(default_fee_rate)
        self._close_via_place_order = bool(close_via_place_order)

    def send_order(self, signal: Signal, *, qty: float) -> OrderResult:
        action = self._to_futures_action(signal.action)
        if action.is_hold:
            return self._rejected(signal, "hold action is not sent to exchange")
        if action.is_close:
            return self._send_close(signal, qty)
        if action.is_long_open:
            return self._send_open(signal, qty, v1_side=1, trade_side="long")
        if action.is_short_open:
            return self._send_open(signal, qty, v1_side=3, trade_side="short")
        return self._rejected(signal, f"unsupported action {signal.action!r}")

    def get_position(self, sym: str) -> Optional[ExchangePosition]:
        target = self._normalize_symbol(sym)
        return self.get_all_positions().get(target)

    def get_all_positions(self) -> Dict[str, ExchangePosition]:
        out: Dict[str, ExchangePosition] = {}
        for raw in self._raw_positions():
            pos = self._normalize_position(raw)
            if pos is not None and pos.qty > 0:
                out[pos.sym] = pos
        return out

    def get_min_notional(self, sym: str) -> float:
        for attr in ("BITGET_MIN_NOTIONAL_USDT", "MIN_ORDER_USDT"):
            try:
                value = getattr(self._order_client, attr)
                if value and float(value) > 0:
                    return float(value)
            except Exception:
                pass
        try:
            value = self._order_client.get_min_notional(sym)  # type: ignore[name-defined]
            if value and float(value) > 0:
                return float(value)
        except Exception:
            pass
        return self._default_min_notional

    def get_account_equity(self) -> float:
        """Read total USDT futures equity for live sizing/status.

        Existing v1 futures clients expose ``account_assets()`` as
        {"USDT": equity, "USDT_AVAIL": available_margin}. Prefer total equity;
        available margin is useful for exchange-side checks, but v2 sizing uses
        the account value as its balance base.
        """
        for source in (self._order_client, self._read_client):
            if source is None:
                continue
            getter = getattr(source, "account_assets", None)
            if callable(getter):
                try:
                    value = self._equity_from_assets(getter())
                    if value > 0:
                        return value
                except Exception:
                    pass

            snapshot = getattr(source, "get_full_snapshot", None)
            if callable(snapshot):
                try:
                    raw = snapshot()
                    if isinstance(raw, dict):
                        value = self._equity_from_assets(raw.get("assets") or raw.get("balances"))
                        if value > 0:
                            return value
                except Exception:
                    pass
        return 0.0

    def _send_open(
        self,
        signal: Signal,
        qty: float,
        *,
        v1_side: int,
        trade_side: str,
    ) -> OrderResult:
        vol = self._contracts_for_qty(signal.sym, qty)
        if vol <= 0:
            return self._rejected(signal, f"qty {qty} is below contract minimum")
        try:
            raw = self._order_client.place_order(
                signal.sym,
                v1_side,
                vol,
                self._leverage,
            )
        except Exception as exc:
            return self._rejected(
                signal,
                f"adapter exception: {type(exc).__name__}: {exc}",
            )
        return self._result_from_raw(signal, raw, qty=qty, side=trade_side)

    def _send_close(self, signal: Signal, qty: float) -> OrderResult:
        existing = self.get_position(signal.sym)
        close_side = existing.side if existing is not None else "long"

        if self._close_via_place_order:
            vol = self._contracts_for_qty(signal.sym, qty)
            if vol <= 0:
                return self._rejected(signal, f"qty {qty} is below contract minimum")
            v1_side = 4 if close_side == "long" else 2
            try:
                raw = self._order_client.place_order(
                    signal.sym,
                    v1_side,
                    vol,
                    self._leverage,
                )
            except Exception as exc:
                return self._rejected(
                    signal,
                    f"adapter exception: {type(exc).__name__}: {exc}",
                )
            return self._result_from_raw(signal, raw, qty=qty, side=close_side)

        try:
            raw = self._order_client.close_all(signal.sym)
        except Exception as exc:
            return self._rejected(
                signal,
                f"adapter exception: {type(exc).__name__}: {exc}",
            )
        return self._result_from_raw(signal, raw, qty=qty, side=close_side)

    def _result_from_raw(
        self,
        signal: Signal,
        raw: Any,
        *,
        qty: float,
        side: str,
    ) -> OrderResult:
        if not isinstance(raw, dict):
            return self._rejected(signal, f"unexpected response type: {type(raw).__name__}")

        success = (
            bool(raw.get("success"))
            or raw.get("code") in (0, 200)
            or raw.get("data") is not None
        )
        if raw.get("skip") or not success:
            return self._rejected(
                signal,
                str(raw.get("msg") or raw.get("message") or raw.get("error") or raw),
            )

        order_id = self._extract_order_id(raw)
        if not order_id:
            return OrderResult(
                status=OrderStatus.PENDING,
                signal_id=signal.id,
                sym=signal.sym,
                message=f"{self.name} accepted order without order id",
            )

        fill_price = self._first_float(
            raw,
            "fillPrice",
            "fill_price",
            "avgPrice",
            "average",
            "price",
            default=signal.price,
        )
        actual_qty = self._actual_qty(raw, fallback=qty)
        fee = self._first_float(raw, "fee", "commission", default=0.0)
        if fee <= 0 and fill_price > 0 and actual_qty > 0:
            fee = actual_qty * fill_price * self._fee_rate(raw)

        trade = Trade(
            signal_id=signal.id,
            bar=signal.bar,
            sym=signal.sym,
            side=side,
            qty=actual_qty,
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

    def _raw_positions(self) -> Iterable[dict]:
        for source in (self._read_client, self._order_client):
            if source is None:
                continue
            getter = getattr(source, "get_futures_positions", None)
            if callable(getter):
                try:
                    positions = getter()
                    if positions:
                        return positions
                except Exception:
                    pass

            snapshot = getattr(source, "get_full_snapshot", None)
            if callable(snapshot):
                try:
                    positions = (
                        snapshot().get("futures", {}).get("positions", [])
                    )
                    if positions:
                        return positions
                except Exception:
                    pass

            req = getattr(source, "_req", None)
            if callable(req):
                try:
                    payload = req("GET", "/api/v1/private/position/open_positions")
                    positions = payload.get("data", []) if isinstance(payload, dict) else []
                    if positions:
                        return positions
                except Exception:
                    pass

            exchange = getattr(source, "exchange", None)
            fetch_positions = getattr(exchange, "fetch_positions", None)
            if callable(fetch_positions):
                try:
                    positions = fetch_positions(
                        params={"productType": "USDT-FUTURES", "marginCoin": "USDT"}
                    )
                    if positions:
                        return positions
                except Exception:
                    pass

        return []

    def _normalize_position(self, raw: dict) -> Optional[ExchangePosition]:
        if not isinstance(raw, dict):
            return None
        sym = self._normalize_symbol(
            raw.get("symbol")
            or raw.get("market")
            or raw.get("instId")
            or raw.get("symbolName")
            or ""
        )
        if not sym:
            return None

        side = self._normalize_side(raw)
        qty = self._position_qty(sym, raw)
        if qty <= 0:
            return None
        entry = self._first_float(
            raw,
            "entry",
            "entryPrice",
            "openAvgPrice",
            "holdAvgPrice",
            "averageOpenPrice",
            default=0.0,
        )
        leverage = int(self._first_float(raw, "leverage", default=1.0) or 1)
        pnl = self._first_float(
            raw,
            "unrealized_pnl",
            "unrealizedPnl",
            "unrealisedPnl",
            "unrealizedValue",
            "holdProfitLoss",
            "profit",
            default=0.0,
        )
        return ExchangePosition(
            sym=sym,
            side=side,
            qty=qty,
            entry=entry,
            leverage=leverage,
            unrealized_pnl=pnl,
        )

    def _contracts_for_qty(self, sym: str, qty: float) -> int:
        meta = self._contract_meta(sym)
        step = self._contract_qty_step(meta)
        raw_vol = float(qty or 0.0) / step if step > 0 else 0.0
        min_vol = max(1, int(float(meta.get("minVol", 1) or 1)))
        vol_unit = max(1, int(float(meta.get("volUnit", 1) or 1)))
        vol = int(math.floor(raw_vol / vol_unit) * vol_unit)
        return vol if vol >= min_vol else 0

    def _position_qty(self, sym: str, raw: dict) -> float:
        qty = self._first_float(raw, "qty", "quantity", "amount", default=0.0)
        if qty > 0:
            return qty
        contracts = self._first_float(raw, "contracts", "holdVol", "total", default=0.0)
        if contracts <= 0:
            return 0.0
        contract_size = self._first_float(
            raw,
            "contract_size",
            "contractSize",
            "amountStep",
            default=0.0,
        )
        if contract_size <= 0:
            contract_size = self._contract_qty_step(self._contract_meta(sym))
        return contracts * contract_size

    def _actual_qty(self, raw: dict, *, fallback: float) -> float:
        amount = self._first_float(raw, "amount", "qty", "quantity", default=0.0)
        if amount > 0:
            return amount
        contracts = self._first_float(raw, "contracts", "vol", default=0.0)
        size = self._first_float(raw, "contractSize", "amountStep", default=0.0)
        if contracts > 0 and size > 0:
            return contracts * size
        return float(fallback)

    def _contract_meta(self, sym: str) -> dict:
        getter = getattr(self._order_client, "_get_contract_meta", None)
        if callable(getter):
            try:
                meta = getter(self._normalize_symbol(sym))
                if isinstance(meta, dict):
                    return meta
            except Exception:
                pass
        return {}

    @staticmethod
    def _contract_qty_step(meta: dict) -> float:
        return max(
            float(
                meta.get("contractSize")
                or meta.get("amountStep")
                or meta.get("sizeIncrement")
                or 1.0
            ),
            1e-12,
        )

    def _fee_rate(self, raw: dict) -> float:
        fee_rate = self._first_float(raw, "feeRate", "takerFeeRate", default=0.0)
        return fee_rate if fee_rate > 0 else self._default_fee_rate

    @staticmethod
    def _first_float(raw: dict, *keys: str, default: float = 0.0) -> float:
        for key in keys:
            try:
                value = raw.get(key)
                if value is None or value == "":
                    continue
                return float(value)
            except (TypeError, ValueError):
                continue
        return float(default)

    @staticmethod
    def _extract_order_id(raw: dict) -> str:
        for key in ("order_id", "orderId", "id", "clientOrderId"):
            value = raw.get(key)
            if value:
                return str(value)
        data = raw.get("data")
        if isinstance(data, dict):
            for key in ("order_id", "orderId", "id", "clientOrderId"):
                value = data.get(key)
                if value:
                    return str(value)
        if data not in (None, "", [], {}):
            return str(data)
        return ""

    @staticmethod
    def _equity_from_assets(raw: Any) -> float:
        if not isinstance(raw, dict):
            return 0.0
        for key in ("USDT", "usdt", "equity", "total", "walletBalance", "marginBalance"):
            try:
                value = raw.get(key)
                if value is not None and float(value) > 0:
                    return float(value)
            except (TypeError, ValueError):
                continue
        return 0.0

    @staticmethod
    def _normalize_symbol(sym: str) -> str:
        value = str(sym or "").upper().strip()
        if ":" in value:
            value = value.split(":", 1)[0]
        if "/" in value:
            value = value.split("/", 1)[0]
        if value.endswith("_USDT"):
            value = value[:-5]
        if value.endswith("USDT") and len(value) > 4:
            value = value[:-4]
        return value

    @staticmethod
    def _normalize_side(raw: dict) -> str:
        side = str(raw.get("side") or raw.get("holdSide") or "").lower()
        if side in ("long", "short"):
            return side
        pos_type = str(raw.get("positionType") or raw.get("posSide") or "").lower()
        if pos_type in ("1", "long", "buy"):
            return "long"
        if pos_type in ("2", "short", "sell"):
            return "short"
        return "long"

    @staticmethod
    def _to_futures_action(action: Action) -> Action:
        if action == Action.SPOT_BUY_HALF:
            return Action.FUT_LONG_HALF
        if action == Action.SPOT_BUY_FULL:
            return Action.FUT_LONG_FULL
        if action == Action.SPOT_SELL_ALL:
            return Action.FUT_CLOSE_ALL
        return Action(action)

    @staticmethod
    def _rejected(signal: Signal, message: str) -> OrderResult:
        return OrderResult(
            status=OrderStatus.REJECTED,
            signal_id=signal.id,
            sym=signal.sym,
            message=message,
        )
