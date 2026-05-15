"""Shared v1 futures client adapter for the v2 Exchange protocol."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from ..domain.types import Action, Signal, Trade
from ..execution.exchange import ExchangePosition, OrderResult, OrderStatus


log = logging.getLogger(__name__)


class ExchangeMetadataUnavailable(RuntimeError):
    """Raised when open-order metadata is unavailable or marked as fallback."""


def load_runtime_leverage(exchange_name: Optional[str] = None, default: int = 2) -> int:
    """Best-effort read of the existing v1 leverage setting."""
    try:
        exchange_name, default = _compat_exchange_arg(exchange_name, default)
        from .exchange_profile import load_exchange_profile

        return load_exchange_profile(
            exchange_name,
            default_leverage=int(default),
        ).leverage
    except Exception:
        return int(default)


def load_runtime_trade_fraction(
    exchange_name: Optional[str] = None,
    default: float = 0.10,
) -> float:
    """Best-effort read of the existing v1 trade_fraction setting."""
    try:
        exchange_name, default = _compat_exchange_arg(exchange_name, default)
        from .exchange_profile import load_exchange_profile

        value = load_exchange_profile(
            exchange_name,
            default_trade_fraction=float(default),
        ).trade_fraction
        return value if 0 < value <= 1.0 else float(default)
    except Exception:
        return float(default)


def _compat_exchange_arg(exchange_name: Optional[str], default):
    if exchange_name is not None and not isinstance(exchange_name, str):
        return None, exchange_name
    return exchange_name, default


@dataclass(frozen=True)
class _PendingOrderContext:
    signal: Signal
    qty: float
    side: str


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
        fail_closed_on_metadata_error: bool = True,
    ) -> None:
        self.name = name
        self._order_client = order_client
        self._read_client = read_client or order_client
        self._leverage = int(leverage if leverage is not None else load_runtime_leverage())
        self._default_min_notional = float(default_min_notional)
        self._default_fee_rate = float(default_fee_rate)
        self._close_via_place_order = bool(close_via_place_order)
        self._fail_closed_on_metadata_error = bool(fail_closed_on_metadata_error)
        self._pending_orders: Dict[str, _PendingOrderContext] = {}

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
        if self._fail_closed_on_metadata_error:
            self._contract_meta(sym, require_reliable=True)
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

    def quantize_order_qty(self, signal: Signal, qty: float) -> float:
        action = self._to_futures_action(signal.action)
        if action.is_open:
            vol = self._contracts_for_open_qty(signal.sym, qty)
            return self._qty_for_contracts(signal.sym, vol) if vol > 0 else 0.0
        if action.is_close:
            vol = self._contracts_for_qty(signal.sym, qty)
            return self._qty_for_contracts(signal.sym, vol) if vol > 0 else 0.0
        return float(qty or 0.0)

    def poll_order(self, order_id: str, signal: Signal) -> OrderResult:
        order_id = str(order_id or "")
        if not order_id:
            return OrderResult(
                status=OrderStatus.REJECTED,
                signal_id=signal.id,
                sym=signal.sym,
                message=f"{self.name} pending order has no order id",
            )

        raw = self._fetch_order_detail(order_id, signal)
        if raw is not None:
            status_text = self._extract_status(raw)
            if status_text in {"rejected", "reject", "failed", "error", "canceled", "cancelled", "expired"}:
                return OrderResult(
                    status=OrderStatus.REJECTED,
                    signal_id=signal.id,
                    sym=signal.sym,
                    exchange_order_id=order_id,
                    message=f"{self.name} order {order_id} {status_text}",
                )
            if self._is_fill_confirmed(raw):
                ctx = self._pending_orders.get(order_id)
                result = self._result_from_raw(
                    signal,
                    self._with_order_id(raw, order_id),
                    qty=(ctx.qty if ctx is not None else 0.0),
                    side=(ctx.side if ctx is not None else signal.action.side or "long"),
                )
                if result.status != OrderStatus.PENDING:
                    self._pending_orders.pop(order_id, None)
                return result

        inferred = self._infer_pending_fill_from_position(order_id, signal)
        if inferred is not None:
            return inferred

        return OrderResult(
            status=OrderStatus.PENDING,
            signal_id=signal.id,
            sym=signal.sym,
            exchange_order_id=order_id,
            message=f"{self.name} order is still pending",
        )

    def get_account_equity(self) -> float:
        """Read total USDT futures equity for live sizing/status.

        Existing v1 futures clients expose ``account_assets()`` as
        {"USDT": equity, "USDT_AVAIL": available_margin}. Prefer total equity;
        available margin is useful for exchange-side checks, but v2 sizing uses
        the account value as its balance base.
        """
        snapshot = self.get_account_snapshot()
        for key in ("current_balance", "futures_equity", "total_assets"):
            value = float(snapshot.get(key, 0.0) or 0.0)
            if value > 0:
                return value
        return 0.0

    def get_account_snapshot(self) -> Dict[str, float]:
        """Read a normalized live account snapshot for operator status.

        The v1 runtime exposes richer snapshots on direct clients. Panteon v2
        keeps only a compact USD view: futures equity/available, spot value,
        total assets and unrealized PnL.
        """
        sources = (self._read_client, self._order_client)
        for source in sources:
            if source is None:
                continue
            snapshot = getattr(source, "get_full_snapshot", None)
            if callable(snapshot):
                try:
                    normalized = self._snapshot_from_full_snapshot(snapshot())
                    if normalized.get("current_balance", 0.0) > 0:
                        return normalized
                except Exception:
                    pass

        for source in sources:
            if source is None:
                continue
            getter = getattr(source, "account_assets", None)
            if callable(getter):
                try:
                    normalized = self._snapshot_from_assets(getter())
                    if normalized.get("current_balance", 0.0) > 0:
                        return normalized
                except Exception:
                    pass

        return {}

    def _send_open(
        self,
        signal: Signal,
        qty: float,
        *,
        v1_side: int,
        trade_side: str,
    ) -> OrderResult:
        try:
            vol = self._contracts_for_open_qty(signal.sym, qty)
        except ExchangeMetadataUnavailable as exc:
            return self._rejected(signal, f"exchange metadata error: {exc}")
        if vol <= 0:
            return self._rejected(signal, f"qty {qty} is below contract minimum")
        sent_qty = self._qty_for_contracts(signal.sym, vol)
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
        result = self._result_from_raw(signal, raw, qty=sent_qty, side=trade_side)
        self._remember_pending(result, signal=signal, qty=sent_qty, side=trade_side)
        return result

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
            result = self._result_from_raw(signal, raw, qty=qty, side=close_side)
            self._remember_pending(result, signal=signal, qty=qty, side=close_side)
            return result

        try:
            raw = self._order_client.close_all(signal.sym)
        except Exception as exc:
            return self._rejected(
                signal,
                f"adapter exception: {type(exc).__name__}: {exc}",
            )
        result = self._result_from_raw(signal, raw, qty=qty, side=close_side)
        self._remember_pending(result, signal=signal, qty=qty, side=close_side)
        return result

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
        status_text = self._extract_status(raw)
        if status_text in {"rejected", "reject", "failed", "error", "canceled", "cancelled", "expired"}:
            return self._rejected(signal, f"{self.name} order {order_id} {status_text}")
        if not self._is_fill_confirmed(raw):
            return OrderResult(
                status=OrderStatus.PENDING,
                signal_id=signal.id,
                sym=signal.sym,
                exchange_order_id=order_id,
                message=f"{self.name} accepted order without confirmed fill",
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

    def _remember_pending(
        self,
        result: OrderResult,
        *,
        signal: Signal,
        qty: float,
        side: str,
    ) -> None:
        if result.status == OrderStatus.PENDING and result.exchange_order_id:
            self._pending_orders[result.exchange_order_id] = _PendingOrderContext(
                signal=signal,
                qty=float(qty or 0.0),
                side=side,
            )
        elif result.exchange_order_id:
            self._pending_orders.pop(result.exchange_order_id, None)

    def _fetch_order_detail(self, order_id: str, signal: Signal) -> Optional[dict]:
        for source in (self._read_client, self._order_client):
            if source is None:
                continue
            for method_name in (
                "get_order",
                "get_order_detail",
                "query_order",
                "order_detail",
                "fetch_order",
            ):
                method = getattr(source, method_name, None)
                if not callable(method):
                    continue
                for args in ((order_id, signal.sym), (order_id,), (signal.sym, order_id)):
                    try:
                        raw = method(*args)
                    except TypeError:
                        continue
                    except Exception:
                        break
                    normalized = self._unwrap_order_payload(raw)
                    if normalized is not None:
                        return normalized
        return None

    @classmethod
    def _unwrap_order_payload(cls, raw: Any) -> Optional[dict]:
        if not isinstance(raw, dict):
            return None
        data = raw.get("data")
        if isinstance(data, dict):
            merged = dict(data)
            for key, value in raw.items():
                merged.setdefault(key, value)
            return merged
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                merged = dict(first)
                for key, value in raw.items():
                    merged.setdefault(key, value)
                return merged
        return raw

    @staticmethod
    def _extract_status(raw: dict) -> str:
        for key in ("status", "state", "orderStatus", "execStatus"):
            value = raw.get(key)
            if value is not None and value != "":
                return str(value).strip().lower()
        return ""

    @staticmethod
    def _with_order_id(raw: dict, order_id: str) -> dict:
        out = dict(raw)
        out.setdefault("order_id", order_id)
        out.setdefault("orderId", order_id)
        out.setdefault("success", True)
        return out

    def _infer_pending_fill_from_position(
        self,
        order_id: str,
        signal: Signal,
    ) -> Optional[OrderResult]:
        ctx = self._pending_orders.get(order_id)
        try:
            pos = self.get_position(signal.sym)
        except Exception:
            pos = None

        if signal.action.is_open:
            expected_side = signal.action.side
            if pos is None or (expected_side and pos.side != expected_side):
                return None
            qty = pos.qty if pos.qty > 0 else (ctx.qty if ctx is not None else 0.0)
            if qty <= 0 or pos.entry <= 0:
                return None
            fee = qty * pos.entry * self._default_fee_rate
            trade = Trade(
                signal_id=signal.id,
                bar=signal.bar,
                sym=signal.sym,
                side=pos.side,
                qty=qty,
                fill_price=pos.entry,
                fee=fee,
                funding=0.0,
                exchange_order_id=order_id,
                timestamp=datetime.now(timezone.utc),
            )
            self._pending_orders.pop(order_id, None)
            return OrderResult(
                status=OrderStatus.FILLED,
                signal_id=signal.id,
                sym=signal.sym,
                exchange_order_id=order_id,
                trade=trade,
            )

        if signal.action.is_close and ctx is not None and pos is None:
            qty = ctx.qty
            if qty <= 0 or signal.price <= 0:
                return None
            fee = qty * signal.price * self._default_fee_rate
            trade = Trade(
                signal_id=signal.id,
                bar=signal.bar,
                sym=signal.sym,
                side=ctx.side or "long",
                qty=qty,
                fill_price=signal.price,
                fee=fee,
                funding=0.0,
                exchange_order_id=order_id,
                timestamp=datetime.now(timezone.utc),
            )
            self._pending_orders.pop(order_id, None)
            return OrderResult(
                status=OrderStatus.FILLED,
                signal_id=signal.id,
                sym=signal.sym,
                exchange_order_id=order_id,
                trade=trade,
            )
        return None

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
        return self._contracts_for_qty_from_meta(qty, meta, floor_to_min=False)

    def _contracts_for_open_qty(self, sym: str, qty: float) -> int:
        meta = self._contract_meta(sym, require_reliable=True)
        return self._contracts_for_qty_from_meta(qty, meta, floor_to_min=True)

    def _contracts_for_qty_from_meta(
        self,
        qty: float,
        meta: dict,
        *,
        floor_to_min: bool,
    ) -> int:
        step = self._contract_qty_step(meta)
        raw_vol = float(qty or 0.0) / step if step > 0 else 0.0
        min_vol = max(1, int(float(meta.get("minVol", 1) or 1)))
        vol_unit = max(1, int(float(meta.get("volUnit", 1) or 1)))
        min_vol = int(math.ceil(min_vol / vol_unit) * vol_unit)
        vol = int(math.floor(raw_vol / vol_unit) * vol_unit)
        if floor_to_min and raw_vol > 0 and vol < min_vol:
            return min_vol
        return vol if vol >= min_vol else 0

    def _qty_for_contracts(self, sym: str, contracts: int) -> float:
        return float(contracts) * self._contract_qty_step(self._contract_meta(sym))

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
        amount = self._first_float(
            raw,
            "filledQty",
            "filled_qty",
            "executedQty",
            "cumExecQty",
            "dealVol",
            "filledAmount",
            "amount",
            "qty",
            "quantity",
            default=0.0,
        )
        if amount > 0:
            return amount
        contracts = self._first_float(raw, "contracts", "vol", default=0.0)
        size = self._first_float(raw, "contractSize", "amountStep", default=0.0)
        if contracts > 0 and size > 0:
            return contracts * size
        return float(fallback)

    @classmethod
    def _is_fill_confirmed(cls, raw: dict) -> bool:
        status_text = cls._extract_status(raw)
        if status_text in {"filled", "closed", "done", "complete", "completed"}:
            return True
        if status_text in {
            "open",
            "new",
            "created",
            "pending",
            "accepted",
            "submitted",
            "partially_filled",
            "partial",
            "live",
        }:
            return False
        return cls._has_fill_evidence(raw)

    @staticmethod
    def _has_fill_evidence(raw: dict) -> bool:
        for key in (
            "fillPrice",
            "fill_price",
            "avgPrice",
            "average",
            "dealAvgPrice",
            "filledAvgPrice",
            "priceAvg",
            "executedPrice",
            "filledQty",
            "filled_qty",
            "executedQty",
            "cumExecQty",
            "dealVol",
            "filledAmount",
        ):
            value = raw.get(key)
            if value not in (None, "", 0, "0"):
                return True
        data = raw.get("data")
        if isinstance(data, dict):
            return V1FuturesExchangeAdapter._has_fill_evidence(data)
        return False

    def _contract_meta(self, sym: str, *, require_reliable: bool = False) -> dict:
        getter = getattr(self._order_client, "_get_contract_meta", None)
        error: Optional[Exception] = None
        if callable(getter):
            try:
                meta = getter(self._normalize_symbol(sym))
                if isinstance(meta, dict):
                    if require_reliable and self._fail_closed_on_metadata_error:
                        self._assert_reliable_contract_meta(sym, meta)
                    return meta
            except ExchangeMetadataUnavailable:
                raise
            except Exception as exc:
                error = exc
        if require_reliable and self._fail_closed_on_metadata_error:
            detail = f": {type(error).__name__}: {error}" if error is not None else ""
            raise ExchangeMetadataUnavailable(
                f"{self.name} contract metadata unavailable for {sym}{detail}"
            )
        return {}

    def _assert_reliable_contract_meta(self, sym: str, meta: dict) -> None:
        if not meta:
            raise ExchangeMetadataUnavailable(f"{self.name} empty contract metadata for {sym}")
        if bool(meta.get("metadataFallback")):
            raise ExchangeMetadataUnavailable(f"{self.name} fallback contract metadata for {sym}")
        source = str(meta.get("metadataSource") or "").strip().lower()
        if source in {"fallback", "default", "synthetic"}:
            raise ExchangeMetadataUnavailable(f"{self.name} fallback contract metadata for {sym}")
        if meta.get("apiAllowed") is False:
            raise ExchangeMetadataUnavailable(f"{self.name} contract API disabled for {sym}")
        state = meta.get("state")
        if state not in (None, "", 0, "0", "online", "enabled", "normal", "live", "trading"):
            raise ExchangeMetadataUnavailable(f"{self.name} contract state {state!r} for {sym}")
        step = self._first_float(meta, "contractSize", "amountStep", "sizeIncrement", default=0.0)
        if step <= 0:
            raise ExchangeMetadataUnavailable(f"{self.name} invalid contract size for {sym}")

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

    @classmethod
    def _snapshot_from_assets(cls, raw: Any) -> Dict[str, float]:
        equity = cls._equity_from_assets(raw)
        available = equity
        if isinstance(raw, dict):
            for key in ("USDT_AVAIL", "available", "availableBalance", "free"):
                try:
                    value = raw.get(key)
                    if value is not None and value != "":
                        available = float(value)
                        break
                except (TypeError, ValueError):
                    continue
        return {
            "current_balance": equity,
            "futures_equity": equity,
            "available_balance": available,
            "spot_assets": 0.0,
            "total_assets": equity,
            "unrealized_pnl": 0.0,
        }

    @classmethod
    def _snapshot_from_full_snapshot(cls, raw: Any) -> Dict[str, float]:
        if not isinstance(raw, dict):
            return {}
        futures = raw.get("futures") if isinstance(raw.get("futures"), dict) else {}
        spot = raw.get("spot") if isinstance(raw.get("spot"), dict) else {}

        futures_equity = cls._first_positive(
            futures.get("equity"),
            raw.get("futures_equity"),
            raw.get("primary_capital"),
            cls._equity_from_assets(raw.get("assets") or raw.get("balances")),
        )
        available = cls._first_positive(
            futures.get("available"),
            futures.get("availableBalance"),
            raw.get("available_balance"),
            futures_equity,
        )
        unrealized = cls._first_float_value(
            futures.get("unrealized"),
            futures.get("unrealized_pnl"),
            raw.get("unrealized_pnl"),
        )
        spot_assets = cls._first_float_value(
            spot.get("total_value"),
            spot.get("total"),
            raw.get("spot_assets"),
        )
        total_assets = cls._first_positive(
            raw.get("total_equity"),
            raw.get("total_assets"),
            futures_equity + spot_assets,
        )
        return {
            "current_balance": futures_equity,
            "futures_equity": futures_equity,
            "available_balance": available,
            "spot_assets": spot_assets,
            "total_assets": total_assets,
            "unrealized_pnl": unrealized,
        }

    @staticmethod
    def _first_positive(*values: Any) -> float:
        for value in values:
            try:
                out = float(value)
                if out > 0:
                    return out
            except (TypeError, ValueError):
                continue
        return 0.0

    @staticmethod
    def _first_float_value(*values: Any) -> float:
        for value in values:
            try:
                if value is not None and value != "":
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
