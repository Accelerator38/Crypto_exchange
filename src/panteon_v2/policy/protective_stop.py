"""Shared protective-stop semantics for replay, paper and Bitget live."""

from __future__ import annotations

import math


class ProtectiveStopError(ValueError):
    """Protective-stop inputs are incomplete or internally inconsistent."""


def protective_stop_price(
    reference_price: float,
    side: str,
    stop_loss_pct: float,
) -> float:
    price = _positive_finite(reference_price, "reference_price")
    pct = _positive_finite(stop_loss_pct, "stop_loss_pct")
    normalized_side = _side(side)
    multiplier = 1.0 - pct / 100.0 if normalized_side == "long" else 1.0 + pct / 100.0
    stop = price * multiplier
    if stop <= 0.0:
        raise ProtectiveStopError("stop_loss_pct produces a nonpositive stop price")
    return stop


def protective_stop_breached(side: str, market_price: float, stop_price: float) -> bool:
    normalized_side = _side(side)
    current = _positive_finite(market_price, "market_price")
    stop = _positive_finite(stop_price, "stop_price")
    return current <= stop if normalized_side == "long" else current >= stop


def intrabar_stop_fill_reference(
    side: str,
    stop_price: float,
    *,
    open_price: float,
    high: float,
    low: float,
    close: float,
) -> float | None:
    """Return a conservative stop-market reference fill when the bar breaches.

    A gap through the stop uses the worse bar open. Normal intrabar breaches use
    the stop trigger itself; the execution model applies adverse slippage after
    this function.
    """

    normalized_side = _side(side)
    stop = _positive_finite(stop_price, "stop_price")
    opened = _positive_finite(open_price, "open_price")
    highest = _positive_finite(high, "high")
    lowest = _positive_finite(low, "low")
    closed = _positive_finite(close, "close")
    if not lowest <= min(opened, closed) <= max(opened, closed) <= highest:
        raise ProtectiveStopError("bar OHLC is inconsistent")
    if normalized_side == "long":
        if lowest > stop:
            return None
        return min(opened, stop)
    if highest < stop:
        return None
    return max(opened, stop)


def _side(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in {"long", "short"}:
        raise ProtectiveStopError(f"unsupported position side: {value!r}")
    return normalized


def _positive_finite(value: float, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtectiveStopError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ProtectiveStopError(f"{name} must be positive and finite")
    return parsed
