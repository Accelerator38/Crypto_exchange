"""Exchange-specific runtime profile for Panteon v2 live adapters."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ExchangeProfile:
    exchange_id: str
    display_name: str
    leverage: int
    trade_fraction: float
    symbols: Optional[Tuple[str, ...]] = None
    spot_fee: float = 0.001
    futures_fee: float = 0.0006
    slippage: float = 0.0
    connector_module: str = ""
    api_module: str = ""
    api_key_env: str = ""
    api_secret_env: str = ""
    api_passphrase_env: str = ""
    trading_mode_env: str = ""
    raw_settings: Mapping[str, Any] = field(default_factory=dict)
    parsed_settings: Mapping[str, Any] = field(default_factory=dict)


def load_exchange_profile(
    exchange_name: Optional[str] = None,
    *,
    default_leverage: int = 2,
    default_trade_fraction: float = 0.10,
) -> ExchangeProfile:
    """Load exchange-specific runtime settings through the v1 registry.

    Settings may be global (``trade_fraction``) or exchange-prefixed
    (``bitget_trade_fraction`` / ``trade_fraction_bitget``). Prefixed values
    always win for the requested exchange.
    """
    exchange_id = _normalize_exchange_id(exchange_name)
    raw: Mapping[str, Any] = {}
    parsed: Mapping[str, Any] = {}
    runtime = None
    try:
        from exchange_registry import load_exchange_runtime  # type: ignore

        runtime = load_exchange_runtime(exchange_id=exchange_id)
        raw = runtime.load_settings() or {}
        parsed = runtime.parse_settings(dict(raw)) or {}
    except Exception:
        raw, parsed = _fallback_connector_settings(exchange_id)

    leverage = int(
        _exchange_value(
            raw,
            parsed,
            exchange_id=exchange_id,
            key="leverage",
            default=default_leverage,
        )
        or default_leverage
    )
    trade_fraction = float(
        _exchange_value(
            raw,
            parsed,
            exchange_id=exchange_id,
            key="trade_fraction",
            default=default_trade_fraction,
        )
        or default_trade_fraction
    )
    if not 0 < trade_fraction <= 1.0:
        trade_fraction = float(default_trade_fraction)

    display = str(getattr(runtime, "display_name", "") or exchange_id.upper())
    env_prefix = display.upper()
    symbols = _symbols_from_value(parsed.get("symbols") if isinstance(parsed, Mapping) else None)

    return ExchangeProfile(
        exchange_id=exchange_id,
        display_name=display,
        leverage=max(1, leverage),
        trade_fraction=trade_fraction,
        symbols=symbols,
        spot_fee=float(parsed.get("spot_fee", 0.001) or 0.001),
        futures_fee=float(parsed.get("futures_fee", 0.0006) or 0.0006),
        slippage=float(parsed.get("slippage", 0.0) or 0.0),
        connector_module=str(getattr(runtime, "connector_module_name", "") or ""),
        api_module=str(getattr(runtime, "api_module_name", "") or ""),
        api_key_env=str(getattr(runtime, "api_key_env", "") or f"{env_prefix}_API_KEY"),
        api_secret_env=str(getattr(runtime, "api_secret_env", "") or f"{env_prefix}_SECRET_KEY"),
        api_passphrase_env=str(getattr(runtime, "api_passphrase_env", "") or f"{env_prefix}_PASSPHRASE"),
        trading_mode_env=str(getattr(runtime, "trading_mode_env", "") or f"{env_prefix}_TRADING_MODE"),
        raw_settings=dict(raw),
        parsed_settings=dict(parsed),
    )


def _normalize_exchange_id(exchange_name: Optional[str]) -> str:
    raw = str(exchange_name or os.getenv("CRYPTO_EXCHANGE") or "mexc").strip().lower()
    return raw or "mexc"


def _exchange_value(
    raw: Mapping[str, Any],
    parsed: Mapping[str, Any],
    *,
    exchange_id: str,
    key: str,
    default: Any,
) -> Any:
    candidates = (
        f"{exchange_id}_{key}",
        f"{key}_{exchange_id}",
        key,
    )
    for name in candidates[:2]:
        if name in raw:
            return _coerce(raw.get(name), default)
    if key in parsed:
        return _coerce(parsed.get(key), default)
    if key in raw:
        return _coerce(raw.get(key), default)
    return default


def _coerce(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    try:
        if isinstance(default, int):
            return int(float(value))
        if isinstance(default, float):
            return float(value)
    except (TypeError, ValueError):
        return default
    return value


def _symbols_from_value(value: Any) -> Optional[Tuple[str, ...]]:
    if value is None:
        return None
    if isinstance(value, str):
        if not value or value.lower() == "all":
            return None
        parts: Sequence[Any] = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        parts = list(value)
    else:
        return None
    out = tuple(str(part).strip().upper() for part in parts if str(part).strip())
    return out or None


def _fallback_connector_settings(exchange_id: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    module_name = "bitget_connector" if exchange_id == "bitget" else "mexc_connector"
    try:
        module = __import__(module_name)
        load = getattr(module, "_load_settings")
        parse = getattr(module, "_parse_settings")
        raw = load() or {}
        return raw, parse(raw) or {}
    except Exception:
        return {}, {}
