from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYMBOLS = ("BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "TRX")
MEXC_CONTRACT_DETAIL_URL = "https://contract.mexc.com/api/v1/contract/detail"
BITGET_CONTRACTS_URL = "https://api.bitget.com/api/v2/mix/market/contracts"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "y", "on", "enabled", "enable"}


def _parse_symbols(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    if raw is None:
        return DEFAULT_SYMBOLS
    if isinstance(raw, str):
        parts = raw.replace(";", ",").split(",")
    else:
        parts = []
        for item in raw:
            parts.extend(str(item or "").replace(";", ",").split(","))
    symbols = tuple(dict.fromkeys(_base_symbol(part) for part in parts if str(part or "").strip()))
    return symbols or DEFAULT_SYMBOLS


def _base_symbol(raw: Any) -> str:
    text = str(raw or "").strip().upper()
    if not text:
        return ""
    if ":" in text:
        text = text.split(":", 1)[0]
    text = text.replace("-", "_").replace("/", "_")
    if "_" in text:
        return text.split("_", 1)[0]
    for suffix in ("USDT", "USDC", "USD"):
        if text.endswith(suffix) and len(text) > len(suffix):
            return text[: -len(suffix)]
    return text


def _price_tick_from_scale(value: Any) -> float:
    scale = _int(value, -1)
    if scale < 0:
        return 0.0
    return 10.0 ** (-scale)


def _mexc_data_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    data = payload.get("data", payload)
    if isinstance(data, Mapping):
        return [data]
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return [item for item in data if isinstance(item, Mapping)]
    return []


def normalize_mexc_contract_info(
    payload: Mapping[str, Any],
    *,
    symbols: Sequence[str] = DEFAULT_SYMBOLS,
) -> dict[str, Any]:
    requested = tuple(dict.fromkeys(_base_symbol(item) for item in symbols if _base_symbol(item)))
    rows_by_symbol: dict[str, Mapping[str, Any]] = {}
    for row in _mexc_data_rows(payload):
        base = _base_symbol(row.get("symbol") or row.get("displayName") or row.get("baseCoin"))
        if base:
            rows_by_symbol[base] = row

    rules: list[dict[str, Any]] = []
    failures: list[str] = []
    for symbol in requested:
        row = rows_by_symbol.get(symbol)
        if row is None:
            failures.append(f"{symbol}:missing")
            continue
        price_tick = _float(row.get("priceUnit"), 0.0) or _price_tick_from_scale(row.get("priceScale"))
        quantity_step = _float(row.get("volUnit"), 0.0) or _price_tick_from_scale(row.get("volScale"))
        min_order_size = _float(row.get("minVol"), 0.0)
        max_order_size = _float(row.get("maxVol"), 0.0)
        contract_size = _float(row.get("contractSize"), 0.0)
        state = str(row.get("state", "")).strip().lower()
        active = _bool(row.get("apiAllowed"), True) and state not in {"offline", "closed", "disabled", "3"}
        rule = {
            "exchange": "MEXC",
            "symbol": symbol,
            "market_symbol": str(row.get("symbol") or f"{symbol}_USDT"),
            "active": bool(active),
            "min_order_size": min_order_size,
            "max_order_size": max_order_size,
            "min_base_amount": min_order_size * contract_size if contract_size else 0.0,
            "price_tick": price_tick,
            "quantity_step": quantity_step,
            "contract_size": contract_size,
            "price_scale": _int(row.get("priceScale"), 0),
            "quantity_scale": _int(row.get("volScale"), 0),
            "amount_scale": _int(row.get("amountScale"), 0),
        }
        rules.append(rule)
        if not active:
            failures.append(f"{symbol}:inactive")
        if min_order_size <= 0:
            failures.append(f"{symbol}:min_order_size_missing")
        if price_tick <= 0:
            failures.append(f"{symbol}:price_tick_missing")
        if quantity_step <= 0:
            failures.append(f"{symbol}:quantity_step_missing")
        if contract_size <= 0:
            failures.append(f"{symbol}:contract_size_missing")

    return {
        "exchange": "MEXC",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requested_symbols": list(requested),
        "passed": not failures and len(rules) == len(requested),
        "rules": rules,
        "failures": failures,
    }


def normalize_bitget_markets(
    markets: Mapping[str, Any],
    *,
    symbols: Sequence[str] = DEFAULT_SYMBOLS,
) -> dict[str, Any]:
    requested = tuple(dict.fromkeys(_base_symbol(item) for item in symbols if _base_symbol(item)))
    by_symbol: dict[str, Mapping[str, Any]] = {}
    for market_symbol, market in markets.items():
        if not isinstance(market, Mapping):
            continue
        base = _base_symbol(market.get("symbol") or market_symbol or market.get("id"))
        if base and (str(market.get("quote") or "").upper() in {"", "USDT"}):
            by_symbol.setdefault(base, market)

    rules: list[dict[str, Any]] = []
    failures: list[str] = []
    for symbol in requested:
        market = by_symbol.get(symbol)
        if market is None:
            failures.append(f"{symbol}:missing")
            continue
        info = market.get("info") if isinstance(market.get("info"), Mapping) else {}
        limits = market.get("limits") if isinstance(market.get("limits"), Mapping) else {}
        amount_limits = limits.get("amount") if isinstance(limits.get("amount"), Mapping) else {}
        cost_limits = limits.get("cost") if isinstance(limits.get("cost"), Mapping) else {}
        precision = market.get("precision") if isinstance(market.get("precision"), Mapping) else {}
        min_order_size = (
            _float(amount_limits.get("min"), 0.0)
            or _float(info.get("minTradeNum"), 0.0)
            or _float(info.get("minTradeAmount"), 0.0)
        )
        max_order_size = (
            _float(amount_limits.get("max"), 0.0)
            or _float(info.get("maxTradeAmount"), 0.0)
            or _float(info.get("maxTradeNum"), 0.0)
        )
        quantity_step = (
            _float(info.get("sizeMultiplier"), 0.0)
            or _float(precision.get("amount"), 0.0)
            or min_order_size
        )
        price_tick = _float(precision.get("price"), 0.0) or _float(info.get("pricePlace"), 0.0)
        contract_size = _float(market.get("contractSize"), 0.0) or _float(info.get("sizeMultiplier"), 0.0)
        min_notional = (
            _float(cost_limits.get("min"), 0.0)
            or _float(info.get("minTradeUSDT"), 0.0)
            or _float(info.get("minTradeUSDT"), 0.0)
        )
        active = _bool(market.get("active"), True)
        rule = {
            "exchange": "BITGET",
            "symbol": symbol,
            "market_symbol": str(market.get("symbol") or ""),
            "market_id": str(market.get("id") or ""),
            "active": bool(active),
            "min_order_size": min_order_size,
            "max_order_size": max_order_size,
            "min_notional_usd": min_notional,
            "price_tick": price_tick,
            "quantity_step": quantity_step,
            "contract_size": contract_size,
        }
        rules.append(rule)
        if not active:
            failures.append(f"{symbol}:inactive")
        if min_order_size <= 0:
            failures.append(f"{symbol}:min_order_size_missing")
        if quantity_step <= 0:
            failures.append(f"{symbol}:quantity_step_missing")
        if contract_size <= 0:
            failures.append(f"{symbol}:contract_size_missing")

    return {
        "exchange": "BITGET",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requested_symbols": list(requested),
        "passed": not failures and len(rules) == len(requested),
        "rules": rules,
        "failures": failures,
    }


def normalize_bitget_contracts(
    payload: Mapping[str, Any],
    *,
    symbols: Sequence[str] = DEFAULT_SYMBOLS,
) -> dict[str, Any]:
    requested = tuple(dict.fromkeys(_base_symbol(item) for item in symbols if _base_symbol(item)))
    data = payload.get("data", payload)
    rows: list[Mapping[str, Any]]
    if isinstance(data, Mapping):
        rows = [data]
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        rows = [item for item in data if isinstance(item, Mapping)]
    else:
        rows = []
    by_symbol: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        base = _base_symbol(row.get("baseCoin") or row.get("symbol") or row.get("symbolName"))
        if base:
            by_symbol[base] = row

    rules: list[dict[str, Any]] = []
    failures: list[str] = []
    for symbol in requested:
        row = by_symbol.get(symbol)
        if row is None:
            failures.append(f"{symbol}:missing")
            continue
        status = str(row.get("symbolStatus") or row.get("status") or "").strip().lower()
        active = status in {"normal", "online", "1", ""} and str(row.get("offTime", "") or "") in {"", "-1"}
        min_order_size = _float(row.get("minTradeNum"), 0.0)
        max_order_size = _float(row.get("maxTradeNum"), 0.0) or _float(row.get("maxTradeAmount"), 0.0)
        quantity_step = _float(row.get("sizeMultiplier"), 0.0) or min_order_size
        min_notional = _float(row.get("minTradeUSDT"), 0.0)
        price_place = _int(row.get("pricePlace"), -1)
        price_end_step = _float(row.get("priceEndStep"), 0.0) or 1.0
        price_tick = price_end_step * (10.0 ** (-price_place)) if price_place >= 0 else 0.0
        contract_size = quantity_step
        rule = {
            "exchange": "BITGET",
            "symbol": symbol,
            "market_symbol": str(row.get("symbol") or f"{symbol}USDT"),
            "market_id": str(row.get("symbol") or f"{symbol}USDT"),
            "active": bool(active),
            "min_order_size": min_order_size,
            "max_order_size": max_order_size,
            "min_notional_usd": min_notional,
            "price_tick": price_tick,
            "quantity_step": quantity_step,
            "contract_size": contract_size,
            "price_place": price_place,
            "volume_place": _int(row.get("volumePlace"), 0),
        }
        rules.append(rule)
        if not active:
            failures.append(f"{symbol}:inactive")
        if min_order_size <= 0:
            failures.append(f"{symbol}:min_order_size_missing")
        if quantity_step <= 0:
            failures.append(f"{symbol}:quantity_step_missing")
        if price_tick <= 0:
            failures.append(f"{symbol}:price_tick_missing")

    return {
        "exchange": "BITGET",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requested_symbols": list(requested),
        "passed": not failures and len(rules) == len(requested),
        "rules": rules,
        "failures": failures,
    }


def fetch_mexc_contract_info(*, symbols: Sequence[str], timeout_sec: float = 15.0) -> dict[str, Any]:
    rows: list[Mapping[str, Any]] = []
    for symbol in symbols:
        contract = f"{_base_symbol(symbol)}_USDT"
        query = urllib.parse.urlencode({"symbol": contract})
        url = f"{MEXC_CONTRACT_DETAIL_URL}?{query}"
        with urllib.request.urlopen(url, timeout=float(timeout_sec)) as response:
            payload = json.loads(response.read().decode("utf-8"))
        data = payload.get("data") if isinstance(payload, Mapping) else None
        if isinstance(data, Mapping):
            rows.append(data)
        elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            rows.extend(item for item in data if isinstance(item, Mapping))
    return normalize_mexc_contract_info({"data": rows}, symbols=symbols)


def fetch_bitget_contracts(*, symbols: Sequence[str], timeout_sec: float = 15.0) -> dict[str, Any]:
    query = urllib.parse.urlencode({"productType": "USDT-FUTURES"})
    url = f"{BITGET_CONTRACTS_URL}?{query}"
    with urllib.request.urlopen(url, timeout=float(timeout_sec)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return normalize_bitget_contracts(payload, symbols=symbols)


def fetch_bitget_markets(*, symbols: Sequence[str]) -> dict[str, Any]:
    try:
        import ccxt  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("ccxt is required to fetch Bitget futures rules") from exc

    client = ccxt.bitget(
        {
            "enableRateLimit": True,
            "options": {"defaultType": "swap", "defaultSubType": "linear"},
        }
    )
    markets = client.load_markets()
    return normalize_bitget_markets(markets, symbols=symbols)


def build_rules_report(
    *,
    exchanges: Sequence[str],
    symbols: Sequence[str],
    timeout_sec: float = 15.0,
) -> dict[str, Any]:
    exchange_reports: dict[str, Any] = {}
    for exchange in exchanges:
        key = str(exchange or "").strip().upper()
        if not key:
            continue
        try:
            if key == "MEXC":
                exchange_reports[key] = fetch_mexc_contract_info(
                    symbols=symbols,
                    timeout_sec=timeout_sec,
                )
            elif key == "BITGET":
                exchange_reports[key] = fetch_bitget_contracts(
                    symbols=symbols,
                    timeout_sec=timeout_sec,
                )
            else:
                exchange_reports[key] = {
                    "exchange": key,
                    "passed": False,
                    "rules": [],
                    "failures": [f"{key}:unsupported_exchange"],
                }
        except Exception as exc:  # pragma: no cover - network/environment dependent
            exchange_reports[key] = {
                "exchange": key,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "requested_symbols": list(symbols),
                "passed": False,
                "rules": [],
                "failures": [f"fetch_failed:{type(exc).__name__}:{exc}"],
            }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbols": list(symbols),
        "passed": bool(exchange_reports) and all(
            bool(item.get("passed")) for item in exchange_reports.values()
        ),
        "exchanges": exchange_reports,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch and normalize public futures contract rules for live pre-flight checks."
    )
    parser.add_argument("--exchange", action="append", dest="exchanges", default=None)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--timeout-sec", type=float, default=15.0)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    exchanges = tuple(args.exchanges or ("MEXC", "BITGET"))
    symbols = _parse_symbols(args.symbols)
    report = build_rules_report(
        exchanges=exchanges,
        symbols=symbols,
        timeout_sec=float(args.timeout_sec),
    )

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
