from __future__ import annotations

from dataclasses import dataclass


def normalize_exchange_id(exchange_id: str | None) -> str:
    return (exchange_id or "").strip().lower()


def normalize_market_type(market_type: str | None) -> str:
    raw = (market_type or "spot").strip().lower()
    aliases = {
        "futures": "linear_futures",
        "future": "linear_futures",
        "perp": "linear_futures",
        "perpetual": "linear_futures",
        "swap": "linear_futures",
        "spot": "spot",
        "inverse": "inverse_futures",
    }
    return aliases.get(raw, raw or "spot")


@dataclass(frozen=True)
class SymbolRef:
    base: str
    quote: str = "USDT"
    market_type: str = "spot"
    venue_symbol: str = ""

    @property
    def slash(self) -> str:
        return f"{self.base}/{self.quote}"

    @property
    def compact(self) -> str:
        return f"{self.base}{self.quote}"

    @property
    def underscored(self) -> str:
        return f"{self.base}_{self.quote}"


def parse_symbol(
    raw_symbol: str | None,
    quote_hint: str = "USDT",
    market_type: str = "spot",
) -> SymbolRef:
    raw = (raw_symbol or "").strip().upper()
    quote_hint = (quote_hint or "USDT").strip().upper()

    if not raw:
        return SymbolRef(base="", quote=quote_hint, market_type=normalize_market_type(market_type))

    cleaned = raw.split(":", 1)[0]
    if "/" in cleaned:
        base, quote = cleaned.split("/", 1)
    elif "_" in cleaned:
        base, quote = cleaned.split("_", 1)
    elif cleaned.endswith(quote_hint) and len(cleaned) > len(quote_hint):
        base, quote = cleaned[:-len(quote_hint)], quote_hint
    else:
        base, quote = cleaned, quote_hint

    return SymbolRef(
        base=base.strip(),
        quote=quote.strip(),
        market_type=normalize_market_type(market_type),
        venue_symbol=raw,
    )


def normalize_base_symbol(raw_symbol: str | None, quote_hint: str = "USDT") -> str:
    return parse_symbol(raw_symbol, quote_hint=quote_hint).base


def symbol_aliases(raw_symbol: str | None, quote_hint: str = "USDT") -> tuple[str, ...]:
    ref = parse_symbol(raw_symbol, quote_hint=quote_hint)
    aliases = [alias for alias in (ref.base, ref.compact, ref.slash, ref.underscored) if alias]
    return tuple(dict.fromkeys(aliases))
