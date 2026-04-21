from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class ExchangeCapabilities:
    supports_spot: bool = True
    supports_linear_futures: bool = True
    supports_inverse_futures: bool = False
    supports_passphrase: bool = False
    supports_reduce_only: bool = False
    supports_hedge_mode: bool = False
    supports_funding: bool = False
    supports_open_interest: bool = False
    supports_copy_trading: bool = False


@dataclass(frozen=True)
class ExchangeSpec:
    exchange_id: str
    display_name: str
    env_prefix: str
    connector_module: str
    api_module: str
    adapter_module: str = ""
    market_types: Tuple[str, ...] = ("spot", "linear_futures")
    capabilities: ExchangeCapabilities = field(default_factory=ExchangeCapabilities)

    @property
    def normalized_id(self) -> str:
        return self.exchange_id.strip().lower()
