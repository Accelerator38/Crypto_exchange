from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Dict

from exchange_core import ExchangeCapabilities, ExchangeSpec
from exchange_symbols import normalize_exchange_id
from exchanges.base import ExchangeAdapterError, ModuleBackedExchangeAdapter


class ExchangeRegistryError(RuntimeError):
    pass


class UnsupportedExchangeError(ExchangeRegistryError):
    pass


class ExchangeModuleImportError(ExchangeRegistryError):
    pass


BUILTIN_EXCHANGES: Dict[str, ExchangeSpec] = {
    "mexc": ExchangeSpec(
        exchange_id="mexc",
        display_name="MEXC",
        env_prefix="MEXC",
        connector_module="mexc_connector",
        api_module="exchange_api_runtime",
        adapter_module="exchanges.mexc_adapter",
        capabilities=ExchangeCapabilities(
            supports_spot=True,
            supports_linear_futures=True,
            supports_funding=True,
            supports_open_interest=True,
            supports_reduce_only=True,
        ),
    ),
    "bitget": ExchangeSpec(
        exchange_id="bitget",
        display_name="BITGET",
        env_prefix="BITGET",
        connector_module="bitget_connector",
        api_module="bitget_api",
        adapter_module="exchanges.bitget_adapter",
        capabilities=ExchangeCapabilities(
            supports_spot=True,
            supports_linear_futures=True,
            supports_passphrase=True,
            supports_funding=True,
            supports_open_interest=True,
            supports_reduce_only=True,
            supports_hedge_mode=True,
            supports_copy_trading=True,
        ),
    ),
    "binance": ExchangeSpec(
        exchange_id="binance",
        display_name="BINANCE",
        env_prefix="BINANCE",
        connector_module="binance_connector",
        api_module="binance_api",
        capabilities=ExchangeCapabilities(
            supports_spot=True,
            supports_linear_futures=True,
            supports_funding=True,
            supports_open_interest=True,
            supports_reduce_only=True,
            supports_copy_trading=True,
        ),
    ),
    "bybit": ExchangeSpec(
        exchange_id="bybit",
        display_name="BYBIT",
        env_prefix="BYBIT",
        connector_module="bybit_connector",
        api_module="bybit_api",
        capabilities=ExchangeCapabilities(
            supports_spot=True,
            supports_linear_futures=True,
            supports_funding=True,
            supports_open_interest=True,
            supports_reduce_only=True,
            supports_copy_trading=True,
        ),
    ),
    "okx": ExchangeSpec(
        exchange_id="okx",
        display_name="OKX",
        env_prefix="OKX",
        connector_module="okx_connector",
        api_module="okx_api",
        capabilities=ExchangeCapabilities(
            supports_spot=True,
            supports_linear_futures=True,
            supports_passphrase=True,
            supports_funding=True,
            supports_open_interest=True,
            supports_reduce_only=True,
        ),
    ),
    "kucoin": ExchangeSpec(
        exchange_id="kucoin",
        display_name="KUCOIN",
        env_prefix="KUCOIN",
        connector_module="kucoin_connector",
        api_module="kucoin_api",
        capabilities=ExchangeCapabilities(
            supports_spot=True,
            supports_linear_futures=True,
            supports_passphrase=True,
            supports_reduce_only=True,
            supports_copy_trading=True,
        ),
    ),
}


def list_registered_exchanges() -> tuple[str, ...]:
    return tuple(sorted(BUILTIN_EXCHANGES))

def _import_module(module_name: str, exchange_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise ExchangeModuleImportError(
            f"Failed to import module '{module_name}' for exchange '{exchange_name}': {exc}"
        ) from exc


def _build_synthetic_spec(
    exchange_id: str,
    connector_module: str,
    api_module: str,
) -> ExchangeSpec:
    normalized = normalize_exchange_id(exchange_id) or "custom"
    display_name = normalized.upper()
    return ExchangeSpec(
        exchange_id=normalized,
        display_name=display_name,
        env_prefix=display_name,
        connector_module=connector_module,
        api_module=api_module,
    )


@dataclass
class ExchangeRuntime:
    spec: ExchangeSpec
    adapter: ModuleBackedExchangeAdapter

    @property
    def exchange_id(self) -> str:
        return self.adapter.exchange_id

    @property
    def display_name(self) -> str:
        return self.adapter.display_name

    @property
    def env_prefix(self) -> str:
        return self.adapter.env_prefix

    @property
    def adapter_module_name(self) -> str:
        return self.adapter.adapter_module_name

    @property
    def connector_module_name(self) -> str:
        return self.adapter.connector_module_name

    @property
    def api_module_name(self) -> str:
        return self.adapter.api_module_name

    @property
    def connector_module(self):
        return self.adapter.connector_module

    @property
    def api_module(self):
        return self.adapter.api_module

    @property
    def capabilities(self) -> ExchangeCapabilities:
        return self.adapter.capabilities

    @property
    def api_key_env(self) -> str:
        return self.adapter.api_key_env

    @property
    def api_secret_env(self) -> str:
        return self.adapter.api_secret_env

    @property
    def api_passphrase_env(self) -> str:
        return self.adapter.api_passphrase_env

    @property
    def trading_mode_env(self) -> str:
        return self.adapter.trading_mode_env

    @property
    def action_names(self) -> dict:
        return self.adapter.action_names

    @property
    def warmup_bars(self) -> int:
        return self.adapter.warmup_bars

    def load_settings(self) -> dict:
        return self.adapter.load_settings()

    def parse_settings(self, raw_cfg: dict) -> dict:
        return self.adapter.parse_settings(raw_cfg)

    def check_connectivity(self) -> bool:
        return self.adapter.check_connectivity()

    def resolve_bridge_class(self):
        return self.adapter.resolve_bridge_class()

    def resolve_direct_client_class(self):
        return self.adapter.resolve_direct_client_class()

    def create_direct_client(
        self,
        api_key: str,
        api_secret: str,
        api_passphrase: str = "",
    ):
        return self.adapter.create_direct_client(api_key, api_secret, api_passphrase)

    @property
    def trading_stats_cls(self):
        return self.adapter.trading_stats_cls

    @property
    def inject_live_state_into_player(self):
        return self.adapter.inject_live_state_into_player

    @property
    def save_dashboard(self):
        return self.adapter.save_dashboard

    @property
    def save_summary_json(self):
        return self.adapter.save_summary_json

    @property
    def check_futures_api_permission(self):
        return self.adapter.check_futures_api_permission


def _build_adapter(
    spec: ExchangeSpec,
    connector_module,
    api_module,
    adapter_module_name: str | None = None,
) -> ModuleBackedExchangeAdapter:
    module_name = adapter_module_name or spec.adapter_module
    if not module_name:
        return ModuleBackedExchangeAdapter(spec, connector_module, api_module)
    adapter_module = _import_module(module_name, spec.display_name)
    adapter_cls = getattr(adapter_module, "ExchangeAdapter", None)
    if adapter_cls is None:
        raise ExchangeRegistryError(
            f"Adapter module '{module_name}' for exchange '{spec.display_name}' "
            "does not expose ExchangeAdapter."
        )
    try:
        return adapter_cls(spec, connector_module, api_module)
    except ExchangeAdapterError as exc:
        raise ExchangeRegistryError(str(exc)) from exc


def load_exchange_runtime(
    exchange_id: str | None = None,
    connector_module: str | None = None,
    api_module: str | None = None,
    adapter_module: str | None = None,
) -> ExchangeRuntime:
    requested = normalize_exchange_id(
        exchange_id or os.getenv("CRYPTO_EXCHANGE") or "mexc"
    )
    spec = BUILTIN_EXCHANGES.get(requested)

    if spec is None:
        if not connector_module or not api_module:
            raise UnsupportedExchangeError(
                f"Unsupported exchange '{requested}'. Registered exchanges: "
                f"{', '.join(list_registered_exchanges())}."
            )
        spec = _build_synthetic_spec(requested, connector_module, api_module)

    connector_name = connector_module or spec.connector_module
    api_name = api_module or spec.api_module
    runtime_spec = spec
    if connector_name != spec.connector_module or api_name != spec.api_module:
        runtime_spec = ExchangeSpec(
            exchange_id=spec.exchange_id,
            display_name=spec.display_name,
            env_prefix=spec.env_prefix,
            connector_module=connector_name,
            api_module=api_name,
            adapter_module=adapter_module or spec.adapter_module,
            market_types=spec.market_types,
            capabilities=spec.capabilities,
        )

    connector = _import_module(connector_name, runtime_spec.display_name)
    api = _import_module(api_name, runtime_spec.display_name)
    adapter = _build_adapter(runtime_spec, connector, api, adapter_module_name=adapter_module)
    return ExchangeRuntime(spec=runtime_spec, adapter=adapter)
