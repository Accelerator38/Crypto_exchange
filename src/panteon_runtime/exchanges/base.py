from __future__ import annotations

from types import ModuleType
from typing import Iterable

from exchange_core import ExchangeCapabilities, ExchangeSpec


class ExchangeAdapterError(RuntimeError):
    pass


DIRECT_CLIENT_CANDIDATES = (
    "ExchangeDirectClient",
    "MexcDirectClient",
    "BitgetDirectClient",
    "BinanceDirectClient",
    "BybitDirectClient",
    "OkxDirectClient",
    "KucoinDirectClient",
)


BRIDGE_CLASS_CANDIDATES = (
    "AgentExchangeBridge",
    "AgentBitgetBridge",
    "AgentMexcBridge",
    "AgentBinanceBridge",
    "AgentBybitBridge",
    "AgentOkxBridge",
    "AgentKucoinBridge",
)


def resolve_required_attr(module: ModuleType, attr_name: str):
    value = getattr(module, attr_name, None)
    if value is None:
        raise ExchangeAdapterError(
            f"Module '{module.__name__}' does not expose required attribute '{attr_name}'."
        )
    return value


def resolve_first_attr(module: ModuleType, attr_names: Iterable[str]):
    for attr_name in attr_names:
        value = getattr(module, attr_name, None)
        if value is not None:
            return value
    raise ExchangeAdapterError(
        f"Module '{module.__name__}' does not expose any of: {', '.join(attr_names)}"
    )


class ModuleBackedExchangeAdapter:
    def __init__(self, spec: ExchangeSpec, connector_module: ModuleType, api_module: ModuleType):
        self.spec = spec
        self.connector_module = connector_module
        self.api_module = api_module

    @property
    def exchange_id(self) -> str:
        return self.spec.normalized_id

    @property
    def display_name(self) -> str:
        return self.spec.display_name

    @property
    def env_prefix(self) -> str:
        return self.spec.env_prefix

    @property
    def adapter_module_name(self) -> str:
        return type(self).__module__

    @property
    def connector_module_name(self) -> str:
        return self.connector_module.__name__

    @property
    def api_module_name(self) -> str:
        return self.api_module.__name__

    @property
    def capabilities(self) -> ExchangeCapabilities:
        return self.spec.capabilities

    @property
    def api_key_env(self) -> str:
        return f"{self.env_prefix}_API_KEY"

    @property
    def api_secret_env(self) -> str:
        return f"{self.env_prefix}_SECRET_KEY"

    @property
    def api_passphrase_env(self) -> str:
        return f"{self.env_prefix}_PASSPHRASE"

    @property
    def trading_mode_env(self) -> str:
        return f"{self.env_prefix}_TRADING_MODE"

    @property
    def action_names(self) -> dict:
        return dict(getattr(self.connector_module, "ACTION_NAMES", {}) or {})

    @property
    def warmup_bars(self) -> int:
        return int(getattr(self.connector_module, "WARMUP_BARS", 0) or 0)

    def load_settings(self) -> dict:
        return resolve_required_attr(self.connector_module, "_load_settings")()

    def parse_settings(self, raw_cfg: dict) -> dict:
        return resolve_required_attr(self.connector_module, "_parse_settings")(raw_cfg)

    def check_connectivity(self) -> bool:
        return bool(resolve_required_attr(self.connector_module, "check_connectivity")())

    def resolve_bridge_class(self):
        return resolve_first_attr(self.connector_module, BRIDGE_CLASS_CANDIDATES)

    def resolve_direct_client_class(self):
        return resolve_first_attr(self.api_module, DIRECT_CLIENT_CANDIDATES)

    def create_direct_client(
        self,
        api_key: str,
        api_secret: str,
        api_passphrase: str = "",
    ):
        factory = getattr(self.api_module, "create_direct_client", None)
        if callable(factory):
            return factory(api_key, api_secret, api_passphrase)
        direct_client_cls = self.resolve_direct_client_class()
        try:
            return direct_client_cls(api_key, api_secret, api_passphrase)
        except TypeError:
            return direct_client_cls(api_key, api_secret)

    @property
    def trading_stats_cls(self):
        return resolve_required_attr(self.api_module, "TradingStats")

    @property
    def inject_live_state_into_player(self):
        return resolve_required_attr(self.api_module, "inject_live_state_into_player")

    @property
    def save_dashboard(self):
        return resolve_required_attr(self.api_module, "save_dashboard")

    @property
    def save_summary_json(self):
        return resolve_required_attr(self.api_module, "save_summary_json")

    @property
    def check_futures_api_permission(self):
        return resolve_required_attr(self.api_module, "_check_futures_api_permission")
