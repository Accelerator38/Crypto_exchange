from __future__ import annotations

from unittest.mock import patch

import pytest


def test_ccxt_direct_client_adds_mandatory_demo_header():
    from panteon_runtime import bitget_api

    client = bitget_api._build_client(
        "swap",
        "demo-key",
        "demo-secret",
        "demo-passphrase",
        demo=True,
    )

    assert client.headers.get("paptrading") == "1"


def test_ccxt_order_client_adds_mandatory_demo_header():
    from panteon_runtime import bitget_connector

    client = bitget_connector._private_client(
        "swap",
        "demo-key",
        "demo-secret",
        "demo-passphrase",
        demo=True,
    )

    assert client.headers.get("paptrading") == "1"


def test_bitget_demo_adapter_never_falls_back_to_live_credentials():
    from panteon_v2.app.bitget_adapter import BitgetExchangeAdapter

    with patch.dict(
        "os.environ",
        {
            "BITGET_TRADING_MODE": "demo_futures",
            "BITGET_API_KEY": "live-key",
            "BITGET_SECRET_KEY": "live-secret",
            "BITGET_PASSPHRASE": "live-passphrase",
        },
        clear=True,
    ):
        with pytest.raises(RuntimeError, match="BITGET_DEMO_API_KEY"):
            BitgetExchangeAdapter(demo=True)


def test_demo_market_data_bridge_uses_live_public_feed_mode():
    from panteon_v2.app.v1_bridge_runner import _bridge_market_data_mode

    assert _bridge_market_data_mode("demo_futures") == "live_futures"
