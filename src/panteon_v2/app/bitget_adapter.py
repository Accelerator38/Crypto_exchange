"""Bitget live futures adapter for Panteon v2."""

from __future__ import annotations

import os
from typing import Any, Optional

from .v1_futures_adapter import V1FuturesExchangeAdapter, load_runtime_leverage


class BitgetExchangeAdapter(V1FuturesExchangeAdapter):
    """Wrap the existing v1 Bitget futures client as a v2 Exchange."""

    def __init__(
        self,
        *,
        order_client: Optional[Any] = None,
        read_client: Optional[Any] = None,
        leverage: Optional[int] = None,
    ) -> None:
        if order_client is None:
            api_key = os.getenv("BITGET_API_KEY", "")
            api_secret = os.getenv("BITGET_SECRET_KEY", "")
            api_passphrase = os.getenv("BITGET_PASSPHRASE", "")
            if not api_key or not api_secret or not api_passphrase:
                raise RuntimeError(
                    "BITGET_API_KEY, BITGET_SECRET_KEY and BITGET_PASSPHRASE are required"
                )
            from bitget_connector import BitgetFuturesClient  # type: ignore

            order_client = BitgetFuturesClient(api_key, api_secret, api_passphrase)

        if read_client is None:
            try:
                api_key = os.getenv("BITGET_API_KEY", "")
                api_secret = os.getenv("BITGET_SECRET_KEY", "")
                api_passphrase = os.getenv("BITGET_PASSPHRASE", "")
                from bitget_api import BitgetDirectClient  # type: ignore

                read_client = BitgetDirectClient(api_key, api_secret, api_passphrase)
            except Exception:
                read_client = order_client

        super().__init__(
            name="BITGET",
            order_client=order_client,
            read_client=read_client,
            leverage=leverage if leverage is not None else load_runtime_leverage(),
            default_min_notional=5.10,
            default_fee_rate=0.0006,
            close_via_place_order=False,
        )
